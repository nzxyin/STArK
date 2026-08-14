#!/bin/bash
#SBATCH --job-name=stark_large_100k
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_large_100k_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_large_100k_%j.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:L40S:2

# Same config as train_preempt.sh (model=large_model train=train_large), but capped at
# 100000 steps instead of train_large.yaml's default 500000. train.py auto-resumes from the
# last checkpoint, so this is safe to run in successive 2-day chunks.
#
# Uses 2 GPUs, not train_large.yaml's default 4: every 4-GPU DDP attempt hung at the first
# call into Aligner.forward's _binarize_attention (confirmed via py-spy — stuck in a CUDA
# device-transfer syscall, not a Python-level slowdown), reproduced identically even on a
# single GPU with no DDP involved at all, across 7 different nodes/3 different GPU models
# that all otherwise passed trivial CUDA sanity checks cleanly. A 2-GPU DDP run completed
# 200/200 steps cleanly with no such hang. Root cause undetermined (not node-specific, not
# GPU-model-specific, not a Python/JIT slowdown) — 2 GPUs is an empirically-verified
# workaround, not a diagnosed fix. accumulate_grad_batches is doubled (2->4) to preserve the
# same effective batch size (128) the paper's 4-GPU config used.
#
# general instead of preempt: general has this cluster's highest scheduling priority (10000)
# and nothing preempts it, whereas preempt (priority 0) sits at the bottom of the preemption
# hierarchy and was repeatedly evicted by general/debug/array jobs, mid-run, with no bound on
# how long it then sits requeued waiting for another allocation. Trading unlimited-but-
# preemptible time for general's guaranteed-but-capped-at-2-days time is a net win here in
# practice. To cover a run that needs more than 2 days, this script chains itself: each link
# queues its own successor (via --dependency=afterany, so it runs regardless of *how* this
# link ends — clean completion, hitting the 2-day wall, or a crash) before it starts training,
# so the chain survives even if this link gets killed by the time limit before reaching any
# of its own post-training cleanup code.

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

# IMPORTANT: this uses its own experiment directory (stark_large_100k), deliberately
# distinct from train_large.yaml's default "ddp_slurm_large_model" — that name is where the
# original paper-reproduction checkpoints (e.g. step=32000-val/loss=6.03.ckpt, referenced in
# testing.ipynb) already live. Do not point this at that directory: train.py's auto-resume
# logic will pick up whatever last.ckpt it finds there, and an incompatible/older checkpoint
# will crash on load (as happened once already) or, worse, get silently overwritten by this
# run's own checkpointing once training starts.
#
# experiment_name is passed directly (not as a literal "{experiment_name}" CLI override)
# since Hydra's override grammar rejects unescaped '{' — that placeholder only resolves when
# it's baked into the YAML default itself, which train.py .format()s at startup.
CKPT_DIR=/data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt
LOG_DIR=/data/user_data/xoy/articulatory-tts/stark_large_100k/log
CHAIN_MARKER=/data/user_data/xoy/articulatory-tts/stark_large_100k/.chain_count
MAX_CHAIN_LENGTH=10  # 10 x 2 days = 20 days of budget; the observed 2-GPU rate needs far less

mkdir -p "$(dirname "$CHAIN_MARKER")"
chain_count=$(cat "$CHAIN_MARKER" 2>/dev/null || echo 0)

# Queue our own successor *before* training starts (not after) — if this link gets killed
# outright by the 2-day time limit, everything below this point never runs, but the successor
# is already safely in the queue. Use the canonical checked-in script path (not $0 — SLURM
# stages/copies the submitted script, so $0 doesn't reliably point back at a resubmittable
# path) via $SLURM_SUBMIT_DIR, the directory `sbatch` was originally invoked from.
successor_id=""
if [ "$chain_count" -lt "$MAX_CHAIN_LENGTH" ]; then
    echo $((chain_count + 1)) > "$CHAIN_MARKER"
    successor_id=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
        "$SLURM_SUBMIT_DIR/scripts/train_large_100k.sh")
    echo "=== queued successor job $successor_id (chain link $((chain_count + 1))/$MAX_CHAIN_LENGTH) ==="
else
    echo "=== reached MAX_CHAIN_LENGTH=$MAX_CHAIN_LENGTH links without finishing — not queueing another successor ==="
fi

echo "=== training on $(hostname), chain link $((chain_count + 1)) ==="
uv run train.py train=train_large model=large_model train.trainer.max_steps=100000 \
    train.trainer.devices=2 \
    train.trainer.accumulate_grad_batches=4 \
    preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
    train.checkpoint.dirpath=$CKPT_DIR \
    train.logger.save_dir=$LOG_DIR
exit_code=$?

if [ $exit_code -eq 0 ]; then
    # trainer.fit() only returns 0 like this because max_steps was actually reached — nothing
    # else in this config stops training early.
    echo "=== training finished successfully (max_steps reached) ==="
    rm -f "$CHAIN_MARKER"
    if [ -n "$successor_id" ]; then
        echo "=== cancelling now-unneeded successor job $successor_id ==="
        scancel "$successor_id"
    fi
    echo "=== submitting eval + push to HF ==="
    sbatch "$SLURM_SUBMIT_DIR/scripts/eval_and_push.sh"
else
    echo "=== training exited with code $exit_code (2-day time limit or a transient error) — successor job $successor_id will continue from the last checkpoint ==="
fi
