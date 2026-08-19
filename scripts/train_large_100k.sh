#!/bin/bash
#SBATCH --job-name=stark_large_100k
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_large_100k_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_large_100k_%j.err
#SBATCH --partition=preempt
#SBATCH --requeue
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
# preempt, not general (switched back after using general for a while): general's fixed
# 8-GPU/user cap turned out to be a worse failure mode than preempt's eviction risk -- this
# job got fully blocked (QOSMaxGRESPerUser, indefinitely pending) on general because *other,
# unrelated* jobs under the same user account (an ablation array + a separate curriculum-
# training job) were using the rest of the 8-GPU budget, with no way to know when they'd free
# up. preempt has a much higher per-user GPU cap (24) and was never actually the bottleneck in
# practice once --requeue is set (SLURM auto-resubmits the same job on eviction, without
# consuming a chain link) -- the original "general has no preemption risk" reasoning undersold
# how easily general's cap gets exhausted by a single account's *other* concurrent work, which
# this script has no visibility into or control over. To cover a run that needs more than 2
# days regardless of partition, this script also chains itself: each link queues its own
# successor (via --dependency=afterany, so it runs regardless of *how* this link ends — clean
# completion, hitting the 2-day wall, a crash, or exhausting requeue attempts) before it starts
# training, so the chain survives even if this link gets killed by the time limit before
# reaching any of its own post-training cleanup code.

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
FASTFAIL_MARKER=/data/user_data/xoy/articulatory-tts/stark_large_100k/.fastfail_count
MAX_CHAIN_LENGTH=10  # 10 x 2 days = 20 days of budget; the observed 2-GPU rate needs far less
FASTFAIL_THRESHOLD_SEC=600   # a legit 2-day-timeout or long-running crash never looks like this
FASTFAIL_LIMIT=3             # this many back-to-back sub-10-min exits => stop chaining, don't retry forever

mkdir -p "$(dirname "$CHAIN_MARKER")"
chain_count=$(cat "$CHAIN_MARKER" 2>/dev/null || echo 0)
fastfail_count=$(cat "$FASTFAIL_MARKER" 2>/dev/null || echo 0)

# Queue our own successor *before* training starts (not after) — if this link gets killed
# outright by the 2-day time limit, everything below this point never runs, but the successor
# is already safely in the queue. Use the canonical checked-in script path (not $0 — SLURM
# stages/copies the submitted script, so $0 doesn't reliably point back at a resubmittable
# path) via $SLURM_SUBMIT_DIR, the directory `sbatch` was originally invoked from.
#
# SLURM_RESTART_COUNT guard: with --requeue set (needed on preempt), a preemption restarts
# THIS SAME job/script from the top rather than ending it — SLURM increments
# SLURM_RESTART_COUNT (0 on the original dispatch) each time. Without this guard, every
# preemption-triggered restart would queue *another* duplicate successor, burning the chain
# budget on mere evictions instead of real completions/crashes. --dependency=afterany already
# only fires once this job reaches a real terminal state (SLURM tracks REQUEUED as non-terminal
# on its own), so one successor queued at the original dispatch is enough.
successor_id=""
if [ "${SLURM_RESTART_COUNT:-0}" -eq 0 ] && [ "$chain_count" -lt "$MAX_CHAIN_LENGTH" ]; then
    echo $((chain_count + 1)) > "$CHAIN_MARKER"
    successor_id=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID \
        "$SLURM_SUBMIT_DIR/scripts/train_large_100k.sh")
    echo "=== queued successor job $successor_id (chain link $((chain_count + 1))/$MAX_CHAIN_LENGTH) ==="
else
    echo "=== reached MAX_CHAIN_LENGTH=$MAX_CHAIN_LENGTH links without finishing — not queueing another successor ==="
fi

echo "=== training on $(hostname), chain link $((chain_count + 1)) ==="
train_start=$(date +%s)
uv run train.py train=train_large model=large_model train.trainer.max_steps=100000 \
    train.trainer.devices=2 \
    train.trainer.accumulate_grad_batches=4 \
    preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
    train.checkpoint.dirpath=$CKPT_DIR \
    train.logger.save_dir=$LOG_DIR
exit_code=$?
train_duration=$(( $(date +%s) - train_start ))

if [ $exit_code -eq 0 ]; then
    # trainer.fit() only returns 0 like this because max_steps was actually reached — nothing
    # else in this config stops training early.
    echo "=== training finished successfully (max_steps reached) ==="
    rm -f "$CHAIN_MARKER" "$FASTFAIL_MARKER"
    if [ -n "$successor_id" ]; then
        echo "=== cancelling now-unneeded successor job $successor_id ==="
        scancel "$successor_id"
    elif [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
        # Known minor gap: on a preemption-requeued execution (restart count > 0) this process
        # never learned its own successor's job id (only the original, restart-count-0
        # execution queued and knows it), so it can't scancel it here. The already-queued
        # successor will still run, but train.py/Lightning will see global_step >= max_steps on
        # resume and return almost immediately without further training -- wasteful (one extra
        # short job, one duplicate eval+push) but not harmful. Not worth the extra state-passing
        # complexity to close given how rare "finishes exactly on a requeued execution" is.
        echo "=== finished on a requeued execution (restart #$SLURM_RESTART_COUNT) — cannot cancel the original successor from here, it will self-detect completion and exit quickly ==="
    fi
    echo "=== submitting eval + push to HF ==="
    sbatch "$SLURM_SUBMIT_DIR/scripts/eval_and_push.sh"
    exit 0
fi

# Non-zero exit. A legitimate 2-day-timeout link ran for ~2 days before SLURM killed it; a real
# crash (bad config, code bug, OOM at startup, etc.) typically dies within seconds-to-minutes.
# Distinguish the two by elapsed wall time so a crash-loop can't silently re-queue its way
# through the entire MAX_CHAIN_LENGTH budget (as happened once already — 10 links x ~31min each
# burned in ~5 hours with no one noticing until the chain was already dead).
if [ "$train_duration" -lt "$FASTFAIL_THRESHOLD_SEC" ]; then
    fastfail_count=$((fastfail_count + 1))
    echo "$fastfail_count" > "$FASTFAIL_MARKER"
    echo "=== training exited with code $exit_code after only ${train_duration}s (looks like a crash, not a timeout) — fastfail_count=$fastfail_count/$FASTFAIL_LIMIT ==="
    if [ "$fastfail_count" -ge "$FASTFAIL_LIMIT" ]; then
        echo "=== $fastfail_count consecutive fast failures — this looks like a crash loop, not transient flakiness. Cancelling successor $successor_id and halting the chain. Fix the root cause, then reset $FASTFAIL_MARKER (or just delete it) and resubmit. ==="
        if [ -n "$successor_id" ]; then
            scancel "$successor_id"
        fi
        exit 1
    fi
else
    # Real progress was made before this exit — a genuine timeout or a one-off transient error.
    # Reset the fast-fail streak so isolated blips don't accumulate toward the breaker.
    rm -f "$FASTFAIL_MARKER"
    echo "=== training exited with code $exit_code after ${train_duration}s (2-day time limit or a transient error) — successor job $successor_id will continue from the last checkpoint ==="
fi
exit $exit_code
