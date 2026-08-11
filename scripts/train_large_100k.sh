#!/bin/bash
#SBATCH --job-name=stark_large_100k
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_large_100k_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_large_100k_%j.err
#SBATCH --partition=preempt
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
#SBATCH --requeue

# Same config as train_preempt.sh (model=large_model train=train_large), but capped at
# 100000 steps instead of train_large.yaml's default 500000. Safe to preempt/requeue —
# train.py auto-resumes from the last checkpoint.

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
#
# Two distinct failure modes observed in practice, handled differently:
#  1. A quick application-level crash (e.g. a transient "CUDA-capable device(s) is/are busy"
#     error) — retried in-place below, since a fresh process on the same node/allocation is
#     often enough.
#  2. A persistent NCCL DDP hang between specific ranks (observed: ranks 2/3 hang on their
#     first collective op, every single one of 10 in-place retries, for the entire 4h13m job —
#     never a single training step logged). Retrying in place doesn't help here since it's the
#     same node/GPUs every time. After a couple of in-place attempts, self-requeue via
#     `scontrol requeue` instead — that returns the job to the SLURM queue for a fresh
#     scheduling decision, which can land on a different node.
CKPT_DIR=/data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt
LOG_DIR=/data/user_data/xoy/articulatory-tts/stark_large_100k/log
ATTEMPT_MARKER=/data/user_data/xoy/articulatory-tts/stark_large_100k/.requeue_attempts
IN_PLACE_RETRIES=2
MAX_REQUEUES=8

mkdir -p "$(dirname "$ATTEMPT_MARKER")"
requeue_count=$(cat "$ATTEMPT_MARKER" 2>/dev/null || echo 0)

for attempt in $(seq 1 $IN_PLACE_RETRIES); do
    echo "=== training attempt $attempt/$IN_PLACE_RETRIES (requeue round $requeue_count/$MAX_REQUEUES) on $(hostname) ==="
    uv run train.py train=train_large model=large_model train.trainer.max_steps=100000 \
        preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
        train.checkpoint.dirpath=$CKPT_DIR \
        train.logger.save_dir=$LOG_DIR
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=== training finished successfully ==="
        rm -f "$ATTEMPT_MARKER"
        exit 0
    fi
    echo "=== training attempt $attempt failed with exit code $exit_code ==="
    sleep 30
done

requeue_count=$((requeue_count + 1))
echo "$requeue_count" > "$ATTEMPT_MARKER"
if [ "$requeue_count" -ge "$MAX_REQUEUES" ]; then
    echo "=== giving up after $requeue_count requeue rounds (${IN_PLACE_RETRIES} in-place attempts each) ==="
    exit 1
fi
echo "=== $IN_PLACE_RETRIES in-place attempts failed on $(hostname) — self-requeueing for a fresh node (round $requeue_count/$MAX_REQUEUES) ==="
scontrol requeue "$SLURM_JOB_ID"
sleep 60  # give the requeue time to take effect before this script instance exits
exit 1
