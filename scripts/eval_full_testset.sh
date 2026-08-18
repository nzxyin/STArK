#!/bin/bash
#SBATCH --job-name=stark_eval_testset
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_eval_testset_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_eval_testset_%j.err
#SBATCH --partition=general
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

# One-off read-only quality check of whatever checkpoint is passed in (mid-training or final —
# this does NOT push anything to the Hub) against the full LibriTTS-R test-clean split: PCC/DTW
# of predicted EMA vs ground truth (Setup 1 predicted-duration + Setup 2 aligner-duration, same
# methodology as eval_and_push.py / the paper's Table 2), plus DNSMOS + UTMOSv2 on the Setup 1
# resynthesized audio vs ground truth vs oracle resynthesis (paper Table 1 methodology).
#
# No longer pinned to L40S (was gpu:L40S:1, cpus-per-gpu=8, mem-per-cpu=8G): L40S nodes are
# heavily contended cluster-wide, which is most of why 3 straight submissions sat PENDING for
# up to an hour with zero runtime. This job doesn't need a specific GPU model — it's single-GPU
# inference only (devices=1, no DDP) — and 3 earlier attempts already got well past model load
# and into real batch processing on L40S with no sign of the DDP-era hang that motivated pinning
# it for *training* (train_large_100k.sh), so that risk looks low here. Any-GPU-type + a smaller
# CPU/mem footprint broadens the pool of nodes/QoS slots that can satisfy the request, which
# should schedule faster on both general (still capped elsewhere) and preempt. num_workers is
# dropped to match (see the uv run invocation below).

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

# Keep UTMOSv2's pretrained-model download and any HF cache off $HOME (tight 100GB quota).
export HF_HOME=/data/user_data/xoy/.cache/huggingface
export TORCH_HOME=/data/user_data/xoy/.cache/torch
mkdir -p "$HF_HOME" "$TORCH_HOME"

CKPT_PATH="${1:?usage: sbatch eval_full_testset.sh <ckpt_path> <results_path>}"
RESULTS_PATH="${2:?usage: sbatch eval_full_testset.sh <ckpt_path> <results_path>}"

# Snapshot the checkpoint before evaluating rather than reading it live: if $CKPT_PATH is an
# actively-training run's last.ckpt (as it is here), training will keep overwriting it
# periodically over the course of this job's up-to-8-hour runtime, and Lightning's
# ModelCheckpoint doesn't write it via an atomic rename — a read mid-write could load a
# truncated/corrupt file. Copying once up front pins us to one consistent, fully-written state.
SNAPSHOT_PATH="/scratch/job_tmp/stark_eval_ckpt_snapshot_${SLURM_JOB_ID}.ckpt"
mkdir -p "$(dirname "$SNAPSHOT_PATH")"
cp "$CKPT_PATH" "$SNAPSHOT_PATH"
echo "=== snapshotted $CKPT_PATH -> $SNAPSHOT_PATH ==="

echo "=== installing eval extras (utmosv2) ==="
uv sync --extra eval

echo "=== evaluating $SNAPSHOT_PATH (snapshot of $CKPT_PATH) on test-clean, writing to $RESULTS_PATH ==="
uv run --extra eval scripts/eval_full_testset.py \
    --checkpoint "$SNAPSHOT_PATH" \
    --dataset_root /data/user_data/xoy/LibriTTS_R/ \
    --num_workers 4 \
    --results_path "$RESULTS_PATH"
exit_code=$?

echo "=== eval script exited with code $exit_code ==="
exit $exit_code
