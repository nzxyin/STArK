#!/bin/bash
#SBATCH --job-name=stark_plot_examples
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_plot_examples_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_plot_examples_%j.err
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

# general (2026-08-20): matches train_large_100k.sh's partition -- STArK was moved off msp
# (deprioritized relative to other, active articulatory-tts work) and off preempt.
#
# Generates example articulatory-trace comparison plots (predicted vs ground truth) plus a
# DTW/PCC summary bar chart, from whatever checkpoint + eval results JSON are passed in. Much
# shorter than eval_full_testset.sh (a handful of example utterances, not the full split), but
# reuses the same lessons: any-GPU-type gres (schedules far faster than pinning L40S), caches
# redirected off $HOME, checkpoint snapshotted before reading (training may be actively
# overwriting last.ckpt while this runs).

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/data/user_data/xoy/.cache/huggingface
export TORCH_HOME=/data/user_data/xoy/.cache/torch
export UV_CACHE_DIR=/data/user_data/xoy/.cache/uv
mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR"

CKPT_PATH="${1:?usage: sbatch plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>}"
RESULTS_JSON="${2:?usage: sbatch plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>}"
OUT_DIR="${3:?usage: sbatch plot_eval_examples.sh <ckpt_path> <results_json_or_NONE> <out_dir>}"

SNAPSHOT_PATH="/scratch/job_tmp/stark_plot_ckpt_snapshot_${SLURM_JOB_ID}.ckpt"
mkdir -p "$(dirname "$SNAPSHOT_PATH")"
cp "$CKPT_PATH" "$SNAPSHOT_PATH"
echo "=== snapshotted $CKPT_PATH -> $SNAPSHOT_PATH ==="

# --extra eval, matching eval_full_testset.sh, even though this script doesn't itself import
# utmosv2: both scripts share ONE .venv (checked out once under $HOME, not per-job), and a plain
# `uv sync` here previously raced a concurrently-running eval_full_testset.sh job, desyncing the
# extra it needed mid-run (ModuleNotFoundError: No module named 'utmosv2') -- confirmed on a
# real concurrent submission of both jobs. Keeping every script's sync spec identical is the
# cheap fix; a fully isolated per-job venv would be the robust one if this keeps happening.
echo "=== syncing env ==="
uv sync --extra eval

RESULTS_ARG=()
if [ "$RESULTS_JSON" != "NONE" ]; then
    RESULTS_ARG=(--results_json "$RESULTS_JSON")
fi

echo "=== generating plots into $OUT_DIR ==="
uv run scripts/plot_eval_examples.py \
    --checkpoint "$SNAPSHOT_PATH" \
    --dataset_root /data/user_data/xoy/LibriTTS_R/ \
    --n 4 \
    --out_dir "$OUT_DIR" \
    "${RESULTS_ARG[@]}"
exit_code=$?
echo "=== plot script exited with code $exit_code ==="

echo "=== files in $OUT_DIR ==="
ls -la "$OUT_DIR"

# Give the caller a window to scp/ssh-cat the PNGs off this node while the allocation is still
# live (pam_slurm_adopt access is revoked the instant this job leaves the queue, and /data isn't
# mounted on the login node at all) -- same pattern used earlier this session to read live logs.
echo "=== holding allocation open for 180s so results can be fetched from this node ==="
sleep 180

exit $exit_code
