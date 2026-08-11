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
# --requeue only covers SLURM-level preemption/timeout — a plain application crash (e.g. a
# transient "CUDA-capable device(s) is/are busy" error, which has happened once already) exits
# the script non-zero and SLURM will NOT auto-restart it, silently ending the run. Retry a
# bounded number of times in-script so a transient failure doesn't kill the whole job; train.py
# auto-resumes from the last checkpoint on each retry either way.
MAX_ATTEMPTS=10
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== training attempt $attempt/$MAX_ATTEMPTS ==="
    uv run train.py train=train_large model=large_model train.trainer.max_steps=100000 \
        preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
        train.checkpoint.dirpath=/data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt \
        train.logger.save_dir=/data/user_data/xoy/articulatory-tts/stark_large_100k/log
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=== training finished successfully ==="
        exit 0
    fi
    echo "=== training attempt $attempt failed with exit code $exit_code, retrying in 60s ==="
    sleep 60
done
echo "=== giving up after $MAX_ATTEMPTS failed attempts ==="
exit 1
