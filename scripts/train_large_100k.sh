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
uv run train.py train=train_large model=large_model train.trainer.max_steps=100000 \
    preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
    train.checkpoint.dirpath=/data/user_data/xoy/articulatory-tts/{experiment_name}/ckpt \
    train.logger.save_dir=/data/user_data/xoy/articulatory-tts/{experiment_name}/log
