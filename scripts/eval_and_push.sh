#!/bin/bash
#SBATCH --job-name=stark_large_eval_push
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_large_eval_push_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_large_eval_push_%j.err
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

# Needs a Hugging Face token cached (e.g. via `hf auth login`) under HF_HOME below.
export HF_HOME="/data/user_data/$USER/.hf_cache"
export PATH="$HOME/.local/bin:$PATH"

cd "$SLURM_SUBMIT_DIR"

uv run python scripts/eval_and_push.py \
    --ckpt_path /data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt/last.ckpt \
    --repo_id nzxyin/stark-large \
    --dataset_root /data/user_data/xoy/LibriTTS_R/ \
    --overrides model=large_model train=train_large \
        train.checkpoint.dirpath=/data/user_data/xoy/articulatory-tts/stark_large_100k/ckpt \
        train.logger.save_dir=/data/user_data/xoy/articulatory-tts/stark_large_100k/log
