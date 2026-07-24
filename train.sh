#!/bin/bash
#SBATCH --job-name=train_articulatory_tts
#SBATCH --output=logs/train/train_articulatory_tts_%j.out
#SBATCH --error=logs/train/train_articulatory_tts_%j.err
#SBATCH --partition=general
#SBATCH --time=2-00:00:00 # Max job run time
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xoy@andrew.cmu.edu

source /home/xoy/articulatory-tts/.venv/bin/activate
uv sync
uv build
# uv run train.py
uv run train.py train=ljspeech_large model=large_model preprocess=ljspeech_preprocess
