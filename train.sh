#!/bin/bash
#SBATCH --job-name=train_articulatory_tts
#SBATCH --output=/path/to/slurm_logs/train/train_articulatory_tts_%j.out # replace with your own NFS log dir
#SBATCH --error=/path/to/slurm_logs/train/train_articulatory_tts_%j.err  # replace with your own NFS log dir
#SBATCH --partition=general
#SBATCH --time=2-00:00:00 # Max job run time
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
# Uncomment and set to your own email to get job-completion notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
uv run train.py train=ljspeech_large model=large_model preprocess=ljspeech_preprocess
