#!/bin/bash
#SBATCH --job-name=train_articulatory_tts
#SBATCH --output=/path/to/slurm_logs/train/train_articulatory_tts_%j.out # replace with your own NFS log dir
#SBATCH --error=/path/to/slurm_logs/train/train_articulatory_tts_%j.err  # replace with your own NFS log dir
#SBATCH --partition=preempt
#SBATCH --time=5-00:00:00 # Max job run time
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
# Uncomment and set to your own email to get job-completion notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com
#SBATCH --requeue

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
# This is the exact config used to train the model reported in the paper
# (checkpoint step 32000; see README for details).
uv run train.py train=train_large model=large_model
