#!/bin/bash
#SBATCH --job-name=process_sparc_ljspeech
#SBATCH --output=/path/to/slurm_logs/preprocess/process_sparc_ljspeech_%j.out # replace with your own NFS log dir
#SBATCH --error=/path/to/slurm_logs/preprocess/process_sparc_ljspeech_%j.err  # replace with your own NFS log dir
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# Uncomment and set to your own email to get job-completion notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

LJSPEECH_ROOT="/path/to/LJSpeech-1.1/preprocessed/" # replace with your own dataset root

uv run process_sparc.py --sparc_dir $LJSPEECH_ROOT --preprocessed_dir $LJSPEECH_ROOT --ema_output_dirname ema_preprocessed
