#!/bin/bash
#SBATCH --job-name=process_sparc_array
#SBATCH --output=/path/to/slurm_logs/preprocess/process_sparc_%A_%a.out # replace with your own NFS log dir
#SBATCH --error=/path/to/slurm_logs/preprocess/process_sparc_%A_%a.err  # replace with your own NFS log dir
#SBATCH --partition=array
#SBATCH --array=0-3
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

LIBRITTS_ROOT="/path/to/LibriTTS_R/" # replace with your own dataset root
SPARC_DIRS=("dev-clean-sparc" "test-clean-sparc" "train-clean-100-sparc" "train-clean-360-sparc")
SPARC_DIR=${LIBRITTS_ROOT}${SPARC_DIRS[${SLURM_ARRAY_TASK_ID}]}
PREPROCESS_DIRS=("dev-clean-preprocessed" "test-clean-preprocessed" "train-clean-100-preprocessed" "train-clean-360-preprocessed")
PREPROCESSED_DIR=${LIBRITTS_ROOT}${PREPROCESS_DIRS[${SLURM_ARRAY_TASK_ID}]}

uv run process_sparc.py --sparc_dir $SPARC_DIR --preprocessed_dir $PREPROCESSED_DIR
