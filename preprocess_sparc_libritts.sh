#!/bin/bash
#SBATCH --job-name=process_sparc_array
#SBATCH --output=logs/preprocess/process_sparc_%A_%a.out
#SBATCH --error=logs/preprocess/process_sparc_%A_%a.err
#SBATCH --partition=array
#SBATCH --array=0-3
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xoy@andrew.cmu.edu

source /home/xoy/articulatory-tts/.venv/bin/activate

LIBRITTS_ROOT="/data/user_data/xoy/LibriTTS_R/"
SPARC_DIRS=("dev-clean-sparc" "test-clean-sparc" "train-clean-100-sparc" "train-clean-360-sparc")
SPARC_DIR=${LIBRITTS_ROOT}${SPARC_DIRS[${SLURM_ARRAY_TASK_ID}]}
PREPROCESS_DIRS=("dev-clean-preprocessed" "test-clean-preprocessed" "train-clean-100-preprocessed" "train-clean-360-preprocessed")
PREPROCESSED_DIR=${LIBRITTS_ROOT}${PREPROCESS_DIRS[${SLURM_ARRAY_TASK_ID}]}

uv run process_sparc.py --sparc_dir $SPARC_DIR --preprocessed_dir $PREPROCESSED_DIR
