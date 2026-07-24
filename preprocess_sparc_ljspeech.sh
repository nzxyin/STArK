#!/bin/bash
#SBATCH --job-name=process_sparc_ljspeech
#SBATCH --output=logs/preprocess/process_sparc_ljspeech_%j.out
#SBATCH --error=logs/preprocess/process_sparc_ljspeech_%j.err
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xoy@andrew.cmu.edu

source /home/xoy/articulatory-tts/.venv/bin/activate

LJSPEECH_ROOT="/data/user_data/xoy/LJSpeech-1.1/preprocessed/"

uv run process_sparc.py --sparc_dir $LJSPEECH_ROOT --preprocessed_dir $LJSPEECH_ROOT
