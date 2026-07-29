#!/bin/bash
#SBATCH --job-name=build_hf_dataset
#SBATCH --output=/data/user_data/xoy/slurm_logs/build_hf_dataset_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/build_hf_dataset_%j.err
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
# Uncomment and set to your own email to get job-completion notifications:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com

# Needs a Hugging Face token cached (e.g. via `hf auth login`) under HF_HOME below.
export HF_HOME="/data/user_data/$USER/.hf_cache"
export PATH="$HOME/.local/bin:$PATH"

cd "$SLURM_SUBMIT_DIR"

uv run --no-project --with huggingface_hub python scripts/build_hf_dataset.py \
    --source_root "/data/user_data/xoy/LibriTTS_R" \
    --staging_root "/data/user_data/xoy/libritts-r-stark-staging" \
    --repo_id "nzxyin/libritts-r-stark" \
    --upload
