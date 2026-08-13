#!/bin/bash
#SBATCH --job-name=stark_diag_2gpu
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_diag_2gpu_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_diag_2gpu_%j.err
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:L40S:2

# Quick diagnostic: real DDP path (2 ranks instead of 4), unpinned node (nodes are
# allocated on-demand here, so pinning to one specific node isn't meaningful — the earlier
# babel-m5-28 pin was purely to reuse an already-verified-healthy node, not because node
# identity itself matters). Short max_steps just to see whether training actually progresses
# past the point where 1-GPU and 4-GPU DDP both hung (Aligner.forward's _binarize_attention
# call, per the py-spy trace).

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

uv run train.py train=train_large model=large_model \
    train.trainer.devices=2 \
    train.trainer.max_steps=200 \
    train.trainer.val_check_interval=1000000 \
    +train.trainer.num_sanity_val_steps=0 \
    train.checkpoint.every_n_train_steps=20 \
    preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
    train.checkpoint.dirpath=/data/user_data/xoy/articulatory-tts/stark_diag_2gpu/ckpt \
    train.logger.save_dir=/data/user_data/xoy/articulatory-tts/stark_diag_2gpu/log
echo "=== diagnostic run exited with code $? ==="
