#!/bin/bash
#SBATCH --job-name=stark_diag_1gpu
#SBATCH --output=/data/user_data/xoy/slurm_logs/stark_diag_1gpu_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/stark_diag_1gpu_%j.err
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1

# Cheap diagnostic: same model/data config as the real 100k-step run, but devices=1
# (no DDP at all) and capped at a couple hundred steps. Purpose: isolate whether the
# repeated stall (first an NCCL rank-2/3 hang, then a second stall that survived
# escalating to fresh nodes) is DDP-specific or something else entirely (dataset,
# model, environment). Uses its own isolated checkpoint dir — not the same as the
# real 100k run or the original paper checkpoints.
#
# every_n_train_steps=20 means a checkpoint should appear quickly if training is
# actually progressing — that's the clearest, least ambiguous signal available
# (Lightning's progress bar/console logging is otherwise unreliable in a
# non-interactive SLURM job, which is part of why the earlier runs were hard to
# read from logs alone).

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

uv run train.py train=train_large model=large_model \
    train.trainer.devices=1 \
    train.trainer.strategy=auto \
    train.trainer.max_steps=200 \
    train.trainer.val_check_interval=1000000 \
    +train.trainer.num_sanity_val_steps=0 \
    train.checkpoint.every_n_train_steps=20 \
    train.datamodule.num_workers=0 \
    preprocess.dataset.dataset_root=/data/user_data/xoy/LibriTTS_R/ \
    train.checkpoint.dirpath=/data/user_data/xoy/articulatory-tts/stark_diag_1gpu/ckpt \
    train.logger.save_dir=/data/user_data/xoy/articulatory-tts/stark_diag_1gpu/log
echo "=== diagnostic run exited with code $? ==="
