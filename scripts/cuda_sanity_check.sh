#!/bin/bash
#SBATCH --job-name=cuda_sanity_check
#SBATCH --output=/data/user_data/xoy/slurm_logs/cuda_sanity_check_%j.out
#SBATCH --error=/data/user_data/xoy/slurm_logs/cuda_sanity_check_%j.err
#SBATCH --partition=general
# debug (this script's original partition) was removed cluster-wide 2026-08-19; general is the
# replacement (see ~/.claude/CLAUDE.md's srun-reliability note for the fuller history).
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:L40S:1

# Minimal, STArK-independent CUDA sanity check: allocate on GPU, do a tiny compute op,
# transfer back to host (the same class of operation — a device-to-host .cpu() transfer —
# that hung inside model.py's _binarize_attention). If this hangs too, it's a cluster/driver
# issue unrelated to STArK's code.

export PATH="$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"

uv run --no-project --with torch python3 -c "
import time
import torch

print('hostname check via /proc:', open('/proc/sys/kernel/hostname').read().strip())
print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))

t0 = time.time()
x = torch.randn(1000, 1000, device='cuda')
torch.cuda.synchronize()
print(f'allocate+sync: {time.time()-t0:.2f}s')

t0 = time.time()
y = (x @ x).sum()
torch.cuda.synchronize()
print(f'matmul+sync: {time.time()-t0:.2f}s, result={y.item():.2f}')

t0 = time.time()
z = x.detach().cpu()
print(f'.cpu() transfer: {time.time()-t0:.2f}s, shape={z.shape}')

t0 = time.time()
w = z.cuda()
torch.cuda.synchronize()
print(f'.cuda() transfer back: {time.time()-t0:.2f}s')

print('ALL CUDA SANITY CHECKS PASSED')
"
echo "=== sanity check exited with code $? ==="
