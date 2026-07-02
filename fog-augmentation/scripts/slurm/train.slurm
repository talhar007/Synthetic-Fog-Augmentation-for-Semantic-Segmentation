#!/usr/bin/env bash
#SBATCH --job-name=fog_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/%j_%x.out
#SBATCH --error=results/slurm_logs/%j_%x.err

# Usage:
#   sbatch --export=CONFIG=configs/photometric_med.yaml scripts/slurm/train.slurm
# Or to submit all experiments:
#   bash scripts/slurm/submit_all.sh

set -euo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Config:   ${CONFIG:-configs/baseline.yaml}"
echo "Started:  $(date)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Load modules ──────────────────────────────────────────────────────────────
module load gcc/v9.3.0
module load cuda/v12.9

export CUDA_HOME=/usr/app-soft1/cuda/v12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ── Activate venv ─────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── Log environment ───────────────────────────────────────────────────────────
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Run training ──────────────────────────────────────────────────────────────
mkdir -p results/slurm_logs

CONFIG="${CONFIG:-configs/baseline.yaml}"
python src/train.py --config "$CONFIG"

echo "Finished: $(date)"
