#!/usr/bin/env bash
#SBATCH --job-name=fog_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_logs/%j_%x.out
#SBATCH --error=results/slurm_logs/%j_%x.err

# Usage:
#   sbatch \
#     --export=CONFIG=configs/photometric_med.yaml,CHECKPOINT=results/photometric_med/best.pth,ACDC_ROOT=/path/to/acdc \
#     scripts/slurm/evaluate.slurm

set -euo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Config:   ${CONFIG:-configs/baseline.yaml}"
echo "Ckpt:     ${CHECKPOINT:-unset}"
echo "ACDC:     ${ACDC_ROOT:-unset}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

module load gcc/v9.3.0
module load cuda/v12.9
export CUDA_HOME=/usr/app-soft1/cuda/v12.9
export PATH=$CUDA_HOME/bin:$PATH

source .venv/bin/activate

CONFIG="${CONFIG:-configs/baseline.yaml}"
CHECKPOINT="${CHECKPOINT}"
ACDC_ROOT="${ACDC_ROOT}"

# Derive output path from config name
EXP_NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['experiment']['name'])")
OUTPUT="results/${EXP_NAME}/acdc_eval.json"

python src/evaluate.py \
    --config     "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --acdc-root  "$ACDC_ROOT" \
    --output     "$OUTPUT"

echo "Results saved to $OUTPUT"
echo "Finished: $(date)"
