#!/usr/bin/env bash
# Submit all 8 training experiments as independent SLURM jobs.
# Each depends only on the pretrained checkpoint being available.
#
# Usage:  bash scripts/slurm/submit_all.sh

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIGS=(
  configs/baseline.yaml
  configs/photometric_low.yaml
  configs/photometric_med.yaml
  configs/photometric_high.yaml
  configs/depthaware_b0005.yaml
  configs/depthaware_b001.yaml
  configs/depthaware_b002.yaml
  configs/combined.yaml
  configs/nn_depth_b0005.yaml
  configs/nn_depth_b001.yaml
  configs/nn_depth_b002.yaml
)

mkdir -p results/slurm_logs

for cfg in "${CONFIGS[@]}"; do
  name=$(python -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['experiment']['name'])")
  jid=$(sbatch --export=CONFIG=$cfg --job-name="fog_${name}" scripts/slurm/train.slurm | awk '{print $NF}')
  echo "Submitted $name  → job $jid"
done
