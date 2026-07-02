#!/usr/bin/env bash
# Run all 8 experiments sequentially on the current machine.
# For HPC batch submission, use scripts/slurm/train.slurm instead.
#
# Usage:
#   bash scripts/run_all_experiments.sh [--cityscapes-root PATH] [--acdc-root PATH]
#
# Prerequisites:
#   1. Download Cityscapes pretrained checkpoint:
#        python src/data/download.py
#   2. (Optional) Pre-render fog cache:
#        python scripts/generate_fog.py --cityscapes-root ... --output-root data/fog_cache

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/taah3149/Documents/Group_Studies/.venv/bin/activate

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

echo "=== Fog Augmentation Experiment Sweep ==="
echo "Timestamp: $(date)"
echo "Configs: ${#CONFIGS[@]}"
echo "(idempotent: already-complete experiments are skipped, partial ones resume from latest.pth)"

for cfg in "${CONFIGS[@]}"; do
  name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment']['name'])")
  out_dir="results/$name"
  log="$out_dir/train.log"
  latest="$out_dir/latest.pth"

  echo ""
  if [[ -f "$out_dir/best.pth" ]] && grep -q "Training complete" "$log" 2>/dev/null; then
    echo ">>> Skipping (already complete): $cfg"
    continue
  fi

  if [[ -f "$latest" ]]; then
    echo ">>> Resuming: $cfg (from $latest)"
    python src/train.py --config "$cfg" --resume "$latest"
  else
    echo ">>> Training (fresh start): $cfg"
    python src/train.py --config "$cfg"
  fi
  echo ">>> Done: $cfg"
done

echo ""
echo "=== All training complete. Evaluating on ACDC fog val... ==="

for cfg in "${CONFIGS[@]}"; do
  name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment']['name'])")
  out_dir="results/$name"
  eval_json="$out_dir/acdc_eval.json"

  if [[ -f "$eval_json" ]]; then
    echo ">>> Skipping eval (already done): $name"
    continue
  fi

  echo ">>> Evaluating: $name"
  python src/evaluate.py \
    --config "$cfg" \
    --checkpoint "$out_dir/best.pth" \
    --output "$eval_json"
done

echo ""
echo "=== Generating results table... ==="
python scripts/compile_results.py
