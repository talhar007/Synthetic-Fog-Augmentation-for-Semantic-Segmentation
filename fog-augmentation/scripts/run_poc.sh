#!/usr/bin/env bash
# POC: does swapping resnet101/DeepLabV3+ for a transformer encoder (SegFormer's
# mit_b2 + FPN decoder) improve ACDC-fog generalization, per the ACDC paper's own
# finding that transformer backbones generalize better under domain shift?
#
# Trains 4 short (10-epoch) runs — resnet101 vs mit_b2, each at baseline (no fog)
# and our best sweep config (nndepth_base_b0005_p30 settings) — then evaluates
# all 4 on ACDC fog val. Meant to catch a clear win/loss before committing to a
# full 50-epoch resweep with a new backbone.
#
# Usage: bash scripts/run_poc.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

source /home/taah3149/Documents/Group_Studies/.venv/bin/activate

ACDC_ROOT="/home/taah3149/Documents/Group_Studies/dataset/acdc"

CONFIGS=(
  configs/poc/resnet101_baseline.yaml
  configs/poc/mit_b2_baseline.yaml
  configs/poc/resnet101_nndepth_best.yaml
  configs/poc/mit_b2_nndepth_best.yaml
)

echo "=== Backbone POC (resnet101/DeepLabV3+ vs mit_b2/FPN, 10 epochs) ==="
echo "Timestamp: $(date)"

for cfg in "${CONFIGS[@]}"; do
  name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment']['name'])")
  out_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['training']['output_dir'])")
  eval_json="$out_dir/acdc_eval.json"
  latest="$out_dir/latest.pth"

  echo ""
  if [[ -f "$eval_json" ]]; then
    echo ">>> Skipping train (already complete): $name"
  elif [[ -f "$latest" ]]; then
    echo ">>> Resuming: $name (from $latest)"
    python src/train.py --config "$cfg" --resume "$latest"
  else
    echo ">>> Training (fresh start): $name"
    python src/train.py --config "$cfg"
  fi

  if [[ -f "$eval_json" ]]; then
    echo ">>> Skipping eval (already done): $name"
  else
    echo ">>> Evaluating: $name"
    python src/evaluate.py \
      --config "$cfg" \
      --checkpoint "$out_dir/best.pth" \
      --acdc-root "$ACDC_ROOT" \
      --output "$eval_json"
  fi

  echo ">>> Done: $name"
done

echo ""
echo "=== POC complete. Results: ==="
for cfg in "${CONFIGS[@]}"; do
  name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['experiment']['name'])")
  out_dir=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['training']['output_dir'])")
  miou=$(python -c "import json; print(json.load(open('$out_dir/acdc_eval.json'))['mIoU'])")
  echo "$name: $miou"
done
