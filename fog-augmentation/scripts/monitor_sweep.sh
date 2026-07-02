#!/usr/bin/env bash
# Quick status overview of the 11-experiment training sweep.
#
# Usage:
#   bash scripts/monitor_sweep.sh          # one-shot status table
#   watch -n 30 bash scripts/monitor_sweep.sh   # auto-refresh every 30s
#
# For live scrolling logs of whichever experiment is currently training:
#   tail -f sweep_run.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIGS=(
  baseline
  photometric_low
  photometric_med
  photometric_high
  depthaware_b0005
  depthaware_b001
  depthaware_b002
  combined
  nn_depth_b0005
  nn_depth_b001
  nn_depth_b002
)

printf "%-20s %-10s %-10s %-12s %s\n" "EXPERIMENT" "STATUS" "EPOCH" "BEST_MIOU" "LAST LOG LINE"
printf "%-20s %-10s %-10s %-12s %s\n" "----------" "------" "-----" "---------" "-------------"

for name in "${CONFIGS[@]}"; do
  dir="results/$name"
  log="$dir/train.log"
  ckpt="$dir/best.pth"
  metrics="$dir/metrics.csv"

  if [[ -f "$ckpt" ]] && grep -q "Training complete" "$log" 2>/dev/null; then
    status="DONE"
  elif [[ -f "$log" ]]; then
    status="RUNNING"
  else
    status="PENDING"
  fi

  epoch="-"
  if [[ -f "$log" ]]; then
    epoch=$(grep -oE "Epoch [0-9]+/[0-9]+" "$log" | tail -1 | sed 's/Epoch //')
    [[ -z "$epoch" ]] && epoch="-"
  fi

  best_miou="-"
  if [[ -f "$metrics" ]]; then
    best_miou=$(awk -F, 'NR>1{if($3>m)m=$3} END{printf "%.4f", m}' "$metrics" 2>/dev/null)
  fi

  last_line="-"
  if [[ -f "$log" ]]; then
    last_line=$(tail -1 "$log" | sed 's/^\[[0-9:]*\] //')
  fi

  printf "%-20s %-10s %-10s %-12s %s\n" "$name" "$status" "$epoch" "$best_miou" "$last_line"
done
