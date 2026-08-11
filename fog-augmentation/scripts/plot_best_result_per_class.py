"""
Per-class IoU breakdown for a single experiment (default: the overall best,
nndepth_base_b001_p03_mitb2) — pulls its row out of the full 58-experiment
per_class_iou_heatmap.png into its own standalone, easy-to-read chart.

Generates:
  Generated_results/best_result_per_class_iou.png

Usage:
    python scripts/plot_best_result_per_class.py
    python scripts/plot_best_result_per_class.py --experiment baseline_mitb2
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"

DEFAULT_EXPERIMENT = "nndepth_base_b001_p03_mitb2"

# Green (good) -> amber -> red (poor), thresholded on IoU.
def tier_color(iou: float) -> str:
    if iou >= 0.70:
        return "#2E8B45"
    if iou >= 0.50:
        return "#DD8452"
    return "#C0392B"


def load_row(experiment: str) -> dict[str, float]:
    with open(RESULTS_DIR / "per_class_iou.csv") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == experiment:
                return {k: float(v) for k, v in row.items() if k != "experiment" and v != "N/A"}
    raise SystemExit(f"No row for experiment={experiment!r} in per_class_iou.csv")


def load_overall_miou(experiment: str) -> float:
    with open(RESULTS_DIR / "miou_table.csv") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == experiment:
                return float(row["mIoU"])
    raise SystemExit(f"No row for experiment={experiment!r} in miou_table.csv")


def plot(experiment: str) -> None:
    per_class = load_row(experiment)
    overall = load_overall_miou(experiment)

    items = sorted(per_class.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [tier_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    y = range(len(names))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("IoU on ACDC fog val (100 real foggy images)")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"Per-Class IoU — {experiment}\nOverall mIoU = {overall:.4f} (mean of the 19 bars below)",
                 fontsize=13, fontweight="bold")

    ax.axvline(overall, color="#444444", linestyle="--", linewidth=1.2)
    ax.text(overall + 0.01, len(names) - 0.3, f"mean = {overall:.3f}", fontsize=9, color="#444444")

    for yi, v in zip(y, values):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2E8B45", "#DD8452", "#C0392B"]]
    ax.legend(handles, ["strong (≥0.70)", "moderate (0.50–0.70)", "weak (<0.50)"],
              loc="lower right", fontsize=9)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "best_result_per_class_iou.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()
    plot(args.experiment)
