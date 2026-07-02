"""
Phase 6 visuals — summary figures for the final results.

Generates:
  results/figures/miou_comparison.png   — sorted bar chart, all 11 experiments
  results/figures/per_class_heatmap.png — per-class IoU heatmap, all 11 x 19 classes

Usage:
    python scripts/plot_results.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"

FAMILY_COLOR = {
    "baseline": "#444444",
    "photometric": "#4C72B0",
    "depthaware": "#DD8452",
    "combined": "#937860",
    "nn_depth": "#55A868",
}


def family_of(name: str) -> str:
    for fam in ("photometric", "depthaware", "nn_depth", "combined", "baseline"):
        if name.startswith(fam):
            return fam
    return "other"


def plot_miou_bar():
    rows = list(csv.DictReader(open(RESULTS_DIR / "miou_table.csv")))
    rows = [r for r in rows if r["mIoU"] != "N/A"]
    rows.sort(key=lambda r: float(r["mIoU"]), reverse=True)

    names = [r["experiment"] for r in rows]
    values = [float(r["mIoU"]) for r in rows]
    colors = [FAMILY_COLOR[family_of(n)] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("mIoU on ACDC Fog Val")
    ax.set_title("Fog Augmentation Comparison — ACDC Fog Val mIoU")
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, max(values) * 1.15)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=fam) for fam, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "miou_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


def plot_per_class_heatmap():
    with open(RESULTS_DIR / "per_class_iou.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        classes = header[1:]
        names, data = [], []
        for row in reader:
            if "N/A" in row:
                continue
            names.append(row[0])
            data.append([float(v) for v in row[1:]])

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Per-Class IoU on ACDC Fog Val")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6)

    fig.colorbar(im, ax=ax, label="IoU")
    plt.tight_layout()
    out = FIG_DIR / "per_class_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_miou_bar()
    plot_per_class_heatmap()
