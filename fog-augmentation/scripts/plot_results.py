"""
Phase 6 visuals — summary figures for the final results.

Generates:
  results/figures/miou_comparison.png   — sorted bar chart, all 11 experiments
  results/figures/per_class_heatmap.png — per-class IoU heatmap, all 11 x 19 classes

Usage:
    python scripts/plot_results.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"

FAMILY_COLOR = {
    "baseline": "#444444",
    "photometric": "#4C72B0",
    "depthaware": "#DD8452",
    "combined": "#937860",
    "nn_depth": "#55A868",
    "other": "#B0B0B0",
}


def family_of(name: str) -> str:
    if name == "baseline" or name.startswith("baseline_"):
        return "baseline"
    if name.startswith("nndepth_base") or name.startswith("nn_depth"):
        return "nn_depth"
    for fam in ("photometric", "depthaware", "combined"):
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

    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(names)), 6))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("mIoU on ACDC Fog Val")
    ax.set_title("Fog Augmentation Comparison — ACDC Fog Val mIoU")
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, max(values) * 1.15)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7, rotation=90)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=fam) for fam, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "miou_comparison.png"
    archive_before_write(out)
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
    avg_miou = data.mean(axis=1, keepdims=True)
    plot_data = np.hstack([data, avg_miou])
    columns = classes + ["mIoU avg"]

    fig, ax = plt.subplots(figsize=(14.8, max(6, 0.35 * len(names))))
    im = ax.imshow(plot_data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Per-Class IoU on ACDC Fog Val")

    # Separator between per-class columns and the aggregate mIoU column
    ax.axvline(len(classes) - 0.5, color="black", linewidth=1.5)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=5)
        ax.text(len(classes), i, f"{avg_miou[i, 0]:.3f}", ha="center", va="center",
                 fontsize=6, fontweight="bold")

    fig.colorbar(im, ax=ax, label="IoU")
    plt.tight_layout()
    out = FIG_DIR / "per_class_heatmap.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_miou_bar()
    plot_per_class_heatmap()
