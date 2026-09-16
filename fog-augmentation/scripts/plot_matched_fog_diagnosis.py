"""
Visualizes the matched-fog Cityscapes-val diagnosis (Section VI-B of gp-paper.tex):
grouped bars of clear-val vs matched-fog-val mIoU for all 11 Phase-1 checkpoints,
showing that most "underperforming" configs actually specialize toward their own
fog domain rather than failing to fit the task.

Generates:
  Generated_results/matched_fog_diagnosis.png

Usage:
    python scripts/plot_matched_fog_diagnosis.py
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
OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"


def plot():
    with open(RESULTS_DIR / "matched_fog_cityscapes_val.csv") as f:
        rows = list(csv.DictReader(f))

    names = [r["experiment"] for r in rows]
    clear = [float(r["clear_cityscapes_val_miou"]) for r in rows]
    fogged = [float(r["matched_fog_cityscapes_val_miou"]) for r in rows]

    y = np.arange(len(names))
    h = 0.36

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y + h / 2, clear, h, label="Clear Cityscapes-val", color="#4C72B0")
    ax.barh(y - h / 2, fogged, h, label="Matched-fog Cityscapes-val", color="#DD8452")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU")
    ax.set_xlim(0, 0.85)
    ax.set_title("Phase 1: Clear vs. Own-Matched-Fog Cityscapes-Val mIoU\n"
                 "Overspecialization to the training fog domain, not simple underfitting",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.7688, color="#444444", linestyle="--", linewidth=1)
    ax.text(0.7688 + 0.005, len(names) - 0.5, "no-fog baseline (0.769)", fontsize=8, color="#444444")

    for yi, (c, f) in zip(y, zip(clear, fogged)):
        ax.text(c + 0.01, yi + h / 2, f"{c:.2f}", va="center", fontsize=7.5)
        ax.text(f + 0.01, yi - h / 2, f"{f:.2f}", va="center", fontsize=7.5, fontweight="bold")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "matched_fog_diagnosis.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    plot()
