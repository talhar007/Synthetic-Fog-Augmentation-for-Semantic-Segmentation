"""
Full-sweep backbone comparison — every one of the 24 fog_prob-sweep configs
(23 fog configs + baseline), resnet101/DeepLabV3+ vs mit_b2/FPN, both trained
the full 50 epochs. Follow-up to the 2-config, 10-epoch backbone POC
(poc_backbone_comparison.png) — this is the complete, fully-converged result.

Generates:
  Generated_results/backbone_full_sweep_comparison.png

Usage:
    python scripts/plot_backbone_full_comparison.py
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


def find_pairs():
    rows = {r["experiment"]: float(r["mIoU"]) for r in csv.DictReader(open(RESULTS_DIR / "miou_table.csv"))
            if r["mIoU"] != "N/A"}
    pairs = []
    for name, miou in rows.items():
        if name.endswith("_mitb2"):
            continue
        mitb2_name = f"{name}_mitb2"
        if mitb2_name in rows:
            pairs.append((name, miou, rows[mitb2_name]))
    pairs.sort(key=lambda p: p[2] - p[1], reverse=True)
    return pairs


def plot():
    pairs = find_pairs()
    names = [p[0] for p in pairs]
    resnet_vals = [p[1] for p in pairs]
    mitb2_vals = [p[2] for p in pairs]
    deltas = [m - r for _, r, m in pairs]

    y = np.arange(len(names))
    height = 0.38

    fig, ax = plt.subplots(figsize=(11, max(8, 0.32 * len(names))))
    ax.barh(y + height / 2, resnet_vals, height, label="resnet101 / DeepLabV3+", color="#4C72B0")
    ax.barh(y - height / 2, mitb2_vals, height, label="mit_b2 / FPN (SegFormer encoder)", color="#55A868")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU on ACDC fog val")
    ax.set_title(f"Backbone Comparison, Full 50-Epoch Sweep — all {len(pairs)} configs improve with mit_b2\n"
                 f"(mean Δ = +{np.mean(deltas):.3f}, range +{min(deltas):.3f} to +{max(deltas):.3f})")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, max(mitb2_vals) * 1.12)

    for yi, (r, m) in zip(y, zip(resnet_vals, mitb2_vals)):
        ax.text(r + 0.005, yi + height / 2, f"{r:.3f}", va="center", fontsize=6.5)
        ax.text(m + 0.005, yi - height / 2, f"{m:.3f}", va="center", fontsize=6.5, fontweight="bold")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "backbone_full_sweep_comparison.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot()
