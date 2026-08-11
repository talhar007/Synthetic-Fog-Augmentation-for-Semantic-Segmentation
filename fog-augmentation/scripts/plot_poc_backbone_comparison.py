"""
Backbone POC results — resnet101/DeepLabV3+ vs mit_b2/FPN (SegFormer's
MixTransformer encoder), matched at 10 epochs each, at baseline (no fog) and
our best full-sweep fog config (nndepth_base_b0005_p30 settings).

Motivated to try a transformer backbone (mit_b2/SegFormer) instead of the
ACDC paper's own CNN-only architectures (DeepLabv2/v3+, RefineNet, DANet,
HRNet — no transformer encoder is evaluated anywhere in that paper). This
POC is the actual evidence: an empirical 10-epoch comparison, not a claim
carried over from the ACDC paper.

Generates:
  Generated_results/poc_backbone_comparison.png

Usage:
    python scripts/plot_poc_backbone_comparison.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"

CONDITIONS = ["Baseline\n(no fog)", "Best fog config\n(NN-depth Base, fog_prob=0.3)"]
RESNET101 = [0.4624, 0.4463]
MIT_B2 = [0.5229, 0.5471]

FULL_SWEEP_BEST = 0.4780  # resnet101, 50 epochs, nndepth_base_b0005_p30 (main sweep)


def plot():
    x = np.arange(len(CONDITIONS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, RESNET101, width, label="resnet101 / DeepLabV3+", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, MIT_B2, width, label="mit_b2 / FPN (SegFormer encoder)", color="#55A868")

    ax.axhline(FULL_SWEEP_BEST, color="#937860", linestyle="--", linewidth=1.3)
    ax.text(len(CONDITIONS) - 0.5, FULL_SWEEP_BEST + 0.006,
             f"Full 50-epoch resnet101 sweep winner: {FULL_SWEEP_BEST:.3f}",
             ha="right", fontsize=8, color="#937860")

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("mIoU on ACDC fog val")
    ax.set_title("Backbone POC — matched at 10 epochs each")
    ax.set_ylim(0, max(MIT_B2) * 1.25)
    ax.legend(loc="upper left", fontsize=9)

    for bars, values in [(bars1, RESNET101), (bars2, MIT_B2)]:
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, (r, m) in enumerate(zip(RESNET101, MIT_B2)):
        delta = m - r
        ax.annotate(
            f"+{delta:.3f}\n({delta/r*100:+.0f}%)",
            xy=(x[i], max(r, m) + 0.04), ha="center", fontsize=9, color="#2c5f2d", fontweight="bold",
        )

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "poc_backbone_comparison.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot()
