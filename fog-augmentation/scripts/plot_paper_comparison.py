"""
Places our best fog-augmentation results in context against numbers
reported in the ACDC paper (Sakaridis et al., ACDC: The Adverse Conditions
Dataset with Correspondences, arXiv:2104.13395v5), for the fog condition.

Caveats (see annotation on the figure): our numbers are on ACDC's fog
*validation* split (100 images, DeepLabv3+/SMP, ImageNet-pretrained
backbone, 50 epochs, 1024x512 crops); the paper's numbers are on the fog
*test* split (500 images, labels withheld, official eval server, 60
epochs, up to 2048x1024). Not decimal-precision comparable, but
informative side by side.

Generates:
  Generated_results/acdc_paper_comparison.png

Usage:
    python scripts/plot_paper_comparison.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"

# (label, mIoU, group)
# groups: "ours" | "paper_no_labels" | "paper_oracle"
ENTRIES = [
    ("Our baseline\n(Cityscapes-only, no fog aug)", 41.4, "ours"),
    ("Paper: DeepLabv3+ source model\n(Cityscapes-only, no fog aug)", 45.7, "paper_no_labels"),
    ("Our best\n(NN-depth Base, fog_prob=0.3)", 47.8, "ours_best"),
    ("Paper: DMAda\n(nighttime-video-adjacent adaptation)", 50.7, "paper_no_labels"),
    ("Paper: CMAda\n(fog-specialized, real Foggy Zurich)", 51.2, "paper_no_labels"),
    ("Paper: GCMA\n(best \"no real ACDC labels\" method)", 52.4, "paper_no_labels"),
    ("Paper oracle\n(full supervision, real ACDC fog labels)", 68.7, "paper_oracle"),
    ("Paper uber oracle\n(supervised on all 4 conditions)", 69.1, "paper_oracle"),
]

GROUP_COLOR = {
    "ours": "#4C72B0",
    "ours_best": "#55A868",
    "paper_no_labels": "#DD8452",
    "paper_oracle": "#937860",
}

GROUP_LABEL = {
    "ours": "Ours (baseline)",
    "ours_best": "Ours (best fog augmentation)",
    "paper_no_labels": "Paper — no real ACDC labels used",
    "paper_oracle": "Paper — full supervision on real ACDC labels",
}


def plot():
    labels = [e[0] for e in ENTRIES]
    values = [e[1] for e in ENTRIES]
    colors = [GROUP_COLOR[e[2]] for e in ENTRIES]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU on ACDC fog (%)")
    ax.set_title("Our Fog Augmentation Results vs. ACDC Paper Benchmarks (Fog)")
    ax.set_xlim(0, max(values) * 1.15)

    for bar, v in zip(bars, values):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                 va="center", fontsize=9, fontweight="bold")

    handles = [Patch(color=c, label=GROUP_LABEL[g]) for g, c in GROUP_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.text(
        0.0, -1.35,
        "Caveat: our numbers are on ACDC fog VAL (100 imgs, DeepLabv3+/SMP, ImageNet-pretrained backbone, 50 epochs, 1024x512 crops).\n"
        "Paper numbers are on ACDC fog TEST (500 imgs, labels withheld, official eval server, 60 epochs, up to 2048x1024). Not decimal-precision comparable.",
        transform=ax.get_yaxis_transform(), fontsize=7.5, color="#555555", ha="left", va="top",
    )

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "acdc_paper_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot()
