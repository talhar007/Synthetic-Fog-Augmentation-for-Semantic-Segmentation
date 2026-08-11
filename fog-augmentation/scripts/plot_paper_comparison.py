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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"

# (label, mIoU, group)
# groups: "ours" | "paper_no_labels" | "paper_oracle"
ENTRIES = [
    ("Our baseline\n(resnet101, Cityscapes-only, no fog aug)", 41.4, "ours"),
    ("Paper: DeepLabv3+ source model\n(Cityscapes-only, no fog aug)", 45.7, "paper_no_labels"),
    ("Our best, resnet101\n(NN-depth Base, fog_prob=0.3)", 47.8, "ours_best"),
    ("Paper: DMAda\n(nighttime-video-adjacent adaptation)", 50.7, "paper_no_labels"),
    ("Paper: CMAda\n(fog-specialized, real Foggy Zurich)", 51.2, "paper_no_labels"),
    ("Paper: GCMA\n(best \"no real ACDC labels\" method)", 52.4, "paper_no_labels"),
    ("Our baseline, mit_b2/FPN\n(Cityscapes-only, no fog aug)", 50.7, "ours_mitb2"),
    ("Our best, mit_b2/FPN\n(NN-depth Base, β=0.01, fog_prob=0.3)", 60.3, "ours_mitb2_best"),
    ("Paper oracle\n(full supervision, real ACDC fog labels)", 68.7, "paper_oracle"),
    ("Paper uber oracle\n(supervised on all 4 conditions)", 69.1, "paper_oracle"),
]

GROUP_COLOR = {
    "ours": "#4C72B0",
    "ours_best": "#55A868",
    "paper_no_labels": "#DD8452",
    "ours_mitb2": "#8172B3",
    "ours_mitb2_best": "#2E7D32",
    "paper_oracle": "#937860",
}

GROUP_LABEL = {
    "ours": "Ours, resnet101 (baseline)",
    "ours_best": "Ours, resnet101 (best fog augmentation)",
    "paper_no_labels": "Paper — no real ACDC labels used",
    "ours_mitb2": "Ours, mit_b2/FPN (baseline)",
    "ours_mitb2_best": "Ours, mit_b2/FPN (best fog augmentation)",
    "paper_oracle": "Paper — full supervision on real ACDC labels",
}


def plot():
    entries = sorted(ENTRIES, key=lambda e: e[1])
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [GROUP_COLOR[e[2]] for e in entries]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU on ACDC fog (%)")
    ax.set_xlim(0, max(values) * 1.15)

    for bar, v in zip(bars, values):
        ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                 va="center", fontsize=9, fontweight="bold")

    fig.suptitle("Our Fog Augmentation Results vs. ACDC Paper Benchmarks (Fog)", fontsize=13, y=0.99)

    handles = [Patch(color=c, label=GROUP_LABEL[g]) for g, c in GROUP_COLOR.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8)

    fig.text(
        0.02, 0.01,
        "Caveat: our numbers are on ACDC fog VAL (100 imgs, SMP backend, ImageNet-pretrained encoder, 50 epochs, 1024x512 crops).\n"
        "Paper numbers are on ACDC fog TEST (500 imgs, labels withheld, official eval server, 60 epochs, up to 2048x1024). Not decimal-precision comparable.",
        fontsize=7.5, color="#555555", ha="left", va="bottom",
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.88))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "acdc_paper_comparison.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot()
