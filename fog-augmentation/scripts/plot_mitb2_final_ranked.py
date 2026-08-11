"""
Final MiT-B2/FPN sweep leaderboard — all 24 configs (23 fog + baseline),
ranked, colored by technique family, with reference lines for the best
ResNet-101 sweep run and the MiT-B2 baseline (no fog). Presentation-oriented
sibling to plot_backbone_full_comparison.py (which pairs each config against
its ResNet-101 counterpart) — this one just shows the finished MiT-B2
leaderboard on its own.

Generates:
  Generated_results/mitb2_sweep_final_ranked.png

Usage:
    python scripts/plot_mitb2_final_ranked.py
"""
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

FAMILY_COLOR = {
    "baseline": "#444444",
    "photometric": "#4C72B0",
    "depth_aware (stereo)": "#DD8452",
    "combined": "#4A3AA7",
    "nn_depth": "#2E8B45",
}


def load():
    with open(RESULTS_DIR / "miou_table.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["mIoU"] != "N/A" and r["experiment"].endswith("_mitb2")]
    rows.sort(key=lambda r: float(r["mIoU"]), reverse=True)
    return rows


def load_resnet_best():
    with open(RESULTS_DIR / "miou_table.csv") as f:
        rows = [float(r["mIoU"]) for r in csv.DictReader(f)
                if r["mIoU"] != "N/A" and not r["experiment"].endswith("_mitb2")]
    return max(rows)


def plot():
    rows = load()
    resnet_best = load_resnet_best()
    baseline_miou = next(float(r["mIoU"]) for r in rows if r["family"] == "baseline")

    names = [r["experiment"].removesuffix("_mitb2") for r in rows]
    values = [float(r["mIoU"]) for r in rows]
    colors = [FAMILY_COLOR.get(r["family"], "#B0B0B0") for r in rows]

    fig, ax = plt.subplots(figsize=(11, max(8, 0.34 * len(names))))
    y = range(len(names))
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mIoU on ACDC fog val (100 real foggy images)")
    ax.set_title(f"MiT-B2 / FPN sweep — {len(names)} of {len(names)} runs complete (ranked)",
                 fontsize=13, fontweight="bold")

    ax.axvline(resnet_best, color="#888888", linestyle=":", linewidth=1.4)
    ax.axvline(baseline_miou, color="#444444", linestyle="--", linewidth=1.4)

    for yi, v in zip(y, values):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    labels = list(FAMILY_COLOR.keys())
    handles += [plt.Line2D([0], [0], color="#888888", linestyle=":", linewidth=1.4),
                plt.Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.4)]
    labels += [f"best ResNet-101 run ({resnet_best:.3f})", f"MiT-B2 baseline ({baseline_miou:.3f})"]
    ax.legend(handles, labels, loc="lower right", fontsize=8.5)
    ax.set_xlim(min(values) - 0.03, max(values) * 1.08)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "mitb2_sweep_final_ranked.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot()
