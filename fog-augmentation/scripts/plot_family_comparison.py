"""
High-level technique comparison — one bar per fog-generation family (baseline,
photometric, depth-aware stereo, NN-depth, combined), each showing that
family's single best-performing config from the full sweep.

Complements plot_results.py's miou_comparison.png (all 34 individual configs):
this is the "which technique wins" view, that one is the "every parameter"
view.

Generates:
  Generated_results/technique_family_comparison.png

Usage:
    python scripts/plot_family_comparison.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"

FAMILY_COLOR = {
    "baseline": "#444444",
    "photometric": "#4C72B0",
    "depth_aware (stereo)": "#DD8452",
    "combined": "#937860",
    "nn_depth": "#55A868",
}

FAMILY_ORDER = ["baseline", "photometric", "depth_aware (stereo)", "nn_depth", "combined"]


def best_per_family():
    rows = list(csv.DictReader(open(RESULTS_DIR / "miou_table.csv")))
    rows = [r for r in rows if r["mIoU"] != "N/A"]

    best = {}
    for r in rows:
        fam = r["family"]
        miou = float(r["mIoU"])
        if fam not in best or miou > best[fam][1]:
            best[fam] = (r["experiment"], miou)
    return best


def plot():
    best = best_per_family()
    families = [f for f in FAMILY_ORDER if f in best]
    names = [best[f][0] for f in families]
    values = [best[f][1] for f in families]
    colors = [FAMILY_COLOR[f] for f in families]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(families, values, color=colors, width=0.6)
    ax.set_ylabel("Best ACDC Fog Val mIoU")
    ax.set_title("Fog Augmentation Technique Comparison\n(best config per family)")
    ax.set_ylim(0, max(values) * 1.25)

    baseline_miou = best["baseline"][1]
    ax.axhline(baseline_miou, color="#444444", linestyle="--", linewidth=1, alpha=0.6)

    for bar, v, name in zip(bars, values, names):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.01, name,
                 ha="center", va="bottom", fontsize=7, rotation=90, color="white")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "technique_family_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    plot()
