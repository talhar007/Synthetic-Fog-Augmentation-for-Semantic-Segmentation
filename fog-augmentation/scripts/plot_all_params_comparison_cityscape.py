"""
In-domain counterpart to plot_results.py's miou_comparison.png (which ranks
every experiment by ACDC *fog* mIoU): this ranks by final-epoch Cityscapes
*val* mIoU instead — the in-domain number, read from each experiment's
metrics.csv.

Scope note: only the 24 MiT-B2 sweep configs are covered. The 34 ResNet-101
experiments had their metrics.csv/train.log deleted during an earlier file
cleanup, so their Cityscapes-val curve is not recoverable — only their final
ACDC score (acdc_eval.json) survived. Says so directly on the chart rather
than silently showing a partial comparison.

Generates:
  Generated_results/all_params_comparison_cityscape.png

Usage:
    python scripts/plot_all_params_comparison_cityscape.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent.parent / "Generated_results"

FAMILY_COLOR = {
    "baseline": "#444444",
    "photometric": "#4C72B0",
    "depthaware": "#DD8452",
    "combined": "#937860",
    "nn_depth": "#55A868",
}


def family_of(name: str) -> str:
    """name is already stripped of the _mitb2 suffix."""
    if name == "baseline":
        return "baseline"
    if name.startswith("nndepth_base"):
        return "nn_depth"
    for fam in ("photometric", "depthaware", "combined"):
        if name.startswith(fam):
            return fam
    return "other"


def load_final_val_miou(exp_dir: Path) -> float | None:
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["val_miou"])


def load_acdc_miou(name: str) -> float | None:
    with open(RESULTS_DIR / "miou_table.csv") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == name and row["mIoU"] != "N/A":
                return float(row["mIoU"])
    return None


def plot():
    exp_dirs = sorted(
        p.parent for p in RESULTS_DIR.glob("*_mitb2/metrics.csv")
        if not p.parent.name.startswith("poc_")
    )
    names = [d.name.removesuffix("_mitb2") for d in exp_dirs]
    cs_values = [load_final_val_miou(d) for d in exp_dirs]
    acdc_values = [load_acdc_miou(d.name) for d in exp_dirs]

    order = sorted(range(len(names)), key=lambda i: cs_values[i], reverse=True)
    names = [names[i] for i in order]
    cs_values = [cs_values[i] for i in order]
    acdc_values = [acdc_values[i] for i in order]
    colors = [FAMILY_COLOR.get(family_of(n), "#B0B0B0") for n in names]

    fig, ax = plt.subplots(figsize=(max(10, 0.42 * len(names)), 6.5))
    x = range(len(names))
    bars = ax.bar(x, cs_values, color=colors, label="Cityscapes val (in-domain)")
    ax.scatter(x, acdc_values, color="black", marker="D", s=18, zorder=3, label="ACDC fog val (for reference)")

    ax.set_ylabel("mIoU")
    ax.set_title("MiT-B2 Sweep — Cityscapes-Val mIoU (in-domain) vs. ACDC Fog mIoU\n"
                 "ResNet-101 experiments excluded: their per-epoch metrics.csv/train.log were deleted in an earlier cleanup",
                 fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.0)

    for xi, v in zip(x, cs_values):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    handles = [Patch(color=c, label=fam) for fam, c in FAMILY_COLOR.items()]
    handles.append(plt.Line2D([0], [0], marker="D", color="black", linestyle="None",
                               markersize=5, label="ACDC fog val (for reference)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "all_params_comparison_cityscape.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}  ({len(names)} MiT-B2 configs)")


if __name__ == "__main__":
    plot()
