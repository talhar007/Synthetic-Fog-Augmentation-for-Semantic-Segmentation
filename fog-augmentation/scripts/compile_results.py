"""
Phase 6 — compile all ACDC evaluation JSONs into a single results table.

Reads results/<experiment>/acdc_eval.json for each experiment and
produces:
  - results/miou_table.csv      (per-experiment mIoU)
  - results/per_class_iou.csv   (per-experiment × per-class IoU)
  - results/summary.md          (markdown table for the paper)

Usage:
    python scripts/compile_results.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

# Experiments that predate fog_prob / mixed-intensity (README's original 11) —
# kept as a labelled "before" group in the summary so the fix's effect is
# visible rather than just re-sorted away.
ORIGINAL_EXPERIMENTS = {
    "baseline",
    "photometric_low", "photometric_med", "photometric_high",
    "depthaware_b0005", "depthaware_b001", "depthaware_b002",
    "combined",
    "nn_depth_b0005", "nn_depth_b001", "nn_depth_b002",
}

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def _family(exp: str) -> str:
    """Coarse technique-family label parsed from the experiment name, for
    grouping/sorting in the summary table."""
    if exp == "baseline" or exp.startswith("baseline_"):
        return "baseline"
    if exp.startswith("combined"):
        return "combined"
    if exp.startswith("photometric"):
        return "photometric"
    if exp.startswith("nndepth_base") or exp.startswith("nn_depth"):
        return "nn_depth"
    if exp.startswith("depthaware"):
        return "depth_aware (stereo)"
    return "other"


def discover_experiments(results_dir: Path) -> list[str]:
    """Every results/<name>/ with an acdc_eval.json, sorted with baseline
    first then alphabetically — picks up sweep results automatically, no
    hardcoded list to keep in sync.

    Excludes poc_* dirs: those are 10-epoch backbone-POC smoke tests (see
    results/poc_run.log), not comparable to the 50-epoch full-sweep runs."""
    names = sorted(
        p.parent.name for p in results_dir.glob("*/acdc_eval.json")
        if not p.parent.name.startswith("poc_")
    )
    return sorted(names, key=lambda n: (n != "baseline", n))


def load_results(results_dir: Path, experiments: list[str]) -> dict[str, dict]:
    data = {}
    for exp in experiments:
        json_path = results_dir / exp / "acdc_eval.json"
        if json_path.exists():
            with open(json_path) as f:
                data[exp] = json.load(f)
        else:
            data[exp] = None
    return data


def compile(results_dir: str = "results") -> None:
    rd = Path(results_dir)
    experiments = discover_experiments(rd)
    all_results = load_results(rd, experiments)

    # ── mIoU table (insertion order = discovery order) ─────────────────────────
    miou_path = rd / "miou_table.csv"
    archive_before_write(miou_path)
    with open(miou_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mIoU", "family", "is_original"])
        for exp, res in all_results.items():
            miou = f"{res['mIoU']:.4f}" if res else "N/A"
            writer.writerow([exp, miou, _family(exp), exp in ORIGINAL_EXPERIMENTS])
    print(f"Saved → {miou_path}")

    # ── Per-class table ───────────────────────────────────────────────────────
    pc_path = rd / "per_class_iou.csv"
    archive_before_write(pc_path)
    with open(pc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + CLASSES)
        for exp, res in all_results.items():
            if res is None:
                writer.writerow([exp] + ["N/A"] * len(CLASSES))
            else:
                row = [exp] + [f"{res['per_class'].get(c, float('nan')):.4f}" for c in CLASSES]
                writer.writerow(row)
    print(f"Saved → {pc_path}")

    # ── Markdown summary ─────────────────────────────────────────────────────
    # Ranked table (all experiments, best first) + a labelled "original 11"
    # sub-table so the before/after effect of the fog_prob/mixing fix reads
    # directly off this file.
    ranked = sorted(
        (exp for exp, res in all_results.items() if res is not None),
        key=lambda e: all_results[e]["mIoU"], reverse=True,
    )
    missing = [exp for exp, res in all_results.items() if res is None]

    md_path = rd / "summary.md"
    archive_before_write(md_path)
    with open(md_path, "w") as f:
        f.write("# Results: mIoU on ACDC Fog Val\n\n")
        f.write("## All experiments, ranked\n\n")
        f.write("| Rank | Experiment | mIoU | Family | Original (pre-fix) |\n")
        f.write("|---|---|---|---|---|\n")
        for i, exp in enumerate(ranked, 1):
            res = all_results[exp]
            orig = "yes" if exp in ORIGINAL_EXPERIMENTS else "—"
            f.write(f"| {i} | {exp} | {res['mIoU']:.4f} | {_family(exp)} | {orig} |\n")
        if missing:
            f.write(f"\n_Not yet evaluated: {', '.join(missing)}_\n")
        f.write("\n_Generated by scripts/compile_results.py_\n")
    print(f"Saved → {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    compile(args.results_dir)
