"""
Phase 6 visuals — training curves parsed directly from train.log.

metrics.csv is rebuilt from an in-memory list each process start, so it loses
pre-resume epochs whenever a run was interrupted and resumed. train.log uses
append-mode logging and survives resumes intact, so it's the reliable source
for full-history curves.

Generates:
  results/figures/training_curves.png — train loss + val mIoU vs epoch, one
  line per results/<experiment>/train.log found on disk (auto-discovered)

Usage:
    python scripts/plot_training_curves.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _archive import archive_before_write

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"


def discover_experiments() -> list[str]:
    """Every results/<name>/ with a train.log, baseline first — picks up
    sweep results automatically instead of a hardcoded list going stale."""
    names = sorted(p.parent.name for p in RESULTS_DIR.glob("*/train.log"))
    return sorted(names, key=lambda n: (n != "baseline", n))


_EPOCH_RE = re.compile(r"Epoch (\d+) complete — avg_loss=([\d.]+)")
_VAL_RE = re.compile(r"val mIoU = ([\d.]+)")


def parse_log(log_path: Path):
    """Return (epochs, train_loss, val_epochs, val_miou) from a train.log."""
    epochs, losses = [], []
    val_epochs, val_mious = [], []
    last_epoch = None
    for line in log_path.read_text().splitlines():
        m = _EPOCH_RE.search(line)
        if m:
            last_epoch = int(m.group(1))
            epochs.append(last_epoch)
            losses.append(float(m.group(2)))
            continue
        m = _VAL_RE.search(line)
        if m and last_epoch is not None:
            val_epochs.append(last_epoch)
            val_mious.append(float(m.group(1)))
    return epochs, losses, val_epochs, val_mious


def plot_curves(experiments: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap("nipy_spectral")

    for i, name in enumerate(experiments):
        log_path = RESULTS_DIR / name / "train.log"
        if not log_path.exists():
            continue
        epochs, losses, val_epochs, val_mious = parse_log(log_path)
        color = cmap(i / max(len(experiments) - 1, 1))
        axes[0].plot(epochs, losses, label=name, color=color)
        axes[1].plot(val_epochs, val_mious, label=name, color=color, marker="o", markersize=3)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss (avg per epoch)")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=7, loc="upper right")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cityscapes Val mIoU")
    axes[1].set_title("Validation mIoU (in-domain, Cityscapes val)")
    axes[1].legend(fontsize=6, loc="lower right", ncol=2)

    plt.tight_layout()
    out = FIG_DIR / "training_curves.png"
    archive_before_write(out)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    experiments = discover_experiments()
    if not experiments:
        raise SystemExit(
            "No results/<experiment>/train.log files found — nothing to plot. "
            "Refusing to overwrite the existing training_curves.png."
        )
    plot_curves(experiments)
