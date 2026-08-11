"""
Phase 6 visuals — qualitative comparison of predictions on real ACDC fog images.

For a handful of ACDC fog val images, shows: input | ground truth | prediction
for baseline + the best-performing condition from each fog technique family.

Usage:
    python scripts/visualize_predictions.py --acdc-root /path/to/acdc --n 3
"""
import argparse
import csv
import sys
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.acdc import ACDCDataset
from data.cityscapes import CityscapesDataset
from models import build_model, forward_eval
from _archive import archive_before_write

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fallback when results/miou_table.csv doesn't exist yet (pre-compile_results.py run).
_FALLBACK_EXPERIMENTS = [
    "baseline",
    "photometric_low",
    "depthaware_b0005",
    "nn_depth_b0005",
    "combined",
]


def select_experiments(results_dir: Path) -> list[str]:
    """The best-mIoU experiment per technique family (including baseline),
    read from miou_table.csv (written by compile_results.py, which tags each
    row with a `family` column) — keeps this figure showing the current
    champion per family (across all backbones) instead of a name hardcoded
    before the sweep existed."""
    table_path = results_dir / "miou_table.csv"
    if not table_path.exists():
        return _FALLBACK_EXPERIMENTS

    best_per_family: dict[str, tuple[str, float]] = {}
    with open(table_path) as f:
        for row in csv.DictReader(f):
            if row["mIoU"] == "N/A":
                continue
            family, miou = row["family"], float(row["mIoU"])
            if family not in best_per_family or miou > best_per_family[family][1]:
                best_per_family[family] = (row["experiment"], miou)

    # baseline first, then the rest in descending mIoU order
    names = sorted(best_per_family, key=lambda f: (f != "baseline", -best_per_family[f][1]))
    return [best_per_family[f][0] for f in names]


def model_arch(results_dir: Path, name: str) -> tuple[str, str]:
    """Read (encoder_name, architecture) that `name` was actually trained
    with, from its saved config.yaml — defaults match the original
    resnet101/DeepLabV3+ configs, which predate these keys existing."""
    cfg_path = results_dir / name / "config.yaml"
    if not cfg_path.exists():
        return "resnet101", "deeplabv3plus"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    return model_cfg.get("encoder_name", "resnet101"), model_cfg.get("architecture", "deeplabv3plus")


def build_eval_transform() -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None,
                      pad_height_divisor=32, pad_width_divisor=32,
                      position="top_left", border_mode=0, value=0, mask_value=255),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acdc-root", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n", type=int, default=3, help="number of ACDC val images")
    parser.add_argument("--save", default="results/figures/qualitative_predictions.png")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_eval_transform()
    raw_ds = ACDCDataset(root=args.acdc_root, split="val", geo_transform=None)

    indices = np.linspace(0, len(raw_ds) - 1, args.n, dtype=int)

    print("Loading models...")
    models = {}
    for name in select_experiments(results_dir):
        ckpt = results_dir / name / "best.pth"
        if not ckpt.exists():
            print(f"  skip {name} (no checkpoint)")
            continue
        encoder_name, architecture = model_arch(results_dir, name)
        m = build_model(
            "configs/_base_/deeplabv3plus_r101.py",
            checkpoint=str(ckpt),
            encoder_name=encoder_name,
            architecture=architecture,
        ).to(device)
        m.eval()
        models[name] = m
        print(f"  loaded {name} (encoder={encoder_name}, architecture={architecture})")

    ncols = 2 + len(models)  # input, GT, + one column per model
    fig, axes = plt.subplots(len(indices), ncols, figsize=(4 * ncols, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        sample = raw_ds[int(idx)]
        img, label = sample["image"], sample["label"]
        H, W = img.shape[:2]

        transformed = transform(image=img, mask=label)
        img_t = transformed["image"].unsqueeze(0).to(device)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Input (real fog)" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(CityscapesDataset.decode_target(label))
        axes[row, 1].set_title("Ground truth" if row == 0 else "")
        axes[row, 1].axis("off")

        for col, name in enumerate(models, start=2):
            with torch.no_grad():
                pred = forward_eval(models[name], img_t)[0].cpu().numpy()
            pred = pred[:H, :W]  # crop padded border back off
            axes[row, col].imshow(CityscapesDataset.decode_target(pred))
            axes[row, col].set_title(name if row == 0 else "")
            axes[row, col].axis("off")

    plt.tight_layout()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(args.save)
    plt.savefig(args.save, dpi=120)
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
