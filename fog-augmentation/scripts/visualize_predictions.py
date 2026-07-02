"""
Phase 6 visuals — qualitative comparison of predictions on real ACDC fog images.

For a handful of ACDC fog val images, shows: input | ground truth | prediction
for baseline + the best-performing condition from each fog technique family.

Usage:
    python scripts/visualize_predictions.py --acdc-root /path/to/acdc --n 3
"""
import argparse
import sys
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.acdc import ACDCDataset
from data.cityscapes import CityscapesDataset
from models import build_model, forward_eval

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# One representative experiment per technique family, plus baseline.
EXPERIMENTS = [
    "baseline",
    "photometric_low",
    "depthaware_b0005",
    "nn_depth_b0005",
    "combined",
]


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
    for name in EXPERIMENTS:
        ckpt = results_dir / name / "best.pth"
        if not ckpt.exists():
            print(f"  skip {name} (no checkpoint)")
            continue
        m = build_model(
            "configs/_base_/deeplabv3plus_r101.py",
            checkpoint=str(ckpt),
        ).to(device)
        m.eval()
        models[name] = m
        print(f"  loaded {name}")

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
    plt.savefig(args.save, dpi=120)
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
