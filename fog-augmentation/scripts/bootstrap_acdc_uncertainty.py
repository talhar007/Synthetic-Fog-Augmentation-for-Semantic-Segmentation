"""
Bootstrap uncertainty estimate over the 100-image ACDC fog val set, for a
representative subset of configurations. Addresses reviewer feedback: with
58 single-seed experiments, small ranking differences shouldn't be read as
meaningful, and the reviewer suggested (as a cheaper alternative to full
multi-seed retraining) bootstrapping the ACDC-val images to estimate how
much the reported mIoU could plausibly vary just from which 100 images
happened to be in the evaluation set. This does NOT replace multi-seed
retraining (it says nothing about training-run-to-training-run variance),
only about finite-evaluation-set variance.

Method: run each checkpoint once over all 100 ACDC val images, keep a
PER-IMAGE confusion matrix (not just the aggregate), then resample images
with replacement B times, summing confusion matrices per resample and
recomputing mIoU from the summed matrix each time (matching how the
official mIoU is computed from one pooled confusion matrix, not a mean of
per-image mIoUs).

Usage:
    python scripts/bootstrap_acdc_uncertainty.py
"""
import csv
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.acdc import ACDCDataset
from models import build_model, forward_eval
from visualize_predictions import model_arch
from _archive import archive_before_write

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
NUM_CLASSES = 19
IGNORE_INDEX = 255
N_BOOTSTRAP = 2000
SEED = 42

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The 5 representative configurations the reviewer suggested.
EXPERIMENTS = [
    "baseline",                     # ResNet-101/DeepLabV3+ no-fog baseline
    "baseline_mitb2",                # MiT-B2/FPN no-fog baseline
    "nndepth_base_b001_p03_mitb2",   # best NN-depth configuration (overall)
    "photometric_med_p03_mitb2",     # best photometric configuration (overall)
    "depthaware_mixed_p03_mitb2",    # best stereo depth-aware configuration (overall)
]


def build_eval_transform() -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=None, min_width=None,
                      pad_height_divisor=32, pad_width_divisor=32,
                      border_mode=0, value=0, mask_value=255),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def confusion_matrix(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    mask = target != IGNORE_INDEX
    pred, target = pred[mask], target[mask]
    pred = np.clip(pred, 0, NUM_CLASSES - 1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    return cm


def miou_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    denom = tp + fn + fp
    iou = np.where(denom > 0, tp / denom, np.nan)
    return float(np.nanmean(iou))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)

    with open("configs/baseline.yaml") as f:
        acdc_root = yaml.safe_load(f)["data"]["acdc_root"]
    transform = build_eval_transform()
    raw_ds = ACDCDataset(root=acdc_root, split="val", geo_transform=None)
    n_images = len(raw_ds)
    print(f"ACDC fog val: {n_images} images, {N_BOOTSTRAP} bootstrap resamples per config")

    rows = []
    for name in EXPERIMENTS:
        ckpt_path = RESULTS_DIR / name / "best.pth"
        encoder_name, architecture = model_arch(RESULTS_DIR, name)
        model = build_model(
            "configs/_base_/deeplabv3plus_r101.py",
            checkpoint=str(ckpt_path),
            encoder_name=encoder_name,
            architecture=architecture,
        ).to(device)
        model.eval()
        print(f"Loaded {name} (encoder={encoder_name}, architecture={architecture})")

        # NOTE: we deliberately do NOT crop the prediction back to the original
        # H,W and compare against the padded mask instead (mask_value=255 in the
        # pad region, so it's excluded from the confusion matrix regardless of
        # padding position). This exactly matches src/evaluate.py's own
        # methodology. An earlier version of this script cropped pred[:H, :W]
        # against the *unpadded* label while PadIfNeeded used its default
        # center-anchored padding (no position="top_left") -- a real bug that
        # silently misaligned prediction and ground truth and produced mIoU
        # values ~1-2 points below the official evaluate.py numbers for every
        # config. Comparing against the padded mask sidesteps the issue
        # entirely rather than relying on a specific padding position.
        per_image_cm = np.zeros((n_images, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for i in range(n_images):
                sample = raw_ds[i]
                img, label = sample["image"], sample["label"]
                t = transform(image=img, mask=label)
                img_t = t["image"].unsqueeze(0).to(device)
                padded_label = t["mask"].numpy()
                pred = forward_eval(model, img_t)[0].cpu().numpy()
                per_image_cm[i] = confusion_matrix(pred, padded_label)

        official_miou = miou_from_confusion(per_image_cm.sum(axis=0))

        boot_mious = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n_images, size=n_images)
            boot_mious[b] = miou_from_confusion(per_image_cm[idx].sum(axis=0))

        lo, hi = np.percentile(boot_mious, [2.5, 97.5])
        row = {
            "experiment": name,
            "official_miou": round(official_miou, 4),
            "bootstrap_mean": round(float(boot_mious.mean()), 4),
            "bootstrap_std": round(float(boot_mious.std()), 4),
            "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4),
        }
        rows.append(row)
        print(f"  {name:32s} official={official_miou:.4f}  "
              f"bootstrap={boot_mious.mean():.4f}+-{boot_mious.std():.4f}  "
              f"95% CI=[{lo:.4f}, {hi:.4f}]")

        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "bootstrap_acdc_uncertainty.csv"
    archive_before_write(out_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
