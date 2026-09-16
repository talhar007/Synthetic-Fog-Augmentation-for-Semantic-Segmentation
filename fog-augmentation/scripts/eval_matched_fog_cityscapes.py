"""
Matched-fog Cityscapes-val evaluation for the Phase-1 checkpoints.

Addresses reviewer feedback: the paper's underfitting explanation for Phase
1's negative result was based on comparing heavily-fogged models' accuracy
on Cityscapes-val -- but that val split is *always* clear regardless of
training fog recipe (fog_transform=None, src/train.py). Poor clear-val
performance is also consistent with an alternative explanation the paper
didn't rule out: the model specialized toward the synthetic fog domain and
lost accuracy on clear imagery (overspecialization / distribution shift),
rather than failing to learn the task at all.

This re-evaluates each existing Phase-1 checkpoint (no retraining) on
Cityscapes-val rendered with the SAME fog config it was trained on, using
the existing fog cache for depth-based techniques and on-the-fly rendering
for photometric. If a checkpoint also performs poorly on its own matched-fog
validation images, that's strong evidence for genuine underfitting. If it
performs well there (much better than on clear images), that points to
overspecialization/distribution shift instead.

Usage:
    python scripts/eval_matched_fog_cityscapes.py
"""
import csv
import sys
from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.cityscapes import CityscapesDataset
from fog.photometric import PhotometricFog
from models import build_model, forward_eval
from utils import SegMetrics
from _archive import archive_before_write

REPO_ROOT = Path(__file__).resolve().parent.parent
CITYSCAPES_ROOT = "/home/taah3149/Documents/Group_Studies/dataset/cityscapes"
RESULTS_DIR = REPO_ROOT / "results"
FOG_CACHE_ROOT = REPO_ROOT / "data" / "fog_cache"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def val_geo() -> A.Compose:
    # Matches src/train.py's build_geo_transform(train=False) exactly, so the
    # "clear" number this script would reproduce is comparable to the
    # best_miou already stored in each checkpoint.
    return A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


# experiment -> (fog_transform, fog_cache_dir) -- mirrors each config's
# actual augmentation block (configs/<name>.yaml)
EXPERIMENTS = {
    "baseline":         dict(fog_transform=None,                       fog_cache_dir=None),
    "photometric_low":  dict(fog_transform=PhotometricFog("low"),       fog_cache_dir=None),
    "photometric_med":  dict(fog_transform=PhotometricFog("medium"),    fog_cache_dir=None),
    "photometric_high": dict(fog_transform=PhotometricFog("high"),      fog_cache_dir=None),
    "depthaware_b0005": dict(fog_transform=None, fog_cache_dir=FOG_CACHE_ROOT / "depthaware_b0005"),
    "depthaware_b001":  dict(fog_transform=None, fog_cache_dir=FOG_CACHE_ROOT / "depthaware_b001"),
    "depthaware_b002":  dict(fog_transform=None, fog_cache_dir=FOG_CACHE_ROOT / "depthaware_b002"),
    "nn_depth_b0005":   dict(fog_transform=None, fog_cache_dir=FOG_CACHE_ROOT / "nn_depth_b0005"),
    "nn_depth_b001":    dict(fog_transform=None, fog_cache_dir=FOG_CACHE_ROOT / "nn_depth_b001"),
    "nn_depth_b002":    dict(fog_transform=None, fog_cache_dir=FOG_CACHE_ROOT / "nn_depth_b002"),
    # combined = stereo beta=0.01 (depthaware_b001 cache) + photometric medium layered on top,
    # matching configs/combined.yaml exactly.
    "combined":         dict(fog_transform=PhotometricFog("medium"), fog_cache_dir=FOG_CACHE_ROOT / "depthaware_b001"),
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    rows = []

    for name, spec in EXPERIMENTS.items():
        ckpt_path = RESULTS_DIR / name / "best.pth"
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        clear_val_miou = float(ckpt["best_miou"])

        model = build_model(
            "configs/_base_/deeplabv3plus_r101.py",
            checkpoint=str(ckpt_path),
        ).to(device)
        model.eval()

        ds = CityscapesDataset(
            root=CITYSCAPES_ROOT, split="val", return_depth=False,
            fog_transform=spec["fog_transform"],
            fog_cache_dir=spec["fog_cache_dir"],
            fog_prob=1.0,  # every image fogged -- matches Phase 1's 100%-fog training regime
            geo_transform=val_geo(),
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        metrics = SegMetrics()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].long().to(device, non_blocking=True)
                preds = forward_eval(model, imgs)
                metrics.update(preds, labels)
        fogged_val_miou = metrics.miou()

        delta = fogged_val_miou - clear_val_miou
        rows.append({
            "experiment": name,
            "clear_cityscapes_val_miou": round(clear_val_miou, 4),
            "matched_fog_cityscapes_val_miou": round(fogged_val_miou, 4),
            "delta_fog_minus_clear": round(delta, 4),
        })
        print(f"{name:20s} clear={clear_val_miou:.4f}  matched-fog={fogged_val_miou:.4f}  "
              f"delta={delta:+.4f}  (n={len(ds)})")

        del model
        torch.cuda.empty_cache()

    out_path = RESULTS_DIR / "matched_fog_cityscapes_val.csv"
    archive_before_write(out_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
