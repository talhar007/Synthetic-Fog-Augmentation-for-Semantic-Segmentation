"""
Evaluate a trained checkpoint on ACDC fog validation split.

Computes mIoU + per-class IoU and writes a JSON results file.

Usage:
    python src/evaluate.py \
        --config  configs/baseline.yaml \
        --checkpoint results/baseline/best.pth \
        --acdc-root /path/to/acdc \
        --output  results/baseline/acdc_eval.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.acdc import ACDCDataset
from models import build_model, forward_eval
from utils import SegMetrics, get_logger

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_eval_transform() -> A.Compose:
    return A.Compose([
        # ACDC images (1080x1920) aren't divisible by 32, which the
        # ResNet-101 5-stage encoder requires. Pad with ignore_index on the
        # mask so the padded border is automatically excluded from mIoU.
        A.PadIfNeeded(
            min_height=None, min_width=None,
            pad_height_divisor=32, pad_width_divisor=32,
            border_mode=0, value=0, mask_value=255,
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def evaluate(config_path: str, checkpoint: str, acdc_root: str | None, output: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if acdc_root is None:
        acdc_root = cfg["data"]["acdc_root"]

    log = get_logger("evaluate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        cfg["model"]["mmseg_config"],
        checkpoint=checkpoint,
        encoder_name=cfg["model"].get("encoder_name", "resnet101"),
        architecture=cfg["model"].get("architecture", "deeplabv3plus"),
    )
    model = model.to(device)
    model.eval()

    # ── ACDC dataset ───────────────────────────────────────────────────────
    split = cfg.get("evaluate", {}).get("split", "val")
    dataset = ACDCDataset(
        root=acdc_root,
        split=split,
        geo_transform=build_eval_transform(),
    )
    log.info(f"ACDC fog {split}: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # ── Evaluation loop ────────────────────────────────────────────────────
    metrics = SegMetrics()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            preds  = forward_eval(model, imgs)
            metrics.update(preds, labels)

            if (i + 1) % 100 == 0:
                log.info(f"  {i+1}/{len(loader)} processed")

    summary = metrics.summary()
    summary["experiment"] = cfg["experiment"]["name"]
    summary["checkpoint"] = str(checkpoint)
    summary["split"] = split

    log.info(f"mIoU = {summary['mIoU']:.4f}")
    for cls, iou in summary["per_class"].items():
        log.info(f"  {cls:<20s} {iou:.4f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved → {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--acdc-root",  default=None,
                        help="Defaults to data.acdc_root in --config if omitted")
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()

    evaluate(args.config, args.checkpoint, args.acdc_root, args.output)
