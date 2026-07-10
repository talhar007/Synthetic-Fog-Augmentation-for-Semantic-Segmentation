"""
Fine-tune DeepLabV3+ on (fog-augmented) Cityscapes and evaluate on ACDC fog val.

Usage:
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/photometric_med.yaml
    python src/train.py --config configs/depthaware_b001.yaml --resume results/baseline/epoch_10.pth

All hyperparameters that differ between conditions are controlled via the YAML config.
The ONLY variable that should differ between conditions is `augmentation`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# --- project imports ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.cityscapes import CityscapesDataset
from models import build_model, forward_train, forward_eval, _detect_backend
from utils import seed_everything, SegMetrics, get_logger

# ImageNet normalisation constants (BGR order in Cityscapes, RGB in our loader)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fog cache (pre-rendered via scripts/generate_fog.py)
# ──────────────────────────────────────────────────────────────────────────────

# beta -> suffix used in scripts/generate_fog.py's VARIANTS keys
_BETA_SUFFIX = {0.005: "b0005", 0.010: "b001", 0.020: "b002"}


def _nn_depth_prefix(aug_cfg: dict) -> str:
    """'nn_depth_base' for the DepthAnythingV2-Base model, else the original
    'nn_depth' (Small model) cache prefix — keeps the two caches separate."""
    return "nn_depth_base" if "Base" in aug_cfg.get("model_id", "") else "nn_depth"


def fog_cache_variant_names(aug_cfg: dict) -> list[str] | None:
    """
    Map an augmentation config to its scripts/generate_fog.py cache directory
    name(s). Normally returns a single-element list; returns all three beta
    variants when beta == 'mixed' (intensity-diversity mode — the dataset then
    picks one at random per sample from the already-rendered caches).

    For 'combined', returns the matching depth-aware variant — that's the
    expensive part worth caching. The photometric layer stays cheap and is
    re-applied fresh every epoch on top of the cached image (see
    build_fog_transform), so it keeps its per-epoch randomness.
    """
    aug_type = aug_cfg.get("type", "none")

    def _beta_variants(prefix: str) -> list[str]:
        beta = aug_cfg["beta"]
        if beta == "mixed":
            return [f"{prefix}_{s}" for s in _BETA_SUFFIX.values()]
        return [f"{prefix}_{_BETA_SUFFIX[beta]}"]

    if aug_type == "depth_aware":
        return _beta_variants("depthaware")
    if aug_type == "nn_depth":
        return _beta_variants(_nn_depth_prefix(aug_cfg))
    if aug_type == "combined":
        technique = aug_cfg.get("depth_technique", "stereo")
        prefix = "depthaware" if technique == "stereo" else _nn_depth_prefix(aug_cfg)
        return [f"{prefix}_{_BETA_SUFFIX[aug_cfg['beta']]}"]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation factory
# ──────────────────────────────────────────────────────────────────────────────

def build_fog_transform(aug_cfg: dict, use_cache: bool = False):
    """
    Return a fog_transform callable or None.

    When use_cache=True, the (expensive) base fog image is supplied by the
    on-disk cache instead — see fog_cache_variant_name. Only transforms that
    still need to run on top of that cached base (e.g. combined's photometric
    layer) are returned here; fully-cached types return None.
    """
    aug_type = aug_cfg.get("type", "none")
    if aug_type == "none":
        return None

    if aug_type == "photometric":
        from fog.photometric import PhotometricFog
        return PhotometricFog(intensity=aug_cfg["intensity"])

    if aug_type == "depth_aware":
        if use_cache:
            return None  # fully supplied by cache
        if aug_cfg["beta"] == "mixed":
            raise ValueError("beta: mixed requires use_fog_cache: true (reuses the 3 rendered caches)")
        from fog.koschmieder import KoschmiederFog
        return KoschmiederFog(beta=aug_cfg["beta"])

    if aug_type == "combined":
        from fog.photometric import PhotometricFog

        photo = PhotometricFog(intensity=aug_cfg["photometric_intensity"])
        if use_cache:
            # depth fog comes from the depthaware_* cache; only layer photometric
            return photo

        from fog.koschmieder import KoschmiederFog
        depth_fog = KoschmiederFog(beta=aug_cfg["beta"])

        def combined_transform(result: dict) -> dict:
            result = depth_fog(result)   # apply depth fog first
            result = photo(result)       # then layer photometric haze
            return result

        return combined_transform

    if aug_type == "nn_depth":
        if use_cache:
            return None  # fully supplied by cache
        if aug_cfg["beta"] == "mixed":
            raise ValueError("beta: mixed requires use_fog_cache: true (reuses the 3 rendered caches)")
        from fog.nn_depth_fog import NNDepthFog
        return NNDepthFog(
            beta=aug_cfg["beta"],
            model_id=aug_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"),
            device=aug_cfg.get("device", None),
        )

    raise ValueError(f"Unknown augmentation type: {aug_type!r}")


def build_geo_transform(cfg: dict, train: bool) -> A.Compose:
    """Build Albumentations pipeline for geometric + normalisation transforms."""
    W, H = cfg["data"]["crop_size"]
    lo, hi = cfg["data"]["scale_range"]

    if train:
        return A.Compose([
            A.RandomScale(scale_limit=(lo - 1, hi - 1), p=1.0),
            A.PadIfNeeded(min_height=H, min_width=W, border_mode=0, value=0, mask_value=255),
            A.RandomCrop(height=H, width=W),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        # Evaluation: no random augmentation
        return A.Compose([
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────────────

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Poly decay: lr = base_lr * (1 - iter/max_iter)^power."""

    def __init__(self, optimizer, max_iter: int, power: float = 1.0, last_epoch: int = -1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(self._step_count / self.max_iter, 1.0)
        factor = (1 - progress) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _archive_prior_run(out_dir: Path, resume: str | None) -> None:
    """
    Preserve run history: if out_dir already holds a previous run (checkpoints
    and/or a train.log) and this is a fresh, non-resumed invocation, move the
    existing contents into out_dir/history/<timestamp>/ before training
    overwrites anything. Resumed runs (--resume passed) are left untouched
    since they're meant to continue building on the current run in place.
    """
    if resume or not out_dir.exists():
        return

    prior_files = [f for f in out_dir.iterdir() if f.name != "history"]
    has_prior_run = any(f.name in ("best.pth", "latest.pth", "train.log") for f in prior_files)
    if not has_prior_run:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = out_dir / "history" / stamp
    archive_dir.mkdir(parents=True)
    for f in prior_files:
        shutil.move(str(f), str(archive_dir / f.name))
    print(f"Archived previous run → {archive_dir}")

def train(cfg: dict) -> None:
    seed_everything(cfg["experiment"]["seed"])

    out_dir = Path(cfg["training"]["output_dir"])
    _archive_prior_run(out_dir, resume=cfg["training"].get("resume"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(cfg["experiment"]["name"], log_file=str(out_dir / "train.log"))

    # Save config + git commit for reproducibility
    (out_dir / "config.yaml").write_text(yaml.dump(cfg))
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        (out_dir / "git_commit.txt").write_text(git_hash.decode().strip())
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    # cfg["model"]["checkpoint"] is the mmseg-format Cityscapes-pretrained
    # init weights — only meaningful (and only guaranteed to exist) for the
    # mmseg backend. The smp backend always starts from ImageNet weights
    # baked into the encoder; pass nothing here, or torch.load would crash
    # on the (often-absent) mmseg checkpoint or load incompatible keys.
    backend = _detect_backend()
    pretrained_init = cfg["model"].get("checkpoint") if backend == "mmseg" else None
    model = build_model(
        cfg["model"]["mmseg_config"],
        checkpoint=pretrained_init,
        backend=backend,
        encoder_name=cfg["model"].get("encoder_name", "resnet101"),
        architecture=cfg["model"].get("architecture", "deeplabv3plus"),
    )
    model = model.to(device)

    # ── Datasets ───────────────────────────────────────────────────────────
    use_cache = cfg["data"].get("use_fog_cache", False)
    fog_cache_dir = None
    fog_cache_dirs = None
    if use_cache:
        variants = fog_cache_variant_names(cfg["augmentation"])
        if variants is None:
            raise ValueError(
                f"use_fog_cache=true but augmentation.type="
                f"{cfg['augmentation'].get('type')!r} has no cache variant"
            )
        fog_cache_root = Path(cfg["data"].get("fog_cache_root", "data/fog_cache"))
        if len(variants) == 1:
            fog_cache_dir = fog_cache_root / variants[0]
            log.info(f"Using fog cache: {fog_cache_dir}")
        else:
            fog_cache_dirs = [fog_cache_root / v for v in variants]
            log.info(f"Using mixed fog cache (random per sample): {fog_cache_dirs}")

    fog_prob = cfg["augmentation"].get("fog_prob", 1.0)
    log.info(f"fog_prob: {fog_prob}")

    need_depth = (not use_cache) and cfg["augmentation"].get("type") in ("depth_aware", "combined")
    fog_transform = build_fog_transform(cfg["augmentation"], use_cache=use_cache)
    train_geo = build_geo_transform(cfg, train=True)
    val_geo   = build_geo_transform(cfg, train=False)

    train_ds = CityscapesDataset(
        root=cfg["data"]["cityscapes_root"],
        split="train",
        return_depth=need_depth,
        fog_transform=fog_transform,
        geo_transform=train_geo,
        fog_cache_dir=fog_cache_dir,
        fog_cache_dirs=fog_cache_dirs,
        fog_prob=fog_prob,
    )
    val_ds = CityscapesDataset(
        root=cfg["data"]["cityscapes_root"],
        split="val",
        return_depth=False,
        fog_transform=None,   # evaluate on clear Cityscapes val
        geo_transform=val_geo,
    )
    log.info(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # ── Optimiser + LR schedule ────────────────────────────────────────────
    total_iters = len(train_loader) * cfg["training"]["epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = PolynomialLR(optimizer, max_iter=total_iters, power=cfg["training"]["poly_power"])

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_miou   = 0.0
    if cfg["training"].get("resume"):
        ckpt = torch.load(cfg["training"]["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou   = ckpt.get("best_miou", 0.0)
        log.info(f"Resumed from epoch {start_epoch}  best_mIoU={best_miou:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────
    metrics_rows = []

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            imgs   = batch["image"].to(device, non_blocking=True)   # [B,3,H,W]
            labels = batch["label"].long().to(device, non_blocking=True)  # [B,H,W]

            optimizer.zero_grad()
            loss, _ = forward_train(model, imgs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (i + 1) % cfg["training"]["log_interval"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"Epoch {epoch+1}/{cfg['training']['epochs']}  "
                    f"iter {i+1}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}  lr={lr:.2e}"
                )

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        log.info(f"Epoch {epoch+1} complete — avg_loss={avg_loss:.4f}  time={elapsed:.0f}s")

        # ── Validation ────────────────────────────────────────────────────
        val_miou = 0.0
        if (epoch + 1) % cfg["training"]["val_interval"] == 0:
            val_miou = evaluate_loader(model, val_loader, device, log)
            row = {"epoch": epoch + 1, "train_loss": avg_loss, "val_miou": val_miou}
            metrics_rows.append(row)
            _save_metrics_csv(metrics_rows, out_dir / "metrics.csv")

            if val_miou > best_miou:
                best_miou = val_miou
                _save_checkpoint(
                    model, optimizer, scheduler, epoch, best_miou,
                    out_dir / "best.pth"
                )
                log.info(f"  ★ New best val mIoU: {best_miou:.4f}")

        # Always save latest
        _save_checkpoint(model, optimizer, scheduler, epoch, best_miou, out_dir / "latest.pth")

    log.info(f"Training complete. Best val mIoU = {best_miou:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_loader(model, loader, device, log=None) -> float:
    model.eval()
    metrics = SegMetrics()
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            preds  = forward_eval(model, imgs)
            metrics.update(preds, labels)
    summary = metrics.summary()
    if log:
        log.info(f"  val mIoU = {summary['mIoU']:.4f}")
    model.train()
    return summary["mIoU"]


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, scheduler, epoch, best_miou, path: Path) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_miou": best_miou,
    }, path)


def _save_metrics_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML experiment config")
    parser.add_argument("--resume", default=None, help="Override resume checkpoint path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["training"]["resume"] = args.resume

    train(cfg)
