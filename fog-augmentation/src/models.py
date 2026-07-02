"""
Model utilities: build DeepLabV3+ and load pretrained weights.

Supports two backends:
  'smp'   — segmentation_models_pytorch (ImageNet pretrained, no CUDA toolkit needed).
            Default on machines without the CUDA toolkit installed.
  'mmseg' — MMSegmentation (Cityscapes pretrained checkpoint, requires mmcv with CUDA ops).
            Preferred on HPC where nvcc is available. Set MMCV_BACKEND=mmseg or
            pass backend='mmseg' to build_model().

On the HPC, install mmcv via:
  pip install mmcv==2.2.0 \\
      -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.5/index.html
Then download the Cityscapes checkpoint:
  python src/data/download.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Auto-detect which backend is available
def _detect_backend() -> str:
    env = os.environ.get("MMCV_BACKEND", "auto")
    if env != "auto":
        return env
    try:
        import mmcv  # noqa
        import mmseg  # noqa
        return "mmseg"
    except ImportError:
        return "smp"


# ──────────────────────────────────────────────────────────────────────────────
# SMP backend (segmentation_models_pytorch)
# ──────────────────────────────────────────────────────────────────────────────

def _build_smp(num_classes: int = 19, checkpoint: Optional[str | Path] = None) -> nn.Module:
    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None,
    )
    if checkpoint:
        state = torch.load(str(checkpoint), map_location="cpu")
        # Accept raw state_dict or wrapped checkpoint
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# MMSeg backend
# ──────────────────────────────────────────────────────────────────────────────

def _build_mmseg(
    mmseg_config: str | Path,
    num_classes: int = 19,
    checkpoint: Optional[str | Path] = None,
) -> nn.Module:
    try:
        from mmseg.utils import register_all_modules
        register_all_modules(init_default_scope=False)
    except ImportError as e:
        raise ImportError(
            "mmsegmentation not available. Either install it on HPC or use backend='smp'."
        ) from e

    from mmengine.config import Config
    from mmseg.models import build_segmentor
    from mmengine.runner import load_checkpoint

    cfg = Config.fromfile(str(mmseg_config))
    model = build_segmentor(cfg.model)
    if checkpoint:
        print(f"Loading MMSeg checkpoint: {checkpoint}")
        load_checkpoint(model, str(checkpoint), map_location="cpu",
                        revise_keys=[(r"^module\.", "")])
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    mmseg_config: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
    num_classes: int = 19,
    backend: str = "auto",
) -> nn.Module:
    """
    Build DeepLabV3+ with pretrained weights.

    Args:
        mmseg_config: path to MMSeg Python config (used only with backend='mmseg')
        checkpoint:   path to .pth file or None
        num_classes:  number of output classes (19 for Cityscapes/ACDC)
        backend:      'auto' | 'smp' | 'mmseg'
    """
    if backend == "auto":
        backend = _detect_backend()

    print(f"Model backend: {backend}")

    if backend == "smp":
        return _build_smp(num_classes=num_classes, checkpoint=checkpoint)
    elif backend == "mmseg":
        assert mmseg_config, "mmseg_config must be provided for backend='mmseg'"
        return _build_mmseg(mmseg_config, num_classes=num_classes, checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'smp' or 'mmseg'.")


# ──────────────────────────────────────────────────────────────────────────────
# Forward pass helpers (backend-agnostic)
# ──────────────────────────────────────────────────────────────────────────────

def _get_logits(model: nn.Module, imgs: torch.Tensor, backend: str) -> torch.Tensor:
    """Return [B, C, H, W] logits at the input resolution."""
    if backend == "smp":
        logits = model(imgs)  # smp returns full-res logits by default
        # If spatial dims don't match input, upsample
        if logits.shape[-2:] != imgs.shape[-2:]:
            logits = F.interpolate(logits, size=imgs.shape[-2:],
                                   mode="bilinear", align_corners=False)
        return logits
    else:
        # mmseg EncoderDecoder
        feats = model.extract_feat(imgs)
        logits = model.decode_head.forward(feats)
        logits = F.interpolate(logits, size=imgs.shape[-2:],
                               mode="bilinear", align_corners=False)
        return logits


def forward_train(
    model: nn.Module,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    backend: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Training forward: returns (total_loss, logits).

    Args:
        imgs:    [B, C, H, W] float32, ImageNet-normalised
        labels:  [B, H, W] int64, trainId 0–18 + ignore 255
        backend: 'auto' | 'smp' | 'mmseg'
    """
    if backend == "auto":
        backend = _detect_backend()

    logits = _get_logits(model, imgs, backend)
    loss = F.cross_entropy(logits, labels, ignore_index=255)

    # Auxiliary head (mmseg only)
    if backend == "mmseg" and getattr(model, "with_auxiliary_head", False):
        feats = model.extract_feat(imgs)
        aux_heads = (
            model.auxiliary_head
            if isinstance(model.auxiliary_head, nn.ModuleList)
            else [model.auxiliary_head]
        )
        for aux in aux_heads:
            aux_logits = aux.forward(feats)
            aux_logits = F.interpolate(aux_logits, size=labels.shape[-2:],
                                       mode="bilinear", align_corners=False)
            loss = loss + 0.4 * F.cross_entropy(aux_logits, labels, ignore_index=255)

    return loss, logits


@torch.no_grad()
def forward_eval(
    model: nn.Module,
    imgs: torch.Tensor,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Inference forward — returns [B, H, W] int64 predicted class indices.
    """
    if backend == "auto":
        backend = _detect_backend()
    logits = _get_logits(model, imgs, backend)
    return logits.argmax(dim=1)
