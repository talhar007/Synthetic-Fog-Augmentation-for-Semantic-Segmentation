"""
Technique 3: Neural-network depth estimation + Koschmieder fog.

Pipeline:
    Clear RGB
        │
        ▼
  [DepthAnythingV2-Metric-Outdoor]   ← pretrained NN, no fine-tuning needed
        │
        ▼
   Metric depth (metres, per pixel)
        │
        ▼
  Koschmieder compositing  (same β-parameterised formula as Technique 2)
        │
        ▼
   Foggy RGB  (label unchanged)

Why this matters vs Technique 2 (stereo Koschmieder):
- Technique 2 uses stereo disparity → calibrated metric depth (high accuracy, needs stereo rig)
- Technique 3 uses monocular RGB → NN-predicted metric depth (lower accuracy, works on any camera)
- The Koschmieder physics formula is identical in both arms.
- Research question: how much does depth-map quality affect fog realism and segmentation mIoU?

Model choices (in order of accuracy / size):
  - Small:  depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf  (~24M params, ~fastest)
  - Base:   depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf   (~97M params)
  - Large:  depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf  (~335M params)
The model is downloaded automatically on first use and cached in ~/.cache/huggingface.

Reference:
  Yang et al., "Depth Anything V2", NeurIPS 2024.
  https://depth-anything-v2.github.io/
"""
from __future__ import annotations

import warnings
from typing import Optional

import cv2
import numpy as np
import torch

from .koschmieder import KoschmiederFog

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class NNDepthEstimator:
    """
    Thin wrapper around DepthAnythingV2 (via HuggingFace transformers).

    The model returns per-pixel depth in metres for outdoor scenes.
    Results are resized back to the input image resolution.

    Args:
        model_id:   HuggingFace model name (see module docstring for options)
        device:     'cuda' | 'cpu' | None (auto)
        max_depth:  clamp predicted depth at this value (metres)
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        max_depth: float = 500.0,
    ):
        from transformers import pipeline as hf_pipeline

        if device is None:
            # This estimator is lazily constructed inside CityscapesDataset
            # .__getitem__, which runs in forked DataLoader worker processes
            # when num_workers>0. CUDA cannot be re-initialised in a forked
            # subprocess that didn't initialise it pre-fork, so default to
            # CPU here regardless of GPU availability. Pass device='cuda'
            # explicitly only if you know num_workers=0.
            device = "cpu"

        self.device = device
        self.max_depth = max_depth
        self.model_id = model_id

        print(f"Loading DepthAnythingV2 ({model_id}) on {device} ...")
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if device == "cuda" else -1,
        )
        print("Depth model loaded.")

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Predict metric depth from an RGB image.

        Args:
            img_rgb: H×W×3 uint8 RGB

        Returns:
            H×W float32 depth map in metres; 0 where the model is uncertain.
        """
        from PIL import Image

        H, W = img_rgb.shape[:2]
        pil_img = Image.fromarray(img_rgb)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._pipe(pil_img)

        depth = np.array(result["depth"], dtype=np.float32)  # model output resolution

        # Resize to original image size (nearest works fine for depth)
        if depth.shape[:2] != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        depth = np.clip(depth, 0.1, self.max_depth)
        return depth


class NNDepthFog:
    """
    Technique 3 fog augmentation: monocular NN depth → Koschmieder compositing.

    Can be used as a CityscapesDataset.fog_transform callable.
    The depth estimator is initialised lazily on first call so that
    DataLoader workers don't each load a separate copy of the network.

    Args:
        beta:        fog attenuation coefficient {0.005, 0.01, 0.02}
        model_id:    DepthAnythingV2 HuggingFace model ID
        device:      'cuda' | 'cpu' | None (auto)
    """

    def __init__(
        self,
        beta: float = 0.01,
        model_id: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        self.beta = beta
        self.model_id = model_id
        self.device = device
        self._estimator: Optional[NNDepthEstimator] = None
        self._fogger = KoschmiederFog(beta=beta)

    def _get_estimator(self) -> NNDepthEstimator:
        if self._estimator is None:
            self._estimator = NNDepthEstimator(
                model_id=self.model_id,
                device=self.device,
            )
        return self._estimator

    def __call__(self, result: dict) -> dict:
        """Apply NN depth fog to result['image'] (H×W×3 uint8 RGB)."""
        img = result["image"]
        depth = self._get_estimator().predict(img)
        result["image"] = self._fogger.apply(img, depth)
        result["nn_depth"] = depth  # store for optional inspection
        # label and all other keys are intentionally unchanged
        return result

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Convenience: apply to a bare numpy image."""
        depth = self._get_estimator().predict(img)
        return self._fogger.apply(img, depth)

    def __repr__(self) -> str:
        return f"NNDepthFog(β={self.beta}, model={self.model_id.split('/')[-1]})"
