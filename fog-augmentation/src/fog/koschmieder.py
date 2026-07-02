"""
Physics-based depth-aware fog via the Koschmieder model.

Fog model (per pixel x):
    I(x) = R(x) · t(x) + L · (1 - t(x))
    t(x) = exp(−β · d(x))

Where:
    R(x) = clear-weather radiance (input pixel value)
    d(x) = per-pixel scene depth in metres (from disparity)
    β    = fog attenuation coefficient (denser fog → larger β)
    L    = atmospheric light (estimated via dark channel prior)
    t(x) = transmittance (how much of the scene survives through the fog)

β values used in the project (matching Foggy Cityscapes):
    β = 0.005  →  visibility ≈ 600 m   (light fog)
    β = 0.010  →  visibility ≈ 300 m   (moderate fog)
    β = 0.020  →  visibility ≈ 150 m   (dense fog)

The Meteorological Optical Range (MOR) relates to β as:
    MOR ≈ ln(20) / β ≈ 3.0 / β   (Koschmieder's law)

Reference implementations (MATLAB):
    https://github.com/sakaridis/fog_simulation-SFSU_synthetic
    https://github.com/sakaridis/fog_simulation_DBF

This is a faithful Python reimplementation that avoids a MATLAB licence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .atmospheric import compute_atmospheric_light
from .depth import complete_depth, disparity_to_depth


# β → approximate visibility in metres
BETA_VISIBILITY = {
    0.005: 600,
    0.010: 300,
    0.020: 150,
}


class KoschmiederFog:
    """
    Depth-aware Koschmieder fog transform.

    Can be used as a CityscapesDataset.fog_transform callable.

    The result dict must contain 'image' (H×W×3 uint8 RGB).
    If 'raw_disp' and 'camera' are present (return_depth=True), they are used
    to compute metric depth.  Otherwise a flat depth (fallback) is used.

    Args:
        beta:              attenuation coefficient {0.005, 0.01, 0.02}
        depth_method:      'simple' | 'ransac' for hole filling
        L:                 atmospheric light (3,) float32 [0,1], or None → DCP estimate
        transmittance_min: clamp transmittance floor to avoid zero-alpha pixels
    """

    def __init__(
        self,
        beta: float = 0.01,
        depth_method: str = "simple",
        L: Optional[np.ndarray] = None,
        transmittance_min: float = 0.0,
    ):
        self.beta = beta
        self.depth_method = depth_method
        self.L = L
        self.transmittance_min = transmittance_min

    def __call__(self, result: dict) -> dict:
        img = result["image"]  # H×W×3 uint8 RGB

        if "raw_disp" in result and "camera" in result:
            depth = disparity_to_depth(result["raw_disp"], result["camera"])
            depth = complete_depth(depth, img, method=self.depth_method)
        else:
            # Fallback: flat depth (photometric-like behaviour)
            depth = np.full(img.shape[:2], 50.0, dtype=np.float32)

        result["image"] = self.apply(img, depth)
        # Label and other keys unchanged
        return result

    def apply(self, img: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Apply Koschmieder fog to a single image + depth map.

        Args:
            img:   H×W×3 uint8 RGB
            depth: H×W float32 metric depth in metres (0 = invalid)

        Returns:
            H×W×3 uint8 RGB foggy image
        """
        img_f = img.astype(np.float32) / 255.0

        # Estimate atmospheric light if not provided
        L = self.L if self.L is not None else compute_atmospheric_light(img)
        L = np.asarray(L, dtype=np.float32).reshape(1, 1, 3)

        # Handle invalid (hole) pixels in depth: assign a far distance
        d = depth.copy()
        d[d <= 0] = depth[depth > 0].max() if (depth > 0).any() else 100.0

        # Transmittance map
        t = np.exp(-self.beta * d).astype(np.float32)
        if self.transmittance_min > 0:
            t = np.maximum(t, self.transmittance_min)
        t = t[:, :, np.newaxis]  # H×W×1 for broadcasting

        # Koschmieder compositing
        foggy = img_f * t + L * (1.0 - t)
        foggy = np.clip(foggy * 255.0, 0, 255).astype(np.uint8)
        return foggy

    def __repr__(self) -> str:
        vis = BETA_VISIBILITY.get(self.beta, f"≈{3.0/self.beta:.0f}")
        return f"KoschmiederFog(β={self.beta}, visibility≈{vis}m)"


# ──────────────────────────────────────────────────────────────────────────────
# Convenience builder
# ──────────────────────────────────────────────────────────────────────────────

def build_koschmieder_fog(beta: float, depth_method: str = "simple") -> KoschmiederFog:
    return KoschmiederFog(beta=beta, depth_method=depth_method)
