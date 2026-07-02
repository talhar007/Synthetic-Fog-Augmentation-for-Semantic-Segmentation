"""
Dark channel prior for atmospheric light estimation.

Ref: He et al., "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009.

The atmospheric light L is estimated as the mean colour of the top-brightest
pixels in the dark channel, then clamped to avoid blowout.  This L is used
as the fog colour in the Koschmieder compositing formula.
"""
from __future__ import annotations

import cv2
import numpy as np


def dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an RGB image.

    The dark channel at pixel p is the minimum intensity over:
      - all colour channels
      - a local patch of radius (patch_size-1)//2 around p

    Args:
        img:        H×W×3 float32 in [0, 1]
        patch_size: local minimum filter size (odd number)

    Returns:
        H×W float32 dark channel in [0, 1]
    """
    min_c = np.min(img, axis=2)                              # H×W
    kernel = np.ones((patch_size, patch_size), np.uint8)
    dark = cv2.erode(min_c, kernel)                          # minimum filter
    return dark


def estimate_atmospheric_light(
    img: np.ndarray,
    dark: np.ndarray,
    top_fraction: float = 0.001,
) -> np.ndarray:
    """
    Estimate atmospheric light from the brightest pixels in the dark channel.

    Steps:
      1. Pick the top `top_fraction` pixels by dark-channel intensity.
      2. Among those pixels in the original image, take the per-channel mean.
      3. Clamp to [0.5, 1.0] so L isn't too dark (prevents division instability).

    Args:
        img:          H×W×3 float32 in [0, 1]
        dark:         H×W float32 dark channel (from dark_channel())
        top_fraction: fraction of brightest haze-opaque pixels to use

    Returns:
        (3,) float32 atmospheric light per colour channel
    """
    flat_dark = dark.flatten()
    n = max(int(flat_dark.size * top_fraction), 1)

    # Indices of the n brightest pixels in the dark channel
    idx = np.argpartition(flat_dark, -n)[-n:]

    flat_img = img.reshape(-1, 3)
    L = np.mean(flat_img[idx], axis=0)
    return np.clip(L, 0.5, 1.0).astype(np.float32)


def compute_atmospheric_light(img_rgb: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Convenience wrapper: uint8 RGB → normalised float → dark channel → L.

    Args:
        img_rgb:    H×W×3 uint8 RGB image
        patch_size: dark channel patch size

    Returns:
        (3,) float32 atmospheric light in [0, 1]
    """
    img_f = img_rgb.astype(np.float32) / 255.0
    dark  = dark_channel(img_f, patch_size=patch_size)
    return estimate_atmospheric_light(img_f, dark)
