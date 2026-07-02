"""
Disparity → metric depth for Cityscapes.

Cityscapes stores stereo disparity as uint16 PNG:
    real_disparity_px = (raw_value - 1) / 256.0   (0 = invalid pixel)
    depth_m = baseline_m * focal_length_px / real_disparity_px

Calibration data come from the JSON files in the `camera/` package:
    camera['extrinsic']['baseline']   (metres, typically 0.2093)
    camera['intrinsic']['fx']         (pixels, typically 2262.5)

Depth completion is needed because the disparity maps have holes (invalid regions).
Two strategies are implemented:
  1. simple_complete  — fast OpenCV inpaint (good enough for Phase 4 start)
  2. ransac_complete  — superpixel-RANSAC plane fitting (closer to Sakaridis et al.)

Usage:
    depth = disparity_to_depth(raw_disp, camera_json)
    depth = complete_depth(depth, guide_rgb, method='simple')
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Disparity → depth
# ──────────────────────────────────────────────────────────────────────────────

def disparity_to_depth(
    raw_disp: np.ndarray,
    camera: dict,
    min_depth: float = 1.0,
    max_depth: float = 500.0,
) -> np.ndarray:
    """
    Convert a Cityscapes uint16 disparity map to metric depth (metres).

    Args:
        raw_disp:  H×W uint16 array as loaded from the disparity PNG.
        camera:    dict from *_camera.json  (must have extrinsic.baseline + intrinsic.fx)
        min_depth: clamp range floor in metres
        max_depth: clamp range ceiling in metres

    Returns:
        H×W float32 depth map; 0.0 marks invalid (hole) pixels.
    """
    baseline = float(camera["extrinsic"]["baseline"])  # metres
    fx       = float(camera["intrinsic"]["fx"])         # pixels

    raw = raw_disp.astype(np.float32)
    valid = raw > 0
    # Cityscapes encoding: raw = disp_px * 256 + 1
    disp = np.where(valid, (raw - 1.0) / 256.0, 0.0)
    depth = np.where(
        (valid) & (disp > 1e-6),
        np.clip(baseline * fx / disp.clip(min=1e-6), min_depth, max_depth),
        0.0,
    )
    return depth.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Depth completion
# ──────────────────────────────────────────────────────────────────────────────

def complete_depth(
    depth: np.ndarray,
    guide_rgb: np.ndarray,
    method: Literal["simple", "ransac"] = "simple",
) -> np.ndarray:
    """
    Fill holes (0-valued pixels) in the depth map.

    Args:
        depth:     H×W float32 from disparity_to_depth (0 = invalid).
        guide_rgb: H×W×3 uint8 RGB image used as spatial guide.
        method:    'simple' (fast, OpenCV inpaint) or 'ransac' (Sakaridis-style).

    Returns:
        H×W float32 depth map with holes filled.
    """
    if method == "simple":
        return _simple_complete(depth, guide_rgb)
    elif method == "ransac":
        return _ransac_complete(depth, guide_rgb)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _simple_complete(depth: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
    """
    Fast depth completion using OpenCV inpaint, then bilateral smoothing.
    The guide image is used for edge-preserving bilateral filter.
    """
    hole_mask = (depth == 0).astype(np.uint8)
    # Normalise depth to [0,1] for inpainting, then scale back
    d_max = depth.max()
    if d_max < 1e-6:
        return depth

    d_norm = (depth / d_max * 255).astype(np.uint8)
    inpainted = cv2.inpaint(d_norm, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    filled = inpainted.astype(np.float32) / 255.0 * d_max

    # Edge-preserving smoothing using bilateral filter on transmittance (Phase 4 does this
    # after computing beta, but smoothing depth here gives similar results)
    smoothed = cv2.bilateralFilter(filled, d=9, sigmaColor=0.5, sigmaSpace=9)
    return smoothed.astype(np.float32)


def _ransac_complete(depth: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
    """
    Superpixel + least-squares plane fitting for depth completion.
    Closer to the Sakaridis et al. pipeline (simplified RANSAC → closed-form lstsq).

    For each SLIC superpixel:
      1. Collect valid (u, v, d) triplets.
      2. Fit plane d = a*u + b*v + c via least squares.
      3. Fill invalid pixels in the superpixel with the fitted plane.
    """
    try:
        from skimage.segmentation import slic
    except ImportError:
        print("scikit-image not available, falling back to simple depth completion.")
        return _simple_complete(depth, guide_rgb)

    segments = slic(guide_rgb, n_segments=300, compactness=10, sigma=1, start_label=0)
    completed = depth.copy()
    H, W = depth.shape

    for seg_id in np.unique(segments):
        seg_mask  = segments == seg_id
        valid_mask = seg_mask & (depth > 0)
        n_valid = valid_mask.sum()
        if n_valid < 6:
            continue

        ys, xs = np.where(valid_mask)
        ds = depth[valid_mask]

        # Fit plane: d = a*x + b*y + c
        A = np.column_stack([xs, ys, np.ones(n_valid)])
        try:
            params, _, _, _ = np.linalg.lstsq(A, ds, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Fill holes in this superpixel
        hole_mask = seg_mask & (depth == 0)
        if hole_mask.sum() > 0:
            hy, hx = np.where(hole_mask)
            pred = params[0] * hx + params[1] * hy + params[2]
            completed[hole_mask] = np.clip(pred, 0.5, 500.0)

    # Smooth the transmittance via guided filter if possible, else bilateral
    try:
        guide_f = guide_rgb.astype(np.float32) / 255.0
        completed_f = completed.astype(np.float32)
        smoothed = cv2.ximgproc.guidedFilter(
            guide=guide_f, src=completed_f, radius=16, eps=1e-4
        )
    except AttributeError:
        smoothed = cv2.bilateralFilter(completed.astype(np.float32), 9, 0.5, 16)

    return np.clip(smoothed, 0.0, 500.0).astype(np.float32)
