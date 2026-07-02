"""
Photometric fog augmentation using Albumentations RandomFog.

Geometry-agnostic: a uniform white-ish haze blended over the entire image,
controlled by fog_coef (higher = denser fog). No depth information needed.

Three intensity levels match the three β levels used in the Koschmieder arm
so the two arms can be compared at nominally equivalent visibility.

  low    → light haze  (~600 m visibility analogue)
  medium → moderate fog (~300 m)
  high   → dense fog    (~150 m)
"""
from __future__ import annotations

import albumentations as A
import numpy as np

# fog_coef_lower/upper control the haze density (0=clear, 1=white).
# alpha_coef controls the size of the fog particles (smaller = finer grain).
INTENSITIES: dict[str, dict] = {
    "low": dict(fog_coef_lower=0.1, fog_coef_upper=0.25, alpha_coef=0.08),
    "medium": dict(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08),
    "high": dict(fog_coef_lower=0.55, fog_coef_upper=0.75, alpha_coef=0.08),
}


class PhotometricFog:
    """
    Callable fog transform that can be used as CityscapesDataset.fog_transform.

    Applies Albumentations RandomFog to `result['image']` and leaves
    `result['label']` (and any other keys) untouched.

    Args:
        intensity: 'low' | 'medium' | 'high'
        p:         probability of applying fog (default 1.0 for deterministic augmentation)
    """

    def __init__(self, intensity: str = "medium", p: float = 1.0):
        assert intensity in INTENSITIES, f"intensity must be one of {list(INTENSITIES)}"
        params = INTENSITIES[intensity]
        self.intensity = intensity
        self.transform = A.RandomFog(**params, p=p)

    def __call__(self, result: dict) -> dict:
        img = result["image"]  # H×W×3 uint8 RGB
        augmented = self.transform(image=img)
        result["image"] = augmented["image"]
        # label and all other keys are intentionally unchanged
        return result

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Convenience: apply to a bare numpy image (H×W×3 uint8)."""
        return self.transform(image=img)["image"]

    def __repr__(self) -> str:
        return f"PhotometricFog(intensity={self.intensity!r})"


def build_photometric_fog(intensity: str, p: float = 1.0) -> PhotometricFog:
    return PhotometricFog(intensity=intensity, p=p)
