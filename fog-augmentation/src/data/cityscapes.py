"""
Cityscapes dataset loader.

Returns image + 19-class trainId label for all splits, plus optional
disparity + camera calibration (needed for depth-aware fog in Phase 4).

Expected on-disk layout after extraction:
  <root>/
    leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
    gtFine/{train,val}/<city>/*_gtFine_labelTrainIds.png
    disparity/{train,val}/<city>/*_disparity.png          (optional)
    camera/{train,val}/<city>/*_camera.json               (optional)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

# Mapping from raw Cityscapes labelId → 19-class trainId.
# Source: cityscapesscripts/helpers/labels.py
_ID_TO_TRAINID: dict[int, int] = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4,
    14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
    21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18,
    -1: 255,
}

_LABELID_TO_TRAINID_LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in _ID_TO_TRAINID.items():
    if 0 <= _k < 256:
        _LABELID_TO_TRAINID_LUT[_k] = _v

# 19 Cityscapes trainId classes, in trainId order (0-18) — same order as the
# PALETTE below, so CLASSES[i] is always the name of PALETTE[i]'s class.
CLASSES: list[str] = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# Official Cityscapes trainId color palette (source: cityscapesscripts/helpers/labels.py).
PALETTE = np.array(
    [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32],
    ],
    dtype=np.uint8,
)


def _convert_label_to_trainid(label_raw: np.ndarray) -> np.ndarray:
    """Map labelId PNG (raw 0-33) to 19-class trainId (0-18 + 255)."""
    return _LABELID_TO_TRAINID_LUT[label_raw.astype(np.uint8)]


class CityscapesDataset(Dataset):
    """
    Args:
        root:         path to the cityscapes root (contains leftImg8bit/, gtFine/, …)
        split:        'train' | 'val' | 'test'
        return_depth: if True, also loads disparity + camera and returns raw_disp + camera_params
        fog_transform: callable(image_rgb, label, **extras) → (image_rgb, label)
                       Applied BEFORE geometric normalisation.  Receives numpy arrays.
                       Ignored when fog_cache_dir is set.
        geo_transform: albumentations Compose applied to both image and label
                       (handles resize/crop/flip/normalize/ToTensor).
        fog_cache_dir: path to pre-rendered fog images (see scripts/generate_fog.py),
                       structured as <fog_cache_dir>/<split>/<city>/<stem>.jpg.
                       When set, the cached image is loaded instead of calling
                       fog_transform — use this for augmentations too expensive
                       to recompute every epoch (e.g. NN depth estimation).
        fog_cache_dirs: list of cache dirs to choose from at random, one per
                       sample (mutually exclusive with fog_cache_dir). Used for
                       "mixed intensity" augmentation: reuses the existing
                       single-intensity caches (e.g. the three depthaware_b*
                       dirs) without re-rendering anything.
        fog_prob:      probability that fog (cache and/or fog_transform) is
                       applied to a given training sample. The remaining
                       (1 - fog_prob) fraction of samples keep the clear image.
                       Default 1.0 preserves the old always-fogged behaviour.
    """

    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        return_depth: bool = False,
        fog_transform: Optional[Callable] = None,
        geo_transform: Optional[Callable] = None,
        fog_cache_dir: Optional[str | Path] = None,
        fog_cache_dirs: Optional[list[str | Path]] = None,
        fog_prob: float = 1.0,
    ):
        assert fog_cache_dir is None or fog_cache_dirs is None, \
            "fog_cache_dir and fog_cache_dirs are mutually exclusive"
        self.root = Path(root)
        self.split = split
        self.return_depth = return_depth
        self.fog_transform = fog_transform
        self.geo_transform = geo_transform
        self.fog_cache_dir = Path(fog_cache_dir) if fog_cache_dir else None
        self.fog_cache_dirs = [Path(p) for p in fog_cache_dirs] if fog_cache_dirs else None
        self.fog_prob = fog_prob
        self.samples = self._find_samples()

    # ------------------------------------------------------------------
    def _find_samples(self) -> list[dict]:
        img_dir = self.root / "leftImg8bit" / self.split
        label_dir = self.root / "gtFine" / self.split
        disp_dir = self.root / "disparity" / self.split
        cam_dir = self.root / "camera" / self.split

        samples = []
        for img_path in sorted(img_dir.rglob("*_leftImg8bit.png")):
            city = img_path.parent.name
            stem = img_path.stem.replace("_leftImg8bit", "")

            # Prefer pre-generated trainId file; fall back to raw labelId file
            train_id_path = label_dir / city / f"{stem}_gtFine_labelTrainIds.png"
            raw_id_path   = label_dir / city / f"{stem}_gtFine_labelIds.png"

            if train_id_path.exists():
                entry: dict = {"img": img_path, "label": train_id_path, "label_is_raw": False}
            elif raw_id_path.exists():
                entry = {"img": img_path, "label": raw_id_path, "label_is_raw": True}
            else:
                continue  # no annotation (e.g. test images)

            if self.return_depth:
                entry["disp"] = disp_dir / city / f"{stem}_disparity.png"
                entry["camera"] = cam_dir / city / f"{stem}_camera.json"

            samples.append(entry)
        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img = cv2.imread(str(s["img"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # H×W×3 uint8

        label = np.array(cv2.imread(str(s["label"]), cv2.IMREAD_GRAYSCALE))  # H×W uint8
        if s.get("label_is_raw"):
            label = _convert_label_to_trainid(label)

        result: dict = {"image": img, "label": label, "img_path": str(s["img"])}

        if self.return_depth and "disp" in s:
            raw_disp = np.array(
                cv2.imread(str(s["disp"]), cv2.IMREAD_UNCHANGED)
            ).astype(np.float32)
            with open(s["camera"]) as f:
                camera = json.load(f)
            result["raw_disp"] = raw_disp
            result["camera"] = camera

        # --- fog augmentation (applied to full-res image before crop) ---
        # Cache supplies the (expensive) base fog image; fog_transform, if also
        # set, layers a further (cheap, per-epoch-random) effect on top — used
        # by the "combined" condition to keep photometric variation alive while
        # reusing a cached depth-aware base.
        #
        # fog_prob gates the whole block: with probability (1 - fog_prob) the
        # sample is left as the clear image already loaded above (a genuine
        # real/generated mix, rather than fog applied to every sample).
        has_fog_source = (
            self.fog_cache_dir is not None
            or self.fog_cache_dirs is not None
            or self.fog_transform is not None
        )
        if has_fog_source and (self.fog_prob >= 1.0 or random.random() < self.fog_prob):
            cache_dir = self.fog_cache_dir
            if self.fog_cache_dirs is not None:
                cache_dir = random.choice(self.fog_cache_dirs)  # "mixed intensity"
            if cache_dir is not None:
                city = s["img"].parent.name
                stem = s["img"].stem.replace("_leftImg8bit", "")
                cached_path = cache_dir / self.split / city / f"{stem}.jpg"
                cached = cv2.imread(str(cached_path))
                if cached is None:
                    raise FileNotFoundError(f"Fog cache miss: {cached_path}")
                result["image"] = cv2.cvtColor(cached, cv2.COLOR_BGR2RGB)
            if self.fog_transform is not None:
                result = self.fog_transform(result)

        # --- geometric transforms (resize / crop / flip / normalize / ToTensor) ---
        if self.geo_transform is not None:
            transformed = self.geo_transform(
                image=result["image"], mask=result["label"]
            )
            result["image"] = transformed["image"]
            result["label"] = transformed["mask"]

        return result

    # ------------------------------------------------------------------
    @staticmethod
    def decode_target(label: np.ndarray) -> np.ndarray:
        """Map trainId 0-18 (+ 255) to a colour image for visualisation."""
        colour = np.zeros((*label.shape, 3), dtype=np.uint8)
        for i, c in enumerate(PALETTE):
            colour[label == i] = c
        colour[label == 255] = [0, 0, 0]
        return colour
