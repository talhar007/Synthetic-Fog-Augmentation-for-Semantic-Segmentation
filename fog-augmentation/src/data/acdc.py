"""
ACDC fog-split loader (evaluation only — never train on this).

Expected on-disk layout:
  <root>/
    rgb_anon/fog/{train,val,test}/<sequence>/*_rgb_anon.png
    gt/fog/{train,val}/<sequence>/*_gt_labelTrainIds.png

ACDC uses the same 19 Cityscapes trainId classes and ignore_index=255.
No class remapping is needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    """
    Args:
        root:          path to the ACDC root (contains rgb_anon/ and gt/)
        split:         'train' | 'val'  (test has no public labels)
        geo_transform: albumentations Compose for resize/normalize/ToTensor
    """

    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        geo_transform: Optional[Callable] = None,
    ):
        if root is None:
            raise RuntimeError(
                "ACDC root is None. Set acdc_root in your config once the dataset is available."
            )
        self.root = Path(root)
        self.split = split
        self.geo_transform = geo_transform
        self.samples = self._find_samples()

    def _find_samples(self) -> list[dict]:
        img_dir = self.root / "rgb_anon" / "fog" / self.split
        label_dir = self.root / "gt" / "fog" / self.split

        samples = []
        for img_path in sorted(img_dir.rglob("*_rgb_anon.png")):
            seq = img_path.parent.name
            stem = img_path.stem.replace("_rgb_anon", "")
            label_path = label_dir / seq / f"{stem}_gt_labelTrainIds.png"
            if label_path.exists():
                samples.append({"img": img_path, "label": label_path})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img = cv2.imread(str(s["img"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = np.array(cv2.imread(str(s["label"]), cv2.IMREAD_GRAYSCALE))

        result = {"image": img, "label": label, "img_path": str(s["img"])}

        if self.geo_transform is not None:
            transformed = self.geo_transform(image=img, mask=label)
            result["image"] = transformed["image"]
            result["label"] = transformed["mask"]

        return result
