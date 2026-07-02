"""mIoU and per-class IoU for 19-class Cityscapes/ACDC evaluation."""
import numpy as np
import torch

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
NUM_CLASSES = 19
IGNORE_INDEX = 255


class SegMetrics:
    """Accumulates confusion matrix across batches, then computes mIoU."""

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred: [B,H,W] int64 class indices; target: [B,H,W] int64."""
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]
        # Clamp out-of-range predictions (safety)
        pred = np.clip(pred, 0, self.num_classes - 1)
        np.add.at(self.confusion, (target, pred), 1)

    def per_class_iou(self) -> np.ndarray:
        tp = np.diag(self.confusion)
        fn = self.confusion.sum(axis=1) - tp
        fp = self.confusion.sum(axis=0) - tp
        denom = tp + fn + fp
        iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def miou(self) -> float:
        iou = self.per_class_iou()
        return float(np.nanmean(iou))

    def summary(self) -> dict:
        iou = self.per_class_iou()
        return {
            "mIoU": float(np.nanmean(iou)),
            "per_class": {cls: float(iou[i]) for i, cls in enumerate(CITYSCAPES_CLASSES)},
        }

    def reset(self) -> None:
        self.confusion[:] = 0
