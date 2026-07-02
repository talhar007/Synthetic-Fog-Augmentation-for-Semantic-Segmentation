"""
Phase 1 sanity check.

Loads a few Cityscapes samples, verifies label trainId mapping,
checks ignore_index=255 is present, and shows image/label side-by-side.

Usage:
    python scripts/sanity_check.py \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --n 4 --save sanity_check.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.cityscapes import CityscapesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--save", default="sanity_check.png")
    args = parser.parse_args()

    ds = CityscapesDataset(root=args.cityscapes_root, split=args.split, return_depth=True)
    print(f"Dataset size ({args.split}): {len(ds)}")
    assert len(ds) > 0, "No samples found! Check --cityscapes-root path."

    fig, axes = plt.subplots(args.n, 3, figsize=(15, 4 * args.n))
    if args.n == 1:
        axes = axes[np.newaxis]

    for row, idx in enumerate(np.linspace(0, len(ds) - 1, args.n, dtype=int)):
        sample = ds[idx]
        img = sample["image"]
        label = sample["label"]

        # ---- label checks ----
        unique = np.unique(label)
        valid_ids = set(range(19)) | {255}
        bad = set(unique) - valid_ids
        assert not bad, f"Unexpected label ids: {bad}"
        has_ignore = 255 in unique
        print(
            f"[{idx}] img={img.shape} label={label.shape} "
            f"classes={sorted(set(unique)-{255})} ignore255={has_ignore}"
        )

        colour_label = CityscapesDataset.decode_target(label)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title("RGB image")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(colour_label)
        axes[row, 1].set_title("Label (colour)")
        axes[row, 1].axis("off")

        if "raw_disp" in sample:
            d = sample["raw_disp"]
            valid = d > 0
            axes[row, 2].imshow(np.where(valid, d, np.nan), cmap="plasma")
            axes[row, 2].set_title(f"Disparity  valid={valid.mean()*100:.1f}%")
        else:
            axes[row, 2].set_visible(False)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(args.save, dpi=100)
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
