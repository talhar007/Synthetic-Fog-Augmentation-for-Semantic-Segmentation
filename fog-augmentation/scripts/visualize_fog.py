"""
Phase 2 visual check.

Shows a Cityscapes image under:
  - clear baseline
  - photometric fog × 3 intensities
  - depth-aware Koschmieder fog (stereo disparity) if available
  - NN depth fog (DepthAnythingV2 + Koschmieder) if --show-nn-depth is set

Usage:
    python scripts/visualize_fog.py \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --idx 0 --save fog_comparison.png

    # Include NN depth column (downloads model on first run):
    python scripts/visualize_fog.py ... --show-nn-depth
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
from fog.photometric import PhotometricFog, INTENSITIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--save", default="fog_comparison.png")
    parser.add_argument("--show-nn-depth", action="store_true",
                        help="Add NN depth fog column (downloads DepthAnythingV2 on first run)")
    args = parser.parse_args()

    ds = CityscapesDataset(root=args.cityscapes_root, split=args.split, return_depth=True)
    sample = ds[args.idx]
    img = sample["image"]

    foggers = {k: PhotometricFog(k) for k in INTENSITIES}

    # Try depth-aware fog (stereo disparity)
    depth_foggy = None
    if "raw_disp" in sample:
        try:
            from fog.depth import disparity_to_depth, complete_depth
            from fog.koschmieder import KoschmiederFog

            depth = disparity_to_depth(sample["raw_disp"], sample["camera"])
            depth = complete_depth(depth, img)
            fogger = KoschmiederFog(beta=0.01)
            depth_foggy = fogger.apply(img, depth)
        except Exception as e:
            print(f"Depth-aware fog skipped ({e})")

    # NN depth fog (optional — loads GPU model)
    nn_foggy = None
    if args.show_nn_depth:
        try:
            from fog.nn_depth_fog import NNDepthFog
            nn_fogger = NNDepthFog(beta=0.01)
            nn_foggy = nn_fogger.apply(img)
        except Exception as e:
            print(f"NN depth fog skipped ({e})")

    panels = [("Clear", img)]
    for name, fogger in foggers.items():
        panels.append((f"Photometric\n{name}", fogger.apply(img)))
    if depth_foggy is not None:
        panels.append(("Depth-aware\n(stereo) β=0.01", depth_foggy))
    if nn_foggy is not None:
        panels.append(("NN depth\n(DAv2) β=0.01", nn_foggy))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Fog augmentation comparison — sample {args.idx}", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.save, dpi=100)
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
