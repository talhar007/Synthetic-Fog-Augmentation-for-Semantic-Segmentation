"""
Side-by-side comparison of every fog generation technique used in the sweep,
on one source image, each at its best-performing config from results/summary.md:
  - Photometric low            (photometric_low_p30, 0.4657 mIoU)
  - Depth-aware stereo, β=0.005 (depthaware_b0005_p10, 0.4589 mIoU)
  - NN-depth (Base model), β=0.005 (nndepth_base_b0005_p30, 0.4780 mIoU — best overall)
  - Combined: stereo + photometric low   (combined_p03, 0.4384 mIoU)
  - Combined: NN-depth (Base) + photometric low (combined_nndepth_p03, 0.4617 mIoU)

Usage:
    python scripts/visualize_all_techniques.py \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --idx 0 --save results/figures/all_techniques_comparison.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.cityscapes import CityscapesDataset
from fog.photometric import PhotometricFog
from fog.depth import disparity_to_depth, complete_depth
from fog.koschmieder import KoschmiederFog
from fog.nn_depth_fog import NNDepthFog

NN_DEPTH_BASE_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--save", default="results/figures/all_techniques_comparison.png")
    args = parser.parse_args()

    ds = CityscapesDataset(root=args.cityscapes_root, split=args.split, return_depth=True)
    sample = ds[args.idx]
    img = sample["image"]

    photo_img = PhotometricFog("low").apply(img)

    depth = disparity_to_depth(sample["raw_disp"], sample["camera"])
    depth = complete_depth(depth, img)
    depthaware_img = KoschmiederFog(beta=0.005).apply(img, depth)

    nn_depth_fogger = NNDepthFog(beta=0.005, model_id=NN_DEPTH_BASE_MODEL)
    nn_depth_img = nn_depth_fogger.apply(img)

    combined_stereo_img = PhotometricFog("low").apply(depthaware_img)
    combined_nndepth_img = PhotometricFog("low").apply(nn_depth_img)

    panels = [
        ("Clear\n(input)", img),
        ("Photometric\n(low)\n0.4657 mIoU", photo_img),
        ("Depth-aware\n(stereo, β=0.005)\n0.4589 mIoU", depthaware_img),
        ("NN-depth\n(Base model, β=0.005)\n0.4780 mIoU — best", nn_depth_img),
        ("Combined\n(stereo + photometric)\n0.4384 mIoU", combined_stereo_img),
        ("Combined\n(NN-depth + photometric)\n0.4617 mIoU", combined_nndepth_img),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.2))
    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle("All Fog Augmentation Techniques — Same Source Image (mIoU = best sweep config)", fontsize=13)
    plt.tight_layout()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save, dpi=150)
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
