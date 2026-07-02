"""
Side-by-side comparison of the 3 fog generation techniques on one image,
at the intensity that performed best for each in the final results
(photometric=low, depth-aware β=0.005, NN-depth β=0.005).

Usage:
    python scripts/visualize_three_techniques.py \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --idx 0 --save results/figures/fog_techniques_comparison.png
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--save", default="results/figures/fog_techniques_comparison.png")
    args = parser.parse_args()

    ds = CityscapesDataset(root=args.cityscapes_root, split=args.split, return_depth=True)
    sample = ds[args.idx]
    img = sample["image"]

    photo_img = PhotometricFog("low").apply(img)

    depth = disparity_to_depth(sample["raw_disp"], sample["camera"])
    depth = complete_depth(depth, img)
    depthaware_img = KoschmiederFog(beta=0.005).apply(img, depth)

    nn_depth_img = NNDepthFog(beta=0.005).apply(img)

    panels = [
        ("Clear (input)", img),
        ("Technique 1: Photometric\n(Albumentations RandomFog, low)", photo_img),
        ("Technique 2: Depth-aware\n(stereo Koschmieder, β=0.005)", depthaware_img),
        ("Technique 3: NN-depth\n(DepthAnythingV2 Koschmieder, β=0.005)", nn_depth_img),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5))
    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.suptitle("The Three Fog Generation Techniques — Same Source Image", fontsize=14)
    plt.tight_layout()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save, dpi=150)
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
