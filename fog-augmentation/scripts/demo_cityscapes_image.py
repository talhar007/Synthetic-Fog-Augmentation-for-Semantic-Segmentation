"""
Full clear -> fog -> prediction demo, like demo_random_image.py, but sourced
from a real Cityscapes image instead of an arbitrary photo — so it supports
EVERY fog type, including depth_aware (stereo Koschmieder), which needs real
stereo disparity that only Cityscapes images have. Reuses show_fog_config.py's
fog-rendering logic (same code path used to render any technique) and
demo_random_image.py's model-loading / legend logic.

Usage:
    python scripts/demo_cityscapes_image.py \
        --config configs/sweep_mitb2/depthaware_b0005_p70_mitb2.yaml \
        --experiment depthaware_b0005_p70_mitb2 \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --filename frankfurt_000001_007973 \
        --save Generated_results/depthaware_b0005_p70_mitb2_demo_cityscapes.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.cityscapes import CityscapesDataset, CLASSES, PALETTE
from models import build_model, forward_eval
from _archive import archive_before_write
from show_fog_config import render
from visualize_predictions import build_eval_transform, model_arch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Fog technique config — any type, incl. depth_aware")
    parser.add_argument("--experiment", required=True, help="results/<experiment> to load best.pth + config.yaml from")
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--filename", default=None,
                         help="Substring of the Cityscapes filename/stem to look up instead of --idx")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--save", default="Generated_results/demo_cityscapes.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    aug_cfg = cfg.get("augmentation", {"type": "none"})

    ds = CityscapesDataset(root=args.cityscapes_root, split=args.split, return_depth=True)
    idx = args.idx
    if args.filename:
        matches = [i for i, s in enumerate(ds.samples) if args.filename in s["img"].name]
        if not matches:
            raise SystemExit(f"No image matching '{args.filename}' found in split={args.split}")
        if len(matches) > 1:
            names = "\n  ".join(ds.samples[i]["img"].name for i in matches)
            raise SystemExit(f"'{args.filename}' matches {len(matches)} images, be more specific:\n  {names}")
        idx = matches[0]
    sample = ds[idx]
    print(f"Using image: {ds.samples[idx]['img']}")

    panels_fog = render(aug_cfg, sample)
    if len(panels_fog) > 1:
        print(f"Note: '{aug_cfg.get('beta') or aug_cfg.get('intensity')}' is a mixed/multi-variant config "
              f"({len(panels_fog)} variants) — using the first ({panels_fog[0][0]}) for the prediction panel.")
    fog_label, fogged_img = panels_fog[0]

    results_dir = Path(args.results_dir)
    ckpt = results_dir / args.experiment / "best.pth"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}")
    encoder_name, architecture = model_arch(results_dir, args.experiment)
    model = build_model(
        "configs/_base_/deeplabv3plus_r101.py",
        checkpoint=str(ckpt),
        encoder_name=encoder_name,
        architecture=architecture,
    ).to(device)
    model.eval()
    print(f"Loaded {args.experiment} (encoder={encoder_name}, architecture={architecture})")

    transform = build_eval_transform()
    H, W = fogged_img.shape[:2]
    transformed = transform(image=fogged_img)
    img_t = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = forward_eval(model, img_t)[0].cpu().numpy()
    pred = pred[:H, :W]

    panels = [
        ("Input (clear)", sample["image"]),
        (fog_label, fogged_img),
        (f"Prediction\n({args.experiment})", CityscapesDataset.decode_target(pred)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.13, wspace=0.04)

    present = sorted(set(np.unique(pred).tolist()) - {255})
    handles = [Patch(color=PALETTE[c] / 255, label=CLASSES[c]) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 10),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(f"Demo on real Cityscapes image (real stereo depth): {ds.samples[idx]['img'].name}\n"
                 f"Prediction is a per-pixel class map (19 fixed Cityscapes categories), not a heatmap.",
                 fontsize=11)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(save_path)
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
