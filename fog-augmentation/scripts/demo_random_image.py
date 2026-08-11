"""
Live-demo tool: fog an arbitrary image (any photo handed to you on the
spot — not from Cityscapes or ACDC) with a chosen technique, then run a
trained model on the fogged result. Shows clear | fogged | prediction.

Only photometric and nn_depth fog types work on an arbitrary image —
depth_aware (stereo Koschmieder) and combined configs with
depth_technique: stereo need Cityscapes' precomputed stereo disparity maps,
which don't exist for a random photo. Use an nn_depth config (or combined
with depth_technique: nn_depth) for a depth-based look instead — it
estimates depth monocularly via DepthAnythingV2, so it works on any image.

Usage:
    python scripts/demo_random_image.py \
        --image /path/to/random_photo.jpg \
        --fog-config configs/sweep_mitb2/nndepth_base_b001_p03_mitb2.yaml \
        --experiment nndepth_base_b001_p03_mitb2 \
        --save Generated_results/demo_random_photo.png
"""
import argparse
import sys
from pathlib import Path

import cv2
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
from fog.photometric import PhotometricFog
from fog.nn_depth_fog import NNDepthFog
from models import build_model, forward_eval
from _archive import archive_before_write
from visualize_predictions import build_eval_transform, model_arch


def render_fog(aug_cfg: dict, img):
    """Same fog classes as show_fog_config.py, minus the depth_aware/stereo
    branch — those need Cityscapes disparity maps this image doesn't have."""
    aug_type = aug_cfg.get("type", "none")

    if aug_type == "none":
        return img, "No fog"

    if aug_type == "photometric":
        intensity = aug_cfg["intensity"]
        if intensity == "mixed":
            intensity = "medium"  # no per-sample RNG context here; show the middle variant
        return PhotometricFog(intensity).apply(img), f"photometric ({intensity})"

    if aug_type == "nn_depth":
        model_id = aug_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
        beta = aug_cfg["beta"]
        if beta == "mixed":
            beta = 0.01
        return NNDepthFog(beta=beta, model_id=model_id).apply(img), f"nn-depth, β={beta}"

    if aug_type == "combined":
        technique = aug_cfg.get("depth_technique", "stereo")
        if technique == "stereo":
            raise SystemExit(
                "This config's depth arm is stereo Koschmieder, which needs Cityscapes' "
                "precomputed disparity maps — not available for an arbitrary image. Pick a "
                "combined config with depth_technique: nn_depth, or a plain nn_depth/"
                "photometric config instead."
            )
        model_id = aug_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
        beta = aug_cfg["beta"]
        if beta == "mixed":
            beta = 0.01
        photo_intensity = aug_cfg["photometric_intensity"]
        depth_img = NNDepthFog(beta=beta, model_id=model_id).apply(img)
        label = f"combined: nn-depth β={beta} + photometric ({photo_intensity})"
        return PhotometricFog(photo_intensity).apply(depth_img), label

    raise SystemExit(
        f"Unsupported augmentation type for arbitrary images: {aug_type!r}. "
        "depth_aware (stereo) requires Cityscapes disparity data and can't be used here."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to any image file (not Cityscapes/ACDC)")
    parser.add_argument("--fog-config", required=True, help="Fog technique config (photometric or nn_depth type)")
    parser.add_argument("--experiment", required=True, help="results/<experiment> to load best.pth + config.yaml from")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-side", type=int, default=2048,
                         help="Downscale so the longer side is at most this many px "
                              "(arbitrary photos can be far larger than Cityscapes/ACDC)")
    parser.add_argument("--save", default="Generated_results/demo_random_image.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = args.max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"Resized {w}x{h} → {img.shape[1]}x{img.shape[0]} (--max-side={args.max_side})")

    with open(args.fog_config) as f:
        fog_cfg = yaml.safe_load(f)
    aug_cfg = fog_cfg.get("augmentation", {"type": "none"})
    fogged_img, fog_label = render_fog(aug_cfg, img)

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
    pred = pred[:H, :W]  # crop the pad-to-32 border back off

    panels = [
        ("Input (clear)", img),
        (fog_label, fogged_img),
        (f"Prediction\n({args.experiment})", CityscapesDataset.decode_target(pred)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.13, wspace=0.04)

    # Legend: this is a per-pixel predicted CLASS map (19 fixed Cityscapes
    # categories, one flat color each) — not a continuous heatmap — so the
    # colors are meaningless without a legend. Only show classes actually
    # present in this prediction, not all 19, to keep it readable.
    present = sorted(set(np.unique(pred).tolist()) - {255})
    handles = [Patch(color=PALETTE[c] / 255, label=CLASSES[c]) for c in present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 10),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(f"Live demo — arbitrary image (not Cityscapes/ACDC): {Path(args.image).name}\n"
                 f"No ground truth exists for this image, so this is qualitative only. "
                 f"Prediction is a per-pixel class map (19 fixed Cityscapes categories), not a heatmap.",
                 fontsize=11)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    archive_before_write(save_path)
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
