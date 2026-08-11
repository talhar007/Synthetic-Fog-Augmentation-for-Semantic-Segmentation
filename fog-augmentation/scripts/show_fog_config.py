"""
Show exactly what fog a given experiment config produces, on one Cityscapes
image — for answering "what does <config_name> actually look like?" on the
spot (e.g. in a live meeting), without training anything.

Reads the config's `augmentation` block and renders it with the same fog
classes used by src/train.py (see build_fog_transform there), so what you see
here is exactly what that config trains on.

Note: `fog_prob` (if present) is a TRAINING-time knob — the probability a
given sample gets fogged each epoch — it does not change how the fog itself
looks. A config with fog_prob=0.7 looks visually identical to the same
technique/intensity at fog_prob=0.1; only the fraction of epochs it's applied
differs. This script prints that reminder whenever fog_prob is set.

`beta: mixed` configs render all three beta variants side by side, since the
dataset picks one at random per sample during training rather than having one
fixed look.

Usage:
    python scripts/show_fog_config.py \
        --config configs/sweep/photometric_low_p70.yaml \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --idx 0 --save /tmp/photometric_low_p70.png

    # or pick a specific photo by filename instead of a numeric index:
    python scripts/show_fog_config.py \
        --config configs/sweep_mitb2/depthaware_b001_p03_mitb2.yaml \
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
        --filename frankfurt_000001_007973 --save /tmp/that_photo.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.cityscapes import CityscapesDataset
from fog.photometric import PhotometricFog
from fog.depth import disparity_to_depth, complete_depth
from fog.koschmieder import KoschmiederFog
from fog.nn_depth_fog import NNDepthFog

BETAS = [0.005, 0.010, 0.020]


def _stereo_depth(sample):
    depth = disparity_to_depth(sample["raw_disp"], sample["camera"])
    return complete_depth(depth, sample["image"])


def render(aug_cfg: dict, sample: dict) -> list[tuple[str, "np.ndarray"]]:
    """Return [(panel_title, image), ...] for the given augmentation config."""
    img = sample["image"]
    aug_type = aug_cfg.get("type", "none")

    if aug_type == "none":
        return [("No fog augmentation", img)]

    if aug_type == "photometric":
        intensity = aug_cfg["intensity"]
        if intensity == "mixed":
            return [(f"photometric ({i})", PhotometricFog(i).apply(img))
                    for i in ("low", "medium", "high")]
        return [(f"photometric ({intensity})", PhotometricFog(intensity).apply(img))]

    if aug_type == "depth_aware":
        depth = _stereo_depth(sample)
        beta = aug_cfg["beta"]
        if beta == "mixed":
            return [(f"depth-aware, β={b}", KoschmiederFog(beta=b).apply(img, depth)) for b in BETAS]
        return [(f"depth-aware, β={beta}", KoschmiederFog(beta=beta).apply(img, depth))]

    if aug_type == "nn_depth":
        model_id = aug_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
        beta = aug_cfg["beta"]
        if beta == "mixed":
            fogger_kwargs = dict(model_id=model_id)
            return [(f"nn-depth, β={b}", NNDepthFog(beta=b, **fogger_kwargs).apply(img)) for b in BETAS]
        return [(f"nn-depth, β={beta}", NNDepthFog(beta=beta, model_id=model_id).apply(img))]

    if aug_type == "combined":
        technique = aug_cfg.get("depth_technique", "stereo")
        photo_intensity = aug_cfg["photometric_intensity"]
        beta = aug_cfg["beta"]
        if technique == "stereo":
            depth_img = KoschmiederFog(beta=beta).apply(img, _stereo_depth(sample))
            label = f"combined: stereo β={beta} + photometric ({photo_intensity})"
        else:
            model_id = aug_cfg.get("model_id", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
            depth_img = NNDepthFog(beta=beta, model_id=model_id).apply(img)
            label = f"combined: nn-depth β={beta} + photometric ({photo_intensity})"
        return [(label, PhotometricFog(photo_intensity).apply(depth_img))]

    raise ValueError(f"Unknown augmentation type: {aug_type!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--filename", default=None,
                         help="Substring of the Cityscapes image filename/stem (e.g. "
                              "'frankfurt_000001_007973') to look up instead of --idx — "
                              "use this when someone points at a specific photo.")
    parser.add_argument("--save", default=None, help="Defaults to <config_name>_fog_sample.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    aug_cfg = cfg.get("augmentation", {"type": "none"})
    name = cfg["experiment"]["name"]

    save_path = Path(args.save) if args.save else Path(f"{name}_fog_sample.png")

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

    panels = [("Clear (input)", sample["image"])] + render(aug_cfg, sample)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, panel_img) in zip(axes, panels):
        ax.imshow(panel_img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    suptitle = f"{name}  —  {cfg['experiment'].get('description', aug_cfg.get('type'))}"
    if "fog_prob" in aug_cfg:
        suptitle += f"\n(fog_prob={aug_cfg['fog_prob']}: training-time sampling frequency only — does not change how the fog looks)"
    plt.suptitle(suptitle, fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
