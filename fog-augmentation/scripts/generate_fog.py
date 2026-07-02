"""
Pre-render fog augmentation to disk (optional cache).

Generates foggy versions of all Cityscapes train images and saves them as
JPEG files. During training, set use_fog_cache: true in the config to load
pre-rendered images instead of generating fog on-the-fly (faster data loading).

Cache layout:
    data/fog_cache/photometric_{low,medium,high}/train/<city>/<stem>.jpg
    data/fog_cache/depth_aware_b{0005,001,002}/train/<city>/<stem>.jpg
    data/fog_cache/nn_depth_b{0005,001,002}/train/<city>/<stem>.jpg
    data/fog_cache/nn_depth_base_b{0005,001,002}/train/<city>/<stem>.jpg   (DepthAnythingV2-Base)

"Mixed intensity" training configs (augmentation.beta / intensity == "mixed")
pick one of the 3 beta/intensity caches at random per sample at train time —
no separate cache needs to be generated for them.

Usage:
    python scripts/generate_fog.py \\
        --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \\
        --output-root    /home/taah3149/Documents/Group_Studies/data/fog_cache \\
        --variants all \\
        --workers 8

NOTE: nn_depth_* variants run DepthAnythingV2 on GPU and are not safe to
parallelise across multiple workers. Pass --workers 1 for those variants.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.cityscapes import CityscapesDataset
from fog.photometric import PhotometricFog, INTENSITIES
from fog.koschmieder import KoschmiederFog
from fog.nn_depth_fog import NNDepthFog, NNDepthEstimator


_SMALL_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
_BASE_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"

VARIANTS = {
    "photometric_low":    lambda: PhotometricFog("low"),
    "photometric_medium": lambda: PhotometricFog("medium"),
    "photometric_high":   lambda: PhotometricFog("high"),
    "depthaware_b0005":   lambda: KoschmiederFog(beta=0.005),
    "depthaware_b001":    lambda: KoschmiederFog(beta=0.010),
    "depthaware_b002":    lambda: KoschmiederFog(beta=0.020),
    "nn_depth_b0005":     lambda: NNDepthFog(beta=0.005, model_id=_SMALL_MODEL_ID),
    "nn_depth_b001":      lambda: NNDepthFog(beta=0.010, model_id=_SMALL_MODEL_ID),
    "nn_depth_b002":      lambda: NNDepthFog(beta=0.020, model_id=_SMALL_MODEL_ID),
    # Base model (~97M params, better depth quality) — upgrade for the
    # nn-depth arm per professor's request that it be the standout technique.
    # Cache dir prefix matches _nn_depth_prefix() in src/train.py.
    "nn_depth_base_b0005": lambda: NNDepthFog(beta=0.005, model_id=_BASE_MODEL_ID),
    "nn_depth_base_b001":  lambda: NNDepthFog(beta=0.010, model_id=_BASE_MODEL_ID),
    "nn_depth_base_b002":  lambda: NNDepthFog(beta=0.020, model_id=_BASE_MODEL_ID),
}

# NN depth variants load a GPU model; serialise them to avoid OOM.
_NN_DEPTH_VARIANTS = {
    "nn_depth_b0005", "nn_depth_b001", "nn_depth_b002",
    "nn_depth_base_b0005", "nn_depth_base_b001", "nn_depth_base_b002",
}

# Variants that share one underlying depth model — the depth map only needs
# to be predicted once per image, then composited at each beta. Keyed by
# cache-dir prefix -> (model_id, {variant_name: beta}).
_NN_DEPTH_FAMILIES = {
    "nn_depth": (_SMALL_MODEL_ID, {"nn_depth_b0005": 0.005, "nn_depth_b001": 0.010, "nn_depth_b002": 0.020}),
    "nn_depth_base": (_BASE_MODEL_ID, {"nn_depth_base_b0005": 0.005, "nn_depth_base_b001": 0.010, "nn_depth_base_b002": 0.020}),
}


def _write_result(result: dict, out_root: Path, variant_name: str, split: str) -> Path:
    img_path = Path(result["img_path"])
    city = img_path.parent.name
    stem = img_path.stem.replace("_leftImg8bit", "")
    out_path = out_root / variant_name / split / city / f"{stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))
    return out_path


def render_sample(args):
    """Pool worker: builds a fresh (cheap) fogger per call. Used for
    photometric/depthaware variants only — never for nn_depth (see
    render_sequential, which avoids reloading the GPU model per image)."""
    idx, cityscapes_root, variant_name, out_root, split = args
    need_depth = variant_name.startswith("depthaware")
    ds = CityscapesDataset(root=cityscapes_root, split=split, return_depth=need_depth)
    sample = ds[idx]
    fogger = VARIANTS[variant_name]()
    result = fogger(sample)
    return _write_result(result, out_root, variant_name, split)


def render_nn_depth_family(root: Path, out_root: Path, family_prefix: str, split: str) -> None:
    """
    Render all beta variants of one nn_depth family (Small or Base) in a
    single pass: the depth map is predicted once per image and reused for
    every beta's Koschmieder composite, instead of rerunning the (expensive,
    especially for the Base model) NN forward pass once per beta as
    render_sequential would if called 3 times.
    """
    model_id, beta_by_variant = _NN_DEPTH_FAMILIES[family_prefix]
    estimator = NNDepthEstimator(model_id=model_id, device="cuda")
    foggers = {name: KoschmiederFog(beta=beta) for name, beta in beta_by_variant.items()}

    ds = CityscapesDataset(root=root, split=split, return_depth=False)
    print(f"  {split}: {len(ds)} images  (family={family_prefix}, variants={list(beta_by_variant)})")
    for i in range(len(ds)):
        sample = ds[i]
        depth = estimator.predict(sample["image"])
        for variant_name, fogger in foggers.items():
            result = dict(sample)
            result["image"] = fogger.apply(sample["image"], depth)
            _write_result(result, out_root, variant_name, split)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(ds)}", end="\r")
    print(f"  {split}: done")


def render_sequential(root: Path, out_root: Path, variant_name: str, split: str) -> None:
    """Single-process rendering with one reused fogger instance — required
    for nn_depth_* so the GPU depth model loads exactly once, not per image.
    Safe to use CUDA here: no multiprocessing fork has happened yet.

    Only used as a fallback when a single nn_depth[_base]_* variant is
    requested on its own; when a full 3-beta family is requested together,
    generate() dispatches to render_nn_depth_family() instead so the (GPU,
    expensive) depth prediction runs once per image instead of 3x."""
    model_id, beta = next(
        (mid, beta_by_variant[variant_name])
        for mid, beta_by_variant in _NN_DEPTH_FAMILIES.values()
        if variant_name in beta_by_variant
    )
    fogger = NNDepthFog(beta=beta, model_id=model_id, device="cuda")

    ds = CityscapesDataset(root=root, split=split, return_depth=False)
    print(f"  {split}: {len(ds)} images")
    for i in range(len(ds)):
        sample = ds[i]
        result = fogger(sample)
        _write_result(result, out_root, variant_name, split)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(ds)}", end="\r")
    print(f"  {split}: done")


def generate(cityscapes_root: str, output_root: str, variants: list[str], workers: int = 4):
    root = Path(cityscapes_root)
    out_root = Path(output_root)
    remaining = list(variants)

    # Dispatch full nn_depth families (all 3 betas requested together) to the
    # single-depth-pass path; anything left over falls through to the normal
    # per-variant loop below.
    for family_prefix, (_, beta_by_variant) in _NN_DEPTH_FAMILIES.items():
        family_variants = list(beta_by_variant)
        if all(v in remaining for v in family_variants):
            print(f"\n=== Rendering {family_prefix} family (single depth pass) ===")
            for split in ("train", "val"):
                render_nn_depth_family(root, out_root, family_prefix, split)
            for v in family_variants:
                remaining.remove(v)

    for variant_name in remaining:
        print(f"\n=== Rendering {variant_name} ===")

        if variant_name in _NN_DEPTH_VARIANTS:
            print("  (sequential, single GPU model instance)")
            for split in ("train", "val"):
                render_sequential(root, out_root, variant_name, split)
            continue

        need_depth = variant_name.startswith("depthaware")
        for split in ("train", "val"):
            ds = CityscapesDataset(root=root, split=split, return_depth=need_depth)
            print(f"  {split}: {len(ds)} images")

            tasks = [(i, str(root), variant_name, out_root, split) for i in range(len(ds))]
            with mp.Pool(workers) as pool:
                for j, _ in enumerate(pool.imap_unordered(render_sample, tasks)):
                    if (j + 1) % 100 == 0:
                        print(f"    {j+1}/{len(ds)}", end="\r")
            print(f"  {split}: done")

    print("\nAll variants rendered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--variants", default="all",
        help=f"Comma-separated list of variants or 'all'. Available: {list(VARIANTS)}"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    chosen = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    generate(args.cityscapes_root, args.output_root, chosen, args.workers)
