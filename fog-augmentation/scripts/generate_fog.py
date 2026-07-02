"""
Pre-render fog augmentation to disk (optional cache).

Generates foggy versions of all Cityscapes train images and saves them as
JPEG files. During training, set use_fog_cache: true in the config to load
pre-rendered images instead of generating fog on-the-fly (faster data loading).

Cache layout:
    data/fog_cache/photometric_{low,medium,high}/train/<city>/<stem>.jpg
    data/fog_cache/depth_aware_b{0005,001,002}/train/<city>/<stem>.jpg
    data/fog_cache/nn_depth_b{0005,001,002}/train/<city>/<stem>.jpg

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
from fog.nn_depth_fog import NNDepthFog


VARIANTS = {
    "photometric_low":    lambda: PhotometricFog("low"),
    "photometric_medium": lambda: PhotometricFog("medium"),
    "photometric_high":   lambda: PhotometricFog("high"),
    "depthaware_b0005":   lambda: KoschmiederFog(beta=0.005),
    "depthaware_b001":    lambda: KoschmiederFog(beta=0.010),
    "depthaware_b002":    lambda: KoschmiederFog(beta=0.020),
    "nn_depth_b0005":     lambda: NNDepthFog(beta=0.005),
    "nn_depth_b001":      lambda: NNDepthFog(beta=0.010),
    "nn_depth_b002":      lambda: NNDepthFog(beta=0.020),
}

# NN depth variants load a GPU model; serialise them to avoid OOM.
_NN_DEPTH_VARIANTS = {"nn_depth_b0005", "nn_depth_b001", "nn_depth_b002"}


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


def render_sequential(root: Path, out_root: Path, variant_name: str, split: str) -> None:
    """Single-process rendering with one reused fogger instance — required
    for nn_depth_* so the GPU depth model loads exactly once, not per image.
    Safe to use CUDA here: no multiprocessing fork has happened yet."""
    from fog.nn_depth_fog import NNDepthFog

    beta = {"nn_depth_b0005": 0.005, "nn_depth_b001": 0.010, "nn_depth_b002": 0.020}[variant_name]
    fogger = NNDepthFog(beta=beta, device="cuda")

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

    for variant_name in variants:
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
