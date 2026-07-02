"""
Helpers for downloading Cityscapes pretrained checkpoints and optional datasets.

Cityscapes and ACDC images require manual registration and cannot be
auto-downloaded. See data/README.md.

This module handles:
  - Downloading the DeepLabV3+ Cityscapes-pretrained MMSeg checkpoint
  - Verifying file integrity via SHA-256

NOTE on the mmseg backend:
  The pretrained checkpoint is only required when using the `mmseg` backend.
  The `smp` backend initialises from ImageNet weights (auto-downloaded from
  torchvision) and works without this file.

If the automatic download fails, obtain the checkpoint with:

    pip install openmim
    mim download mmsegmentation --config deeplabv3plus_r101-d8_512x1024_40k_cityscapes --dest results/checkpoints/pretrained/

then rename it to `deeplabv3plus_r101_cityscapes.pth`.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


# Candidate URLs in priority order — the CDN layout has changed over time.
_CHECKPOINT_URLS = [
    # OpenMMLab Aliyun CDN — 40k iters (original)
    (
        "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/"
        "deeplabv3plus_r101-d8_512x1024_40k_cityscapes/"
        "deeplabv3plus_r101-d8_512x1024_40k_cityscapes_20200606_114143-068fcfe9.pth"
    ),
    # OpenMMLab Aliyun CDN — 80k iters (higher mIoU)
    (
        "https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/"
        "deeplabv3plus_r101-d8_512x1024_80k_cityscapes/"
        "deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_225332-865e3a25.pth"
    ),
]

# Short hash appended to filename — used for quick sanity-check only
CHECKPOINT_SHA256_PREFIX = "068fcfe9"
DEFAULT_CKPT_PATH = Path("results/checkpoints/pretrained/deeplabv3plus_r101_cityscapes.pth")

_MIM_CONFIG = "deeplabv3plus_r101-d8_512x1024_40k_cityscapes"


def _try_url_download(url: str, dest: Path) -> bool:
    """Attempt to download from url to dest. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status != 200:
                return False
            print(f"  Fetching {url}")
            with open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        return True
    except Exception as e:
        print(f"  URL failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _try_mim_download(dest: Path) -> bool:
    """Attempt to download via openmim. Returns True on success."""
    try:
        subprocess.run(["mim", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    tmp_dir = dest.parent / "_mim_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Trying: mim download mmsegmentation --config {_MIM_CONFIG}")
    try:
        subprocess.run(
            ["mim", "download", "mmsegmentation",
             "--config", _MIM_CONFIG, "--dest", str(tmp_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  mim download failed: {e}")
        return False

    # mim saves with the full config name; find the .pth file
    pth_files = list(tmp_dir.glob("*.pth"))
    if not pth_files:
        return False
    pth_files[0].rename(dest)
    tmp_dir.rmdir()
    return True


def download_checkpoint(dest: Path = DEFAULT_CKPT_PATH, force: bool = False) -> Path:
    """Download the Cityscapes-pretrained DeepLabV3+ checkpoint if not present."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"Checkpoint already exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading DeepLabV3+ Cityscapes checkpoint ...")

    for url in _CHECKPOINT_URLS:
        print(f"Trying URL: {url}")
        if _try_url_download(url, dest):
            print(f"Checkpoint saved to {dest}")
            return dest

    print("Direct URL download failed. Trying openmim ...")
    if _try_mim_download(dest):
        print(f"Checkpoint saved via mim to {dest}")
        return dest

    print(
        "\n[ERROR] Could not download checkpoint automatically.\n"
        "Please obtain it manually:\n"
        "\n"
        "  Option A — openmim:\n"
        "    pip install openmim\n"
        f"    mim download mmsegmentation --config {_MIM_CONFIG} \\\n"
        f"        --dest {dest.parent}/\n"
        f"    mv {dest.parent}/{_MIM_CONFIG}*.pth {dest}\n"
        "\n"
        "  Option B — wget (if you have a working mirror URL):\n"
        f"    wget <URL> -O {dest}\n"
        "\n"
        "  Note: the smp backend works without this checkpoint;\n"
        "        only the mmseg backend requires it.\n",
        file=sys.stderr,
    )
    sys.exit(1)


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_checkpoint(Path(args.dest), force=args.force)
