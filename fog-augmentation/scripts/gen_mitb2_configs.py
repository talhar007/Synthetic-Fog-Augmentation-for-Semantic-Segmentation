"""
Clones every config in configs/sweep/*.yaml plus configs/baseline.yaml into
configs/sweep_mitb2/, changing only the backbone (encoder_name: mit_b2,
architecture: fpn) — everything else (fog technique, fog_prob, epochs,
LR, resolution) stays identical, so this is a clean apples-to-apples
resweep isolating the backbone POC's finding.

Usage:
    python scripts/gen_mitb2_configs.py
"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_CONFIGS = sorted((REPO_ROOT / "configs" / "sweep").glob("*.yaml")) + [REPO_ROOT / "configs" / "baseline.yaml"]
OUT_DIR = REPO_ROOT / "configs" / "sweep_mitb2"


def transform(cfg: dict) -> dict:
    cfg["model"]["encoder_name"] = "mit_b2"
    cfg["model"]["architecture"] = "fpn"
    old_name = cfg["experiment"]["name"]
    new_name = f"{old_name}_mitb2"
    cfg["experiment"]["name"] = new_name
    cfg["experiment"]["description"] = cfg["experiment"].get("description", "") + " [backbone: mit_b2/FPN]"
    cfg["training"]["output_dir"] = f"results/{new_name}"
    return cfg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in SRC_CONFIGS:
        with open(src) as f:
            cfg = yaml.safe_load(f)
        cfg = transform(cfg)
        out_path = OUT_DIR / f"{cfg['experiment']['name']}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Wrote {out_path}")
    print(f"\n{len(SRC_CONFIGS)} configs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
