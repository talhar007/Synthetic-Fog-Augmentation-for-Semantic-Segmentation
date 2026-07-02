"""
Generate the fog_prob / mixed-intensity sweep configs (23 runs) that fix the
"every augmented condition loses to baseline" problem: the original 11
configs applied fog to 100% of training images at one fixed intensity, with
no diversity. This sweep is the single source of truth for that matrix —
re-run it any time to regenerate configs/sweep/ identically.

See PRESENTATION_BRIEF.md / README.md "Reproducing this project" for the
full pipeline this plugs into.

Usage:
    python scripts/gen_sweep_configs.py
    # writes 23 YAML files to configs/sweep/
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SWEEP_DIR = CONFIGS_DIR / "sweep"

NN_DEPTH_BASE_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"

# fog_prob grid, centred on the professor's "mostly real, some generated"
# suggestion (0.1 ~= 90% real / 10% generated).
FOG_PROB_SWEEP = [0.1, 0.3, 0.5, 0.7]
FIXED_FOG_PROB = 0.3  # used for stages B/C/D (see plan)


def _base_common() -> dict:
    return {
        "model": {
            "mmseg_config": "configs/_base_/deeplabv3plus_r101.py",
            "checkpoint": "results/checkpoints/pretrained/deeplabv3plus_r101_cityscapes.pth",
            "num_classes": 19,
        },
        "data": {
            "cityscapes_root": "/home/taah3149/Documents/Group_Studies/dataset/cityscapes",
            "acdc_root": "/home/taah3149/Documents/Group_Studies/dataset/acdc",
            "img_size": [2048, 1024],
            "crop_size": [1024, 512],
            "scale_range": [0.5, 2.0],
            "batch_size": 4,
            "num_workers": 4,
            "use_fog_cache": False,
        },
        "training": {
            "epochs": 50,
            "lr": 6.0e-5,
            "weight_decay": 0.01,
            "poly_power": 1.0,
            "val_interval": 5,
            "log_interval": 50,
            "resume": None,
        },
        "evaluate": {"split": "val"},
    }


def _make(name: str, description: str, augmentation: dict, use_fog_cache: bool = False) -> dict:
    cfg = _base_common()
    cfg["experiment"] = {"name": name, "seed": 42, "description": description}
    cfg["augmentation"] = augmentation
    cfg["data"]["use_fog_cache"] = use_fog_cache
    cfg["training"]["output_dir"] = f"results/{name}"
    return cfg


def build_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}

    # ── Stage A: fog_prob sweep at each family's lightest (best-known) intensity ──
    for p in FOG_PROB_SWEEP:
        p_tag = f"p{int(p * 100):02d}"

        name = f"photometric_low_{p_tag}"
        configs[name] = _make(
            name, f"Photometric low, fog_prob={p} ({int(p*100)}% of training images fogged)",
            {"type": "photometric", "intensity": "low", "fog_prob": p},
        )

        name = f"depthaware_b0005_{p_tag}"
        configs[name] = _make(
            name, f"Depth-aware (stereo) beta=0.005, fog_prob={p}",
            {"type": "depth_aware", "beta": 0.005, "depth_method": "simple", "fog_prob": p},
            use_fog_cache=True,
        )

        name = f"nndepth_base_b0005_{p_tag}"
        configs[name] = _make(
            name, f"NN-depth (DepthAnythingV2-Base) beta=0.005, fog_prob={p}",
            {"type": "nn_depth", "beta": 0.005, "model_id": NN_DEPTH_BASE_MODEL_ID, "fog_prob": p},
            use_fog_cache=True,
        )

    # ── Stage B: mixed-intensity variant per family, fixed fog_prob ──
    p = FIXED_FOG_PROB
    configs["photometric_mixed_p03"] = _make(
        "photometric_mixed_p03", f"Photometric, random intensity per sample, fog_prob={p}",
        {"type": "photometric", "intensity": "mixed", "fog_prob": p},
    )
    configs["depthaware_mixed_p03"] = _make(
        "depthaware_mixed_p03", f"Depth-aware (stereo), random beta per sample, fog_prob={p}",
        {"type": "depth_aware", "beta": "mixed", "depth_method": "simple", "fog_prob": p},
        use_fog_cache=True,
    )
    configs["nndepth_base_mixed_p03"] = _make(
        "nndepth_base_mixed_p03", f"NN-depth (Base), random beta per sample, fog_prob={p}",
        {"type": "nn_depth", "beta": "mixed", "model_id": NN_DEPTH_BASE_MODEL_ID, "fog_prob": p},
        use_fog_cache=True,
    )

    # ── Stage C: remaining fixed intensities at fog_prob=0.3 ──
    for intensity in ("medium", "high"):
        name = f"photometric_{'med' if intensity == 'medium' else 'high'}_p03"
        configs[name] = _make(
            name, f"Photometric {intensity}, fog_prob={p}",
            {"type": "photometric", "intensity": intensity, "fog_prob": p},
        )
    for beta, tag in ((0.010, "b001"), (0.020, "b002")):
        name = f"depthaware_{tag}_p03"
        configs[name] = _make(
            name, f"Depth-aware (stereo) beta={beta}, fog_prob={p}",
            {"type": "depth_aware", "beta": beta, "depth_method": "simple", "fog_prob": p},
            use_fog_cache=True,
        )
        name = f"nndepth_base_{tag}_p03"
        configs[name] = _make(
            name, f"NN-depth (Base) beta={beta}, fog_prob={p}",
            {"type": "nn_depth", "beta": beta, "model_id": NN_DEPTH_BASE_MODEL_ID, "fog_prob": p},
            use_fog_cache=True,
        )

    # ── Stage D: combined, with the mixing fix and the upgraded NN-depth arm ──
    configs["combined_p03"] = _make(
        "combined_p03",
        f"Combined stereo depth-aware (beta=0.005) + photometric low, fog_prob={p}",
        {
            "type": "combined", "depth_technique": "stereo", "beta": 0.005,
            "photometric_intensity": "low", "depth_method": "simple", "fog_prob": p,
        },
        use_fog_cache=True,
    )
    configs["combined_nndepth_p03"] = _make(
        "combined_nndepth_p03",
        f"Combined NN-depth (Base, beta=0.005) + photometric low, fog_prob={p}",
        {
            "type": "combined", "depth_technique": "nn_depth", "beta": 0.005,
            "model_id": NN_DEPTH_BASE_MODEL_ID, "photometric_intensity": "low", "fog_prob": p,
        },
        use_fog_cache=True,
    )

    return configs


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    configs = build_configs()
    for name, cfg in configs.items():
        out_path = SWEEP_DIR / f"{name}.yaml"
        out_path.write_text(yaml.dump(copy.deepcopy(cfg), sort_keys=False))
        print(f"Wrote {out_path}")
    print(f"\n{len(configs)} sweep configs written to {SWEEP_DIR}/")


if __name__ == "__main__":
    main()
