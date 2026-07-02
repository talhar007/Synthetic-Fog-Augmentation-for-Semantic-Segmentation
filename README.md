# Synthetic Fog Augmentation for Semantic Segmentation

TU Ilmenau, Dept. of Virtual Worlds and Digital Games  
Controlled comparison of photometric vs. depth-aware (Koschmieder) fog augmentation for DeepLabV3+ trained on Cityscapes, evaluated on the ACDC fog benchmark.

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Depth-aware (Foggy Cityscapes) vs. photometric (RandomFog) fine-tuning → mIoU on ACDC fog? |
| **RQ2** | Which β ∈ {0.005, 0.01, 0.02} transfers best to real fog? |
| **RQ3** | Does photometric + depth-aware combined beat either alone? |

---

## Repository structure

```
fog-augmentation/
├── configs/                    # One YAML per experiment condition
│   ├── _base_/deeplabv3plus_r101.py  # MMSeg model architecture
│   ├── baseline.yaml
│   ├── photometric_{low,med,high}.yaml
│   ├── depthaware_b{0005,001,002}.yaml
│   ├── nn_depth_b{0005,001,002}.yaml
│   ├── combined.yaml
│   └── sweep/                  # fog_prob / mixed-intensity sweep (generated,
│                                # see gen_sweep_configs.py — not hand-edited)
├── data/README.md              # How to obtain Cityscapes + ACDC
├── src/
│   ├── data/                   # Dataset loaders (Cityscapes, ACDC)
│   ├── fog/                    # Augmentation modules
│   │   ├── photometric.py      # Albumentations RandomFog (3 intensities + mixed)
│   │   ├── depth.py            # Disparity → metric depth + hole filling
│   │   ├── atmospheric.py      # Dark channel prior for atmospheric light
│   │   ├── koschmieder.py      # β-parameterised fog compositing
│   │   └── nn_depth_fog.py     # DepthAnythingV2 (Small/Base) + Koschmieder
│   ├── models.py               # DeepLabV3+ build (MMSeg or SMP backend)
│   ├── train.py                # Fine-tuning loop (AdamW + poly LR)
│   ├── evaluate.py             # mIoU + per-class IoU on ACDC fog val
│   └── utils/                  # Seeding, metrics, logging
├── scripts/
│   ├── sanity_check.py         # Visualise Cityscapes data + labels
│   ├── visualize_fog.py        # Side-by-side clear vs. all fog variants
│   ├── generate_fog.py         # Pre-render fog to disk (optional cache)
│   ├── gen_sweep_configs.py    # Generate configs/sweep/*.yaml (single source of truth)
│   ├── compile_results.py      # Aggregate JSONs → CSV + Markdown table (auto-discovers experiments)
│   ├── plot_results.py / plot_training_curves.py / visualize_predictions.py  # Figures
│   ├── run_all_experiments.sh  # Sequential run on local machine
│   └── slurm/
│       ├── train.slurm         # SLURM job template (one GPU, 12 h)
│       ├── evaluate.slurm      # SLURM eval job
│       ├── submit_all.sh       # Submit the original 11 jobs
│       └── submit_sweep.sh     # Submit the 23 configs/sweep/ jobs (chained train→eval)
└── results/                    # Metrics, checkpoints, qualitative samples
```

---

## Quick-start

### 1. Environment

```bash
cd fog-augmentation

# Activate the shared venv
source /home/taah3149/Documents/Group_Studies/.venv/bin/activate

# On the HPC, load GCC 9 + CUDA 12.9 before using PyTorch / mmcv:
module load gcc/v9.3.0
module load cuda/v12.9
export CUDA_HOME=/usr/app-soft1/cuda/v12.9
```

### 2. Get datasets

See [data/README.md](data/README.md) for manual download links (both datasets require free registration).

After downloading, extract Cityscapes under:
```
/home/taah3149/Documents/Group_Studies/dataset/cityscapes/
```

### 3. Download the pretrained checkpoint

```bash
python src/data/download.py
# saves to: results/checkpoints/pretrained/deeplabv3plus_r101_cityscapes.pth
```

### 4. Sanity check

```bash
python scripts/sanity_check.py \
    --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
    --save results/sanity.png
```

### 5. Visualise fog

```bash
python scripts/visualize_fog.py \
    --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
    --save results/fog_comparison.png
```

### 6. Train one experiment

```bash
python src/train.py --config configs/photometric_med.yaml
```

### 7. Run all experiments on HPC (SLURM)

```bash
bash scripts/slurm/submit_all.sh
# Check queue: squeue -u $USER
```

### 8. Evaluate on ACDC fog val

```bash
# Once ACDC is downloaded:
python src/evaluate.py \
    --config     configs/photometric_med.yaml \
    --checkpoint results/photometric_med/best.pth \
    --acdc-root  /path/to/acdc \
    --output     results/photometric_med/acdc_eval.json
```

### 9. Compile results table

```bash
python scripts/compile_results.py
# Writes: results/miou_table.csv, results/per_class_iou.csv, results/summary.md
```

---

## Experiment matrix

### Original 11 (fog applied to 100% of training images — kept as the "before" baseline)

| Condition | Augmentation | β / intensity | Output dir |
|---|---|---|---|
| Baseline | None | — | `results/baseline` |
| Photometric low/med/high | RandomFog | low / medium / high | `results/photometric_{low,med,high}` |
| Depth-aware (stereo) | Koschmieder | β = 0.005 / 0.01 / 0.02 | `results/depthaware_b{0005,001,002}` |
| NN-depth (Small model) | DepthAnythingV2-Small + Koschmieder | β = 0.005 / 0.01 / 0.02 | `results/nn_depth_b{0005,001,002}` |
| Combined | Koschmieder + RandomFog | 0.01 + medium | `results/combined` |

All 11 share identical hyperparameters and `augmentation.fog_prob` (implicitly 1.0 — fog on every
sample). **Only the augmentation differs.** This is the methodology that produced
`PRESENTATION_BRIEF.md`'s finding that no augmented condition beat baseline.

### Sweep (23 runs, `configs/sweep/`) — fixes fog_prob + intensity diversity

Generated by `scripts/gen_sweep_configs.py`; see that file for the exact matrix. Same 50-epoch /
lr=6e-5 / batch-4 budget as the original 11 — the only new variable is `augmentation.fog_prob`
and, for 3 of the 23, `intensity: mixed` / `beta: mixed`. Also upgrades the NN-depth arm to the
DepthAnythingV2 **Base** model (cache dirs `nn_depth_base_*`).

| Stage | Configs | Purpose |
|---|---|---|
| A | `{photometric_low,depthaware_b0005,nndepth_base_b0005}_p{01,03,05,07}` | fog_prob sweep at each family's lightest intensity |
| B | `{photometric,depthaware,nndepth_base}_mixed_p03` | random intensity/β per sample |
| C | `{photometric_med,photometric_high,depthaware_b001,depthaware_b002,nndepth_base_b001,nndepth_base_b002}_p03` | remaining intensities under the fix |
| D | `combined_p03`, `combined_nndepth_p03` | RQ3 retest, incl. the upgraded NN-depth arm |

---

## Fog mixing & diversity config keys

Added to fix the "every augmented condition loses to baseline" problem (see
`PRESENTATION_BRIEF.md`): applying fog to 100% of training images at one fixed intensity gave the
model no exposure to clear-weather statistics and no augmentation diversity. New `augmentation.*`
config keys:

| Key | Applies to | Meaning |
|---|---|---|
| `fog_prob` | all types | Probability a given training sample gets fogged; the rest keep the clear image. Default `1.0` (old behaviour). `0.1` ≈ "90% real / 10% generated". |
| `intensity: mixed` | `photometric` | Randomly picks low/medium/high per sample instead of one fixed intensity. |
| `beta: mixed` | `depth_aware`, `nn_depth` | Randomly picks one of the 3 pre-rendered β caches per sample. Requires `use_fog_cache: true` (no new rendering needed — reuses the existing single-β caches). |
| `depth_technique: stereo \| nn_depth` | `combined` | Which cached depth arm to layer photometric fog on top of. |
| `model_id` (with `"Base"` in the name) | `nn_depth`, `combined` w/ `depth_technique: nn_depth` | Selects the `nn_depth_base_*` cache instead of the original `nn_depth_*` (Small model) cache. |

Implementation: `CityscapesDataset.fog_prob` / `fog_cache_dirs` in `src/data/cityscapes.py`, wired
from config via `fog_cache_variant_names()` in `src/train.py`.

---

## Model backend

| Situation | Backend | Starting weights |
|---|---|---|
| Local development (no nvcc) | `segmentation_models_pytorch` | ImageNet pretrained |
| HPC (gcc 9 + cuda loaded) | MMSegmentation | **Cityscapes pretrained** (downloaded via `src/data/download.py`) |

The backend is selected automatically. To force a backend:  
`MMCV_BACKEND=smp python src/train.py ...` or `MMCV_BACKEND=mmseg ...`

---

## mmcv installation (HPC, one-time)

```bash
module load gcc/v9.3.0
module load cuda/v12.9
export CUDA_HOME=/usr/app-soft1/cuda/v12.9
pip install wheel ninja
pip install mmcv==2.2.0 --no-build-isolation   # compiles CUDA ops (~10 min)
pip install mmsegmentation mmengine
```

---

## Reproducibility

- Seed: set in each config (`seed: 42`), passed to `seed_everything()` which seeds Python, NumPy,
  and PyTorch. DataLoader workers each get an independent, deterministic `random`/`numpy`/`torch`
  seed (`base_seed + worker_id`) automatically — this is handled internally by PyTorch's
  `_worker_loop` (confirmed on torch 2.5.1; no manual `worker_init_fn` needed).
- Git commit + config YAML are saved alongside each run in `results/<exp>/`.
- All experiments use the same crop size, LR, weight decay, and number of epochs; only
  `augmentation` (and, in the sweep, `fog_prob`) differs between conditions.

### Reproducing every result in this project, from a clean checkout

```bash
# 1. Environment
cd fog-augmentation
source /home/taah3149/Documents/Group_Studies/.venv/bin/activate
module load gcc/v9.3.0 && module load cuda/v12.9
export CUDA_HOME=/usr/app-soft1/cuda/v12.9

# 2. Datasets — see data/README.md for download links (both need free registration).
#    Extract Cityscapes to dataset/cityscapes/, ACDC to dataset/acdc/ (paths in each config's
#    data.cityscapes_root / data.acdc_root).

# 3. Pretrained checkpoint
python src/data/download.py

# 4. Fog cache (required for depth_aware / nn_depth / combined / sweep — cheap ones like
#    photometric compute on the fly and don't need this)
python scripts/generate_fog.py \
    --cityscapes-root /home/taah3149/Documents/Group_Studies/dataset/cityscapes \
    --output-root     data/fog_cache \
    --variants all       # includes the new nn_depth_base_* (Base model) variants

# 5. Train + evaluate the original 11 (the "before" methodology)
bash scripts/run_all_experiments.sh    # trains + evaluates on ACDC fog val, idempotent/resumable

# 6. Generate and run the fog_prob / mixed-intensity sweep (the fix, 23 runs)
python scripts/gen_sweep_configs.py    # writes configs/sweep/*.yaml
bash scripts/slurm/submit_sweep.sh     # SLURM: submits all 23, each train job chained to its eval
# (no SLURM available? run individually instead:
#    for cfg in configs/sweep/*.yaml; do python src/train.py --config "$cfg"; done)

# 7. Compile everything (34 experiments total) into results/summary.md,
#    results/miou_table.csv, results/per_class_iou.csv
python scripts/compile_results.py

# 8. Regenerate figures
python scripts/plot_results.py
python scripts/plot_training_curves.py
python scripts/visualize_predictions.py --acdc-root /home/taah3149/Documents/Group_Studies/dataset/acdc
python scripts/visualize_three_techniques.py   # if present, see script for args
```

`results/summary.md`'s ranked table is the canonical numbers table — it's regenerated fresh by
step 7 above and tags each row with its technique `family` and whether it's part of the original
(pre-fix) 11, so the before/after comparison for the professor is a direct read of that file.
