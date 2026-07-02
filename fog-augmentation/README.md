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
│   └── combined.yaml
├── data/README.md              # How to obtain Cityscapes + ACDC
├── src/
│   ├── data/                   # Dataset loaders (Cityscapes, ACDC)
│   ├── fog/                    # Augmentation modules
│   │   ├── photometric.py      # Albumentations RandomFog (3 intensities)
│   │   ├── depth.py            # Disparity → metric depth + hole filling
│   │   ├── atmospheric.py      # Dark channel prior for atmospheric light
│   │   └── koschmieder.py      # β-parameterised fog compositing
│   ├── models.py               # DeepLabV3+ build (MMSeg or SMP backend)
│   ├── train.py                # Fine-tuning loop (AdamW + poly LR)
│   ├── evaluate.py             # mIoU + per-class IoU on ACDC fog val
│   └── utils/                  # Seeding, metrics, logging
├── scripts/
│   ├── sanity_check.py         # Visualise Cityscapes data + labels
│   ├── visualize_fog.py        # Side-by-side clear vs. all fog variants
│   ├── generate_fog.py         # Pre-render fog to disk (optional cache)
│   ├── compile_results.py      # Aggregate JSONs → CSV + Markdown table
│   ├── run_all_experiments.sh  # Sequential run on local machine
│   └── slurm/
│       ├── train.slurm         # SLURM job template (one GPU, 12 h)
│       ├── evaluate.slurm      # SLURM eval job
│       └── submit_all.sh       # Submit all 8 jobs at once
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

| Condition | Augmentation | β / intensity | Output dir |
|---|---|---|---|
| Baseline | None | — | `results/baseline` |
| Photometric low | RandomFog | low | `results/photometric_low` |
| Photometric med | RandomFog | medium | `results/photometric_med` |
| Photometric high | RandomFog | high | `results/photometric_high` |
| Depth-aware β=0.005 | Koschmieder | 0.005 | `results/depthaware_b0005` |
| Depth-aware β=0.01 | Koschmieder | 0.010 | `results/depthaware_b001` |
| Depth-aware β=0.02 | Koschmieder | 0.020 | `results/depthaware_b002` |
| Combined | Koschmieder + RandomFog | 0.01 + medium | `results/combined` |

All conditions share identical hyperparameters. **Only the augmentation differs.**

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

- Seed: set in each config (`seed: 42`), passed to `seed_everything()` which seeds Python, NumPy, and PyTorch.
- Git commit + config YAML are saved alongside each run in `results/<exp>/`.
- All experiments use the same crop size, LR, weight decay, and number of epochs.
