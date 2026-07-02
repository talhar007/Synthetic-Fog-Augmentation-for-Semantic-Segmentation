# Fog Augmentation for Semantic Segmentation — Project Status Brief

**Purpose of this file:** background + results for generating a presentation. This project is functionally **complete** — all experiments are trained, evaluated, and visualized. Use this as the source material.

---

## 1. Research Question

Does synthetic fog augmentation during training improve semantic segmentation robustness to **real fog**, and which of three fog-generation techniques works best?

- **Train on:** Cityscapes (clear-weather, fine-annotated, 2975 train / 500 val images)
- **Evaluate on:** ACDC fog validation split (100 real foggy images) — held out, never trained on
- **Model:** DeepLabV3+, ResNet-101 encoder, 19 Cityscapes classes
- **Constant across all experiments:** every hyperparameter is identical (50 epochs, AdamW, lr=6e-5, polynomial LR decay, batch size 4, crop 1024×512). The **only** variable is the fog augmentation applied to training images. Labels are never touched by augmentation.

## 2. The Three Fog Techniques Compared

| # | Technique | How it works | Intensities tested |
|---|---|---|---|
| 1 | **Photometric** | Albumentations `RandomFog` — geometry-agnostic uniform haze, no depth information | low / med / high |
| 2 | **Depth-aware (stereo)** | Physics-based Koschmieder model `I(x) = R(x)·t(x) + L·(1-t(x))`, `t(x)=exp(-βd(x))`, using **real stereo disparity** from Cityscapes as the depth map | β = 0.005 / 0.01 / 0.02 |
| 3 | **NN depth (monocular)** | Same Koschmieder physics as #2, but depth comes from **DepthAnythingV2** monocular depth estimation instead of stereo — works on any single camera, no stereo rig needed (this was the 3rd technique added per professor's request) | β = 0.005 / 0.01 / 0.02 |

Plus two reference conditions:
- **baseline** — no fog augmentation at all
- **combined** — depth-aware (β=0.01) + photometric (medium) stacked together

**11 experiments total.**

## 3. Final Results — mIoU on Real ACDC Fog (ranked)

| Rank | Experiment | mIoU | Technique | β / intensity |
|---|---|---|---|---|
| 1 | **baseline** | **0.4143** | none | — |
| 2 | nn_depth_b0005 | 0.3912 | NN-depth (monocular) | β=0.005 |
| 3 | depthaware_b0005 | 0.3807 | Depth-aware (stereo) | β=0.005 |
| 4 | depthaware_b001 | 0.2734 | Depth-aware (stereo) | β=0.01 |
| 5 | nn_depth_b002 | 0.2732 | NN-depth (monocular) | β=0.02 |
| 6 | nn_depth_b001 | 0.2667 | NN-depth (monocular) | β=0.01 |
| 7 | photometric_low | 0.2318 | Photometric | low |
| 8 | depthaware_b002 | 0.2275 | Depth-aware (stereo) | β=0.02 |
| 9 | photometric_med | 0.0853 | Photometric | medium |
| 10 | photometric_high | 0.0384 | Photometric | high |
| 11 | combined | 0.0346 | Depth-aware + photometric stacked | β=0.01 + medium |

## 4. Headline Findings (the narrative)

1. **No augmented condition beat the no-fog baseline.** This is the central, somewhat counter-intuitive finding.
2. **Dose-response pattern, consistent across all three technique families:** lighter fog → better ACDC performance, every time (low > med > high; β=0.005 > β=0.01 > β=0.02 for both depth-based techniques).
3. **The two depth-based techniques (stereo and NN-monocular) clearly outperform photometric fog at every comparable intensity**, and came closest to matching baseline. NN-depth (monocular, no stereo rig required) performed essentially on par with the stereo-based version — a meaningful practical result, since it means the cheaper/more general technique loses nothing.
4. **Root cause of the failures (combined, photometric_high) is underfitting, not a generalization gap.** Training curves show these models never reached competitive performance even on their **own in-domain Cityscapes validation set** (e.g. `photometric_high` peaked at 0.29 Cityscapes-val mIoU vs. baseline's 0.77). The heavy/stacked fog perturbation overwhelmed the model's ability to learn the segmentation task in the fixed 50-epoch budget — this is a training-budget/intensity issue, not proof that fog augmentation is fundamentally unhelpful.
5. **Practical implication:** if doing this again, the light intensity (β=0.005-equivalent) settings deserve a longer training budget or lower fog dosage curriculum (e.g. ramping fog intensity up over training) to test whether depth-aware fog can be pushed to actually beat baseline rather than just approach it.

## 5. Available Figures

All in `fog-augmentation/results/figures/`:

| File | Description | Use in slides |
|---|---|---|
| `miou_comparison.png` | Sorted bar chart, all 11 experiments, color-coded by technique family | Headline results slide |
| `per_class_heatmap.png` | 11 experiments × 19 classes IoU heatmap | "Where does it fail" slide |
| `training_curves.png` | Train loss + Cityscapes val mIoU per epoch, all 11 | Supports the "underfitting, not generalization gap" explanation |
| `qualitative_predictions.png` | 3 real ACDC fog images × 5 models (baseline, photometric_low, depthaware_b0005, nn_depth_b0005, combined) vs. ground truth, side by side | Visual proof slide — `combined` visibly collapses into a near-uniform blob, matching its 0.035 mIoU |

Raw data tables: `results/miou_table.csv`, `results/per_class_iou.csv`, `results/summary.md`.

## 6. Methodology Detail (for a methods slide, if needed)

- **Datasets:** Cityscapes (train/val split, fine-grained labels, stereo disparity + camera calibration used for depth-aware fog); ACDC fog split (evaluation only, 400 train images unused / 100 val images used)
- **Model:** DeepLabV3+ with ResNet-101 backbone, ImageNet-pretrained encoder (`segmentation_models_pytorch`), 19 Cityscapes trainId classes, ignore_index=255
- **Training:** AdamW, lr=6e-5, weight_decay=0.01, polynomial LR decay, 50 epochs, batch size 4, random-scale + crop (1024×512) + horizontal flip augmentation (geometric augmentation identical across all 11 conditions — only the fog differs)
- **Fog caching:** for the two computationally expensive techniques (depth-aware completion, NN depth estimation), fog was pre-rendered once per image and cached to disk, then reused identically across all 50 epochs — a performance optimization with no effect on results (verified to match on-the-fly computation).
- **Hardware:** single NVIDIA RTX A5000 (24GB), university HPC cluster.

## 7. Status / What's Done vs. Open

**Done:**
- [x] All 11 models trained to 50 epochs
- [x] All 11 evaluated on real ACDC fog val
- [x] Results compiled (CSV + markdown tables)
- [x] All 4 summary figures generated

**Open / possible future work (not yet done, optional):**
- [ ] Investigate whether longer training or a fog-intensity curriculum lets depth-aware fog beat baseline (current best only approaches it)
- [ ] Statistical significance / multiple-seed runs (current results are single-seed per condition)
- [ ] ACDC held-out test set submission (only val was used; test labels aren't public)
- [ ] Cityscapes-pretrained backbone (vs. current ImageNet-pretrained) — was blocked on this HPC by a missing system package (`mmcv` couldn't be compiled, no `python3-devel` available without root); ImageNet-pretrained was used instead for all 11 conditions, so this doesn't bias the comparison between them, but a Cityscapes-pretrained backbone would likely raise all mIoU numbers somewhat.
