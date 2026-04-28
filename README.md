# Onco-GPT-X

**An End-to-End AI Pipeline for Melanoma Analysis from Dermoscopy Images**

Tisha Patel · Research Apprenticeship · Prof. Divya Chaudhary  
Khoury College of Computer Sciences, Northeastern University

---

## Overview

Onco-GPT-X is a five-module deep learning pipeline that integrates preprocessing, segmentation, classification, explainability, and generative AI into a unified clinical decision-support system for melanoma detection. Unlike existing approaches that address individual components in isolation, Onco-GPT-X chains all five stages end-to-end — from a raw dermoscopy image to an integrated clinical report with diagnosis, lesion boundary, visual explanation, and counterfactual comparisons.

## Pipeline Architecture

```
Raw Dermoscopy Image
        │
        ▼
┌──────────────────────────┐
│  Module 1: Preprocessing │  DullRazor hair removal → CLAHE contrast enhancement
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Module 2: Segmentation  │  PVTv2-B2 + UNet decoder → lesion mask
│  (ISIC 2018 Task 1)      │  Dice: 0.9140 | IoU: 0.84
└──────────┬───────────────┘
           │  masked/cropped lesion
           ▼
┌──────────────────────────┐
│  Module 3: Classification│  EfficientNet-B3 + Focal Loss, 5-Fold CV, TTA
│  (SIIM-ISIC 2020)        │  Best mean fold AUC: 0.9170 (B3+SE, merged data)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Module 4: Explainability│  Grad-CAM on all classification models
│                          │  60 heatmap visualizations (20 images × 3 models)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Module 5: Counterfactual│  Conditional DDPM + SDEdit on HAM10000
│  Generation              │  "What would this lesion look like if benign?"
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Integrated Clinical     │  Diagnosis + mask + heatmap + counterfactuals
│  Report                  │  in a single composite output
└──────────────────────────┘
```

## Datasets

| Dataset | Module | Size | Task |
|---|---|---|---|
| SIIM-ISIC 2020 | Classification | ~33,126 images, 1.76% melanoma | Binary melanoma detection |
| ISIC 2018 Task 1 | Segmentation | 2,594 image-mask pairs | Lesion boundary segmentation |
| HAM10000 | Generative | 10,015 images, 7 classes | Counterfactual generation |

SIIM-ISIC 2020 images sourced via Kaggle mirror (`bitthal/resize-jpg-siimisic-melanoma-classification`). All classification splits are stratified by `patient_id` to prevent data leakage.

## Key Results

### Classification — Architecture Comparison (5-Fold CV)

| Family | Model | Mean Fold AUC | OOF AUC | Dataset |
|---|---|---|---|---|
| CNN | EfficientNet-B3 + Focal Loss (384px) | 0.8995 | 0.8963 | ISIC 2020 |
| CNN | **EfficientNet-B3 + SE (merged)** | **0.9170** | **0.9101** | ISIC 2019+2020 |
| Transformer | SwinV2-Small (256px) | 0.8551 | 0.7886 | ISIC 2020 |
| Transformer | **SwinV2-Small staged (merged)** | **0.9134** | **0.9098** | ISIC 2019+2020 |
| Transformer | **ViT-Small staged (merged)** | **0.9154** | **0.9117** | ISIC 2019+2020 |
| Hybrid | EfficientNet-B3 + CBAM | 0.9035* | — | ISIC 2020 |

*Single-split validation AUC.

**Finding:** Transformers underperform CNNs on ISIC 2020 alone but match/exceed them when paired with merged datasets and staged training — a genuinely interesting result showing data scaling matters more than architecture choice.

### Segmentation

| Model | Val Dice | Val IoU |
|---|---|---|
| U-Net + EfficientNet-B3 encoder | 0.9104 | ~0.83 |
| **PVTv2-B2 + UNet decoder** | **0.9140** | **~0.84** |

### Counterfactual Generation

Conditional DDPM trained for 200+ epochs on HAM10000 (128×128). Training loss converged from 0.06 → 0.007. SDEdit generates class-conditional counterfactuals at noise strength 0.4 (t=400/1000), preserving original lesion structure while transforming appearance.

### Held-Out Test Set

Best single-fold test AUC: **0.9068** (SwinV2-Small staged, fold 1) with 53.16% sensitivity at 0.27 operating threshold.

## Ablation Study — Class Balancing (Module 3, In Progress)

Benchmarking against a target OOF AUC of **0.9295** (knowledge distillation paper). Baseline: OOF AUC 0.9220 (EfficientNet-B3 + Focal Loss, 384px, 5-fold CV + TTA).

| Experiment | Technique | Status |
|---|---|---|
| Exp 1 | Two-stage training (warm-up + fine-tune) | Complete — best val AUC 0.8812, underperformed |
| Exp 2 | OHEM oversampling | Running |
| Exp 3 | SMOTE in latent embedding space | Running |
| Exp 4 | Conditional GAN augmentation | Planned |
| Exp 5 | DDPM-based augmentation (Module 5 infra) | Planned |
| Exp 6 | Best combination | Planned |

## Project Structure

```
/scratch/patel.tis/OncoX/
├── src/
│   ├── data/                  # Shared preprocessing & dataset loaders
│   ├── classification/        # Module 3: Multi-model classification
│   ├── segmentation/          # Module 2: PVTv2-UNet lesion segmentation
│   ├── xai/                   # Module 4: Grad-CAM explainability
│   ├── diffusion/             # Module 5: Conditional DDPM + SDEdit
│   └── integration/           # End-to-end pipeline & report generation
├── jobs/                      # SLURM submission scripts
├── data/
│   ├── raw/                   # Downloaded datasets (SIIM-ISIC, HAM10000, ISIC 2018)
│   ├── metadata/              # Split CSVs (train/val/test by patient_id)
│   └── processed/
├── models/checkpoints/        # Saved model weights (.pth)
└── results/                   # Output metrics, figures, reports
```

## Technical Details

### Training Infrastructure
- **Cluster:** Northeastern Explorer HPC (`login.explorer.northeastern.edu`)
- **Scheduler:** SLURM (gpu partition, wall time < 8 hours)
- **GPUs:** V100, H200, T4
- **Environment:** Python 3.9, PyTorch 2.8.0, timm 1.0.24

### Core Libraries
PyTorch, timm, albumentations, scikit-learn, OpenCV, HuggingFace diffusers (for pretrained DDPM comparison)

### Key Design Decisions

**Segmentation before classification.** The pipeline segments the lesion first, then feeds the masked/cropped region to the classifier. This enforces clinical relevance — the model classifies the lesion, not background skin or artifacts.

**Patient-level splitting.** All train/val/test splits are stratified by `patient_id`. SIIM-ISIC 2020 contains multiple images per patient; random splitting would leak patient-specific features across splits and inflate metrics.

**Focal Loss over BCE.** Standard binary cross-entropy with `pos_weight` required careful tuning (56.0 → 8.0) to avoid NaN instability. Focal Loss naturally down-weights easy negatives without manual weight tuning.

**EfficientNet-B3 over B5.** B5 was abandoned due to CUDA multiprocessing errors and T4 numerical precision issues. B3 was retained as the stable, performant baseline.

**Merged ISIC 2019+2020 datasets.** Merging external data from ISIC 2019 boosted all architectures by 2–3 AUC points and was the single most impactful change. Transformers benefited disproportionately, closing the gap with CNNs.

## Prior Coursework Foundation

Segmentation architectures (DS²Net, PVTv2+UNet, MALUNet, TransFuse) were benchmarked during coursework on ISIC 2018, revealing that all individual models plateau near ~85% mIoU with ~2,300 training samples — a data-limited bottleneck. Ensembles + TTA broke past that ceiling. This insight informed the pipeline's data-scaling strategy.

## Future Work

- Complete ablation experiments 4–6 and select the best class-balancing approach
- Treatment response prediction module using Anti-PD-1 immunotherapy data (TCIA) with longitudinal CT delta-radiomics
- Clinical user study with dermatologists on the integrated report format
- FID validation for DDPM-generated images

## Citation

If referencing this work, please cite:

```
Patel, T. (2026). Onco-GPT-X: An End-to-End AI Pipeline for Melanoma Analysis
from Dermoscopy Images. Research Apprenticeship, Northeastern University.
Advisor: Prof. Divya Chaudhary.
```

## Acknowledgments

Prof. Divya Chaudhary (advisor), Khoury College of Computer Sciences, Northeastern University Research Computing (Explorer HPC).
