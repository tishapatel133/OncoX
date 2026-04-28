# Onco-GPT-X

**An End-to-End AI Pipeline for Melanoma Analysis from Dermoscopy Images**

Tisha Patel · Prof. Divya Chaudhary  
Khoury College of Computer Sciences, Northeastern University

---

## Overview

Onco-GPT-X is a five-module deep learning pipeline for melanoma detection that integrates preprocessing, segmentation, classification, explainability, and counterfactual generation into a unified clinical decision-support system. Given a raw dermoscopy image, it produces a diagnosis, lesion boundary, visual explanation, and counterfactual comparisons in a single report.

## Datasets

| Dataset | Size | Purpose |
|---|---|---|
| SIIM-ISIC 2020 | ~33K images, 1.76% melanoma | Binary classification |
| ISIC 2018 Task 1 | 2,594 image-mask pairs | Segmentation |
| HAM10000 | 10,015 images, 7 classes | Counterfactual generation |

## Environment

- **Cluster:** Northeastern Explorer HPC (SLURM, V100/H200 GPUs)
- **Stack:** Python 3.9, PyTorch 2.8.0, timm 1.0.24, albumentations, scikit-learn, OpenCV

## Acknowledgments

Prof. Divya Chaudhary (advisor), Khoury College of Computer Sciences, Northeastern University Research Computing.
