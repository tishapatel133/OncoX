# Onco-GPT-X

## Overview

Onco-GPT-X is a melanoma analysis pipeline that integrates classification, explainability, and generative components for dermoscopic images. This study focuses on single-model classification under extreme class imbalance on the SIIM-ISIC 2020 dataset (~33K images, 1.76% melanoma prevalence), a substantially harder benchmark than the balanced multi-class tasks on which most published models report near-perfect accuracy. Fourteen backbones across convolutional, transformer, and hybrid families are evaluated against five class-balancing strategies, four attention mechanisms, and three masking configurations using patient-level five-fold stratified cross-validation. The strongest configuration, a staged SwinV2-Small pretrained on combined ISIC 2019+2020 data, achieves a peak per-fold validation AUC of 0.9169 and a held-out test AUC of 0.9068, exceeding every single-stage configuration evaluated.
