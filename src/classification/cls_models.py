"""

Onco-GPT-X Module 1: Classification Models

- EfficientNet-B3 + CBAM

- SwinV2-Tiny

- PVTv2-B2

- Meta-Ensemble

"""



import torch

import torch.nn as nn

import timm

from cbam import CBAM





# ──────────────────────────────────────────────

# Model 1: EfficientNet-B3 + CBAM

# ──────────────────────────────────────────────

class EfficientNetCBAM(nn.Module):

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        self.backbone = timm.create_model(

            'efficientnet_b3', pretrained=pretrained, num_classes=0

        )

        feat_dim = self.backbone.num_features  # 1536 for B3

        self.cbam = CBAM(feat_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(feat_dim, num_classes),

        )



    def forward_features(self, x):

        x = self.backbone.forward_features(x)  # (B, C, H, W)

        x = self.cbam(x)

        return x



    def forward(self, x):

        x = self.forward_features(x)

        x = self.pool(x).flatten(1)

        return self.head(x)





# ──────────────────────────────────────────────

# Model 2: SwinV2-Tiny

# ──────────────────────────────────────────────

class SwinV2Classifier(nn.Module):

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        self.backbone = timm.create_model(

            'swinv2_tiny_window8_256', pretrained=pretrained, num_classes=0

        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(feat_dim, num_classes),

        )



    def forward(self, x):

        x = self.backbone(x)

        return self.head(x)





# ──────────────────────────────────────────────

# Model 3: PVTv2-B2

# ──────────────────────────────────────────────

class PVTv2Classifier(nn.Module):

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        self.backbone = timm.create_model(

            'pvt_v2_b2', pretrained=pretrained, num_classes=0

        )

        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(feat_dim, num_classes),

        )



    def forward(self, x):

        x = self.backbone(x)

        return self.head(x)





# ──────────────────────────────────────────────

# Meta-Ensemble: Stacks predictions from all 3

# ──────────────────────────────────────────────

class MetaEnsemble(nn.Module):

    """

    Takes sigmoid outputs from 3 models and learns

    the optimal combination via a small meta-classifier.

    """

    def __init__(self, n_models=3, num_classes=1):

        super().__init__()

        self.meta = nn.Sequential(

            nn.Linear(n_models, 16),

            nn.ReLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(16, num_classes),

        )



    def forward(self, preds_list):

        """preds_list: list of (B, 1) sigmoid outputs from each model"""

        stacked = torch.cat(preds_list, dim=1)  # (B, n_models)

        return self.meta(stacked)





def build_model(name, num_classes=1, pretrained=True):

    """Factory function to build any model by name."""

    models = {

        'efficientnet_cbam': EfficientNetCBAM,

        'swinv2':            SwinV2Classifier,

        'pvtv2':             PVTv2Classifier,

    }

    if name not in models:

        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")

    return models[name](num_classes=num_classes, pretrained=pretrained)
