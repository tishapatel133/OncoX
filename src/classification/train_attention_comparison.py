"""

Attention mechanism comparison on melanoma classification.

6 experiments: Baseline, SE-Net, ECA-Net, MHSA, Multi-scale, MTA-ViT

Phase 1: fold 0 only for fast comparison

"""



import os, random, math

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, accuracy_score

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm

import argparse



SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



ROOT        = "/scratch/patel.tis/OncoX"

IMG_DIR     = os.path.join(ROOT, "data/raw/isic2020_clean")

META_CSV    = os.path.join(ROOT, "data/metadata/train.csv")

RESULTS_DIR = os.path.join(ROOT, "results/classification")

CKPT_DIR    = os.path.join(ROOT, "models/checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)



# ───────────────────────── ATTENTION MODULES ─────────────────────────



class SEBlock(nn.Module):

    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, channels // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channels // reduction, channels, bias=False),

            nn.Sigmoid()

        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y



class ECABlock(nn.Module):

    """Efficient Channel Attention — no FC, just 1D conv."""

    def __init__(self, channels, gamma=2, b=1):

        super().__init__()

        k = int(abs((math.log(channels, 2) + b) / gamma))

        k = k if k % 2 else k + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = F.adaptive_avg_pool2d(x, 1)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        return x * self.sigmoid(y)



class MHSABlock(nn.Module):

    """Multi-Head Self-Attention on spatial feature map."""

    def __init__(self, channels, num_heads=8):

        super().__init__()

        self.num_heads = num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, 1)

        self.proj = nn.Conv2d(channels, channels, 1)

        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):

        b, c, h, w = x.shape

        qkv = self.qkv(x).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-2, -1) @ k) * self.scale

        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)

        return x + self.proj(out)



class MultiScaleAttention(nn.Module):

    """Attention at 3 scales in parallel, fused."""

    def __init__(self, channels):

        super().__init__()

        self.branch1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        self.branch2 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)

        self.branch3 = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)

        self.fuse = nn.Conv2d(channels * 3, channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b1 = self.branch1(x); b2 = self.branch2(x); b3 = self.branch3(x)

        attn = self.sigmoid(self.fuse(torch.cat([b1, b2, b3], dim=1)))

        return x * attn



# MTA — adapted from Meta's paper for vision (bidirectional, no causal mask)

class MTAAttention(nn.Module):
    """Multi-Token Attention adapted for vision (bidirectional).
    Simplified version of Golovneva et al. 2025 for image tokens.
    """
    def __init__(self, dim, num_heads=6, cq=3, ck=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        # key-query conv: applied on attention logits, preserves head count
        self.kq_conv = nn.Conv2d(num_heads, num_heads, kernel_size=(cq, ck),
                                  padding=(cq // 2, ck // 2), groups=num_heads)
        # head mixing conv: 1x1 across heads dimension (no groups for full mixing)
        self.head_conv = nn.Conv2d(num_heads, num_heads, kernel_size=1)
        self.norm = nn.GroupNorm(num_heads, num_heads)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # attn_logits shape: [B, num_heads, N, N]
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        # pre-softmax key-query convolution (groups=num_heads preserves shape)
        attn_logits = self.kq_conv(attn_logits)
        attn = attn_logits.softmax(dim=-1)
        # post-softmax head mixing
        attn = self.head_conv(attn)
        attn = self.norm(attn)
        # output: [B, num_heads, N, head_dim] -> [B, N, num_heads, head_dim] -> [B, N, C]
        out = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        return self.proj(out)

# ───────────────────────── MODEL WRAPPERS ─────────────────────────



class EffNetWithAttention(nn.Module):

    """EfficientNet-B3 with an attention module inserted before classifier."""

    def __init__(self, attention_type="none", dropout=0.3):

        super().__init__()

        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0, global_pool="")

        feat_dim = self.backbone.num_features  # 1536



        if attention_type == "se":

            self.attention = SEBlock(feat_dim)

        elif attention_type == "eca":

            self.attention = ECABlock(feat_dim)

        elif attention_type == "mhsa":

            self.attention = MHSABlock(feat_dim)

        elif attention_type == "multiscale":

            self.attention = MultiScaleAttention(feat_dim)

        else:

            self.attention = nn.Identity()



        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(feat_dim, 1)



    def forward(self, x):

        feat = self.backbone.forward_features(x)

        feat = self.attention(feat)

        feat = self.pool(feat).flatten(1)

        return self.classifier(self.dropout(feat))



class MTAViT(nn.Module):

    """ViT-Small where last N attention layers are replaced with MTA."""

    def __init__(self, replace_last_n=4, dropout=0.3):

        super().__init__()

        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)

        dim = self.backbone.embed_dim

        # replace last N blocks' attention with MTA

        n_blocks = len(self.backbone.blocks)

        for i in range(n_blocks - replace_last_n, n_blocks):

            self.backbone.blocks[i].attn = MTAAttention(dim, num_heads=6)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(dim, 1)

    def forward(self, x):

        feat = self.backbone.forward_features(x)

        # grab CLS token

        if feat.dim() == 3:

            feat = feat[:, 0]

        return self.classifier(self.dropout(feat))



# ───────────────────────── DATA + TRAINING ─────────────────────────



def get_transforms(img_size, train=True):

    if train:

        return A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), A.CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16, p=0.3), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])

    return A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])



class MelanomaDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):

        self.df = df.reset_index(drop=True)

        self.img_dir = img_dir

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img = np.array(Image.open(os.path.join(self.img_dir, row["image_name"] + ".jpg")).convert("RGB"))

        label = float(row["target"])

        if self.transform:

            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.float32)



class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):

        super().__init__()

        self.alpha = alpha; self.gamma = gamma

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt = torch.exp(-bce)

        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()



def compute_metrics(labels, probs, threshold=0.5):

    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    tn, fp, fn, tp = cm

    return {

        "AUC": round(roc_auc_score(labels, probs), 4),

        "F1": round(f1_score(labels, preds, zero_division=0), 4),

        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),

        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),

        "Precision": round(precision_score(labels, preds, zero_division=0), 4),

        "MCC": round(matthews_corrcoef(labels, preds), 4),

        "Accuracy": round(accuracy_score(labels, preds), 4),

    }



def find_best_threshold(labels, probs):

    best_f1, best_thresh = 0, 0.5

    for t in np.arange(0.1, 0.9, 0.01):

        preds = (probs >= t).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:

            best_f1 = f1; best_thresh = t

    return best_thresh



def train_one(model, train_loader, val_loader, val_labels, device, epochs=15, lr=2e-4, patience=5):

    scaler = torch.cuda.amp.GradScaler()

    criterion = FocalLoss(0.25, 2.0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1)



    best_auc = 0.0

    best_probs = None

    patience_counter = 0



    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0

        for imgs, labels in train_loader:

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                logits = model(imgs)

                if logits.dim() > 1: logits = logits.squeeze(1)

                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer); scaler.update(); scheduler.step()

            total_loss += loss.item()



        # validate

        model.eval()

        probs = []

        with torch.no_grad():

            for imgs, _ in val_loader:

                imgs = imgs.to(device)

                with torch.cuda.amp.autocast():

                    logits = model(imgs)

                    if logits.dim() > 1: logits = logits.squeeze(1)

                p = torch.sigmoid(logits).cpu().numpy()

                probs.extend(np.nan_to_num(p, nan=0.5))

        probs = np.array(probs)

        auc = roc_auc_score(val_labels, probs)

        print(f"  Epoch {epoch:02d}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_auc={auc:.4f}")

        if auc > best_auc:

            best_auc = auc; best_probs = probs; patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(f"  Early stopping at epoch {epoch}")

                break

    return best_auc, best_probs



def run_experiment(name, model_builder, img_size, train_df, val_df, device, batch_size=16, epochs=15):

    print(f"\n{'='*60}\nExperiment: {name}\n{'='*60}")

    train_ds = MelanomaDataset(train_df, IMG_DIR, get_transforms(img_size, train=True))

    val_ds = MelanomaDataset(val_df, IMG_DIR, get_transforms(img_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    val_labels = val_df["target"].values.astype(int)



    try:

        model = model_builder().to(device)

        best_auc, best_probs = train_one(model, train_loader, val_loader, val_labels, device, epochs=epochs)

        thresh = find_best_threshold(val_labels, best_probs)

        metrics = compute_metrics(val_labels, best_probs, thresh)

        metrics["Experiment"] = name

        metrics["Threshold"] = round(thresh, 2)

        print(f"  Best AUC: {best_auc:.4f}  |  F1: {metrics['F1']}  |  Sens: {metrics['Sensitivity']}")

        del model; torch.cuda.empty_cache()

        return metrics

    except Exception as e:

        print(f"  ERROR: {e}")

        import traceback; traceback.print_exc()

        return None



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--experiments", nargs="+", default=["baseline", "se", "eca", "mhsa", "multiscale", "mta"])

    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    train_idx, val_idx = split[0]

    train_df = df.iloc[train_idx]

    val_df = df.iloc[val_idx]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Melanoma: {train_df['target'].sum()}")



    configs = {

        "baseline":   ("EfficientNet-B3 no-attn", lambda: EffNetWithAttention("none"),        384, 16),

        "se":         ("EfficientNet-B3 + SE",    lambda: EffNetWithAttention("se"),          384, 16),

        "eca":        ("EfficientNet-B3 + ECA",   lambda: EffNetWithAttention("eca"),         384, 16),

        "mhsa":       ("EfficientNet-B3 + MHSA",  lambda: EffNetWithAttention("mhsa"),        384, 16),

        "multiscale": ("EfficientNet-B3 + MultiScale", lambda: EffNetWithAttention("multiscale"), 384, 16),

        "mta":        ("ViT-Small + MTA (novel)", lambda: MTAViT(replace_last_n=4),           224, 32),

    }



    all_results = []

    for exp_key in args.experiments:

        if exp_key not in configs:

            print(f"Unknown experiment: {exp_key}"); continue

        name, builder, img_size, batch_size = configs[exp_key]

        result = run_experiment(name, builder, img_size, train_df, val_df, device, batch_size=batch_size)

        if result:

            all_results.append(result)



    if not all_results:

        print("No results."); return



    results_df = pd.DataFrame(all_results)

    out_path = os.path.join(RESULTS_DIR, "attention_comparison.csv")

    results_df.to_csv(out_path, index=False)



    print("\n========== ATTENTION COMPARISON RESULTS ==========")

    print(f"{'Experiment':<32} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} {'MCC':>6}")

    print("-" * 72)

    for r in sorted(all_results, key=lambda x: x["AUC"], reverse=True):

        print(f"{r['Experiment']:<32} {r['AUC']:>6} {r['F1']:>6} {r['Sensitivity']:>6} {r['Specificity']:>6} {r['MCC']:>6}")

    print(f"\nSaved to: {out_path}")



if __name__ == "__main__":

    main()
