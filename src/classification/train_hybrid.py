"""

Hybrid CNN + Transformer for melanoma classification (scaled up).

Architecture:

  EfficientNet-B3 backbone (pretrained)

  -> SE channel attention

  -> Token projection (1536 -> 768)

  -> 12 custom transformer blocks (768d, 12 heads, MLP x4)

  -> Dropout 0.4

  -> Global avg pool -> Classifier

Approximate size: ~85M parameters.

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

from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,

                              matthews_corrcoef, confusion_matrix, accuracy_score)

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm



SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



ROOT        = "/scratch/patel.tis/OncoX"

IMG_DIR     = os.path.join(ROOT, "data/raw/isic2020_clean")

META_CSV    = os.path.join(ROOT, "data/metadata/train.csv")

RESULTS_DIR = os.path.join(ROOT, "results/classification")

CKPT_DIR    = os.path.join(ROOT, "models/checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)



# ───────────────────────── MODULES ─────────────────────────



class SEBlock(nn.Module):

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



class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.0):

        super().__init__()

        self.num_heads = num_heads

        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(dropout)



    def forward(self, x):

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj_drop(self.proj(out))



class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.4):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout=dropout)

        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(

            nn.Linear(dim, int(dim * mlp_ratio)),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(int(dim * mlp_ratio), dim),

            nn.Dropout(dropout),

        )

    def forward(self, x):

        x = x + self.drop1(self.attn(self.norm1(x)))

        x = x + self.mlp(self.norm2(x))

        return x



class HybridCNNTransformer(nn.Module):

    def __init__(self, transformer_dim=768, num_blocks=12, num_heads=12, dropout=0.4):

        super().__init__()

        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0, global_pool="")

        feat_dim = self.backbone.num_features  # 1536

        self.se = SEBlock(feat_dim)

        self.proj_in = nn.Conv2d(feat_dim, transformer_dim, kernel_size=1)

        self.norm_in = nn.LayerNorm(transformer_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 144, transformer_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.ModuleList([

            TransformerBlock(transformer_dim, num_heads=num_heads, mlp_ratio=4.0, dropout=dropout)

            for _ in range(num_blocks)

        ])

        self.norm_out = nn.LayerNorm(transformer_dim)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(transformer_dim, 1)



    def forward(self, x):

        feat = self.backbone.forward_features(x)

        feat = self.se(feat)

        feat = self.proj_in(feat)

        B, C, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)

        tokens = self.norm_in(tokens)

        if tokens.shape[1] != self.pos_embed.shape[1]:

            pos = F.interpolate(

                self.pos_embed.transpose(1, 2).unsqueeze(0),

                size=tokens.shape[1], mode="linear"

            ).squeeze(0).transpose(1, 2)

        else:

            pos = self.pos_embed

        tokens = tokens + pos

        for block in self.transformer:

            tokens = block(tokens)

        tokens = self.norm_out(tokens)

        feat_pooled = tokens.mean(dim=1)

        return self.classifier(self.dropout(feat_pooled))



# ───────────────────────── DATA + TRAINING ─────────────────────────



def get_transforms(img_size, train=True):

    if train:

        return A.Compose([

            A.Resize(img_size, img_size),

            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),

            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            A.CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16, p=0.3),

            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

            ToTensorV2()

        ])

    return A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

        ToTensorV2()

    ])



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



def find_best_threshold(labels, probs):

    best_f1, best_thresh = 0, 0.5

    for t in np.arange(0.1, 0.9, 0.01):

        preds = (probs >= t).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:

            best_f1 = f1; best_thresh = t

    return best_thresh



def compute_metrics(labels, probs, threshold):

    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()

    return {

        "AUC": round(roc_auc_score(labels, probs), 4),

        "F1": round(f1_score(labels, preds, zero_division=0), 4),

        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),

        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),

        "Precision": round(precision_score(labels, preds, zero_division=0), 4),

        "MCC": round(matthews_corrcoef(labels, preds), 4),

        "Accuracy": round(accuracy_score(labels, preds), 4),

        "Threshold": round(threshold, 2),

    }



def main():

    IMG_SIZE = 384; EPOCHS = 20; BATCH_SIZE = 8; LR = 1e-4; PATIENCE = 7; WORKERS = 4



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    train_idx, val_idx = split[0]

    train_df = df.iloc[train_idx]

    val_df = df.iloc[val_idx]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Melanoma: {train_df['target'].sum()}")



    train_ds = MelanomaDataset(train_df, IMG_DIR, get_transforms(IMG_SIZE, train=True))

    val_ds   = MelanomaDataset(val_df,   IMG_DIR, get_transforms(IMG_SIZE, train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)

    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)



    model = HybridCNNTransformer(transformer_dim=768, num_blocks=12, num_heads=12, dropout=0.4).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"Total parameters:     {total_params:.2f}M")

    print(f"Trainable parameters: {trainable_params:.2f}M")



    scaler = torch.cuda.amp.GradScaler()

    criterion = FocalLoss(0.25, 2.0)

    backbone_params = list(model.backbone.parameters())

    new_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    optimizer = AdamW([

        {"params": backbone_params, "lr": LR / 10},

        {"params": new_params,      "lr": LR},

    ], weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, max_lr=[LR/10, LR], steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1)



    val_labels = val_df["target"].values.astype(int)

    best_auc = 0.0

    best_probs = None

    patience_counter = 0

    ckpt_path = os.path.join(CKPT_DIR, "hybrid_cnn_transformer_best.pt")

    log_rows = []



    for epoch in range(1, EPOCHS + 1):

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

        avg_loss = total_loss / len(train_loader)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  val_auc={auc:.4f}")

        log_rows.append({"epoch": epoch, "loss": avg_loss, "val_auc": auc})



        if auc > best_auc:

            best_auc = auc; best_probs = probs; patience_counter = 0

            torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)

            print(f"  New best: {best_auc:.4f}")

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(f"  Early stopping at epoch {epoch}"); break



    pd.DataFrame(log_rows).to_csv(os.path.join(RESULTS_DIR, "hybrid_log.csv"), index=False)

    if best_probs is not None:

        thresh = find_best_threshold(val_labels, best_probs)

        metrics = compute_metrics(val_labels, best_probs, thresh)

        print(f"\n========== HYBRID CNN+TRANSFORMER RESULTS ==========")

        print(f"  Model size:    {total_params:.2f}M parameters")

        print(f"  Best Val AUC:  {metrics['AUC']}")

        print(f"  F1:            {metrics['F1']}")

        print(f"  Sensitivity:   {metrics['Sensitivity']}")

        print(f"  Specificity:   {metrics['Specificity']}")

        print(f"  Precision:     {metrics['Precision']}")

        print(f"  MCC:           {metrics['MCC']}")

        print(f"  Accuracy:      {metrics['Accuracy']}")

        print(f"  Threshold:     {metrics['Threshold']}")

        print(f"\n  Baseline B3 (no-attn):  0.8944")

        print(f"  Best attn (B3+SE):      0.8997")

        print(f"  Target to beat:         0.9220")

        pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "hybrid_metrics.csv"), index=False)



if __name__ == "__main__":

    main()
