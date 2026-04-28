"""

Final 5-fold CV on EfficientNet-B3 + SE attention.

This is the winner from the attention comparison (fold 0 AUC 0.8997).

Full 5-fold CV gives the proper OOF AUC for paper reporting.

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



class EffNetB3_SE(nn.Module):

    def __init__(self, dropout=0.3):

        super().__init__()

        self.backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0, global_pool="")

        feat_dim = self.backbone.num_features

        self.se = SEBlock(feat_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x):

        feat = self.backbone.forward_features(x)

        feat = self.se(feat)

        feat = self.pool(feat).flatten(1)

        return self.classifier(self.dropout(feat))



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



def train_fold(model, train_loader, val_loader, val_labels, device, epochs=20, lr=2e-4, patience=7):

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

            print(f"    New best: {best_auc:.4f}")

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(f"  Early stopping at epoch {epoch}"); break

    return best_auc, best_probs



def main():

    IMG_SIZE = 384; EPOCHS = 20; BATCH_SIZE = 16; LR = 2e-4; PATIENCE = 7; WORKERS = 4



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    print(f"Total: {len(df)}  |  Melanoma: {df['target'].sum()}")



    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))



    fold_aucs = []

    oof_preds = np.zeros(len(df))

    oof_labels = df["target"].values



    for fold, (train_idx, val_idx) in enumerate(split):

        print(f"\n{'='*60}")

        print(f"Fold {fold+1}/5")

        print(f"{'='*60}")

        train_df = df.iloc[train_idx]

        val_df = df.iloc[val_idx]

        val_labels = val_df["target"].values.astype(int)



        train_ds = MelanomaDataset(train_df, IMG_DIR, get_transforms(IMG_SIZE, train=True))

        val_ds = MelanomaDataset(val_df, IMG_DIR, get_transforms(IMG_SIZE, train=False))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)

        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)



        model = EffNetB3_SE().to(device)

        best_auc, best_probs = train_fold(model, train_loader, val_loader, val_labels, device, epochs=EPOCHS, lr=LR, patience=PATIENCE)

        fold_aucs.append(best_auc)

        oof_preds[val_idx] = best_probs

        torch.save({"model_state": model.state_dict(), "auc": best_auc}, os.path.join(CKPT_DIR, f"b3_se_5fold_fold{fold+1}.pt"))

        del model; torch.cuda.empty_cache()

        print(f"  Fold {fold+1} best AUC: {best_auc:.4f}")



    oof_auc = roc_auc_score(oof_labels, oof_preds)

    mean_auc = np.mean(fold_aucs)

    std_auc = np.std(fold_aucs)



    thresh = find_best_threshold(oof_labels.astype(int), oof_preds)

    metrics = compute_metrics(oof_labels.astype(int), oof_preds, thresh)



    print(f"\n{'='*60}")

    print(f"FINAL 5-FOLD RESULTS — EfficientNet-B3 + SE")

    print(f"{'='*60}")

    print(f"  Fold AUCs: {[round(a,4) for a in fold_aucs]}")

    print(f"  Mean fold AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    print(f"  OOF AUC: {oof_auc:.4f}")

    print(f"  F1: {metrics['F1']}")

    print(f"  Sensitivity: {metrics['Sensitivity']}")

    print(f"  Specificity: {metrics['Specificity']}")

    print(f"  Precision: {metrics['Precision']}")

    print(f"  MCC: {metrics['MCC']}")

    print(f"  Accuracy: {metrics['Accuracy']}")

    print(f"  Threshold: {metrics['Threshold']}")



    summary = {

        "model": "EfficientNet-B3 + SE (5-fold)",

        "fold_aucs": str(fold_aucs),

        "mean_auc": round(mean_auc, 4),

        "std_auc": round(std_auc, 4),

        "oof_auc": round(oof_auc, 4),

        **metrics

    }

    pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, "b3_se_5fold_summary.csv"), index=False)



if __name__ == "__main__":

    main()
