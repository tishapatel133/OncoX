"""

Experiment 1: Two-stage training for class balancing

Stage 1 — balanced 1:1 subset, BCE loss, 10 epochs (learn lesion features)

Stage 2 — full imbalanced dataset, Focal Loss, 15 epochs (calibrate probabilities)



Usage: python src/classification/train_two_stage.py

"""



import os

import random

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import roc_auc_score

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm

import argparse



# ── reproducibility ──────────────────────────────────────────────────────────

SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



# ── paths ─────────────────────────────────────────────────────────────────────

ROOT        = "/scratch/patel.tis/OncoX"

IMG_DIR     = os.path.join(ROOT, "data/raw/isic2020_clean")

META_CSV    = os.path.join(ROOT, "data/metadata/train.csv")

RESULTS_DIR = os.path.join(ROOT, "results/classification")

CKPT_DIR    = os.path.join(ROOT, "models/checkpoints")

LOG_DIR     = os.path.join(ROOT, "models/logs")

os.makedirs(RESULTS_DIR, exist_ok=True)

os.makedirs(CKPT_DIR, exist_ok=True)



# ── augmentations ─────────────────────────────────────────────────────────────

def get_transforms(img_size, train=True):

    if train:

        return A.Compose([

            A.Resize(img_size, img_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

            A.CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16, p=0.3),

            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

            ToTensorV2(),

        ])

    return A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

        ToTensorV2(),

    ])



# ── dataset ───────────────────────────────────────────────────────────────────

class MelanomaDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):

        self.df        = df.reset_index(drop=True)

        self.img_dir   = img_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        row   = self.df.iloc[idx]

        img   = np.array(Image.open(os.path.join(self.img_dir, row["image_name"] + ".jpg")).convert("RGB"))

        label = float(row["target"])

        if self.transform:

            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.float32)



# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma



    def forward(self, logits, targets):

        bce   = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt    = torch.exp(-bce)

        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()



# ── model ─────────────────────────────────────────────────────────────────────

def build_model(dropout=0.3):

    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)

    in_features = model.num_features

    model.classifier = nn.Sequential(

        nn.Dropout(dropout),

        nn.Linear(in_features, 1)

    )

    return model



# ── balanced sampler ──────────────────────────────────────────────────────────

def make_balanced_subset(df, ratio=1.0):

    """Return a df with 1:ratio melanoma:benign balance."""

    pos = df[df["target"] == 1]

    neg = df[df["target"] == 0].sample(n=int(len(pos) * ratio), random_state=SEED)

    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)

    print(f"  Balanced subset: {len(pos)} melanoma + {len(neg)} benign = {len(balanced)} total")

    return balanced



# ── training loop (one epoch) ─────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):

    model.train()

    total_loss = 0

    for imgs, labels in loader:

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            logits = model(imgs).squeeze(1)

            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# ── validation ────────────────────────────────────────────────────────────────

def validate(model, loader, device):

    model.eval()

    preds, labels_all = [], []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)

            with torch.cuda.amp.autocast():

                logits = model(imgs).squeeze(1)

            preds.extend(torch.sigmoid(logits).cpu().numpy())

            labels_all.extend(labels.numpy())

    return roc_auc_score(labels_all, preds)



# ── main ──────────────────────────────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--img_size",       type=int,   default=384)

    parser.add_argument("--stage1_epochs",  type=int,   default=10)

    parser.add_argument("--stage2_epochs",  type=int,   default=15)

    parser.add_argument("--batch_size",     type=int,   default=16)

    parser.add_argument("--lr",             type=float, default=2e-4)

    parser.add_argument("--focal_alpha",    type=float, default=0.25)

    parser.add_argument("--focal_gamma",    type=float, default=2.0)

    parser.add_argument("--patience",       type=int,   default=7)

    parser.add_argument("--workers",        type=int,   default=0)

    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    # load metadata

    df  = pd.read_csv(META_CSV)

    print(f"Total samples: {len(df)}  |  Melanoma: {df['target'].sum()}")



    # patient-level train/val split (80/20)

    sgkf  = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    train_idx, val_idx = split[0]   # use fold 0 for this single experiment

    train_df = df.iloc[train_idx]

    val_df   = df.iloc[val_idx]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")



    train_tfm = get_transforms(args.img_size, train=True)

    val_tfm   = get_transforms(args.img_size, train=False)

    val_ds    = MelanomaDataset(val_df, IMG_DIR, val_tfm)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,

                            num_workers=args.workers, pin_memory=True)



    model  = build_model().to(device)

    scaler = torch.cuda.amp.GradScaler()



    log_rows = []

    best_auc = 0.0

    ckpt_path = os.path.join(CKPT_DIR, "two_stage_best.pt")



    # ── STAGE 1: balanced 1:1 subset, BCE ─────────────────────────────────────

    print("\n========== STAGE 1: Balanced 1:1 — BCE Loss ==========")

    balanced_df   = make_balanced_subset(train_df, ratio=1.0)

    stage1_ds     = MelanomaDataset(balanced_df, IMG_DIR, train_tfm)

    stage1_loader = DataLoader(stage1_ds, batch_size=args.batch_size, shuffle=True,

                               num_workers=args.workers, pin_memory=True, drop_last=True)



    criterion_s1 = nn.BCEWithLogitsLoss()

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    scheduler    = OneCycleLR(optimizer, max_lr=args.lr,

                              steps_per_epoch=len(stage1_loader),

                              epochs=args.stage1_epochs, pct_start=0.1)



    for epoch in range(1, args.stage1_epochs + 1):

        loss = train_epoch(model, stage1_loader, optimizer, scheduler,

                           criterion_s1, device, scaler)

        auc  = validate(model, val_loader, device)

        print(f"  S1 Epoch {epoch:02d}/{args.stage1_epochs}  loss={loss:.4f}  val_auc={auc:.4f}")

        log_rows.append({"stage": 1, "epoch": epoch, "loss": loss, "val_auc": auc})

        if auc > best_auc:

            best_auc = auc

            torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)

            print(f"    ✅ New best: {best_auc:.4f}")



    # ── STAGE 2: full dataset, Focal Loss ─────────────────────────────────────

    print("\n========== STAGE 2: Full dataset — Focal Loss ==========")

    stage2_ds     = MelanomaDataset(train_df, IMG_DIR, train_tfm)

    stage2_loader = DataLoader(stage2_ds, batch_size=args.batch_size, shuffle=True,

                               num_workers=args.workers, pin_memory=True, drop_last=True)



    criterion_s2 = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    # lower LR for fine-tuning

    optimizer    = AdamW(model.parameters(), lr=args.lr / 5, weight_decay=1e-2)

    scheduler    = OneCycleLR(optimizer, max_lr=args.lr / 5,

                              steps_per_epoch=len(stage2_loader),

                              epochs=args.stage2_epochs, pct_start=0.1)



    patience_counter = 0

    for epoch in range(1, args.stage2_epochs + 1):

        loss = train_epoch(model, stage2_loader, optimizer, scheduler,

                           criterion_s2, device, scaler)

        auc  = validate(model, val_loader, device)

        print(f"  S2 Epoch {epoch:02d}/{args.stage2_epochs}  loss={loss:.4f}  val_auc={auc:.4f}")

        log_rows.append({"stage": 2, "epoch": epoch, "loss": loss, "val_auc": auc})

        if auc > best_auc:

            best_auc = auc

            patience_counter = 0

            torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)

            print(f"    ✅ New best: {best_auc:.4f}")

        else:

            patience_counter += 1

            if patience_counter >= args.patience:

                print(f"  Early stopping at epoch {epoch}")

                break



    # save log

    log_df = pd.DataFrame(log_rows)

    log_path = os.path.join(RESULTS_DIR, "two_stage_log.csv")

    log_df.to_csv(log_path, index=False)

    print(f"\nBest Val AUC: {best_auc:.4f}")

    print(f"Checkpoint:   {ckpt_path}")

    print(f"Log:          {log_path}")



if __name__ == "__main__":

    main()
