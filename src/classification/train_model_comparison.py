"""
Model comparison: Multiple architectures
Focal Loss + patient-level 5-fold CV
EfficientNet, SwinV2, PVTv2, ConvNeXt, MaxViT families
Supports --start_fold to resume from a specific fold
"""

import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
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

def get_img_size(model_name):
    if "maxvit" in model_name or model_name.endswith("_224"):
        return 224
    if any(x in model_name for x in ["window8_256", "window16_256"]):
        return 256
    return 384

def get_batch_size(model_name):
    if any(x in model_name for x in ["b7", "b8", "large", "base"]):
        return 8
    if any(x in model_name for x in ["b5", "b6", "small", "b4"]):
        return 12
    return 16

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
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

def build_model(model_name, dropout=0.3):
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=1, drop_rate=dropout)
        print(f"  Loaded pretrained weights for {model_name}")
    except RuntimeError:
        available = [m for m in timm.list_pretrained() if model_name.split(".")[0] in m]
        if available:
            fallback = available[0]
            print(f"  No weights for {model_name}, using fallback: {fallback}")
            model = timm.create_model(fallback, pretrained=True, num_classes=1, drop_rate=dropout)
        else:
            print(f"  No pretrained weights found for {model_name}, using random init")
            model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=dropout)
    return model

def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            if logits.dim() > 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                if logits.dim() > 1:
                    logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.5)
            preds.extend(probs)
            labels_all.extend(labels.numpy())
    return roc_auc_score(labels_all, preds)

def train_model(model_name, img_size, df, device, start_fold=0):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}  |  img_size: {img_size}  |  start_fold: {start_fold}")
    print(f"{'='*60}")

    EPOCHS = 25; LR = 2e-4; PATIENCE = 7; WORKERS = 4
    BATCH_SIZE = get_batch_size(model_name)
    print(f"  Batch size: {BATCH_SIZE}")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    fold_aucs = []
    oof_preds = np.zeros(len(df))
    oof_labels = df["target"].values

    for fold, (train_idx, val_idx) in enumerate(split):
        if fold < start_fold:
            print(f"\n--- Fold {fold+1}/5 --- SKIPPED (resuming)")
            ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fold{fold+1}.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                saved_auc = ckpt.get("auc", 0.0)
                fold_aucs.append(saved_auc)
                print(f"  Loaded existing fold {fold+1} AUC: {saved_auc:.4f}")
                # reload predictions for OOF
                val_df = df.iloc[val_idx]
                model = build_model(model_name).to(device)
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                val_ds = MelanomaDataset(val_df, IMG_DIR, get_transforms(img_size, train=False))
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)
                fold_preds = []
                with torch.no_grad():
                    for imgs, _ in val_loader:
                        imgs = imgs.to(device)
                        with torch.cuda.amp.autocast():
                            logits = model(imgs)
                            if logits.dim() > 1:
                                logits = logits.squeeze(1)
                        p = torch.sigmoid(logits).cpu().numpy()
                        fold_preds.extend(np.nan_to_num(p, nan=0.5))
                oof_preds[val_idx] = np.array(fold_preds)
                del model; torch.cuda.empty_cache()
            else:
                print(f"  WARNING: checkpoint not found for fold {fold+1}, using 0.5 preds")
                fold_aucs.append(0.0)
                oof_preds[val_idx] = 0.5
            continue

        print(f"\n--- Fold {fold+1}/5 ---")
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_ds = MelanomaDataset(train_df, IMG_DIR, get_transforms(img_size, train=True))
        val_ds   = MelanomaDataset(val_df,   IMG_DIR, get_transforms(img_size, train=False))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)

        model     = build_model(model_name).to(device)
        scaler    = torch.cuda.amp.GradScaler()
        criterion = FocalLoss(0.25, 2.0)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1)

        best_auc = 0.0
        best_preds = None
        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_fold{fold+1}.pt")
        patience_counter = 0
        log_rows = []

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
            auc  = validate(model, val_loader, device)
            print(f"  Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  val_auc={auc:.4f}")
            log_rows.append({"model": model_name, "fold": fold+1, "epoch": epoch, "loss": loss, "val_auc": auc})
            if auc > best_auc:
                best_auc = auc; patience_counter = 0
                torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)
                model.eval()
                fold_preds = []
                with torch.no_grad():
                    for imgs, _ in val_loader:
                        imgs = imgs.to(device)
                        with torch.cuda.amp.autocast():
                            logits = model(imgs)
                            if logits.dim() > 1:
                                logits = logits.squeeze(1)
                        p = torch.sigmoid(logits).cpu().numpy()
                        fold_preds.extend(np.nan_to_num(p, nan=0.5))
                best_preds = np.array(fold_preds)
                print(f"  New best: {best_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}"); break

        if best_preds is None:
            best_preds = np.full(len(val_idx), 0.5)

        fold_aucs.append(best_auc)
        oof_preds[val_idx] = best_preds
        pd.DataFrame(log_rows).to_csv(os.path.join(RESULTS_DIR, f"{model_name}_fold{fold+1}_log.csv"), index=False)
        print(f"  Fold {fold+1} best AUC: {best_auc:.4f}")
        del model; torch.cuda.empty_cache()

    oof_auc = roc_auc_score(oof_labels, oof_preds)
    mean_auc = np.mean(fold_aucs)
    print(f"\n{model_name} — Fold AUCs: {[round(a,4) for a in fold_aucs]}")
    print(f"{model_name} — Mean fold AUC: {mean_auc:.4f}")
    print(f"{model_name} — OOF AUC:       {oof_auc:.4f}")

    pd.DataFrame([{"model": model_name, "fold_aucs": str(fold_aucs), "mean_auc": mean_auc, "oof_auc": oof_auc}]).to_csv(
        os.path.join(RESULTS_DIR, f"{model_name}_summary.csv"), index=False)
    return oof_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["efficientnet_b4", "efficientnet_b5", "efficientnet_b6"])
    parser.add_argument("--start_fold", type=int, default=0, help="Resume from this fold index (0-based)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(META_CSV)
    print(f"Total: {len(df)}  |  Melanoma: {df['target'].sum()}")

    models_to_train = [(m, get_img_size(m)) for m in args.models]
    print(f"Models to train: {[m for m,_ in models_to_train]}")
    print(f"Starting from fold: {args.start_fold + 1}")

    results = {}
    for model_name, img_size in models_to_train:
        try:
            oof_auc = train_model(model_name, img_size, df, device, start_fold=args.start_fold)
            results[model_name] = oof_auc
        except Exception as e:
            print(f"  ERROR training {model_name}: {e}")
            import traceback; traceback.print_exc()
            continue

    print("\n========== FINAL COMPARISON ==========")
    print(f"  efficientnet_b3 (baseline OOF): 0.8963")
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = " <-- NEW BEST" if auc > 0.8963 else ""
        print(f"  {name}: {auc:.4f}{marker}")
    print("======================================")

if __name__ == "__main__":
    main()