"""

Best single-model training pipeline.

Improvements over baseline:

  1. Hair-removed + CLAHE preprocessed images

  2. Focal Loss (handles extreme class imbalance better than BCE)

  3. Larger image size (384x384)

  4. Stronger augmentation

  5. TTA (Test-Time Augmentation) at inference



Target: beat Stanford paper AUC ~0.87 and approach 0.9295.



Focal Loss reference: Lin et al. (2017) RetinaNet, ICCV

"""

import os, sys, time, argparse

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from pathlib import Path

from PIL import Image

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, roc_curve

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm



BASE = Path("/scratch/patel.tis/OncoX")

sys.path.insert(0, str(BASE / "src"))

sys.path.insert(0, str(BASE / "src" / "classification"))



# ── Args ─────────────────────────────────────────────────

parser = argparse.ArgumentParser()

parser.add_argument("--model",      default="efficientnet_b3",

                    choices=["efficientnet_b3", "efficientnet_b4",

                             "efficientnet_b5", "efficientnet_cbam"])

parser.add_argument("--img_size",   type=int,   default=384)

parser.add_argument("--epochs",     type=int,   default=25)

parser.add_argument("--batch_size", type=int,   default=16)

parser.add_argument("--lr",         type=float, default=2e-4)

parser.add_argument("--focal_alpha",type=float, default=0.25,

                    help="Focal loss alpha — down-weights easy negatives")

parser.add_argument("--focal_gamma",type=float, default=2.0,

                    help="Focal loss gamma — focusing parameter")

parser.add_argument("--grad_clip",  type=float, default=1.0)

parser.add_argument("--patience",   type=int,   default=7)

parser.add_argument("--workers",    type=int,   default=4)

parser.add_argument("--use_clean",  action="store_true",

                    help="Use hair-removed + CLAHE images")

parser.add_argument("--tta",        action="store_true",

                    help="Use TTA at final test evaluation")

args = parser.parse_args()



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Image directory — clean or original


clean_dir_file = BASE / "data/metadata/img_dir_clean.txt"
clean_img_dir  = Path("/scratch/patel.tis/OncoX/data/raw/isic2020_clean")
if args.use_clean and clean_img_dir.exists() and any(clean_img_dir.iterdir()):
    IMG_DIR = str(clean_img_dir)
    print(f"Using CLEAN images: {IMG_DIR}")
else:
    IMG_DIR = Path(BASE / "data/metadata/img_dir.txt").read_text().strip()
    print(f"Using ORIGINAL images: {IMG_DIR}")


CKPT_DIR = BASE / "models/checkpoints/classification"

RES_DIR  = BASE / "results/classification"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

RES_DIR.mkdir(parents=True, exist_ok=True)



# ── Focal Loss ────────────────────────────────────────────

class FocalLoss(nn.Module):

    """

    Focal Loss — Lin et al. (2017) ICCV.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)



    Key benefit over BCE on imbalanced data:

    - gamma > 0 reduces loss contribution from easy negatives

    - The (1-p_t)^gamma factor down-weights well-classified examples

    - Lets the model focus on hard misclassified melanoma cases

    """

    def __init__(self, alpha=0.25, gamma=2.0):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma



    def forward(self, logits, targets):

        bce  = F.binary_cross_entropy_with_logits(

                   logits, targets, reduction="none")

        p_t  = torch.exp(-bce)                        # probability of correct class

        fl   = self.alpha * (1 - p_t) ** self.gamma * bce

        return fl.mean()



# ── Dataset ──────────────────────────────────────────────

class ISICBestDataset(Dataset):

    def __init__(self, csv_path, img_dir, is_train=True):

        self.df      = pd.read_csv(csv_path)

        self.img_dir = Path(img_dir)

        self.tf      = self._build_transform(is_train)



    def _build_transform(self, is_train):

        sz = args.img_size

        if is_train:

            return A.Compose([

                A.Resize(sz, sz),

                A.HorizontalFlip(p=0.5),

                A.VerticalFlip(p=0.5),

                A.RandomRotate90(p=0.5),

                A.Transpose(p=0.5),

                A.ShiftScaleRotate(shift_limit=0.1,

                                   scale_limit=0.15,

                                   rotate_limit=45, p=0.5),

                A.OneOf([

                    A.RandomBrightnessContrast(p=1),

                    A.HueSaturationValue(p=1),

                    A.RandomGamma(p=1),

                ], p=0.5),

                A.OneOf([

                    A.GaussianBlur(blur_limit=3, p=1),

                    A.MedianBlur(blur_limit=3, p=1),

                    A.GaussNoise(p=1),

                ], p=0.3),

                A.CoarseDropout(max_holes=8,

                                max_height=sz//8,

                                max_width=sz//8, p=0.3),

                A.Normalize(mean=(0.485,0.456,0.406),

                            std=(0.229,0.224,0.225)),

                ToTensorV2(),

            ])

        return A.Compose([

            A.Resize(sz, sz),

            A.Normalize(mean=(0.485,0.456,0.406),

                        std=(0.229,0.224,0.225)),

            ToTensorV2(),

        ])



    def __len__(self): return len(self.df)



    def __getitem__(self, idx):

        row   = self.df.iloc[idx]

        label = int(row["target"])

        path  = self.img_dir / f"{row['image_name']}.jpg"

        img   = np.array(Image.open(path).convert("RGB"))

        t     = self.tf(image=img)["image"]

        return t, torch.tensor(label, dtype=torch.float32)



# ── TTA transforms ────────────────────────────────────────

def get_tta_transforms():

    sz = args.img_size

    base = [

        A.Resize(sz, sz),

        A.Normalize(mean=(0.485,0.456,0.406),

                    std=(0.229,0.224,0.225)),

        ToTensorV2(),

    ]

    return [

        A.Compose(base),

        A.Compose([A.HorizontalFlip(p=1)] + base),

        A.Compose([A.VerticalFlip(p=1)]   + base),

        A.Compose([A.Transpose(p=1)]      + base),

        A.Compose([A.RandomRotate90(p=1)] + base),

    ]



# ── Model ─────────────────────────────────────────────────

def build_model(name):

    if name == "efficientnet_cbam":

        from cls_models import build_model as bm

        return bm("efficientnet_cbam", num_classes=1, pretrained=True)



    # Map to timm model names

    name_map = {

        "efficientnet_b3": "efficientnet_b3",

        "efficientnet_b4": "efficientnet_b4",

        "efficientnet_b5": "efficientnet_b5",

    }

    backbone = timm.create_model(name_map[name],

                                 pretrained=True,

                                 num_classes=0)  # remove head

    in_features = backbone.num_features

    model = nn.Sequential(

        backbone,

        nn.Dropout(0.3),

        nn.Linear(in_features, 1)

    )

    return model



# ── Training ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler):

    model.train()

    losses, preds_all, labels_all = [], [], []

    for imgs, labels in tqdm(loader, desc="Train", leave=False):

        imgs   = imgs.to(DEVICE, non_blocking=True)

        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            out  = model(imgs).squeeze(1)

            loss = criterion(out, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        losses.append(loss.item())

        preds_all.extend(torch.sigmoid(out).detach().cpu().numpy())

        labels_all.extend(labels.cpu().numpy())

    auc = roc_auc_score(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0

    return np.mean(losses), auc



def evaluate(model, loader, criterion):

    model.eval()

    losses, preds_all, labels_all = [], [], []

    with torch.no_grad():

        for imgs, labels in tqdm(loader, desc="Val", leave=False):

            imgs   = imgs.to(DEVICE, non_blocking=True)

            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda"):

                out  = model(imgs).squeeze(1)

                loss = criterion(out, labels)

            losses.append(loss.item())

            preds_all.extend(torch.sigmoid(out).cpu().numpy())

            labels_all.extend(labels.cpu().numpy())

    preds_all  = np.array(preds_all)

    labels_all = np.array(labels_all)

    auc  = roc_auc_score(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0

    pbin = (preds_all > 0.5).astype(int)

    acc  = accuracy_score(labels_all, pbin)

    f1   = f1_score(labels_all, pbin, zero_division=0)

    return np.mean(losses), auc, acc, f1



def evaluate_with_tta(model, csv_path, img_dir):

    """TTA inference — average predictions across 5 augmentation views."""

    print("  Running TTA evaluation...")

    df      = pd.read_csv(csv_path)

    tta_tfs = get_tta_transforms()

    all_preds = np.zeros(len(df))



    for tf in tta_tfs:

        ds  = ISICBestDataset.__new__(ISICBestDataset)

        ds.df      = df

        ds.img_dir = Path(img_dir)

        ds.tf      = tf

        loader = DataLoader(ds, batch_size=args.batch_size,

                            shuffle=False, num_workers=args.workers,

                            pin_memory=True)

        preds = []

        model.eval()

        with torch.no_grad():

            for imgs, _ in tqdm(loader, desc="  TTA", leave=False):

                imgs = imgs.to(DEVICE)

                out  = torch.sigmoid(model(imgs).squeeze(1))

                preds.extend(out.cpu().numpy())

        all_preds += np.array(preds)



    all_preds /= len(tta_tfs)

    labels = df["target"].values



    auc  = roc_auc_score(labels, all_preds)

    fpr, tpr, thresholds = roc_curve(labels, all_preds)

    opt_thresh = thresholds[np.argmax(tpr - fpr)]

    pbin = (all_preds >= opt_thresh).astype(int)



    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(labels, pbin).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    bacc = (sens + spec) / 2



    print(f"  TTA AUC:          {auc:.4f}")

    print(f"  Balanced Acc:     {bacc:.4f}")

    print(f"  Sensitivity:      {sens:.4f}")

    print(f"  Specificity:      {spec:.4f}")

    print(f"  Optimal threshold:{opt_thresh:.4f}")

    return auc



# ── Main ─────────────────────────────────────────────────

def main():

    suffix = f"{'_clean' if args.use_clean else ''}_{args.img_size}"

    run_name = f"{args.model}{suffix}"



    print("="*60)

    print(f"Best Single Model Training — {run_name}")

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print(f"Image size: {args.img_size}  |  Focal Loss: alpha={args.focal_alpha} gamma={args.focal_gamma}")

    print("="*60)



    train_ds = ISICBestDataset(BASE/"data/metadata/train.csv",

                               IMG_DIR, is_train=True)

    val_ds   = ISICBestDataset(BASE/"data/metadata/val.csv",

                               IMG_DIR, is_train=False)



    train_loader = DataLoader(train_ds, batch_size=args.batch_size,

                              shuffle=True,  num_workers=args.workers,

                              pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,

                              shuffle=False, num_workers=args.workers,

                              pin_memory=True)



    print(f"  train → {len(train_ds):,} | val → {len(val_ds):,}")



    model     = build_model(args.model).to(DEVICE)

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = OneCycleLR(optimizer, max_lr=args.lr,

                           steps_per_epoch=len(train_loader),

                           epochs=args.epochs, pct_start=0.1)

    scaler    = torch.amp.GradScaler("cuda")



    best_auc, patience_cnt = 0.0, 0

    log_rows  = []

    ckpt_path = CKPT_DIR / f"best_{run_name}.pth"

    t0 = time.time()



    for epoch in range(1, args.epochs + 1):

        te = time.time()

        tr_loss, tr_auc             = train_one_epoch(model, train_loader,

                                                       optimizer, criterion,

                                                       scaler, scheduler)

        vl_loss, vl_auc, acc, f1    = evaluate(model, val_loader, criterion)

        elapsed = time.time() - te



        log_rows.append([epoch, tr_loss, tr_auc, vl_loss,

                         vl_auc, acc, f1,

                         optimizer.param_groups[0]["lr"], elapsed])



        status = ""

        if vl_auc > best_auc:

            best_auc     = vl_auc

            patience_cnt = 0

            torch.save({"model_state": model.state_dict(),

                        "epoch": epoch, "auc": best_auc,

                        "args": vars(args)}, ckpt_path)

            status = f"  ✅ New best! AUC: {best_auc:.4f}"

        else:

            patience_cnt += 1

            status = f"  No improvement ({patience_cnt}/{args.patience})"



        total = (time.time() - t0) / 60

        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.0f}s, {total:.1f}min)")

        print(f"  Train → Loss: {tr_loss:.4f}  AUC: {tr_auc:.4f}")

        print(f"  Val   → Loss: {vl_loss:.4f}  AUC: {vl_auc:.4f}"

              f"  Acc: {acc:.4f}  F1: {f1:.4f}")

        print(status)



        if patience_cnt >= args.patience:

            print(f"\n⏹ Early stopping at epoch {epoch}")

            break



    pd.DataFrame(log_rows,

                 columns=["epoch","tr_loss","tr_auc","vl_loss",

                          "vl_auc","vl_acc","vl_f1","lr","time_s"]

                 ).to_csv(RES_DIR / f"{run_name}_log.csv", index=False)



    print(f"\n{'='*60}")

    print(f"Training done — Best Val AUC: {best_auc:.4f}")

    print(f"Checkpoint: {ckpt_path}")



    # TTA evaluation on val set

    if args.tta:

        print("\n── TTA Evaluation on Val Set ──")

        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        model.load_state_dict(state["model_state"])

        tta_auc = evaluate_with_tta(model,

                                    BASE/"data/metadata/val.csv",

                                    IMG_DIR)

        print(f"\nFinal Val AUC with TTA: {tta_auc:.4f}")



    print(f"{'='*60}")



if __name__ == "__main__":

    main()
