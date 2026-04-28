"""

Phase 2: Retrain classifier on seg-masked images.

Seg model runs first → mask generated → crop applied → classifier trained.

Trains both EfficientNet-CBAM and SwinV2 and compares.

"""

import os, sys, time, argparse

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from pathlib import Path

from PIL import Image

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

import albumentations as A

from albumentations.pytorch import ToTensorV2

import cv2



BASE = Path("/scratch/patel.tis/OncoX")

sys.path.insert(0, str(BASE / "src"))

sys.path.insert(0, str(BASE / "src" / "classification"))

sys.path.insert(0, str(BASE / "src" / "segmentation"))



from seg_models import PVTv2UNet

from cls_models import build_model



# ── Args ─────────────────────────────────────────────────

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True,

                    choices=["efficientnet_cbam", "swinv2"])

parser.add_argument("--strategy", default="crop",

                    choices=["hard_mask", "crop"],

                    help="Masking strategy to use for training")

parser.add_argument("--epochs",     type=int,   default=25)

parser.add_argument("--batch_size", type=int,   default=32)

parser.add_argument("--img_size",   type=int,   default=224)

parser.add_argument("--lr",         type=float, default=3e-4)

parser.add_argument("--pos_weight", type=float, default=8.0)

parser.add_argument("--grad_clip",  type=float, default=1.0)

parser.add_argument("--patience",   type=int,   default=7)

parser.add_argument("--workers",    type=int,   default=4)

args = parser.parse_args()



DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR  = Path(BASE / "data/metadata/img_dir.txt").read_text().strip()

SEG_CKPT = BASE / "models/checkpoints/segmentation/best_pvtv2_unet.pth"

CKPT_DIR = BASE / "models/checkpoints/classification"

RES_DIR  = BASE / "results/classification"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

RES_DIR.mkdir(parents=True, exist_ok=True)



# ── Load segmentation model once (shared across dataset) ─

def load_seg_model():

    model = PVTv2UNet(pretrained=False)

    state = torch.load(SEG_CKPT, map_location=DEVICE, weights_only=False)

    model.load_state_dict(state["model_state"] if "model_state" in state else state)

    model.to(DEVICE).eval()

    for p in model.parameters():

        p.requires_grad = False   # frozen — only used for mask generation

    print("PVTv2-UNet segmentation model loaded and frozen.")

    return model



# ── Masking utilities ────────────────────────────────────

seg_tf = A.Compose([

    A.Resize(256, 256),

    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

    ToTensorV2(),

])



def get_mask(seg_model, img_np):

    t = seg_tf(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = torch.sigmoid(seg_model(t)).squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]),

                      interpolation=cv2.INTER_NEAREST)

    return mask



def apply_strategy(img_np, mask, strategy):

    if strategy == "hard_mask":

        out = img_np.copy()

        out[mask == 0] = 0

        return out

    else:  # crop

        coords = cv2.findNonZero(mask)

        if coords is None:

            return img_np

        x, y, w, h = cv2.boundingRect(coords)

        pad = int(0.1 * max(w, h))

        x1 = max(0, x - pad)

        y1 = max(0, y - pad)

        x2 = min(img_np.shape[1], x + w + pad)

        y2 = min(img_np.shape[0], y + h + pad)

        return img_np[y1:y2, x1:x2]



# ── Dataset ──────────────────────────────────────────────

class MaskedISICDataset(Dataset):

    def __init__(self, csv_path, img_dir, seg_model, strategy, is_train=True):

        self.df        = pd.read_csv(csv_path)

        self.img_dir   = Path(img_dir)

        self.seg_model = seg_model

        self.strategy  = strategy

        self.transform = self._build_transform(is_train)



    def _build_transform(self, is_train):

        if is_train:

            return A.Compose([

                A.Resize(args.img_size, args.img_size),

                A.HorizontalFlip(p=0.5),

                A.VerticalFlip(p=0.5),

                A.RandomRotate90(p=0.5),

                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,

                                   rotate_limit=30, p=0.5),

                A.RandomBrightnessContrast(p=0.3),

                A.HueSaturationValue(p=0.3),

                A.Normalize(mean=(0.485,0.456,0.406),

                            std=(0.229,0.224,0.225)),

                ToTensorV2(),

            ])

        return A.Compose([

            A.Resize(args.img_size, args.img_size),

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



        # segment → mask → apply strategy

        mask    = get_mask(self.seg_model, img)

        img_out = apply_strategy(img, mask, self.strategy)



        tensor = self.transform(image=img_out)["image"]

        return tensor, torch.tensor(label, dtype=torch.float32)



# ── Training loop ────────────────────────────────────────

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

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

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

            out    = model(imgs).squeeze(1)

            loss   = criterion(out, labels)

            losses.append(loss.item())

            preds_all.extend(torch.sigmoid(out).cpu().numpy())

            labels_all.extend(labels.cpu().numpy())

    auc  = roc_auc_score(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0

    pbin = [1 if p > 0.5 else 0 for p in preds_all]

    acc  = accuracy_score(labels_all, pbin)

    f1   = f1_score(labels_all, pbin, zero_division=0)

    return np.mean(losses), auc, acc, f1



# ── Main ─────────────────────────────────────────────────

def main():

    print("="*60)

    print(f"Phase 2: Retrain {args.model} on {args.strategy} masked images")

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("="*60)



    seg_model = load_seg_model()



    train_ds = MaskedISICDataset(BASE/"data/metadata/train.csv",

                                  IMG_DIR, seg_model, args.strategy, is_train=True)

    val_ds   = MaskedISICDataset(BASE/"data/metadata/val.csv",

                                  IMG_DIR, seg_model, args.strategy, is_train=False)



    train_loader = DataLoader(train_ds, batch_size=args.batch_size,

                              shuffle=True,  num_workers=0,

                              pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,

                              shuffle=False, num_workers=0,

                              pin_memory=True)



    print(f"  train → {len(train_ds)} samples | val → {len(val_ds)} samples")



    model     = build_model(args.model, num_classes=1, pretrained=True).to(DEVICE)

    pw        = torch.tensor([args.pos_weight], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = OneCycleLR(optimizer, max_lr=args.lr,

                           steps_per_epoch=len(train_loader),

                           epochs=args.epochs, pct_start=0.1)

    scaler    = torch.amp.GradScaler("cuda")



    best_auc, patience_cnt = 0.0, 0

    log_rows = []

    ckpt_path = CKPT_DIR / f"best_{args.model}_masked_{args.strategy}.pth"

    t0 = time.time()



    for epoch in range(1, args.epochs + 1):

        te = time.time()

        tr_loss, tr_auc               = train_one_epoch(model, train_loader,

                                                         optimizer, criterion,

                                                         scaler, scheduler)

        vl_loss, vl_auc, vl_acc, vl_f1 = evaluate(model, val_loader, criterion)

        elapsed = time.time() - te



        log_rows.append([epoch, tr_loss, tr_auc, vl_loss,

                         vl_auc, vl_acc, vl_f1,

                         optimizer.param_groups[0]["lr"], elapsed])



        status = ""

        if vl_auc > best_auc:

            best_auc = vl_auc

            patience_cnt = 0

            torch.save({"model": model.state_dict(),

                        "epoch": epoch, "auc": best_auc}, ckpt_path)

            status = f"  ✅ New best! AUC: {best_auc:.4f}"

        else:

            patience_cnt += 1

            status = f"  No improvement ({patience_cnt}/{args.patience})"



        total = (time.time() - t0) / 60

        print(f"\nEpoch {epoch}/{args.epochs}  ({elapsed:.0f}s, {total:.1f}min)")

        print(f"  Train → Loss: {tr_loss:.4f}  AUC: {tr_auc:.4f}")

        print(f"  Val   → Loss: {vl_loss:.4f}  AUC: {vl_auc:.4f}"

              f"  Acc: {vl_acc:.4f}  F1: {vl_f1:.4f}")

        print(status)



        if patience_cnt >= args.patience:

            print(f"\n⏹ Early stopping at epoch {epoch}")

            break



    pd.DataFrame(log_rows, columns=["epoch","tr_loss","tr_auc","vl_loss",

                                     "vl_auc","vl_acc","vl_f1","lr","time_s"]

                 ).to_csv(RES_DIR / f"{args.model}_masked_{args.strategy}_log.csv",

                          index=False)



    print(f"\n{'='*60}")

    print(f"Done — Best Val AUC: {best_auc:.4f}")

    print(f"Checkpoint: {ckpt_path}")

    print(f"{'='*60}")



if __name__ == "__main__":

    main()
