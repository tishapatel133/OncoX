"""

Full upgrade pipeline targeting AUC 0.93-0.94:

  1. 5-fold stratified cross-validation (patient-level)

  2. MixUp augmentation

  3. Label smoothing

  4. Patient metadata fusion (age, sex, anatomical site)

  5. EfficientNet-B5 backbone

  6. Focal Loss

  7. TTA at inference

  8. Larger image size (384px)



References:

  - MixUp: Zhang et al. (2018) ICLR

  - Label Smoothing: Szegedy et al. (2016) CVPR

  - Focal Loss: Lin et al. (2017) ICCV

  - Metadata fusion: Ha et al. (2020) SIIM-ISIC winning solution

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

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from sklearn.model_selection import StratifiedGroupKFold

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm



BASE    = Path("/scratch/patel.tis/OncoX")

sys.path.insert(0, str(BASE / "src"))



# ── Args ─────────────────────────────────────────────────

parser = argparse.ArgumentParser()

parser.add_argument("--model",       default="efficientnet_b5",

                    choices=["efficientnet_b3","efficientnet_b4","efficientnet_b5"])

parser.add_argument("--img_size",    type=int,   default=384)

parser.add_argument("--epochs",      type=int,   default=20)

parser.add_argument("--batch_size",  type=int,   default=16)

parser.add_argument("--lr",          type=float, default=2e-4)

parser.add_argument("--focal_alpha", type=float, default=0.25)

parser.add_argument("--focal_gamma", type=float, default=2.0)

parser.add_argument("--label_smooth",type=float, default=0.05,

                    help="Label smoothing epsilon")

parser.add_argument("--mixup_alpha", type=float, default=0.4,

                    help="MixUp alpha — 0 disables MixUp")

parser.add_argument("--n_folds",     type=int,   default=5)

parser.add_argument("--grad_clip",   type=float, default=1.0)

parser.add_argument("--patience",    type=int,   default=5)

parser.add_argument("--workers",     type=int,   default=4)

parser.add_argument("--use_meta",    action="store_true",

                    help="Fuse patient metadata (age, sex, site)")

parser.add_argument("--tta_folds",   type=int,   default=5,

                    help="Number of TTA augmentations at inference")

args = parser.parse_args()



DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR  = Path(BASE / "data/metadata/img_dir.txt").read_text().strip()

CKPT_DIR = BASE / "models/checkpoints/classification"

RES_DIR  = BASE / "results/classification"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

RES_DIR.mkdir(parents=True, exist_ok=True)



# ── Metadata encoding ─────────────────────────────────────

SEX_MAP  = {"male": 0, "female": 1}

SITE_MAP = {

    "head/neck": 0, "upper extremity": 1, "lower extremity": 2,

    "torso": 3, "palms/soles": 4, "oral/genital": 5, "unknown": 6

}

META_DIM = 10  # age(1) + sex(2) + site(7) one-hot → 9 features total



def encode_meta(row):

    """Encode patient metadata into a fixed-length float vector."""

    # Age — normalize to [0,1] range (max age ~90)

    age = float(row.get("age_approx", 45) or 45) / 90.0



    # Sex — one-hot (male, female)

    sex_str = str(row.get("sex", "")).lower()

    sex = [0.0, 0.0]

    if sex_str == "male":   sex[0] = 1.0

    elif sex_str == "female": sex[1] = 1.0

    else: sex = [0.5, 0.5]  # unknown → neutral



    # Anatomical site — one-hot (7 categories)

    site_str = str(row.get("anatom_site_general_challenge", "")).lower()

    site = [0.0] * 7

    idx = SITE_MAP.get(site_str, 6)

    site[idx] = 1.0



    return np.array([age] + sex + site, dtype=np.float32)



# ── Focal Loss with label smoothing ──────────────────────

class FocalLossWithSmoothing(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.05):

        super().__init__()

        self.alpha     = alpha

        self.gamma     = gamma

        self.smoothing = smoothing



    def forward(self, logits, targets):

        # Apply label smoothing

        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(

                  logits, targets_smooth, reduction="none")

        p_t = torch.exp(-bce)

        fl  = self.alpha * (1 - p_t) ** self.gamma * bce

        return fl.mean()



# ── MixUp ─────────────────────────────────────────────────

def mixup_data(x, y, meta, alpha=0.4):

    """MixUp augmentation — Zhang et al. (2018) ICLR."""

    if alpha <= 0:

        return x, y, meta, y, meta, 1.0

    lam = np.random.beta(alpha, alpha)

    bs  = x.size(0)

    idx = torch.randperm(bs, device=x.device)

    mixed_x    = lam * x    + (1 - lam) * x[idx]

    mixed_meta = lam * meta + (1 - lam) * meta[idx] if meta is not None else None

    y_a, y_b   = y, y[idx]

    return mixed_x, y_a, mixed_meta, y_b, None, lam



def mixup_criterion(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



# ── Model with optional metadata fusion ──────────────────

class EfficientNetWithMeta(nn.Module):

    """

    EfficientNet backbone + optional metadata fusion head.

    Architecture:

      Image → EfficientNet backbone → 2048-d features

      Meta  → MLP → 64-d features

      Concat → Dropout → Linear → logit

    """

    def __init__(self, backbone_name, use_meta=True):

        super().__init__()

        self.use_meta = use_meta



        # Image backbone

        self.backbone = timm.create_model(

            backbone_name, pretrained=True, num_classes=0)

        img_features = self.backbone.num_features



        if use_meta:

            # Metadata MLP

            self.meta_mlp = nn.Sequential(

                nn.Linear(META_DIM, 64),

                nn.BatchNorm1d(64),

                nn.SiLU(),

                nn.Dropout(0.3),

                nn.Linear(64, 64),

                nn.SiLU(),

            )

            fusion_dim = img_features + 64

        else:

            fusion_dim = img_features



        # Classification head

        self.head = nn.Sequential(

            nn.Dropout(0.4),

            nn.Linear(fusion_dim, 512),

            nn.SiLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 1)

        )



    def forward(self, img, meta=None):

        img_feat = self.backbone(img)

        if self.use_meta and meta is not None:

            meta_feat = self.meta_mlp(meta)

            feat = torch.cat([img_feat, meta_feat], dim=1)

        else:

            feat = img_feat

        return self.head(feat)



# ── Dataset ──────────────────────────────────────────────

class ISICDataset(Dataset):

    def __init__(self, df, img_dir, is_train=True):

        self.df      = df.reset_index(drop=True)

        self.img_dir = Path(img_dir)

        self.tf      = self._build_transform(is_train)

        self.use_meta = args.use_meta



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

                    A.GaussNoise(p=1),

                ], p=0.3),

                A.CoarseDropout(p=0.3),

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

        meta  = torch.tensor(encode_meta(row), dtype=torch.float32) if self.use_meta else torch.zeros(META_DIM)


        return t, torch.tensor(label, dtype=torch.float32), meta



# ── TTA transforms ────────────────────────────────────────

TTA_TRANSFORMS = [

    A.Compose([A.Resize(args.img_size, args.img_size),

               A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

               ToTensorV2()]),

    A.Compose([A.Resize(args.img_size, args.img_size), A.HorizontalFlip(p=1),

               A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

               ToTensorV2()]),

    A.Compose([A.Resize(args.img_size, args.img_size), A.VerticalFlip(p=1),

               A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

               ToTensorV2()]),

    A.Compose([A.Resize(args.img_size, args.img_size), A.Transpose(p=1),

               A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

               ToTensorV2()]),

    A.Compose([A.Resize(args.img_size, args.img_size), A.RandomRotate90(p=1),

               A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

               ToTensorV2()]),

]



# ── Train one epoch ───────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):

    model.train()

    losses, preds_all, labels_all = [], [], []



    for imgs, labels, meta in tqdm(loader, desc="Train", leave=False):

        imgs   = imgs.to(DEVICE, non_blocking=True)

        labels = labels.to(DEVICE, non_blocking=True)

        meta   = meta.to(DEVICE, non_blocking=True)



        # MixUp

        imgs, y_a, meta_m, y_b, _, lam = mixup_data(

            imgs, labels, meta, args.mixup_alpha)



        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            out  = model(imgs, meta_m).squeeze(1)

            if lam < 1.0:

                loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            else:

                loss = criterion(out, labels)



        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Skip step if gradients contain NaN
        found_inf = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                found_inf = True
                break
        if not found_inf:
            scaler.step(optimizer)
        else:
            optimizer.zero_grad()
        scaler.update()

        scheduler.step()



        losses.append(loss.item())

        p = torch.sigmoid(out).detach().cpu().numpy()
        p = np.nan_to_num(p, nan=0.5)
        preds_all.extend(p)

        labels_all.extend(labels.cpu().numpy())



    auc = roc_auc_score(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0


    return np.mean(losses), auc



# ── Validation ────────────────────────────────────────────

def validate(model, loader, criterion):

    model.eval()

    losses, preds_all, labels_all = [], [], []

    with torch.no_grad():

        for imgs, labels, meta in tqdm(loader, desc="Val", leave=False):

            imgs   = imgs.to(DEVICE, non_blocking=True)

            labels = labels.to(DEVICE, non_blocking=True)

            meta   = meta.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda"):

                out  = model(imgs, meta).squeeze(1)

                loss = criterion(out, labels)

            losses.append(loss.item())

            p = torch.sigmoid(out).cpu().numpy(); p = np.nan_to_num(p, nan=0.5); preds_all.extend(p)

            labels_all.extend(labels.cpu().numpy())



    preds_all  = np.array(preds_all)

    labels_all = np.array(labels_all)

    auc = roc_auc_score(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0


    return np.mean(losses), auc, preds_all, labels_all



# ── TTA inference for one fold model ─────────────────────

def tta_predict(model, df, img_dir):

    """Average predictions over TTA_FOLDS augmentation views."""

    all_preds = np.zeros(len(df))

    for tf in TTA_TRANSFORMS[:args.tta_folds]:

        ds = ISICDataset(df, img_dir, is_train=False)

        ds.tf = tf

        loader = DataLoader(ds, batch_size=args.batch_size,

                            shuffle=False, num_workers=args.workers,

                            pin_memory=True)

        preds = []

        model.eval()

        with torch.no_grad():

            for imgs, _, meta in loader:

                imgs = imgs.to(DEVICE)

                meta = meta.to(DEVICE)

                out  = torch.sigmoid(model(imgs, meta).squeeze(1))

                preds.extend(out.cpu().numpy())

        all_preds += np.array(preds)

    return all_preds / args.tta_folds



# ── Main: 5-fold CV ───────────────────────────────────────

def main():

    run_name = f"{args.model}_kfold{args.n_folds}_meta{int(args.use_meta)}_384"

    print("="*65)

    print(f"5-Fold CV Training — {run_name}")

    print(f"GPU : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print(f"Img : {args.img_size}px | Focal α={args.focal_alpha} γ={args.focal_gamma}")

    print(f"Meta: {args.use_meta} | MixUp α={args.mixup_alpha} | Smooth={args.label_smooth}")

    print("="*65)



    # Load full training data

    full_df = pd.read_csv(BASE / "data/metadata/train.csv")

    val_df  = pd.read_csv(BASE / "data/metadata/val.csv")



    # Combine train+val for k-fold (use all available labeled data)

    all_df  = pd.concat([full_df, val_df], ignore_index=True)

    print(f"Total samples: {len(all_df)} | Melanoma: {all_df['target'].sum()}")



    # Patient-level stratified k-fold

    # Groups = patient_id ensures no patient leaks across folds

    sgkf   = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,

                                   random_state=42)

    groups = all_df["patient_id"].values

    labels = all_df["target"].values



    criterion = FocalLossWithSmoothing(

        alpha=args.focal_alpha,

        gamma=args.focal_gamma,

        smoothing=args.label_smooth

    )



    fold_aucs  = []

    oof_preds  = np.zeros(len(all_df))   # out-of-fold predictions

    oof_labels = labels.copy()



    for fold, (tr_idx, vl_idx) in enumerate(

            sgkf.split(all_df, labels, groups)):



        print(f"\n{'='*65}")

        print(f"FOLD {fold+1}/{args.n_folds}")

        print(f"{'='*65}")



        tr_df = all_df.iloc[tr_idx]

        vl_df = all_df.iloc[vl_idx]

        print(f"  Train: {len(tr_df)} | Val: {len(vl_df)}")

        print(f"  Train melanoma: {tr_df['target'].sum()} "

              f"({tr_df['target'].mean()*100:.1f}%)")

        print(f"  Val   melanoma: {vl_df['target'].sum()} "

              f"({vl_df['target'].mean()*100:.1f}%)")



        train_ds = ISICDataset(tr_df, IMG_DIR, is_train=True)

        val_ds   = ISICDataset(vl_df, IMG_DIR, is_train=False)



        train_loader = DataLoader(train_ds, batch_size=args.batch_size,

                                  shuffle=True,  num_workers=args.workers,

                                  pin_memory=True)

        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,

                                  shuffle=False, num_workers=args.workers,

                                  pin_memory=True)



        model     = EfficientNetWithMeta(args.model,

                                         use_meta=args.use_meta).to(DEVICE)

        optimizer = AdamW(model.parameters(),

                          lr=args.lr, weight_decay=1e-4)

        scheduler = OneCycleLR(optimizer, max_lr=args.lr,

                               steps_per_epoch=len(train_loader),

                               epochs=args.epochs, pct_start=0.1)

        scaler    = torch.amp.GradScaler("cuda")



        best_auc, patience_cnt = 0.0, 0

        ckpt_path = CKPT_DIR / f"best_{run_name}_fold{fold+1}.pth"

        log_rows  = []

        t0        = time.time()



        for epoch in range(1, args.epochs + 1):

            te = time.time()

            tr_loss, tr_auc = train_epoch(model, train_loader,

                                           optimizer, criterion,

                                           scaler, scheduler)

            vl_loss, vl_auc, _, _ = validate(model, val_loader, criterion)

            elapsed = time.time() - te



            log_rows.append([fold+1, epoch, tr_loss, tr_auc,

                             vl_loss, vl_auc, elapsed])



            improved = vl_auc > best_auc

            if improved:

                best_auc     = vl_auc

                patience_cnt = 0

                torch.save({"model_state": model.state_dict(),

                            "fold": fold+1, "epoch": epoch,

                            "auc": best_auc}, ckpt_path)



            total = (time.time() - t0) / 60

            mark  = "✅" if improved else f"  ({patience_cnt+1}/{args.patience})"

            print(f"  Ep {epoch:02d}  "

                  f"Tr {tr_auc:.4f}  Vl {vl_auc:.4f}  "

                  f"{mark}  [{total:.1f}min]")



            if not improved:

                patience_cnt += 1

                if patience_cnt >= args.patience:

                    print(f"  ⏹ Early stop at epoch {epoch}")

                    break



        print(f"\n  Fold {fold+1} best val AUC: {best_auc:.4f}")

        fold_aucs.append(best_auc)



        # OOF TTA predictions with best checkpoint

        print(f"  Generating OOF TTA predictions...")

        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        model.load_state_dict(state["model_state"])

        oof_preds[vl_idx] = tta_predict(model, vl_df, IMG_DIR)



        # Save fold log

        pd.DataFrame(log_rows,

                     columns=["fold","epoch","tr_loss","tr_auc",

                              "vl_loss","vl_auc","time_s"]

                     ).to_csv(RES_DIR / f"{run_name}_fold{fold+1}_log.csv",

                              index=False)



    # ── Final OOF AUC ────────────────────────────────────

    oof_auc = roc_auc_score(oof_labels, oof_preds)

    fpr, tpr, thresholds = roc_curve(oof_labels, oof_preds)

    opt_thresh = thresholds[np.argmax(tpr - fpr)]

    pbin = (oof_preds >= opt_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(oof_labels, pbin).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0

    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    bacc = (sens + spec) / 2



    print(f"\n{'='*65}")

    print(f"5-FOLD CV COMPLETE")

    print(f"{'='*65}")

    print(f"  Per-fold AUCs : {[round(a,4) for a in fold_aucs]}")

    print(f"  Mean fold AUC : {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    print(f"  OOF AUC (TTA) : {oof_auc:.4f}")

    print(f"  Balanced Acc  : {bacc:.4f}")

    print(f"  Sensitivity   : {sens:.4f}")

    print(f"  Specificity   : {spec:.4f}")

    print(f"  Opt threshold : {opt_thresh:.4f}")



    # Save OOF predictions and summary

    oof_df = all_df[["image_name","target"]].copy()

    oof_df["oof_pred"] = oof_preds

    oof_df.to_csv(RES_DIR / f"{run_name}_oof_preds.csv", index=False)



    summary = {

        "run": run_name,

        "fold_aucs": fold_aucs,

        "mean_auc": float(np.mean(fold_aucs)),

        "std_auc":  float(np.std(fold_aucs)),

        "oof_auc":  float(oof_auc),

        "balanced_acc": float(bacc),

        "sensitivity":  float(sens),

        "specificity":  float(spec),

    }

    pd.DataFrame([summary]).to_csv(

        RES_DIR / f"{run_name}_summary.csv", index=False)

    print(f"\nSaved OOF predictions and summary to {RES_DIR}")



if __name__ == "__main__":

    main()
