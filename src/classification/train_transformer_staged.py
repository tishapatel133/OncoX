"""

Two-stage fine-tuning for transformers on merged ISIC 2019+2020.

Stage 1: freeze backbone, train classifier head (15 epochs, LR 2e-4)

Stage 2: unfreeze last N layers, continue training (15 epochs, LR 1e-4)

Paper-style approach from Garcia et al. 2025.

"""



import os, random, argparse

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

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



ROOT         = "/scratch/patel.tis/OncoX"

IMG_DIR_2020 = os.path.join(ROOT, "data/raw/isic2020_clean")

IMG_DIR_2019 = os.path.join(ROOT, "data/raw/isic2019_clean")

MERGED_CSV   = os.path.join(ROOT, "data/metadata/train_merged_2019_2020.csv")

TEST_CSV     = os.path.join(ROOT, "data/metadata/test.csv")

RESULTS_DIR  = os.path.join(ROOT, "results/classification")

CKPT_DIR     = os.path.join(ROOT, "models/checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)





# ────────── MODEL BUILDERS ──────────



def build_vit_small(dropout=0.3):

    backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)

    dim = backbone.embed_dim

    classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    return BackboneClassifier(backbone, classifier, img_size=224)



def build_swinv2_small(dropout=0.3):

    backbone = timm.create_model("swinv2_small_window8_256", pretrained=True, num_classes=0)

    dim = backbone.num_features

    classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 1))

    return BackboneClassifier(backbone, classifier, img_size=256)





class BackboneClassifier(nn.Module):

    def __init__(self, backbone, classifier, img_size):

        super().__init__()

        self.backbone = backbone

        self.classifier = classifier

        self.img_size = img_size

    def forward(self, x):

        feat = self.backbone(x)

        if feat.dim() > 2:

            feat = feat.mean(dim=list(range(1, feat.dim()-1))) if feat.dim() == 4 else feat.mean(dim=1)

        return self.classifier(feat)





# ────────── STAGED FREEZE/UNFREEZE ──────────



def freeze_backbone(model):

    for p in model.backbone.parameters():

        p.requires_grad = False

    for p in model.classifier.parameters():

        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_total = sum(p.numel() for p in model.parameters())

    print(f"  Stage 1 (frozen backbone): {n_train/1e6:.2f}M / {n_total/1e6:.2f}M trainable")



def unfreeze_last_layers(model, model_type):

    """Unfreeze last N layers. For ViT: last 4 of 12 blocks. For Swin: last stage (layers[3])."""

    if model_type == "vit_small":

        for p in model.backbone.parameters():

            p.requires_grad = False

        for i in range(8, 12):

            for p in model.backbone.blocks[i].parameters():

                p.requires_grad = True

        for p in model.backbone.norm.parameters():

            p.requires_grad = True

    elif model_type == "swinv2_small":

        for p in model.backbone.parameters():

            p.requires_grad = False

        # swinv2 has 4 stages accessible via .layers[0..3]

        for p in model.backbone.layers[3].parameters():

            p.requires_grad = True

        if hasattr(model.backbone, 'norm'):

            for p in model.backbone.norm.parameters():

                p.requires_grad = True

    for p in model.classifier.parameters():

        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_total = sum(p.numel() for p in model.parameters())

    print(f"  Stage 2 (unfrozen last layers): {n_train/1e6:.2f}M / {n_total/1e6:.2f}M trainable")





# ────────── DATA ──────────



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



class MelanomaMergedDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df.reset_index(drop=True)

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        src = row.get("source", "isic2020")

        img_dir = IMG_DIR_2019 if src == "isic2019" else IMG_DIR_2020

        img_path = os.path.join(img_dir, row["image_name"] + ".jpg")

        img = np.array(Image.open(img_path).convert("RGB"))

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





# ────────── METRICS ──────────



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





# ────────── TRAINING ──────────



def run_stage(model, train_loader, val_loader, val_labels, device, epochs, lr, stage_name):

    scaler = torch.cuda.amp.GradScaler()

    criterion = FocalLoss(0.25, 2.0)

    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(trainable, lr=lr, weight_decay=1e-2)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)



    best_auc = 0.0

    best_probs = None



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

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)

            scaler.step(optimizer); scaler.update()

        scheduler.step()



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

        print(f"    [{stage_name}] Epoch {epoch:02d}/{epochs}  val_auc={auc:.4f}")

        if auc > best_auc:

            best_auc = auc; best_probs = probs

    return best_auc, best_probs





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["vit_small", "swinv2_small"], required=True)

    parser.add_argument("--start_fold", type=int, default=1)

    args = parser.parse_args()



    BATCH = 32 if args.model == "vit_small" else 16

    WORKERS = 4

    EPOCHS_S1 = 15

    EPOCHS_S2 = 15



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    print(f"Model: {args.model}")



    df = pd.read_csv(MERGED_CSV)

    print(f"Merged CSV: {len(df)} rows, {int(df['target'].sum())} melanoma")



    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))



    fold_aucs = []

    oof_preds = np.zeros(len(df))

    oof_labels = df["target"].values



    # build once to get img_size

    model_stub = build_vit_small() if args.model == "vit_small" else build_swinv2_small()

    IMG_SIZE = model_stub.img_size

    del model_stub

    print(f"Image size: {IMG_SIZE}")



    for fold, (train_idx, val_idx) in enumerate(split):

        if (fold + 1) < args.start_fold:

            print(f"Skipping fold {fold+1}")

            continue

        print(f"\n{'='*60}\nFold {fold+1}/5\n{'='*60}")



        train_df = df.iloc[train_idx]

        val_df = df.iloc[val_idx]

        val_labels = val_df["target"].values.astype(int)

        print(f"  Train: {len(train_df)} ({int(train_df['target'].sum())} mel)")

        print(f"  Val:   {len(val_df)} ({int(val_df['target'].sum())} mel)")



        train_ds = MelanomaMergedDataset(train_df, get_transforms(IMG_SIZE, train=True))

        val_ds = MelanomaMergedDataset(val_df, get_transforms(IMG_SIZE, train=False))

        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)

        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)



        model = build_vit_small().to(device) if args.model == "vit_small" else build_swinv2_small().to(device)



        # STAGE 1: frozen backbone

        print(f"\n  --- Stage 1: frozen backbone, train classifier head ---")

        freeze_backbone(model)

        _, _ = run_stage(model, train_loader, val_loader, val_labels, device, EPOCHS_S1, lr=2e-4, stage_name="S1")



        # STAGE 2: unfreeze last layers

        print(f"\n  --- Stage 2: unfreeze last layers, LR 1e-4 ---")

        unfreeze_last_layers(model, args.model)

        best_auc, best_probs = run_stage(model, train_loader, val_loader, val_labels, device, EPOCHS_S2, lr=1e-4, stage_name="S2")



        fold_aucs.append(best_auc)

        oof_preds[val_idx] = best_probs

        torch.save({"model_state": model.state_dict(), "auc": best_auc},

                   os.path.join(CKPT_DIR, f"{args.model}_staged_merged_fold{fold+1}.pt"))

        del model; torch.cuda.empty_cache()

        print(f"  Fold {fold+1} best AUC (after 2 stages): {best_auc:.4f}")



    if len(fold_aucs) == 5:

        oof_auc = roc_auc_score(oof_labels, oof_preds)

        mean_auc = np.mean(fold_aucs)

        std_auc = np.std(fold_aucs)

        thresh = find_best_threshold(oof_labels.astype(int), oof_preds)

        metrics = compute_metrics(oof_labels.astype(int), oof_preds, thresh)



        print(f"\n{'='*60}")

        print(f"FINAL — {args.model} staged, merged 2019+2020")

        print(f"{'='*60}")

        print(f"  Fold AUCs:      {[round(a,4) for a in fold_aucs]}")

        print(f"  Mean fold AUC:  {mean_auc:.4f} ± {std_auc:.4f}")

        print(f"  OOF AUC:        {oof_auc:.4f}")

        print(f"  F1: {metrics['F1']}  Sens: {metrics['Sensitivity']}  Spec: {metrics['Specificity']}  MCC: {metrics['MCC']}")



        summary = {

            "model": f"{args.model}_staged_merged",

            "fold_aucs": str(fold_aucs),

            "mean_auc": round(mean_auc, 4),

            "std_auc": round(std_auc, 4),

            "oof_auc": round(oof_auc, 4),

            **metrics

        }

        pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, f"{args.model}_staged_merged_summary.csv"), index=False)



if __name__ == "__main__":

    main()
