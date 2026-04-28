"""

Experiment 2: Online Hard Example Mining (OHEM)

Oversample melanoma cases where the model is most uncertain (highest loss).

Runs every epoch — hard examples get upweighted dynamically.

"""



import os, random

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import roc_auc_score

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

os.makedirs(CKPT_DIR, exist_ok=True)



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

        self.alpha = alpha

        self.gamma = gamma

    def forward(self, logits, targets):

        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        pt = torch.exp(-bce)

        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()



def build_model(dropout=0.3):

    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)

    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.num_features, 1))

    return model



def compute_sample_weights(model, df, img_dir, img_size, device, batch_size=64):

    """Compute per-sample loss to use as sampling weights — hard examples get higher weight."""

    model.eval()

    tfm = get_transforms(img_size, train=False)

    ds = MelanomaDataset(df, img_dir, tfm)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    all_losses = []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast():

                logits = model(imgs).squeeze(1)

                losses = criterion(logits, labels)

            all_losses.extend(losses.cpu().numpy())

    weights = np.array(all_losses)

    # boost melanoma weights on top of loss-based weights

    for i, row in df.reset_index(drop=True).iterrows():

        if row["target"] == 1:

            weights[i] *= 5.0

    weights = weights / weights.sum()

    return weights



def train_epoch_weighted(model, df, img_dir, img_size, optimizer, scheduler, criterion, device, scaler, batch_size, workers):

    weights = compute_sample_weights(model, df, img_dir, img_size, device)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    tfm = get_transforms(img_size, train=True)

    ds = MelanomaDataset(df, img_dir, tfm)

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True)

    model.train()

    total_loss = 0

    for imgs, labels in loader:

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            logits = model(imgs).squeeze(1)

            loss = criterion(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)



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



def main():

    IMG_SIZE = 384; EPOCHS = 25; BATCH_SIZE = 16; LR = 2e-4; PATIENCE = 7; WORKERS = 0



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    train_df = df.iloc[split[0][0]]

    val_df   = df.iloc[split[0][1]]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}")



    val_ds = MelanomaDataset(val_df, IMG_DIR, get_transforms(IMG_SIZE, train=False))

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)



    model = build_model().to(device)

    scaler = torch.cuda.amp.GradScaler()

    criterion = FocalLoss(0.25, 2.0)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # scheduler steps are approximate since loader rebuilds each epoch

    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_df)//BATCH_SIZE, epochs=EPOCHS, pct_start=0.1)



    log_rows = []

    best_auc = 0.0

    ckpt_path = os.path.join(CKPT_DIR, "ohem_best.pt")

    patience_counter = 0



    for epoch in range(1, EPOCHS + 1):

        print(f"\nEpoch {epoch}/{EPOCHS} — computing sample weights...")

        loss = train_epoch_weighted(model, train_df, IMG_DIR, IMG_SIZE, optimizer, scheduler, criterion, device, scaler, BATCH_SIZE, WORKERS)

        auc = validate(model, val_loader, device)

        print(f"  loss={loss:.4f}  val_auc={auc:.4f}")

        log_rows.append({"epoch": epoch, "loss": loss, "val_auc": auc})

        if auc > best_auc:

            best_auc = auc

            patience_counter = 0

            torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)

            print(f"  New best: {best_auc:.4f}")

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(f"Early stopping at epoch {epoch}")

                break



    pd.DataFrame(log_rows).to_csv(os.path.join(RESULTS_DIR, "ohem_log.csv"), index=False)

    print(f"\nBest Val AUC: {best_auc:.4f}")



if __name__ == "__main__":

    main()
