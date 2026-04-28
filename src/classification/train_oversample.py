"""

Experiment 4: Random oversampling with heavy augmentation

Oversample melanoma to 1:4 ratio (melanoma:benign) but apply

strong augmentation every time so repeats look different.

Also tests class-weighted sampler at 1:4 ratio as second approach.

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



def get_transforms(img_size, strong=False):

    if strong:

        return A.Compose([

            A.Resize(img_size, img_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.7),

            A.OneOf([A.ElasticTransform(p=1.0), A.GridDistortion(p=1.0), A.OpticalDistortion(p=1.0)], p=0.4),

            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),

            A.OneOf([A.GaussNoise(p=1.0), A.GaussianBlur(p=1.0), A.MotionBlur(p=1.0)], p=0.3),

            A.CoarseDropout(max_holes=12, max_height=img_size//12, max_width=img_size//12, p=0.4),

            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

            ToTensorV2()

        ])

    if img_size > 0:

        return A.Compose([

            A.Resize(img_size, img_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

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



def get_val_transforms(img_size):

    return A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

        ToTensorV2()

    ])



class MelanomaDataset(Dataset):

    def __init__(self, df, img_dir, transform=None, oversample_pos=False, oversample_ratio=4):

        self.img_dir = img_dir

        self.transform = transform

        if oversample_pos:

            pos = df[df["target"] == 1]

            neg = df[df["target"] == 0]

            n_repeats = max(1, int(len(neg) / (len(pos) * oversample_ratio)))

            df = pd.concat([neg] + [pos] * n_repeats).sample(frac=1, random_state=SEED)

            print(f"  Oversampled: {len(pos)} x{n_repeats} melanoma + {len(neg)} benign = {len(df)} total")

        self.df = df.reset_index(drop=True)



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



def build_model(dropout=0.3):

    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)

    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.num_features, 1))

    return model



def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):

    model.train()

    total_loss = 0

    for imgs, labels in loader:

        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            loss = criterion(model(imgs).squeeze(1), labels)

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

                logits = model(imgs).squeeze(1)

            probs = torch.sigmoid(logits).cpu().numpy()

            probs = np.nan_to_num(probs, nan=0.5)

            preds.extend(probs)

            labels_all.extend(labels.numpy())

    return roc_auc_score(labels_all, preds)



def run_experiment(name, train_df, val_loader, img_size, device, use_oversample, use_weighted_sampler, epochs, batch_size, lr, patience, workers):

    print(f"\n{'='*50}")

    print(f"Running: {name}")

    print(f"{'='*50}")



    model = build_model().to(device)

    scaler = torch.cuda.amp.GradScaler()

    criterion = FocalLoss(0.25, 2.0)



    # build dataset

    strong_tfm = get_transforms(img_size, strong=True)

    train_ds = MelanomaDataset(train_df, IMG_DIR, strong_tfm, oversample_pos=use_oversample, oversample_ratio=4)



    # weighted sampler targets 1:4 ratio in each batch

    if use_weighted_sampler:

        labels_arr = train_ds.df["target"].values

        class_counts = np.bincount(labels_arr.astype(int))

        weights = np.where(labels_arr == 1, len(labels_arr)/class_counts[1], len(labels_arr)/(class_counts[0]*4))

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True, drop_last=True)

    else:

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)



    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1)



    log_rows = []

    best_auc = 0.0

    ckpt_path = os.path.join(CKPT_DIR, f"{name.replace(' ', '_')}_best.pt")

    patience_counter = 0



    for epoch in range(1, epochs + 1):

        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)

        auc = validate(model, val_loader, device)

        print(f"  Epoch {epoch:02d}/{epochs}  loss={loss:.4f}  val_auc={auc:.4f}")

        log_rows.append({"experiment": name, "epoch": epoch, "loss": loss, "val_auc": auc})

        if auc > best_auc:

            best_auc = auc; patience_counter = 0

            torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)

            print(f"  New best: {best_auc:.4f}")

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(f"  Early stopping at epoch {epoch}"); break



    print(f"\n{name} — Best Val AUC: {best_auc:.4f}")

    return log_rows, best_auc



def main():

    IMG_SIZE = 384; EPOCHS = 25; BATCH_SIZE = 16; LR = 2e-4; PATIENCE = 7; WORKERS = 0



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    train_df = df.iloc[split[0][0]]

    val_df   = df.iloc[split[0][1]]

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Melanoma in train: {train_df['target'].sum():.0f}")



    val_ds = MelanomaDataset(val_df, IMG_DIR, get_val_transforms(IMG_SIZE))

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=WORKERS, pin_memory=True)



    all_logs = []

    results = {}



    # Approach A: random oversampling + strong augmentation

    logs, auc = run_experiment("oversample_strong_aug", train_df, val_loader, IMG_SIZE, device, use_oversample=True, use_weighted_sampler=False, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE, workers=WORKERS)

    all_logs.extend(logs); results["oversample_strong_aug"] = auc



    # Approach B: weighted sampler 1:4 ratio + strong augmentation

    logs, auc = run_experiment("weighted_sampler_1to4", train_df, val_loader, IMG_SIZE, device, use_oversample=False, use_weighted_sampler=True, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE, workers=WORKERS)

    all_logs.extend(logs); results["weighted_sampler_1to4"] = auc



    pd.DataFrame(all_logs).to_csv(os.path.join(RESULTS_DIR, "oversample_log.csv"), index=False)



    print("\n========== FINAL RESULTS ==========")

    for name, auc in results.items():

        print(f"  {name}: {auc:.4f}")

    print(f"  Baseline: 0.9220")

    print("====================================")



if __name__ == "__main__":

    main()
