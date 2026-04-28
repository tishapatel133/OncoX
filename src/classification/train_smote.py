"""

Experiment 3: SMOTE in latent embedding space

Step 1 — Extract EfficientNet-B3 embeddings for all train samples

Step 2 — Apply SMOTE in embedding space to balance classes

Step 3 — Train a classifier head on balanced embeddings

Step 4 — Fine-tune full model end-to-end on original data

"""



import os, random

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE

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

        return A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()])

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



def build_model(dropout=0.3):

    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)

    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.num_features, 1))

    return model



def extract_embeddings(model, df, img_dir, img_size, device, batch_size=64, workers=0):

    """Extract penultimate layer features (before classifier head)."""

    backbone = nn.Sequential(*list(model.children())[:-1])  # drop classifier

    backbone.eval().to(device)

    tfm = get_transforms(img_size, train=False)

    ds = MelanomaDataset(df, img_dir, tfm)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    embeddings, labels_all = [], []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)

            with torch.cuda.amp.autocast():

                feats = model.forward_features(imgs)

                feats = model.global_pool(feats)

            embeddings.extend(feats.cpu().numpy())

            labels_all.extend(labels.numpy())

    return np.array(embeddings), np.array(labels_all)



def train_head_on_embeddings(embeddings, labels, in_features, device, epochs=10, batch_size=256):

    """Train just the classifier head on SMOTE-balanced embeddings."""

    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, 1)).to(device)

    optimizer = AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)

    criterion = FocalLoss(0.25, 2.0)

    ds = TensorDataset(torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    head.train()

    for epoch in range(epochs):

        total_loss = 0

        for feats, lbls in loader:

            feats, lbls = feats.to(device), lbls.to(device)

            optimizer.zero_grad()

            loss = criterion(head(feats).squeeze(1), lbls)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"  Head epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")

    return head



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



def main():

    IMG_SIZE = 384; EPOCHS = 20; BATCH_SIZE = 16; LR = 2e-4; PATIENCE = 7; WORKERS = 0



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



    # ── Step 1: extract embeddings ────────────────────────────────────────────

    print("\n--- Step 1: Extracting embeddings ---")

    model = build_model().to(device)

    embeddings, labels = extract_embeddings(model, train_df, IMG_DIR, IMG_SIZE, device)

    print(f"  Embeddings shape: {embeddings.shape}  |  Melanoma: {labels.sum():.0f}")



    # ── Step 2: SMOTE in embedding space ──────────────────────────────────────

    print("\n--- Step 2: Applying SMOTE ---")

    smote = SMOTE(random_state=SEED, k_neighbors=5)

    emb_balanced, lbl_balanced = smote.fit_resample(embeddings, labels.astype(int))

    print(f"  After SMOTE: {emb_balanced.shape[0]} samples  |  Melanoma: {lbl_balanced.sum()}")



    # ── Step 3: train head on balanced embeddings ─────────────────────────────

    print("\n--- Step 3: Training classifier head on SMOTE embeddings ---")

    head = train_head_on_embeddings(emb_balanced, lbl_balanced, model.num_features, device)

    # transfer head weights back into model

    # check for NaN in head weights before transferring
    if not torch.isnan(head[1].weight.data).any():
        model.classifier[1].weight.data = head[1].weight.data
        model.classifier[1].bias.data   = head[1].bias.data
    else:
        print("  WARNING: NaN in head weights, skipping transfer")



    # ── Step 4: fine-tune full model end-to-end ───────────────────────────────

    print("\n--- Step 4: Fine-tuning full model ---")

    train_tfm = get_transforms(IMG_SIZE, train=True)

    train_ds  = MelanomaDataset(train_df, IMG_DIR, train_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)

    criterion = FocalLoss(0.25, 2.0)

    optimizer = AdamW(model.parameters(), lr=LR / 5, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, max_lr=LR / 5, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1)

    scaler = torch.cuda.amp.GradScaler()



    log_rows = []

    best_auc = 0.0

    ckpt_path = os.path.join(CKPT_DIR, "smote_best.pt")

    patience_counter = 0



    for epoch in range(1, EPOCHS + 1):

        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)

        auc  = validate(model, val_loader, device)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  val_auc={auc:.4f}")

        log_rows.append({"epoch": epoch, "loss": loss, "val_auc": auc})

        if auc > best_auc:

            best_auc = auc; patience_counter = 0

            torch.save({"model_state": model.state_dict(), "auc": auc}, ckpt_path)

            print(f"  New best: {best_auc:.4f}")

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(f"Early stopping at epoch {epoch}"); break



    pd.DataFrame(log_rows).to_csv(os.path.join(RESULTS_DIR, "smote_log.csv"), index=False)

    print(f"\nBest Val AUC: {best_auc:.4f}")



if __name__ == "__main__":

    main()
