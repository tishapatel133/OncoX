"""

Full evaluation of best model checkpoint.

Reports: AUC, Accuracy, F1, Precision, Recall/Sensitivity, Specificity, MCC

Tries multiple checkpoint key formats automatically.

"""



import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix, roc_curve)

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm



SEED = 42

ROOT        = "/scratch/patel.tis/OncoX"

IMG_DIR     = os.path.join(ROOT, "data/raw/isic2020_clean")

META_CSV    = os.path.join(ROOT, "data/metadata/train.csv")

RESULTS_DIR = os.path.join(ROOT, "results/classification")

CKPT_DIR    = os.path.join(ROOT, "models/checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)



def get_val_transforms(img_size=384):

    return A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),

        ToTensorV2()

    ])



def get_tta_transforms(img_size=384):

    return [

        A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.RandomRotate90(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.Transpose(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

    ]



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



def build_model():

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)

    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.num_features, 1))

    return model



def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        print(f"  Loaded via key: model_state")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print(f"  Loaded via key: model")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print(f"  Loaded via key: state_dict")
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        print(f"  Loaded as raw state dict")
    else:
        state_dict = ckpt
        print(f"  Loaded as raw state dict")
    # remap old classifier keys (classifier.weight -> classifier.1.weight)
    remapped = {}
    for k, v in state_dict.items():
        if k == "classifier.weight":
            remapped["classifier.1.weight"] = v
            print(f"  Remapped: classifier.weight -> classifier.1.weight")
        elif k == "classifier.bias":
            remapped["classifier.1.bias"] = v
            print(f"  Remapped: classifier.bias -> classifier.1.bias")
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=False)
    return model


def get_predictions(model, df, img_dir, device, use_tta=True):

    model.eval()

    if use_tta:

        transforms_list = get_tta_transforms()

    else:

        transforms_list = [get_val_transforms()]



    all_probs = []

    for tfm in transforms_list:

        ds = MelanomaDataset(df, img_dir, tfm)

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        probs = []

        with torch.no_grad():

            for imgs, _ in loader:

                imgs = imgs.to(device)

                with torch.cuda.amp.autocast():

                    logits = model(imgs).squeeze(1)

                p = torch.sigmoid(logits).cpu().numpy()

                p = np.nan_to_num(p, nan=0.5)

                probs.extend(p)

        all_probs.append(np.array(probs))



    return np.mean(all_probs, axis=0)



def find_best_threshold(labels, probs):

    """Find threshold that maximizes F1 score."""

    thresholds = np.arange(0.1, 0.9, 0.01)

    best_f1, best_thresh = 0, 0.5

    for t in thresholds:

        preds = (probs >= t).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)

        if f1 > best_f1:

            best_f1 = f1

            best_thresh = t

    return best_thresh, best_f1



def compute_metrics(labels, probs, threshold):

    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {

        "AUC":         round(roc_auc_score(labels, probs), 4),

        "Accuracy":    round(accuracy_score(labels, preds), 4),

        "F1":          round(f1_score(labels, preds, zero_division=0), 4),

        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),

        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),

        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),

        "MCC":         round(matthews_corrcoef(labels, preds), 4),

        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),

        "Threshold":   round(threshold, 2),

    }



def evaluate_checkpoint(name, ckpt_path, val_df, device):

    print(f"\n{'='*55}")

    print(f"Evaluating: {name}")

    print(f"Checkpoint: {ckpt_path}")

    print(f"{'='*55}")



    if not os.path.exists(ckpt_path):

        print(f"  SKIPPED — checkpoint not found")

        return None



    model = build_model().to(device)

    model = load_checkpoint(model, ckpt_path)

    model.to(device)



    labels = val_df["target"].values.astype(int)



    print("  Running inference with TTA...")

    probs_tta = get_predictions(model, val_df, IMG_DIR, device, use_tta=True)



    print("  Running inference without TTA...")

    probs_no_tta = get_predictions(model, val_df, IMG_DIR, device, use_tta=False)



    # find best threshold on no-TTA probs

    best_thresh, _ = find_best_threshold(labels, probs_no_tta)

    print(f"  Best threshold (F1-optimized): {best_thresh:.2f}")



    metrics_tta    = compute_metrics(labels, probs_tta,    best_thresh)

    metrics_no_tta = compute_metrics(labels, probs_no_tta, best_thresh)

    metrics_default = compute_metrics(labels, probs_no_tta, 0.5)



    print(f"\n  Results at threshold=0.50 (default):")

    for k, v in metrics_default.items():

        print(f"    {k}: {v}")



    print(f"\n  Results at threshold={best_thresh:.2f} (F1-optimized):")

    for k, v in metrics_no_tta.items():

        print(f"    {k}: {v}")



    print(f"\n  Results with TTA at threshold={best_thresh:.2f}:")

    for k, v in metrics_tta.items():

        print(f"    {k}: {v}")



    return {

        "name": name,

        "default_thresh": metrics_default,

        "best_thresh_no_tta": metrics_no_tta,

        "best_thresh_tta": metrics_tta,

    }



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    val_df = df.iloc[split[0][1]].reset_index(drop=True)

    print(f"Val samples: {len(val_df)}  |  Melanoma: {val_df['target'].sum():.0f}")



    checkpoints = [

        ("Baseline (0.9220)",     "best_efficientnet_b3.pth"),

        ("Two-stage (0.8812)",    "two_stage_best.pt"),

        ("OHEM (0.8777)",         "ohem_best.pt"),

        ("SMOTE (0.8599)",        "smote_best.pt"),

        ("Oversample",            "oversample_strong_aug_best.pt"),

    ]



    all_results = []

    for name, fname in checkpoints:

        ckpt_path = os.path.join(CKPT_DIR, fname)

        result = evaluate_checkpoint(name, ckpt_path, val_df, device)

        if result:

            all_results.append(result)



    # save summary table

    rows = []

    for r in all_results:

        for setting, metrics in [("default_thresh", r["default_thresh"]), ("best_thresh_no_tta", r["best_thresh_no_tta"]), ("best_thresh_tta", r["best_thresh_tta"])]:

            row = {"Model": r["name"], "Setting": setting}

            row.update(metrics)

            rows.append(row)



    summary_df = pd.DataFrame(rows)

    out_path = os.path.join(RESULTS_DIR, "full_evaluation.csv")

    summary_df.to_csv(out_path, index=False)

    print(f"\nFull results saved to: {out_path}")



    # print clean comparison table

    print("\n========== SUMMARY: AUC + Key Metrics (TTA, best threshold) ==========")

    print(f"{'Model':<28} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} {'Prec':>6} {'MCC':>6} {'Acc':>6}")

    print("-" * 76)

    for r in all_results:

        m = r["best_thresh_tta"]

        print(f"{r['name']:<28} {m['AUC']:>6} {m['F1']:>6} {m['Sensitivity']:>6} {m['Specificity']:>6} {m['Precision']:>6} {m['MCC']:>6} {m['Accuracy']:>6}")



if __name__ == "__main__":

    main()
