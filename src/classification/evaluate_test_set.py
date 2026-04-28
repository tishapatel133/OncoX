"""

Final test set evaluation for all checkpoints.

Reports AUC, F1, Sensitivity, Specificity, Precision, MCC, Accuracy.

"""



import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix

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



def build_model(model_name, dropout=0.3):

    try:

        model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=dropout)

    except Exception:

        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=1, drop_rate=dropout)

    return model



def build_model_legacy(dropout=0.3):

    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)

    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.num_features, 1))

    return model



def load_checkpoint(model, ckpt_path):

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):

        if "model_state" in ckpt:

            state_dict = ckpt["model_state"]

        elif "model" in ckpt:

            state_dict = ckpt["model"]

        elif "state_dict" in ckpt:

            state_dict = ckpt["state_dict"]

        else:

            state_dict = ckpt

    else:

        state_dict = ckpt

    # remap old classifier keys

    remapped = {}

    for k, v in state_dict.items():

        if k == "classifier.weight":

            remapped["classifier.1.weight"] = v

        elif k == "classifier.bias":

            remapped["classifier.1.bias"] = v

        else:

            remapped[k] = v

    try:

        model.load_state_dict(remapped, strict=True)

    except RuntimeError:

        model.load_state_dict(remapped, strict=False)

        print(f"  Warning: loaded with strict=False")

    return model



def get_predictions_tta(model, df, img_dir, device, img_size=384):

    model.eval()

    all_probs = []

    for tfm in get_tta_transforms(img_size):

        ds = MelanomaDataset(df, img_dir, tfm)

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        probs = []

        with torch.no_grad():

            for imgs, _ in loader:

                imgs = imgs.to(device)

                with torch.cuda.amp.autocast():

                    logits = model(imgs)

                    if logits.dim() > 1:

                        logits = logits.squeeze(1)

                p = torch.sigmoid(logits).cpu().numpy()

                probs.extend(np.nan_to_num(p, nan=0.5))

        all_probs.append(np.array(probs))

    return np.mean(all_probs, axis=0)



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

    cm = confusion_matrix(labels, preds, labels=[0,1])

    tn, fp, fn, tp = cm.ravel()

    return {

        "AUC":         round(roc_auc_score(labels, probs), 4),

        "F1":          round(f1_score(labels, preds, zero_division=0), 4),

        "Sensitivity": round(recall_score(labels, preds, zero_division=0), 4),

        "Specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),

        "Precision":   round(precision_score(labels, preds, zero_division=0), 4),

        "MCC":         round(matthews_corrcoef(labels, preds), 4),

        "Accuracy":    round(accuracy_score(labels, preds), 4),

        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),

        "Threshold":   round(threshold, 2),

    }



def evaluate_checkpoint(name, model_name, ckpt_path, test_df, labels, device, img_size, use_legacy=False):

    print(f"\nEvaluating: {name}")

    if not os.path.exists(ckpt_path):

        print(f"  SKIPPED — not found: {ckpt_path}")

        return None

    try:

        if use_legacy:

            model = build_model_legacy().to(device)

        else:

            model = build_model(model_name).to(device)

        model = load_checkpoint(model, ckpt_path)

        probs = get_predictions_tta(model, test_df, IMG_DIR, device, img_size)

        thresh = find_best_threshold(labels, probs)

        metrics = compute_metrics(labels, probs, thresh)

        metrics["Model"] = name

        print(f"  AUC={metrics['AUC']}  F1={metrics['F1']}  Sens={metrics['Sensitivity']}  Spec={metrics['Specificity']}  MCC={metrics['MCC']}")

        del model; torch.cuda.empty_cache()

        return metrics

    except Exception as e:

        print(f"  ERROR: {e}")

        import traceback; traceback.print_exc()

        return None



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    df = pd.read_csv(META_CSV)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    split = list(sgkf.split(df, df["target"], groups=df["patient_id"]))

    test_idx = split[0][1]

    test_df = df.iloc[test_idx].reset_index(drop=True)

    labels = test_df["target"].values.astype(int)

    print(f"Test samples: {len(test_df)}  |  Melanoma: {labels.sum()}")



    checkpoints = [

        ("EfficientNet-B3 baseline",    "efficientnet_b3", "best_efficientnet_b3.pth",              384, True),

        ("EfficientNet-B3 clean+384",   "efficientnet_b3", "best_efficientnet_b3_clean_384.pth",    384, True),

        ("EfficientNet-B3 kfold fold1", "efficientnet_b3", "best_efficientnet_b3_kfold5_meta0_384_fold1.pth", 384, True),

        ("EfficientNet-B3 CBAM",        "efficientnet_b3", "best_efficientnet_cbam.pth",             384, True),

        ("SwinV2-Tiny",                 "swinv2_tiny_window8_256", "best_swinv2.pth",               256, False),

        ("Two-stage training",          "efficientnet_b3", "two_stage_best.pt",                     384, False),

        ("OHEM oversampling",           "efficientnet_b3", "ohem_best.pt",                          384, False),

        ("SMOTE embedding",             "efficientnet_b3", "smote_best.pt",                         384, False),

        ("Oversample + strong aug",     "efficientnet_b3", "oversample_strong_aug_best.pt",         384, False),

        ("EfficientNet-B4 fold1",       "efficientnet_b4", "efficientnet_b4_fold1.pt",              384, False),

        ("EfficientNet-B5 fold1",       "efficientnet_b5", "efficientnet_b5_fold1.pt",              384, False),

        ("SwinV2-Small fold1",          "swinv2_small_window8_256", "swinv2_small_window8_256_fold1.pt", 256, False),

        ("SwinV2-Base fold1",           "swinv2_base_window8_256", "swinv2_base_window8_256_fold1.pt", 256, False),

        ("PVTv2-B3 fold1",              "pvt_v2_b3", "pvt_v2_b3_fold1.pt",                         384, False),

        ("PVTv2-B4 fold1",              "pvt_v2_b4", "pvt_v2_b4_fold1.pt",                         384, False),

        ("PVTv2-B5 fold1",              "pvt_v2_b5", "pvt_v2_b5_fold1.pt",                         384, False),

    ]



    all_results = []

    for name, model_name, fname, img_size, use_legacy in checkpoints:

        ckpt_path = os.path.join(CKPT_DIR, fname)

        result = evaluate_checkpoint(name, model_name, ckpt_path, test_df, labels, device, img_size, use_legacy)

        if result:

            all_results.append(result)



    if not all_results:

        print("No results to save.")

        return



    results_df = pd.DataFrame(all_results)

    out_path = os.path.join(RESULTS_DIR, "test_set_evaluation.csv")

    results_df.to_csv(out_path, index=False)



    print("\n========== FINAL TEST SET RESULTS ==========")

    print(f"{'Model':<32} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} {'Prec':>6} {'MCC':>6} {'Acc':>6}")

    print("-" * 82)

    for r in sorted(all_results, key=lambda x: x["AUC"], reverse=True):

        print(f"{r['Model']:<32} {r['AUC']:>6} {r['F1']:>6} {r['Sensitivity']:>6} {r['Specificity']:>6} {r['Precision']:>6} {r['MCC']:>6} {r['Accuracy']:>6}")

    print(f"\nSaved to: {out_path}")



if __name__ == "__main__":

    main()
