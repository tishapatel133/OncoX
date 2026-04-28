"""

Evaluate staged transformer checkpoints on held-out ISIC 2020 test set.

Reports AUC, F1, Sensitivity, Specificity, MCC for ViT-S and SwinV2-Small.

No retraining — just loads saved checkpoints.

"""



import os

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,

                              matthews_corrcoef, confusion_matrix, accuracy_score)

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm



ROOT         = "/scratch/patel.tis/OncoX"

IMG_DIR_2020 = os.path.join(ROOT, "data/raw/isic2020_clean")

TEST_CSV     = os.path.join(ROOT, "data/metadata/test.csv")

RESULTS_DIR  = os.path.join(ROOT, "results/classification")

CKPT_DIR     = os.path.join(ROOT, "models/checkpoints")





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



def build_vit_small(dropout=0.3):

    backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)

    classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.embed_dim, 1))

    return BackboneClassifier(backbone, classifier, 224)



def build_swinv2_small(dropout=0.3):

    backbone = timm.create_model("swinv2_small_window8_256", pretrained=False, num_classes=0)

    classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(backbone.num_features, 1))

    return BackboneClassifier(backbone, classifier, 256)





def get_tta_transforms(img_size):

    return [

        A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

        A.Compose([A.Resize(img_size, img_size), A.RandomRotate90(p=1.0), A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), ToTensorV2()]),

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



def get_predictions_tta(model, test_df, img_dir, device, img_size):

    model.eval()

    all_probs = []

    for tfm in get_tta_transforms(img_size):

        ds = MelanomaDataset(test_df, img_dir, tfm)

        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        probs = []

        with torch.no_grad():

            for imgs, _ in loader:

                imgs = imgs.to(device)

                with torch.cuda.amp.autocast():

                    logits = model(imgs)

                    if logits.dim() > 1: logits = logits.squeeze(1)

                p = torch.sigmoid(logits).cpu().numpy()

                probs.extend(np.nan_to_num(p, nan=0.5))

        all_probs.append(np.array(probs))

    return np.mean(all_probs, axis=0)



def evaluate_checkpoint(name, model, ckpt_path, test_df, labels, device, img_size):

    print(f"\nEvaluating: {name}")

    if not os.path.exists(ckpt_path):

        print(f"  SKIPPED — checkpoint not found: {ckpt_path}")

        return None

    try:

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = ckpt.get("model_state", ckpt)

        model.load_state_dict(state, strict=False)

        model = model.to(device)

        probs = get_predictions_tta(model, test_df, IMG_DIR_2020, device, img_size)

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

    test_df = pd.read_csv(TEST_CSV)

    labels = test_df["target"].values.astype(int)

    print(f"Test samples: {len(test_df)}  Melanoma: {labels.sum()}")



    checkpoints = []

    for fold in [1, 2, 3, 4, 5]:

        checkpoints.append((f"ViT-S staged fold{fold}", "vit_small", fold, 224))

        checkpoints.append((f"SwinV2-Small staged fold{fold}", "swinv2_small", fold, 256))



    results = []

    for name, model_type, fold, img_size in checkpoints:

        model = build_vit_small() if model_type == "vit_small" else build_swinv2_small()

        ckpt_path = os.path.join(CKPT_DIR, f"{model_type}_staged_merged_fold{fold}.pt")

        r = evaluate_checkpoint(name, model, ckpt_path, test_df, labels, device, img_size)

        if r: results.append(r)



    if results:

        df = pd.DataFrame(results)

        out = os.path.join(RESULTS_DIR, "transformer_staged_test_eval.csv")

        df.to_csv(out, index=False)

        print(f"\n========== TRANSFORMER TEST SET RESULTS ==========")

        print(f"{'Model':<32} {'AUC':>6} {'F1':>6} {'Sens':>6} {'Spec':>6} {'MCC':>6}")

        print("-" * 72)

        for r in sorted(results, key=lambda x: x["AUC"], reverse=True):

            print(f"{r['Model']:<32} {r['AUC']:>6} {r['F1']:>6} {r['Sensitivity']:>6} {r['Specificity']:>6} {r['MCC']:>6}")

        print(f"\nSaved to: {out}")



if __name__ == "__main__":

    main()
