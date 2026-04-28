"""

Phase 1: Inference test — Seg → Mask → Classify

Tests both masking strategies (hard mask + crop) on existing checkpoints.

No retraining. Just checks if AUC improves before committing to full retrain.

"""

import os, sys

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from pathlib import Path

from PIL import Image

from tqdm import tqdm

from sklearn.metrics import roc_auc_score

import albumentations as A

from albumentations.pytorch import ToTensorV2

import timm

import cv2



BASE = Path("/scratch/patel.tis/OncoX")

sys.path.insert(0, str(BASE / "src"))

sys.path.insert(0, str(BASE / "src" / "classification"))

sys.path.insert(0, str(BASE / "src" / "segmentation"))



from seg_models import PVTv2UNet

from cls_models import build_model



# ── Config ──────────────────────────────────────────────

SEG_CKPT   = BASE / "models/checkpoints/segmentation/best_pvtv2_unet.pth"

CLS_CKPTS  = {
    "efficientnet_cbam": (BASE / "models/checkpoints/classification/best_efficientnet_cbam.pth", 224),
    "swinv2":            (BASE / "models/checkpoints/classification/best_swinv2.pth", 256),
}

IMG_DIR    = Path(BASE / "data/metadata/img_dir.txt").read_text().strip()

VAL_CSV    = BASE / "data/metadata/val.csv"

SEG_SIZE   = 256

CLS_SIZE   = 224

BATCH      = 16

WORKERS    = 4

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = BASE / "results/masked_inference"

OUT_DIR.mkdir(parents=True, exist_ok=True)



# ── Transforms ───────────────────────────────────────────

seg_tf = A.Compose([

    A.Resize(SEG_SIZE, SEG_SIZE),

    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

    ToTensorV2(),

])



cls_tf = A.Compose([

    A.Resize(CLS_SIZE, CLS_SIZE),

    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),

    ToTensorV2(),

])



# ── Dataset ──────────────────────────────────────────────

class ValDataset(Dataset):

    def __init__(self, csv_path, img_dir):

        self.df      = pd.read_csv(csv_path)

        self.img_dir = Path(img_dir)



    def __len__(self): return len(self.df)



    def __getitem__(self, idx):

        row   = self.df.iloc[idx]

        iname = row["image_name"]

        label = int(row["target"])

        path  = self.img_dir / f"{iname}.jpg"

        img   = np.array(Image.open(path).convert("RGB"))

        return img, label, iname   # return raw numpy so we can apply seg mask later



def collate_raw(batch):

    imgs, labels, names = zip(*batch)

    return list(imgs), torch.tensor(labels, dtype=torch.float32), list(names)



# ── Load segmentation model ───────────────────────────────

def load_seg():

    model = PVTv2UNet(pretrained=False)

    state = torch.load(SEG_CKPT, map_location=DEVICE, weights_only=False)

    model.load_state_dict(state["model_state"] if "model_state" in state else state)

    model.to(DEVICE).eval()

    print(f"Segmentation model loaded — PVTv2-UNet")

    return model



# ── Load classification model ─────────────────────────────

def load_cls(name):
    ckpt_path, img_size = CLS_CKPTS[name]
    model = build_model(name, num_classes=1, pretrained=False)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(DEVICE).eval()
    print(f"Classifier loaded — {name} (img_size={img_size})")
    return model, img_size



# ── Generate mask for one image ───────────────────────────

def get_mask(seg_model, img_np):

    """img_np: H x W x 3 uint8. Returns binary mask H x W (0/1 uint8)."""

    t = seg_tf(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        pred = torch.sigmoid(seg_model(t)).squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)

    # resize back to original image size

    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]),

                      interpolation=cv2.INTER_NEAREST)

    return mask



# ── Masking strategies ────────────────────────────────────

def apply_hard_mask(img_np, mask):

    """Multiply image by mask — background becomes black."""

    masked = img_np.copy()

    masked[mask == 0] = 0

    return masked



def apply_crop(img_np, mask):

    """Crop bounding box of mask region, resize to CLS_SIZE."""

    coords = cv2.findNonZero(mask)

    if coords is None:          # empty mask — fall back to full image

        return img_np

    x, y, w, h = cv2.boundingRect(coords)

    # add 10% padding around bounding box

    pad = int(0.1 * max(w, h))

    x1 = max(0, x - pad)

    y1 = max(0, y - pad)

    x2 = min(img_np.shape[1], x + w + pad)

    y2 = min(img_np.shape[0], y + h + pad)

    return img_np[y1:y2, x1:x2]



# ── Run inference for one strategy ───────────────────────

def run_inference(cls_model, seg_model, loader, strategy, tf=None):
    if tf is None:
        tf = cls_tf
    """strategy: 'original' | 'hard_mask' | 'crop'"""

    all_preds, all_labels = [], []



    for imgs_np, labels, _ in tqdm(loader, desc=f"  [{strategy}]"):

        batch_tensors = []

        for img_np in imgs_np:

            if strategy == "original":

                processed = img_np

            else:

                mask = get_mask(seg_model, img_np)

                if strategy == "hard_mask":

                    processed = apply_hard_mask(img_np, mask)

                else:  # crop

                    processed = apply_crop(img_np, mask)



            t = cls_tf(image=processed)["image"]

            batch_tensors.append(t)



        batch = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad():

            preds = torch.sigmoid(cls_model(batch)).squeeze(-1).cpu().numpy()



        all_preds.extend(preds.tolist())

        all_labels.extend(labels.numpy().tolist())



    auc = roc_auc_score(all_labels, all_preds)

    return auc



# ── Main ─────────────────────────────────────────────────

def main():

    print("="*55)

    print("Phase 1: Seg → Mask → Classify Inference Test")

    print(f"Device: {DEVICE}")

    print("="*55)



    dataset = ValDataset(VAL_CSV, IMG_DIR)

    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=False,

                         num_workers=WORKERS, collate_fn=collate_raw)



    seg_model = load_seg()



    results = {}

    for cls_name in CLS_CKPTS:

        print(f"\n── Classifier: {cls_name} ──")

        cls_model, img_size = load_cls(cls_name)
        cls_tf_local = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
        auc_orig = run_inference(cls_model, seg_model, loader, "original", cls_tf_local)
        auc_hard = run_inference(cls_model, seg_model, loader, "hard_mask", cls_tf_local)
        auc_crop = run_inference(cls_model, seg_model, loader, "crop", cls_tf_local)



        results[cls_name] = {

            "original":  auc_orig,

            "hard_mask": auc_hard,

            "crop":      auc_crop,

        }



        print(f"\n  Results for {cls_name}:")

        print(f"    Original (no mask)  : AUC {auc_orig:.4f}")

        print(f"    Hard mask           : AUC {auc_hard:.4f}  ({auc_hard-auc_orig:+.4f})")

        print(f"    Crop to lesion      : AUC {auc_crop:.4f}  ({auc_crop-auc_orig:+.4f})")



    # Summary

    print("\n" + "="*55)

    print("SUMMARY — Val AUC Comparison")

    print("="*55)

    print(f"{'Model':<25} {'Original':>10} {'HardMask':>10} {'Crop':>10}")

    print("-"*55)

    for name, r in results.items():

        print(f"{name:<25} {r['original']:>10.4f} {r['hard_mask']:>10.4f} {r['crop']:>10.4f}")



    # Save

    df = pd.DataFrame(results).T

    df.to_csv(OUT_DIR / "inference_comparison.csv")

    print(f"\nSaved to {OUT_DIR}/inference_comparison.csv")



    # Recommendation

    print("\n── Recommendation ──")

    best_auc, best_combo = 0, ""

    for name, r in results.items():

        for strategy, auc in r.items():

            if auc > best_auc:

                best_auc = auc

                best_combo = f"{name} + {strategy}"

    print(f"  Best combo : {best_combo}  (AUC {best_auc:.4f})")

    if best_auc > max(r["original"] for r in results.values()):

        print("  >> Masking IMPROVES performance — proceed to Phase 2 retraining.")

    else:

        print("  >> Masking did NOT improve performance on existing checkpoints.")

        print("     Retraining from scratch on masked images may still help.")



if __name__ == "__main__":

    main()
