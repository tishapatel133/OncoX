import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import timm
import time

sys.path.append('/scratch/patel.tis/OncoX')
from src.data.dataset import get_dataloaders

BASE = Path("/scratch/patel.tis/OncoX")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=25)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--img_size',    type=int,   default=224)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--model',       type=str,   default='efficientnet_b3')
    p.add_argument('--pos_weight',  type=float, default=8.0)
    p.add_argument('--grad_clip',   type=float, default=1.0)
    p.add_argument('--num_workers', type=int,   default=4)
    return p.parse_args()


def safe_auc(labels, preds):
    """Compute AUC safely — returns 0.0 if only one class present."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    return roc_auc_score(labels, preds)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    model.train()
    losses, preds_all, labels_all = [], [], []

    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, targets in pbar:
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        out  = model(imgs).squeeze(1)
        loss = criterion(out, targets)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        preds_all.extend(torch.sigmoid(out).detach().cpu().numpy())
        labels_all.extend(targets.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    auc = safe_auc(labels_all, preds_all)
    return np.mean(losses), auc


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds_all, labels_all = [], [], []

    for imgs, targets in tqdm(loader, desc='Val', leave=False):
        imgs    = imgs.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        out  = model(imgs).squeeze(1)
        loss = criterion(out, targets)

        losses.append(loss.item())
        preds_all.extend(torch.sigmoid(out).detach().cpu().numpy())
        labels_all.extend(targets.cpu().numpy())

    auc = safe_auc(labels_all, preds_all)

    # Additional metrics at threshold 0.5
    preds_bin = (np.array(preds_all) > 0.5).astype(int)
    labels_np = np.array(labels_all).astype(int)
    acc = accuracy_score(labels_np, preds_bin)
    f1  = f1_score(labels_np, preds_bin, zero_division=0)

    return np.mean(losses), auc, acc, f1


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {vars(args)}\n")

    # ── Find image directory ──
    img_dir_file = BASE / "data/metadata/img_dir.txt"
    if img_dir_file.exists():
        img_dir = Path(img_dir_file.read_text().strip())
    else:
        # Fallback: try common locations
        for candidate in [
            BASE / "data/raw/isic2020/jpeg/train",
            BASE / "data/raw/isic2020/train",
        ]:
            if candidate.exists():
                img_dir = candidate
                break
        else:
            print("ERROR: Cannot find image directory!")
            sys.exit(1)
    print(f"Image dir: {img_dir}\n")

    # ── Data ──
    loaders = get_dataloaders(
        base_path   = BASE,
        img_dir     = img_dir,
        batch_size  = args.batch_size,
        img_size    = args.img_size,
        num_workers = args.num_workers,
    )

    if 'train' not in loaders or 'val' not in loaders:
        print("ERROR: Missing train or val dataloader!")
        sys.exit(1)

    # ── Model ──
    model = timm.create_model(
        args.model,
        pretrained  = True,
        num_classes = 1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} — {total_params/1e6:.1f}M params ({train_params/1e6:.1f}M trainable)\n")

    # ── Loss with moderate class weighting ──
    # pos_weight=8 is much more stable than 56 while still addressing imbalance
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer with weight decay ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── OneCycleLR: built-in warmup + cosine decay ──
    total_steps = len(loaders['train']) * args.epochs
    scheduler   = OneCycleLR(
        optimizer,
        max_lr       = args.lr,
        total_steps  = total_steps,
        pct_start    = 0.1,   # 10% warmup
        anneal_strategy = 'cos',
    )

    # ── Training loop ──
    best_auc = 0
    log_rows = []
    ckpt_dir = BASE / "models/checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    patience, patience_counter = 7, 0

    print("=" * 60)
    print("Starting training")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        tr_loss, tr_auc = train_epoch(
            model, loaders['train'], criterion, optimizer, scheduler, device, args.grad_clip
        )
        vl_loss, vl_auc, vl_acc, vl_f1 = val_epoch(
            model, loaders['val'], criterion, device
        )

        elapsed = time.time() - epoch_start
        total_elapsed = (time.time() - start_time) / 60

        row = {
            'epoch': epoch + 1,
            'tr_loss': round(tr_loss, 4),
            'tr_auc':  round(tr_auc, 4),
            'vl_loss': round(vl_loss, 4),
            'vl_auc':  round(vl_auc, 4),
            'vl_acc':  round(vl_acc, 4),
            'vl_f1':   round(vl_f1, 4),
            'lr':      round(optimizer.param_groups[0]['lr'], 6),
            'time_s':  round(elapsed, 1),
        }
        log_rows.append(row)

        print(f"\nEpoch {epoch+1}/{args.epochs}  ({elapsed:.0f}s, total {total_elapsed:.1f}min)")
        print(f"  Train → Loss: {tr_loss:.4f}  AUC: {tr_auc:.4f}")
        print(f"  Val   → Loss: {vl_loss:.4f}  AUC: {vl_auc:.4f}  Acc: {vl_acc:.4f}  F1: {vl_f1:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if vl_auc > best_auc:
            best_auc = vl_auc
            patience_counter = 0
            torch.save({
                'epoch':      epoch + 1,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'best_auc':    best_auc,
                'args':        vars(args),
            }, ckpt_dir / f"best_{args.model}.pth")
            print(f"  ✅ New best! AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")

        # Save logs every epoch
        pd.DataFrame(log_rows).to_csv(
            BASE / "results/metrics/training_log.csv", index=False
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch+1}")
            break

    total_time = (time.time() - start_time) / 60
    print(f"\n{'=' * 60}")
    print(f"Training complete! Best Val AUC: {best_auc:.4f}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Best model saved: {ckpt_dir / f'best_{args.model}.pth'}")
    print(f"Training log: {BASE / 'results/metrics/training_log.csv'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()