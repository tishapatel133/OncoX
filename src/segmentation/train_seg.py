"""

Onco-GPT-X Module 2: Segmentation Training

Benchmarks multiple architectures on ISIC 2018 Task 1.

"""



import os, sys, time, argparse

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

from pathlib import Path

from tqdm import tqdm



sys.path.insert(0, str(Path(__file__).parent))

from seg_dataset import get_seg_dataloaders

from seg_models import build_seg_model



BASE = Path("/scratch/patel.tis/OncoX")

DATA = BASE / "data/raw/isic2018_seg"





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, required=True,

                   choices=['unet_efficientnet', 'pvtv2_unet', 'swinv2_unet'])

    p.add_argument('--epochs', type=int, default=100)

    p.add_argument('--batch_size', type=int, default=8)

    p.add_argument('--img_size', type=int, default=256)

    p.add_argument('--lr', type=float, default=1e-4)

    p.add_argument('--grad_clip', type=float, default=1.0)

    p.add_argument('--patience', type=int, default=15)

    p.add_argument('--workers', type=int, default=4)

    return p.parse_args()





class DiceBCELoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()



    def forward(self, pred, target):

        bce = self.bce(pred, target)

        pred_sig = torch.sigmoid(pred)

        flat_p = pred_sig.view(-1)

        flat_t = target.view(-1)

        inter = (flat_p * flat_t).sum()

        dice = 1 - (2.0 * inter + self.smooth) / (flat_p.sum() + flat_t.sum() + self.smooth)

        return bce + dice





def compute_metrics(pred, target, threshold=0.5):

    pred_bin = (torch.sigmoid(pred) > threshold).float()

    flat_p = pred_bin.view(-1)

    flat_t = target.view(-1)

    inter = (flat_p * flat_t).sum().item()

    union = flat_p.sum().item() + flat_t.sum().item() - inter

    iou = inter / (union + 1e-8)

    dice = (2.0 * inter) / (flat_p.sum().item() + flat_t.sum().item() + 1e-8)

    return iou, dice





def train_one_epoch(model, loader, criterion, optimizer, device, gc):

    model.train()

    losses, ious, dices = [], [], []

    for imgs, msks in tqdm(loader, desc='Train', leave=False):

        imgs = imgs.to(device, non_blocking=True)

        msks = msks.to(device, non_blocking=True)

        optimizer.zero_grad()

        out = model(imgs)

        loss = criterion(out, msks)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gc)

        optimizer.step()

        losses.append(loss.item())

        iou, dice = compute_metrics(out.detach(), msks)

        ious.append(iou)

        dices.append(dice)

    return np.mean(losses), np.mean(ious), np.mean(dices)





@torch.no_grad()

def evaluate(model, loader, criterion, device):

    model.eval()

    losses, ious, dices = [], [], []

    for imgs, msks in tqdm(loader, desc='Val', leave=False):

        imgs = imgs.to(device, non_blocking=True)

        msks = msks.to(device, non_blocking=True)

        out = model(imgs)

        loss = criterion(out, msks)

        losses.append(loss.item())

        iou, dice = compute_metrics(out, msks)

        ious.append(iou)

        dices.append(dice)

    return np.mean(losses), np.mean(ious), np.mean(dices)





def main():

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    if device.type == 'cuda':

        print(f"GPU: {torch.cuda.get_device_name(0)}")



    mn = args.model

    lr = args.lr

    isz = args.img_size

    if mn == 'swinv2_unet':

        isz = 256

    print(f"\nModel: {mn}, lr={lr}, img_size={isz}, epochs={args.epochs}")



    loaders = get_seg_dataloaders(DATA, img_size=isz, batch_size=args.batch_size,

                                  num_workers=args.workers)

    model = build_seg_model(mn, num_classes=1, pretrained=True).to(device)

    np_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Params: {np_m:.1f}M\n")



    criterion = DiceBCELoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)



    best_dice, pc = 0, 0

    rows = []

    ckpt = BASE / f"models/checkpoints/segmentation/best_{mn}.pth"

    ckpt.parent.mkdir(parents=True, exist_ok=True)

    lp = BASE / f"results/segmentation/{mn}_log.csv"

    lp.parent.mkdir(parents=True, exist_ok=True)



    t0 = time.time()

    for ep in range(args.epochs):

        et = time.time()

        tl, ti, td = train_one_epoch(model, loaders['train'], criterion,

                                      optimizer, device, args.grad_clip)

        vl, vi, vd = evaluate(model, loaders['val'], criterion, device)

        scheduler.step()

        el = time.time() - et

        tm = (time.time() - t0) / 60



        rows.append(dict(epoch=ep+1, tr_loss=round(tl,4), tr_iou=round(ti,4),

                         tr_dice=round(td,4), vl_loss=round(vl,4),

                         vl_iou=round(vi,4), vl_dice=round(vd,4),

                         lr=round(optimizer.param_groups[0]['lr'],6),

                         time_s=round(el,1)))



        print(f"\nEpoch {ep+1}/{args.epochs} ({el:.0f}s, {tm:.1f}min)")

        print(f"  Train -> Loss: {tl:.4f}  IoU: {ti:.4f}  Dice: {td:.4f}")

        print(f"  Val   -> Loss: {vl:.4f}  IoU: {vi:.4f}  Dice: {vd:.4f}")



        if vd > best_dice:

            best_dice = vd

            pc = 0

            torch.save({'epoch': ep+1, 'model_state': model.state_dict(),

                        'best_dice': best_dice, 'best_iou': vi,

                        'model_name': mn}, ckpt)

            print(f"  New best! Dice: {best_dice:.4f} IoU: {vi:.4f}")

        else:

            pc += 1

            print(f"  No improvement ({pc}/{args.patience})")



        pd.DataFrame(rows).to_csv(lp, index=False)

        if pc >= args.patience:

            print(f"\nEarly stopping at epoch {ep+1}")

            break



    tt = (time.time() - t0) / 60

    print(f"\n{mn} done! Best Dice: {best_dice:.4f} in {tt:.1f}min")

    print(f"Checkpoint: {ckpt}")





if __name__ == "__main__":

    main()
