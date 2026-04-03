import os, sys, time, argparse

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.optim import AdamW

from torch.optim.lr_scheduler import OneCycleLR

from pathlib import Path

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression

import pickle



sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).parent.parent))

from cls_models import build_model

from data.dataset import ISICDataset, get_transforms, get_dataloaders



BASE = Path("/scratch/patel.tis/OncoX")





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, required=True,

                   choices=['efficientnet_cbam', 'swinv2', 'pvtv2', 'meta'])

    p.add_argument('--epochs', type=int, default=25)

    p.add_argument('--batch_size', type=int, default=32)

    p.add_argument('--img_size', type=int, default=224)

    p.add_argument('--lr', type=float, default=3e-4)

    p.add_argument('--pos_weight', type=float, default=8.0)

    p.add_argument('--grad_clip', type=float, default=1.0)

    p.add_argument('--patience', type=int, default=7)

    p.add_argument('--workers', type=int, default=4)

    return p.parse_args()





def safe_auc(labels, preds):

    if len(np.unique(labels)) < 2:

        return 0.0

    return roc_auc_score(labels, preds)





def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, gc):

    model.train()

    losses, pa, la = [], [], []

    for imgs, tgts in tqdm(loader, desc='Train', leave=False):

        imgs = imgs.to(device, non_blocking=True)

        tgts = tgts.float().to(device, non_blocking=True)

        optimizer.zero_grad()

        out = model(imgs).squeeze(1)

        loss = criterion(out, tgts)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gc)

        optimizer.step()

        scheduler.step()

        losses.append(loss.item())

        pa.extend(torch.sigmoid(out).detach().cpu().numpy())

        la.extend(tgts.cpu().numpy())

    return np.mean(losses), safe_auc(la, pa)





@torch.no_grad()

def evaluate(model, loader, criterion, device):

    model.eval()

    losses, pa, la = [], [], []

    for imgs, tgts in tqdm(loader, desc='Val', leave=False):

        imgs = imgs.to(device, non_blocking=True)

        tgts = tgts.float().to(device, non_blocking=True)

        out = model(imgs).squeeze(1)

        loss = criterion(out, tgts)

        losses.append(loss.item())

        pa.extend(torch.sigmoid(out).detach().cpu().numpy())

        la.extend(tgts.cpu().numpy())

    auc = safe_auc(la, pa)

    pb = (np.array(pa) > 0.5).astype(int)

    ln = np.array(la).astype(int)

    acc = accuracy_score(ln, pb)

    f1 = f1_score(ln, pb, zero_division=0)

    return np.mean(losses), auc, acc, f1, np.array(pa), np.array(la)





@torch.no_grad()

def collect_preds(model, loader, device):

    model.eval()

    pa, la = [], []

    for imgs, tgts in loader:

        imgs = imgs.to(device, non_blocking=True)

        out = model(imgs).squeeze(1)

        pa.extend(torch.sigmoid(out).cpu().numpy())

        la.extend(tgts.numpy())

    return np.array(pa), np.array(la)





def get_loaders(args, img_size):

    img_dir_file = BASE / "data/metadata/img_dir.txt"

    if img_dir_file.exists():

        img_dir = Path(img_dir_file.read_text().strip())

    else:

        img_dir = BASE / "data/raw/isic2020/300x300/train"

    print(f"Image dir: {img_dir}")

    return get_dataloaders(

        base_path=BASE, img_dir=img_dir,

        batch_size=args.batch_size, img_size=img_size,

        num_workers=args.workers

    )





def train_single(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    if device.type == 'cuda':

        print(f"GPU: {torch.cuda.get_device_name(0)}")



    mn = args.model

    lr = args.lr

    isz = args.img_size

    if mn == 'swinv2':

        lr = min(lr, 1e-4)

        isz = 256

    elif mn == 'pvtv2':

        lr = min(lr, 1e-4)



    print(f"\nTraining: {mn}, lr={lr}, img={isz}")



    loaders = get_loaders(args, isz)

    model = build_model(mn, num_classes=1, pretrained=True).to(device)

    np_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Params: {np_m:.1f}M\n")



    pw = torch.tensor([args.pos_weight]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ts = len(loaders['train']) * args.epochs

    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=ts,

                           pct_start=0.1, anneal_strategy='cos')



    best_auc, pc = 0, 0

    rows = []

    ckpt = BASE / f"models/checkpoints/classification/best_{mn}.pth"

    ckpt.parent.mkdir(parents=True, exist_ok=True)

    lp = BASE / f"results/classification/{mn}_log.csv"

    lp.parent.mkdir(parents=True, exist_ok=True)



    t0 = time.time()

    for ep in range(args.epochs):

        et = time.time()

        tl, ta = train_one_epoch(model, loaders['train'], criterion,

                                 optimizer, scheduler, device, args.grad_clip)

        vl, va, vc, vf, _, _ = evaluate(model, loaders['val'], criterion, device)

        el = time.time() - et

        tm = (time.time() - t0) / 60



        rows.append(dict(epoch=ep+1, tr_loss=round(tl,4), tr_auc=round(ta,4),

                         vl_loss=round(vl,4), vl_auc=round(va,4),

                         vl_acc=round(vc,4), vl_f1=round(vf,4),

                         lr=round(optimizer.param_groups[0]['lr'],6),

                         time_s=round(el,1)))



        print(f"\nEpoch {ep+1}/{args.epochs} ({el:.0f}s, {tm:.1f}min)")

        print(f"  Train -> Loss: {tl:.4f}  AUC: {ta:.4f}")

        print(f"  Val   -> Loss: {vl:.4f}  AUC: {va:.4f}  Acc: {vc:.4f}  F1: {vf:.4f}")



        if va > best_auc:

            best_auc = va

            pc = 0

            torch.save({'epoch': ep+1, 'model_state': model.state_dict(),

                        'best_auc': best_auc, 'model_name': mn}, ckpt)

            print(f"  New best! AUC: {best_auc:.4f}")

        else:

            pc += 1

            print(f"  No improvement ({pc}/{args.patience})")



        pd.DataFrame(rows).to_csv(lp, index=False)

        if pc >= args.patience:

            print(f"\nEarly stopping at epoch {ep+1}")

            break



    tt = (time.time() - t0) / 60

    print(f"\n{mn} done! Best AUC: {best_auc:.4f} in {tt:.1f}min")

    print(f"Checkpoint: {ckpt}")





def train_meta(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Training Meta-Ensemble...")



    cfgs = [('efficientnet_cbam', 224), ('swinv2', 256), ('pvtv2', 224)]

    vp_all, vl = [], None



    for mn, isz in cfgs:

        cp = BASE / f"models/checkpoints/classification/best_{mn}.pth"

        if not cp.exists():

            print(f"WARNING: {cp} not found, skipping")

            continue

        print(f"\nLoading {mn}...")

        model = build_model(mn, num_classes=1, pretrained=False).to(device)

        st = torch.load(cp, map_location=device, weights_only=False)

        model.load_state_dict(st['model_state'])

        print(f"  AUC from training: {st['best_auc']:.4f}")

        loaders = get_loaders(args, isz)

        p, l = collect_preds(model, loaders['val'], device)

        vp_all.append(p)

        if vl is None:

            vl = l

        del model

        torch.cuda.empty_cache()



    if not vp_all:

        print("ERROR: No predictions collected!")

        return



    X = np.column_stack(vp_all)

    print(f"\nMeta input: {X.shape}, pos rate: {np.mean(vl):.3f}")



    clf = LogisticRegression(class_weight='balanced', max_iter=1000)

    clf.fit(X, vl)

    mp = clf.predict_proba(X)[:, 1]

    ma = safe_auc(vl, mp)

    print(f"Meta-ensemble AUC: {ma:.4f}")

    print(f"Weights: {clf.coef_[0]}")



    ap = np.mean(np.column_stack(vp_all), axis=1)

    aa = safe_auc(vl, ap)

    print(f"Simple average AUC: {aa:.4f}")



    mp_path = BASE / "models/checkpoints/classification/meta_ensemble.pkl"

    with open(mp_path, 'wb') as f:

        pickle.dump(clf, f)

    print(f"Meta-ensemble saved: {mp_path}")





def main():

    args = parse_args()

    if args.model == 'meta':

        train_meta(args)

    else:

        train_single(args)





if __name__ == "__main__":

    main()
