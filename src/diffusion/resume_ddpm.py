"""

Onco-GPT-X Module 4: Resume DDPM Training

Loads last checkpoint and continues training for more epochs.

"""



import os, sys, time, argparse, glob

import numpy as np

import torch

import torch.nn as nn

from torch.optim import AdamW

from pathlib import Path

from tqdm import tqdm

import torchvision.utils as vutils



sys.path.insert(0, str(Path(__file__).parent))

from ddpm_dataset import get_ddpm_dataloader, CLASS_NAMES, NUM_CLASSES

from ddpm_model import ConditionalUNet, GaussianDiffusion



BASE = Path("/scratch/patel.tis/OncoX")

DATA = BASE / "data/raw/ham10000"





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument('--total_epochs', type=int, default=800)

    p.add_argument('--batch_size', type=int, default=16)

    p.add_argument('--img_size', type=int, default=128)

    p.add_argument('--lr', type=float, default=1e-4)

    p.add_argument('--timesteps', type=int, default=1000)

    p.add_argument('--save_every', type=int, default=50)

    p.add_argument('--sample_every', type=int, default=50)

    p.add_argument('--workers', type=int, default=4)

    return p.parse_args()





def find_latest_checkpoint():

    ckpt_dir = BASE / "models/checkpoints/diffusion"

    ckpts = sorted(glob.glob(str(ckpt_dir / "ddpm_epoch_*.pth")))

    if ckpts:

        return ckpts[-1]

    final = ckpt_dir / "ddpm_final.pth"

    if final.exists():

        return str(final)

    return None





def train():

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    if device.type == 'cuda':

        print(f"GPU: {torch.cuda.get_device_name(0)}")



    loader = get_ddpm_dataloader(DATA, img_size=args.img_size,

                                  batch_size=args.batch_size,

                                  num_workers=args.workers)



    model = ConditionalUNet(

        img_ch=3, base_ch=128, ch_mult=(1, 2, 4),

        num_classes=NUM_CLASSES, time_dim=256

    ).to(device)



    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)



    start_epoch = 0

    ckpt_path = find_latest_checkpoint()

    if ckpt_path:

        print(f"Resuming from: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt['model_state'])

        if 'optimizer_state' in ckpt:

            optimizer.load_state_dict(ckpt['optimizer_state'])

        start_epoch = ckpt['epoch']

        print(f"Resuming from epoch {start_epoch}, loss {ckpt.get('loss', 'N/A')}")

    else:

        print("No checkpoint found, training from scratch")



    np_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Model params: {np_m:.1f}M")

    print(f"Training epochs {start_epoch+1} to {args.total_epochs}\n")



    ckpt_dir = BASE / "models/checkpoints/diffusion"

    sample_dir = BASE / "results/diffusion/samples"

    log_path = BASE / "results/diffusion/ddpm_log.txt"



    fixed_labels = torch.arange(NUM_CLASSES).to(device)



    t0 = time.time()

    for epoch in range(start_epoch + 1, args.total_epochs + 1):

        model.train()

        epoch_loss = []

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.total_epochs}', leave=False)



        for imgs, labels in pbar:

            imgs = imgs.to(device)

            labels = labels.to(device)

            t = torch.randint(0, args.timesteps, (imgs.shape[0],), device=device)

            noisy_imgs, noise = diffusion.q_sample(imgs, t)

            pred_noise = model(noisy_imgs, t, labels)

            loss = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss.append(loss.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}")



        avg_loss = np.mean(epoch_loss)

        elapsed = (time.time() - t0) / 60

        log_line = f"Epoch {epoch}/{args.total_epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}min"

        print(log_line)

        with open(log_path, 'a') as f:

            f.write(log_line + '\n')



        if epoch % args.sample_every == 0:

            print(f"  Generating samples...")

            samples = diffusion.sample(model, NUM_CLASSES, fixed_labels,

                                        img_size=args.img_size)

            grid = vutils.make_grid(samples, nrow=NUM_CLASSES, padding=2)

            vutils.save_image(grid, sample_dir / f"epoch_{epoch:04d}.png")



        if epoch % args.save_every == 0:

            torch.save({

                'epoch': epoch,

                'model_state': model.state_dict(),

                'optimizer_state': optimizer.state_dict(),

                'loss': avg_loss,

            }, ckpt_dir / f"ddpm_epoch_{epoch:04d}.pth")

            print(f"  Checkpoint saved")



    torch.save({

        'epoch': args.total_epochs,

        'model_state': model.state_dict(),

        'loss': avg_loss,

    }, ckpt_dir / "ddpm_final.pth")



    total = (time.time() - t0) / 60

    print(f"\nTraining complete! {total:.1f} minutes, final loss: {avg_loss:.4f}")





if __name__ == "__main__":

    train()
