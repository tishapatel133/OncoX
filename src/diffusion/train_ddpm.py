"""

Onco-GPT-X Module 4: Train Conditional DDPM

Trains class-conditional diffusion model on HAM10000.

Generates counterfactual skin lesion images.

"""



import os, sys, time, argparse

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

    p.add_argument('--epochs', type=int, default=200)

    p.add_argument('--batch_size', type=int, default=16)

    p.add_argument('--img_size', type=int, default=128)

    p.add_argument('--lr', type=float, default=2e-4)

    p.add_argument('--timesteps', type=int, default=1000)

    p.add_argument('--save_every', type=int, default=25)

    p.add_argument('--sample_every', type=int, default=25)

    p.add_argument('--workers', type=int, default=4)

    return p.parse_args()





def train():

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    if device.type == 'cuda':

        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Config: {vars(args)}\n")



    # Data

    loader = get_ddpm_dataloader(DATA, img_size=args.img_size,

                                  batch_size=args.batch_size,

                                  num_workers=args.workers)



    # Model

    model = ConditionalUNet(

        img_ch=3, base_ch=128, ch_mult=(1, 2, 4),

        num_classes=NUM_CLASSES, time_dim=256

    ).to(device)

    np_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Model params: {np_m:.1f}M\n")



    # Diffusion

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)



    # Optimizer

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)



    # Directories

    ckpt_dir = BASE / "models/checkpoints/diffusion"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = BASE / "results/diffusion/samples"

    sample_dir.mkdir(parents=True, exist_ok=True)

    log_path = BASE / "results/diffusion/ddpm_log.txt"

    log_path.parent.mkdir(parents=True, exist_ok=True)



    # Fixed labels for sampling (one of each class)

    fixed_labels = torch.arange(NUM_CLASSES).to(device)



    print("=" * 60)

    print("Starting DDPM Training")

    print("=" * 60)



    t0 = time.time()

    for epoch in range(1, args.epochs + 1):

        model.train()

        epoch_loss = []

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)



        for imgs, labels in pbar:

            imgs = imgs.to(device)

            labels = labels.to(device)



            # Sample random timesteps

            t = torch.randint(0, args.timesteps, (imgs.shape[0],), device=device)



            # Forward diffusion

            noisy_imgs, noise = diffusion.q_sample(imgs, t)



            # Predict noise

            pred_noise = model(noisy_imgs, t, labels)



            # MSE loss

            loss = nn.functional.mse_loss(pred_noise, noise)



            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()



            epoch_loss.append(loss.item())

            pbar.set_postfix(loss=f"{loss.item():.4f}")



        avg_loss = np.mean(epoch_loss)

        elapsed = (time.time() - t0) / 60

        log_line = f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}min"

        print(log_line)



        with open(log_path, 'a') as f:

            f.write(log_line + '\n')



        # Save samples

        if epoch % args.sample_every == 0 or epoch == 1:

            print(f"  Generating samples...")

            samples = diffusion.sample(model, NUM_CLASSES, fixed_labels,

                                        img_size=args.img_size)

            grid = vutils.make_grid(samples, nrow=NUM_CLASSES, padding=2)

            vutils.save_image(grid, sample_dir / f"epoch_{epoch:04d}.png")

            print(f"  Saved samples to {sample_dir / f'epoch_{epoch:04d}.png'}")



        # Save checkpoint

        if epoch % args.save_every == 0:

            torch.save({

                'epoch': epoch,

                'model_state': model.state_dict(),

                'optimizer_state': optimizer.state_dict(),

                'loss': avg_loss,

            }, ckpt_dir / f"ddpm_epoch_{epoch:04d}.pth")

            print(f"  Checkpoint saved")



    # Save final model

    torch.save({

        'epoch': args.epochs,

        'model_state': model.state_dict(),

        'loss': avg_loss,

    }, ckpt_dir / "ddpm_final.pth")



    total = (time.time() - t0) / 60

    print(f"\n{'=' * 60}")

    print(f"DDPM Training Complete! {total:.1f} minutes")

    print(f"Final loss: {avg_loss:.4f}")

    print(f"Checkpoints: {ckpt_dir}")

    print(f"Samples: {sample_dir}")

    print(f"{'=' * 60}")





if __name__ == "__main__":

    train()
