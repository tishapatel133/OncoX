"""

Onco-GPT-X Module 4 Option 2: Fine-tune Pretrained Diffusion Model

Uses HuggingFace diffusers pretrained DDPM and fine-tunes on HAM10000.

Produces higher quality images than training from scratch.

"""



import sys, time, argparse

import numpy as np

import torch

import torch.nn as nn

from torch.optim import AdamW

from pathlib import Path

from tqdm import tqdm

import torchvision.utils as vutils

from diffusers import DDPMScheduler, UNet2DModel



sys.path.insert(0, str(Path(__file__).parent))

from ddpm_dataset import get_ddpm_dataloader, CLASS_NAMES, NUM_CLASSES



BASE = Path("/scratch/patel.tis/OncoX")

DATA = BASE / "data/raw/ham10000"





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument('--epochs', type=int, default=200)

    p.add_argument('--batch_size', type=int, default=8)

    p.add_argument('--img_size', type=int, default=128)

    p.add_argument('--lr', type=float, default=1e-4)

    p.add_argument('--save_every', type=int, default=50)

    p.add_argument('--sample_every', type=int, default=50)

    p.add_argument('--workers', type=int, default=4)

    return p.parse_args()





class ConditionalDDPM(nn.Module):
    """Wraps diffusers UNet2DModel with class conditioning."""
    def __init__(self, num_classes=7, img_size=128):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
            num_class_embeds=num_classes,
        )

    def forward(self, x, t, class_labels):
        return self.unet(x, t, class_labels=class_labels).sample



def train():

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    if device.type == 'cuda':

        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Config: {vars(args)}\n")



    loader = get_ddpm_dataloader(DATA, img_size=args.img_size,

                                  batch_size=args.batch_size,

                                  num_workers=args.workers)



    model = ConditionalDDPM(num_classes=NUM_CLASSES, img_size=args.img_size).to(device)

    np_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"Pretrained DDPM params: {np_m:.1f}M\n")



    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)



    ckpt_dir = BASE / "models/checkpoints/diffusion_pretrained"

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = BASE / "results/diffusion_pretrained/samples"

    sample_dir.mkdir(parents=True, exist_ok=True)

    log_path = BASE / "results/diffusion_pretrained/finetune_log.txt"

    log_path.parent.mkdir(parents=True, exist_ok=True)



    print("=" * 60)

    print("Fine-tuning Pretrained Diffusion Model")

    print("=" * 60)



    t0 = time.time()

    for epoch in range(1, args.epochs + 1):

        model.train()

        epoch_loss = []

        pbar = tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False)



        for imgs, labels in pbar:

            imgs = imgs.to(device)

            labels = labels.to(device)

            noise = torch.randn_like(imgs)

            t = torch.randint(0, 1000, (imgs.shape[0],), device=device).long()

            noisy = scheduler.add_noise(imgs, noise, t)

            pred = model(noisy, t, labels)

            loss = nn.functional.mse_loss(pred, noise)



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



        if epoch % args.sample_every == 0 or epoch == 1:

            print("  Generating samples...")

            model.eval()

            with torch.no_grad():

                x = torch.randn(NUM_CLASSES, 3, args.img_size, args.img_size).to(device)

                labels_sample = torch.arange(NUM_CLASSES).to(device)

                for i, ts in enumerate(scheduler.timesteps):

                    t_batch = torch.full((NUM_CLASSES,), ts, device=device).long()

                    pred = model(x, t_batch, labels_sample)

                    x = scheduler.step(pred, ts, x).prev_sample

                x = (x + 1) / 2

                x = x.clamp(0, 1)

            grid = vutils.make_grid(x, nrow=NUM_CLASSES, padding=2)

            vutils.save_image(grid, sample_dir / f"epoch_{epoch:04d}.png")



        if epoch % args.save_every == 0:

            torch.save({

                'epoch': epoch,

                'model_state': model.state_dict(),

                'loss': avg_loss,

            }, ckpt_dir / f"pretrained_epoch_{epoch:04d}.pth")

            print("  Checkpoint saved")



    torch.save({

        'epoch': args.epochs,

        'model_state': model.state_dict(),

        'loss': avg_loss,

    }, ckpt_dir / "pretrained_final.pth")



    total = (time.time() - t0) / 60

    print(f"\nFine-tuning complete! {total:.1f}min, final loss: {avg_loss:.4f}")





if __name__ == "__main__":

    train()
