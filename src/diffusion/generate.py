"""

Onco-GPT-X Module 4: Generate Counterfactual Images

Load trained DDPM and generate images for each class.

"""



import sys, argparse

import torch

import torchvision.utils as vutils

from pathlib import Path



sys.path.insert(0, str(Path(__file__).parent))

from ddpm_model import ConditionalUNet, GaussianDiffusion

from ddpm_dataset import CLASS_NAMES, NUM_CLASSES



BASE = Path("/scratch/patel.tis/OncoX")





def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument('--checkpoint', type=str, default=str(BASE / "models/checkpoints/diffusion/ddpm_final.pth"))

    p.add_argument('--n_per_class', type=int, default=8)

    p.add_argument('--img_size', type=int, default=128)

    p.add_argument('--timesteps', type=int, default=1000)

    p.add_argument('--output_dir', type=str, default=str(BASE / "results/diffusion/generated"))

    return p.parse_args()





def main():

    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")



    # Load model

    model = ConditionalUNet(

        img_ch=3, base_ch=128, ch_mult=(1, 2, 4),

        num_classes=NUM_CLASSES, time_dim=256

    ).to(device)



    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state'])

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")



    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    # Generate images for each class

    for cls_idx, cls_name in enumerate(CLASS_NAMES):

        print(f"\nGenerating {args.n_per_class} images for class: {cls_name} ({cls_idx})")

        labels = torch.full((args.n_per_class,), cls_idx, dtype=torch.long).to(device)

        samples = diffusion.sample(model, args.n_per_class, labels, img_size=args.img_size)



        # Save grid

        grid = vutils.make_grid(samples, nrow=4, padding=2)

        vutils.save_image(grid, out_dir / f"class_{cls_name}_grid.png")



        # Save individual images

        cls_dir = out_dir / cls_name

        cls_dir.mkdir(exist_ok=True)

        for i in range(args.n_per_class):

            vutils.save_image(samples[i], cls_dir / f"{cls_name}_{i:03d}.png")



        print(f"  Saved to {out_dir / cls_name}")



    # Generate counterfactual grid: same noise, different classes

    print("\n>>> Generating counterfactual comparison grid...")

    torch.manual_seed(42)

    noise = torch.randn(1, 3, args.img_size, args.img_size).to(device)

    noise = noise.repeat(NUM_CLASSES, 1, 1, 1)

    labels = torch.arange(NUM_CLASSES).to(device)



    model.eval()

    x = noise.clone()

    for i in reversed(range(args.timesteps)):

        t = torch.full((NUM_CLASSES,), i, dtype=torch.long, device=device)

        x = diffusion.p_sample(model, x, t, labels)

    x = (x + 1) / 2

    x = x.clamp(0, 1)



    grid = vutils.make_grid(x, nrow=NUM_CLASSES, padding=2)

    vutils.save_image(grid, out_dir / "counterfactual_same_noise_all_classes.png")

    print(f"  Counterfactual grid saved!")

    print(f"\nAll generation complete! Output: {out_dir}")





if __name__ == "__main__":

    main()
