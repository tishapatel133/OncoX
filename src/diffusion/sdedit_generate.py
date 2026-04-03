"""

Onco-GPT-X Module 4: SDEdit Counterfactual Generation

Given a real patient image, generates counterfactual versions

showing what the lesion would look like under different diagnoses.



SDEdit approach:

1. Take real image

2. Add noise to partially destroy it (controlled by strength parameter)

3. Denoise conditioned on target class

4. Result preserves structure but transforms appearance

"""



import sys, argparse, glob

import numpy as np

import torch

import torchvision.transforms as T

import torchvision.utils as vutils

import cv2

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image



sys.path.insert(0, str(Path(__file__).parent))

from ddpm_model import ConditionalUNet, GaussianDiffusion

from ddpm_dataset import CLASS_NAMES, CLASS_MAP, NUM_CLASSES



BASE = Path("/scratch/patel.tis/OncoX")





def load_and_preprocess(img_path, img_size=128):

    img = cv2.imread(str(img_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = img.copy()

    transform = T.Compose([

        T.ToPILImage(),

        T.Resize((img_size, img_size)),

        T.ToTensor(),

        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

    ])

    tensor = transform(img)

    return tensor, original





def sdedit_transform(model, diffusion, img_tensor, target_class, strength=0.5, device='cuda'):

    """

    SDEdit: Add noise to real image, then denoise with target class condition.

    strength: 0.0 = no change, 1.0 = generate from scratch

    Typical good range: 0.3-0.6

    """

    model.eval()

    img = img_tensor.unsqueeze(0).to(device)

    target = torch.tensor([target_class]).to(device)



    # Determine how many steps to noise (strength controls this)

    t_start = int(diffusion.timesteps * strength)



    # Forward: add noise up to t_start

    t = torch.tensor([t_start]).to(device)

    noisy, _ = diffusion.q_sample(img, t)



    # Reverse: denoise from t_start back to 0, conditioned on target class

    x = noisy

    for i in reversed(range(t_start)):

        t_batch = torch.tensor([i]).to(device)

        x = diffusion.p_sample(model, x, t_batch, target)



    x = (x + 1) / 2

    x = x.clamp(0, 1)

    return x.squeeze(0)





def generate_counterfactual_grid(model, diffusion, img_path, output_path,

                                  strengths=[0.3, 0.4, 0.5], device='cuda'):

    """

    Generate a comprehensive counterfactual grid for one patient image.

    Rows: different noise strengths

    Cols: original + each target class

    """

    img_tensor, original = load_and_preprocess(img_path)

    stem = Path(img_path).stem



    target_classes = [0, 1, 2, 3, 4, 5, 6]

    target_names = CLASS_NAMES



    n_rows = len(strengths) + 1

    n_cols = len(target_classes) + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))



    # Row 0: Original image repeated

    orig_display = cv2.resize(original, (128, 128))

    for j in range(n_cols):

        axes[0, j].imshow(orig_display)

        axes[0, j].axis('off')

        if j == 0:

            axes[0, j].set_title('Original', fontsize=10, fontweight='bold')

        else:

            axes[0, j].set_title(f'{target_names[j-1]}', fontsize=10)



    axes[0, 0].set_ylabel('Input', fontsize=10, fontweight='bold')



    # Rows 1+: SDEdit results at different strengths

    for i, s in enumerate(strengths):

        axes[i+1, 0].imshow(orig_display)

        axes[i+1, 0].axis('off')

        axes[i+1, 0].set_ylabel(f'Strength={s}', fontsize=10, fontweight='bold')



        for j, cls_idx in enumerate(target_classes):

            print(f"  Generating: strength={s}, class={target_names[cls_idx]}...")

            cf = sdedit_transform(model, diffusion, img_tensor, cls_idx,

                                   strength=s, device=device)

            cf_np = cf.permute(1, 2, 0).cpu().numpy()

            axes[i+1, j+1].imshow(np.clip(cf_np, 0, 1))

            axes[i+1, j+1].axis('off')



    plt.suptitle(f'Onco-GPT-X Counterfactual Analysis | {stem}\n'

                 f'Columns: target diagnosis | Rows: transformation strength',

                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close()

    print(f"  Saved: {output_path}")





def generate_clinical_comparison(model, diffusion, img_path, output_path,

                                  strength=0.4, device='cuda'):

    """

    Generate a clean clinical comparison: original vs top 4 counterfactuals.

    This is the version used in the integration pipeline.

    """

    img_tensor, original = load_and_preprocess(img_path)

    stem = Path(img_path).stem



    key_classes = [

        (0, 'Melanocytic Nevus (benign)'),

        (1, 'Melanoma (malignant)'),

        (2, 'Benign Keratosis'),

        (3, 'Basal Cell Carcinoma'),

    ]



    fig, axes = plt.subplots(1, 5, figsize=(20, 4))



    orig_display = cv2.resize(original, (128, 128))

    axes[0].imshow(orig_display)

    axes[0].set_title('Original\nPatient Image', fontsize=11, fontweight='bold')

    axes[0].axis('off')



    for i, (cls_idx, cls_label) in enumerate(key_classes):

        cf = sdedit_transform(model, diffusion, img_tensor, cls_idx,

                               strength=strength, device=device)

        cf_np = cf.permute(1, 2, 0).cpu().numpy()

        axes[i+1].imshow(np.clip(cf_np, 0, 1))

        axes[i+1].set_title(f'If {cls_label}', fontsize=10)

        axes[i+1].axis('off')



    plt.suptitle(f'Counterfactual Analysis: What would this lesion look like under different diagnoses?',

                 fontsize=13, fontweight='bold')

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    plt.close()





def main():

    p = argparse.ArgumentParser()

    p.add_argument('--n_samples', type=int, default=5)

    p.add_argument('--strength', type=float, default=0.4)

    p.add_argument('--img_size', type=int, default=128)

    args = p.parse_args()



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")



    # Load best DDPM checkpoint

    ckpt_dir = BASE / "models/checkpoints/diffusion"

    final = ckpt_dir / "ddpm_final.pth"

    if final.exists():

        ckpt_path = final

    else:

        ckpts = sorted(glob.glob(str(ckpt_dir / "ddpm_epoch_*.pth")))

        ckpt_path = ckpts[-1] if ckpts else None



    if ckpt_path is None:

        print("ERROR: No DDPM checkpoint found!")

        return



    print(f"Loading checkpoint: {ckpt_path}")

    model = ConditionalUNet(

        img_ch=3, base_ch=128, ch_mult=(1, 2, 4),

        num_classes=NUM_CLASSES, time_dim=256

    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state'])

    print(f"Loaded epoch {ckpt['epoch']}, loss {ckpt.get('loss', 'N/A')}")



    diffusion = GaussianDiffusion(timesteps=1000, device=device)



    # Get sample images

    import pandas as pd

    img_dir = Path(open(BASE / "data/metadata/img_dir.txt").read().strip())

    val_df = pd.read_csv(BASE / "data/metadata/val.csv")

    pos = val_df[val_df['target'] == 1].head(args.n_samples)

    neg = val_df[val_df['target'] == 0].head(args.n_samples)

    samples = pd.concat([pos, neg])



    out_dir = BASE / "results/diffusion/counterfactuals"

    out_dir.mkdir(parents=True, exist_ok=True)



    for _, row in samples.iterrows():

        path = img_dir / f"{row['image_name']}.jpg"

        if not path.exists():

            continue

        print(f"\nProcessing: {row['image_name']} (target={row['target']})")



        # Clinical comparison (clean, 1 row)

        generate_clinical_comparison(

            model, diffusion, str(path),

            out_dir / f"clinical_{row['image_name']}.png",

            strength=args.strength, device=device

        )



        # Full grid (all classes x multiple strengths)

        generate_counterfactual_grid(

            model, diffusion, str(path),

            out_dir / f"grid_{row['image_name']}.png",

            strengths=[0.3, 0.5, 0.7], device=device

        )



    print(f"\nAll counterfactuals saved to {out_dir}")





if __name__ == "__main__":

    main()
