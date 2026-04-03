"""

Onco-GPT-X Module 5: End-to-End Integration Pipeline

Takes a dermoscopy image and runs the full pipeline:

  1. Classify (melanoma vs benign)

  2. Segment (lesion boundary)

  3. Explain (Grad-CAM heatmap)

  4. Generate counterfactual (DDPM)

  5. Produce integrated report

"""



import sys, argparse

import numpy as np

import torch

import torch.nn.functional as F

import cv2

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from pathlib import Path

import albumentations as A

from albumentations.pytorch import ToTensorV2



sys.path.insert(0, str(Path(__file__).parent.parent / "classification"))

sys.path.insert(0, str(Path(__file__).parent.parent / "segmentation"))

sys.path.insert(0, str(Path(__file__).parent.parent / "xai"))

sys.path.insert(0, str(Path(__file__).parent.parent / "diffusion"))



from cls_models import build_model as build_cls

from seg_models import build_seg_model

from gradcam import GradCAM, get_target_layer

from ddpm_model import ConditionalUNet, GaussianDiffusion

from ddpm_dataset import CLASS_NAMES, NUM_CLASSES



BASE = Path("/scratch/patel.tis/OncoX")





def load_image(path, img_size=224):

    img = cv2.imread(str(path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = img.copy()

    t = A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ToTensorV2()

    ])

    tensor = t(image=img)['image'].unsqueeze(0)

    return tensor, original





def run_classification(img_tensor, device):

    model = build_cls('efficientnet_cbam', num_classes=1, pretrained=False).to(device)

    ckpt = torch.load(BASE / "models/checkpoints/classification/best_efficientnet_cbam.pth",

                      map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state'])

    model.eval()

    with torch.no_grad():

        out = model(img_tensor.to(device))

        prob = torch.sigmoid(out).item()

    label = "MELANOMA" if prob > 0.5 else "BENIGN"

    confidence = prob if prob > 0.5 else 1 - prob

    return label, prob, confidence, model





def run_segmentation(img_path, device):

    img = cv2.imread(str(img_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    t = A.Compose([

        A.Resize(256, 256),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ToTensorV2()

    ])

    tensor = t(image=img)['image'].unsqueeze(0).to(device)



    model = build_seg_model('pvtv2_unet', num_classes=1, pretrained=False).to(device)

    ckpt = torch.load(BASE / "models/checkpoints/segmentation/best_pvtv2_unet.pth",

                      map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state'])

    model.eval()



    with torch.no_grad():

        pred = model(tensor)

        mask = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()



    area_pixels = mask.sum()

    area_pct = area_pixels / mask.size * 100

    perimeter = cv2.findContours((mask * 255).astype(np.uint8),

                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = perimeter[0]

    perim_len = sum(cv2.arcLength(c, True) for c in contours) if contours else 0



    return mask, area_pixels, area_pct, perim_len





def run_gradcam(img_tensor, model, device):

    target_layer = get_target_layer(model, 'efficientnet_cbam')

    gc = GradCAM(model, target_layer)

    tensor = img_tensor.to(device).requires_grad_(True)

    cam, _ = gc.generate(tensor)

    return cam





def run_counterfactual(device, img_size=128):

    model = ConditionalUNet(

        img_ch=3, base_ch=128, ch_mult=(1, 2, 4),

        num_classes=NUM_CLASSES, time_dim=256

    ).to(device)



    ckpt_path = BASE / "models/checkpoints/diffusion/ddpm_final.pth"

    if not ckpt_path.exists():

        ckpts = sorted((BASE / "models/checkpoints/diffusion").glob("ddpm_epoch_*.pth"))

        if ckpts:

            ckpt_path = ckpts[-1]

        else:

            return None



    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state'])

    diffusion = GaussianDiffusion(timesteps=1000, device=device)



    torch.manual_seed(42)

    labels = torch.tensor([0, 1, 2, 3]).to(device)

    names = [CLASS_NAMES[i] for i in [0, 1, 2, 3]]

    samples = diffusion.sample(model, 4, labels, img_size=img_size)

    return samples, names





def generate_report(img_path, output_dir, device):

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(img_path).stem



    print(f"\n{'='*60}")

    print(f"ONCO-GPT-X INTEGRATED ANALYSIS: {stem}")

    print(f"{'='*60}")



    # 1. Classification

    print("\n[1/4] Running classification...")

    tensor, original = load_image(img_path, 224)

    label, prob, conf, cls_model = run_classification(tensor, device)

    print(f"  Diagnosis: {label} (melanoma prob: {prob:.3f}, confidence: {conf:.1%})")



    # 2. Segmentation

    print("\n[2/4] Running segmentation...")

    mask, area_px, area_pct, perim = run_segmentation(img_path, device)

    print(f"  Lesion area: {area_px:.0f} pixels ({area_pct:.1f}% of image)")

    print(f"  Perimeter: {perim:.1f} pixels")



    # 3. Grad-CAM

    print("\n[3/4] Running Grad-CAM explainability...")

    cam = run_gradcam(tensor, cls_model, device)

    print(f"  Heatmap generated")



    # 4. Counterfactual

    print("\n[4/4] Generating counterfactual images...")

    cf_result = run_counterfactual(device)



    # Create visualization

    fig = plt.figure(figsize=(20, 12))



    # Row 1: Original + Classification + Segmentation

    ax1 = fig.add_subplot(2, 4, 1)

    ax1.imshow(cv2.resize(original, (224, 224)))

    ax1.set_title('Original Image', fontsize=11)

    ax1.axis('off')



    ax2 = fig.add_subplot(2, 4, 2)

    ax2.imshow(cv2.resize(original, (224, 224)))

    ax2.text(112, 200, f"{label}\n{prob:.3f}", ha='center', fontsize=14,

             color='red' if label == 'MELANOMA' else 'green', fontweight='bold',

             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_title('Classification', fontsize=11)

    ax2.axis('off')



    ax3 = fig.add_subplot(2, 4, 3)

    ax3.imshow(mask, cmap='gray')

    ax3.set_title(f'Segmentation\nArea: {area_pct:.1f}%', fontsize=11)

    ax3.axis('off')



    ax4 = fig.add_subplot(2, 4, 4)

    overlay = cv2.resize(original, (cam.shape[1], cam.shape[0]))

    hm = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

    blend = np.uint8(0.4 * hm + 0.6 * overlay)

    ax4.imshow(blend)

    ax4.set_title('Grad-CAM Explanation', fontsize=11)

    ax4.axis('off')



    # Row 2: Counterfactual images

    if cf_result is not None:

        samples, names = cf_result

        for i in range(4):

            ax = fig.add_subplot(2, 4, 5 + i)

            img_np = samples[i].permute(1, 2, 0).cpu().numpy()

            ax.imshow(np.clip(img_np, 0, 1))

            ax.set_title(f'Counterfactual:\n{names[i]}', fontsize=11)

            ax.axis('off')



    plt.suptitle(f'Onco-GPT-X Analysis Report | {stem}', fontsize=16, fontweight='bold')

    plt.tight_layout()

    plt.savefig(output_dir / f"report_{stem}.png", dpi=150, bbox_inches='tight')

    plt.close()



    # Text report

    report = f"""

ONCO-GPT-X CLINICAL ANALYSIS REPORT

====================================

Image: {stem}



1. CLASSIFICATION

   Diagnosis: {label}

   Melanoma probability: {prob:.3f}

   Confidence: {conf:.1%}



2. SEGMENTATION

   Lesion area: {area_px:.0f} pixels ({area_pct:.1f}% of image)

   Perimeter: {perim:.1f} pixels



3. EXPLAINABILITY

   Grad-CAM heatmap highlights regions driving the classification.

   Visual output saved.



4. COUNTERFACTUAL GENERATION

   Generated synthetic images showing alternative diagnoses

   for clinical comparison.

====================================

"""

    with open(output_dir / f"report_{stem}.txt", 'w') as f:

        f.write(report)



    print(f"\nReport saved: {output_dir / f'report_{stem}.png'}")

    print(report)





if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument('--n_samples', type=int, default=5)

    args = p.parse_args()



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")



    import pandas as pd

    img_dir = Path(open(BASE / "data/metadata/img_dir.txt").read().strip())

    val_df = pd.read_csv(BASE / "data/metadata/val.csv")



    pos = val_df[val_df['target'] == 1].head(args.n_samples)

    neg = val_df[val_df['target'] == 0].head(args.n_samples)

    samples = pd.concat([pos, neg])



    out = BASE / "results/integration"

    for _, row in samples.iterrows():

        path = img_dir / f"{row['image_name']}.jpg"

        if path.exists():

            generate_report(str(path), out, device)
