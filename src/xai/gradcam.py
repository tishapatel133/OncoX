"""

Onco-GPT-X Module 3: Explainable AI

Grad-CAM visualizations for classification models.

Shows which regions of the image influenced the prediction.

"""



import sys

import numpy as np

import torch

import torch.nn.functional as F

import cv2

import matplotlib.pyplot as plt

from pathlib import Path

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2



sys.path.insert(0, str(Path(__file__).parent.parent / "classification"))

from cls_models import build_model



BASE = Path("/scratch/patel.tis/OncoX")





class GradCAM:

    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.gradients = None

        self.activations = None

        self._register_hooks()



    def _register_hooks(self):

        def forward_hook(module, input, output):

            self.activations = output.detach()



        def backward_hook(module, grad_in, grad_out):

            self.gradients = grad_out[0].detach()



        self.target_layer.register_forward_hook(forward_hook)

        self.target_layer.register_full_backward_hook(backward_hook)



    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = 0

        self.model.zero_grad()
        score = output[0, class_idx] if output.shape[1] > 1 else output[0, 0]
        score.backward()

        grads = self.gradients
        acts = self.activations

        # Handle transformer outputs (3D: B, tokens, dim) vs CNN (4D: B, C, H, W)
        if grads.dim() == 3:
            B, N, C = grads.shape
            h = w = int(N ** 0.5)
            if h * w != N:
                h = w = int(np.ceil(N ** 0.5))
                grads = F.pad(grads, (0, 0, 0, h * w - N))
                acts = F.pad(acts, (0, 0, 0, h * w - N))
            grads = grads.reshape(B, h, w, C).permute(0, 3, 1, 2)
            acts = acts.reshape(B, h, w, C).permute(0, 3, 1, 2)

        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, torch.sigmoid(output).item()




def get_target_layer(model, model_name):

    """Get the appropriate layer for Grad-CAM based on model type."""

    if 'efficientnet' in model_name:

        return model.backbone.conv_head if hasattr(model, 'backbone') else model.backbone.blocks[-1]

    elif 'swinv2' in model_name:

        return model.backbone.layers[-1].blocks[-1].norm2

    elif 'pvtv2' in model_name:

        return model.backbone.stages[-1].blocks[-1].norm2

    raise ValueError(f"No target layer defined for {model_name}")





def preprocess_image(img_path, img_size=224):

    """Load and preprocess a single image."""

    img = cv2.imread(str(img_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original = img.copy()



    transform = A.Compose([

        A.Resize(img_size, img_size),

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ToTensorV2()

    ])

    tensor = transform(image=img)['image'].unsqueeze(0)

    return tensor, original





def overlay_heatmap(img, cam, alpha=0.4):

    """Overlay Grad-CAM heatmap on original image."""

    img_resized = cv2.resize(img, (cam.shape[1], cam.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.float32(heatmap) * alpha + np.float32(img_resized) * (1 - alpha)

    overlay = np.uint8(np.clip(overlay, 0, 255))

    return overlay





def generate_gradcam_report(model_name, checkpoint_path, image_paths, output_dir, img_size=224):

    """Generate Grad-CAM visualizations for a list of images."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)



    # Load model

    model = build_model(model_name, num_classes=1, pretrained=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state'])

    model.eval()

    print(f"Loaded {model_name} (AUC: {ckpt.get('best_auc', 'N/A')})")



    target_layer = get_target_layer(model, model_name)

    gradcam = GradCAM(model, target_layer)



    for img_path in image_paths:

        img_path = Path(img_path)

        tensor, original = preprocess_image(img_path, img_size)

        tensor = tensor.to(device).requires_grad_(True)



        cam, pred = gradcam.generate(tensor)



        overlay = overlay_heatmap(original, cam)



        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.resize(original, (img_size, img_size)))

        axes[0].set_title('Original Image')

        axes[0].axis('off')



        axes[1].imshow(cam, cmap='jet')

        axes[1].set_title('Grad-CAM Heatmap')

        axes[1].axis('off')



        axes[2].imshow(overlay)

        axes[2].set_title(f'Overlay (pred: {pred:.3f})')

        axes[2].axis('off')



        plt.suptitle(f'{model_name} | {img_path.stem}', fontsize=14)

        plt.tight_layout()

        plt.savefig(output_dir / f"gradcam_{model_name}_{img_path.stem}.png", dpi=150, bbox_inches='tight')

        plt.close()

        print(f"  {img_path.stem}: pred={pred:.3f}")



    print(f"Saved {len(image_paths)} Grad-CAM visualizations to {output_dir}")





if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--model', type=str, default='efficientnet_cbam')

    p.add_argument('--n_samples', type=int, default=20)

    args = p.parse_args()



    mn = args.model

    isz = 256 if 'swinv2' in mn else 224

    ckpt = BASE / f"models/checkpoints/classification/best_{mn}.pth"



    # Use some validation images

    img_dir = Path(open(BASE / "data/metadata/img_dir.txt").read().strip())

    val_csv = BASE / "data/metadata/val.csv"

    import pandas as pd

    df = pd.read_csv(val_csv)



    # Get mix of positive and negative samples

    pos = df[df['target'] == 1].head(args.n_samples // 2)

    neg = df[df['target'] == 0].head(args.n_samples // 2)

    samples = pd.concat([pos, neg])

    paths = [img_dir / f"{r['image_name']}.jpg" for _, r in samples.iterrows()]



    out = BASE / f"results/xai/gradcam_{mn}"

    generate_gradcam_report(mn, ckpt, paths, out, img_size=isz)
