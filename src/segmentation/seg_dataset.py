"""

Onco-GPT-X Module 2: Segmentation Dataset

ISIC 2018 Task 1 — Skin Lesion Segmentation

"""



import cv2

import numpy as np

from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import albumentations as A

from albumentations.pytorch import ToTensorV2





class ISIC2018SegDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None):

        self.image_paths = image_paths

        self.mask_paths = mask_paths

        self.transform = transform



    def __len__(self):

        return len(self.image_paths)



    def __getitem__(self, idx):

        img = cv2.imread(str(self.image_paths[idx]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        msk = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        msk = (msk > 127).astype(np.float32)



        if self.transform:

            aug = self.transform(image=img, mask=msk)

            img = aug['image']

            msk = aug['mask'].unsqueeze(0)



        return img, msk





def get_seg_transforms(phase, img_size=256):

    if phase == 'train':

        return A.Compose([

            A.Resize(img_size, img_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.ElasticTransform(alpha=120, sigma=6, p=0.3),

            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ToTensorV2()

        ])

    else:

        return A.Compose([

            A.Resize(img_size, img_size),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ToTensorV2()

        ])





def get_seg_dataloaders(data_dir, img_size=256, batch_size=8, num_workers=4, val_ratio=0.15, seed=42):

    data_dir = Path(data_dir)

    img_dir = data_dir / "ISIC2018_Task1-2_Training_Input"

    msk_dir = data_dir / "ISIC2018_Task1_Training_GroundTruth"



    all_masks = sorted(msk_dir.glob("*_segmentation.png"))

    all_images = []

    valid_masks = []



    for mp in all_masks:

        name = mp.stem.replace("_segmentation", "")

        ip = img_dir / f"{name}.jpg"

        if ip.exists():

            all_images.append(ip)

            valid_masks.append(mp)



    print(f"Found {len(all_images)} matched image-mask pairs")



    train_imgs, val_imgs, train_msks, val_msks = train_test_split(

        all_images, valid_masks, test_size=val_ratio, random_state=seed

    )



    print(f"  Train: {len(train_imgs)}")

    print(f"  Val:   {len(val_imgs)}")



    train_ds = ISIC2018SegDataset(train_imgs, train_msks, get_seg_transforms('train', img_size))

    val_ds = ISIC2018SegDataset(val_imgs, val_msks, get_seg_transforms('val', img_size))



    loaders = {

        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True,

                            num_workers=num_workers, pin_memory=True, drop_last=True),

        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,

                          num_workers=num_workers, pin_memory=True),

    }

    return loaders
