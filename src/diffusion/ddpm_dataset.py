"""

Onco-GPT-X Module 4: DDPM Dataset

HAM10000 with class-conditional labels for diffusion training.

"""



import cv2

import numpy as np

import pandas as pd

from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T





# Class mapping for HAM10000

CLASS_MAP = {

    'nv': 0,     # melanocytic nevi

    'mel': 1,    # melanoma

    'bkl': 2,    # benign keratosis

    'bcc': 3,    # basal cell carcinoma

    'akiec': 4,  # actinic keratosis

    'vasc': 5,   # vascular lesion

    'df': 6,     # dermatofibroma

}

NUM_CLASSES = 7

CLASS_NAMES = list(CLASS_MAP.keys())





class HAM10000Diffusion(Dataset):

    def __init__(self, img_dir, metadata_csv, img_size=128, transform=None):

        self.img_dir = Path(img_dir)

        self.df = pd.read_csv(metadata_csv)

        self.img_size = img_size



        # Drop duplicates — keep one image per lesion for cleaner training

        self.df = self.df.drop_duplicates(subset='lesion_id', keep='first').reset_index(drop=True)



        # Verify images exist

        valid = []

        for _, row in self.df.iterrows():

            p = self.img_dir / f"{row['image_id']}.jpg"

            if p.exists():

                valid.append(True)

            else:

                valid.append(False)

        self.df = self.df[valid].reset_index(drop=True)



        self.transform = transform or T.Compose([

            T.ToPILImage(),

            T.Resize((img_size, img_size)),

            T.RandomHorizontalFlip(),

            T.RandomVerticalFlip(),

            T.ToTensor(),

            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

        ])



        print(f"DDPM dataset: {len(self.df)} images, {self.df['dx'].nunique()} classes")

        print(f"Distribution:\n{self.df['dx'].value_counts().to_string()}")



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = self.img_dir / f"{row['image_id']}.jpg"

        img = cv2.imread(str(img_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        label = CLASS_MAP[row['dx']]

        return img, label





def get_ddpm_dataloader(data_dir, img_size=128, batch_size=16, num_workers=4):

    data_dir = Path(data_dir)

    ds = HAM10000Diffusion(

        img_dir=data_dir / "images",

        metadata_csv=data_dir / "HAM10000_metadata.csv",

        img_size=img_size,

    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,

                        num_workers=num_workers, pin_memory=True, drop_last=True)

    return loader
