"""

Onco-GPT-X Module 2: Segmentation Models

- U-Net with EfficientNet-B3 encoder (baseline)

- U-Net with PVTv2-B2 encoder

- U-Net with SwinV2-Tiny encoder

DS2Net should be added from your coursework code.

"""



import torch

import torch.nn as nn

import torch.nn.functional as F

import timm





class DecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

        )



    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)





class UNetEfficientNet(nn.Module):

    """U-Net with EfficientNet-B3 encoder — strong baseline."""

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        enc = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)

        self.enc_layers = enc

        chs = enc.feature_info.channels()



        self.dec4 = DecoderBlock(chs[4], chs[3], 256)

        self.dec3 = DecoderBlock(256, chs[2], 128)

        self.dec2 = DecoderBlock(128, chs[1], 64)

        self.dec1 = DecoderBlock(64, chs[0], 32)

        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.head = nn.Conv2d(32, num_classes, 1)



    def forward(self, x):

        feats = self.enc_layers(x)

        d = self.dec4(feats[4], feats[3])

        d = self.dec3(d, feats[2])

        d = self.dec2(d, feats[1])

        d = self.dec1(d, feats[0])

        d = self.final_up(d)

        d = F.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.head(d)





class PVTv2UNet(nn.Module):

    """U-Net with PVTv2-B2 encoder — transformer-based."""

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        self.enc = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True)

        chs = self.enc.feature_info.channels()



        self.dec3 = DecoderBlock(chs[3], chs[2], 256)

        self.dec2 = DecoderBlock(256, chs[1], 128)

        self.dec1 = DecoderBlock(128, chs[0], 64)

        self.final_up = nn.Sequential(

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

        )

        self.head = nn.Conv2d(32, num_classes, 1)



    def forward(self, x):

        feats = self.enc(x)

        d = self.dec3(feats[3], feats[2])

        d = self.dec2(d, feats[1])

        d = self.dec1(d, feats[0])

        d = self.final_up(d)

        d = F.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.head(d)





class SwinV2UNet(nn.Module):

    """U-Net with SwinV2-Tiny encoder — shifted window attention."""

    def __init__(self, num_classes=1, pretrained=True):

        super().__init__()

        self.enc = timm.create_model('swinv2_tiny_window8_256', pretrained=pretrained, features_only=True)

        chs = self.enc.feature_info.channels()



        self.dec3 = DecoderBlock(chs[3], chs[2], 256)

        self.dec2 = DecoderBlock(256, chs[1], 128)

        self.dec1 = DecoderBlock(128, chs[0], 64)

        self.final_up = nn.Sequential(

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),

            nn.ReLU(inplace=True),

        )

        self.head = nn.Conv2d(32, num_classes, 1)



    def forward(self, x):

        feats = self.enc(x)

        d = self.dec3(feats[3], feats[2])

        d = self.dec2(d, feats[1])

        d = self.dec1(d, feats[0])

        d = self.final_up(d)

        d = F.interpolate(d, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.head(d)





def build_seg_model(name, num_classes=1, pretrained=True):

    models = {

        'unet_efficientnet': UNetEfficientNet,

        'pvtv2_unet': PVTv2UNet,

        'swinv2_unet': SwinV2UNet,

    }

    if name not in models:

        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")

    return models[name](num_classes=num_classes, pretrained=pretrained)
