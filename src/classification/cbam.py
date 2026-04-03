"""

Convolutional Block Attention Module (CBAM)

Woo et al., 2018 - "CBAM: Convolutional Block Attention Module"



Two sequential sub-modules:

1. Channel Attention: "WHAT" to focus on

2. Spatial Attention: "WHERE" to focus

"""



import torch

import torch.nn as nn





class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        mid = max(channels // reduction, 8)

        self.shared_mlp = nn.Sequential(

            nn.Linear(channels, mid, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(mid, channels, bias=False),

        )



    def forward(self, x):

        b, c, _, _ = x.size()

        avg_pool = x.mean(dim=[2, 3])                          # (B, C)

        max_pool = x.amax(dim=[2, 3])                          # (B, C)

        attn = torch.sigmoid(

            self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)

        )

        return x * attn.unsqueeze(-1).unsqueeze(-1)





class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        pad = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)



    def forward(self, x):

        avg_pool = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)

        max_pool = x.amax(dim=1, keepdim=True)                 # (B, 1, H, W)

        attn = torch.sigmoid(

            self.conv(torch.cat([avg_pool, max_pool], dim=1))

        )

        return x * attn





class CBAM(nn.Module):

    """Apply channel attention then spatial attention sequentially."""

    def __init__(self, channels, reduction=16, spatial_kernel=7):

        super().__init__()

        self.channel = ChannelAttention(channels, reduction)

        self.spatial = SpatialAttention(spatial_kernel)



    def forward(self, x):

        x = self.channel(x)

        x = self.spatial(x)

        return x
