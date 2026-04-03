"""

Onco-GPT-X Module 4: Conditional DDPM

Class-conditional denoising diffusion for counterfactual generation.

Uses HuggingFace diffusers UNet2DModel with class embedding.

"""



import torch

import torch.nn as nn

import torch.nn.functional as F

import math





class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.dim = dim



    def forward(self, t):

        device = t.device

        half = self.dim // 2

        emb = math.log(10000) / (half - 1)

        emb = torch.exp(torch.arange(half, device=device) * -emb)

        emb = t[:, None].float() * emb[None, :]

        return torch.cat([emb.sin(), emb.cos()], dim=-1)





class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):

        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()



    def forward(self, x, t):

        h = self.conv1(F.silu(self.norm1(x)))

        h = h + self.time_mlp(t)[:, :, None, None]

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        return h + self.skip(x)





class Downsample(nn.Module):

    def __init__(self, ch):

        super().__init__()

        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)



    def forward(self, x):

        return self.conv(x)





class Upsample(nn.Module):

    def __init__(self, ch):

        super().__init__()

        self.conv = nn.Conv2d(ch, ch, 3, padding=1)



    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return self.conv(x)





class ConditionalUNet(nn.Module):

    """

    Simple conditional U-Net for DDPM.

    Conditions on: timestep + class label.

    """

    def __init__(self, img_ch=3, base_ch=128, ch_mult=(1, 2, 4), num_classes=7, time_dim=256):

        super().__init__()

        self.time_dim = time_dim



        # Time embedding

        self.time_emb = nn.Sequential(

            SinusoidalPosEmb(time_dim),

            nn.Linear(time_dim, time_dim),

            nn.SiLU(),

            nn.Linear(time_dim, time_dim),

        )



        # Class embedding

        self.class_emb = nn.Embedding(num_classes, time_dim)



        # Initial conv

        self.init_conv = nn.Conv2d(img_ch, base_ch, 3, padding=1)



        # Encoder

        chs = [base_ch * m for m in ch_mult]

        self.down_blocks = nn.ModuleList()

        self.downsamples = nn.ModuleList()

        in_ch = base_ch

        for out_ch in chs:

            self.down_blocks.append(nn.ModuleList([

                ResBlock(in_ch, out_ch, time_dim),

                ResBlock(out_ch, out_ch, time_dim),

            ]))

            self.downsamples.append(Downsample(out_ch))

            in_ch = out_ch



        # Bottleneck

        self.mid1 = ResBlock(chs[-1], chs[-1], time_dim)

        self.mid2 = ResBlock(chs[-1], chs[-1], time_dim)



        # Decoder

        self.up_blocks = nn.ModuleList()

        self.upsamples = nn.ModuleList()

        for i, out_ch in enumerate(reversed(chs)):

            skip_ch = out_ch

            self.upsamples.append(Upsample(in_ch))

            self.up_blocks.append(nn.ModuleList([

                ResBlock(in_ch + skip_ch, out_ch, time_dim),

                ResBlock(out_ch, out_ch, time_dim),

            ]))

            in_ch = out_ch



        # Output

        self.out_norm = nn.GroupNorm(8, base_ch)

        self.out_conv = nn.Conv2d(base_ch, img_ch, 3, padding=1)



    def forward(self, x, t, class_labels):

        # Embeddings

        t_emb = self.time_emb(t)

        c_emb = self.class_emb(class_labels)

        emb = t_emb + c_emb



        # Encoder

        x = self.init_conv(x)

        skips = [x]

        for (r1, r2), down in zip(self.down_blocks, self.downsamples):

            x = r1(x, emb)

            x = r2(x, emb)

            skips.append(x)

            x = down(x)



        # Bottleneck

        x = self.mid1(x, emb)

        x = self.mid2(x, emb)



        # Decoder

        for (r1, r2), up in zip(self.up_blocks, self.upsamples):

            x = up(x)

            s = skips.pop()

            if x.shape[2:] != s.shape[2:]:

                x = F.interpolate(x, size=s.shape[2:], mode='nearest')

            x = torch.cat([x, s], dim=1)

            x = r1(x, emb)

            x = r2(x, emb)



        x = self.out_conv(F.silu(self.out_norm(x)))

        return x





class GaussianDiffusion:

    """DDPM forward/reverse process."""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):

        self.timesteps = timesteps

        self.device = device



        betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        alphas = 1.0 - betas

        alpha_bar = torch.cumprod(alphas, dim=0)



        self.betas = betas

        self.alphas = alphas

        self.alpha_bar = alpha_bar

        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)

        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)



    def q_sample(self, x0, t, noise=None):

        """Forward process: add noise to x0."""

        if noise is None:

            noise = torch.randn_like(x0)

        sa = self.sqrt_alpha_bar[t][:, None, None, None]

        som = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        return sa * x0 + som * noise, noise



    @torch.no_grad()

    def p_sample(self, model, x, t, class_labels):

        """Reverse process: denoise one step."""

        beta = self.betas[t][:, None, None, None]

        somb = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]

        sa_inv = 1.0 / torch.sqrt(self.alphas[t])[:, None, None, None]



        pred_noise = model(x, t, class_labels)

        mean = sa_inv * (x - (beta / somb) * pred_noise)



        if t[0] > 0:

            noise = torch.randn_like(x)

            return mean + torch.sqrt(beta) * noise

        return mean



    @torch.no_grad()

    def sample(self, model, n_samples, class_labels, img_size=128, channels=3):

        """Generate images from noise."""

        model.eval()

        x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)

        class_labels = class_labels.to(self.device)



        for i in reversed(range(self.timesteps)):

            t = torch.full((n_samples,), i, dtype=torch.long, device=self.device)

            x = self.p_sample(model, x, t, class_labels)



        # Denormalize from [-1, 1] to [0, 1]

        x = (x + 1) / 2

        x = x.clamp(0, 1)

        return x
