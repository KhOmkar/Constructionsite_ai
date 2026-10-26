"""
Generative Adversarial Network (GAN) Architectures for Construction Site Synthesis.

Includes 3 distinct Generative Architectures:
1. DCGAN (Deep Convolutional GAN) — Unconditional site scene synthesis with progressive upsampling.
2. cGAN (Conditional GAN) — Targeted synthesis conditioned on OSHA safety violation categories.
3. WGAN-GP (Wasserstein GAN with Gradient Penalty) — Stable Earth Mover's Distance synthesis avoiding mode collapse.

Also includes Geodesic Latent Space Interpolation (SLERP).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. DCGAN (Deep Convolutional GAN) — Baseline Unconditional
# ==============================================================================

class Generator(nn.Module):
    """
    DCGAN Generator: Maps latent noise z ~ N(0, I) to synthetic construction site images.
    Inputs:
        x: Tensor of shape [B, latent_dim, 1, 1] (noise vector)
    Outputs:
        img: Tensor of shape [B, 3, 128, 128] (values in [-1, 1] via Tanh)
    """
    def __init__(self, latent_dim=100, out_channels=3, features=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.net = nn.Sequential(
            # Input: [B, latent_dim, 1, 1] -> [B, features*16, 4, 4]
            nn.ConvTranspose2d(latent_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),
            
            # [B, features*16, 4, 4] -> [B, features*8, 8, 8]
            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            
            # [B, features*8, 8, 8] -> [B, features*4, 16, 16]
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            
            # [B, features*4, 16, 16] -> [B, features*2, 32, 32]
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            
            # [B, features*2, 32, 32] -> [B, features, 64, 64]
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            # [B, features, 64, 64] -> [B, 3, 128, 128]
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator: Classifies construction site images as real or generated.
    Inputs:
        x: Tensor of shape [B, 3, 128, 128]
    Outputs:
        validity: Tensor of shape [B, 1, 1, 1] (probability in [0, 1] via Sigmoid)
    """
    def __init__(self, in_channels=3, features=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # [B, 3, 128, 128] -> [B, features, 64, 64]
            nn.Conv2d(in_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, features, 64, 64] -> [B, features*2, 32, 32]
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, features*2, 32, 32] -> [B, features*4, 16, 16]
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, features*4, 16, 16] -> [B, features*8, 8, 8]
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, features*8, 8, 8] -> [B, features*16, 4, 4]
            nn.Conv2d(features * 8, features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # [B, features*16, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2d(features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# 2. Conditional GAN (cGAN) — Targeted Hazard Category Generation
# ==============================================================================

GAN_CATEGORIES = [
    {"id": 0, "name": "Excavator & Earthwork Zone", "hazard_level": "MODERATE", "desc": "Active trench excavation with heavy earthmoving machinery"},
    {"id": 1, "name": "Structural Steel & High Elevation", "hazard_level": "HIGH", "desc": "Framing and crane lifts with fall-from-height exposure"},
    {"id": 2, "name": "PPE Compliant Inspection Crew", "hazard_level": "LOW", "desc": "Personnel fully equipped with Class E hardhats and ANSI safety vests"},
    {"id": 3, "name": "Critical Violation: Machine Proximity", "hazard_level": "CRITICAL", "desc": "Workers detected directly inside heavy equipment swing envelope"},
]
NUM_GAN_CATEGORIES = len(GAN_CATEGORIES)


class ConditionalGenerator(nn.Module):
    """
    Conditional GAN Generator: Synthesizes specific construction site scenarios
    conditioned on target safety category labels.
    Inputs:
        z: Tensor [B, latent_dim, 1, 1]
        c: Tensor [B] (integer class indices in [0, num_classes-1])
    Outputs:
        img: Tensor [B, 3, 128, 128]
    """
    def __init__(self, latent_dim=100, num_classes=4, embed_dim=32, out_channels=3, features=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, embed_dim)

        combined_dim = latent_dim + embed_dim

        self.net = nn.Sequential(
            nn.ConvTranspose2d(combined_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        # z: [B, latent_dim, 1, 1], c: [B]
        c_emb = self.class_emb(c).unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, 1, 1]
        x = torch.cat([z, c_emb], dim=1)                      # [B, latent_dim+embed_dim, 1, 1]
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator: Evaluates realism conditioned on category label.
    """
    def __init__(self, in_channels=3, num_classes=4, img_size=128, features=64):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

        self.net = nn.Sequential(
            # Input: 3 image channels + 1 label channel = 4 channels
            nn.Conv2d(in_channels + 1, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        c_map = self.class_emb(c).view(-1, 1, self.img_size, self.img_size)
        inp = torch.cat([x, c_map], dim=1)
        return self.net(inp)


# ==============================================================================
# 3. WGAN-GP (Wasserstein GAN with Gradient Penalty) — Stable Earth Mover Synthesis
# ==============================================================================

class WGANGenerator(nn.Module):
    """
    WGAN-GP Generator: Maps latent vector to synthetic construction site images
    optimized with Wasserstein distance.
    """
    def __init__(self, latent_dim=128, out_channels=3, features=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 16, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class WGANCritic(nn.Module):
    """
    WGAN-GP Critic: Evaluates the Wasserstein-1 Earth Mover's Distance.
    Uses LayerNorm (or no norm) instead of BatchNorm to satisfy 1-Lipschitz continuity.
    Outputs an unbounded scalar score f(x) (No Sigmoid).
    """
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1),
            nn.InstanceNorm2d(features * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1),
            nn.InstanceNorm2d(features * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 2, 1),
            nn.InstanceNorm2d(features * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, features * 16, 4, 2, 1),
            nn.InstanceNorm2d(features * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 16, 1, 4, 1, 0)
            # No Sigmoid! Unbounded critic score
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


# ==============================================================================
# 4. Latent Space Interpolation (SLERP / Geodesic Walk)
# ==============================================================================

def slerp(val: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) along the unit hyper-sphere in latent space.
    Interpolates smoothly between two latent noise vectors z1 and z2.
    """
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=1, keepdim=True), -1.0, 1.0))
    so = torch.sin(omega)
    if torch.abs(so).item() < 1e-6:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
