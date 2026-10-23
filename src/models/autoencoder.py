"""
Convolutional Autoencoder (AE) and Feature Classification for Construction Site Safety.

Module Architectures:
- ConvBlock: Dual Conv2d + BatchNorm2d + LeakyReLU block with residual capability.
- AutoencoderV2: U-Net style Autoencoder with multi-scale skip connections and multi-scale feature extraction.
- EncoderClassifier: Supervised classification head on top of pre-trained AutoencoderV2 encoder.
- ConvAutoencoder, ConvEncoder, ConvDecoder: Standard bottleneck Autoencoder (legacy support).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SAFETY_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle"
]
NUM_CLASSES = len(SAFETY_CLASSES)


# ==============================================================================
# Autoencoder V2 (Optimized U-Net Style Architecture)
# ==============================================================================

class ConvBlock(nn.Module):
    """Dual-layer Convolutional block with BatchNorm and LeakyReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AutoencoderV2(nn.Module):
    """
    Optimized Convolutional Autoencoder with multi-scale skip connections.
    Learns rich hierarchical visual features from construction site images.
    """
    def __init__(self, base=32):
        super().__init__()
        self.base = base

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 4)

        self.pool = nn.MaxPool2d(2)

        # Spatial bottleneck: (base*4) x 8 x 8 = 128 x 8 x 8
        self.bottleneck = ConvBlock(base * 4, base * 4)

        # Decoder with spatial skip connections
        self.up4 = nn.ConvTranspose2d(base * 4, base * 4, 2, 2)
        self.dec4 = ConvBlock(base * 8, base * 4)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, 2, 2)
        self.dec3 = ConvBlock(base * 8, base * 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base)

        self.up1 = nn.ConvTranspose2d(base, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base)

        self.output = nn.Sequential(
            nn.Conv2d(base, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Extract spatial bottleneck representation [B, 128, 8, 8]."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))
        return z

    def encode_multiscale(self, x):
        """
        Extract pooled multi-scale features across all 5 encoder levels.
        Output dimension: 32 + 64 + 128 + 128 + 128 = 480 dims.
        """
        e1 = self.enc1(x)                                   # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))                       # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))                       # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))                       # (B, 128, 16, 16)
        z = self.bottleneck(self.pool(e4))                  # (B, 128, 8, 8)

        def gap(t):
            return F.adaptive_avg_pool2d(t, 1).flatten(1)

        return torch.cat([gap(e1), gap(e2), gap(e3), gap(e4), gap(z)], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        z = self.bottleneck(self.pool(e4))

        d4 = self.up4(z)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1), z


# ==============================================================================
# Supervised Classification Head (Autoencoder Feature Classification)
# ==============================================================================

class EncoderClassifier(nn.Module):
    """
    Classification network utilizing representations from pre-trained AutoencoderV2.
    Trained progressively (Phase 1: Head-only, Phase 2: enc4+bottleneck, Phase 3: Full fine-tune).
    """
    def __init__(self, ae: AutoencoderV2, num_classes: int = 10, drop: float = 0.4):
        super().__init__()
        self.ae = ae
        feat_dim = ae.base * 4  # 128 channels

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),       # (B, 128, 4, 4)
            nn.Flatten(),                  # (B, 128*4*4 = 2048)
            nn.Linear(feat_dim * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(drop * 0.5),
            nn.Linear(128, num_classes),
        )

    def set_encoder_frozen(self, frozen=True):
        for p in self.ae.parameters():
            p.requires_grad = not frozen

    def unfreeze_layer(self, *modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True

    def forward(self, x):
        z = self.ae.encode(x)
        return self.head(z)


# ==============================================================================
# Legacy ConvAutoencoder (for backward compatibility)
# ==============================================================================

class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ConvDecoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        return self.deconv(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
