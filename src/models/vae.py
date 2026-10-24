"""
Convolutional Variational Autoencoder (VAE) for Construction Site Digital Twin Modeling.

Module Architecture:
- VAEEncoder: 4-layer CNN feature extractor + Dual linear projection heads (mu, logvar).
- VAEDecoder: Linear expansion + 4-layer Transposed CNN upsampler + Sigmoid.
- ConvVAE: Reparameterization trick (z = mu + eps * std), sampling from prior z ~ N(0, I), and ELBO loss calculation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    """
    Encodes image [B, 3, 128, 128] into Gaussian parameter vectors mu and logvar [B, latent_dim].
    
    Inputs:
        x: Tensor of shape [B, in_channels, 128, 128]
    Outputs:
        mu: Tensor of shape [B, latent_dim] (Mean vector)
        logvar: Tensor of shape [B, latent_dim] (Log-variance vector)
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            # 3 x 128 x 128 -> 32 x 64 x 64
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        features = self.conv(x)
        flat = self.flatten(features)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Reconstructs image [B, 3, 128, 128] from sampled latent state z [B, latent_dim].
    
    Inputs:
        z: Tensor of shape [B, latent_dim]
    Outputs:
        recon: Tensor of shape [B, out_channels, 128, 128]
    """
    def __init__(self, out_channels=3, latent_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            # 256 x 8 x 8 -> 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 x 16 x 16 -> 64 x 32 x 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 x 32 x 32 -> 32 x 64 x 64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32 x 64 x 64 -> 3 x 128 x 128
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        recon = self.deconv(x)
        return recon


class ConvVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Construction Site Images.
    
    Contract:
        Input:       x in R^[B, 3, 128, 128]
        Latent:      mu, logvar in R^[B, 256]
        Sampled:     z in R^[B, 256]
        Output:      recon in R^[B, 3, 128, 128]
        Generative:  sample(N) -> R^[N, 3, 128, 128]
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = VAEDecoder(out_channels=in_channels, latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + eps * exp(0.5 * logvar), where eps ~ N(0, I)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, device="cpu"):
        """
        Generates novel construction site states by sampling directly from standard Gaussian prior z ~ N(0, I).
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def ssim_loss(img1, img2, window_size=7):
        """
        Differentiable Structural Similarity (SSIM) Loss.
        Computes 1.0 - SSIM across local image patches to preserve fine geometric edges & textures.
        """
        channels = img1.size(1)
        # 1D triangular/gaussian kernel approximation
        kernel_1d = torch.tensor([0.06, 0.12, 0.20, 0.24, 0.20, 0.12, 0.06], device=img1.device, dtype=img1.dtype)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        kernel = kernel_2d.repeat(channels, 1, 1, 1)
        pad = window_size // 2

        mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
        return 1.0 - ssim_map.mean()

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0, normalize_pixels=True, free_bits=0.0):
        """
        Calibrated Evidence Lower Bound (ELBO) Loss with Structural SSIM + L1/MSE formulation.

        Formulation:
            L_total = L_recon + beta * D_KL(q(z|x) || p(z))
            L_recon = 0.5 * L_ssim + 0.3 * L1 + 0.2 * MSE

        Args:
            recon_x: Reconstructed tensor [B, C, H, W]
            x: Ground truth input tensor [B, C, H, W]
            mu: Mean vector [B, latent_dim]
            logvar: Log-variance vector [B, latent_dim]
            beta: Trade-off parameter for KL divergence.
            normalize_pixels: Whether to balance loss scales per pixel for high reconstruction accuracy.
            free_bits: Minimum bits per latent dimension.
        """
        batch_size = x.size(0)
        num_pixels = x.size(1) * x.size(2) * x.size(3)

        # 1. Multi-Component Reconstruction Loss (SSIM structural + L1 sharp edge + MSE stability)
        if normalize_pixels:
            loss_ssim = ConvVAE.ssim_loss(recon_x, x)
            l1_loss = F.l1_loss(recon_x, x, reduction='mean')
            mse_loss = F.mse_loss(recon_x, x, reduction='mean')
            recon_loss = 0.5 * loss_ssim + 0.3 * l1_loss + 0.2 * mse_loss
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / batch_size

        # 2. KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if free_bits > 0.0:
            kld_per_dim = torch.clamp(kld_per_dim, min=free_bits)

        kld_loss_raw = torch.sum(kld_per_dim) / batch_size

        if normalize_pixels:
            # Scaled KL divergence normalized to image dimensions
            kld_scaled = kld_loss_raw / num_pixels
            total_loss = recon_loss + beta * kld_scaled
        else:
            total_loss = recon_loss + beta * kld_loss_raw

        return total_loss, recon_loss, kld_loss_raw
