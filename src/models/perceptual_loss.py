"""
Perceptual and Structural Edge Loss Modules for Construction Site Deep Learning.

Implements:
1. VGGPerceptualLoss: Deep feature matching (VGG16 conv layers) to eliminate L2 regression blur.
2. SSIMLoss: Structural similarity loss to enforce outline and contour preservation.
3. SobelEdgeLoss: High-frequency gradient edge loss to preserve scaffolding and machinery boundaries.
4. CompositePerceptualLoss: Multi-objective loss combining MSE, Perceptual, SSIM, and Edge losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    Extracts deep perceptual features from VGG16 to penalize semantic/perceptual differences.
    """
    def __init__(self, feature_layers=(3, 8, 15, 22), weights=(1.0, 1.0, 1.0, 1.0)):
        super().__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        except Exception:
            # Fallback if offline / weights unavailable
            vgg = models.vgg16(weights=None).features
            
        self.layers = nn.ModuleList()
        prev_idx = 0
        for idx in feature_layers:
            self.layers.append(vgg[prev_idx:idx + 1])
            prev_idx = idx + 1
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.weights = weights
        # VGG standard normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize to ImageNet stats
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        x_p = pred
        x_t = target
        for layer, w in zip(self.layers, self.weights):
            x_p = layer(x_p)
            x_t = layer(x_t)
            loss += w * F.l1_loss(x_p, x_t)
            
        return loss


class SobelEdgeLoss(nn.Module):
    """
    Penalizes differences in spatial gradients (edges, scaffolding, contours).
    """
    def __init__(self):
        super().__init__()
        # Sobel filters
        gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def _get_edges(self, x):
        # Convert RGB to grayscale
        gray = x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
        edge_x = F.conv2d(gray, self.gx, padding=1)
        edge_y = F.conv2d(gray, self.gy, padding=1)
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

    def forward(self, pred, target):
        pred_edges = self._get_edges(pred)
        target_edges = self._get_edges(target)
        return F.l1_loss(pred_edges, target_edges)


class SSIMLoss(nn.Module):
    """
    SSIM Loss (1 - SSIM) for structural coherence.
    """
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        k = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
        gauss_1d = torch.exp(-k**2 / (2 * sigma**2))
        gauss_2d = (gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        window = (gauss_2d / gauss_2d.sum()).repeat(3, 1, 1, 1)
        self.register_buffer("window", window)

    def forward(self, pred, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(pred, self.window, padding=self.window_size // 2, groups=3)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, self.window, padding=self.window_size // 2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=3) - mu2_sq
        sigma12 = F.conv2d(pred * target, self.window, padding=self.window_size // 2, groups=3) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()


class CompositePerceptualLoss(nn.Module):
    """
    Full composite loss combining:
    L = w_mse * MSE + w_percep * VGG + w_ssim * SSIM + w_edge * Sobel
    """
    def __init__(self, w_mse=1.0, w_percep=0.1, w_ssim=0.2, w_edge=0.1):
        super().__init__()
        self.w_mse = w_mse
        self.w_percep = w_percep
        self.w_ssim = w_ssim
        self.w_edge = w_edge
        
        self.mse = nn.MSELoss()
        self.perceptual = VGGPerceptualLoss()
        self.ssim = SSIMLoss()
        self.edge = SobelEdgeLoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_p = self.perceptual(pred, target)
        loss_s = self.ssim(pred, target)
        loss_e = self.edge(pred, target)
        
        total = (
            self.w_mse * loss_mse +
            self.w_percep * loss_p +
            self.w_ssim * loss_s +
            self.w_edge * loss_e
        )
        return total, {"mse": loss_mse.item(), "perceptual": loss_p.item(), "ssim": loss_s.item(), "edge": loss_e.item()}
