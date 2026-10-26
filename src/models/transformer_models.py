"""
Vision Transformer (ViT) & Safety Intelligence Architectures for Construction Sites.

Features:
1. PatchEmbedding: 2D Image Patch Partitioner (converts [B, C, H, W] -> [B, N, D]).
2. TransformerEncoderBlock: Multi-Head Self-Attention (MHSA) + MLP with Pre-LayerNorm & Residuals.
3. VisionTransformerAutoencoder (ViT-AE): Self-attention autoencoder for digital twin representations.
4. ConstructionHazardViT: 8-Head Multi-Head Self-Attention Vision Transformer with Explainable Attention Rollout.
5. SafetyIntelligenceEngine: Combines object detection with spatial ViT attention and rule-based OSHA reasoning.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAFETY_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]
NUM_CLASSES = len(SAFETY_CLASSES)


class PatchEmbedding(nn.Module):
    """
    Splits image [B, C, H, W] into non-overlapping patches of size [P, P]
    and linearly projects them to embedding dimension D.
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Input spatial resolution ({H}x{W}) does not match model resolution ({self.img_size}x{self.img_size})"
        )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block with Multi-Head Self-Attention (MHSA) and MLP.
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        norm_x = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(norm_x, norm_x, norm_x, need_weights=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        else:
            attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x


class VisionTransformerAutoencoder(nn.Module):
    """
    Vision Transformer Autoencoder (ViT-AE) for Construction Site Images.
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        depth=4,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_channels)

    def unpatchify(self, patch_preds):
        p = self.patch_size
        h = w = self.img_size // p
        B = patch_preds.shape[0]
        x = patch_preds.reshape(shape=(B, h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(B, self.in_channels, h * p, w * p))

    def forward(self, x):
        tokens = self.patch_embed(x) + self.pos_embed
        for block in self.encoder_blocks:
            tokens = block(tokens)
        latent_tokens = self.encoder_norm(tokens)

        dec_tokens = self.decoder_proj(latent_tokens) + self.decoder_pos_embed
        for block in self.decoder_blocks:
            dec_tokens = block(dec_tokens)
        dec_tokens = self.decoder_norm(dec_tokens)

        patch_preds = self.decoder_pred(dec_tokens)
        recon = torch.sigmoid(self.unpatchify(patch_preds))
        return recon, latent_tokens


class ConstructionHazardViT(nn.Module):
    """
    Vision Transformer (ViT) for Construction Site Hazard Intelligence & Spatial Saliency.
    Operates on 64 spatial patches (16x16 on 128x128) + 1 [CLS] token.
    Extracts 8-Head Multi-Head Self-Attention (MHSA) rollout for explainability.
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Patch Tokenizer
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # 2. [CLS] token + 1D learnable positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 4. Multi-Label Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 5. Hazard Map Decoder
        self.hazard_decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 16 -> 32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # 32 -> 64
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),           # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        patches = self.patch_embed(x)

        # Prepend [CLS] token -> [B, 65, 256]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)

        attn_weights_list = []
        for block in self.blocks:
            if return_attention:
                tokens, attn_w = block(tokens, return_attention=True)
                attn_weights_list.append(attn_w)
            else:
                tokens = block(tokens, return_attention=False)

        tokens = self.norm(tokens)

        # [CLS] representation for multi-label safety prediction
        cls_rep = tokens[:, 0]
        logits = self.classifier(cls_rep)
        probs = torch.sigmoid(logits)

        # Spatial tokens -> 2D feature map for Hazard Heatmap [B, 256, 8, 8]
        spatial_tokens = tokens[:, 1:]
        grid_dim = int(self.num_patches ** 0.5)
        feat_map = spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, grid_dim, grid_dim)
        hazard_map = self.hazard_decoder(feat_map)

        # OSHA Risk Level Score (0.0 to 1.0)
        p_no_hardhat = probs[:, 2]
        p_no_vest = probs[:, 4]
        p_worker = probs[:, 5]
        p_machinery = probs[:, 8]
        p_vehicle = probs[:, 9]

        machinery_presence = torch.clamp(p_machinery + p_vehicle, 0.0, 1.0)
        proximity_risk = p_worker * machinery_presence
        raw_risk = p_no_hardhat * 0.35 + p_no_vest * 0.35 + proximity_risk * 0.30
        risk_score = torch.clamp(raw_risk, 0.0, 1.0).unsqueeze(1)

        if return_attention and attn_weights_list:
            last_attn = attn_weights_list[-1]  # [B, 65, 65]
            # Attention from [CLS] token (idx 0) to all 64 spatial patches (idx 1..64)
            cls_to_patches = last_attn[:, 0, 1:].reshape(B, 1, grid_dim, grid_dim)
            upsampled_attn = F.interpolate(cls_to_patches, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            min_v = upsampled_attn.amin(dim=(2, 3), keepdim=True)
            max_v = upsampled_attn.amax(dim=(2, 3), keepdim=True)
            norm_attn = (upsampled_attn - min_v) / (max_v - min_v + 1e-6)
            return logits, probs, hazard_map, risk_score, norm_attn

        return logits, probs, hazard_map, risk_score
