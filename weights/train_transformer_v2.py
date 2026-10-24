"""
Training Pipeline for Upgraded Construction Progress Transformer with Perceptual Loss.

Trains with:
1. L1 + MSE Reconstruction Loss
2. Deep Perceptual VGG Loss (LPIPS)
3. SSIM Structural Coherence Loss
4. Sobel Edge Gradient Loss
5. Spatial Risk Supervision
"""
import os
import sys
import torch
import torch.optim as optim

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.transformer_models import ConstructionProgressTransformer
from src.models.perceptual_loss import CompositePerceptualLoss
from src.data.curate_datasets import get_dataloader

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 8
BATCH_SIZE = 16
LR = 2e-4


def train_upgraded_transformer():
    print("=" * 60)
    print("  Training Upgraded Progress Transformer (Perceptual + SSIM)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # 1. Initialize Model
    model = ConstructionProgressTransformer(
        img_size=128, patch_size=16, in_channels=3, num_stages=5,
        embed_dim=256, depth=4, num_heads=8
    ).to(DEVICE)

    # 2. Losses and Optimizer
    perceptual_loss_fn = CompositePerceptualLoss(
        w_mse=1.0, w_percep=0.15, w_ssim=0.25, w_edge=0.15
    ).to(DEVICE)
    risk_loss_fn = torch.nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 3. DataLoader
    dataloader = get_dataloader(batch_size=BATCH_SIZE, length=240)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_p_loss = 0.0
        total_r_loss = 0.0

        for current_img, stage_id, delta_days, future_target, risk_target in dataloader:
            current_img = current_img.to(DEVICE)
            stage_id = stage_id.to(DEVICE)
            delta_days = delta_days.to(DEVICE)
            future_target = future_target.to(DEVICE)
            risk_target = risk_target.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            pred_future, pred_risk = model(current_img, stage_id, delta_days)

            # Losses
            p_loss, loss_dict = perceptual_loss_fn(pred_future, future_target)
            r_loss = risk_loss_fn(pred_risk, risk_target)

            loss = p_loss + 0.5 * r_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_r_loss += r_loss.item()

        scheduler.step()
        n = len(dataloader)
        print(f"  Epoch [{epoch+1}/{EPOCHS}] — Loss: {total_loss/n:.4f} (Percep: {total_p_loss/n:.4f}, Risk: {total_r_loss/n:.4f})")

    # 4. Save Weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(WEIGHTS_DIR, "transformer_weights.pkl")
    torch.save(model.state_dict(), save_path)
    print("=" * 60)
    print(f"  [SUCCESS] Upgraded weights saved to: {save_path}")
    print("=" * 60)
    return model


if __name__ == "__main__":
    train_upgraded_transformer()
