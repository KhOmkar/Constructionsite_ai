"""
Construction Site Safety Vision Transformer (ViT) Training Script.
Trained on the 2,801 Real Construction Images (Roboflow CSS & ConstructionSite-10k).

Usage:
  Full Training (6 epochs):
    python train_transformer.py

  Quick Demo Training for Live Review (2 epochs, ~60 seconds):
    python train_transformer.py --quick

  Custom:
    python train_transformer.py --epochs 4 --batch-size 32
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.models.transformer_models import ConstructionHazardViT
from src.data.css_dataset import get_css_dataloaders, CSS_CLASSES

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
FEAT_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(FEAT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epochs=6, batch_size=32, lr=3e-4, quick=False):
    if quick:
        epochs = 2
        max_train = 600
        print("\n>>> RUNNING IN QUICK DEMO MODE (2 Epochs on 600 Real Images) <<<")
    else:
        max_train = None

    print("=" * 72)
    print("  TRAINING VISION TRANSFORMER (ViT) FOR CONSTRUCTION SAFETY & HAZARDS")
    print(f"  Hardware Acceleration : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU Device            : {torch.cuda.get_device_name(0)}")
    print(f"  Epochs                : {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 72)

    # 1. Dataloaders
    train_loader, valid_loader, test_loader = get_css_dataloaders(
        batch_size=batch_size, img_size=128, max_train=max_train
    )
    print(f"[OK] Dataset Loaded: {len(train_loader.dataset)} Train | {len(valid_loader.dataset)} Valid | {len(test_loader.dataset)} Test")

    # 2. Model
    model = ConstructionHazardViT(
        img_size=128, patch_size=16, in_channels=3, num_classes=10,
        embed_dim=256, depth=4, num_heads=8
    ).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model Instantiated: {params:,} parameters (4.10M)")

    # 3. Loss & Optimizer
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_map = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "cls_loss": [], "map_loss": [],
        "val_loss": [], "val_f1": []
    }

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_map = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            hazard_gt = batch["hazard_map"].to(DEVICE)

            optimizer.zero_grad()
            logits, probs, pred_hazard, risk_score = model(images)

            loss_c = criterion_cls(logits, labels)
            loss_m = criterion_map(pred_hazard, hazard_gt)
            loss = loss_c + 0.6 * loss_m

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_c.item()
            total_map += loss_m.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                hazard_gt = batch["hazard_map"].to(DEVICE)

                logits, probs, pred_hazard, _ = model(images)
                loss_c = criterion_cls(logits, labels)
                loss_m = criterion_map(pred_hazard, hazard_gt)
                val_loss += (loss_c + 0.6 * loss_m).item()

                preds_binary = (probs > 0.40).float()
                all_preds.append(preds_binary.cpu())
                all_targets.append(labels.cpu())

        n_tr = len(train_loader)
        n_va = len(valid_loader)
        avg_tr = total_loss / n_tr
        avg_va = val_loss / n_va

        cat_preds = torch.cat(all_preds, dim=0).numpy()
        cat_targets = torch.cat(all_targets, dim=0).numpy()
        tp = np.sum((cat_preds == 1) & (cat_targets == 1))
        fp = np.sum((cat_preds == 1) & (cat_targets == 0))
        fn = np.sum((cat_preds == 0) & (cat_targets == 1))
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)

        history["train_loss"].append(avg_tr)
        history["cls_loss"].append(total_cls / n_tr)
        history["map_loss"].append(total_map / n_tr)
        history["val_loss"].append(avg_va)
        history["val_f1"].append(float(f1))

        print(f"  Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_tr:.4f} (Cls: {total_cls/n_tr:.4f}, Map: {total_map/n_tr:.4f}) | Val Loss: {avg_va:.4f} | Val F1: {f1:.3f}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n[OK] Training completed in {elapsed:.1f}s ({elapsed/epochs:.1f}s/epoch)")

    # 4. Save Weights
    hazard_w = os.path.join(WEIGHTS_DIR, "hazard_vit_weights.pkl")
    trans_w = os.path.join(WEIGHTS_DIR, "transformer_weights.pkl")
    torch.save(model.state_dict(), hazard_w)
    torch.save(model.state_dict(), trans_w)
    print(f"[SUCCESS] Weights saved to:\n  - {hazard_w}\n  - {trans_w}")

    # 5. Generate Review Figures
    print("\n--- Generating Defense Visual Artifacts ---")
    _plot_loss_curves(history)
    _plot_attention_rollout_samples(model, valid_loader)
    _plot_per_class_metrics(cat_preds, cat_targets)

    print("\n======================================================================")
    print("  [SUCCESS] TRANSFORMER TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("======================================================================")


def _plot_loss_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='#0B0F19')
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.set_facecolor('#111827')
    ax1.plot(epochs, history["train_loss"], label="Total Train Loss", color="#00F0FF", linewidth=2.5)
    ax1.plot(epochs, history["cls_loss"], label="Safety Cls Loss (BCE)", color="#10B981", linestyle="--", linewidth=1.8)
    ax1.plot(epochs, history["map_loss"], label="Hazard Map Loss (MSE)", color="#F59E0B", linestyle=":", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#EC4899", linewidth=2.0)
    ax1.set_title("ViT Multi-Task Loss Convergence", color='white', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Epoch", color='#94A3B8', fontsize=11)
    ax1.set_ylabel("Loss", color='#94A3B8', fontsize=11)
    ax1.tick_params(colors='#94A3B8')
    ax1.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
    ax1.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white')

    ax2.set_facecolor('#111827')
    ax2.plot(epochs, history["val_f1"], label="Validation Safety F1-Score", color="#38BDF8", linewidth=2.8, marker='o')
    ax2.set_title("OSHA Multi-Label Safety F1 Trajectory", color='white', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Epoch", color='#94A3B8', fontsize=11)
    ax2.set_ylabel("F1 Score", color='#94A3B8', fontsize=11)
    ax2.tick_params(colors='#94A3B8')
    ax2.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
    ax2.legend(facecolor='#1E293B', edgecolor='#334155', labelcolor='white')

    plt.tight_layout()
    out_p = os.path.join(FEAT_DIR, "transformer_loss_curve.png")
    fig.savefig(out_p, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {out_p}")


def _plot_attention_rollout_samples(model, valid_loader):
    model.eval()
    batch = next(iter(valid_loader))
    images = batch["image"][:4].to(DEVICE)
    raw_images = batch["raw_image"][:4]
    hazard_gt = batch["hazard_map"][:4]

    with torch.no_grad():
        logits, probs, pred_hazard, risk_scores, norm_attn = model(images, return_attention=True)

    fig, axes = plt.subplots(4, 4, figsize=(16, 15), facecolor='#0B0F19')
    col_titles = [
        "Real Construction Photo (640x640)",
        "Ground-Truth Hazard Mask",
        "8-Head ViT Attention Saliency",
        "Predicted Spatial Risk Heatmap"
    ]

    for row in range(4):
        raw_np = raw_images[row].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(raw_np)
        detected = [CSS_CLASSES[i] for i in range(10) if probs[row, i] > 0.40]
        title_str = ", ".join(detected[:2]) if detected else "Compliant / Safe"
        axes[row, 0].set_title(f"Scene #{row+1}: {title_str}", color='white', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')

        gt_np = hazard_gt[row, 0].numpy()
        axes[row, 1].imshow(gt_np, cmap='jet')
        axes[row, 1].set_title("OSHA Equipment Danger Mask", color='#F59E0B', fontsize=10)
        axes[row, 1].axis('off')

        attn_np = norm_attn[row, 0].cpu().numpy()
        axes[row, 2].imshow(attn_np, cmap='plasma')
        axes[row, 2].set_title("8-Head ViT Attention Saliency", color='#38BDF8', fontsize=10)
        axes[row, 2].axis('off')

        pred_np = pred_hazard[row, 0].cpu().numpy()
        axes[row, 3].imshow(pred_np, cmap='jet')
        rk = float(risk_scores[row, 0].cpu().item() * 100)
        status = "CRITICAL" if rk > 55 else ("MODERATE" if rk > 30 else "SAFE")
        status_color = "#EF4444" if rk > 55 else ("#F59E0B" if rk > 30 else "#10B981")
        axes[row, 3].set_title(f"Risk: {rk:.1f}% [{status}]", color=status_color, fontsize=10, fontweight='bold')
        axes[row, 3].axis('off')

    for col in range(4):
        axes[0, col].text(0.5, 1.15, col_titles[col], transform=axes[0, col].transAxes,
                         ha='center', va='bottom', color='#00F0FF', fontsize=12, fontweight='bold')

    plt.tight_layout()
    out_p = os.path.join(FEAT_DIR, "transformer_attention_maps.png")
    fig.savefig(out_p, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {out_p}")


def _plot_per_class_metrics(preds, targets):
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor='#0B0F19')
    ax.set_facecolor('#111827')

    class_f1s = []
    for c in range(10):
        tp = np.sum((preds[:, c] == 1) & (targets[:, c] == 1))
        fp = np.sum((preds[:, c] == 1) & (targets[:, c] == 0))
        fn = np.sum((preds[:, c] == 0) & (targets[:, c] == 1))
        f1_c = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        class_f1s.append(f1_c)

    colors = ['#10B981' if c not in [2, 3, 4, 8, 9] else '#EF4444' for c in range(10)]
    bars = ax.bar(CSS_CLASSES, class_f1s, color=colors, edgecolor='#1E293B', width=0.6)

    for bar, score in zip(bars, class_f1s):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{score:.2f}",
                ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    ax.set_title("Per-Class ViT Recognition & Hazard Detection F1-Scores", color='white', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("F1 Score", color='#94A3B8', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors='#94A3B8', axis='x', rotation=25)
    ax.tick_params(colors='#94A3B8', axis='y')
    ax.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF", axis='y')

    out_p = os.path.join(FEAT_DIR, "transformer_metrics.png")
    fig.savefig(out_p, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [OK] Saved: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Construction Hazard Vision Transformer")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick 2-epoch demo training")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, quick=args.quick)
