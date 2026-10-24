"""
Production Training Pipeline for Calibrated High-Fidelity ConvVAE.

Key Enhancements:
1. Combined L1 + MSE Reconstruction Loss (eliminates blurriness from pure MSE).
2. KL Annealing Schedule (beta ramp-up) to prevent early latent collapse.
3. Eval-mode deterministic mu reparameterization.
4. Auto-exports checkpoints to weights/ and artifacts to data/processed/features/.
"""
import os
import sys
import time
import glob
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# Set project paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.vae import ConvVAE

# Dataset directories
DATASET_CANDIDATES = [
    r"C:\Users\mrpat\Downloads\Constructio_Site_AI\ConstructionSiteSafetyImageDatasetRoboflow\css-data",
    str(BASE_DIR / "data"),
    str(BASE_DIR / "css-data"),
]
DATASET_PATH = next((p for p in DATASET_CANDIDATES if os.path.exists(p)), DATASET_CANDIDATES[0])

OUT_DIR = BASE_DIR / "data" / "processed" / "features"
WEIGHTS_DIR = BASE_DIR / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


class ConstructionDataset(Dataset):
    def __init__(self, img_dir, is_train=True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        with Image.open(self.img_paths[idx]) as im:
            return self.transform(im.convert("RGB"))


def train(epochs=25, batch_size=64, lr=3e-4, warm_start=True, device="cpu"):
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print("=" * 70)
    print(f"[*] Training High-Fidelity ConvVAE on {device.type.upper()}")
    print(f"    Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print(f"    Dataset: {DATASET_PATH}")
    print("=" * 70)

    train_dir = os.path.join(DATASET_PATH, "train", "images")
    val_dir = os.path.join(DATASET_PATH, "valid", "images")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(DATASET_PATH, "test", "images")

    train_ds = ConstructionDataset(train_dir, is_train=True)
    val_ds = ConstructionDataset(val_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded: {len(train_ds):,} Train Images | {len(val_ds):,} Val Images")

    # Initialize model
    model = ConvVAE(in_channels=3, latent_dim=256).to(device)

    # Warm start from existing checkpoint if available
    init_epoch = 1
    existing_ckpt = OUT_DIR / "vae_model.pth"
    if not existing_ckpt.exists():
        existing_ckpt = WEIGHTS_DIR / "vae_model.pth"

    if warm_start and existing_ckpt.exists():
        try:
            ckpt = torch.load(existing_ckpt, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            print(f" [OK] Warm-starting from existing weights: {existing_ckpt.name}")
        except Exception as e:
            print(f" [WARN] Failed warm start: {e}. Training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "val_mse": [],
        "val_kld": []
    }

    best_val_loss = float("inf")
    best_ssim = 0.0

    print("\n" + "=" * 78)
    print(f" {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'PSNR (dB)':>9} | {'SSIM':>7} | {'KLD':>7} | {'Time':>6}")
    print("=" * 78)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── KL Annealing schedule (Ramp beta from 0.0 to 0.001 over first 15 epochs) ──
        beta = min(0.001, 0.001 * (epoch / 15.0))

        # ── 1. Train Step ──
        model.train()
        train_loss_accum = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, r_loss, k_loss = model.loss_function(recon, batch, mu, logvar, beta=beta, normalize_pixels=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_accum += loss.item()

        train_loss = train_loss_accum / max(len(train_loader), 1)

        # ── 2. Validation Step (model.eval() uses pure mu, no noise) ──
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        val_kld_accum = 0.0
        val_psnr_list = []
        val_ssim_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss, r_loss, k_loss = model.loss_function(recon, batch, mu, logvar, beta=beta, normalize_pixels=True)

                val_loss_accum += loss.item()
                val_mse = F.mse_loss(recon, batch).item()
                val_mse_accum += val_mse

                kld_raw = float((-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()) / batch.size(0)
                val_kld_accum += kld_raw

                # PSNR & SSIM on a small subset for speed
                b_orig = batch.cpu().permute(0, 2, 3, 1).numpy()
                b_rec = recon.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
                for i in range(min(8, b_orig.shape[0])):
                    s = ssim_fn(b_orig[i], b_rec[i], channel_axis=2, data_range=1.0)
                    p = psnr_fn(b_orig[i], b_rec[i], data_range=1.0)
                    val_ssim_list.append(s)
                    val_psnr_list.append(p)

        val_loss = val_loss_accum / max(len(val_loader), 1)
        val_mse = val_mse_accum / max(len(val_loader), 1)
        val_kld = val_kld_accum / max(len(val_loader), 1)
        val_psnr = np.mean(val_psnr_list) if val_psnr_list else 10 * np.log10(1.0 / (val_mse + 1e-10))
        val_ssim = np.mean(val_ssim_list) if val_ssim_list else 0.5

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(float(val_psnr))
        history["val_ssim"].append(float(val_ssim))
        history["val_mse"].append(float(val_mse))
        history["val_kld"].append(float(val_kld))

        print(f" {epoch:5d} | {train_loss:11.5f} | {val_loss:11.5f} | {val_psnr:8.2f} dB | {val_ssim:7.4f} | {val_kld:7.1f} | {elapsed:5.1f}s")

        # Save best model
        if val_loss < best_val_loss or val_ssim > best_ssim:
            best_val_loss = min(best_val_loss, val_loss)
            best_ssim = max(best_ssim, val_ssim)

            save_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": {
                    "img_size": 128,
                    "latent_dim": 256,
                    "epochs": 100,
                    "calibrated_loss": "L1 + 0.5*MSE + Beta-Annealed KLD",
                    "final_ssim": float(val_ssim),
                    "final_psnr": float(val_psnr)
                }
            }
            torch.save(save_payload, OUT_DIR / "vae_model.pth")
            torch.save(save_payload, WEIGHTS_DIR / "vae_model.pth")

    print("=" * 78)
    print(f"[COMPLETE] Training Complete! Best Val SSIM: {best_ssim:.4f}")

    # ── 3. Generate Dashboard Artifacts ──
    print("Generating updated dashboard evaluation artifacts...")
    try:
        # A. Loss Curve
        plt.figure(figsize=(10, 4.5))
        plt.plot(history["train_loss"], label="Train Loss (L1 + 0.5*MSE)", color="#00f0ff", lw=2)
        plt.plot(history["val_loss"], label="Val Loss", color="#a855f7", lw=2, linestyle="--")
        plt.title("Calibrated VAE Training Loss Curve", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "vae_loss_curve.png", dpi=150)
        plt.close()

        # B. Generative Samples from Prior
        model.eval()
        with torch.no_grad():
            z_sample = torch.randn(8, 256, device=device) * 0.85
            gen_imgs = model.decode(z_sample).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
        for i in range(8):
            axes[i].imshow(gen_imgs[i])
            axes[i].axis("off")
            axes[i].set_title(f"Gen {i+1}", fontsize=9)
        plt.suptitle("VAE Generative Sampling: z ~ N(0, I) -> Decoder", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "vae_generative_samples.png", dpi=150)
        plt.close()

        # C. Original vs Reconstructed
        sample_batch = next(iter(val_loader))[:5].to(device)
        with torch.no_grad():
            recon_batch, _, _ = model(sample_batch)
        b_in = sample_batch.cpu().permute(0, 2, 3, 1).numpy()
        b_out = recon_batch.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i in range(5):
            axes[0, i].imshow(b_in[i])
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Original {i+1}", fontsize=9)
            axes[1, i].imshow(b_out[i])
            axes[1, i].axis("off")
            axes[1, i].set_title(f"100e VAE Rec {i+1}", fontsize=9)
        plt.suptitle("Calibrated VAE: Original vs Reconstructed", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "vae_original_vs_reconstructed.png", dpi=150)
        plt.close()

        print(" [OK] Generated vae_loss_curve.png, vae_generative_samples.png, and vae_original_vs_reconstructed.png")
    except Exception as e:
        print(f" [WARN] Artifact plotting error: {e}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warm_start", action="store_true", default=True)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, warm_start=args.warm_start)
