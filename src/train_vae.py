"""
Standalone VAE Training Script for Construction Site AI.

Trains ConvVAE for 100 epochs (configurable) on the Roboflow dataset
and exports updated weights for both the notebook and backend inference server.
"""

import os
import sys
import glob
import time
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set up project path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.vae import ConvVAE

class ConstructionImageDataset(Dataset):
    """Loads construction site images from dataset directories."""
    def __init__(self, root_dirs, img_size=128):
        self.img_size = img_size
        self.image_paths = []
        
        for root in root_dirs:
            if not os.path.exists(root):
                continue
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
                self.image_paths.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))
                
        # Remove duplicates
        self.image_paths = sorted(list(set(self.image_paths)))
        print(f"Loaded {len(self.image_paths)} images for training.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            return self.transform(image)
        except Exception:
            # Fallback black image on corrupt file
            return torch.zeros(3, self.img_size, self.img_size)


def train(epochs=100, batch_size=32, lr=1e-3, latent_dim=256, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f" Training ConvVAE for {epochs} Epochs")
    print(f" Device      : {device}")
    print(f" Latent Dim  : {latent_dim}")
    print(f" Batch Size  : {batch_size}")
    print(f" Learning Rate: {lr}")
    print(f"==========================================\n")

    # Locate dataset
    data_roots = [
        r"C:\Users\mrpat\Downloads\Constructio_Site_AI\ConstructionSiteSafetyImageDatasetRoboflow",
        os.path.join(PROJECT_ROOT, "data"),
        os.path.join(PROJECT_ROOT, "ConstructionSiteSafetyImageDatasetRoboflow"),
    ]
    
    dataset = ConstructionImageDataset(data_roots, img_size=128)
    if len(dataset) == 0:
        print("[ERROR] No images found. Check data directory.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = ConvVAE(in_channels=3, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')
    history = {'loss': []}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, images in enumerate(loader):
            images = images.to(device)
            recon, mu, logvar = model(images)
            
            # ELBO loss (MSE reconstruction + KL divergence)
            loss, recon_loss, kld_loss = ConvVAE.loss_function(recon, images, mu, logvar, beta=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader.dataset)
        history['loss'].append(avg_loss)
        scheduler.step(avg_loss)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:3d}/{epochs}] | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint periodically & best
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save for backend
            backend_weight_path = os.path.join(PROJECT_ROOT, "weights", "vae_weights.pkl")
            torch.save(model.state_dict(), backend_weight_path)
            
            # Save for notebook/features directory
            notebook_weight_dir = os.path.join(PROJECT_ROOT, "data", "processed", "features")
            os.makedirs(notebook_weight_dir, exist_ok=True)
            notebook_weight_path = os.path.join(notebook_weight_dir, "vae_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'epochs': epoch,
                'latent_dim': latent_dim
            }, notebook_weight_path)

    print(f"\n[DONE] Training complete! Best Loss: {best_loss:.4f}")
    print(f"Saved weights to:")
    print(f" - {os.path.join(PROJECT_ROOT, 'weights', 'vae_weights.pkl')}")
    print(f" - {os.path.join(PROJECT_ROOT, 'data', 'processed', 'features', 'vae_model.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrain Construction VAE Model")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 0.001)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
