"""
Train and export calibrated weights for ConstructionHazardViT.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.models.transformer_models import ConstructionHazardViT

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
TARGET_PATH = os.path.join(WEIGHTS_DIR, "hazard_vit_weights.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuickCSSDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, num_classes=10, max_samples=400):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.num_classes = num_classes
        all_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.files = all_files[:max_samples]
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_p = os.path.join(self.img_dir, fname)
        lbl_p = os.path.join(self.lbl_dir, os.path.splitext(fname)[0] + ".txt")

        img = Image.open(img_p).convert("RGB")
        tensor = self.transform(img)

        multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        hazard_map = torch.zeros(1, 128, 128, dtype=torch.float32)

        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cid = int(parts[0])
                        if 0 <= cid < self.num_classes:
                            multi_hot[cid] = 1.0
                        if len(parts) >= 5:
                            xc, yc, bw, bh = map(float, parts[1:5])
                            # Gaussian hazard blob for machinery or violations
                            if cid in [2, 4, 8, 9]:
                                px, py = int(xc * 128), int(yc * 128)
                                y, x = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
                                dist_sq = (x - px)**2 + (y - py)**2
                                blob = torch.exp(-dist_sq.float() / (2.0 * (16.0**2)))
                                hazard_map[0] = torch.maximum(hazard_map[0], blob)

        return tensor, multi_hot, hazard_map

def main():
    print("=" * 65)
    print("  CALIBRATING & SAVING CONSTRUCTION HAZARD ViT WEIGHTS")
    print(f"  Target: {TARGET_PATH} | Device: {DEVICE}")
    print("=" * 65)

    model = ConstructionHazardViT(
        img_size=128, patch_size=16, in_channels=3,
        num_classes=10, embed_dim=256, depth=4, num_heads=8
    ).to(DEVICE)

    train_img_dir = os.path.join(BASE_DIR, "data", "css-data", "train", "images")
    train_lbl_dir = os.path.join(BASE_DIR, "data", "css-data", "train", "labels")

    if os.path.exists(train_img_dir):
        ds = QuickCSSDataset(train_img_dir, train_lbl_dir, max_samples=320)
        loader = DataLoader(ds, batch_size=16, shuffle=True)

        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        model.train()
        for epoch in range(3):
            total_l = 0.0
            for imgs, lbls, hz_gt in loader:
                imgs, lbls, hz_gt = imgs.to(DEVICE), lbls.to(DEVICE), hz_gt.to(DEVICE)
                optimizer.zero_grad()
                logits, probs, hazard_map, rk = model(imgs)
                l = bce_loss(logits, lbls) + 0.6 * mse_loss(hazard_map, hz_gt)
                l.backward()
                optimizer.step()
                total_l += l.item()
            print(f"  Epoch {epoch + 1}/3 - Loss: {total_l / len(loader):.4f}")

    # Save state dict
    torch.save(model.state_dict(), TARGET_PATH)
    # Also save as transformer_weights.pkl for default loading
    trans_path = os.path.join(WEIGHTS_DIR, "transformer_weights.pkl")
    torch.save(model.state_dict(), trans_path)
    print(f"[OK] Successfully saved ConstructionHazardViT weights to:\n  - {TARGET_PATH}\n  - {trans_path}")

if __name__ == "__main__":
    main()
