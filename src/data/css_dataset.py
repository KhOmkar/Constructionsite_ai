"""
Roboflow Construction Site Safety (CSS) Dataset Loader.
Loads 2,801 high-resolution images and YOLOv8 annotations:
  0: Hardhat
  1: Mask
  2: NO-Hardhat
  3: NO-Mask
  4: NO-Safety Vest
  5: Person
  6: Safety Cone
  7: Safety Vest
  8: machinery
  9: vehicle
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

CSS_CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
]

VIOLATION_CLASSES = {2: "NO-Hardhat", 3: "NO-Mask", 4: "NO-Safety Vest"}
MACHINERY_CLASSES = {8: "machinery", 9: "vehicle"}


class CSSDataset(Dataset):
    def __init__(self, root_dir=None, split="train", img_size=128, max_samples=None):
        super().__init__()
        self.img_size = img_size
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if root_dir is None:
            root_dir = os.path.join(base_dir, "data", "css-data")
            if not os.path.exists(root_dir):
                root_dir = r"D:\Constructio_Site_AI(1)\Constructio_Site_AI\ConstructionSiteSafetyImageDatasetRoboflow\css-data"
            
        self.img_dir = os.path.join(root_dir, split, "images")
        self.lbl_dir = os.path.join(root_dir, split, "labels")
        
        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")) + 
                                  glob.glob(os.path.join(self.img_dir, "*.png")))
        
        if max_samples and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
            
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.raw_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def _parse_yolo_labels(self, label_path):
        labels_multi_hot = torch.zeros(10, dtype=torch.float32)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        if 0 <= cls_id < 10:
                            labels_multi_hot[cls_id] = 1.0
                            boxes.append((cls_id, cx, cy, w, h))
        return labels_multi_hot, boxes

    def _build_ground_truth_hazard_map(self, boxes):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, self.img_size),
            torch.linspace(0, 1, self.img_size),
            indexing='ij'
        )
        hazard = torch.zeros(self.img_size, self.img_size, dtype=torch.float32)
        
        for cls_id, cx, cy, w, h in boxes:
            if cls_id in VIOLATION_CLASSES:
                weight = 1.0
                sigma = max(max(w, h) * 0.75, 0.08)
            elif cls_id in MACHINERY_CLASSES:
                weight = 0.85
                sigma = max(max(w, h) * 0.90, 0.12)
            elif cls_id == 5:
                weight = 0.40
                sigma = 0.06
            else:
                weight = 0.15
                sigma = 0.05
                
            dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2
            patch = weight * torch.exp(-dist_sq / (2 * (sigma**2)))
            hazard = torch.maximum(hazard, patch)
            
        return hazard.unsqueeze(0)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, stem + ".txt")
        
        img = Image.open(img_path).convert("RGB")
        norm_img = self.transform(img)
        raw_img = self.raw_transform(img)
        
        multi_hot, boxes = self._parse_yolo_labels(lbl_path)
        hazard_map = self._build_ground_truth_hazard_map(boxes)
        
        return {
            "image": norm_img,
            "raw_image": raw_img,
            "labels": multi_hot,
            "hazard_map": hazard_map,
            "boxes_count": len(boxes),
            "filename": os.path.basename(img_path)
        }


def get_css_dataloaders(batch_size=32, img_size=128, max_train=None):
    train_ds = CSSDataset(split="train", img_size=img_size, max_samples=max_train)
    valid_ds = CSSDataset(split="valid", img_size=img_size)
    test_ds = CSSDataset(split="test", img_size=img_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
