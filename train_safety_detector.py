"""
Training Pipeline for Construction Site Safety & PPE Object Detector (YOLOv8).
Trained on 2,605 Real Images from Roboflow CSS & ConstructionSite 10k.

Classes (10):
  0: Hardhat, 1: Mask, 2: NO-Hardhat, 3: NO-Mask, 4: NO-Safety Vest,
  5: Person, 6: Safety Cone, 7: Safety Vest, 8: machinery, 9: vehicle
"""

import os
import sys
import shutil
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(BASE_DIR, "data", "construction_safety.yaml")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

TARGET_WEIGHTS = os.path.join(WEIGHTS_DIR, "safety_detector_best.pt")

def train_safety_detector(epochs=15, batch=16, imgsz=640):
    print("=" * 75)
    print("  TRAINING CONSTRUCTION SAFETY OBJECT DETECTOR (YOLOv8)")
    print(f"  Dataset Config : {YAML_PATH}")
    print(f"  Target Epochs  : {epochs} | Batch: {batch} | ImgSz: {imgsz}")
    print("=" * 75)

    # 1. Initialize pretrained YOLOv8n
    model = YOLO("yolov8n.pt")

    # 2. Train on the 10 construction safety classes
    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0, # GPU device
        workers=2,
        project=os.path.join(BASE_DIR, "runs", "train"),
        name="safety_yolo",
        exist_ok=True,
        save=True,
        verbose=True
    )

    # 3. Copy best weights to weights/safety_detector_best.pt
    best_pt = os.path.join(BASE_DIR, "runs", "train", "safety_yolo", "weights", "best.pt")
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, TARGET_WEIGHTS)
        print(f"\n[OK] Successfully saved best weights to: {TARGET_WEIGHTS}")
    else:
        last_pt = os.path.join(BASE_DIR, "runs", "train", "safety_yolo", "weights", "last.pt")
        if os.path.exists(last_pt):
            shutil.copy2(last_pt, TARGET_WEIGHTS)
            print(f"\n[OK] Successfully saved last weights to: {TARGET_WEIGHTS}")

    # 4. Evaluate on test set
    print("\n>>> Running validation benchmark on test set...")
    metrics = model.val(data=YAML_PATH, split="test")
    print(f"mAP@50: {metrics.box.map50:.4f} | mAP@50-95: {metrics.box.map:.4f}")
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_safety_detector(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
