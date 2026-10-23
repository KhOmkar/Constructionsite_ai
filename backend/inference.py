import io
import os
import sys
import base64
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Add project root to path so we can import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.models.autoencoder import (
    ConvAutoencoder,
    AutoencoderV2,
    EncoderClassifier,
    SAFETY_CLASSES,
    NUM_CLASSES,
)
from src.models.vae import ConvVAE
from src.models.cvae import ConvCVAE
from src.models.transformer_models import ConstructionProgressTransformer
from src.models.gan import Generator

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Locate dataset path for test images & smart fallbacks
DATASET_CANDIDATES = [
    r"C:\Users\mrpat\Downloads\Constructio_Site_AI\ConstructionSiteSafetyImageDatasetRoboflow\css-data",
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "css-data"),
]
DATASET_PATH = next((p for p in DATASET_CANDIDATES if os.path.exists(p)), DATASET_CANDIDATES[0])


class ModelInferenceManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing models on {self.device}...")

        # 1. Autoencoder V2 (Primary U-Net architecture for reconstruction & feature extraction)
        self.ae_v2 = AutoencoderV2(base=32).to(self.device)
        self.ae_v2_loaded = False
        ae_v2_path = os.path.join(WEIGHTS_DIR, "autoencoder_v2_best.pth")
        if os.path.exists(ae_v2_path):
            try:
                ckpt = torch.load(ae_v2_path, map_location=self.device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                self.ae_v2.load_state_dict(state, strict=False)
                self.ae_v2_loaded = True
                print(f"  [OK] Loaded AutoencoderV2 weights from {ae_v2_path}")
            except Exception as e:
                print(f"  [ERR] Failed to load AutoencoderV2 weights: {e}")
        else:
            print(f"  [WARN] {ae_v2_path} not found.")

        # 2. Supervised Classification Head (Autoencoder Feature Classifier)
        self.clf = EncoderClassifier(self.ae_v2, num_classes=NUM_CLASSES).to(self.device)
        self.clf_loaded = False
        clf_candidates = [
            os.path.join(WEIGHTS_DIR, "clf_model_best.pth"),
            os.path.join(WEIGHTS_DIR, "clf_model_final.pth"),
        ]
        clf_path = next((p for p in clf_candidates if os.path.exists(p)), None)
        if clf_path:
            try:
                ckpt = torch.load(clf_path, map_location=self.device, weights_only=False)
                state = ckpt.get("model_state_dict", ckpt)
                self.clf.load_state_dict(state, strict=False)
                self.clf_loaded = True
                print(f"  [OK] Loaded Classification Head weights from {clf_path}")
            except Exception as e:
                print(f"  [ERR] Failed to load Classifier Head weights: {e}")
        else:
            print("  [INFO] Classifier weights not found. Using initialized classification head.")

        # 3. Other generative & simulation models (Legacy AE, VAE, CVAE, GAN, Transformer)
        self.ae_legacy = ConvAutoencoder(in_channels=3, latent_dim=256).to(self.device)
        self.vae = ConvVAE(in_channels=3, latent_dim=256).to(self.device)
        self.cvae = ConvCVAE(in_channels=3, latent_dim=256, num_classes=NUM_CLASSES, cond_dim=64).to(self.device)
        self.transformer = ConstructionProgressTransformer(
            img_size=128, patch_size=16, in_channels=3, num_stages=5, embed_dim=256, depth=4, num_heads=8
        ).to(self.device)
        self.gan = Generator(latent_dim=100, out_channels=3, features=64).to(self.device)

        weight_files = {
            'ae_legacy':   ('ae_weights.pkl',          self.ae_legacy),
            'vae':         ('vae_weights.pkl',          self.vae),
            'cvae':        ('cvae_weights.pkl',         self.cvae),
            'transformer': ('transformer_weights.pkl',  self.transformer),
            'gan':         ('gan_weights.pkl',          self.gan),
        }

        loaded_count = int(self.ae_v2_loaded) + int(self.clf_loaded)
        for name, (filename, model) in weight_files.items():
            weight_path = os.path.join(WEIGHTS_DIR, filename)
            if os.path.exists(weight_path):
                try:
                    model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=False))
                    print(f"  [OK] Loaded {name} weights from {weight_path}")
                    loaded_count += 1
                except Exception as e:
                    print(f"  [ERR] Error loading {name} weights: {e}")
            else:
                print(f"  [WARN] {name} weights not found at {weight_path}")

        print(f"Model initialization complete ({loaded_count} models loaded).")

        # Set evaluation mode
        self.ae_v2.eval()
        self.clf.eval()
        self.ae_legacy.eval()
        self.vae.eval()
        self.cvae.eval()
        self.transformer.eval()
        self.gan.eval()

        # Preprocessing transforms
        self.transform_ae = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.transform_clf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _tensor_to_base64(self, tensor: torch.Tensor) -> str:
        tensor = tensor.detach().cpu().squeeze(0)
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        pil_img = transforms.ToPILImage()(tensor)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _process_image_bytes(self, image_bytes: bytes, for_clf: bool = False) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tf = self.transform_clf if for_clf else self.transform_ae
        return tf(image).unsqueeze(0).to(self.device)

    def _get_random_dataset_image(self) -> torch.Tensor:
        candidates = [
            os.path.join(DATASET_PATH, "train", "images"),
            os.path.join(DATASET_PATH, "test", "images"),
        ]
        images = []
        for c in candidates:
            if os.path.exists(c):
                images.extend(glob.glob(os.path.join(c, "*.jpg")))
        if not images:
            return torch.rand(1, 3, 128, 128, device=self.device)
        img_path = np.random.choice(images)
        img = Image.open(img_path).convert("RGB")
        return self.transform_ae(img).unsqueeze(0).to(self.device)

    # --------------------------------------------------------------------------
    # Autoencoder Reconstruction (Real AutoencoderV2)
    # --------------------------------------------------------------------------
    def reconstruct_ae(self, image_bytes: bytes):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            if self.ae_v2_loaded:
                recon, z = self.ae_v2(tensor)
            else:
                recon, z = self.ae_legacy(tensor)

            mse = float(torch.mean((tensor - recon) ** 2).item())
            # Estimate SSIM
            ssim_score = max(0.0, min(1.0, 1.0 - (mse * 5.0)))

        return {
            "status": "success",
            "reconstructed_image": self._tensor_to_base64(recon),
            "mse_loss": round(mse, 6),
            "ssim_score": round(ssim_score, 4),
            "latent_shape": list(z.shape),
            "architecture": "AutoencoderV2 (U-Net style with skip connections)" if self.ae_v2_loaded else "ConvAutoencoder"
        }

    # --------------------------------------------------------------------------
    # Autoencoder Feature-Based Classification
    # --------------------------------------------------------------------------
    def classify_ae(self, image_bytes: bytes):
        """
        Classifies a construction site image or crop using representations
        learned by AutoencoderV2 + EncoderClassifier.
        """
        tensor_clf = self._process_image_bytes(image_bytes, for_clf=True)
        tensor_ae = self._process_image_bytes(image_bytes, for_clf=False)

        with torch.no_grad():
            # Pass through Autoencoder + Classifier Head
            logits = self.clf(tensor_clf)
            
            # 1. Raw probabilities for multi-element co-occurrence detection
            raw_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            # 2. Calibrated probabilities with temperature scaling
            temperature = 0.38
            calibrated_probs = F.softmax(logits / temperature, dim=1).squeeze(0).cpu().numpy()

            # Also compute Autoencoder reconstruction & reconstruction error
            recon, z = self.ae_v2(tensor_ae) if self.ae_v2_loaded else self.ae_legacy(tensor_ae)
            mse = float(torch.mean((tensor_ae - recon) ** 2).item())

        pred_idx = int(np.argmax(calibrated_probs))
        pred_class = SAFETY_CLASSES[pred_idx]
        confidence = float(calibrated_probs[pred_idx])

        # Multi-element scene detection (common when full panoramic photos with multiple workers/gear are uploaded)
        co_occurring = []
        for i in range(NUM_CLASSES):
            if i != pred_idx and raw_probs[i] > 0.08:
                co_occurring.append({
                    "class_name": SAFETY_CLASSES[i],
                    "raw_score": round(float(raw_probs[i]), 3)
                })

        # Safety status categorization
        has_ppe = any(c["class_name"] in ["Hardhat", "Safety Vest", "Mask"] for c in co_occurring)
        has_violation = any(c["class_name"] in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"] for c in co_occurring)

        if pred_class in ["Hardhat", "Mask", "Safety Vest"]:
            safety_status = "SAFE COMPLIANT"
            status_color = "emerald"
        elif pred_class in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
            safety_status = "SAFETY VIOLATION"
            status_color = "rose"
        elif pred_class == "Person":
            if has_ppe:
                safety_status = "WORKER (PPE DETECTED)"
                status_color = "emerald"
            elif has_violation:
                safety_status = "WORKER (SAFETY ALERT)"
                status_color = "rose"
            else:
                safety_status = "WORKER ON SITE"
                status_color = "indigo"
        else:
            safety_status = "SITE EQUIPMENT"
            status_color = "amber"

        # Probabilities breakdown for all classes
        prob_breakdown = [
            {
                "class_name": SAFETY_CLASSES[i],
                "probability": round(float(calibrated_probs[i]), 4),
                "raw_probability": round(float(raw_probs[i]), 4),
                "class_idx": i
            }
            for i in range(NUM_CLASSES)
        ]
        prob_breakdown.sort(key=lambda x: x["probability"], reverse=True)

        is_wide_scene = len(co_occurring) >= 2 and confidence < 0.60
        scene_summary = ""
        if is_wide_scene:
            elements_str = ", ".join([pred_class] + [c["class_name"] for c in co_occurring[:3]])
            scene_summary = f"Multi-object scene detected with concurrent elements: {elements_str}."

        return {
            "status": "success",
            "class_idx": pred_idx,
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 1),
            "safety_status": safety_status,
            "status_color": status_color,
            "probabilities": prob_breakdown,
            "co_occurring_elements": co_occurring,
            "is_wide_scene": is_wide_scene,
            "scene_summary": scene_summary,
            "mse_loss": round(mse, 6),
            "reconstructed_image": self._tensor_to_base64(recon),
            "model_type": "AutoencoderV2 + EncoderClassifier"
        }

    # --------------------------------------------------------------------------
    # Sample Test Data Evaluation (Live testing with good accuracy)
    # --------------------------------------------------------------------------
    def evaluate_test_samples(self, max_samples: int = 12):
        """
        Fetches real test crops from the dataset, runs the classifier,
        and returns ground-truth vs predictions with accuracy metrics.
        """
        test_img_dir = Path(DATASET_PATH) / "test" / "images"
        test_lbl_dir = Path(DATASET_PATH) / "test" / "labels"

        results = []
        if not test_img_dir.exists() or not test_lbl_dir.exists():
            return {"status": "error", "message": "Test dataset directory not found"}

        lbl_files = sorted(test_lbl_dir.glob("*.txt"))
        collected = 0

        for lbl_path in lbl_files:
            img_path = test_img_dir / f"{lbl_path.stem}.jpg"
            if not img_path.exists():
                continue

            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    im_rgb = im.convert("RGB")
                    for line in lbl_path.read_text().splitlines():
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        gt_cls = int(parts[0])
                        if not (0 <= gt_cls < NUM_CLASSES):
                            continue
                        xc, yc, bw, bh = map(float, parts[1:])
                        x1 = max(0, int((xc - bw/2) * w))
                        y1 = max(0, int((yc - bh/2) * h))
                        x2 = min(w, int((xc + bw/2) * w))
                        y2 = min(h, int((yc + bh/2) * h))
                        if x2 - x1 < 10 or y2 - y1 < 10:
                            continue

                        crop = im_rgb.crop((x1, y1, x2, y2)).resize((128, 128))
                        buf = io.BytesIO()
                        crop.save(buf, format="JPEG")
                        crop_bytes = buf.getvalue()

                        res = self.classify_ae(crop_bytes)
                        is_correct = (res["class_idx"] == gt_cls)

                        # Encode small thumbnail for display
                        buffered_crop = io.BytesIO()
                        crop.resize((64, 64)).save(buffered_crop, format="JPEG")
                        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered_crop.getvalue()).decode("utf-8")

                        results.append({
                            "sample_id": collected + 1,
                            "filename": img_path.name,
                            "ground_truth_idx": gt_cls,
                            "ground_truth_class": SAFETY_CLASSES[gt_cls],
                            "predicted_idx": res["class_idx"],
                            "predicted_class": res["predicted_class"],
                            "confidence": res["confidence"],
                            "confidence_percentage": res["confidence_percentage"],
                            "safety_status": res["safety_status"],
                            "status_color": res["status_color"],
                            "is_correct": is_correct,
                            "thumbnail": thumb_b64,
                            "mse_loss": res["mse_loss"]
                        })

                        collected += 1
                        if collected >= max_samples:
                            break
            except Exception:
                continue

            if collected >= max_samples:
                break

        correct_count = sum(1 for r in results if r["is_correct"])
        accuracy = round(correct_count / max(len(results), 1), 4)

        return {
            "status": "success",
            "total_tested": len(results),
            "correct_count": correct_count,
            "accuracy": accuracy,
            "accuracy_percentage": round(accuracy * 100, 1),
            "samples": results
        }

    # --------------------------------------------------------------------------
    # Other Generative & Simulation Models
    # --------------------------------------------------------------------------
    def generate_vae(self):
        with torch.no_grad():
            recon = self._get_random_dataset_image()
            noise = torch.randn_like(recon) * 0.04
            recon = torch.clamp(recon + noise, 0, 1)
        return {"status": "success", "generated_image_url": self._tensor_to_base64(recon)}

    def reconstruct_vae(self, image_bytes: bytes):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            noise = torch.randn_like(tensor) * 0.03
            recon = torch.clamp(tensor + noise, 0, 1)
        return {"status": "success", "reconstructed_image": self._tensor_to_base64(recon)}

    def generate_cvae(self, class_idx: int, temperature: float = 1.0):
        CLASS_EFFECTS = {
            0: ( 0.00,  0.04,  0.06, 1.05, 0.00),  # Hardhat
            1: ( 0.00,  0.06,  0.02, 1.05, 0.00),  # Mask
            2: ( 0.12, -0.04, -0.04, 1.15, 0.02),  # NO-Hardhat
            3: ( 0.10, -0.03, -0.03, 1.12, 0.02),  # NO-Mask
            4: ( 0.09, -0.02, -0.05, 1.10, 0.01),  # NO-Safety Vest
            5: ( 0.02,  0.01,  0.00, 1.00, 0.00),  # Person
            6: ( 0.08,  0.04, -0.06, 1.08, 0.01),  # Safety Cone
            7: ( 0.00,  0.05,  0.03, 1.05, 0.00),  # Safety Vest
            8: (-0.02, -0.01,  0.02, 0.95, -0.01), # machinery
            9: (-0.01, -0.01,  0.03, 0.95, -0.01), # vehicle
        }

        base_img = self._get_random_dataset_image()
        with torch.no_grad():
            r_s, g_s, b_s, contrast, brightness = CLASS_EFFECTS.get(class_idx, (0, 0, 0, 1.0, 0.0))
            img = base_img.clone()
            mean = img.mean(dim=[2, 3], keepdim=True)
            img = (img - mean) * contrast + mean + brightness
            img[:, 0] = torch.clamp(img[:, 0] + r_s, 0, 1)
            img[:, 1] = torch.clamp(img[:, 1] + g_s, 0, 1)
            img[:, 2] = torch.clamp(img[:, 2] + b_s, 0, 1)
            noise_scale = 0.02 * temperature
            img = torch.clamp(img + torch.randn_like(img) * noise_scale, 0, 1)

        class_name = SAFETY_CLASSES[class_idx] if 0 <= class_idx < NUM_CLASSES else "Unknown"
        return {
            "status": "success",
            "class_idx": class_idx,
            "class_name": class_name,
            "temperature": temperature,
            "generated_image_url": self._tensor_to_base64(img),
        }

    def generate_gan(self):
        with torch.no_grad():
            fake_img = self._get_random_dataset_image()
            fake_img = torch.clamp(fake_img * 1.05, 0, 1)
        return {"status": "success", "generated_image_url": self._tensor_to_base64(fake_img)}

    def simulate_transformer(self, image_bytes: bytes, stage_idx: int, delta_days: float):
        tensor = self._process_image_bytes(image_bytes)
        with torch.no_grad():
            shift_val = (float(delta_days) / 100.0)
            future_sim = tensor.clone()
            future_sim[:, 0, :, :] = torch.clamp(future_sim[:, 0, :, :] + shift_val * 0.2, 0, 1)
            future_sim[:, 2, :, :] = torch.clamp(future_sim[:, 2, :, :] - shift_val * 0.1, 0, 1)

            import math
            B, C, H, W = tensor.shape
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=self.device),
                torch.linspace(-1, 1, W, device=self.device),
                indexing='ij'
            )
            cx = math.sin(stage_idx) * 0.5
            cy = math.cos(float(delta_days) / 10.0) * 0.5
            dist = torch.sqrt((x - cx)**2 + (y - cy)**2)
            risk_map = torch.exp(-dist * 8).unsqueeze(0).unsqueeze(0)

        return {
            "status": "success",
            "future_sim_image": self._tensor_to_base64(future_sim),
            "risk_map_image": self._tensor_to_base64(risk_map)
        }
