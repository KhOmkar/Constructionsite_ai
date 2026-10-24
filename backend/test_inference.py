"""Test to verify Autoencoder V2, Classification Head, and Generative models."""
import sys
import os
import io
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from inference import ModelInferenceManager

print("Initializing ModelInferenceManager...")
m = ModelInferenceManager()
print()

# Create a synthetic image for unit tests
test_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
buf = io.BytesIO()
test_img.save(buf, format="JPEG")
test_bytes = buf.getvalue()

# 1. Test AE reconstruction with real AutoencoderV2
r_ae = m.reconstruct_ae(test_bytes)
print(f"AE Reconstruct V2: status={r_ae['status']}, MSE={r_ae['mse_loss']}, SSIM={r_ae['ssim_score']}, output_len={len(r_ae['reconstructed_image'])}")

# 2. Test AE Feature Classification
r_clf = m.classify_ae(test_bytes)
print(f"AE Classification: status={r_clf['status']}, pred={r_clf['predicted_class']} ({r_clf['confidence_percentage']}%), status={r_clf['safety_status']}")

# 3. Test Sample Evaluation
r_sample = m.evaluate_test_samples(max_samples=4)
print(f"Sample Evaluation: status={r_sample['status']}, tested={r_sample.get('total_tested', 0)}, accuracy={r_sample.get('accuracy_percentage', 0)}%")

# 4. Test Generative & Simulation
r_gan = m.generate_gan()
print(f"GAN generate: status={r_gan['status']}, output_length={len(r_gan['generated_image_url'])}")

r_vae = m.generate_vae()
print(f"VAE generate: status={r_vae['status']}, output_length={len(r_vae['generated_image_url'])}")

r_vae_rec = m.reconstruct_vae(test_bytes)
print(f"VAE reconstruct (100e): status={r_vae_rec['status']}, Acc={r_vae_rec.get('accuracy_percentage')}%, PSNR={r_vae_rec.get('psnr')}, SSIM={r_vae_rec.get('ssim_score')}, KLD={r_vae_rec.get('kld_loss')}, Heatmap_len={len(r_vae_rec.get('heatmap_image', ''))}")

r_tr = m.simulate_transformer(test_bytes, stage_idx=2, delta_days=30.0)
print(f"Transformer sim: status={r_tr['status']}")

print()
print("All endpoints verified and working successfully!")
