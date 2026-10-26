"""
Test FastAPI backend endpoints including all Transformer routes.
"""
import os
import sys
import io
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_endpoints():
    print("=" * 60)
    print("  Testing FastAPI Backend & Transformer Endpoints")
    print("=" * 60)

    # 1. Test Root
    res = client.get("/")
    assert res.status_code == 200, f"Root endpoint failed: {res.status_code}"
    print(f"[OK] GET / -> {res.json()}")

    # 2. Test Presets
    res = client.get("/api/transformer/presets")
    assert res.status_code == 200, f"Presets endpoint failed: {res.status_code}"
    presets = res.json()
    assert isinstance(presets, list) and len(presets) > 0, "Presets list empty"
    print(f"[OK] GET /api/transformer/presets -> {len(presets)} presets available:")
    for p in presets:
        print(f"     - Preset ID: {p.get('id')} | Title: {p.get('title')} | Stage: {p.get('stage_idx')} | Days: {p.get('days')}")

    # 3. Test Transformer Simulate with a dummy image
    dummy_img = Image.fromarray(np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8))
    buf = io.BytesIO()
    dummy_img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_site.jpg", buf, "image/jpeg")}
    data = {"stage_idx": "2", "delta_days": "45.0"}
    res = client.post("/api/transformer/simulate", files=files, data=data)
    assert res.status_code == 200, f"Simulate endpoint failed: {res.status_code}, {res.text}"
    sim_data = res.json()
    assert sim_data.get("status") == "success", f"Sim status not success: {sim_data}"
    assert "future_sim_image" in sim_data and sim_data["future_sim_image"].startswith("data:image/"), "Missing future_sim_image"
    assert "risk_map_image" in sim_data and sim_data["risk_map_image"].startswith("data:image/"), "Missing risk_map_image"
    assert "attention_map_image" in sim_data and sim_data["attention_map_image"].startswith("data:image/"), "Missing attention_map_image"
    assert "metrics" in sim_data, "Missing metrics"

    m = sim_data["metrics"]
    print(f"[OK] POST /api/transformer/simulate -> Status: {sim_data['status']}")
    print(f"     - Future Sim Image Base64 Length : {len(sim_data['future_sim_image'])} chars")
    print(f"     - Risk Map Image Base64 Length   : {len(sim_data['risk_map_image'])} chars")
    print(f"     - Attention Map Base64 Length    : {len(sim_data['attention_map_image'])} chars")
    print(f"     - Metrics: SSIM={m.get('ssim')}, PSNR={m.get('psnr')} dB, MSE={m.get('mse')}, Mean Risk={m.get('mean_risk')}, Peak Risk={m.get('peak_risk')}, Attn Focus={m.get('attention_focus_ratio')}")

    print("=" * 60)
    print("  [SUCCESS] All FastAPI Transformer Endpoints Validated Cleanly!")
    print("=" * 60)

if __name__ == "__main__":
    test_endpoints()
