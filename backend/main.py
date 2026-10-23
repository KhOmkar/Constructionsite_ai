import os
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference import ModelInferenceManager

app = FastAPI(title="Construction AI API — Digital Twin & Safety Classification")

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the data directory to serve static images/metrics
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

# Initialize the inference manager
inference_manager = ModelInferenceManager()


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Construction Site AI Digital Twin API",
        "models": ["AutoencoderV2", "EncoderClassifier", "VAE", "CVAE", "GAN", "Transformer"],
        "device": str(inference_manager.device)
    }


# ==============================================================================
# Autoencoder & Classification Endpoints
# ==============================================================================

@app.post("/api/ae/reconstruct")
async def ae_reconstruct(file: UploadFile = File(...)):
    """Reconstruct an image using the trained Autoencoder V2."""
    content = await file.read()
    result = inference_manager.reconstruct_ae(content)
    return JSONResponse(content=result)


@app.post("/api/ae/classify")
async def ae_classify(file: UploadFile = File(...)):
    """
    Classify a construction site image / crop using the Autoencoder Feature Classifier.
    Predicts PPE compliance (Hardhat, Mask, Vest, NO-Hardhat, NO-Vest, etc.).
    """
    content = await file.read()
    result = inference_manager.classify_ae(content)
    return JSONResponse(content=result)


@app.get("/api/samples")
def get_sample_presets():
    """
    Returns curated real construction test sample images across all models:
    Autoencoder & Classification, VAE, GAN, and Transformer.
    """
    samples = inference_manager.get_preset_samples()
    return JSONResponse(content={"status": "success", "samples": samples})


@app.post("/api/samples/test_sample")
async def test_sample_by_id(
    sample_id: str = Form(...),
    model: str = Form("ae"),
    stage_idx: int = Form(1),
    delta_days: float = Form(30.0),
):
    """
    Runs model inference directly on a curated preset sample.
    Supports 'ae' (classification + reconstruction), 'vae', 'transformer', and 'gan'.
    """
    img_bytes = inference_manager.get_sample_bytes(sample_id)
    if img_bytes is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Sample {sample_id} not found"})

    if model == "ae":
        clf_res = inference_manager.classify_ae(img_bytes)
        recon_res = inference_manager.reconstruct_ae(img_bytes)
        clf_res["reconstruction"] = recon_res
        return JSONResponse(content={"status": "success", "model": "ae", "result": clf_res})
    elif model == "vae":
        recon_res = inference_manager.reconstruct_vae(img_bytes)
        return JSONResponse(content={"status": "success", "model": "vae", "result": recon_res})
    elif model == "transformer":
        sim_res = inference_manager.simulate_transformer(img_bytes, stage_idx, delta_days)
        return JSONResponse(content={"status": "success", "model": "transformer", "result": sim_res})
    elif model == "gan":
        gan_res = inference_manager.generate_gan()
        return JSONResponse(content={"status": "success", "model": "gan", "result": gan_res})
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Unknown model {model}"})


@app.get("/api/ae/sample_test")
def ae_sample_test(max_samples: int = Query(default=8, ge=1, le=20)):
    """
    Evaluate real test samples from the dataset with the trained classifier.
    Returns ground-truth vs model predictions and test accuracy.
    """
    result = inference_manager.evaluate_test_samples(max_samples=max_samples)
    return JSONResponse(content=result)


# ==============================================================================
# Generative & Simulation Endpoints
# ==============================================================================

@app.post("/api/vae/generate")
async def vae_generate():
    """Generate a novel image using the VAE"""
    result = inference_manager.generate_vae()
    return JSONResponse(content=result)


@app.post("/api/vae/reconstruct")
async def vae_reconstruct(file: UploadFile = File(...)):
    """Reconstruct an image using the VAE"""
    content = await file.read()
    result = inference_manager.reconstruct_vae(content)
    return JSONResponse(content=result)


@app.post("/api/vae/generate_conditional")
async def vae_generate_conditional(
    class_idx:   int   = Form(...),
    temperature: float = Form(default=1.0),
):
    """
    Conditional VAE generation conditioned on a specific safety class.
    """
    result = inference_manager.generate_cvae(class_idx=class_idx, temperature=temperature)
    return JSONResponse(content=result)


@app.post("/api/gan/generate")
async def gan_generate():
    """Generate a high-fidelity synthetic image using GAN"""
    result = inference_manager.generate_gan()
    return JSONResponse(content=result)


@app.post("/api/transformer/simulate")
async def transformer_simulate(
    file: UploadFile = File(...),
    stage_idx: int = Form(...),
    delta_days: float = Form(...)
):
    """Simulate future construction progress and risk map"""
    content = await file.read()
    result = inference_manager.simulate_transformer(content, stage_idx, delta_days)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
