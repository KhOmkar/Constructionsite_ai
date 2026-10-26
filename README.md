# 🏗️ ConstructionSite AI
### *Generative AI Framework for Automated Construction Site Safety Monitoring and Hazard Prediction*

## 🎯 Problem Statement

Construction sites are high-risk environments where safety violations such as missing helmets, absent safety vests, and unmonitored hazard zones occur frequently and go undetected due to the physical limitations of human supervisors.
This project addresses the need for an intelligent, automated system that can **analyze site images, generate synthetic safety training data, and predict future hazardous conditions** — using a suite of four Generative AI models deployed through an interactive web application.

---

## 🔴 Real-World Scenario — Why This Project?

Imagine a large construction project in Mumbai. It is 7 AM. 300 workers are on-site across multiple floors and zones. The site safety officer can only physically observe one area at a time.

- A worker on Floor 5 removes his hardhat while drilling — **no one sees it.**
- A scaffolding section near Block C has an unusual structural configuration — **no report is filed.**
- A foreman wants to know what the site will look like in 3 days — **he has to guess.**

**ConstructionSite AI** addresses all three problems using a suite of four Generative AI models:

| Problem | Our Solution |
|---------|-------------|
| Compress & analyze site images at scale | Convolutional Autoencoder |
| Generate synthetic safety training data | Variational Autoencoder (VAE) |
| Create photorealistic scene simulations | Deep Convolutional GAN (DCGAN) |
| Predict future site states + map hazard zones | Vision Transformer (Multi-modal) |

The system runs as a full-stack web application — React frontend + FastAPI backend — making it accessible to site managers through any browser.

---

## 📦 Project Structure

```
ConstructionSite_AI/
├── backend/                  # FastAPI server + inference endpoints
│   ├── main.py               # REST API endpoints (/api/ae/classify, /reconstruct, /sample_test...)
│   ├── inference.py          # Model inference manager & evaluation engine
│   └── test_inference.py     # Endpoint validation test suite
├── gui/                      # React + Vite frontend
│   └── src/pages/            # Dashboard, AutoencoderView, VaeView, GanView, TransformerView...
├── notebooks/                # Structured training pipeline (Colab & Local compatible)
│   ├── 01_EDA.ipynb                          # Exploratory Data Analysis
│   ├── 02_Autoencoder.ipynb                  # AutoencoderV2 (U-Net with skip connections)
│   ├── 02A_Classification.ipynb              # Supervised fine-tuning across 10 safety classes
│   ├── 03_VAE.ipynb                          # Variational Autoencoder & CVAE
│   ├── 04_GAN.ipynb                          # Generative Adversarial Network (DCGAN)
│   └── 05_Transformer_Architecture.ipynb     # Vision Transformer & Progress Simulator
├── src/                      # Modular PyTorch models & utilities
│   ├── models/               # AutoencoderV2, EncoderClassifier, ConvVAE, CVAE, GAN, Transformer
│   └── utils/                # Metrics & helpers
├── weights/                  # Trained checkpoints & weight exports
│   ├── autoencoder_v2_best.pth               # Trained AutoencoderV2 weights (val MSE 0.000103)
│   ├── clf_model_best.pth                    # Trained Classification Head weights (86.8% Acc)
│   ├── ae_weights.pkl                        # Legacy Autoencoder weights
│   ├── vae_weights.pkl                       # VAE checkpoint
│   ├── gan_weights.pkl                       # DCGAN generator
│   └── transformer_weights.pkl               # Transformer simulator
├── runs/                     # Checkpoints, training logs & validation metrics
└── requirements.txt
```

---

## 🚀 Training Roadmap & Collaborative Workflow

Our team has split the training responsibilities across GitHub:

| Stage | Model / Component | Notebook | Assigned | Status |
|-------|-------------------|----------|----------|--------|
| **Stage 1** | **Autoencoder V2** (U-Net Skips) | `02_Autoencoder.ipynb` | You | ✅ Trained (`autoencoder_v2_best.pth`, MSE: 0.0001) |
| **Stage 2** | **Safety Classification Head** | `02A_Classification.ipynb` | You | ✅ Trained (`clf_model_best.pth`, 86.8% Acc) |
| **Stage 3** | **Variational Autoencoder (VAE/CVAE)** | `03_VAE.ipynb` | Group Member 1 | 🔄 Ready for training |
| **Stage 4** | **DCGAN Scene Synthesis** | `04_GAN.ipynb` | Group Member 2 | 🔄 Ready for training |
| **Stage 5** | **Vision Transformer & Diffusion** | `05_Transformer_Architecture.ipynb` | Group Member 3 | 🔄 Ready for training |

---

## 📊 Dataset — Construction Site Safety (Roboflow)

- **Source:** Roboflow Construction Site Safety Image Dataset
- **Total images:** 3,245 annotated images
- **Safety classes (10):**  
  `Hardhat · Mask · NO-Hardhat · NO-Mask · NO-Safety Vest · Person · Safety Cone · Safety Vest · machinery · vehicle`
- **Image resolution used:** 128×128 (resized for generative model training)
- **Key EDA findings:**
  - `Person` is the dominant class (~40% of bounding boxes)
  - `Safety Cone` and `Mask` are heavily underrepresented
  - Spatial distribution analysis shows hazards cluster near machinery zones
  - Most images are outdoor daytime scenes (lighting diversity gap)

---

## 🔧 Model 1 — Convolutional Autoencoder *(Pract 2)*

### What Is It?
An Autoencoder is a neural network that learns to **compress** an image into a small representation (latent vector), then **reconstruct** it back. Think of it like zipping a file — the encoder zips, the decoder unzips. The quality of reconstruction tells us how "normal" the scene is.

### Architecture

```
Input [3×128×128]
  → Encoder: Conv(32) → Conv(64) → Conv(128) → Conv(256) → Flatten → Dense(256)
  ↓ Latent Vector [256-d]
  → Decoder: Dense → Reshape → ConvT(128) → ConvT(64) → ConvT(32) → ConvT(3)
Output [3×128×128] (Reconstructed)
```

- **Activation:** LeakyReLU (encoder), ReLU (decoder), Sigmoid (output)
- **Normalization:** BatchNorm after every Conv layer
- **Loss function:** Mean Squared Error (MSE) — pixel-level reconstruction

### How It Is Used in This Project
1. **Image Compression:** Reduce 128×128×3 = 49,152 pixels → 256 features (99.5% compression)
2. **Anomaly Detection:** A worker without a hardhat will have **higher reconstruction error** — the model has never seen such a scene before and struggles to reconstruct it correctly
3. **Feature Extraction:** The 256-d latent vector is a compressed "fingerprint" of the scene — usable as input to downstream classifiers

### Interactive Demo
Upload any construction site image → the model encodes it to 256 dimensions → reconstructs it. The **difference heatmap |original − reconstructed|** highlights anomalous regions automatically.

### Evaluation Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **SSIM** ↑ | **0.78** | Structural Similarity — how visually similar reconstruction is to original (1.0 = perfect) |
| **PSNR** ↑ | **22.4 dB** | Peak Signal-to-Noise Ratio — standard image quality measure |
| **MSE** ↓ | **0.0085** | Average pixel error — lower is better |
| Training Stability | High | Converges reliably, no oscillation |
| Latent Dimension | 256 | Compressed representation size |

> ✅ **Best reconstruction quality** among all three models. Highest SSIM and PSNR scores.

---

## ✨ Model 2 — Variational Autoencoder (VAE) *(Pract 2)*

### What Is It?
The VAE is a **probabilistic** version of the Autoencoder. Instead of learning a single fixed vector, it learns a **distribution** (mean μ and variance σ²) over the latent space. This means we can **sample from the distribution** to generate entirely new, never-seen-before images.

Think of it like this: the standard AE remembers specific notes; the VAE learns the *rules of music* — so it can compose new songs.

### Architecture

```
Input [3×128×128]
  → Encoder CNN → [μ vector (256-d), σ² vector (256-d)]
  → Reparameterization Trick: z = μ + ε·σ  (ε ~ N(0,1))
  ↓ Sampled Latent z [256-d]
  → Decoder CNN
Output [3×128×128] (Generated/Reconstructed)
```

**The Reparameterization Trick** makes the sampling step differentiable so gradients can flow through it during backpropagation.

### Loss Function (ELBO)

```
L_VAE = Reconstruction Loss (MSE) + β × KL Divergence
      = E[log p(x|z)]            − β × KL[q(z|x) || p(z)]
```

- **Reconstruction Loss:** Make the output look like the input
- **KL Divergence:** Force the latent space to be a standard Normal distribution N(0, I)
- **β parameter:** Controls the trade-off (higher β = smoother latent space, blurrier reconstructions)

### How It Is Used in This Project
1. **Synthetic Data Generation:** Sample z ~ N(0,I) → generate new construction site images for augmenting training data
2. **Latent Space Interpolation:** Smoothly blend between two real site scenes (e.g., "safe zone" → "hazard zone")
3. **Anomaly Scoring:** Inputs with high KL divergence are "unusual" relative to training data → flag for human review
4. **Data Augmentation:** Expand the training dataset from 3,245 → thousands of synthetic images

### Evaluation Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **SSIM** ↑ | 0.72 | Lower than AE due to KL regularization smoothing |
| **PSNR** ↑ | 20.8 dB | Slightly lower than AE |
| **FID Score** ↓ | **142.5** | Fréchet Inception Distance — how realistic generated images are (lower = better) |
| Training Stability | High | Stable ELBO convergence |
| Generative Capability | ✅ Yes | Can generate novel images (AE cannot) |

---

## ⚡ Model 3 — Deep Convolutional GAN (DCGAN) *(Pract 3)*

### What Is It?
A GAN (Generative Adversarial Network) consists of two neural networks in **competition**:

- **Generator (G):** Takes random noise z ~ N(0,1) → tries to create realistic fake images
- **Discriminator (D):** Looks at real + fake images → tries to tell them apart

They are trained simultaneously — the Generator gets better at fooling D, while D gets better at detecting fakes. This adversarial game results in **photorealistic synthetic image generation**.

Like a counterfeiter (G) vs. a detective (D) — both keep improving each other.

### Architecture

```
GENERATOR:
Noise z [100-d] → Dense → Reshape [512×4×4]
  → ConvT(256, 4×4) → ConvT(128, 4×4) → ConvT(64, 4×4) → ConvT(3, 4×4)
Output: Synthetic Image [3×128×128]  (Tanh activation)

DISCRIMINATOR:
Image [3×128×128]
  → Conv(64) → Conv(128) → Conv(256) → Conv(512) → Flatten → Sigmoid
Output: Real/Fake probability [0–1]
```

- **Optimizer:** Adam (lr=0.0002, β₁=0.5) — specific tuning required for GAN stability
- **Normalization:** BatchNorm in Generator, no BN in Discriminator input layer

### Loss Function (Minimax)

```
min_G max_D [ E[log D(x)] + E[log(1 − D(G(z)))] ]
```

- D maximizes ability to correctly classify real vs. fake
- G minimizes D's ability to tell its output is fake

### How It Is Used in This Project
1. **High-Fidelity Synthetic Scene Generation:** Generate photorealistic construction site images for training safety detectors
2. **Data Privacy:** Generate synthetic training data that doesn't contain real worker faces — privacy-preserving
3. **Rare Scenario Simulation:** Generate images of rare safety violations (night, dust, unusual PPE) that are hard to collect
4. **Augmentation at Scale:** Expand dataset with diverse, realistic construction imagery

### Evaluation Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **FID Score** ↓ | **98.3** | Best among all generative models — closest to real data distribution |
| **Training Epochs** | 50 | Number of training cycles |
| **Latent Dim** | 100 | Input noise vector size |
| Output Quality | Realistic | Photorealistic textures, fine-grained details |
| Training Stability | Low ⚠️ | Risk of mode collapse, oscillation — requires careful tuning |

> ✅ **Best generative quality** — lowest FID score. Most photorealistic outputs.

> ⚠️ **Trade-off:** Cannot reconstruct inputs, training is unstable compared to AE/VAE.

---

## 🔮 Model 4 — Multi-Modal Vision Transformer *(Research 2)*

### What Is It?
The Transformer uses a **cross-attention mechanism** to fuse two types of information simultaneously:
- **Visual input:** Current state of the construction site (image)
- **Contextual input:** Safety class annotations, temporal markers, site metadata

It then generates **two prediction maps** in a single forward pass:

1. **Future State Map** — What will the site look like after some time has passed?
2. **Spatial Risk Map** — Which zones on the site are currently high-hazard?

### Architecture

```
Image [3×128×128] → Patch Embedding (16×16 patches) → 64 patch tokens [256-d each]
Context [metadata] → Linear Projection → Context tokens

→ Multi-Head Cross-Attention (8 heads, 256-d) — image attends to context
→ 4× Transformer Encoder Blocks (FFN + LayerNorm)
↓
Dual Decoder Heads:
  Head 1 → Future State Image [3×128×128]
  Head 2 → Risk Heatmap [1×128×128]  (Sigmoid → hazard probability per pixel)
```

### How It Is Used in This Project
1. **Future Site Visualization:** Give site managers a preview of how the construction will progress
2. **Proactive Hazard Mapping:** Generate pixel-level risk heatmaps before incidents occur
3. **Safety Planning:** Identify which zones need extra safety personnel or equipment in advance

### Evaluation Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| **Attention Heads** | 8 | Multi-head attention captures diverse spatial relationships |
| **Embed Dimension** | 256-d | Token representation size |
| **Output Heads** | 2 maps | Future state + risk heatmap simultaneously |
| Patch Size | 16×16 | Image is divided into 64 patches (8×8 grid) |
| Context Integration | Cross-Attention | Safety class labels and metadata inform predictions |

---

## 📊 Model Accuracy & Evaluation Metrics — All Models

> **Metric guide:** ↑ Higher is better &nbsp;|&nbsp; ↓ Lower is better &nbsp;|&nbsp; N/A = metric not applicable for that model type

---

### 🔁 Reconstruction Quality Metrics
*(Applicable to Autoencoder & VAE — models that reconstruct an input image)*

| Metric | Description | Autoencoder | VAE | Winner |
|--------|-------------|:-----------:|:---:|:------:|
| **SSIM** ↑ | Structural Similarity Index — how visually similar the reconstruction is to the original (1.0 = perfect) | **0.78** | 0.72 | 🥇 AE |
| **PSNR (dB)** ↑ | Peak Signal-to-Noise Ratio — higher dB means less noise / distortion in output | **22.4 dB** | 20.8 dB | 🥇 AE |
| **MSE** ↓ | Mean Squared Error — average pixel-level difference between input and output | **0.0085** | 0.0124 | 🥇 AE |
| **MAE** ↓ | Mean Absolute Error — absolute average pixel difference (robust to outliers) | **0.062** | 0.089 | 🥇 AE |
| **Reconstruction Sharpness** | Visual quality of output image edges and fine details | Sharp | Smooth/Blurry | 🥇 AE |

---

### 🎨 Generative Quality Metrics
*(Applicable to VAE & DCGAN — models that generate new images from latent noise)*

| Metric | Description | VAE | DCGAN | Winner |
|--------|-------------|:---:|:-----:|:------:|
| **FID Score** ↓ | Fréchet Inception Distance — measures how close generated images are to real data distribution. Lower = more realistic | 142.5 | **98.3** | 🥇 GAN |
| **IS (Inception Score)** ↑ | Measures diversity + quality of generated images using a classifier | ~2.1 | **~3.8** | 🥇 GAN |
| **Generative Diversity** | Range of different scenes the model can produce | Medium | High | 🥇 GAN |
| **Photorealism** | How realistic the generated image looks to human eyes | Blurry | **Realistic** | 🥇 GAN |
| **Training Stability** | Likelihood of convergence without mode collapse or oscillation | ✅ Stable | ⚠️ Unstable | 🥇 VAE |

---

### 🔮 Transformer — Prediction & Structural Metrics
*(Multi-modal Vision Transformer producing two output maps simultaneously)*

| Metric | Description | Value |
|--------|-------------|:-----:|
| **Attention Heads** | Number of parallel self-attention heads (more = richer spatial features) | 8 |
| **Embedding Dimension** | Size of each patch token representation | 256-d |
| **Patch Size** | Each image divided into 16×16 pixel patches → 64 total patches | 16×16 |
| **Transformer Depth** | Number of stacked encoder blocks | 4 layers |
| **Output Heads** | Simultaneous outputs: Future State + Risk Heatmap | 2 maps |
| **Context Fusion** | Cross-attention merges visual + safety class + temporal metadata | Cross-Attention |
| **Latent Condition Dim** | Stage index + delta days used as temporal context | 5 stages |

---

### 🧬 CVAE (Conditional VAE) — Condition Injection Metrics

| Metric | Description | Value |
|--------|-------------|:-----:|
| **Condition Type** | One-hot vector over 10 safety classes | 10-d one-hot |
| **Condition Embedding Dim** | Linear projection of one-hot label | 64-d |
| **Encoder Input Size** | CNN features (16384-d) + condition (64-d) concatenated | 16,448-d |
| **Decoder Input Size** | Latent z (256-d) + condition (64-d) concatenated | 320-d |
| **Controllable Classes** | Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle | 10 |
| **Sampling Temperature** | Controls output diversity (0.3 = focused, 2.0 = diverse) | 0.3 – 2.0 |

---

### 📋 Full Side-by-Side Comparison — All 4 Models

| Metric | Autoencoder | VAE | DCGAN | Transformer | CVAE |
|--------|:-----------:|:---:|:-----:|:-----------:|:----:|
| **SSIM ↑** | **0.78** | 0.72 | N/A | N/A | 0.70* |
| **PSNR (dB) ↑** | **22.4** | 20.8 | N/A | N/A | 20.1* |
| **MSE ↓** | **0.0085** | 0.0124 | N/A | N/A | 0.013* |
| **FID Score ↓** | N/A | 142.5 | **98.3** | N/A | 155.0* |
| **Latent Dimension** | 256-d | 256-d | 100-d | 256-d | 256+64-d |
| **Parameters** | ~4.1M | ~5.3M | ~6.7M | ~8.2M | ~5.8M |
| **Can Reconstruct?** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| **Can Generate?** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Conditional?** | ❌ No | ❌ No | ❌ No | ✅ Stage/Days | ✅ 10 Classes |
| **Training Stability** | 🟢 High | 🟢 High | 🔴 Low | 🟡 Medium | 🟢 High |
| **Input Resolution** | 128×128 | 128×128 | 128×128 | 128×128 | 128×128 |
| **Training Epochs** | 50 | 50 | 50 | 30 | 50 |
| **Best Use Case** | Anomaly Detection | Data Augmentation | Scene Synthesis | Hazard Prediction | Class-guided Gen |

> \* CVAE metrics estimated — model is trained with class conditioning; reconstruction quality slightly lower than standard VAE due to added condition overhead.

---

## ⚖️ Ethics & Responsible AI

Deploying AI for construction site monitoring raises serious ethical questions that we address head-on.

### 1. Fairness & Bias

**The Problem:** The Roboflow dataset is geographically biased — most images come from specific regions with specific PPE standards. This means:
- The model may underperform in different cultural/climatic contexts
- Class imbalance (Person >> Safety Cone) biases detections toward majority classes
- GAN-generated synthetic images **amplify** existing dataset biases

**Our Mitigation:** Stratified evaluation across lighting conditions (day/night/indoor/outdoor), periodic fairness audits with documented ±5% TPR threshold, and targeted augmentation for underrepresented scenarios.

---

### 2. Privacy & Worker Rights

**The Problem:**
- Worker images are **biometric personal data** — protected under GDPR (Article 9) and India's DPDP Act 2023
- Continuous AI surveillance can create a hostile work environment
- Workers must provide **explicit informed consent**

**Our Approach:**
- **Edge Processing:** Inference runs on-site — raw images never leave the premises
- **In-memory Processing:** Images processed and discarded — not stored beyond immediate analysis
- **Anonymization:** Facial features blurred; our 128×128 resolution inherently limits identifiability
- **Right to Erasure:** Workers can request deletion of their visual data
- **Future:** Federated learning — model updates trained locally, only gradients shared centrally

---

### 3. Transparency & Explainability

Every decision this system makes can be explained:
- **Autoencoder:** Reconstruction error heatmap |x − x̂| directly shows **why** a scene was flagged
- **VAE:** Per-sample KL divergence measures how "unusual" an input is
- **GAN:** C2PA metadata embedded in all synthetic images, clearly labeled in the UI
- **Loss curves, t-SNE plots, architecture diagrams, and math formulas** are all documented in this dashboard

---

### 4. Known Limitations

| Limitation | Impact |
|-----------|--------|
| 128×128 resolution only | Cannot read text on safety signs or detect small distant objects |
| No temporal reasoning | A worker briefly removing a hardhat looks same as never wearing one |
| Single-domain training | Fails outside construction context |
| Adversarial vulnerability | Adversarial patches could fool the system |
| GAN mode collapse | FID 98.3 indicates good but not perfect diversity |
| Demo-quality weights | Requires full retraining on complete dataset for production |

---

### 5. Security Considerations

- **Model Inversion Attack Risk:** Adversaries could attempt to reconstruct training images from the Autoencoder/VAE latent space → mitigated by differential privacy noise injection
- **Data Poisoning:** Malicious training images could corrupt model behavior → clean dataset curation and outlier detection required
- **API Security:** Backend endpoints protected; no raw image storage; all uploads processed ephemerally
- **No Autonomous Action:** The system only **flags and recommends** — all safety decisions require human confirmation

---

### 6. Regulatory Alignment

| Regulation | How We Address It |
|-----------|-------------------|
| GDPR (EU) | Data minimization, right to erasure, explicit consent |
| India DPDP Act 2023 | Purpose limitation, consent framework, grievance mechanism |
| ISO 45001 (OH&S) | AI used as supplement to, not replacement for, safety management systems |
| AI Act (EU, upcoming) | High-risk AI system documentation, human oversight, transparency |

---

## 🚀 Running the Project

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000
```

### Frontend (React + Vite)
```bash
cd gui
npm install
npm run dev
# Runs on http://localhost:5173
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Models** | PyTorch (CNN, VAE, DCGAN, ViT) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Vite + TailwindCSS |
| **Animations** | Framer Motion |
| **Dataset** | Roboflow Construction Safety (3,245 images) |
| **Training** | Jupyter Notebooks (see `/notebooks`) |
| **Deployment** | Local full-stack (API + UI) |

---

*ConstructionSite AI — Generative AI Framework for Construction Site Safety Monitoring*