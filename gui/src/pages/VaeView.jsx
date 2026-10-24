import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Loader2, Upload, ArrowRight, Hash, Cpu, Layers, Activity,
         Image as ImageIcon, FlaskConical, Thermometer, Zap, CheckCircle2, AlertTriangle } from 'lucide-react';
import axios from 'axios';
import SampleImageSelector from '../components/SampleImageSelector';

const API = 'http://localhost:8000';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

// ── Safety class definitions ──────────────────────────────────────────
const SAFETY_CLASSES = [
  { idx: 0, label: 'Hardhat',        type: 'safe',    emoji: '⛑️'  },
  { idx: 1, label: 'Mask',           type: 'safe',    emoji: '😷'  },
  { idx: 2, label: 'NO-Hardhat',     type: 'danger',  emoji: '🚨'  },
  { idx: 3, label: 'NO-Mask',        type: 'danger',  emoji: '⚠️'  },
  { idx: 4, label: 'NO-Safety Vest', type: 'danger',  emoji: '🔴'  },
  { idx: 5, label: 'Person',         type: 'neutral', emoji: '🧑'  },
  { idx: 6, label: 'Safety Cone',    type: 'safe',    emoji: '🔶'  },
  { idx: 7, label: 'Safety Vest',    type: 'safe',    emoji: '🦺'  },
  { idx: 8, label: 'machinery',      type: 'neutral', emoji: '🏗️' },
  { idx: 9, label: 'vehicle',        type: 'neutral', emoji: '🚛'  },
];

const CLASS_STYLE = {
  safe:    { border: 'border-emerald-500/40', bg: 'bg-emerald-500/10',  text: 'text-emerald-300',  badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',  ring: 'ring-emerald-400' },
  danger:  { border: 'border-rose-500/40',    bg: 'bg-rose-500/10',     text: 'text-rose-300',     badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',           ring: 'ring-rose-400'    },
  neutral: { border: 'border-cyan-500/40',    bg: 'bg-cyan-500/10',     text: 'text-cyan-300',     badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',        ring: 'ring-cyan-400'   },
};

export default function VaeView() {
  // Standard VAE state
  const [generatedUrl, setGeneratedUrl]         = useState(null);
  const [isGenerating, setIsGenerating]         = useState(false);
  const [selectedFile, setSelectedFile]         = useState(null);
  const [previewUrl, setPreviewUrl]             = useState(null);
  const [reconstructedUrl, setReconstructedUrl] = useState(null);
  const [isReconstructing, setIsReconstructing] = useState(false);
  const [reconstructionMetrics, setReconstructionMetrics] = useState(null);

  // CVAE conditional generation state
  const [selectedClass, setSelectedClass]       = useState(null);
  const [temperature, setTemperature]           = useState(1.0);
  const [cvaeUrl, setCvaeUrl]                   = useState(null);
  const [isCvaeGenerating, setIsCvaeGenerating] = useState(false);
  const [cvaeResult, setCvaeResult]             = useState(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API}/api/vae/generate`);
      setGeneratedUrl(response.data.generated_image_url);
    } catch (error) {
      console.error('Error generating image:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setReconstructedUrl(null);
    }
  };

  const handleReconstruct = async () => {
    if (!selectedFile) return;
    setIsReconstructing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await axios.post(`${API}/api/vae/reconstruct`, formData);
      setReconstructedUrl(response.data.reconstructed_image);
      setReconstructionMetrics(response.data.metrics || null);
    } catch (error) {
      console.error('Error reconstructing:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsReconstructing(false);
    }
  };

  // ── CVAE conditional generation handler ────────────────────────────
  const handleCvaeGenerate = async () => {
    if (selectedClass === null) return;
    setIsCvaeGenerating(true);
    setCvaeUrl(null);
    setCvaeResult(null);
    try {
      const formData = new FormData();
      formData.append('class_idx',   selectedClass.idx);
      formData.append('temperature', temperature);
      const response = await axios.post(`${API}/api/vae/generate_conditional`, formData);
      setCvaeUrl(response.data.generated_image_url);
      setCvaeResult(response.data);
    } catch (error) {
      console.error('CVAE generation error:', error);
      alert('Failed to connect to backend. Make sure server is running.');
    } finally {
      setIsCvaeGenerating(false);
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-6xl mx-auto"
    >
      {/* Header */}
      <header>
        <motion.div variants={itemVariants} className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-secondary to-purple-400 flex items-center justify-center shadow-lg shadow-secondary/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-3xl font-black gradient-text">Variational Autoencoder (VAE)</h2>
            <div className="flex items-center gap-2 flex-wrap mt-1">
              <span className="badge badge-secondary">PRACT 2 — Generative Modeling &amp; Latent Interpolation</span>
              <span className="badge bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 font-mono text-xs shadow-sm">
                ✓ 100 Epochs Trained (vae_model.pth Active)
              </span>
            </div>
          </div>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 mt-2 max-w-3xl">
          Probabilistic encoder maps inputs to a Gaussian distribution in latent space (μ, σ²). 
          The reparameterization trick enables gradient-based sampling: z = μ + ε·σ, where ε ~ N(0,I). 
          Novel construction scenes are generated by sampling directly from the prior z ~ N(0, I).
        </motion.p>
      </header>

      {/* Architecture Diagram */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          Network Architecture (with Reparameterization Trick)
        </div>
        <div className="arch-flow flex-wrap justify-center">
          <div className="arch-block arch-block-input">
            <div className="text-xs opacity-60">Input</div>
            <div className="font-bold">3×128×128</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-encoder">
            <div className="text-xs opacity-60">4× Conv+BN</div>
            <div className="font-bold">Encoder</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-encoder">
            <div className="text-xs opacity-60">FC → 256</div>
            <div className="font-bold">Flatten</div>
          </div>
          <span className="arch-arrow">→</span>

          {/* Split into mu and logvar */}
          <div className="flex flex-col gap-2 items-center">
            <div className="arch-block arch-block-latent" style={{ minWidth: '80px' }}>
              <div className="text-xs opacity-60">FC Head</div>
              <div className="font-bold">μ</div>
            </div>
            <div className="arch-block arch-block-latent" style={{ minWidth: '80px' }}>
              <div className="text-xs opacity-60">FC Head</div>
              <div className="font-bold">log σ²</div>
            </div>
          </div>

          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-latent animate-pulse-glow">
            <div className="text-xs opacity-60">z = μ + ε·σ</div>
            <div className="font-bold">Reparam.</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-decoder">
            <div className="text-xs opacity-60">FC → 256×8×8</div>
            <div className="font-bold">Expand</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-decoder">
            <div className="text-xs opacity-60">4× DeConv+BN</div>
            <div className="font-bold">Decoder</div>
          </div>
          <span className="arch-arrow">→</span>
          <div className="arch-block arch-block-output">
            <div className="text-xs opacity-60">Sigmoid</div>
            <div className="font-bold">3×128×128</div>
          </div>
        </div>
      </motion.div>

      {/* Model Info + Math */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Cpu className="w-4 h-4" /></div>
            Model Configuration
          </div>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Latent Dim</span>
              <span className="info-value">256</span>
            </div>
            <div className="info-item">
              <span className="info-label">Parameters</span>
              <span className="info-value">~5.3M</span>
            </div>
            <div className="info-item">
              <span className="info-label">Input Size</span>
              <span className="info-value">128×128</span>
            </div>
            <div className="info-item">
              <span className="info-label">Channels</span>
              <span className="info-value">3 (RGB)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Encoder</span>
              <span className="info-value">4 Conv</span>
            </div>
            <div className="info-item">
              <span className="info-label">μ, σ Heads</span>
              <span className="info-value">2 × FC</span>
            </div>
            <div className="info-item">
              <span className="info-label">β (KL weight)</span>
              <span className="info-value">1.0</span>
            </div>
            <div className="info-item">
              <span className="info-label">Prior</span>
              <span className="info-value">N(0, I)</span>
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Hash className="w-4 h-4" /></div>
            Mathematical Formulation (ELBO)
          </div>
          <div className="math-block">
            <div>
              <span className="math-fn">ELBO Loss</span><span className="math-op">: </span>
              <span className="math-var">L</span> <span className="math-op">=</span> <span className="math-var">L</span><span className="math-sub">recon</span> <span className="math-op">+</span> <span className="math-var">β</span> <span className="math-op">·</span> <span className="math-var">D</span><span className="math-sub">KL</span>
            </div>
            <div className="mt-3">
              <span className="math-fn">Reconstruction</span><span className="math-op">: </span>
              <span className="math-var">L</span><span className="math-sub">recon</span> <span className="math-op">=</span> <span className="math-fn">MSE</span>(<span className="math-var">x</span>, <span className="math-var">x̂</span>)
              <span className="math-op"> = </span>
              <span className="math-op">Σ</span>(<span className="math-var">x</span><span className="math-sub">i</span> <span className="math-op">−</span> <span className="math-var">x̂</span><span className="math-sub">i</span>)<span className="math-sup">2</span>
            </div>
            <div className="mt-3">
              <span className="math-fn">KL Divergence</span><span className="math-op">: </span>
              <span className="math-var">D</span><span className="math-sub">KL</span> <span className="math-op">=</span> <span className="math-op">−½ · Σ</span>(1 <span className="math-op">+</span> <span className="math-fn">log</span> <span className="math-var">σ</span><span className="math-sup">2</span> <span className="math-op">−</span> <span className="math-var">μ</span><span className="math-sup">2</span> <span className="math-op">−</span> <span className="math-var">σ</span><span className="math-sup">2</span>)
            </div>
            <div className="mt-3">
              <span className="math-fn">Reparameterization</span><span className="math-op">: </span>
              <span className="math-var">z</span> <span className="math-op">=</span> <span className="math-var">μ</span> <span className="math-op">+</span> <span className="math-var">ε</span> <span className="math-op">·</span> <span className="math-var">σ</span>
              <span className="math-op"> , where </span>
              <span className="math-var">ε</span> <span className="math-op">~</span> <span className="math-fn">N</span>(0, <span className="math-var">I</span>)
            </div>
            <span className="math-label">Maximize Evidence Lower Bound = minimize reconstruction + regularize latent to prior</span>
          </div>
        </motion.div>
      </div>

      {/* Preset Test Samples for VAE */}
      <SampleImageSelector
        onSelectSample={(sample, file) => {
          setSelectedFile(file);
          setPreviewUrl(sample.image);
          setReconstructedUrl(null);
          // Auto-trigger VAE reconstruction
          if (file) {
            setIsReconstructing(true);
            const formData = new FormData();
            formData.append('file', file);
            axios.post(`${API}/api/vae/reconstruct`, formData)
              .then(res => {
                setReconstructedUrl(res.data.reconstructed_image);
                setReconstructionMetrics(res.data.metrics || null);
              })
              .catch(err => console.error(err))
              .finally(() => setIsReconstructing(false));
          }
          // Also preselect matching class in CVAE if available
          if (sample.expected_class) {
            const matchCls = SAFETY_CLASSES.find(c => c.label.toLowerCase() === sample.expected_class.toLowerCase());
            if (matchCls) setSelectedClass(matchCls);
          }
        }}
        filterModel="vae"
        title="Sample Images For Testing VAE Reconstruction"
        subtitle="Select any test construction image below to compress into latent space and reconstruct with ConvVAE."
      />

      {/* Two-mode Inference: Generate + Reconstruct */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Generate Mode */}
        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Sparkles className="w-4 h-4" /></div>
            Generate Novel Sample (z ~ N(0, I))
          </div>
          <div className="flex flex-col items-center gap-5">
            <button onClick={handleGenerate} disabled={isGenerating} className="btn-gradient w-full">
              {isGenerating ? (
                <>Sampling <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Sample Latent Space <Sparkles className="w-5 h-5" /></>
              )}
            </button>
            <div className="w-full flex justify-center">
              <div className="w-56 h-56 rounded-2xl border border-white/10 bg-black/40 flex items-center justify-center overflow-hidden shadow-inner">
                {generatedUrl ? (
                  <motion.img initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                    src={generatedUrl} alt="Generated VAE Sample" className="w-full h-full object-cover" />
                ) : (
                  <div className="text-slate-600 text-sm font-medium flex flex-col items-center">
                    <Sparkles className="w-10 h-10 mb-2 opacity-20" />
                    Click to generate
                  </div>
                )}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Reconstruct Mode */}
        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><ImageIcon className="w-4 h-4" /></div>
            Reconstruct Uploaded Image
          </div>
          <div className="flex flex-col items-center gap-4">
            <div className="grid grid-cols-2 gap-4 w-full">
              {/* Input side */}
              <div className="flex flex-col items-center">
                <span className="text-[11px] font-mono text-slate-400 mb-1.5 uppercase font-semibold">Input (Original)</span>
                <label className="w-40 h-40 rounded-xl border border-dashed border-white/20 bg-white/[0.02] flex items-center justify-center overflow-hidden cursor-pointer hover:border-cyan-400/50 transition-all">
                  {previewUrl ? (
                    <img src={previewUrl} alt="Input" className="w-full h-full object-cover" />
                  ) : (
                    <div className="flex flex-col items-center py-3 text-center px-2">
                      <Upload className="w-6 h-6 mb-1 text-slate-500" />
                      <span className="text-[10px] text-slate-400">Upload site image</span>
                    </div>
                  )}
                  <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                </label>
              </div>

              {/* Reconstructed side */}
              <div className="flex flex-col items-center">
                <span className="text-[11px] font-mono text-slate-400 mb-1.5 uppercase font-semibold">100e VAE Output</span>
                <div className="w-40 h-40 rounded-xl border border-white/10 bg-black/40 flex items-center justify-center overflow-hidden shadow-inner">
                  {reconstructedUrl ? (
                    <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      src={reconstructedUrl} alt="VAE Reconstruction" className="w-full h-full object-cover" />
                  ) : (
                    <div className="text-slate-600 text-xs text-center px-2">Awaiting upload</div>
                  )}
                </div>
              </div>
            </div>

            <button onClick={handleReconstruct} disabled={!selectedFile || isReconstructing} className="btn-secondary w-full">
              {isReconstructing ? (
                <>Reconstructing <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Reconstruct with VAE <ArrowRight className="w-5 h-5" /></>
              )}
            </button>

            {reconstructionMetrics && (
              <div className="w-full grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-white/[0.03] border border-white/10 text-center font-mono">
                <div>
                  <div className="text-[9px] uppercase tracking-wider text-slate-500">MSE Loss</div>
                  <div className="text-xs font-bold text-cyan-400">{reconstructionMetrics.mse}</div>
                </div>
                <div>
                  <div className="text-[9px] uppercase tracking-wider text-slate-500">PSNR</div>
                  <div className="text-xs font-bold text-emerald-400">{reconstructionMetrics.psnr}</div>
                </div>
                <div>
                  <div className="text-[9px] uppercase tracking-wider text-slate-500">Latent Norm</div>
                  <div className="text-xs font-bold text-purple-400">{Number(reconstructionMetrics.latent_mu_norm || 0).toFixed(1)}</div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* ── CVAE Conditional Generation Panel ───────────────────────────── */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><FlaskConical className="w-4 h-4" /></div>
          Conditional Generation (CVAE)
          <span className="ml-auto text-xs font-semibold px-2.5 py-1 rounded-full bg-purple-100 text-purple-700 border border-purple-200">
            NEW — Feature Injection
          </span>
        </div>
        <p className="text-sm text-slate-500 mb-6 -mt-3 max-w-2xl">
          Select a safety class to <strong className="text-slate-700">condition the generation</strong> — the CVAE injects
          a one-hot class embedding into both the encoder and decoder, steering the latent space
          toward scenes associated with that class.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* Left: Controls */}
          <div className="space-y-6">

            {/* Class selector grid */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3 block">
                Step 1 — Select Safety Class Condition
              </label>
              <div className="grid grid-cols-2 gap-2">
                {SAFETY_CLASSES.map((cls) => {
                  const s = CLASS_STYLE[cls.type];
                  const isSelected = selectedClass?.idx === cls.idx;
                  return (
                    <button
                      key={cls.idx}
                      onClick={() => { setSelectedClass(cls); setCvaeUrl(null); setCvaeResult(null); }}
                      className={`flex items-center gap-2.5 px-3 py-2.5 rounded-xl border text-sm font-medium
                                  transition-all duration-200 text-left backdrop-blur-sm
                                  ${isSelected
                                    ? `${s.border} ${s.bg} ${s.text} ring-2 ${s.ring} shadow-md`
                                    : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/20 hover:bg-white/[0.06] hover:text-white'
                                  }`}
                    >
                      <span className="text-base flex-shrink-0">{cls.emoji}</span>
                      <span className="leading-tight">{cls.label}</span>
                      {isSelected && <CheckCircle2 className="w-3.5 h-3.5 ml-auto flex-shrink-0" />}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Temperature slider */}
            <div>
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
                <Thermometer className="w-3.5 h-3.5 text-cyan-400" />
                Step 2 — Sampling Temperature
                <span className="ml-auto font-black text-cyan-400 font-mono text-sm">{temperature.toFixed(1)}</span>
              </label>
              <input
                type="range" min="0.3" max="2.0" step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer
                           bg-gradient-to-r from-cyan-500 via-indigo-500 to-rose-500"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1.5">
                <span>0.3 — Focused</span>
                <span>1.0 — Balanced</span>
                <span>2.0 — Diverse</span>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Low temperature → latent z stays near μ (predictable).
                High temperature → z explores further from μ (creative, varied).
              </p>
            </div>

            {/* Generate button */}
            <button
              onClick={handleCvaeGenerate}
              disabled={selectedClass === null || isCvaeGenerating}
              className="btn-primary w-full disabled:opacity-40"
            >
              {isCvaeGenerating ? (
                <><Loader2 className="w-5 h-5 animate-spin" /> Generating conditioned scene...</>
              ) : (
                <><Zap className="w-5 h-5" /> Generate — Conditioned on "{selectedClass?.label ?? '…'}"
                </>
              )}
            </button>

            {/* Architecture note */}
            <div className="p-4 rounded-xl bg-white/[0.03] border border-white/10 text-xs text-slate-400 space-y-1.5">
              <div className="font-semibold text-slate-200 mb-2 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" /> CVAE Condition Injection Points
              </div>
              <div>🔹 <strong>Encoder:</strong> one-hot → Linear(64) → concat with CNN features → μ, σ heads</div>
              <div>🔹 <strong>Decoder:</strong> one-hot → Linear(64) → concat with z → FC expansion → deconv</div>
              <div className="pt-1 text-[10px] text-cyan-400/80 font-mono">
                Latent dim: 256 · Condition dim: 64 · Total encoder input: 16,448-d
              </div>
            </div>
          </div>

          {/* Right: Output */}
          <div className="space-y-4">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
              Generated Scene
            </label>

            <div className="relative w-full aspect-square rounded-2xl overflow-hidden border-2
                            border-dashed border-white/15 bg-black/40 flex items-center justify-center">
              {isCvaeGenerating && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm z-10">
                  <Loader2 className="w-10 h-10 animate-spin text-cyan-400 mb-3" />
                  <p className="text-sm text-slate-300 font-medium">Sampling from conditioned latent space…</p>
                  <p className="text-xs text-slate-500 mt-1">z ~ N(0,I) conditioned on {selectedClass?.label}</p>
                </div>
              )}
              {cvaeUrl ? (
                <motion.img
                  key={cvaeUrl}
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4 }}
                  src={cvaeUrl} alt="CVAE Generated"
                  className="w-full h-full object-cover"
                />
              ) : !isCvaeGenerating && (
                <div className="flex flex-col items-center text-slate-500">
                  <FlaskConical className="w-12 h-12 mb-3 opacity-30 text-cyan-400" />
                  <p className="text-sm">Select a class and generate</p>
                </div>
              )}
            </div>

            {/* Result metadata */}
            {cvaeResult && (
              <motion.div
                initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl border bg-black/40 border-white/10 space-y-3 backdrop-blur-md"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-medium">Condition Applied</span>
                  <span className={`text-xs font-bold px-2.5 py-1 rounded-full border
                    ${CLASS_STYLE[cvaeResult ? SAFETY_CLASSES[cvaeResult.class_idx]?.type : 'neutral']?.badge}`}>
                    {cvaeResult.class_name}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-medium">Temperature</span>
                  <span className="text-xs font-bold text-slate-200 font-mono">{cvaeResult.temperature.toFixed(1)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-medium">Class Type</span>
                  <span className={`text-xs font-semibold flex items-center gap-1
                    ${SAFETY_CLASSES[cvaeResult.class_idx]?.type === 'danger' ? 'text-rose-400' :
                      SAFETY_CLASSES[cvaeResult.class_idx]?.type === 'safe' ? 'text-emerald-400' : 'text-slate-300'}`}>
                    {SAFETY_CLASSES[cvaeResult.class_idx]?.type === 'danger'
                      ? <><AlertTriangle className="w-3 h-3" /> Violation / Hazard</>
                      : SAFETY_CLASSES[cvaeResult.class_idx]?.type === 'safe'
                      ? <><CheckCircle2 className="w-3 h-3" /> PPE Compliant</>
                      : 'Neutral'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-400 font-medium">Latent z dimension</span>
                  <span className="text-xs font-mono font-bold text-cyan-400">256 + 64 cond = 320</span>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </motion.div>

      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><Activity className="w-4 h-4" /></div>
          Training Metrics &amp; Latent Space Analysis
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-panel-hover p-4 h-80 flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">ELBO Loss Curve (Reconstruction + KL)</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/processed/features/vae_loss_curve.png`} className="w-full h-full object-contain" alt="VAE Loss"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
          <div className="glass-panel-hover p-4 h-80 flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">Latent Manifold (t-SNE Projection)</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/processed/features/vae_tsne.png`} className="w-full h-full object-contain" alt="VAE t-SNE"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Saved Outputs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div variants={itemVariants} className="glass-panel-hover p-6">
          <h3 className="text-lg font-bold text-slate-200 mb-3">VAE Generative Samples (from Prior)</h3>
          <div className="image-frame h-64">
            <img src={`${API}/data/processed/features/vae_generative_samples.png`} alt="VAE Samples" className="w-full h-full object-contain"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
          </div>
        </motion.div>
        <motion.div variants={itemVariants} className="glass-panel-hover p-6">
          <h3 className="text-lg font-bold text-slate-200 mb-3">Original vs Reconstructed</h3>
          <div className="image-frame h-64">
            <img src={`${API}/data/processed/features/vae_original_vs_reconstructed.png`} alt="VAE Comparison" className="w-full h-full object-contain"
              onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
            <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
          </div>
        </motion.div>
      </div>

      {/* SSIM/PSNR Metrics */}
      <motion.div variants={itemVariants} className="glass-panel-hover p-6">
        <h3 className="text-lg font-bold text-slate-200 mb-3">VAE Metrics (SSIM / PSNR)</h3>
        <div className="image-frame h-64">
          <img src={`${API}/data/processed/features/vae_metrics.png`} alt="VAE Metrics" className="w-full h-full object-contain"
            onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
          <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
        </div>
      </motion.div>
    </motion.div>
  );
}
