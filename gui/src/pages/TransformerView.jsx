import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, Play, Clock, AlertTriangle, Loader2, Layers, Cpu, Hash, Activity, Eye, Crosshair } from 'lucide-react';
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

const stages = [
  { value: '0', label: 'Excavation', desc: 'Ground clearing & foundation dig' },
  { value: '1', label: 'Substructure', desc: 'Foundation & underground work' },
  { value: '2', label: 'Framing', desc: 'Structural frame erection' },
  { value: '3', label: 'Facade', desc: 'External wall & cladding' },
  { value: '4', label: 'Finishing', desc: 'Interior fit-out & commissioning' },
];

export default function TransformerView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [simulatedUrl, setSimulatedUrl] = useState(null);
  const [riskMapUrl, setRiskMapUrl] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  
  const [stage, setStage] = useState('1');
  const [days, setDays] = useState('30');

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setSimulatedUrl(null);
      setRiskMapUrl(null);
    }
  };

  const handleSimulate = async () => {
    if (!selectedFile) return;
    setIsSimulating(true);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('stage_idx', stage);
    formData.append('delta_days', days);

    try {
      const response = await axios.post(`${API}/api/transformer/simulate`, formData);
      setSimulatedUrl(response.data.future_sim_image);
      setRiskMapUrl(response.data.risk_map_image);
    } catch (error) {
      console.error('Error simulating progress:', error);
      alert('Failed to connect to backend. Make sure the server is running.');
    } finally {
      setIsSimulating(false);
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
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-3xl font-black gradient-text">Construction Progress Transformer</h2>
            <span className="badge badge-primary">Research Phase 2 — Multi-Modal Progress Simulation</span>
          </div>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 mt-2 max-w-3xl">
          Multi-modal Vision Transformer that fuses current site imagery with construction stage and time horizon 
          conditioning via cross-attention. Predicts future visual states and spatial risk hazard maps 
          for proactive safety intelligence.
        </motion.p>
      </header>

      {/* Architecture Diagram */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          Cross-Attention Transformer Architecture
        </div>

        {/* Visual Tokenization Path */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-cyan-400 mb-2 ml-1">Visual Tokenization (Image → Patch Tokens)</h4>
          <div className="arch-flow flex-wrap">
            <div className="arch-block arch-block-input">
              <div className="text-xs opacity-60">Current State</div>
              <div className="font-bold">3×128×128</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-encoder">
              <div className="text-xs opacity-60">Patch Embed</div>
              <div className="font-bold">16×16 patches</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-encoder">
              <div className="text-xs opacity-60">+ Pos Embed</div>
              <div className="font-bold">64 × 256-d</div>
            </div>
          </div>
        </div>

        {/* Conditioning Path */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-amber-400 mb-2 ml-1">Multi-Modal Conditioning Tokens</h4>
          <div className="arch-flow flex-wrap">
            <div className="arch-block arch-block-latent">
              <div className="text-xs opacity-60">Stage ID</div>
              <div className="font-bold">Embedding</div>
              <div className="text-xs opacity-60">[B, 1, 256]</div>
            </div>
            <span className="arch-arrow">+</span>
            <div className="arch-block arch-block-latent">
              <div className="text-xs opacity-60">Δ Days</div>
              <div className="font-bold">Time MLP</div>
              <div className="text-xs opacity-60">[B, 1, 256]</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-latent animate-pulse-glow">
              <div className="text-xs opacity-60">Concat</div>
              <div className="font-bold">Context</div>
              <div className="text-xs opacity-60">[B, 2, 256]</div>
            </div>
          </div>
        </div>

        {/* Cross-Attention + Prediction */}
        <div>
          <h4 className="text-sm font-semibold text-emerald-400 mb-2 ml-1">Cross-Attention Simulation → Dual Prediction Heads</h4>
          <div className="arch-flow flex-wrap">
            <div className="arch-block arch-block-encoder">
              <div className="text-xs opacity-60">Self-Attn</div>
              <div className="font-bold">MHSA</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-decoder">
              <div className="text-xs opacity-60">Cross-Attn</div>
              <div className="font-bold">Q:vis, KV:ctx</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-encoder">
              <div className="text-xs opacity-60">MLP + Norm</div>
              <div className="font-bold">×4 blocks</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="flex flex-col gap-2 items-center">
              <div className="arch-block arch-block-output">
                <div className="text-xs opacity-60">Unpatchify</div>
                <div className="font-bold">Future 3×128²</div>
              </div>
              <div className="arch-block arch-block-discriminator">
                <div className="text-xs opacity-60">Unpatchify</div>
                <div className="font-bold">Risk 1×128²</div>
              </div>
            </div>
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
              <span className="info-label">Patch Size</span>
              <span className="info-value">16×16</span>
            </div>
            <div className="info-item">
              <span className="info-label">Num Patches</span>
              <span className="info-value">64</span>
            </div>
            <div className="info-item">
              <span className="info-label">Embed Dim</span>
              <span className="info-value">256</span>
            </div>
            <div className="info-item">
              <span className="info-label">Attn Heads</span>
              <span className="info-value">8</span>
            </div>
            <div className="info-item">
              <span className="info-label">Depth</span>
              <span className="info-value">4 blocks</span>
            </div>
            <div className="info-item">
              <span className="info-label">Stages</span>
              <span className="info-value">5 classes</span>
            </div>
            <div className="info-item">
              <span className="info-label">Input</span>
              <span className="info-value">128×128</span>
            </div>
            <div className="info-item">
              <span className="info-label">Outputs</span>
              <span className="info-value">Image + Risk</span>
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Hash className="w-4 h-4" /></div>
            Mathematical Formulation
          </div>
          <div className="math-block">
            <div>
              <span className="math-fn">Patch Tokenize</span><span className="math-op">: </span>
              <span className="math-var">T</span><span className="math-sub">vis</span> <span className="math-op">=</span> <span className="math-fn">PatchEmbed</span>(<span className="math-var">x</span>) <span className="math-op">+</span> <span className="math-var">E</span><span className="math-sub">pos</span>
            </div>
            <div className="mt-2">
              <span className="math-fn">Context</span><span className="math-op">: </span>
              <span className="math-var">C</span> <span className="math-op">=</span> [<span className="math-fn">Emb</span>(<span className="math-var">s</span>), <span className="math-fn">MLP</span>(<span className="math-var">Δt</span>)]
            </div>
            <div className="mt-2">
              <span className="math-fn">Self-Attn</span><span className="math-op">: </span>
              <span className="math-var">T</span> <span className="math-op">=</span> <span className="math-var">T</span> <span className="math-op">+</span> <span className="math-fn">MHSA</span>(<span className="math-var">T</span>, <span className="math-var">T</span>, <span className="math-var">T</span>)
            </div>
            <div className="mt-2">
              <span className="math-fn">Cross-Attn</span><span className="math-op">: </span>
              <span className="math-var">T</span> <span className="math-op">=</span> <span className="math-var">T</span> <span className="math-op">+</span> <span className="math-fn">MHCA</span>(<span className="math-var">T</span>, <span className="math-var">C</span>, <span className="math-var">C</span>)
            </div>
            <div className="mt-3 pt-3 border-t border-slate-700/50">
              <span className="math-fn">x̂</span><span className="math-sub">future</span> <span className="math-op">= σ(</span><span className="math-fn">Unpatchify</span>(<span className="math-fn">FC</span><span className="math-sub">img</span>(<span className="math-var">T</span>))<span className="math-op">)</span>
            </div>
            <div className="mt-1">
              <span className="math-fn">R</span><span className="math-sub">risk</span> <span className="math-op">= σ(</span><span className="math-fn">Unpatchify</span>(<span className="math-fn">FC</span><span className="math-sub">risk</span>(<span className="math-var">T</span>))<span className="math-op">)</span>
            </div>
            <span className="math-label">Visual tokens attend to stage + time context via cross-attention for conditioned future prediction</span>
          </div>
        </motion.div>
      </div>

      {/* Preset Test Samples for Transformer 4D Simulation */}
      <SampleImageSelector
        onSelectSample={(sample, file) => {
          setSelectedFile(file);
          setPreviewUrl(sample.image);
          setSimulatedUrl(null);
          setRiskMapUrl(null);
          const stageVal = String(sample.default_stage !== undefined ? sample.default_stage : '1');
          const daysVal = String(sample.default_days !== undefined ? sample.default_days : '30');
          setStage(stageVal);
          setDays(daysVal);
          // Auto-trigger simulation
          if (file) {
            setIsSimulating(true);
            const formData = new FormData();
            formData.append('file', file);
            formData.append('stage_idx', stageVal);
            formData.append('delta_days', daysVal);
            axios.post(`${API}/api/transformer/simulate`, formData)
              .then(res => {
                setSimulatedUrl(res.data.future_sim_image);
                setRiskMapUrl(res.data.risk_map_image);
              })
              .catch(err => console.error(err))
              .finally(() => setIsSimulating(false));
          }
        }}
        filterModel="transformer"
        title="Sample Images For Testing 4D Digital Twin Simulation"
        subtitle="Select real construction scenes (earthwork trenches, structural framing, tower cranes) to simulate future visual states & spatial risk hazard maps."
      />

      {/* Interactive Simulation */}
      <motion.div variants={itemVariants} className="glass-panel p-8 relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500/8 blur-[100px] rounded-full pointer-events-none" />
        
        <div className="section-title relative z-10">
          <div className="section-icon"><Activity className="w-4 h-4" /></div>
          Live Simulation — Upload, Configure &amp; Predict
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 relative z-10">
          {/* Input & Controls */}
          <div className="flex flex-col gap-5">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Current State Image</label>
              <label className="upload-zone h-44 w-full">
                {previewUrl ? (
                  <img src={previewUrl} alt="Input" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                ) : (
                  <div className="flex flex-col items-center justify-center py-4">
                    <Upload className="w-8 h-8 mb-2 text-slate-500" />
                    <p className="text-sm text-slate-400"><span className="font-semibold text-primary">Click to upload</span></p>
                    <span className="text-xs text-slate-500 mt-1">Construction site image</span>
                  </div>
                )}
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Target Construction Stage</label>
              <select 
                value={stage} 
                onChange={(e) => setStage(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-slate-200 focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
              >
                {stages.map((s) => (
                  <option key={s.value} value={s.value}>{s.label} — {s.desc}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="flex items-center justify-between text-sm font-medium text-slate-400 mb-2">
                <span>Time Horizon</span>
                <span className="text-primary font-bold">{days} Days</span>
              </label>
              <input 
                type="range" 
                min="1" max="90" 
                value={days} 
                onChange={(e) => setDays(e.target.value)}
                className="w-full accent-primary"
              />
              <div className="flex justify-between text-xs text-slate-600 mt-1">
                <span>1 day</span>
                <span>90 days</span>
              </div>
            </div>

            <button
              onClick={handleSimulate}
              disabled={!selectedFile || isSimulating}
              className="btn-primary w-full text-lg py-4"
            >
              {isSimulating ? (
                <>Simulating <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Run Simulation <Play className="w-5 h-5" /></>
              )}
            </button>
          </div>

          {/* Output: Future State + Risk Map */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="glass-panel-hover p-5 flex-1">
              <div className="flex items-center gap-2 mb-3">
                <Clock className="w-4.5 h-4.5 text-cyan-400" />
                <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Predicted Future State</h4>
              </div>
              <div className="image-frame h-52">
                {simulatedUrl ? (
                  <motion.img initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                    src={simulatedUrl} alt="Future Simulation" className="w-full h-full object-cover" />
                ) : (
                  <div className="text-slate-600 text-sm font-medium flex flex-col items-center">
                    <Eye className="w-10 h-10 mb-2 opacity-20" />
                    Awaiting simulation
                  </div>
                )}
              </div>
            </div>

            <div className="glass-panel-hover p-5 flex-1 border-l-4 border-l-rose-500/50">
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="w-4.5 h-4.5 text-rose-400" />
                <h4 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Spatial Risk Hazard Map</h4>
              </div>
              <div className="image-frame h-52">
                {riskMapUrl ? (
                  <div className="relative w-full h-full">
                    <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      src={riskMapUrl} alt="Risk Map" className="w-full h-full object-cover opacity-90" />
                    <div className="absolute inset-0 bg-gradient-to-t from-slate-900/60 via-transparent to-transparent pointer-events-none" />
                  </div>
                ) : (
                  <div className="text-slate-600 text-sm font-medium flex flex-col items-center">
                    <Crosshair className="w-10 h-10 mb-2 opacity-20" />
                    Awaiting simulation
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Construction Stages Reference */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          Construction Stage Taxonomy
        </div>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {stages.map((s, i) => (
            <div key={s.value} className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center group hover:border-primary/30 transition-all">
              <div className="text-2xl font-black text-primary mb-1">{i}</div>
              <div className="text-sm font-bold text-slate-200">{s.label}</div>
              <div className="text-xs text-slate-500 mt-1">{s.desc}</div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Key Insights */}
      <motion.div variants={itemVariants} className="glass-panel-accent p-6">
        <div className="section-title">
          <div className="section-icon"><Activity className="w-4 h-4" /></div>
          Research Context &amp; Capabilities
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-3 text-sm text-slate-300 leading-relaxed">
            <p>
              <strong className="text-slate-100">Multi-Modal Conditioning:</strong> Unlike standard image-to-image models, 
              this transformer conditions on both discrete stage tokens (via learned embeddings) and continuous time 
              signals (via MLP projection), enabling fine-grained control over the simulation horizon.
            </p>
            <p>
              <strong className="text-slate-100">Cross-Attention Mechanism:</strong> Visual patch tokens attend to the 
              2-token conditioning context (stage + time) through dedicated cross-attention layers, allowing the model 
              to learn stage-specific and time-dependent visual transformations.
            </p>
          </div>
          <div className="space-y-3 text-sm text-slate-300 leading-relaxed">
            <p>
              <strong className="text-slate-100">Dual Prediction Heads:</strong> The architecture produces both a 
              predicted future visual state and a spatial risk hazard map from the same latent representation, 
              enabling simultaneous progress monitoring and safety intelligence.
            </p>
            <p>
              <strong className="text-slate-100">Patch-based Reconstruction:</strong> The unpatchify operation 
              reconstructs full-resolution images from per-patch predictions using Einstein summation, preserving 
              spatial coherence across patch boundaries.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
