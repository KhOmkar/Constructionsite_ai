import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, Play, AlertTriangle, Loader2, Layers, Cpu, Hash, 
  Activity, Eye, Crosshair, ShieldAlert, CheckCircle2, XCircle, 
  Sparkles, RefreshCw, AlertOctagon, HelpCircle, HardHat, FileText
} from 'lucide-react';
import axios from 'axios';

const API = 'http://localhost:8000';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

export default function TransformerView() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [attentionMapUrl, setAttentionMapUrl] = useState(null);
  const [riskMapUrl, setRiskMapUrl] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [activeTab, setActiveTab] = useState('detections'); // 'detections' | 'attention' | 'risk_map'
  const [resultData, setResultData] = useState(null);

  // 1-Click Evaluation Presets
  const presets = [
    {
      id: 'rule4-blindspot',
      title: 'Rule 4: Excavator Blindspot',
      desc: 'Worker positioned within heavy machinery swing radius',
      tag: 'Critical Struck-By Hazard',
      rule: 'OSHA 1926.600',
      color: 'rose',
      image: '/data/css-data/test/images/004063_jpg.rf.1b7cdc4035bcb24ef69b8798b444053e.jpg'
    },
    {
      id: 'rule2-ppe',
      title: 'Rule 2: PPE Non-Compliance',
      desc: 'Ground crew missing required hardhat and safety vest',
      tag: 'PPE Violation',
      rule: 'OSHA 1926.100',
      color: 'amber',
      image: '/data/css-data/test/images/002551_jpg.rf.ce4b9f934161faa72c80dc6898d37b2d.jpg'
    },
    {
      id: 'rule3-trench',
      title: 'Rule 3: Trench Excavation',
      desc: 'Deep excavation with earthmoving machinery active',
      tag: 'Excavation Safety',
      rule: 'OSHA 1926.651',
      color: 'cyan',
      image: '/data/css-data/test/images/004763_jpg.rf.46484e6ca73caeaa9de45822cf1085a9.jpg'
    },
    {
      id: 'rule1-heights',
      title: 'Rule 1: Fall Protection & Heights',
      desc: 'Structural work with edge exposure & personnel monitoring',
      tag: 'Fall Protection',
      rule: 'OSHA 1926.501',
      color: 'purple',
      image: '/data/css-data/test/images/000005_jpg.rf.96e9379ccae638140c4a90fc4b700a2b.jpg'
    }
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setAnnotatedUrl(null);
      setAttentionMapUrl(null);
      setRiskMapUrl(null);
      setResultData(null);
    }
  };

  const handleRunAssessment = async (fileToUse = selectedFile) => {
    if (!fileToUse) return;
    setIsSimulating(true);

    const formData = new FormData();
    formData.append('file', fileToUse);

    try {
      const response = await axios.post(`${API}/api/transformer/hazard`, formData);
      const data = response.data;
      setResultData(data);
      setAnnotatedUrl(data.annotated_image);
      setAttentionMapUrl(data.attention_map);
      setRiskMapUrl(data.risk_map_image);
    } catch (error) {
      console.error('Error running transformer hazard assessment:', error);
      alert('Failed to connect to backend. Please ensure the FastAPI server is running.');
    } finally {
      setIsSimulating(false);
    }
  };

  const handleSelectPreset = async (preset) => {
    try {
      setIsSimulating(true);
      setPreviewUrl(preset.image);
      setAnnotatedUrl(null);
      setAttentionMapUrl(null);
      setRiskMapUrl(null);
      setResultData(null);

      const resp = await fetch(preset.image);
      const blob = await resp.blob();
      const file = new File([blob], `${preset.id}.jpg`, { type: 'image/jpeg' });
      setSelectedFile(file);
      await handleRunAssessment(file);
    } catch (err) {
      console.error('Error loading preset:', err);
      // Fallback: send as form file
      setIsSimulating(false);
    }
  };

  const handleAuditGanScene = async () => {
    setIsSimulating(true);
    try {
      const response = await axios.post(`${API}/api/transformer/audit_gan`);
      const data = response.data;
      setResultData(data);
      setPreviewUrl(data.original_image);
      setAnnotatedUrl(data.annotated_image);
      setAttentionMapUrl(data.attention_map);
      setRiskMapUrl(data.risk_map_image);
    } catch (error) {
      console.error('Error auditing GAN scene:', error);
      alert('Failed to audit GAN scene. Ensure backend is running.');
    } finally {
      setIsSimulating(false);
    }
  };

  const handleSelectPreset = (p) => {
    setActivePreset(p.id);
    setPreviewUrl(p.image_url);
    setSelectedRule(p.stage_idx.toString());
    const blob = dataURLtoBlob(p.image_url);
    const file = new File([blob], `${p.id}.jpg`, { type: 'image/jpeg' });
    setSelectedFile(file);
    triggerSimulation(file, p.stage_idx);
  };

  // Filter detected classes based on dynamic threshold
  const activeClassList = classProbabilities.length > 0
    ? classProbabilities.filter(c => c.probability >= detectionThreshold)
    : detectedClasses.map(name => ({ class_name: name, probability: 85, is_detected: true }));

  const getTagColor = (clsName) => {
    if (clsName.startsWith('NO-') || clsName.includes('Hazard')) {
      return 'bg-rose-50 text-rose-700 border-rose-300 ring-rose-500/20';
    }
    if (clsName === 'machinery' || clsName === 'vehicle' || clsName === 'Safety Cone') {
      return 'bg-amber-50 text-amber-800 border-amber-300 ring-amber-500/20';
    }
    if (clsName === 'Person') {
      return 'bg-blue-50 text-blue-700 border-blue-300 ring-blue-500/20';
    }
    return 'bg-emerald-50 text-emerald-800 border-emerald-300 ring-emerald-500/20';
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-7xl mx-auto pb-12"
    >
      {/* Header */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <motion.div variants={itemVariants} className="flex items-center gap-3 mb-2">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-500 via-blue-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/25">
              <ShieldAlert className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-black gradient-text tracking-tight">Vision Transformer (ViT) Hazard Intelligence</h2>
              <span className="badge badge-primary">Research Phase 2 — 8-Head Self-Attention Saliency &amp; OSHA Safety Reasoning</span>
            </div>
          </motion.div>
          <motion.p variants={itemVariants} className="text-slate-400 text-sm max-w-3xl leading-relaxed">
            Multi-head Vision Transformer combining 64 spatial patch tokens with 8-head self-attention rollout 
            and deep object detection. Pinpoints worker-machinery proximity hazards, identifies PPE non-compliance, 
            and generates automated OSHA standards audits in real time (~14 ms).
          </motion.p>
        </div>

        <motion.div variants={itemVariants} className="flex items-center gap-3">
          <button
            onClick={handleAuditGanScene}
            disabled={isSimulating}
            className="px-4 py-2.5 rounded-xl bg-purple-500/10 hover:bg-purple-500/20 border border-purple-500/30 text-purple-300 font-semibold text-xs flex items-center gap-2 transition-all shadow-sm"
          >
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span>Audit GAN Synthetic Site</span>
          </button>
        </motion.div>
      </header>

      {/* 1-Click Evaluation Presets */}
      <motion.div variants={itemVariants} className="glass-panel p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="section-title text-sm m-0">
            <div className="section-icon"><Crosshair className="w-4 h-4" /></div>
            1-Click Benchmark Evaluation Presets (Real Columbia/UBC &amp; Roboflow CSS Imagery)
          </div>
          <span className="text-xs font-mono text-slate-400">4 OSHA Safety Rules Tested</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handleSelectPreset(preset)}
              disabled={isSimulating}
              className="p-3.5 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-white/5 hover:border-cyan-500/40 text-left transition-all group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] font-mono font-bold text-cyan-400">{preset.rule}</span>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                    preset.color === 'rose' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' :
                    preset.color === 'amber' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30' :
                    preset.color === 'cyan' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' :
                    'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                  }`}>
                    {preset.tag}
                  </span>
                </div>
                <h4 className="text-sm font-bold text-slate-100 group-hover:text-cyan-300 transition-colors">
                  {preset.title}
                </h4>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">
                  {preset.desc}
                </p>
              </div>
              <div className="mt-3 text-[11px] font-semibold text-cyan-400/80 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                <span>Evaluate Sample</span>
                <span>→</span>
              </div>
            </button>
          ))}
        </div>
      </motion.div>

      {/* Main Interactive Workstation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Upload & Live Controls */}
        <motion.div variants={itemVariants} className="lg:col-span-4 flex flex-col gap-5">
          <div className="glass-panel p-5 space-y-4">
            <div className="section-title text-sm">
              <div className="section-icon"><Upload className="w-4 h-4" /></div>
              Site Feed Input &amp; Inference
            </div>

            <label className="upload-zone h-52 w-full flex flex-col items-center justify-center p-3 relative group">
              {previewUrl ? (
                <img 
                  src={previewUrl} 
                  alt="Input Site Feed" 
                  className="w-full h-full object-cover rounded-lg group-hover:opacity-90 transition-opacity" 
                />
              ) : (
                <div className="flex flex-col items-center justify-center text-center p-4">
                  <div className="w-12 h-12 rounded-full bg-cyan-500/10 flex items-center justify-center mb-3 text-cyan-400">
                    <Upload className="w-6 h-6" />
                  </div>
                  <p className="text-sm font-bold text-slate-200">Click to upload site photo</p>
                  <span className="text-xs text-slate-400 mt-1">High-Res CCTV, Drone, or Safety Inspection Image</span>
                </div>
              )}
              <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
            </label>

            <button
              onClick={() => handleRunAssessment()}
              disabled={!selectedFile || isSimulating}
              className="btn-primary w-full py-3.5 flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 font-bold"
            >
              {isSimulating ? (
                <>Analyzing Site Safety <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Execute ViT Hazard Inference <Play className="w-5 h-5" /></>
              )}
            </button>
          </div>

          {/* Quick Metrics Bar */}
          {resultData && (
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }} 
              animate={{ opacity: 1, scale: 1 }}
              className="glass-panel p-5 space-y-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono uppercase text-slate-400 font-semibold">OSHA Risk Index</span>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-black border ${
                  resultData.severity === 'CRITICAL HAZARD' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' :
                  resultData.severity === 'MODERATE HAZARD' ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' :
                  'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                }`}>
                  {resultData.severity}
                </span>
              </div>

              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-black text-white">{resultData.risk_percentage}</span>
                <span className="text-xs text-slate-400 font-mono">Normalized Violation Severity</span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                <div 
                  className={`h-full transition-all duration-700 ${
                    resultData.risk_score >= 0.65 ? 'bg-rose-500' :
                    resultData.risk_score >= 0.35 ? 'bg-amber-500' : 'bg-emerald-500'
                  }`}
                  style={{ width: `${resultData.risk_score * 100}%` }}
                />
              </div>

              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800 text-xs">
                <div className="p-2 rounded-lg bg-slate-900/60 border border-white/5">
                  <div className="text-slate-400">Workers Active</div>
                  <div className="text-base font-bold text-amber-400">{resultData.counts.workers}</div>
                </div>
                <div className="p-2 rounded-lg bg-slate-900/60 border border-white/5">
                  <div className="text-slate-400">Machinery Units</div>
                  <div className="text-base font-bold text-cyan-400">{resultData.counts.machinery}</div>
                </div>
                <div className="p-2 rounded-lg bg-slate-900/60 border border-white/5">
                  <div className="text-slate-400">PPE Violations</div>
                  <div className="text-base font-bold text-rose-400">{resultData.counts.violations}</div>
                </div>
                <div className="p-2 rounded-lg bg-slate-900/60 border border-white/5">
                  <div className="text-slate-400">GPU Latency</div>
                  <div className="text-base font-bold text-emerald-400 font-mono">~{resultData.latency_ms} ms</div>
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Right Column: Tri-Pane Saliency & Bounding Box View */}
        <motion.div variants={itemVariants} className="lg:col-span-8 flex flex-col gap-5">
          <div className="glass-panel p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-2">
                <Activity className="w-4.5 h-4.5 text-cyan-400" />
                <h3 className="text-base font-bold text-white tracking-wide">
                  Tri-Pane Hazard Saliency &amp; Attention Rollout
                </h3>
              </div>

              {/* View Switcher Tabs */}
              <div className="flex items-center bg-slate-900/80 p-1 rounded-xl border border-white/10">
                <button
                  onClick={() => setActiveTab('detections')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'detections' 
                      ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20' 
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  OSHA Detections
                </button>
                <button
                  onClick={() => setActiveTab('attention')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    activeTab === 'attention' 
                      ? 'bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20' 
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  8-Head ViT Attention
                </button>
              </div>
            </div>

            {/* Display Canvas Frame */}
            <div className="relative w-full aspect-video rounded-xl overflow-hidden bg-slate-950 border border-white/10 flex items-center justify-center">
              {isSimulating ? (
                <div className="flex flex-col items-center gap-3 text-cyan-400">
                  <Loader2 className="w-10 h-10 animate-spin" />
                  <span className="text-xs font-mono font-semibold">Extracting 8-Head ViT Attention Rollout...</span>
                </div>
              ) : activeTab === 'detections' && annotatedUrl ? (
                <motion.img 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  src={annotatedUrl} 
                  alt="Detected Objects" 
                  className="w-full h-full object-contain" 
                />
              ) : activeTab === 'attention' && attentionMapUrl ? (
                <div className="relative w-full h-full">
                  <motion.img 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }} 
                  src={attentionMapUrl} 
                  alt="8-Head Attention Rollout" 
                  className="w-full h-full object-contain" 
                />
                  <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-lg bg-slate-950/80 border border-white/10 text-[11px] font-mono text-cyan-300">
                    Plasma Colormap: Class Token → 64 Patch Tokens (Self-Attention Rollout)
                  </div>
                </div>
              ) : previewUrl ? (
                <img src={previewUrl} alt="Original Site" className="w-full h-full object-contain" />
              ) : (
                <div className="text-center text-slate-500 text-xs">
                  <Eye className="w-10 h-10 mx-auto mb-2 opacity-30" />
                  Select an evaluation preset or upload a site image to begin
                </div>
              )}
            </div>

            {/* Dual-Pane Thumbnails Preview */}
            {resultData && (
              <div className="grid grid-cols-2 gap-3 mt-4">
                <button
                  onClick={() => setActiveTab('detections')}
                  className={`p-2 rounded-xl bg-slate-900/60 border text-left transition-all ${
                    activeTab === 'detections' ? 'border-cyan-500 shadow-md shadow-cyan-500/10' : 'border-white/5'
                  }`}
                >
                  <span className="text-[10px] font-mono uppercase text-slate-400 block mb-1 font-semibold">1. Detected Objects &amp; PPE</span>
                  <div className="aspect-video rounded-lg overflow-hidden bg-slate-950">
                    <img src={annotatedUrl} alt="Detected Objects" className="w-full h-full object-cover" />
                  </div>
                </button>

                <button
                  onClick={() => setActiveTab('attention')}
                  className={`p-2 rounded-xl bg-slate-900/60 border text-left transition-all ${
                    activeTab === 'attention' ? 'border-cyan-500 shadow-md shadow-cyan-500/10' : 'border-white/5'
                  }`}
                >
                  <span className="text-[10px] font-mono uppercase text-cyan-300 block mb-1 font-semibold">2. 8-Head ViT Attention</span>
                  <div className="aspect-video rounded-lg overflow-hidden bg-slate-950">
                    <img src={attentionMapUrl} alt="Attention Map" className="w-full h-full object-cover" />
                  </div>
                </button>
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Structured OSHA Safety Audit Breakdown & Natural Language Reasoning */}
      {resultData && (
        <motion.div 
          initial={{ opacity: 0, y: 16 }} 
          animate={{ opacity: 1, y: 0 }} 
          className="grid grid-cols-1 lg:grid-cols-12 gap-6"
        >
          {/* Rule-by-Rule Compliance Table */}
          <div className="lg:col-span-7 glass-panel p-6 space-y-4">
            <div className="section-title text-sm">
              <div className="section-icon"><FileText className="w-4 h-4" /></div>
              OSHA Standards Compliance Breakdown (Automated Site Audit)
            </div>

            <div className="space-y-3">
              {resultData.osha_rules.map((rule) => (
                <div 
                  key={rule.rule_id} 
                  className="p-3.5 rounded-xl bg-slate-900/70 border border-white/5 hover:border-white/10 transition-all flex items-start justify-between gap-3"
                >
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold text-cyan-400">{rule.rule_id}: {rule.standard}</span>
                      <span className="text-xs font-bold text-slate-200">— {rule.title}</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">{rule.finding}</p>
                  </div>

                  <span className={`px-2.5 py-1 rounded-full text-[11px] font-mono font-bold whitespace-nowrap border ${
                    rule.status === 'VIOLATION' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' :
                    rule.status === 'CAUTION' ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' :
                    'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                  }`}>
                    {rule.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Natural Language Engineering Reasoning */}
          <div className="lg:col-span-5 glass-panel p-6 space-y-4 border-l-4 border-l-cyan-500">
            <div className="section-title text-sm">
              <div className="section-icon"><HardHat className="w-4 h-4" /></div>
              Autonomous Safety Reasoning &amp; Corrective Action
            </div>

            <div className="p-4 rounded-xl bg-slate-900/80 border border-white/5 text-xs text-slate-300 leading-relaxed space-y-3 font-sans">
              <p className="font-semibold text-slate-100">{resultData.reasoning}</p>
              
              <div className="pt-3 border-t border-slate-800 space-y-1.5 text-slate-400">
                <div className="font-bold text-slate-200">Recommended Action Plan:</div>
                <ul className="list-disc list-inside space-y-1 text-slate-400">
                  {resultData.counts.violations > 0 && (
                    <li>Issue immediate stop-work order to non-compliant personnel until ANSI Type R vests and Class E hardhats are donned.</li>
                  )}
                  {resultData.counts.machinery > 0 && (
                    <li>Enforce a minimum 3-meter safety perimeter and assign a dedicated spotter for active excavator/heavy machinery swing radius.</li>
                  )}
                  <li>Log incident in digital twin safety dashboard for daily subcontractor safety audit records.</li>
                </ul>
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 px-1">
              <span>Model: ConstructionHazardViT (4.10M Parameters)</span>
              <span>Tokens: 65 | Heads: 8</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Model Architecture & Mathematical Explainability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Cpu className="w-4 h-4" /></div>
            Vision Transformer Architecture Specifications
          </div>
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">Patch Size (P)</span>
              <span className="info-value">16×16 pixels</span>
            </div>
            <div className="info-item">
              <span className="info-label">Spatial Tokens (N)</span>
              <span className="info-value">64 patches</span>
            </div>
            <div className="info-item">
              <span className="info-label">Total Tokens</span>
              <span className="info-value">65 (64 + 1 [CLS])</span>
            </div>
            <div className="info-item">
              <span className="info-label">Embedding Dim (D)</span>
              <span className="info-value">256 channels</span>
            </div>
            <div className="info-item">
              <span className="info-label">Attention Heads</span>
              <span className="info-value">8 Heads (32-d/head)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Encoder Depth</span>
              <span className="info-value">4 Blocks (Pre-LN)</span>
            </div>
            <div className="info-item">
              <span className="info-label">Total Parameters</span>
              <span className="info-value">4.10M Params</span>
            </div>
            <div className="info-item">
              <span className="info-label">Inference Latency</span>
              <span className="info-value">~14.1 ms (RTX 3050)</span>
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Hash className="w-4 h-4" /></div>
            Mathematical Formulation of Attention Rollout
          </div>
          <div className="math-block">
            <div>
              <span className="math-fn">Self-Attention</span><span className="math-op">: </span>
              <span className="math-var">A</span> <span className="math-op">= Softmax(</span>
              <span className="math-var">Q K</span><span className="math-sub">T</span> <span className="math-op">/ √</span><span className="math-var">d</span><span className="math-sub">k</span>
              <span className="math-op">)</span>
            </div>
            <div className="mt-2">
              <span className="math-fn">Attention Rollout</span><span className="math-op">: </span>
              <span className="math-var">A</span><span className="math-sub">rollout</span> <span className="math-op">= ∏</span>
              <span className="math-op">(0.5 · </span><span className="math-var">I</span> <span className="math-op">+ 0.5 · </span><span className="math-var">A</span><span className="math-sub">l</span><span className="math-op">)</span>
            </div>
            <div className="mt-2">
              <span className="math-fn">OSHA Proximity Risk</span><span className="math-op">: </span>
              <span className="math-var">R</span> <span className="math-op">= 0.35·P(NO-Hardhat) + 0.35·P(NO-Vest) + 0.30·[P(Worker) × P(Machinery)]</span>
            </div>
            <div className="mt-3 pt-3 border-t border-slate-700/50">
              <span className="math-label">The [CLS] token aggregates global safety context across all 64 patches to expose OSHA violation probability and spatial danger zones.</span>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}
