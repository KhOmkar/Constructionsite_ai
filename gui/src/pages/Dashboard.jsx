import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import axios from 'axios';
import {
  Activity, ShieldAlert, BarChart3, Maximize2, Info,
  Box, Sparkles, Image as ImageIcon, Layers, ArrowRight,
  BookOpen, Scale, TrendingUp, Database, Cpu, Play,
  CheckCircle2, AlertTriangle, Zap, RefreshCw, Eye
} from 'lucide-react';
import SampleImageSelector from '../components/SampleImageSelector';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.07 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
};

const API = 'http://localhost:8000';

/* -- Stat Card --------------------------------------------------------- */
const StatCard = ({ title, value, icon: Icon, trend, accentColor = '#003153' }) => (
  <motion.div variants={itemVariants}
    className="bg-white border border-slate-200 rounded-2xl p-5 relative overflow-hidden
               shadow-sm hover:shadow-md hover:border-primary/30 transition-all duration-300 group">
    <div className="absolute top-0 right-0 w-24 h-24 rounded-full -mr-8 -mt-8 blur-2xl opacity-60
                    group-hover:opacity-100 transition-opacity"
      style={{ background: `${accentColor}18` }} />
    <div className="flex justify-between items-start mb-4 relative z-10">
      <div className="w-10 h-10 rounded-xl flex items-center justify-center border border-slate-200 bg-slate-50 shadow-sm">
        <Icon className="w-5 h-5" style={{ color: accentColor }} />
      </div>
      {trend !== undefined && (
        <span className={`text-xs font-bold px-2 py-1 rounded-full
          ${trend > 0 ? 'bg-emerald-50 text-emerald-600 border border-emerald-200'
          : trend < 0 ? 'bg-rose-50 text-rose-600 border border-rose-200'
          : 'bg-slate-100 text-slate-500 border border-slate-200'}`}>
          {trend > 0 ? '+' : ''}{trend}%
        </span>
      )}
    </div>
    <p className="text-xs uppercase tracking-wider font-semibold text-slate-500 relative z-10">{title}</p>
    <p className="text-2xl font-black mt-1 text-slate-800 relative z-10">{value}</p>
  </motion.div>
);

/* -- Model Card -------------------------------------------------------- */
const ModelCard = ({ name, icon: Icon, description, path, color, metrics, badge }) => (
  <motion.div variants={itemVariants}>
    <Link to={path} className="block h-full">
      <div className="bg-white border border-slate-200 rounded-2xl p-6 h-full group cursor-pointer
                      relative overflow-hidden shadow-sm hover:shadow-lg hover:border-slate-300
                      transition-all duration-300">
        <div className="absolute top-0 right-0 w-40 h-40 rounded-full -mr-16 -mt-16 blur-3xl
                        opacity-0 group-hover:opacity-100 transition-opacity duration-700"
          style={{ background: `radial-gradient(circle, ${color}20, transparent)` }} />
        <div className="flex items-center justify-between mb-4 relative z-10">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl flex items-center justify-center border border-slate-200 shadow-sm"
              style={{ background: `${color}12` }}>
              <Icon className="w-5 h-5" style={{ color }} />
            </div>
            <div>
              <h3 className="font-bold text-base text-slate-800 leading-tight">{name}</h3>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full border"
                style={{ color, borderColor: `${color}40`, background: `${color}10` }}>
                {badge}
              </span>
            </div>
          </div>
          <ArrowRight className="w-4 h-4 text-slate-400 group-hover:translate-x-1 transition-transform" />
        </div>
        <p className="text-sm text-slate-500 mb-4 leading-relaxed relative z-10">{description}</p>
        <div className="grid grid-cols-3 gap-2 relative z-10">
          {metrics.map((m, i) => (
            <div key={i} className="p-2 rounded-xl bg-slate-50 border border-slate-100 text-center">
              <div className="text-[10px] text-slate-400 font-medium">{m.label}</div>
              <div className="text-sm font-bold text-slate-700 mt-0.5">{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </Link>
  </motion.div>
);

/* -- EDA Chart --------------------------------------------------------- */
const EDAChart = ({ title, imgUrl, className = '' }) => (
  <motion.div variants={itemVariants}
    className={`bg-white border border-slate-200 rounded-2xl p-4 flex flex-col shadow-sm
                hover:shadow-md hover:border-primary/20 transition-all duration-300 group ${className}`}>
    <div className="flex justify-between items-center mb-3">
      <h4 className="text-sm font-semibold text-slate-700">{title}</h4>
      <button className="p-1.5 text-slate-400 hover:text-primary transition-colors
                         opacity-0 group-hover:opacity-100 bg-slate-50 border border-slate-200 rounded-lg">
        <Maximize2 className="w-3.5 h-3.5" />
      </button>
    </div>
    <div className="relative flex-1 w-full min-h-[180px] rounded-xl overflow-hidden
                    bg-slate-50 border border-slate-100 flex items-center justify-center">
      <img src={imgUrl} alt={title}
        className="w-full h-full object-contain transform group-hover:scale-[1.02] transition-transform duration-500"
        onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
      />
      <div className="absolute inset-0 hidden items-center justify-center flex-col text-slate-400 gap-2">
        <Info className="w-6 h-6 opacity-40" />
        <span className="text-xs text-slate-400">Run notebook first</span>
      </div>
    </div>
  </motion.div>
);

/* ======================================================================
   DASHBOARD PAGE
   ====================================================================== */
export default function Dashboard() {
  const [selectedSample, setSelectedSample] = useState(null);
  const [activeModel, setActiveModel] = useState('ae');
  const [stageIdx, setStageIdx] = useState(1);
  const [deltaDays, setDeltaDays] = useState(30);
  const [isRunningTest, setIsRunningTest] = useState(false);
  const [testResult, setTestResult] = useState(null);

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
    if (sample.default_stage !== undefined) setStageIdx(sample.default_stage);
    if (sample.default_days !== undefined) setDeltaDays(sample.default_days);
    if (sample.recommended_models && !sample.recommended_models.includes(activeModel)) {
      setActiveModel(sample.recommended_models[0]);
    }
    // Auto-trigger test on sample selection for instant feedback!
    runLiveTest(sample, sample.recommended_models ? sample.recommended_models[0] : activeModel);
  };

  const runLiveTest = async (sampleToUse = selectedSample, modelToUse = activeModel) => {
    if (!sampleToUse) return;
    setIsRunningTest(true);
    setTestResult(null);

    const formData = new FormData();
    formData.append('sample_id', sampleToUse.id);
    formData.append('model', modelToUse);
    formData.append('stage_idx', stageIdx);
    formData.append('delta_days', deltaDays);

    try {
      const res = await axios.post(`${API}/api/samples/test_sample`, formData);
      setTestResult(res.data);
    } catch (err) {
      console.error('Error running sample test:', err);
      alert('Failed to connect to backend server. Make sure port 8000 is running.');
    } finally {
      setIsRunningTest(false);
    }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-7xl mx-auto"
    >
      {/* Header */}
      <header className="mb-2 flex flex-col md:flex-row md:items-end justify-between gap-5 relative z-10">
        <div>
          <motion.div variants={itemVariants} className="flex items-center gap-3 mb-2">
            <div className="w-11 h-11 rounded-xl bg-primary flex items-center justify-center shadow-md shadow-primary/20">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-black tracking-tight text-slate-800 leading-none">
                ConstructionSite AI
              </h2>
              <p className="text-[11px] text-slate-400 font-semibold uppercase tracking-widest mt-0.5">
                Digital Twin Lab &middot; Generative AI Framework
              </p>
            </div>
          </motion.div>
          <motion.p variants={itemVariants} className="text-slate-500 text-base leading-relaxed max-w-xl">
            Generative Framework for Construction Site Monitoring &amp; Safety Analysis
          </motion.p>
        </div>
        <motion.div variants={itemVariants} className="flex gap-3">
          <Link to="/compare" className="btn-secondary py-2.5 px-5 text-sm">
            View Analysis <BarChart3 className="w-4 h-4" />
          </Link>
          <Link to="/transformer" className="btn-primary py-2.5 px-5 text-sm">
            Run Simulation <ArrowRight className="w-4 h-4" />
          </Link>
        </motion.div>
      </header>

      {/* Hero Feature Maps */}
      <motion.div variants={itemVariants} className="w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

          {/* Future State */}
          <div className="group cursor-pointer">
            <div className="relative w-full aspect-[4/3] bg-slate-100 overflow-hidden mb-3 rounded-2xl
                            border border-slate-200 shadow-sm hover:shadow-md transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-primary/5
                              group-hover:from-primary/15 group-hover:to-primary/10 transition-all duration-700
                              flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20
                                  flex items-center justify-center mx-auto mb-3">
                    <Maximize2 className="w-8 h-8 text-primary opacity-70" />
                  </div>
                  <span className="text-xs font-bold tracking-widest uppercase text-primary/60">
                    Simulation Map Data
                  </span>
                </div>
              </div>
              <div className="absolute top-3 right-3">
                <span className="bg-white/80 backdrop-blur-sm text-primary text-[10px] font-bold
                                 px-2.5 py-1 rounded-full border border-primary/20 shadow-sm">
                  LIVE PREVIEW
                </span>
              </div>
            </div>
            <h3 className="text-lg font-black text-slate-800 uppercase tracking-tight">Future State</h3>
            <p className="text-slate-500 text-sm mt-0.5">Predicted visual progression of site</p>
          </div>

          {/* Spatial Risk */}
          <div className="group cursor-pointer">
            <div className="relative w-full aspect-[4/3] bg-slate-800 overflow-hidden mb-3 rounded-2xl
                            border border-slate-700 shadow-sm hover:shadow-lg transition-all duration-300">
              <div className="absolute inset-0 bg-gradient-to-tr from-slate-900 to-slate-800
                              flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-400/20
                                  flex items-center justify-center mx-auto mb-3">
                    <ShieldAlert className="w-8 h-8 text-rose-400 opacity-70" />
                  </div>
                  <span className="text-xs font-bold tracking-widest uppercase text-slate-400">
                    Risk Map Data
                  </span>
                </div>
              </div>
              <div className="absolute top-3 right-3">
                <span className="bg-slate-900/80 backdrop-blur-sm text-rose-400 text-[10px] font-bold
                                 px-2.5 py-1 rounded-full border border-rose-400/20 shadow-sm">
                  HAZARD ZONES
                </span>
              </div>
            </div>
            <h3 className="text-lg font-black text-slate-800 uppercase tracking-tight">Spatial Risk</h3>
            <p className="text-slate-500 text-sm mt-0.5">Hazard detection zones &amp; analysis</p>
          </div>

        </div>
      </motion.div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Safety Classes"   value="10"    icon={ShieldAlert} trend={0}   accentColor="#003153" />
        <StatCard title="Training Images"  value="3,245" icon={Database}    trend={12}  accentColor="#0891b2" />
        <StatCard title="Total Detections" value="14.2K" icon={TrendingUp}  trend={8}   accentColor="#059669" />
        <StatCard title="Models Deployed"  value="4 / 4" icon={Cpu}         trend={100} accentColor="#7c3aed" />
      </div>

      {/* ======================================================================
         INTERACTIVE LIVE MODEL TESTING STATION (WITH REAL DATASET SAMPLES)
         ====================================================================== */}
      <motion.div variants={itemVariants} className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="section-title mb-0">
            <div className="section-icon bg-primary text-white"><Zap className="w-4 h-4" /></div>
            Multi-Model Live Testing Station
          </div>
          <span className="text-xs font-bold text-slate-400 uppercase tracking-widest hidden sm:inline">
            1-Click Dataset Benchmarking
          </span>
        </div>

        {/* Sample Image Selector Carousel */}
        <SampleImageSelector
          onSelectSample={handleSelectSample}
          selectedSampleId={selectedSample?.id}
          title="Sample Images For Testing On Dashboard"
          subtitle="Click any real test image below to automatically load and run it across any model."
        />

        {/* Interactive Testing Bench */}
        {selectedSample && (
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-md space-y-6">
            {/* Model Selection Tabs & Run Button */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Target Model:</span>
                <div className="flex flex-wrap gap-1.5 p-1 bg-slate-100 rounded-xl">
                  {[
                    { id: 'ae', label: 'Autoencoder & PPE', icon: Box, color: '#0891b2' },
                    { id: 'vae', label: 'VAE Reconstructor', icon: Sparkles, color: '#7c3aed' },
                    { id: 'transformer', label: 'Transformer 4D', icon: Layers, color: '#003153' },
                    { id: 'gan', label: 'DCGAN Benchmark', icon: ImageIcon, color: '#059669' }
                  ].map(tab => {
                    const TabIcon = tab.icon;
                    const isActive = activeModel === tab.id;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => {
                          setActiveModel(tab.id);
                          runLiveTest(selectedSample, tab.id);
                        }}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                          isActive
                            ? 'bg-white text-slate-800 shadow-sm'
                            : 'text-slate-500 hover:text-slate-800'
                        }`}
                      >
                        <TabIcon className="w-3.5 h-3.5" style={{ color: tab.color }} />
                        {tab.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => runLiveTest(selectedSample, activeModel)}
                  disabled={isRunningTest}
                  className="btn-primary py-2 px-4 text-xs flex items-center gap-2 shadow-md shadow-primary/20"
                >
                  {isRunningTest ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      Running Neural Inference...
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      Run Live Test
                    </>
                  )}
                </button>

                <Link
                  to={activeModel === 'ae' ? '/autoencoder' : activeModel === 'vae' ? '/vae' : activeModel === 'transformer' ? '/transformer' : '/gan'}
                  className="btn-secondary py-2 px-3 text-xs flex items-center gap-1.5"
                >
                  <span>Open Deep Dive</span>
                  <ArrowRight className="w-3 h-3" />
                </Link>
              </div>
            </div>

            {/* Test Arena: Input Details + Real-time Results */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              {/* Left Column: Selected Sample Card */}
              <div className="lg:col-span-4 bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Selected Test Input</span>
                    <h4 className="font-bold text-slate-800 text-sm">{selectedSample.title}</h4>
                  </div>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded border bg-white shadow-sm text-slate-700">
                    {selectedSample.category}
                  </span>
                </div>

                <div className="aspect-square w-full rounded-xl overflow-hidden bg-slate-900 border border-slate-200 relative group">
                  <img
                    src={selectedSample.image}
                    alt={selectedSample.title}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                    <span className="text-[10px] font-mono text-white/90 truncate">{selectedSample.id}</span>
                  </div>
                </div>

                <p className="text-xs text-slate-500 leading-relaxed">{selectedSample.description}</p>

                {/* Additional controls for Transformer */}
                {activeModel === 'transformer' && (
                  <div className="pt-3 border-t border-slate-200 space-y-2">
                    <div className="flex justify-between text-xs font-semibold text-slate-600">
                      <span>Stage Horizon:</span>
                      <span className="text-primary font-bold">
                        {stageIdx === 0 ? 'Stage 0 (Groundwork)' : stageIdx === 1 ? 'Stage 1 (Foundation)' : 'Stage 2 (Superstructure)'}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-1">
                      {[0, 1, 2].map(st => (
                        <button
                          key={st}
                          onClick={() => { setStageIdx(st); runLiveTest(selectedSample, 'transformer'); }}
                          className={`py-1 rounded text-[10px] font-bold border ${
                            stageIdx === st ? 'bg-primary text-white border-primary' : 'bg-white text-slate-600 border-slate-200'
                          }`}
                        >
                          Stage {st}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Right Column: Live Model Output Display */}
              <div className="lg:col-span-8 bg-slate-50 border border-slate-200 rounded-xl p-5 min-h-[320px] flex flex-col justify-center">
                {isRunningTest && (
                  <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-3">
                    <RefreshCw className="w-8 h-8 text-primary animate-spin" />
                    <span className="text-sm font-semibold text-slate-600">Executing {activeModel.toUpperCase()} inference on test sample...</span>
                    <span className="text-xs text-slate-400 font-mono">PyTorch tensor pipeline active</span>
                  </div>
                )}

                {!isRunningTest && !testResult && (
                  <div className="text-center py-12 text-slate-400 space-y-2">
                    <Zap className="w-8 h-8 text-slate-300 mx-auto" />
                    <p className="text-sm font-semibold text-slate-600">Ready to test</p>
                    <p className="text-xs text-slate-400">Click &quot;Run Live Test&quot; above to see real-time inference output</p>
                  </div>
                )}

                {!isRunningTest && testResult?.result && (
                  <div className="space-y-4">
                    {/* Autoencoder & Classification Result */}
                    {activeModel === 'ae' && (
                      <div className="space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-3 p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
                          <div className="flex items-center gap-3">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-black text-white shadow-sm ${
                              testResult.result.status_color === 'emerald' ? 'bg-emerald-600' :
                              testResult.result.status_color === 'rose' ? 'bg-rose-600' : 'bg-primary'
                            }`}>
                              {testResult.result.confidence_percentage}%
                            </div>
                            <div>
                              <div className="text-xs text-slate-400 font-bold uppercase tracking-wider">Classification Prediction</div>
                              <div className="text-lg font-black text-slate-800">{testResult.result.predicted_class}</div>
                            </div>
                          </div>
                          <span className={`px-3 py-1 rounded-full text-xs font-black border ${
                            testResult.result.status_color === 'emerald'
                              ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                              : testResult.result.status_color === 'rose'
                              ? 'bg-rose-50 text-rose-700 border-rose-300'
                              : 'bg-primary/10 text-primary border-primary/20'
                          }`}>
                            {testResult.result.safety_status}
                          </span>
                        </div>

                        {/* Reconstruction Comparison */}
                        {testResult.result.reconstructed_image && (
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                              <span className="text-[10px] font-bold text-slate-400 uppercase block mb-2">Original Sample</span>
                              <div className="aspect-square rounded-lg overflow-hidden bg-slate-100 border border-slate-100">
                                <img src={selectedSample.image} alt="Original" className="w-full h-full object-cover" />
                              </div>
                            </div>
                            <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                              <div className="flex justify-between items-center mb-2 px-1">
                                <span className="text-[10px] font-bold text-primary uppercase">Autoencoder Reconstructed</span>
                                <span className="text-[10px] font-mono text-slate-400">MSE: {testResult.result.mse_loss}</span>
                              </div>
                              <div className="aspect-square rounded-lg overflow-hidden bg-slate-100 border border-primary/20">
                                <img src={testResult.result.reconstructed_image} alt="Reconstructed" className="w-full h-full object-cover" />
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* VAE Result */}
                    {activeModel === 'vae' && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                            <span className="text-[10px] font-bold text-slate-400 uppercase block mb-2">Input Prior</span>
                            <div className="aspect-square rounded-lg overflow-hidden bg-slate-100">
                              <img src={selectedSample.image} alt="Input" className="w-full h-full object-cover" />
                            </div>
                          </div>
                          <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                            <span className="text-[10px] font-bold text-secondary uppercase block mb-2">VAE Latent Reconstruction</span>
                            <div className="aspect-square rounded-lg overflow-hidden bg-slate-100 border border-secondary/30">
                              <img src={testResult.result.reconstructed_image} alt="VAE Reconstructed" className="w-full h-full object-cover" />
                            </div>
                          </div>
                        </div>
                        <div className="p-3 bg-white rounded-xl border border-slate-200 text-xs text-slate-600 flex items-center justify-between">
                          <span>Latent Representation: <strong className="font-mono text-secondary">z ~ N(μ, σ²), d=256</strong></span>
                          <span className="badge badge-secondary">Probabilistic Encoding</span>
                        </div>
                      </div>
                    )}

                    {/* Transformer Result */}
                    {activeModel === 'transformer' && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                            <span className="text-[10px] font-bold text-cyan-600 uppercase block mb-2">Predicted Future Progression</span>
                            <div className="aspect-square rounded-lg overflow-hidden bg-slate-100 border border-cyan-200">
                              <img src={testResult.result.future_sim_image} alt="Future State" className="w-full h-full object-cover" />
                            </div>
                          </div>
                          <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                            <span className="text-[10px] font-bold text-rose-600 uppercase block mb-2">Spatial Risk Hazard Map</span>
                            <div className="aspect-square rounded-lg overflow-hidden bg-slate-100 border border-rose-200">
                              <img src={testResult.result.risk_map_image} alt="Risk Map" className="w-full h-full object-cover" />
                            </div>
                          </div>
                        </div>
                        <div className="p-3 bg-white rounded-xl border border-slate-200 text-xs text-slate-600 flex items-center justify-between">
                          <span>Digital Twin Horizon: <strong className="text-primary font-bold">+{deltaDays} Days</strong></span>
                          <span className="text-xs font-mono text-emerald-600 font-bold">Attention Map Synthesized</span>
                        </div>
                      </div>
                    )}

                    {/* GAN Result */}
                    {activeModel === 'gan' && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                            <span className="text-[10px] font-bold text-slate-400 uppercase block mb-2">Real Site Benchmark</span>
                            <div className="aspect-square rounded-lg overflow-hidden bg-slate-100">
                              <img src={selectedSample.image} alt="Real Benchmark" className="w-full h-full object-cover" />
                            </div>
                          </div>
                          <div className="bg-white rounded-xl p-3 border border-slate-200 text-center">
                            <span className="text-[10px] font-bold text-emerald-600 uppercase block mb-2">DCGAN Synthetic Generation</span>
                            <div className="aspect-square rounded-lg overflow-hidden bg-slate-100 border border-emerald-300">
                              <img src={testResult.result.generated_image_url} alt="GAN Generated" className="w-full h-full object-cover" />
                            </div>
                          </div>
                        </div>
                        <div className="p-3 bg-white rounded-xl border border-slate-200 text-xs text-slate-600 flex items-center justify-between">
                          <span>Adversarial Synthesis: <strong className="font-mono text-emerald-600">z ~ N(0, I), FID: 98.3</strong></span>
                          <span className="badge badge-success">Photorealistic Synthesis</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* Model Cards */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          Implemented Models
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          <ModelCard
            name="Autoencoder & Classification"
            icon={Box}
            badge="86.8% Accuracy"
            description="AutoencoderV2 with U-Net skip connections & supervised classification head. Classifies construction images across 10 safety & PPE compliance categories."
            path="/autoencoder"
            color="#0891b2"
            metrics={[
              { label: 'Accuracy', value: '86.8%' },
              { label: 'Classes',  value: '10 PPE' },
              { label: 'Val MSE',  value: '0.0001' },
            ]}
          />
          <ModelCard
            name="VAE"
            icon={Sparkles}
            badge="PRACT 2"
            description="Probabilistic generative model with reparameterization trick. Generates novel construction scenes from structured latent space z ~ N(0, I)."
            path="/vae"
            color="#7c3aed"
            metrics={[
              { label: 'SSIM',   value: '0.72'  },
              { label: 'FID',    value: '142.5' },
              { label: 'Latent', value: '256-d' },
            ]}
          />
          <ModelCard
            name="DCGAN"
            icon={ImageIcon}
            badge="PRACT 3"
            description="Deep Convolutional GAN with adversarial training for photorealistic synthetic scene generation. Lowest FID score among all models."
            path="/gan"
            color="#059669"
            metrics={[
              { label: 'FID',    value: '98.3'  },
              { label: 'Latent', value: '100-d' },
              { label: 'Epochs', value: '50'    },
            ]}
          />
          <ModelCard
            name="Transformer"
            icon={Layers}
            badge="R2"
            description="Multi-modal Vision Transformer with cross-attention for future site state prediction and spatial risk hazard mapping."
            path="/transformer"
            color="#003153"
            metrics={[
              { label: 'Heads',   value: '8'      },
              { label: 'Embed',   value: '256-d'  },
              { label: 'Outputs', value: '2 maps' },
            ]}
          />
        </div>
      </motion.div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <motion.div variants={itemVariants}>
          <Link to="/compare" className="block">
            <div className="bg-white border border-slate-200 rounded-2xl p-5 flex items-center gap-4 group
                            cursor-pointer shadow-sm hover:shadow-md hover:border-amber-200 transition-all duration-300">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: 'linear-gradient(135deg, #f59e0b, #ef4444)',
                         boxShadow: '0 4px 14px rgba(245,158,11,0.25)' }}>
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-bold text-slate-800 group-hover:text-amber-600 transition-colors">
                  Comparative Analysis
                </h4>
                <p className="text-sm text-slate-500 mt-0.5 leading-snug">
                  Performance metrics, strengths/weaknesses, visual comparison across all models.
                </p>
              </div>
              <ArrowRight className="w-5 h-5 text-slate-400 group-hover:text-amber-500 group-hover:translate-x-1 transition-all flex-shrink-0" />
            </div>
          </Link>
        </motion.div>

        <motion.div variants={itemVariants}>
          <Link to="/ethics" className="block">
            <div className="bg-white border border-slate-200 rounded-2xl p-5 flex items-center gap-4 group
                            cursor-pointer shadow-sm hover:shadow-md hover:border-rose-200 transition-all duration-300">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{ background: 'linear-gradient(135deg, #ef4444, #7c3aed)',
                         boxShadow: '0 4px 14px rgba(239,68,68,0.25)' }}>
                <Scale className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="font-bold text-slate-800 group-hover:text-rose-600 transition-colors">
                  Ethics &amp; Responsible AI
                </h4>
                <p className="text-sm text-slate-500 mt-0.5 leading-snug">
                  Bias analysis, misuse prevention, privacy, sustainability, and regulatory compliance.
                </p>
              </div>
              <ArrowRight className="w-5 h-5 text-slate-400 group-hover:text-rose-500 group-hover:translate-x-1 transition-all flex-shrink-0" />
            </div>
          </Link>
        </motion.div>
      </div>

      {/* EDA Visualisations */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><BookOpen className="w-4 h-4" /></div>
          Exploratory Data Analysis
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <EDAChart title="Class Distribution"  imgUrl={`${API}/data/processed/features/class_distribution.png`} className="lg:col-span-2 h-[320px]" />
          <EDAChart title="BBox Spatial Density" imgUrl={`${API}/data/processed/features/bbox_density.png`}       className="h-[320px]" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
          <EDAChart title="Co-occurrence Matrix" imgUrl={`${API}/data/processed/features/co_occurrence.png`}       className="h-[300px]" />
          <EDAChart title="Resolution Analysis"  imgUrl={`${API}/data/processed/features/resolution_analysis.png`} className="h-[300px]" />
          <EDAChart title="Pixel Intensity"      imgUrl={`${API}/data/processed/features/pixel_intensity.png`}     className="h-[300px]" />
        </div>
      </motion.div>

    </motion.div>
  );
}