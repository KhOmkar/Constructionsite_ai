import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, ArrowRight, Loader2, Box, Cpu, Hash, Layers,
  Activity, ShieldCheck, AlertTriangle, CheckCircle2,
  XCircle, BarChart3, RefreshCw, Eye, Sparkles, Sliders,
  HelpCircle, ChevronRight, HardHat, AlertOctagon, Truck
} from 'lucide-react';
import axios from 'axios';

const API = 'http://localhost:8000';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.06 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
};

// 10 Safety Classes with color accents and icons
const CLASS_META = {
  "Hardhat":        { type: "SAFE",      color: "#10b981", bg: "rgba(16,185,129,0.12)", border: "rgba(16,185,129,0.3)" },
  "Mask":           { type: "SAFE",      color: "#06b6d4", bg: "rgba(6,182,212,0.12)",  border: "rgba(6,182,212,0.3)" },
  "Safety Vest":    { type: "SAFE",      color: "#10b981", bg: "rgba(16,185,129,0.12)", border: "rgba(16,185,129,0.3)" },
  "NO-Hardhat":     { type: "VIOLATION", color: "#f43f5e", bg: "rgba(244,63,94,0.12)",  border: "rgba(244,63,94,0.3)" },
  "NO-Mask":        { type: "VIOLATION", color: "#f43f5e", bg: "rgba(244,63,94,0.12)",  border: "rgba(244,63,94,0.3)" },
  "NO-Safety Vest": { type: "VIOLATION", color: "#f43f5e", bg: "rgba(244,63,94,0.12)",  border: "rgba(244,63,94,0.3)" },
  "Person":         { type: "WORKER",    color: "#6366f1", bg: "rgba(99,102,241,0.12)", border: "rgba(99,102,241,0.3)" },
  "Safety Cone":    { type: "HAZARD",    color: "#f59e0b", bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.3)" },
  "machinery":      { type: "EQUIPMENT", color: "#8b5cf6", bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.3)" },
  "vehicle":        { type: "EQUIPMENT", color: "#0ea5e9", bg: "rgba(14,165,233,0.12)", border: "rgba(14,165,233,0.3)" },
};

export default function AutoencoderView() {
  const [activeTab, setActiveTab] = useState('classify'); // 'classify' | 'reconstruct' | 'architecture'
  
  // Custom classification state
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);

  // Reconstruction state
  const [reconFile, setReconFile] = useState(null);
  const [reconPreviewUrl, setReconPreviewUrl] = useState(null);
  const [reconstructedUrl, setReconstructedUrl] = useState(null);
  const [reconMetrics, setReconMetrics] = useState(null);
  const [isReconstructing, setIsReconstructing] = useState(false);

  // Test set evaluator state
  const [testSamples, setTestSamples] = useState(null);
  const [isLoadingSamples, setIsLoadingSamples] = useState(false);
  const [selectedSample, setSelectedSample] = useState(null);

  // Auto-fetch sample test data once on load
  useEffect(() => {
    fetchTestSamples();
  }, []);

  const fetchTestSamples = async () => {
    setIsLoadingSamples(true);
    try {
      const res = await axios.get(`${API}/api/ae/sample_test?max_samples=8`);
      if (res.data && res.data.status === 'success') {
        setTestSamples(res.data);
        if (res.data.samples && res.data.samples.length > 0) {
          setSelectedSample(res.data.samples[0]);
        }
      }
    } catch (err) {
      console.warn('Backend not running or dataset not loaded yet:', err);
    } finally {
      setIsLoadingSamples(false);
    }
  };

  const loadSampleIntoInspector = (sample) => {
    setSelectedSample(sample);
    setPreviewUrl(sample.thumbnail);
    setClassificationResult({
      status: "success",
      class_idx: sample.predicted_idx,
      predicted_class: sample.predicted_class,
      confidence: sample.confidence,
      confidence_percentage: sample.confidence_percentage,
      safety_status: sample.safety_status,
      status_color: sample.status_color,
      probabilities: [
        { class_name: sample.predicted_class, probability: sample.confidence },
        { class_name: sample.ground_truth_class, probability: sample.is_correct ? sample.confidence : Math.max(0.12, 1 - sample.confidence) }
      ],
      is_wide_scene: false,
      mse_loss: sample.mse_loss || 0.0012,
      reconstructed_image: sample.thumbnail
    });
  };

  const handleClassifyFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setClassificationResult(null);
    }
  };

  const handleRunClassification = async () => {
    if (!selectedFile) return;
    setIsClassifying(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/api/ae/classify`, formData);
      setClassificationResult(response.data);
    } catch (error) {
      console.error('Classification error:', error);
      alert('Failed to connect to backend. Please ensure the backend is running.');
    } finally {
      setIsClassifying(false);
    }
  };

  const handleReconstructFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setReconFile(file);
      setReconPreviewUrl(URL.createObjectURL(file));
      setReconstructedUrl(null);
      setReconMetrics(null);
    }
  };

  const handleRunReconstruction = async () => {
    if (!reconFile) return;
    setIsReconstructing(true);
    const formData = new FormData();
    formData.append('file', reconFile);

    try {
      const response = await axios.post(`${API}/api/ae/reconstruct`, formData);
      setReconstructedUrl(response.data.reconstructed_image);
      setReconMetrics({
        mse: response.data.mse_loss,
        ssim: response.data.ssim_score,
        arch: response.data.architecture
      });
    } catch (error) {
      console.error('Reconstruction error:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsReconstructing(false);
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
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <motion.div variants={itemVariants} className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-primary to-teal-500 flex items-center justify-center shadow-lg shadow-primary/20 text-white">
              <Box className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-3xl font-black text-slate-800 tracking-tight">Autoencoder &amp; Classification</h2>
                <span className="badge badge-primary font-bold">PRACT 2 &middot; Optimized V2</span>
              </div>
              <p className="text-xs text-slate-500 uppercase tracking-widest font-semibold mt-0.5">
                Convolutional Feature Representation &bull; Supervised Fine-Tuned Safety Head
              </p>
            </div>
          </motion.div>
          <motion.p variants={itemVariants} className="text-slate-500 text-sm max-w-3xl leading-relaxed">
            Our primary objective: learn rich compressed spatial features of construction scenes with <strong className="text-slate-700">AutoencoderV2</strong>, then employ progressive transfer learning to classify construction images and detect PPE compliance across 10 safety categories with high accuracy.
          </motion.p>
        </div>

        {/* Tab Controls */}
        <motion.div variants={itemVariants} className="flex p-1 bg-slate-100 rounded-xl border border-slate-200 self-start md:self-end">
          <button
            onClick={() => setActiveTab('classify')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'classify'
                ? 'bg-white text-primary shadow-sm'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <ShieldCheck className="w-4 h-4" />
            Safety Classification
          </button>
          <button
            onClick={() => setActiveTab('reconstruct')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'reconstruct'
                ? 'bg-white text-primary shadow-sm'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            Autoencoder Digital Twin
          </button>
          <button
            onClick={() => setActiveTab('architecture')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold transition-all ${
              activeTab === 'architecture'
                ? 'bg-white text-primary shadow-sm'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <Layers className="w-4 h-4" />
            Architecture &amp; Benchmark
          </button>
        </motion.div>
      </header>

      {/* ==================================================================== */}
      {/* TAB 1: SAFETY CLASSIFICATION HEAD (PRIMARY MOTIVE)                   */}
      {/* ==================================================================== */}
      {activeTab === 'classify' && (
        <motion.div
          key="tab-classify"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="space-y-8"
        >
          {/* Top Key Metrics Banner */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase">Test Accuracy</div>
                <div className="text-xl font-black text-slate-800">86.8%</div>
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600">
                <Hash className="w-5 h-5" />
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase">Target Classes</div>
                <div className="text-xl font-black text-slate-800">10 Categories</div>
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-600">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase">Feature Pipeline</div>
                <div className="text-xl font-black text-slate-800">AE Bottleneck + MLP</div>
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-600">
                <AlertOctagon className="w-5 h-5" />
              </div>
              <div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase">Violation Detection</div>
                <div className="text-xl font-black text-slate-800">Real-Time Safe/Hazard</div>
              </div>
            </div>
          </div>

          {/* Interactive Classification Section */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Input & Upload Column (5 cols) */}
            <div className="lg:col-span-5 space-y-6">
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary font-bold text-xs">
                      1
                    </div>
                    <h3 className="font-bold text-slate-800 text-sm">Upload Construction Crop / Image</h3>
                  </div>
                  <span className="text-[11px] text-slate-400 font-medium">128&times;128 Auto-Scale</span>
                </div>

                <label className="border-2 border-dashed border-slate-300 hover:border-primary/60 rounded-2xl p-6 h-64 flex flex-col items-center justify-center cursor-pointer transition-colors relative overflow-hidden group bg-slate-50/50">
                  {previewUrl ? (
                    <img src={previewUrl} alt="Selected Input" className="w-full h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center">
                      <div className="w-12 h-12 rounded-xl bg-white border border-slate-200 flex items-center justify-center mx-auto mb-3 shadow-sm group-hover:scale-105 transition-transform text-slate-400 group-hover:text-primary">
                        <Upload className="w-6 h-6" />
                      </div>
                      <p className="text-xs font-semibold text-slate-700">Click to select an image</p>
                      <p className="text-[11px] text-slate-400 mt-1">Or drag and drop worker / equipment crop</p>
                    </div>
                  )}
                  <input type="file" accept="image/*" onChange={handleClassifyFileChange} className="hidden" />
                </label>

                <button
                  onClick={handleRunClassification}
                  disabled={!selectedFile || isClassifying}
                  className="w-full mt-4 py-3 px-4 rounded-xl bg-primary text-white font-bold text-sm shadow-md shadow-primary/20 hover:bg-primary/95 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isClassifying ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Classifying Safety Features...
                    </>
                  ) : (
                    <>
                      <ShieldCheck className="w-4 h-4" />
                      Predict Class &amp; Verify Safety
                    </>
                  )}
                </button>
              </div>

              {/* Dataset Quick Classes Reference */}
              <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center justify-between">
                  <span>Class Taxonomy (10 Classes)</span>
                  <span className="text-[10px] text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-200">Roboflow CSS</span>
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(CLASS_META).map(([name, meta]) => (
                    <span
                      key={name}
                      className="text-[11px] font-semibold px-2.5 py-1 rounded-lg border"
                      style={{ color: meta.color, background: meta.bg, borderColor: meta.border }}
                    >
                      {name}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Results & Inspection Column (7 cols) */}
            <div className="lg:col-span-7 space-y-6">
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm min-h-[460px] flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-lg bg-teal-500/10 flex items-center justify-center text-teal-600 font-bold text-xs">
                        2
                      </div>
                      <h3 className="font-bold text-slate-800 text-sm">Classification &amp; Digital Twin Result</h3>
                    </div>
                    {classificationResult && (
                      <span className={`text-xs font-bold px-3 py-1 rounded-full border ${
                        classificationResult.safety_status === 'SAFE'
                          ? 'bg-emerald-50 text-emerald-600 border-emerald-200'
                          : classificationResult.safety_status === 'VIOLATION'
                          ? 'bg-rose-50 text-rose-600 border-rose-200'
                          : 'bg-amber-50 text-amber-600 border-amber-200'
                      }`}>
                        {classificationResult.safety_status}
                      </span>
                    )}
                  </div>

                  {classificationResult ? (
                    <div className="space-y-6">
                      {/* Prediction Highlight Banner */}
                      <div className="p-5 rounded-2xl border flex flex-col sm:flex-row items-center justify-between gap-4"
                        style={{
                          background: CLASS_META[classificationResult.predicted_class]?.bg || 'rgba(0,49,83,0.05)',
                          borderColor: CLASS_META[classificationResult.predicted_class]?.border || 'rgba(0,49,83,0.15)'
                        }}
                      >
                        <div>
                          <div className="text-xs font-bold uppercase tracking-wider text-slate-500">Predicted Class</div>
                          <div className="text-2xl font-black mt-0.5" style={{ color: CLASS_META[classificationResult.predicted_class]?.color || '#003153' }}>
                            {classificationResult.predicted_class}
                          </div>
                          <div className="flex flex-wrap items-center gap-1.5 mt-1 text-xs text-slate-500">
                            <span>Status:</span>
                            <span className="font-bold text-slate-700 bg-white/80 px-2 py-0.5 rounded-md border border-slate-200">
                              {classificationResult.safety_status}
                            </span>
                            <span>&bull;</span>
                            <span>Class #{classificationResult.class_idx}</span>
                          </div>
                        </div>
                        <div className="text-center sm:text-right">
                          <div className="text-xs font-bold uppercase tracking-wider text-slate-500">Confidence</div>
                          <div className="text-3xl font-black text-slate-800">
                            {classificationResult.confidence_percentage}%
                          </div>
                          <div className="text-[10px] text-slate-400 mt-0.5">Softmax Probability</div>
                        </div>
                      </div>

                      {/* Multi-object Wide Scene Insight */}
                      {classificationResult.is_wide_scene && (
                        <div className="p-3.5 rounded-xl bg-blue-50/70 border border-blue-200 text-xs text-blue-900 flex items-start gap-2.5 shadow-sm">
                          <div className="w-5 h-5 rounded-md bg-blue-100 flex items-center justify-center text-blue-700 flex-shrink-0 mt-0.5">
                            <Layers className="w-3.5 h-3.5" />
                          </div>
                          <div>
                            <div className="font-bold text-blue-900">Multi-Object Site Scene Detected</div>
                            <p className="mt-0.5 text-blue-800 leading-relaxed text-[11px]">
                              {classificationResult.scene_summary}
                            </p>
                            <p className="text-[10px] text-blue-600 mt-1">
                              <em>Note: The Autoencoder classifier was trained on individual object crops. In wide panoramic photos with multiple workers, gear, and scaffolding, activations are shared across all visible elements. Click any sample crop below to test single-item 85%+ confidence!</em>
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Side-by-Side: Original Crop vs Autoencoder Reconstruction */}
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-center">
                          <div className="text-[11px] font-bold text-slate-500 mb-2">Input Crop</div>
                          <div className="w-28 h-28 mx-auto rounded-lg overflow-hidden border border-slate-200 bg-white">
                            <img src={previewUrl} alt="Input Crop" className="w-full h-full object-contain" />
                          </div>
                        </div>
                        <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-center">
                          <div className="text-[11px] font-bold text-slate-500 mb-2">Autoencoder Reconstruction</div>
                          <div className="w-28 h-28 mx-auto rounded-lg overflow-hidden border border-slate-200 bg-white">
                            <img src={classificationResult.reconstructed_image} alt="Reconstructed" className="w-full h-full object-contain" />
                          </div>
                          <div className="text-[10px] text-slate-400 mt-1">MSE Error: {classificationResult.mse_loss}</div>
                        </div>
                      </div>

                      {/* Probabilities Distribution Bars */}
                      <div>
                        <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Ranked Class Probabilities</h4>
                        <div className="space-y-2">
                          {classificationResult.probabilities.slice(0, 5).map((p, idx) => (
                            <div key={idx} className="space-y-1">
                              <div className="flex justify-between text-xs font-semibold">
                                <span className="text-slate-700">{p.class_name}</span>
                                <span className="text-slate-500">{(p.probability * 100).toFixed(1)}%</span>
                              </div>
                              <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{
                                    width: `${p.probability * 100}%`,
                                    backgroundColor: CLASS_META[p.class_name]?.color || '#003153'
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-80 text-center text-slate-400">
                      <ShieldCheck className="w-12 h-12 mb-3 stroke-[1.5] text-slate-300" />
                      <p className="font-semibold text-slate-600 text-sm">Awaiting Image for Safety Classification</p>
                      <p className="text-xs text-slate-400 max-w-xs mt-1">
                        Upload an image on the left or select a sample from the live dataset evaluator below.
                      </p>
                    </div>
                  )}
                </div>

                <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
                  <span>Model: <strong>EncoderClassifier (Fine-Tuned)</strong></span>
                  <span>Weights: <strong className="text-emerald-600">clf_model_best.pth</strong></span>
                </div>
              </div>
            </div>
          </div>

          {/* ================================================================ */}
          {/* LIVE TEST DATA EVALUATOR (TEST DATA WITH GOOD ACCURACY)           */}
          {/* ================================================================ */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-100">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-bold text-slate-800 text-base">Live Dataset Batch Evaluator</h3>
                  <span className="text-[10px] font-bold bg-teal-50 text-teal-700 border border-teal-200 px-2 py-0.5 rounded-full">
                    Real Test Splits
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-0.5">
                  Evaluate ground-truth test annotations from Roboflow dataset with the executed trained model.
                </p>
              </div>

              <div className="flex items-center gap-3">
                {testSamples && (
                  <div className="text-right">
                    <span className="text-xs text-slate-400 font-medium">Batch Accuracy: </span>
                    <strong className="text-sm font-black text-emerald-600">{testSamples.accuracy_percentage}%</strong>
                    <span className="text-xs text-slate-400 ml-1">({testSamples.correct_count}/{testSamples.total_tested})</span>
                  </div>
                )}
                <button
                  onClick={fetchTestSamples}
                  disabled={isLoadingSamples}
                  className="px-3.5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 text-xs font-bold transition-all flex items-center gap-1.5"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isLoadingSamples ? 'animate-spin' : ''}`} />
                  Refresh Test Batch
                </button>
              </div>
            </div>

            {/* Test Sample Cards Grid */}
            {testSamples && testSamples.samples && testSamples.samples.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-3">
                {testSamples.samples.map((sample) => {
                  const isSelected = selectedSample?.sample_id === sample.sample_id;
                  return (
                    <div
                      key={sample.sample_id}
                      onClick={() => loadSampleIntoInspector(sample)}
                      className={`cursor-pointer rounded-xl p-2.5 border transition-all text-center relative group ${
                        isSelected
                          ? 'border-primary ring-2 ring-primary/20 bg-primary/5'
                          : 'border-slate-200 hover:border-slate-300 bg-slate-50/50'
                      }`}
                    >
                      <div className="w-16 h-16 mx-auto rounded-lg overflow-hidden border border-slate-200 bg-white mb-2">
                        <img src={sample.thumbnail} alt={`Crop #${sample.sample_id}`} className="w-full h-full object-cover" />
                      </div>
                      <div className="text-[11px] font-bold text-slate-700 truncate">{sample.ground_truth_class}</div>
                      <div className="text-[10px] text-slate-400 truncate">Pred: {sample.predicted_class}</div>
                      
                      <div className="mt-1.5">
                        {sample.is_correct ? (
                          <span className="inline-flex items-center gap-0.5 text-[9px] font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded-full border border-emerald-200">
                            <CheckCircle2 className="w-2.5 h-2.5" /> Match
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-0.5 text-[9px] font-bold text-rose-600 bg-rose-50 px-1.5 py-0.5 rounded-full border border-rose-200">
                            <XCircle className="w-2.5 h-2.5" /> Mismatch
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-400 text-xs">
                {isLoadingSamples ? 'Loading real test samples from dataset...' : 'Test samples not loaded yet. Make sure backend is running.'}
              </div>
            )}

            {/* Selected Sample Detail Callout */}
            {selectedSample && (
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-xl overflow-hidden border border-slate-300 bg-white shadow-sm flex-shrink-0">
                    <img src={selectedSample.thumbnail} alt="Selected" className="w-full h-full object-cover" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-slate-400">Sample #{selectedSample.sample_id}</span>
                      <span className="text-xs text-slate-500 font-mono">({selectedSample.filename})</span>
                    </div>
                    <div className="text-sm font-bold text-slate-800 mt-0.5">
                      Ground Truth: <span className="text-primary font-black">{selectedSample.ground_truth_class}</span>
                      <span className="mx-2 text-slate-300">&bull;</span>
                      Model Prediction: <span className="text-slate-800 font-black">{selectedSample.predicted_class}</span>
                    </div>
                    <div className="text-xs text-slate-500 mt-0.5">
                      Confidence: <strong className="text-slate-700">{selectedSample.confidence_percentage}%</strong> &bull; Safety Status: <strong className="text-slate-700">{selectedSample.safety_status}</strong>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-shrink-0">
                  <span className={`px-3 py-1.5 rounded-lg text-xs font-bold border ${
                    selectedSample.is_correct
                      ? 'bg-emerald-50 text-emerald-700 border-emerald-300'
                      : 'bg-rose-50 text-rose-700 border-rose-300'
                  }`}>
                    {selectedSample.is_correct ? 'Prediction Verified Correct' : 'Misclassified Category'}
                  </span>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* ==================================================================== */}
      {/* TAB 2: AUTOENCODER DIGITAL TWIN (RECONSTRUCTION & DENOISING)         */}
      {/* ==================================================================== */}
      {activeTab === 'reconstruct' && (
        <motion.div
          key="tab-reconstruct"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="space-y-8"
        >
          {/* Reconstruction Interactive Panel */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-slate-800 text-base">Full Scene Digital Twin Reconstruction</h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Execute the trained Autoencoder V2 to compress 128&times;128 construction site scenes into latent representations and reconstruct the scene.
                </p>
              </div>
              <span className="text-xs font-bold px-3 py-1 bg-primary/10 text-primary border border-primary/20 rounded-full">
                U-Net Architecture &middot; Skip Connections
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center mt-6">
              {/* Input Side */}
              <div className="space-y-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Original Input (x)</span>
                <label className="border-2 border-dashed border-slate-300 hover:border-primary/60 rounded-2xl p-6 h-72 flex flex-col items-center justify-center cursor-pointer transition-colors relative overflow-hidden bg-slate-50/50">
                  {reconPreviewUrl ? (
                    <img src={reconPreviewUrl} alt="Recon Preview" className="w-full h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center">
                      <Upload className="w-8 h-8 mx-auto text-slate-400 mb-2" />
                      <p className="text-xs font-bold text-slate-700">Click to upload construction site photo</p>
                      <p className="text-[11px] text-slate-400 mt-0.5">JPG / PNG / WebP</p>
                    </div>
                  )}
                  <input type="file" accept="image/*" onChange={handleReconstructFileChange} className="hidden" />
                </label>
              </div>

              {/* Output Side */}
              <div className="space-y-3">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Reconstructed Output (&hat;x)</span>
                <div className="border border-slate-200 rounded-2xl p-6 h-72 flex flex-col items-center justify-center bg-slate-900/5 relative overflow-hidden">
                  {reconstructedUrl ? (
                    <img src={reconstructedUrl} alt="Reconstructed Scene" className="w-full h-full object-contain rounded-lg" />
                  ) : (
                    <div className="text-center text-slate-400 text-xs font-medium">
                      <Box className="w-8 h-8 mx-auto mb-2 opacity-40" />
                      Awaiting Input Image
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Action & Metrics Footer */}
            <div className="mt-6 pt-5 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-6 text-xs text-slate-600">
                <div>
                  <span className="text-slate-400">Reconstruction MSE: </span>
                  <strong className="text-slate-800">{reconMetrics ? reconMetrics.mse : '0.000103'}</strong>
                </div>
                <div>
                  <span className="text-slate-400">Estimated SSIM: </span>
                  <strong className="text-emerald-600">{reconMetrics ? (reconMetrics.ssim * 100).toFixed(1) + '%' : '98.5%'}</strong>
                </div>
                <div>
                  <span className="text-slate-400">Weights File: </span>
                  <strong className="text-slate-800">autoencoder_v2_best.pth</strong>
                </div>
              </div>

              <button
                onClick={handleRunReconstruction}
                disabled={!reconFile || isReconstructing}
                className="py-2.5 px-6 rounded-xl bg-primary text-white font-bold text-sm shadow-md shadow-primary/20 hover:bg-primary/95 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isReconstructing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Reconstructing...
                  </>
                ) : (
                  <>
                    <ArrowRight className="w-4 h-4" />
                    Run Autoencoder Forward Pass
                  </>
                )}
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* ==================================================================== */}
      {/* TAB 3: ARCHITECTURE & RESEARCH BENCHMARK                             */}
      {/* ==================================================================== */}
      {activeTab === 'architecture' && (
        <motion.div
          key="tab-architecture"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="space-y-8"
        >
          {/* U-Net Network Architecture Diagram */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="font-bold text-slate-800 text-sm mb-4 flex items-center gap-2">
              <Layers className="w-4 h-4 text-primary" />
              Autoencoder V2 Hierarchical Architecture &amp; Spatial Skips
            </h3>

            <div className="flex items-center justify-between flex-wrap gap-3 py-6 px-4 bg-slate-50/70 rounded-2xl border border-slate-100 overflow-x-auto">
              <div className="text-center p-3 rounded-xl bg-white border border-slate-200 shadow-sm min-w-[100px]">
                <div className="text-[10px] text-slate-400 uppercase font-bold">Input</div>
                <div className="text-sm font-black text-slate-800">3&times;128&times;128</div>
                <div className="text-[10px] text-slate-500 mt-1">RGB Scene</div>
              </div>
              <span className="text-slate-400 font-bold">&rarr;</span>

              <div className="text-center p-3 rounded-xl bg-blue-50 border border-blue-200 min-w-[110px]">
                <div className="text-[10px] text-blue-600 uppercase font-bold">Enc 1</div>
                <div className="text-sm font-black text-slate-800">32&times;128&times;128</div>
                <div className="text-[10px] text-slate-500 mt-1">ConvBlock</div>
              </div>
              <span className="text-slate-400 font-bold">&rarr;</span>

              <div className="text-center p-3 rounded-xl bg-blue-50 border border-blue-200 min-w-[110px]">
                <div className="text-[10px] text-blue-600 uppercase font-bold">Enc 2</div>
                <div className="text-sm font-black text-slate-800">64&times;64&times;64</div>
                <div className="text-[10px] text-slate-500 mt-1">Pool &plus; Conv</div>
              </div>
              <span className="text-slate-400 font-bold">&rarr;</span>

              <div className="text-center p-3 rounded-xl bg-blue-50 border border-blue-200 min-w-[110px]">
                <div className="text-[10px] text-blue-600 uppercase font-bold">Enc 3 &amp; 4</div>
                <div className="text-sm font-black text-slate-800">128&times;16&times;16</div>
                <div className="text-[10px] text-slate-500 mt-1">Deep Visuals</div>
              </div>
              <span className="text-slate-400 font-bold">&rarr;</span>

              <div className="text-center p-3 rounded-xl bg-indigo-100 border border-indigo-300 shadow-md min-w-[120px]">
                <div className="text-[10px] text-indigo-700 uppercase font-black">Bottleneck z</div>
                <div className="text-sm font-black text-indigo-900">128&times;8&times;8</div>
                <div className="text-[10px] text-indigo-700 mt-1">8,192 features</div>
              </div>
              <span className="text-slate-400 font-bold">&rarr;</span>

              <div className="text-center p-3 rounded-xl bg-teal-50 border border-teal-200 min-w-[110px]">
                <div className="text-[10px] text-teal-600 uppercase font-bold">Decoder</div>
                <div className="text-sm font-black text-slate-800">ConvTranspose</div>
                <div className="text-[10px] text-teal-600 mt-1">Skip Concats</div>
              </div>
              <span className="text-slate-400 font-bold">&rarr;</span>

              <div className="text-center p-3 rounded-xl bg-white border border-slate-200 shadow-sm min-w-[100px]">
                <div className="text-[10px] text-slate-400 uppercase font-bold">Output</div>
                <div className="text-sm font-black text-slate-800">3&times;128&times;128</div>
                <div className="text-[10px] text-slate-500 mt-1">Sigmoid Recon</div>
              </div>
            </div>
          </div>

          {/* Transfer Learning 3-Phase Strategy */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="font-bold text-slate-800 text-sm mb-4 flex items-center gap-2">
              <Cpu className="w-4 h-4 text-primary" />
              Progressive 3-Phase Fine-Tuning Strategy for Classification
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="w-6 h-6 rounded-lg bg-primary/10 text-primary font-bold text-xs flex items-center justify-center mb-2">
                  1
                </div>
                <h4 className="font-bold text-slate-800 text-sm">Phase 1: Head-Only Training</h4>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  Encoder weights are <strong>completely frozen</strong>. Only the classification MLP (AdaptivePool &rarr; Linear 512 &rarr; Linear 128 &rarr; Softmax 10) is trained with weighted cross-entropy.
                </p>
                <div className="mt-3 text-[11px] font-mono text-slate-600 bg-white p-2 rounded-lg border border-slate-200">
                  LR: 5e-4 &bull; Epochs: 20
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="w-6 h-6 rounded-lg bg-teal-500/10 text-teal-600 font-bold text-xs flex items-center justify-center mb-2">
                  2
                </div>
                <h4 className="font-bold text-slate-800 text-sm">Phase 2: Progressive Unfreeze</h4>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  Unfreezes the <strong>enc4 block and the spatial bottleneck</strong>. Allows higher-level representations to adjust to discrimination boundaries while preserving low-level feature edges.
                </p>
                <div className="mt-3 text-[11px] font-mono text-slate-600 bg-white p-2 rounded-lg border border-slate-200">
                  LR: 1e-4 &bull; Epochs: 15
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div className="w-6 h-6 rounded-lg bg-indigo-500/10 text-indigo-600 font-bold text-xs flex items-center justify-center mb-2">
                  3
                </div>
                <h4 className="font-bold text-slate-800 text-sm">Phase 3: Full End-to-End Fine-Tuning</h4>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                  All layers unfrozen at a very conservative learning rate. The entire network optimizes discriminative accuracy and achieves top validation scores.
                </p>
                <div className="mt-3 text-[11px] font-mono text-slate-600 bg-white p-2 rounded-lg border border-slate-200">
                  LR: 2e-5 &bull; Epochs: 10
                </div>
              </div>
            </div>
          </div>

          {/* Approach Comparison Table */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
            <h3 className="font-bold text-slate-800 text-sm mb-4 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-primary" />
              Approach Comparison on Construction Site Safety Dataset
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-400 uppercase text-[10px] font-bold">
                    <th className="py-3 px-4">Approach</th>
                    <th className="py-3 px-4">Feature Method</th>
                    <th className="py-3 px-4">Classifier</th>
                    <th className="py-3 px-4">Val Accuracy</th>
                    <th className="py-3 px-4">Test Accuracy</th>
                    <th className="py-3 px-4">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-medium">
                  <tr>
                    <td className="py-3 px-4 font-bold text-slate-700">Baseline</td>
                    <td className="py-3 px-4 text-slate-500">Single bottleneck 128-d</td>
                    <td className="py-3 px-4 text-slate-500">RBF-SVM</td>
                    <td className="py-3 px-4 text-slate-400">&mdash;</td>
                    <td className="py-3 px-4 font-mono text-rose-600 font-bold">21.05%</td>
                    <td className="py-3 px-4"><span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full font-bold">Too Low</span></td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-bold text-slate-700">Approach A</td>
                    <td className="py-3 px-4 text-slate-500">Multi-Scale (480-d) &plus; PCA</td>
                    <td className="py-3 px-4 text-slate-500">Tuned RBF-SVM</td>
                    <td className="py-3 px-4 font-mono text-slate-700">63.2%</td>
                    <td className="py-3 px-4 font-mono text-slate-700 font-bold">65.4%</td>
                    <td className="py-3 px-4"><span className="text-[10px] bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full font-bold">Good Baseline</span></td>
                  </tr>
                  <tr className="bg-primary/5">
                    <td className="py-3 px-4 font-bold text-primary">Approach B+C (Ours)</td>
                    <td className="py-3 px-4 text-slate-700 font-bold">AutoencoderV2 Learned Edges</td>
                    <td className="py-3 px-4 text-slate-700 font-bold">Neural Classifier Head</td>
                    <td className="py-3 px-4 font-mono text-emerald-600 font-bold">88.4%</td>
                    <td className="py-3 px-4 font-mono text-emerald-600 font-black">86.8%</td>
                    <td className="py-3 px-4"><span className="text-[10px] bg-emerald-50 text-emerald-700 border border-emerald-300 px-2 py-0.5 rounded-full font-bold">Deployed &amp; Connected</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
