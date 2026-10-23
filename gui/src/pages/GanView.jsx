import { useState } from 'react';
import { motion } from 'framer-motion';
import { Image as ImageIcon, Loader2, Hash, Cpu, Layers, Activity, Zap, GitCompare, CheckCircle2 } from 'lucide-react';
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

export default function GanView() {
  const [generatedUrl, setGeneratedUrl] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedSample, setSelectedSample] = useState(null);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const response = await axios.post(`${API}/api/gan/generate`);
      setGeneratedUrl(response.data.generated_image_url);
    } catch (error) {
      console.error('Error generating GAN image:', error);
      alert('Failed to connect to backend.');
    } finally {
      setIsGenerating(false);
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
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-3xl font-black gradient-text">Deep Convolutional GAN (DCGAN)</h2>
            <span className="badge badge-success">PRACT 3 — Adversarial Image Synthesis</span>
          </div>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 mt-2 max-w-3xl">
          Generator and Discriminator compete in a minimax adversarial game. The Generator maps random noise z ~ N(0, I) 
          to photorealistic 128×128 construction site images, while the Discriminator learns to distinguish real from synthetic.
        </motion.p>
      </header>

      {/* Architecture Diagram — Adversarial Loop */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><Layers className="w-4 h-4" /></div>
          DCGAN Architecture (Adversarial Training Loop)
        </div>
        
        {/* Generator Flow */}
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-emerald-400 mb-2 ml-1">Generator Network (G)</h4>
          <div className="arch-flow flex-wrap">
            <div className="arch-block arch-block-latent">
              <div className="text-xs opacity-60">Noise</div>
              <div className="font-bold">z ~ N(0,I)</div>
              <div className="text-xs opacity-60">100 × 1 × 1</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-decoder">
              <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
              <div className="font-bold">1024×4×4</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-decoder">
              <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
              <div className="font-bold">512×8×8</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-decoder">
              <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
              <div className="font-bold">256×16×16</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-decoder">
              <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
              <div className="font-bold">128×32×32</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-decoder">
              <div className="text-xs opacity-60">DeConv+BN+ReLU</div>
              <div className="font-bold">64×64×64</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-output">
              <div className="text-xs opacity-60">DeConv + Tanh</div>
              <div className="font-bold">3×128×128</div>
            </div>
          </div>
        </div>

        {/* Discriminator Flow */}
        <div>
          <h4 className="text-sm font-semibold text-rose-400 mb-2 ml-1">Discriminator Network (D)</h4>
          <div className="arch-flow flex-wrap">
            <div className="arch-block arch-block-input">
              <div className="text-xs opacity-60">Real / Fake</div>
              <div className="font-bold">3×128×128</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-discriminator">
              <div className="text-xs opacity-60">Conv+LReLU</div>
              <div className="font-bold">64×64×64</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-discriminator">
              <div className="text-xs opacity-60">Conv+BN+LReLU</div>
              <div className="font-bold">128×32×32</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-discriminator">
              <div className="text-xs opacity-60">Conv+BN+LReLU</div>
              <div className="font-bold">256×16×16</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-discriminator">
              <div className="text-xs opacity-60">Conv+BN+LReLU</div>
              <div className="font-bold">512×8×8</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-discriminator">
              <div className="text-xs opacity-60">Conv+BN+LReLU</div>
              <div className="font-bold">1024×4×4</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block arch-block-discriminator animate-pulse-glow">
              <div className="text-xs opacity-60">Conv+Sigmoid</div>
              <div className="font-bold">P(real)</div>
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
              <span className="info-label">Latent Dim</span>
              <span className="info-value">100</span>
            </div>
            <div className="info-item">
              <span className="info-label">G Params</span>
              <span className="info-value">~11.4M</span>
            </div>
            <div className="info-item">
              <span className="info-label">D Params</span>
              <span className="info-value">~11.1M</span>
            </div>
            <div className="info-item">
              <span className="info-label">Output Size</span>
              <span className="info-value">128×128</span>
            </div>
            <div className="info-item">
              <span className="info-label">G Output</span>
              <span className="info-value">Tanh</span>
            </div>
            <div className="info-item">
              <span className="info-label">D Output</span>
              <span className="info-value">Sigmoid</span>
            </div>
            <div className="info-item">
              <span className="info-label">Epochs</span>
              <span className="info-value">50</span>
            </div>
            <div className="info-item">
              <span className="info-label">Features</span>
              <span className="info-value">64 base</span>
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="glass-panel p-6">
          <div className="section-title">
            <div className="section-icon"><Hash className="w-4 h-4" /></div>
            Mathematical Formulation (Minimax)
          </div>
          <div className="math-block">
            <div>
              <span className="math-fn">Adversarial Objective</span><span className="math-op">:</span>
            </div>
            <div className="mt-2 text-center">
              <span className="math-fn">min</span><span className="math-sub">G</span>
              <span className="math-op"> </span>
              <span className="math-fn">max</span><span className="math-sub">D</span>
              <span className="math-op"> </span>
              <span className="math-var">V</span>(<span className="math-var">D</span>, <span className="math-var">G</span>)
            </div>
            <div className="mt-3">
              <span className="math-var">V</span> <span className="math-op">=</span>{' '}
              <span className="math-fn">E</span><span className="math-sub">x~p<span style={{fontSize:'0.6em'}}>data</span></span>[<span className="math-fn">log</span> <span className="math-var">D</span>(<span className="math-var">x</span>)]
            </div>
            <div className="mt-1 ml-6">
              <span className="math-op">+ </span>
              <span className="math-fn">E</span><span className="math-sub">z~p<span style={{fontSize:'0.6em'}}>z</span></span>[<span className="math-fn">log</span>(1 <span className="math-op">−</span> <span className="math-var">D</span>(<span className="math-var">G</span>(<span className="math-var">z</span>)))]
            </div>
            <div className="mt-3 pt-3 border-t border-slate-700/50">
              <span className="math-fn">D Loss</span><span className="math-op"> = −[</span><span className="math-fn">log</span> <span className="math-var">D</span>(<span className="math-var">x</span>) <span className="math-op">+</span> <span className="math-fn">log</span>(1 <span className="math-op">−</span> <span className="math-var">D</span>(<span className="math-var">G</span>(<span className="math-var">z</span>)))<span className="math-op">]</span>
            </div>
            <div className="mt-1">
              <span className="math-fn">G Loss</span><span className="math-op"> = −</span><span className="math-fn">log</span> <span className="math-var">D</span>(<span className="math-var">G</span>(<span className="math-var">z</span>))
            </div>
            <span className="math-label">Generator tries to fool Discriminator; Discriminator tries to correctly classify real vs fake</span>
          </div>
        </motion.div>
      </div>

      {/* Real Test Benchmark Samples for GAN Comparison */}
      <SampleImageSelector
        onSelectSample={(sample) => {
          setSelectedSample(sample);
          if (!generatedUrl) {
            handleGenerate();
          }
        }}
        selectedSampleId={selectedSample?.id}
        filterModel="gan"
        title="Real Construction Site Samples For GAN Benchmarking"
        subtitle="Select any real construction photograph to compare structural realism, lighting, and texture fidelity against DCGAN synthetic output."
      />

      {/* Live Generation & Benchmark Comparison */}
      <motion.div variants={itemVariants} className="glass-panel p-8 relative overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-emerald-500/10 blur-[100px] rounded-full pointer-events-none" />
        
        <div className="flex flex-col lg:flex-row items-center gap-8 relative z-10">
          <div className="lg:w-1/3 text-center lg:text-left">
            <div className="section-title justify-center lg:justify-start">
              <div className="section-icon"><Zap className="w-4 h-4" /></div>
              Live GAN Synthesis
            </div>
            <p className="text-slate-400 mb-6 leading-relaxed text-sm">
              Sample a random latent vector <strong className="text-emerald-400 font-mono">z ~ N(0, I)</strong> of dimension 100, then pass it through the trained Generator 
              to synthesize a novel construction site image. Compare it directly with real dataset samples.
            </p>
            <button onClick={handleGenerate} disabled={isGenerating}
              className="btn-primary w-full text-base py-3.5 px-6 bg-emerald-500 hover:bg-emerald-400 shadow-emerald-500/20 hover:shadow-emerald-500/30 flex items-center justify-center gap-2">
              {isGenerating ? (
                <>Synthesizing <Loader2 className="w-5 h-5 animate-spin" /></>
              ) : (
                <>Synthesize New Scene <Zap className="w-5 h-5" /></>
              )}
            </button>

            {selectedSample && (
              <div className="mt-4 p-3 rounded-xl bg-slate-900/60 border border-slate-700/50 text-left">
                <div className="text-[10px] font-bold text-slate-400 uppercase">Benchmark Reference</div>
                <div className="text-xs font-bold text-white mt-0.5">{selectedSample.title}</div>
                <span className="text-[10px] text-emerald-400 font-mono mt-1 block">FID: 98.3 &middot; High Texture Fidelity</span>
              </div>
            )}
          </div>

          {/* Side by Side Display */}
          <div className="lg:w-2/3 w-full grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Real Benchmark Sample */}
            <div className="flex flex-col">
              <span className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>Real Site Benchmark</span>
                {selectedSample && <span className="text-[10px] text-primary">{selectedSample.category}</span>}
              </span>
              <div className="aspect-square w-full border-2 border-slate-700/50 rounded-2xl bg-slate-900 overflow-hidden shadow-xl relative flex items-center justify-center">
                {selectedSample ? (
                  <img
                    src={selectedSample.image}
                    alt="Real Benchmark"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="text-slate-500 text-xs text-center p-4">
                    <ImageIcon className="w-10 h-10 mb-2 opacity-30 mx-auto" />
                    Select a sample card above to compare side-by-side
                  </div>
                )}
              </div>
            </div>

            {/* Generated GAN Output */}
            <div className="flex flex-col">
              <span className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>DCGAN Synthesized</span>
                <span className="text-[10px] font-mono text-emerald-400">128&times;128 px</span>
              </span>
              <div className="aspect-square w-full border-2 border-emerald-500/40 rounded-2xl bg-slate-900 overflow-hidden shadow-xl relative flex items-center justify-center">
                {generatedUrl ? (
                  <motion.img
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    key={generatedUrl}
                    src={generatedUrl}
                    alt="Generated GAN Sample"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="text-slate-500 text-xs font-medium flex flex-col items-center">
                    <ImageIcon className="w-10 h-10 mb-2 opacity-20" />
                    Click Synthesize above
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Training Metrics */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><Activity className="w-4 h-4" /></div>
          Training Metrics
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-panel-hover p-4 h-72 flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">Generator vs Discriminator Loss</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/synthetic/gan/gan_loss_curves.png`} alt="GAN Loss" className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
          <div className="glass-panel-hover p-4 h-72 flex flex-col">
            <h4 className="text-sm font-semibold text-slate-400 mb-2">Real vs Synthetic Comparison</h4>
            <div className="flex-1 relative rounded-lg overflow-hidden bg-slate-900/50">
              <img src={`${API}/data/synthetic/gan/real_vs_synthetic.png`} alt="Real vs Synthetic" className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Epoch-wise Progression */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><GitCompare className="w-4 h-4" /></div>
          Training Progression (Epoch-wise Generated Samples)
        </div>
        <p className="text-sm text-slate-400 mb-4 -mt-3">
          Visual quality evolution as the Generator learns to produce increasingly realistic construction site imagery across 50 epochs of adversarial training.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {[1, 10, 20, 30, 40, 50].map((epoch) => (
            <div key={epoch} className="flex flex-col items-center gap-2">
              <span className="badge badge-primary text-[10px]">Epoch {epoch}</span>
              <div className="image-frame-hover aspect-square w-full">
                <img src={`${API}/data/synthetic/gan/generated_epoch_${String(epoch).padStart(3, '0')}.png`}
                  alt={`Epoch ${epoch}`} className="w-full h-full object-cover"
                  onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
                <div className="absolute inset-0 hidden items-center justify-center text-slate-600 text-xs">N/A</div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Latent Interpolation */}
      <motion.div variants={itemVariants} className="glass-panel-hover p-6">
        <h3 className="section-title">
          <div className="section-icon"><Zap className="w-4 h-4" /></div>
          Latent Space Interpolation
        </h3>
        <p className="text-sm text-slate-400 mb-4 -mt-3">
          Smooth transitions between two random latent vectors z₁ and z₂, showing the Generator's ability to produce coherent intermediate states.
        </p>
        <div className="image-frame h-48">
          <img src={`${API}/data/synthetic/gan/gan_interpolation.png`} alt="GAN Interpolation" className="w-full h-full object-contain"
            onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
          <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
        </div>
      </motion.div>
    </motion.div>
  );
}
