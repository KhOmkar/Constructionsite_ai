import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Zap, Target, Award, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const API = 'http://localhost:8000';

/* ═══ Metric data for all 4 models ═══ */
const metricsData = {
  headers: ['Metric', 'Autoencoder (CAE)', 'VAE (CVAE)', 'DCGAN', 'Hazard ViT (120 Ep)', 'Best'],
  rows: [
    { metric: 'SSIM ↑', ae: '0.78', vae: '0.72', gan: 'N/A*', vit: '0.81', best: 'ViT / AE', bestColor: 'text-cyan-400' },
    { metric: 'PSNR (dB) ↑', ae: '22.4', vae: '20.8', gan: 'N/A*', vit: '22.4', best: 'AE / ViT', bestColor: 'text-primary' },
    { metric: 'Safety F1-Score ↑', ae: 'N/A', vae: '0.64', gan: 'N/A', vit: '0.846', best: 'ViT', bestColor: 'text-cyan-400' },
    { metric: 'Epochs Trained', ae: '30', vae: '30', gan: '50', vit: '120 Epochs', best: 'ViT', bestColor: 'text-cyan-400' },
    { metric: 'FID Score ↓', ae: 'N/A', vae: '142.5', gan: '98.3', vit: 'N/A†', best: 'GAN', bestColor: 'text-emerald-400' },
    { metric: 'Latent / Token Repr', ae: '256-d vector', vae: '256-d Gaussian', gan: '100-d noise', vit: '64 + 1 [CLS] tokens', best: 'ViT', bestColor: 'text-cyan-400' },
    { metric: 'Output Modalities', ae: 'Reconstruction', vae: 'Synthesis', gan: 'Synthesis', vit: 'Classes + Risk + Attn', best: 'ViT', bestColor: 'text-cyan-400' },
    { metric: 'Training Stability', ae: 'High', vae: 'High', gan: 'Low', vit: 'High (AdamW + Cosine)', best: 'AE / ViT', bestColor: 'text-primary' },
    { metric: 'Spatial Hazard Mapping', ae: 'None', vae: 'None', gan: 'None', vit: 'Dynamic Thermal Jet', best: 'ViT', bestColor: 'text-cyan-400' },
    { metric: 'Dataset Foundation', ae: 'Roboflow CSS', vae: 'Roboflow CSS', gan: 'Roboflow CSS', vit: 'CS-10k & Roboflow', best: 'ViT', bestColor: 'text-cyan-400' },
  ]
};

const modelComparison = [
  {
    name: 'Autoencoder',
    color: 'primary',
    icon: '🔧',
    strengths: [
      'Best reconstruction quality (highest SSIM 0.78 & PSNR 22.4 dB)',
      'Stable and fast deterministic convergence during training',
      'Compact 256-d bottleneck ideal for edge camera bandwidth compression',
      'Reconstruction error heatmap |x − x̂| enables visual anomaly detection',
    ],
    weaknesses: [
      'Cannot generate novel samples — only reconstruct inputs',
      'Latent space is unstructured (no smooth interpolation)',
      'Prone to memorization with small datasets',
    ],
    useCases: [
      'Image compression for site surveillance',
      'Anomaly detection in safety gear',
      'Feature extraction for downstream classifiers',
    ],
    rating: { quality: 90, stability: 95, generation: 10, speed: 92 },
  },
  {
    name: 'VAE & CVAE',
    color: 'secondary',
    icon: '✨',
    strengths: [
      'Can both reconstruct AND generate novel images from Gaussian prior',
      'Structured latent space enables smooth interpolation between scenes',
      'Conditional CVAE generates specific safety classes (Hardhat, Vest)',
      'Data augmentation via synthetic sample generation',
    ],
    weaknesses: [
      'Reconstructions are blurrier than AE (KL regularization trade-off)',
      'Lower SSIM/PSNR compared to deterministic AE',
      'Requires careful β-tuning for KL vs reconstruction balance',
    ],
    useCases: [
      'Synthetic data augmentation for safety model training',
      'Latent space exploration and interpolation',
      'Probabilistic anomaly detection via log-likelihood',
    ],
    rating: { quality: 72, stability: 88, generation: 75, speed: 85 },
  },
  {
    name: 'DCGAN',
    color: 'emerald',
    icon: '⚡',
    strengths: [
      'Best generative quality — most photorealistic outputs',
      'Lowest FID score (98.3) indicating closest match to real data distribution',
      'Adversarial training captures fine-grained high-frequency textures',
      'Excellent for large-scale synthetic data generation',
    ],
    weaknesses: [
      'Training instability — mode collapse and oscillation risk',
      'Cannot reconstruct input images (generation only)',
      'Requires careful hyperparameter tuning (lr, β₁, β₂)',
      'No meaningful latent space for downstream analysis',
    ],
    useCases: [
      'High-fidelity synthetic scene generation',
      'Data augmentation with realistic construction imagery',
      'Privacy-preserving training data synthesis',
    ],
    rating: { quality: 85, stability: 45, generation: 95, speed: 60 },
  },
  {
    name: 'Hazard Transformer (ViT)',
    color: 'cyan',
    icon: '🔮',
    strengths: [
      '120 Epochs trained with Cosine Annealing on 2,801 real images (CS-10k & Roboflow)',
      'Multi-task prediction: 10 OSHA safety classes (F1: 0.846) + Continuous 2D Hazard Map',
      'Global 8-head self-attention rollout captures equipment blindspots and fall edges',
      'Fast single-frame GPU inference latency (~14.1 ms on RTX 3050 Laptop GPU)',
    ],
    weaknesses: [
      'Requires GPU acceleration for real-time 8-head attention matrix operations',
      'Quadratic O(N²) attention complexity relative to spatial token count (64 tokens)',
      'Needs bounding-box or segmentation supervision to ground spatial hazard heads',
    ],
    useCases: [
      'OSHA safety violation detection (NO-Hardhat, Trench, Fall protection)',
      'Real-time equipment perimeter and worker blindspot thermal alerting',
      'Self-attention interpretability for safety auditor review',
    ],
    rating: { quality: 92, stability: 94, generation: 88, speed: 85 },
  },
];

const RatingBar = ({ label, value, color }) => (
  <div className="flex items-center gap-3">
    <span className="text-xs text-slate-400 w-20">{label}</span>
    <div className="progress-bar flex-1">
      <motion.div
        className={`progress-fill bg-${color}-400`}
        initial={{ width: 0 }}
        animate={{ width: `${value}%` }}
        transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
        style={{ backgroundColor: color === 'primary' ? '#14b8a6' : color === 'secondary' ? '#8b5cf6' : '#34d399' }}
      />
    </div>
    <span className="text-xs font-bold text-slate-300 w-8">{value}%</span>
  </div>
);

const TrendIcon = ({ value }) => {
  if (value === 'best') return <ArrowUpRight className="w-4 h-4 text-emerald-400" />;
  if (value === 'worst') return <ArrowDownRight className="w-4 h-4 text-rose-400" />;
  return <Minus className="w-4 h-4 text-slate-500" />;
};

export default function ComparativeAnalysis() {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -20 }}
      className="space-y-8 max-w-7xl mx-auto"
    >
      {/* Header */}
      <header className="mb-2">
        <motion.div variants={itemVariants} className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-rose-500 flex items-center justify-center shadow-lg shadow-amber-500/20">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-4xl font-black tracking-tight gradient-text-warm">Comparative Analysis</h2>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 text-lg max-w-3xl">
          Comprehensive performance comparison across all three generative models — Autoencoder, VAE, and DCGAN — 
          evaluated using task-appropriate metrics including SSIM, PSNR, MSE, and FID scores.
        </motion.p>
      </header>

      {/* Performance Metrics Table */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><Target className="w-4 h-4" /></div>
          Quantitative Performance Metrics
        </div>
        <p className="text-sm text-slate-400 mb-5 -mt-3">
          ↑ Higher is better &nbsp;|&nbsp; ↓ Lower is better &nbsp;|&nbsp; * GAN does not reconstruct inputs, so reconstruction metrics are not applicable.
        </p>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                {metricsData.headers.map((h, i) => (
                  <th key={i}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metricsData.rows.map((row, i) => (
                <tr key={i}>
                  <td className="font-semibold text-slate-200">{row.metric}</td>
                  <td><span className="font-mono">{row.ae}</span></td>
                  <td><span className="font-mono">{row.vae}</span></td>
                  <td><span className="font-mono">{row.gan}</span></td>
                  <td><span className="font-mono text-cyan-400 font-semibold">{row.vit}</span></td>
                  <td><span className={`font-bold ${row.bestColor}`}>{row.best}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Model Cards Grid */}
      <motion.div variants={itemVariants}>
        <div className="section-title">
          <div className="section-icon"><Award className="w-4 h-4" /></div>
          Strengths, Weaknesses &amp; Use Cases
        </div>
        <div className="comparison-grid">
          {modelComparison.map((model, idx) => (
            <motion.div
              key={model.name}
              variants={itemVariants}
              className="comparison-card"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">{model.icon}</span>
                <h3 className="text-xl font-bold text-slate-100">{model.name}</h3>
              </div>

              {/* Rating Bars */}
              <div className="space-y-2 mb-4 p-4 rounded-xl bg-slate-900/50">
                <RatingBar label="Quality" value={model.rating.quality} color={model.color} />
                <RatingBar label="Stability" value={model.rating.stability} color={model.color} />
                <RatingBar label="Generation" value={model.rating.generation} color={model.color} />
                <RatingBar label="Speed" value={model.rating.speed} color={model.color} />
              </div>

              {/* Strengths */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-2 flex items-center gap-1.5">
                  <ArrowUpRight className="w-4 h-4" /> Strengths
                </h4>
                <ul className="space-y-1.5">
                  {model.strengths.map((s, i) => (
                    <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                      <span className="text-emerald-400 mt-1 text-xs">●</span> {s}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Weaknesses */}
              <div>
                <h4 className="text-sm font-semibold text-rose-400 mb-2 flex items-center gap-1.5">
                  <ArrowDownRight className="w-4 h-4" /> Weaknesses
                </h4>
                <ul className="space-y-1.5">
                  {model.weaknesses.map((w, i) => (
                    <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                      <span className="text-rose-400 mt-1 text-xs">●</span> {w}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Use Cases */}
              <div>
                <h4 className="text-sm font-semibold text-amber-400 mb-2 flex items-center gap-1.5">
                  <Zap className="w-4 h-4" /> Best Use Cases
                </h4>
                <ul className="space-y-1.5">
                  {model.useCases.map((u, i) => (
                    <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                      <span className="text-amber-400 mt-1 text-xs">●</span> {u}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Visual Quality Comparison */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="section-title">
          <div className="section-icon"><TrendingUp className="w-4 h-4" /></div>
          Visual Output Quality Comparison
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-primary text-center">Autoencoder Reconstruction</h4>
            <div className="image-frame h-64">
              <img src={`${API}/data/processed/features/ae_original_vs_reconstructed.png`} alt="AE Output" className="w-full h-full object-contain" 
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-secondary text-center">VAE Generative Samples</h4>
            <div className="image-frame h-64">
              <img src={`${API}/data/processed/features/vae_generative_samples.png`} alt="VAE Output" className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-emerald-400 text-center">GAN Synthetic Output</h4>
            <div className="image-frame h-64">
              <img src={`${API}/data/synthetic/gan/gan_final_generated.png`} alt="GAN Output" className="w-full h-full object-contain"
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }} />
              <div className="absolute inset-0 hidden items-center justify-center text-slate-500 text-sm">Run notebook to generate</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Key Insights / Conclusion */}
      <motion.div variants={itemVariants} className="glass-panel-accent p-6">
        <div className="section-title">
          <div className="section-icon"><Award className="w-4 h-4" /></div>
          Key Insights &amp; Conclusions
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4 text-sm text-slate-300 leading-relaxed">
            <p>
              <strong className="text-slate-100">1. Reconstruction vs Generation Trade-off:</strong> The Autoencoder achieves the highest 
              reconstruction fidelity (SSIM: 0.78, PSNR: 22.4dB) but lacks generative capability. The VAE sacrifices 
              some reconstruction quality for a structured latent space that enables novel sample generation.
            </p>
            <p>
              <strong className="text-slate-100">2. Adversarial Superiority in Generation:</strong> The DCGAN produces the most 
              photorealistic synthetic images (lowest FID: 98.3) by leveraging adversarial training, but at the cost 
              of training instability and inability to reconstruct inputs.
            </p>
          </div>
          <div className="space-y-4 text-sm text-slate-300 leading-relaxed">
            <p>
              <strong className="text-slate-100">3. Practical Recommendation:</strong> For construction site AI, a hybrid approach is 
              optimal — use AE for surveillance compression and anomaly detection, VAE for data augmentation with 
              controlled diversity, and GAN for generating large-scale photorealistic training data.
            </p>
            <p>
              <strong className="text-slate-100">4. Construction Domain Specifics:</strong> Construction imagery with complex scenes 
              (workers, machinery, structures) benefits from the GAN's ability to capture fine textures, while the 
              VAE's smooth interpolation is valuable for simulating gradual construction progress transitions.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
