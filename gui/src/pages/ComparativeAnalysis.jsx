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

/* ═══ Metric data for all models ═══ */
const metricsData = {
  headers: ['Metric', 'Autoencoder', 'VAE', 'DCGAN', 'WGAN-GP / cGAN', 'Diffusion (DDIM)', 'Vision Transformer', 'Best'],
  rows: [
    { metric: 'SSIM ↑', ae: '0.88', vae: '0.76', dcgan: 'N/A*', wgan: 'N/A*', diffusion: '0.70', vit: '0.82', best: 'AE', bestColor: 'text-primary' },
    { metric: 'PSNR (dB) ↑', ae: '24.2', vae: '21.5', dcgan: 'N/A*', wgan: 'N/A*', diffusion: '12.8', vit: '23.1', best: 'AE', bestColor: 'text-primary' },
    { metric: 'MSE ↓', ae: '0.0051', vae: '0.0112', dcgan: 'N/A*', wgan: 'N/A*', diffusion: '0.0522', vit: '0.0078', best: 'AE', bestColor: 'text-primary' },
    { metric: 'FID Score ↓', ae: 'N/A', vae: '142.5', dcgan: '98.3', wgan: '82.1', diffusion: '138.9', vit: 'N/A', best: 'WGAN-GP', bestColor: 'text-cyan-400' },
    { metric: 'Inception Score ↑', ae: 'N/A', vae: '2.85', dcgan: '3.42', wgan: '3.88', diffusion: '1.39', vit: 'N/A', best: 'WGAN-GP', bestColor: 'text-cyan-400' },
    { metric: 'Classification Acc', ae: '87.4%', vae: 'N/A', dcgan: 'N/A', wgan: 'N/A', diffusion: 'N/A', vit: '91.8%', best: 'ViT', bestColor: 'text-purple-400' },
    { metric: 'Latent Dim / Prior', ae: '256', vae: '256', dcgan: '100', wgan: '128', diffusion: '128×128 Noise', vit: '65 Tokens', best: '—', bestColor: 'text-slate-400' },
    { metric: 'Training Stability', ae: 'High', vae: 'High', dcgan: 'Moderate', wgan: 'Optimal (Lipschitz)', diffusion: 'Optimal (MSE Closed Form)', vit: 'High', best: 'Diffusion/AE', bestColor: 'text-teal-400' },
    { metric: 'Closed-Loop ViT Audit', ae: 'Supported', vae: 'Supported', dcgan: 'Audited', wgan: 'Audited', diffusion: 'Audited (DDIM)', vit: 'Evaluator', best: 'ViT+Diffusion/GAN', bestColor: 'text-indigo-400' },
  ]
};

const modelComparison = [
  {
    name: 'Autoencoder & Classifier',
    color: 'primary',
    icon: '🔧',
    strengths: [
      'Best reconstruction fidelity (highest SSIM & PSNR)',
      'Stable and fast convergence during training',
      '87.42% accuracy for 10-class OSHA PPE feature classification',
      'Learns dense spatial representations with multi-scale skips',
    ],
    weaknesses: [
      'Cannot generate novel samples — only reconstructs inputs',
      'Latent space is unstructured (no smooth interpolation)',
      'Prone to memorization with small datasets',
    ],
    useCases: [
      'Image compression for site surveillance',
      'Anomaly detection in safety gear',
      'Supervised PPE compliance inspection',
    ],
    rating: { quality: 92, stability: 95, generation: 15, speed: 92 },
  },
  {
    name: 'Variational AE (VAE/CVAE)',
    color: 'secondary',
    icon: '✨',
    strengths: [
      'Can both reconstruct AND generate novel images',
      'Structured latent space enables smooth interpolation',
      'Principled probabilistic framework (ELBO objective)',
      'Calibrated N(0, I) manifold with 100-epoch training',
    ],
    weaknesses: [
      'Reconstructions are blurrier than AE (KL regularization trade-off)',
      'Requires temperature scaling to avoid average gray collapse',
      'Requires careful balance for KL vs reconstruction MSE',
    ],
    useCases: [
      'Synthetic data augmentation for safety model training',
      'Latent space exploration and interpolation',
      'Probabilistic anomaly detection via log-likelihood',
    ],
    rating: { quality: 78, stability: 88, generation: 80, speed: 85 },
  },
  {
    name: 'Adversarial GAN Suite',
    color: 'emerald',
    icon: '⚡',
    strengths: [
      '3 Specialized Architectures: DCGAN, cGAN, and WGAN-GP',
      'cGAN conditions on 4 targeted OSHA safety categories',
      'WGAN-GP eliminates mode collapse via Earth Mover distance (FID 82.1)',
      'Geodesic SLERP walk & closed-loop Vision Transformer validation',
    ],
    weaknesses: [
      'DCGAN has oscillation risk during training',
      'Cannot reconstruct input images (pure generative synthesis)',
      'Requires gradient penalty (λ=10) and instance norm tuning',
    ],
    useCases: [
      'Targeted hazard generation (excavators, PPE violations)',
      'Smooth continuous latent walks for civil simulation',
      'Synthetic augmentation of rare dangerous site scenarios',
    ],
    rating: { quality: 94, stability: 88, generation: 96, speed: 90 },
  },
  {
    name: 'Denoising Diffusion Model (DDPM/DDIM)',
    color: 'primary',
    icon: '🌌',
    strengths: [
      '25-Step non-Markovian DDIM acceleration (38.5x speedup vs DDPM)',
      'Time & Class-Conditioned U-Net with Bottleneck Spatial Self-Attention',
      'Multi-scale temporal error tracking (Early, Mid, Late-t decomposition)',
      'Eliminates adversarial min-max oscillation entirely via convex MSE',
    ],
    weaknesses: [
      'Multi-step iterative inference (191.1 ms vs single-step GAN 6 ms)',
      'Higher compute requirements during reverse denoising trajectory',
      'Sensitive to cosine vs linear beta variance schedules',
    ],
    useCases: [
      'Controlled synthesis of hazardous site conditions',
      'Augmenting civil engineering datasets for edge detection',
      'Closed-loop training with downstream Vision Transformers',
    ],
    rating: { quality: 91, stability: 96, generation: 94, speed: 76 },
  },
  {
    name: 'Vision Transformer (ViT)',
    color: 'secondary',
    icon: '👁️',
    strengths: [
      '8-Head Multi-Head Self-Attention rollout explainability (Plasma)',
      '4-Rule OSHA Compliance Audit (1926.501, 1926.100, 1926.651, 1926.600)',
      'Struck-by machine proximity & worker analytics',
      'Natural language hazard reasoning for industrial site stewards',
    ],
    weaknesses: [
      'Higher computational footprint for self-attention matrices',
      'Requires high-resolution patch embeddings for small PPE tokens',
      'Needs bounding-box detector pairing for sub-pixel localization',
    ],
    useCases: [
      'Automated site safety compliance audit',
      'Closed-loop audit of synthetic GAN generated scenes',
      'Visual attention heatmaps and safety risk scoring',
    ],
    rating: { quality: 96, stability: 92, generation: 70, speed: 84 },
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
                  <td><span className="font-mono">{row.dcgan}</span></td>
                  <td><span className="font-mono text-cyan-300 font-semibold">{row.wgan}</span></td>
                  <td><span className="font-mono text-teal-300 font-semibold">{row.diffusion}</span></td>
                  <td><span className="font-mono text-purple-300 font-semibold">{row.vit}</span></td>
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
