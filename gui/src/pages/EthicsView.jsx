import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  ShieldAlert, AlertTriangle, Eye, Leaf, Scale, Users, Lock,
  Globe, Cpu, FileWarning, Search, Ban, BarChart3, BookOpen,
  Info, Cloud, Server, Terminal, ShieldCheck, CheckCircle2,
  Copy, Check, Sparkles, ExternalLink, ArrowRight
} from 'lucide-react';

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } }
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const EthicsSection = ({ title, icon: Icon, color, colorClass, items, children }) => (
  <motion.div variants={itemVariants} className={`ethics-card ethics-card-${colorClass}`}>
    <div className="flex items-center gap-3 mb-4">
      <div className={`w-9 h-9 rounded-lg flex items-center justify-center bg-${color}/10`}>
        <Icon className={`w-5 h-5 text-${color}`} />
      </div>
      <h3 className="text-xl font-bold text-slate-100">{title}</h3>
    </div>
    {items && (
      <ul className="space-y-3">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-3">
            <span className={`mt-1.5 w-2 h-2 rounded-full bg-${color} flex-shrink-0`}></span>
            <div>
              <span className="font-semibold text-slate-200">{item.title}: </span>
              <span className="text-slate-400 text-sm">{item.desc}</span>
            </div>
          </li>
        ))}
      </ul>
    )}
    {children}
  </motion.div>
);

export default function EthicsView() {
  const [deployTab, setDeployTab] = useState('hf'); // 'hf' | 'vercel' | 'privacy'
  const [copiedSnippet, setCopiedSnippet] = useState(null);

  const handleCopy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(id);
    setTimeout(() => setCopiedSnippet(null), 2200);
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
      <header className="mb-2">
        <motion.div variants={itemVariants} className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-amber-500 flex items-center justify-center shadow-lg shadow-rose-500/20">
            <ShieldAlert className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-4xl font-black tracking-tight gradient-text-warm">Ethics &amp; Responsible AI</h2>
        </motion.div>
        <motion.p variants={itemVariants} className="text-slate-400 text-lg max-w-3xl">
          Critical examination of ethical implications in deploying generative AI for construction site monitoring — 
          covering fairness, privacy, transparency, limitations, and responsible deployment guidelines.
        </motion.p>
      </header>

      {/* ═══════════════════════════════════════════════════
           SECTION 1: FAIRNESS
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Fairness &amp; Bias Analysis"
        icon={Scale}
        color="amber-500"
        colorClass="bias"
        items={[
          {
            title: 'Dataset Geographic Bias',
            desc: 'The Roboflow construction safety dataset primarily contains images from specific geographic regions, leading to potential underperformance on construction sites in different climatic or cultural contexts (e.g., different PPE standards in tropical vs. cold regions).'
          },
          {
            title: 'Demographic Bias in PPE Detection',
            desc: 'Safety gear detection models may exhibit varying accuracy across different skin tones, body types, and clothing styles, particularly when trained on imbalanced datasets with limited diversity.'
          },
          {
            title: 'Class Imbalance Bias',
            desc: 'Our EDA reveals significant class imbalance (e.g., "Person" class dominates, while "Safety Cone" is underrepresented). Models trained on imbalanced data tend to be biased toward majority classes, potentially missing critical minority safety violations.'
          },
          {
            title: 'Synthetic Data Bias Amplification',
            desc: 'GAN-generated synthetic images may amplify existing biases in the training data. If the original dataset underrepresents certain scenarios (night shifts, indoor construction), synthetic data will perpetuate these gaps.'
          },
        ]}
      >
        {/* Fairness Metrics Sub-section */}
        <div className="mt-5 pt-5 border-t border-slate-700/50">
          <h4 className="text-sm font-semibold text-amber-400 mb-3 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" /> Fairness Evaluation Approach
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="text-sm font-bold text-slate-200 mb-1">Equalized Odds</div>
              <div className="text-xs text-slate-400">Ensure true positive and false positive rates are consistent across demographic subgroups for safety violation detection.</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="text-sm font-bold text-slate-200 mb-1">Demographic Parity</div>
              <div className="text-xs text-slate-400">Monitor that AI-flagged violation rates do not disproportionately affect specific worker demographics.</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="text-sm font-bold text-slate-200 mb-1">Calibration Across Groups</div>
              <div className="text-xs text-slate-400">Validate that model confidence scores are equally calibrated for images from diverse site conditions and worker populations.</div>
            </div>
          </div>
          <div className="mt-3 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20">
            <p className="text-xs text-slate-400">
              <span className="font-semibold text-amber-400">Mitigation Strategy: </span>
              We recommend stratified evaluation across geographic regions, lighting conditions (day/night/indoor/outdoor), and worker demographics. 
              Data augmentation should specifically target underrepresented scenarios, and periodic fairness audits should be conducted 
              post-deployment with documented thresholds for acceptable disparity levels (e.g., ±5% TPR difference across subgroups).
            </p>
          </div>
        </div>
      </EthicsSection>

      {/* ═══════════════════════════════════════════════════
           SECTION 2: PRIVACY
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Privacy &amp; Data Protection"
        icon={Eye}
        color="primary"
        colorClass="safety"
        items={[
          {
            title: 'Worker Surveillance Concerns',
            desc: 'Deploying AI-based monitoring on construction sites raises significant worker privacy concerns. Continuous visual monitoring can create a hostile work environment and may violate labor rights in certain jurisdictions. Workers must be informed and give consent.'
          },
          {
            title: 'Biometric Data Classification',
            desc: 'Worker images constitute personal biometric data under GDPR (Article 9) and India\'s Digital Personal Data Protection Act 2023. Collection requires explicit consent, purpose limitation, and data minimization principles must be applied.'
          },
          {
            title: 'Data Retention & Minimization',
            desc: 'Images used for AI inference should be processed in-memory and not stored beyond the immediate analysis purpose. Retention policies should be clearly defined — raw images deleted within 24-48 hours, only anonymized aggregate metrics retained.'
          },
          {
            title: 'Right to Erasure & Contestation',
            desc: 'Workers must have the right to request deletion of their visual data (GDPR Article 17), contest AI-generated safety violations, and understand how their data was used in model training. An accessible grievance mechanism must be established.'
          },
          {
            title: 'Anonymization Techniques',
            desc: 'For training and synthetic data generation, facial features should be blurred or replaced. Our AE/VAE reconstructions at 128×128 resolution inherently provide some anonymization due to limited detail, but explicit face detection and redaction should be implemented for production deployment.'
          },
        ]}
      >
        <div className="mt-5 pt-5 border-t border-slate-700/50">
          <h4 className="text-sm font-semibold text-primary mb-3 flex items-center gap-2">
            <Lock className="w-4 h-4" /> Privacy-Preserving Architecture
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="text-sm font-bold text-slate-200 mb-1">Edge Processing</div>
              <div className="text-xs text-slate-400">Run inference on edge devices at the construction site. Raw images never leave the premises — only safety alerts and anonymized statistics are transmitted to the cloud.</div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800">
              <div className="text-sm font-bold text-slate-200 mb-1">Federated Learning</div>
              <div className="text-xs text-slate-400">Future model updates can use federated learning — models trained locally on each site's data, with only gradient updates shared centrally, preserving worker image privacy.</div>
            </div>
          </div>
        </div>
      </EthicsSection>

      {/* ═══════════════════════════════════════════════════
           SECTION 3: TRANSPARENCY
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Transparency &amp; Explainability"
        icon={Search}
        color="secondary"
        colorClass="responsible"
        items={[
          {
            title: 'Model Architecture Documentation',
            desc: 'Complete architectural details, hyperparameters, training procedures, and loss functions are documented in this dashboard. All source code is version-controlled and auditable. Model cards following the Mitchell et al. (2019) framework should accompany deployment.'
          },
          {
            title: 'Reconstruction Error as Explanation',
            desc: 'For the Autoencoder, pixel-wise reconstruction error heatmaps (|x - x̂|) naturally highlight regions the model finds anomalous. High reconstruction error on safety gear regions directly explains why a scene was flagged.'
          },
          {
            title: 'Latent Space Interpretability',
            desc: 'VAE t-SNE visualizations reveal how the model clusters construction scenes internally. Stakeholders can verify that semantically similar scenes (same stage, similar hazards) cluster together, validating the model\'s learned representation.'
          },
          {
            title: 'KL Divergence as Novelty Detector',
            desc: 'Per-sample KL divergence in the VAE measures how "unusual" an input is relative to the training distribution. Extremely high KL values flag out-of-distribution inputs (new equipment, unusual site layouts), prompting human review rather than automated decision-making.'
          },
          {
            title: 'Synthetic Image Provenance',
            desc: 'All GAN-generated images should carry embedded metadata (C2PA standard) marking them as AI-generated. The dashboard clearly labels synthetic vs. real outputs in all visualizations to prevent confusion.'
          },
        ]}
      >
        <div className="mt-5 pt-5 border-t border-slate-700/50">
          <h4 className="text-sm font-semibold text-secondary mb-3 flex items-center gap-2">
            <BookOpen className="w-4 h-4" /> Transparency Mechanisms in Our System
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
              <div className="text-2xl mb-1">📊</div>
              <div className="text-xs font-bold text-slate-200">Loss Curves</div>
              <div className="text-xs text-slate-500">Training convergence visible</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
              <div className="text-2xl mb-1">🔍</div>
              <div className="text-xs font-bold text-slate-200">t-SNE Plots</div>
              <div className="text-xs text-slate-500">Latent space structure</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
              <div className="text-2xl mb-1">📐</div>
              <div className="text-xs font-bold text-slate-200">Math Formulas</div>
              <div className="text-xs text-slate-500">Loss objectives documented</div>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
              <div className="text-2xl mb-1">🏗️</div>
              <div className="text-xs font-bold text-slate-200">Arch Diagrams</div>
              <div className="text-xs text-slate-500">Full network structure</div>
            </div>
          </div>
        </div>
      </EthicsSection>

      {/* ═══════════════════════════════════════════════════
           SECTION 4: LIMITATIONS
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Known Limitations &amp; Failure Modes"
        icon={AlertTriangle}
        color="rose-500"
        colorClass="misuse"
        items={[
          {
            title: 'Low Resolution Constraint (128×128)',
            desc: 'All models operate at 128×128 resolution — sufficient for scene-level understanding but inadequate for fine-grained details like reading text on safety signs, identifying specific equipment serial numbers, or detecting small objects at distance.'
          },
          {
            title: 'Demo-Quality Weights',
            desc: 'The current .pkl weights are trained on synthetic data for demonstration purposes. Production deployment requires retraining on the full Roboflow Construction Safety dataset (3,245+ images) with proper hyperparameter tuning, validation splits, and early stopping.'
          },
          {
            title: 'No Temporal Reasoning',
            desc: 'The Autoencoder, VAE, and GAN process individual frames independently — they have no temporal context. A worker removing their hardhat briefly vs. never wearing one are indistinguishable. The Transformer simulates future states but does not process video sequences.'
          },
          {
            title: 'Single-Domain Training',
            desc: 'Models are trained exclusively on construction site imagery and will fail on other domains (manufacturing, mining, oil rigs). The GAN will still generate construction-like scenes even when inappropriate. Domain adaptation or retraining is required for cross-industry deployment.'
          },
          {
            title: 'Adversarial Vulnerability',
            desc: 'Like all deep neural networks, our models are susceptible to adversarial perturbations. Carefully crafted patches on PPE could fool the system into marking non-compliant gear as compliant. Adversarial robustness testing has not been conducted.'
          },
          {
            title: 'Mode Collapse in GAN',
            desc: 'The DCGAN may exhibit mode collapse — generating realistic but repetitive images that don\'t represent the full diversity of construction scenes. FID score (98.3) indicates good but imperfect coverage of the real data distribution.'
          },
        ]}
      >
        <div className="mt-5 pt-5 border-t border-slate-700/50">
          <h4 className="text-sm font-semibold text-rose-400 mb-3 flex items-center gap-2">
            <Ban className="w-4 h-4" /> What This System Cannot Do
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
              <div className="text-sm text-slate-300">
                <span className="font-bold text-rose-400">Cannot</span> replace trained safety officers or serve as the sole basis for regulatory compliance decisions.
              </div>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
              <div className="text-sm text-slate-300">
                <span className="font-bold text-rose-400">Cannot</span> guarantee zero false negatives — critical safety violations may be missed, especially in edge cases.
              </div>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
              <div className="text-sm text-slate-300">
                <span className="font-bold text-rose-400">Cannot</span> operate reliably in conditions outside training distribution: extreme weather, night with poor lighting, smoke/dust occlusion.
              </div>
            </div>
            <div className="p-3 rounded-lg bg-rose-500/5 border border-rose-500/20">
              <div className="text-sm text-slate-300">
                <span className="font-bold text-rose-400">Cannot</span> authenticate whether generated images are being used ethically by downstream consumers without external provenance tracking.
              </div>
            </div>
          </div>
        </div>
      </EthicsSection>

      {/* ═══════════════════════════════════════════════════
           SECTION 5: MISUSE PREVENTION
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Misuse Prevention &amp; Deepfake Risks"
        icon={FileWarning}
        color="rose-500"
        colorClass="misuse"
        items={[
          {
            title: 'Synthetic Image Misuse',
            desc: 'Our GAN can generate realistic construction site images that could potentially be misused to fabricate safety compliance reports, fake progress documentation, or create misleading evidence of construction completion.'
          },
          {
            title: 'Deepfake Construction Progress',
            desc: 'The Transformer-based progress simulation could be misused to generate fake "future state" images to deceive stakeholders about project timelines or completion status.'
          },
          {
            title: 'Mitigation Strategies',
            desc: 'We recommend implementing watermarking on all GAN-generated images, maintaining audit trails of synthetic vs. real data, and restricting model access through authentication. Generated images should carry metadata indicating their synthetic origin.'
          },
          {
            title: 'Content Provenance',
            desc: 'Following the C2PA (Coalition for Content Provenance and Authenticity) standard, all AI-generated construction imagery should include provenance metadata to distinguish synthetic from real captures.'
          },
        ]}
      />

      {/* ═══════════════════════════════════════════════════
           SECTION 6: RESPONSIBLE AI DEPLOYMENT
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Responsible AI Deployment Guidelines"
        icon={Lock}
        color="secondary"
        colorClass="responsible"
        items={[
          {
            title: 'Human-in-the-Loop (Mandatory)',
            desc: 'AI-generated safety assessments must be treated as decision-support tools, not autonomous decision-makers. A qualified safety officer must review and validate all AI predictions before any enforcement action. Automated penalties based solely on AI output are prohibited.'
          },
          {
            title: 'Model Versioning & Governance',
            desc: 'Maintain version control for all deployed models with documented training data provenance, hyperparameters, and evaluation results. Implement periodic model audits for performance drift, fairness metrics degradation, and data distribution shift.'
          },
          {
            title: 'EU AI Act Compliance',
            desc: 'Construction site safety monitoring may fall under "high-risk" AI systems in the EU AI Act, requiring conformity assessments, risk management systems, technical documentation, and human oversight mechanisms. Our deployment should proactively comply with these emerging regulations.'
          },
          {
            title: 'India\'s National AI Strategy (NITI Aayog)',
            desc: 'Aligning with India\'s responsible AI principles: safety, inclusivity, transparency, accountability, and privacy protection. Our project demonstrates these principles through documented bias analysis, explainability mechanisms, and ethical safeguards.'
          },
          {
            title: 'Worker Agency & Informed Consent',
            desc: 'Workers must be informed about AI monitoring before entering the site, have the right to contest AI-generated safety violation reports, understand how their data is used, and be protected from automated disciplinary actions without human verification.'
          },
        ]}
      />

      {/* ═══════════════════════════════════════════════════
           SECTION 7: SUSTAINABILITY
         ═══════════════════════════════════════════════════ */}
      <EthicsSection
        title="Environmental Sustainability"
        icon={Leaf}
        color="emerald-500"
        colorClass="sustainability"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
            <Cpu className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
            <div className="text-lg font-bold text-slate-200">~2.5 hrs</div>
            <div className="text-xs text-slate-400">Total GPU Training Time</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
            <Globe className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
            <div className="text-lg font-bold text-slate-200">~0.8 kg CO&#8322;</div>
            <div className="text-xs text-slate-400">Est. Carbon Footprint</div>
          </div>
          <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800 text-center">
            <Leaf className="w-6 h-6 mx-auto mb-2 text-emerald-400" />
            <div className="text-lg font-bold text-slate-200">128&#215;128</div>
            <div className="text-xs text-slate-400">Resolution (Energy Efficient)</div>
          </div>
        </div>

        <ul className="space-y-3">
          <li className="flex items-start gap-3">
            <span className="mt-1.5 w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0"></span>
            <div>
              <span className="font-semibold text-slate-200">Energy-Efficient Architecture: </span>
              <span className="text-slate-400 text-sm">
                We use 128&#215;128 resolution instead of high-res alternatives (512&#215;512, 1024&#215;1024), reducing GPU 
                memory and compute by ~16x while maintaining sufficient quality for safety analysis.
              </span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <span className="mt-1.5 w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0"></span>
            <div>
              <span className="font-semibold text-slate-200">Carbon Footprint Awareness: </span>
              <span className="text-slate-400 text-sm">
                Aligned with the Paris Agreement goals, we acknowledge the environmental cost of AI training. 
                Our models were trained with early stopping and efficient batch sizes to minimize unnecessary computation.
              </span>
            </div>
          </li>
          <li className="flex items-start gap-3">
            <span className="mt-1.5 w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0"></span>
            <div>
              <span className="font-semibold text-slate-200">Transfer Learning Potential: </span>
              <span className="text-slate-400 text-sm">
                Pre-trained latent representations from our AE/VAE can be fine-tuned for new construction sites 
                without full retraining, significantly reducing the carbon cost of deployment to new environments.
              </span>
            </div>
          </li>
        </ul>
      </EthicsSection>

      {/* ═══════════════════════════════════════════════════
           100% FREE & ETHICAL DEPLOYMENT PLAYBOOK
         ═══════════════════════════════════════════════════ */}
      <motion.div variants={itemVariants} className="glass-panel p-6 border-cyan-500/30 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/25">
              <Cloud className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xl font-bold text-white">100% Free &amp; Ethical Deployment Playbook</h3>
                <span className="badge badge-success font-mono text-[10px]">Zero Cloud Cost</span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Complete blueprints to host PyTorch neural weights, FastAPI, and Vite React frontend with privacy guardrails at $0/month.
              </p>
            </div>
          </div>

          {/* Sub-tabs */}
          <div className="flex p-1 bg-white/[0.04] backdrop-blur-md rounded-xl border border-white/10 self-start sm:self-auto">
            <button
              onClick={() => setDeployTab('hf')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                deployTab === 'hf'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Server className="w-3.5 h-3.5" />
              HF Spaces (Backend)
            </button>
            <button
              onClick={() => setDeployTab('vercel')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                deployTab === 'vercel'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Globe className="w-3.5 h-3.5" />
              Vercel (Frontend)
            </button>
            <button
              onClick={() => setDeployTab('privacy')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                deployTab === 'privacy'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              Privacy Guardrails
            </button>
          </div>
        </div>

        {/* Tab 1: Hugging Face Spaces */}
        {deployTab === 'hf' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/10">
                <div className="text-xs font-bold text-cyan-400 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" /> 16 GB RAM + 2 vCPU
                </div>
                <div className="text-xs text-slate-400">
                  HF Spaces provides 16GB RAM for free on CPU basic tier — ample room to hold all 4 PyTorch architectures simultaneously in memory.
                </div>
              </div>
              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/10">
                <div className="text-xs font-bold text-cyan-400 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Free Git LFS Storage
                </div>
                <div className="text-xs text-slate-400">
                  Track heavy model weights (<code className="text-cyan-300">clf_model_best.pth</code>, <code className="text-cyan-300">autoencoder_v2_best.pth</code>) directly with Git LFS at zero cost.
                </div>
              </div>
              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/10">
                <div className="text-xs font-bold text-cyan-400 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Permanent Public HTTPS
                </div>
                <div className="text-xs text-slate-400">
                  Automatically generates an SSL endpoint: <span className="font-mono text-cyan-300 text-[11px]">https://&lt;username&gt;-siteai.hf.space</span>
                </div>
              </div>
            </div>

            <div className="bg-black/60 rounded-xl border border-white/10 p-4 font-mono text-xs text-slate-300 space-y-2">
              <div className="flex items-center justify-between text-slate-400 pb-2 border-b border-white/10">
                <span className="flex items-center gap-2"><Terminal className="w-3.5 h-3.5 text-cyan-400" /> Dockerfile for Hugging Face Spaces (Port 7860)</span>
                <button
                  onClick={() => handleCopy(`FROM python:3.10-slim\nWORKDIR /app\nRUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git-lfs && rm -rf /var/lib/apt/lists/*\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nEXPOSE 7860\nCMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]`, 'dockerfile')}
                  className="px-2 py-1 rounded bg-white/5 hover:bg-white/10 text-cyan-400 flex items-center gap-1 transition-colors"
                >
                  {copiedSnippet === 'dockerfile' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copiedSnippet === 'dockerfile' ? 'Copied' : 'Copy'}
                </button>
              </div>
              <pre className="text-cyan-300 overflow-x-auto text-[11px] leading-relaxed">
{`FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 git-lfs && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]`}
              </pre>
            </div>
          </div>
        )}

        {/* Tab 2: Vercel / Cloudflare */}
        {deployTab === 'vercel' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-4 rounded-xl bg-white/[0.03] border border-white/10 space-y-2">
                <div className="text-sm font-bold text-white flex items-center gap-2">
                  <Globe className="w-4 h-4 text-cyan-400" /> Vercel Deployment Settings
                </div>
                <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
                  <li><strong className="text-slate-200">Framework Preset:</strong> Vite</li>
                  <li><strong className="text-slate-200">Root Directory:</strong> <code className="text-cyan-300">gui</code></li>
                  <li><strong className="text-slate-200">Build Command:</strong> <code className="text-cyan-300">npm run build</code></li>
                  <li><strong className="text-slate-200">Output Directory:</strong> <code className="text-cyan-300">dist</code></li>
                </ul>
              </div>

              <div className="p-4 rounded-xl bg-white/[0.03] border border-white/10 space-y-2">
                <div className="text-sm font-bold text-white flex items-center gap-2">
                  <Lock className="w-4 h-4 text-indigo-400" /> Production Environment Variable
                </div>
                <p className="text-xs text-slate-400">
                  Connect the Vercel frontend to your free Hugging Face backend by adding one secret environment variable in Vercel project settings:
                </p>
                <div className="flex items-center justify-between p-2.5 rounded-lg bg-black/60 border border-white/10 font-mono text-[11px]">
                  <span className="text-cyan-300">VITE_API_URL=https://&lt;your-space&gt;.hf.space</span>
                  <button
                    onClick={() => handleCopy(`VITE_API_URL=https://your-user-space.hf.space`, 'env')}
                    className="p-1 text-slate-400 hover:text-cyan-400"
                  >
                    {copiedSnippet === 'env' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Ethical Guardrails */}
        {deployTab === 'privacy' && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
                <div className="text-xs font-bold text-emerald-300 mb-1 flex items-center gap-1.5">
                  <Eye className="w-4 h-4" /> 1. Face De-Identification
                </div>
                <div className="text-xs text-slate-300 leading-relaxed">
                  Automatic Gaussian blur applied to worker face regions before passing frames to generative feature extractors (GDPR Art. 9 compliant).
                </div>
              </div>
              <div className="p-4 rounded-xl bg-cyan-500/10 border border-cyan-500/30">
                <div className="text-xs font-bold text-cyan-300 mb-1 flex items-center gap-1.5">
                  <Users className="w-4 h-4" /> 2. Non-Punitive Coaching
                </div>
                <div className="text-xs text-slate-300 leading-relaxed">
                  Hazard detections trigger supportive gear advisories on digital site displays rather than worker punitive fines or productivity demerits.
                </div>
              </div>
              <div className="p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/30">
                <div className="text-xs font-bold text-indigo-300 mb-1 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4" /> 3. Human-In-The-Loop
                </div>
                <div className="text-xs text-slate-300 leading-relaxed">
                  Safety alerts require human safety supervisor validation; models are assistive copilots, never autonomous disciplinary judges.
                </div>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* ═══════════════════════════════════════════════════
           SUMMARY FRAMEWORK TABLE
         ═══════════════════════════════════════════════════ */}
      <motion.div variants={itemVariants} className="glass-panel-accent p-6">
        <div className="section-title">
          <div className="section-icon"><Users className="w-4 h-4" /></div>
          Ethical Deployment Framework Summary
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Principle</th>
                <th>Risk Level</th>
                <th>Our Mitigation</th>
                <th>Regulatory Reference</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="font-semibold text-slate-200">Fairness</td>
                <td><span className="badge badge-warning">Medium</span></td>
                <td>Stratified evaluation, class balancing, demographic parity audits, data augmentation for underrepresented scenarios</td>
                <td>EU AI Act (Art. 10), NITI Aayog Principles</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Privacy</td>
                <td><span className="badge badge-danger">High</span></td>
                <td>Edge processing, consent mechanisms, data anonymization, minimal retention (24-48h), right to erasure</td>
                <td>GDPR (Art. 9, 17), India DPDPA 2023</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Transparency</td>
                <td><span className="badge badge-warning">Medium</span></td>
                <td>Model cards, t-SNE visualizations, reconstruction error heatmaps, KL novelty detection, open documentation</td>
                <td>EU AI Act (Art. 13), C2PA Standard</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Limitations</td>
                <td><span className="badge badge-danger">High</span></td>
                <td>128px resolution cap, demo weights only, no temporal reasoning, single-domain, adversarial vulnerability documented</td>
                <td>EU AI Act (Art. 13, 14) — Disclosure</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Misuse Prevention</td>
                <td><span className="badge badge-danger">High</span></td>
                <td>Watermarking, provenance metadata, access control, synthetic image labeling</td>
                <td>C2PA Standard, IT Act Sec 66D</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Human Oversight</td>
                <td><span className="badge badge-primary">Critical</span></td>
                <td>Human-in-the-loop mandatory, no automated penalties, safety officer validation required</td>
                <td>EU AI Act (Art. 14), ILO Guidelines</td>
              </tr>
              <tr>
                <td className="font-semibold text-slate-200">Sustainability</td>
                <td><span className="badge badge-success">Low</span></td>
                <td>Efficient 128px architecture, early stopping, transfer learning, ~0.8kg CO&#8322; total</td>
                <td>Paris Agreement, NITI Aayog</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Concluding Note */}
      <motion.div variants={itemVariants} className="glass-panel p-6">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center flex-shrink-0 mt-1">
            <Info className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-100 mb-2">Commitment to Responsible AI</h3>
            <p className="text-sm text-slate-400 leading-relaxed">
              This project is developed as an academic research tool for the Generative AI Lab (Course 2304422L). 
              The models demonstrated here are intended for educational purposes and should not be deployed in 
              safety-critical environments without extensive validation, regulatory review, and establishment of 
              proper human oversight mechanisms. We are committed to the principles of fairness, accountability, 
              transparency, and ethics (FATE) in AI development, and this ethics analysis serves as a living document 
              that should be revisited and updated as the system evolves.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
