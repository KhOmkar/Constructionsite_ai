import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Image as ImageIcon, Sparkles, Layers, Box,
  BarChart3, ShieldAlert, Zap, Rocket, Cpu, HelpCircle
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

import Dashboard from './pages/Dashboard';
import AutoencoderView from './pages/AutoencoderView';
import VaeView from './pages/VaeView';
import GanView from './pages/GanView';
import TransformerView from './pages/TransformerView';
import DiffusionView from './pages/DiffusionView';
import ComparativeAnalysis from './pages/ComparativeAnalysis';
import EthicsView from './pages/EthicsView';
import AIChatbot from './components/AIChatbot';

function Sidebar() {
  const location = useLocation();

  const navGroups = [
    {
      label: 'Mission Control',
      items: [
        { path: '/', label: 'Overview Dashboard', icon: LayoutDashboard },
      ],
    },
    {
      label: 'Generative AI Suite',
      items: [
        { path: '/autoencoder', label: 'Autoencoder & PPE', icon: Box, badge: '87.4%' },
        { path: '/vae', label: 'Variational AE', icon: Sparkles, badge: 'CVAE' },
        { path: '/gan', label: 'Adversarial GAN Suite', icon: Zap, badge: '3 Models' },
        { path: '/diffusion', label: 'Diffusion Model', icon: Sparkles, badge: 'DDIM 40x' },
        { path: '/transformer', label: 'Vision Transformer (ViT)', icon: Layers, badge: '8-Head' },
      ],
    },
    {
      label: 'Audit & Deployment',
      items: [
        { path: '/compare', label: 'Comparative Benchmark', icon: BarChart3 },
        { path: '/ethics', label: 'Ethics & Free Deploy', icon: ShieldAlert, badge: 'FREE' },
      ],
    },
  ];

  return (
    <div className="w-64 h-screen border-r border-white/10 bg-slate-950/90 backdrop-blur-2xl flex flex-col shrink-0 shadow-2xl relative z-20">
      {/* Brand Header */}
      <div className="flex items-center gap-3 px-5 pt-6 pb-5 border-b border-white/5">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 via-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <Layers className="text-slate-950 w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <h1 className="text-base font-black tracking-tight text-white leading-tight">
              Construc<span className="text-cyan-400">.AI</span>
            </h1>
            <span className="text-[9px] px-1.5 py-0.2 rounded font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
              v2.0
            </span>
          </div>
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold mt-0.5">
            Digital Twin &bull; GenAI
          </p>
        </div>
      </div>

      {/* Real-time System Status Pill */}
      <div className="px-4 py-3 bg-slate-900/40 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] font-semibold text-slate-300">Neural Engine</span>
        </div>
        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
          API Active
        </span>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-5">
        {navGroups.map((group, gi) => (
          <div key={gi}>
            <div className="text-[10px] uppercase tracking-widest text-slate-500 font-bold px-3 mb-2">
              {group.label}
            </div>
            <div className="flex flex-col gap-1">
              {group.items.map((item) => {
                const isActive = location.pathname === item.path;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group ${
                      isActive
                        ? 'bg-gradient-to-r from-cyan-500/15 to-indigo-500/10 text-cyan-300 font-semibold border border-cyan-500/30 shadow-[0_0_15px_rgba(0,240,255,0.1)]'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                    }`}
                  >
                    {isActive && (
                      <motion.div
                        layoutId="active-indicator"
                        className="absolute left-0 w-1 h-6 bg-cyan-400 rounded-r-full shadow-[0_0_10px_#00f0ff]"
                        initial={false}
                        transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                      />
                    )}
                    <Icon className={`w-4 h-4 transition-transform group-hover:scale-110 ${isActive ? 'text-cyan-400' : 'text-slate-400 group-hover:text-slate-200'}`} />
                    <span className="text-xs font-medium">{item.label}</span>
                    {item.badge && (
                      <span className={`ml-auto text-[9px] px-1.5 py-0.5 rounded font-mono font-bold ${
                        item.badge === 'FREE'
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                          : isActive
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                          : 'bg-white/5 text-slate-400'
                      }`}>
                        {item.badge}
                      </span>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Deployment Footer Box */}
      <div className="p-3 border-t border-white/5 bg-slate-900/30">
        <div className="p-3 rounded-xl bg-gradient-to-br from-indigo-950/60 to-purple-950/40 border border-indigo-500/20 text-center">
          <div className="flex items-center justify-center gap-1.5 text-xs font-bold text-indigo-300 mb-1">
            <Rocket className="w-3.5 h-3.5 text-cyan-400" /> Free Deployment
          </div>
          <p className="text-[10px] text-slate-400 leading-tight">
            Deploy free to Vercel &amp; Hugging Face Spaces.
          </p>
          <Link
            to="/ethics"
            className="mt-2.5 inline-block text-[10px] font-bold text-cyan-400 hover:underline"
          >
            View Free Guide &rarr;
          </Link>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="flex w-full min-h-screen bg-[#080c14] text-slate-100 font-sans antialiased overflow-hidden selection:bg-cyan-500/30 selection:text-cyan-200">
        <Sidebar />
        <main className="flex-1 h-screen overflow-y-auto overflow-x-hidden relative">
          {/* Ambient Cyber Grid & Glow */}
          <div className="absolute inset-0 bg-grid bg-radial-glow pointer-events-none" />

          {/* Top Header Live Ticker */}
          <div className="sticky top-0 z-30 px-8 py-2.5 bg-slate-950/70 backdrop-blur-md border-b border-white/5 flex items-center justify-between text-[11px] text-slate-400 font-mono">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5 text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Cluster: Localhost:8000
              </span>
              <span className="hidden sm:inline text-slate-600">|</span>
              <span className="hidden sm:inline text-slate-400">
                Models: AutoencoderV2 &bull; VAE &bull; DCGAN &bull; ViT
              </span>
            </div>
            <div className="flex items-center gap-3">
              <span className="px-2 py-0.5 rounded bg-white/5 text-slate-300 border border-white/10">
                Precision: FP32
              </span>
              <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-bold">
                Deployable Suite
              </span>
            </div>
          </div>

          {/* Page Viewport */}
          <div className="relative z-10 p-8 min-h-[calc(100%-41px)]">
            <AnimatePresence mode="wait">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/autoencoder" element={<AutoencoderView />} />
                <Route path="/vae" element={<VaeView />} />
                <Route path="/gan" element={<GanView />} />
                <Route path="/diffusion" element={<DiffusionView />} />
                <Route path="/transformer" element={<TransformerView />} />
                <Route path="/compare" element={<ComparativeAnalysis />} />
                <Route path="/ethics" element={<EthicsView />} />
              </Routes>
            </AnimatePresence>
          </div>
        </main>

        {/* Global Floating AI Copilot & Deployment Chatbot */}
        <AIChatbot />
      </div>
    </Router>
  );
}

export default App;
