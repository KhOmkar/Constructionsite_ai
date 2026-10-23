import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Sparkles, CheckCircle2, AlertTriangle, Cpu, Layers, Box, ArrowRight, Zap, RefreshCw } from 'lucide-react';

const API = 'http://localhost:8000';

// Helper to convert base64 data URI to a standard File object
export function dataURLtoFile(dataurl, filename) {
  try {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  } catch (err) {
    console.error("Failed to convert dataURL to file:", err);
    return null;
  }
}

export default function SampleImageSelector({
  onSelectSample,
  selectedSampleId = null,
  filterModel = null, // 'ae', 'vae', 'transformer', 'gan', or null for all
  title = "Verified Construction Test Samples",
  subtitle = "Select any curated dataset sample image below to run instant inference without uploading local files."
}) {
  const [samples, setSamples] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeFilter, setActiveFilter] = useState(filterModel || 'all');

  useEffect(() => {
    let isMounted = true;
    const fetchSamples = async () => {
      try {
        setLoading(true);
        const res = await axios.get(`${API}/api/samples`);
        if (isMounted && res.data?.samples) {
          setSamples(res.data.samples);
        }
      } catch (err) {
        console.warn('Could not fetch sample presets:', err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchSamples();
    return () => { isMounted = false; };
  }, []);

  const filteredSamples = samples.filter(s => {
    if (activeFilter === 'all') return true;
    return s.recommended_models?.includes(activeFilter);
  });

  const handleSelect = (sample) => {
    if (!onSelectSample) return;
    const file = dataURLtoFile(sample.image, `${sample.id}.jpg`);
    onSelectSample(sample, file);
  };

  const getBadgeStyle = (color) => {
    switch (color) {
      case 'emerald':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
      case 'rose':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      case 'amber':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      case 'cyan':
        return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40';
      case 'indigo':
        return 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40';
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <div className="glass-panel p-5 space-y-4 border border-white/10 shadow-xl">
      {/* Header and Filter Tabs */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#00f0ff]" />
            <h3 className="font-extrabold text-white text-base tracking-tight">{title}</h3>
            <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
              {filteredSamples.length} Real Samples
            </span>
          </div>
          {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
        </div>

        {/* Model Filter Pills */}
        {!filterModel && (
          <div className="flex flex-wrap items-center gap-1.5 p-1 bg-slate-950/70 border border-white/10 rounded-xl text-xs font-semibold">
            {[
              { id: 'all', label: 'All Models' },
              { id: 'ae', label: 'Autoencoder / PPE' },
              { id: 'vae', label: 'VAE Latent' },
              { id: 'transformer', label: 'Transformer 4D' },
              { id: 'gan', label: 'GAN Benchmark' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveFilter(tab.id)}
                className={`px-3 py-1.5 rounded-lg text-xs transition-all ${
                  activeFilter === tab.id
                    ? 'bg-gradient-to-r from-cyan-400 to-sky-500 text-slate-950 font-black shadow-md shadow-cyan-500/20'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Loading Skeleton */}
      {loading && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {[1, 2, 3, 4, 5, 6, 7, 8].map(n => (
            <div key={n} className="h-36 rounded-xl bg-slate-800/50 animate-pulse border border-white/5" />
          ))}
        </div>
      )}

      {/* Samples Grid / Carousel */}
      {!loading && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          {filteredSamples.map(sample => {
            const isSelected = selectedSampleId === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                onClick={() => handleSelect(sample)}
                className={`group text-left p-2.5 rounded-xl border transition-all duration-300 flex flex-col relative overflow-hidden cursor-pointer ${
                  isSelected
                    ? 'bg-cyan-500/15 border-cyan-400 shadow-[0_0_20px_rgba(0,240,255,0.25)] ring-1 ring-cyan-400/50 scale-[1.02]'
                    : 'bg-slate-950/60 hover:bg-slate-900 border-white/10 hover:border-cyan-500/40 hover:shadow-lg'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-slate-900 mb-2 border border-white/10">
                  <img
                    src={sample.thumbnail || sample.image}
                    alt={sample.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                  {isSelected && (
                    <div className="absolute top-1 right-1 bg-cyan-400 text-slate-950 rounded-full p-0.5 shadow-lg">
                      <CheckCircle2 className="w-3.5 h-3.5 stroke-[3]" />
                    </div>
                  )}
                  <span className={`absolute bottom-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded border backdrop-blur-md shadow-sm ${getBadgeStyle(sample.badge_color)}`}>
                    {sample.badge}
                  </span>
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider truncate">
                    {sample.category}
                  </div>
                  <div className={`text-xs font-bold truncate transition-colors ${isSelected ? 'text-cyan-300' : 'text-slate-200 group-hover:text-cyan-400'}`}>
                    {sample.title}
                  </div>
                </div>

                {/* Quick Action Link */}
                <div className="mt-1 flex items-center justify-between text-[10px] font-bold text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span>Run Live</span>
                  <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
