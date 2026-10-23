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
  title = "Test With Pre-loaded Dataset Samples",
  subtitle = "Select any verified construction site sample image below to run instant inference without uploading local files."
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
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      case 'rose':
        return 'bg-rose-50 text-rose-700 border-rose-200';
      case 'amber':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'cyan':
        return 'bg-cyan-50 text-cyan-700 border-cyan-200';
      case 'indigo':
        return 'bg-indigo-50 text-indigo-700 border-indigo-200';
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header and Filter Tabs */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse" />
            <h3 className="font-bold text-slate-800 text-base">{title}</h3>
            <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
              {filteredSamples.length} Samples Ready
            </span>
          </div>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        </div>

        {/* Model Filter Pills */}
        {!filterModel && (
          <div className="flex items-center gap-1.5 p-1 bg-slate-100 rounded-xl text-xs font-semibold">
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
                className={`px-3 py-1.5 rounded-lg transition-all ${
                  activeFilter === tab.id
                    ? 'bg-white text-slate-800 shadow-sm font-bold'
                    : 'text-slate-500 hover:text-slate-800'
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
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[1, 2, 3, 4].map(n => (
            <div key={n} className="h-32 rounded-xl bg-slate-100 animate-pulse border border-slate-200" />
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
                className={`group text-left p-2 rounded-xl border transition-all duration-200 flex flex-col relative overflow-hidden ${
                  isSelected
                    ? 'bg-primary/5 border-primary shadow-md ring-2 ring-primary/20'
                    : 'bg-slate-50 hover:bg-white border-slate-200 hover:border-slate-300 hover:shadow-sm'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative aspect-square w-full rounded-lg overflow-hidden bg-slate-200 mb-2 border border-slate-100">
                  <img
                    src={sample.thumbnail || sample.image}
                    alt={sample.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  {isSelected && (
                    <div className="absolute top-1 right-1 bg-primary text-white rounded-full p-0.5 shadow">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </div>
                  )}
                  <span className={`absolute bottom-1 left-1 text-[9px] font-bold px-1.5 py-0.5 rounded border shadow-sm ${getBadgeStyle(sample.badge_color)}`}>
                    {sample.badge}
                  </span>
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider truncate">
                    {sample.category}
                  </div>
                  <div className="text-xs font-bold text-slate-800 truncate group-hover:text-primary transition-colors">
                    {sample.title}
                  </div>
                </div>

                {/* Quick Action Link */}
                <div className="mt-1 flex items-center justify-between text-[10px] font-bold text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                  <span>Load Sample</span>
                  <ArrowRight className="w-3 h-3" />
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
