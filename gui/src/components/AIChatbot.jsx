import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageSquare, X, Send, Sparkles, Bot, User, ShieldCheck,
  Rocket, AlertCircle, Copy, Check, ChevronDown, RefreshCw, Zap
} from 'lucide-react';
import axios from 'axios';

const API = 'http://localhost:8000';

const INITIAL_MESSAGES = [
  {
    id: 1,
    sender: 'bot',
    text: "👋 **Welcome to SiteAI Copilot!**\n\nI am your intelligent assistant for **Free Ethical Deployment**, Model Verification, and Worker Safety Monitoring.\n\nHow can I help you take this Generative AI framework to production?",
    time: 'Just now'
  }
];

const PROMPT_SUGGESTIONS = [
  { label: '🚀 Free Deployment Guide', query: 'How do I deploy this project for free?' },
  { label: '🛡️ Ethical AI & Privacy', query: 'What are the ethical privacy rules for site monitoring?' },
  { label: '🎯 Why test on crops?', query: 'Why does the safety classifier need cropped images?' },
  { label: '🔬 4 Generative Models', query: 'Explain the 4 generative AI models in this project' }
];

export default function AIChatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleCopy = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSend = async (messageText = inputMessage) => {
    const textToSend = messageText.trim();
    if (!textToSend || isTyping) return;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: textToSend,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsTyping(true);

    try {
      const res = await axios.post(`${API}/api/chat`, { message: textToSend });
      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: res.data?.reply || "I processed your request, but received an empty response.",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.warn("Backend chat endpoint unreachable, using intelligent local fallback:", err);
      // Fallback response generator if backend is momentarily offline
      let fallbackText = "### 🚀 100% Free Ethical Deployment Blueprint\n\n" +
        "You can deploy this full project for **$0/month** without paying for cloud servers:\n\n" +
        "1. **Frontend (Vercel / Cloudflare Pages):** Free unlimited hosting for your React/Vite dashboard.\n" +
        "2. **Backend & Models (Hugging Face Spaces):** Free 16GB RAM + 2 vCPU container for your PyTorch FastAPI backend with Git LFS for `.pth` model weights.\n" +
        "3. **Ethical Guardrails:** Automated face blurring for CCTV privacy & human-in-the-loop review for OSHA safety compliance.";

      const lower = textToSend.toLowerCase();
      if (lower.includes("crop") || lower.includes("acc") || lower.includes("fail") || lower.includes("guess")) {
        fallbackText = "### 🎯 Model Accuracy & Image Crops\n\n" +
          "- **Test Accuracy:** The classifier checkpoint (`clf_model_best.pth`) has **87.42% accuracy**.\n" +
          "- **Why wide scenes fail:** The model was trained on **36,698 bounding-box crops** (helmets, vests, workers). When an entire panoramic 1080p image is resized to 128x128, a helmet becomes only 4 pixels!\n" +
          "- **Tip:** Crop the target worker or equipment, or use the sample selector on the dashboard for instant 99% confidence.";
      } else if (lower.includes("ethic") || lower.includes("privacy") || lower.includes("worker")) {
        fallbackText = "### 🛡️ Ethical AI & Privacy Rules\n\n" +
          "1. **Face Anonymization:** Automatically obfuscate worker faces before feature extraction (GDPR Article 9).\n" +
          "2. **Non-Punitive Safety:** Deploy as a hazard warning system to prevent injuries, not a surveillance punitive tracker.\n" +
          "3. **Human-in-the-Loop:** Safety stewards must review low-confidence alerts before actions are taken.";
      }

      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: fallbackText,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, botMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  // Helper to format basic markdown-style text into neat HTML elements
  const renderFormattedText = (text) => {
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      if (line.startsWith('### ')) {
        return <h4 key={idx} className="text-sm font-black text-cyan-400 mt-2 mb-1">{line.replace('### ', '')}</h4>;
      }
      if (line.startsWith('#### ')) {
        return <h5 key={idx} className="text-xs font-bold text-slate-200 mt-2 mb-1">{line.replace('#### ', '')}</h5>;
      }
      if (line.startsWith('- ') || line.startsWith('  - ')) {
        return (
          <li key={idx} className="ml-3 text-xs text-slate-300 leading-relaxed list-disc">
            <span dangerouslySetInnerHTML={{ __html: formatInline(line.replace(/^(\s*-\s*)/, '')) }} />
          </li>
        );
      }
      if (/^\d+\.\s/.test(line)) {
        return (
          <div key={idx} className="text-xs text-slate-300 leading-relaxed mt-1">
            <span dangerouslySetInnerHTML={{ __html: formatInline(line) }} />
          </div>
        );
      }
      if (line.trim() === '') {
        return <div key={idx} className="h-1.5" />;
      }
      return (
        <p key={idx} className="text-xs text-slate-300 leading-relaxed">
          <span dangerouslySetInnerHTML={{ __html: formatInline(line) }} />
        </p>
      );
    });
  };

  const formatInline = (str) => {
    return str
      .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
      .replace(/`([^`]+)`/g, '<code class="bg-slate-900 px-1.5 py-0.5 rounded text-cyan-300 font-mono text-[11px] border border-cyan-500/20">$1</code>')
      .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-cyan-400 underline hover:text-cyan-300">$1</a>');
  };

  return (
    <>
      {/* Floating Toggle Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <AnimatePresence>
          {!isOpen && (
            <motion.button
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0, opacity: 0 }}
              onClick={() => setIsOpen(true)}
              className="relative group p-4 rounded-2xl bg-gradient-to-r from-cyan-400 via-indigo-500 to-purple-500 text-slate-950 font-bold shadow-2xl shadow-cyan-500/30 hover:shadow-cyan-400/50 hover:scale-105 active:scale-95 transition-all flex items-center gap-2.5 cursor-pointer border border-cyan-300/40"
            >
              <div className="relative">
                <Bot className="w-6 h-6 text-slate-950" />
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-slate-950 animate-pulse" />
              </div>
              <div className="text-left hidden sm:block">
                <div className="text-xs font-black tracking-tight leading-none text-slate-950">SiteAI Copilot</div>
                <div className="text-[10px] text-slate-900 font-medium">Free Deployment Advisor</div>
              </div>
            </motion.button>
          )}
        </AnimatePresence>
      </div>

      {/* Floating Chat Modal */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 30, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 30, scale: 0.95 }}
            transition={{ type: 'spring', damping: 25, stiffness: 350 }}
            className="fixed bottom-6 right-6 z-50 w-[420px] max-w-[calc(100vw-2rem)] h-[620px] max-h-[calc(100vh-4rem)] rounded-2xl bg-slate-950/95 border border-cyan-500/30 shadow-2xl shadow-cyan-950/80 backdrop-blur-2xl flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="p-4 bg-gradient-to-r from-slate-900 via-slate-900 to-slate-950 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-400 to-indigo-500 flex items-center justify-center text-slate-950 shadow-md shadow-cyan-500/20 font-bold">
                  <Sparkles className="w-5 h-5 text-slate-950" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-black text-white tracking-tight">SiteAI Copilot</h3>
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> Live
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400">Generative AI &amp; Ethical Deployment</p>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setMessages(INITIAL_MESSAGES)}
                  title="Reset conversation"
                  className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Quick Action Suggestion Chips */}
            <div className="p-3 bg-slate-900/60 border-b border-white/5 overflow-x-auto flex gap-1.5 no-scrollbar">
              {PROMPT_SUGGESTIONS.map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(item.query)}
                  className="flex-shrink-0 px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-cyan-500/15 border border-white/10 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300 text-[11px] font-medium transition-all"
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Chat Messages Body */}
            <div className="flex-1 p-4 overflow-y-auto space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex gap-2.5 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.sender === 'bot' && (
                    <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400 flex-shrink-0 mt-0.5">
                      <Bot className="w-4 h-4" />
                    </div>
                  )}

                  <div className={`relative group max-w-[85%] rounded-2xl p-3.5 ${
                    msg.sender === 'user'
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-medium rounded-tr-none shadow-md shadow-cyan-500/20'
                      : 'bg-slate-900/90 border border-white/10 text-slate-200 rounded-tl-none shadow-md'
                  }`}>
                    {msg.sender === 'bot' ? (
                      <div>
                        {renderFormattedText(msg.text)}
                        <button
                          onClick={() => handleCopy(msg.id, msg.text)}
                          className="mt-2.5 pt-2 border-t border-white/5 text-[10px] text-slate-400 hover:text-cyan-300 flex items-center gap-1 transition-colors"
                        >
                          {copiedId === msg.id ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-400" /> Copied blueprint
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" /> Copy guide
                            </>
                          )}
                        </button>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-950 font-semibold leading-relaxed">{msg.text}</p>
                    )}

                    <span className="text-[9px] opacity-50 block mt-1 text-right">
                      {msg.time}
                    </span>
                  </div>

                  {msg.sender === 'user' && (
                    <div className="w-7 h-7 rounded-lg bg-slate-800 border border-white/10 flex items-center justify-center text-slate-300 flex-shrink-0 mt-0.5">
                      <User className="w-4 h-4" />
                    </div>
                  )}
                </div>
              ))}

              {isTyping && (
                <div className="flex gap-2.5 items-center">
                  <div className="w-7 h-7 rounded-lg bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400 flex-shrink-0">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="p-3 rounded-2xl bg-slate-900 border border-white/10 rounded-tl-none flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Bar */}
            <div className="p-3 bg-slate-900/90 border-t border-white/10">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSend();
                }}
                className="flex items-center gap-2"
              >
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  placeholder="Ask about free deployment, models, or ethics..."
                  className="flex-1 bg-slate-950 border border-white/15 focus:border-cyan-400 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none transition-colors"
                />
                <button
                  type="submit"
                  disabled={!inputMessage.trim() || isTyping}
                  className="p-2.5 rounded-xl bg-gradient-to-r from-cyan-400 to-indigo-500 text-slate-950 font-bold hover:brightness-110 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
