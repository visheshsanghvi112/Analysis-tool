'use client';

import React, { useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  X,
  TrendingUp,
  LayoutGrid,
  Briefcase,
  Zap,
  Brain,
  Newspaper,
  Calculator,
  FileText,
  BookOpen,
  ExternalLink,
  Search,
  CheckCircle2,
  Activity,
  Layers,
  BarChart3,
  SlidersHorizontal,
  ChevronRight
} from 'lucide-react';

export default function SideNavDrawer({ isOpen, onClose, onOpenSearch }) {
  const pathname = usePathname();

  // Close on Escape key
  useEffect(() => {
    if (isOpen) {
      const handleKeyDown = (e) => {
        if (e.key === 'Escape') onClose();
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const NAV_LINKS = [
    {
      href: '/',
      label: 'Market Dashboard',
      description: 'Live terminal, charts & fundamentals',
      icon: TrendingUp,
      active: pathname === '/'
    },
    {
      href: '/intraday',
      label: 'Intraday Trading Desk',
      description: 'VWAP bands, Volume Profile, Camarilla & ORB',
      icon: Activity,
      active: pathname === '/intraday'
    },
    {
      href: '/browse',
      label: 'Browse & Screener',
      description: 'Explore 7,954 stocks, ETFs & sectors',
      icon: LayoutGrid,
      active: pathname === '/browse'
    },
    {
      href: '/portfolio',
      label: 'Portfolio Optimizer',
      description: 'Modern Portfolio Theory, GMV & Monte Carlo',
      icon: Briefcase,
      active: pathname === '/portfolio'
    },
    {
      href: '/features',
      label: 'Quantitative Models',
      description: 'Deep dive into mathematical engines',
      icon: Zap,
      active: pathname === '/features'
    }
  ];

  const QUICK_JUMP_TOOLS = [
    {
      label: '6-Model ML Predictions',
      desc: 'RF, XGBoost, LightGBM & HMM regimes',
      icon: Brain,
      color: 'text-emerald-400',
      action: () => {
        onClose();
        const el = document.getElementById('ml-prediction-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    },
    {
      label: 'Scrapling Live News Reader',
      desc: '100% real-time corporate catalysts',
      icon: Newspaper,
      color: 'text-cyan-400',
      action: () => {
        onClose();
        const el = document.getElementById('news-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    },
    {
      label: 'DCF & DuPont Valuation',
      desc: 'Intrinsic value & financial forensics',
      icon: BarChart3,
      color: 'text-amber-400',
      action: () => {
        onClose();
        const el = document.getElementById('valuation-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    },
    {
      label: 'Monte Carlo 99% VaR',
      desc: 'Stochastic risk simulation',
      icon: Activity,
      color: 'text-purple-400',
      action: () => {
        onClose();
        const el = document.getElementById('monte-carlo-section');
        if (el) el.scrollIntoView({ behavior: 'smooth' });
      }
    },
    {
      label: 'Export Executive Research Memo',
      desc: 'Print-ready institutional A4 PDF',
      icon: FileText,
      color: 'text-blue-400',
      action: () => {
        onClose();
        window.dispatchEvent(new CustomEvent('open-report-modal'));
      }
    }
  ];

  return (
    <div 
      className="fixed inset-0 z-[130] flex justify-end bg-black/80 backdrop-blur-md transition-opacity duration-200"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-sm sm:max-w-md bg-[#0b0c10] border-l border-white/10 h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-250 select-none overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Side Menu Header ─────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.08] bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-white flex items-center justify-center shadow-lg shadow-white/10">
              <TrendingUp className="w-5 h-5 text-black" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-white tracking-wide">StockIQ Pro</h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 font-bold border border-blue-500/30">
                  v2.4
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Institutional Workstation</p>
            </div>
          </div>

          <button 
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition-colors"
            title="Close menu (Esc)"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* ── Spotlight Search Quick Trigger ───────────────────────────── */}
        <div className="p-4 border-b border-white/[0.06]">
          <button
            onClick={() => {
              onClose();
              if (onOpenSearch) onOpenSearch();
            }}
            className="w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-[#14151c] border border-white/10 hover:border-blue-500/50 hover:bg-white/[0.04] transition-all text-slate-400 text-xs group"
          >
            <div className="flex items-center gap-2.5">
              <Search className="w-4 h-4 text-slate-400 group-hover:text-blue-400 transition-colors" />
              <span>Search 7,954+ instruments…</span>
            </div>
            <kbd className="text-[10px] px-2 py-0.5 rounded bg-black/50 border border-white/10 text-slate-500 font-mono">
              ⌘K
            </kbd>
          </button>
        </div>

        {/* ── Scrollable Body ──────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-5">
          {/* Main Navigation */}
          <div>
            <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Workstation Navigation
            </div>
            <div className="space-y-1">
              {NAV_LINKS.map((link) => {
                const Icon = link.icon;
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={onClose}
                    className={`flex items-start gap-3 px-3 py-2.5 rounded-xl transition-all ${
                      link.active
                        ? 'bg-blue-600/15 border border-blue-500/30 text-white'
                        : 'text-slate-300 hover:text-white hover:bg-white/[0.04] border border-transparent'
                    }`}
                  >
                    <div className={`p-1.5 rounded-lg mt-0.5 ${link.active ? 'bg-blue-500/20 text-blue-400' : 'bg-white/[0.05] text-slate-400'}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-semibold leading-tight">{link.label}</div>
                      <div className="text-[11px] text-slate-500 leading-normal mt-0.5">{link.description}</div>
                    </div>
                    {link.active && (
                      <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
                    )}
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Quick Quantitative Jump Tools */}
          <div>
            <div className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Analysis Engines &amp; Tools
            </div>
            <div className="space-y-1">
              {QUICK_JUMP_TOOLS.map((tool, idx) => {
                const ToolIcon = tool.icon;
                return (
                  <button
                    key={idx}
                    onClick={tool.action}
                    className="w-full text-left flex items-start gap-3 px-3 py-2 rounded-xl text-slate-300 hover:text-white hover:bg-white/[0.04] transition-all group"
                  >
                    <div className={`p-1.5 rounded-lg mt-0.5 bg-white/[0.05] ${tool.color} group-hover:scale-105 transition-transform`}>
                      <ToolIcon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-semibold text-slate-200 group-hover:text-white leading-tight">{tool.label}</div>
                      <div className="text-[10px] text-slate-500 leading-tight mt-0.5">{tool.desc}</div>
                    </div>
                    <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-all mt-1" />
                  </button>
                );
              })}
            </div>
          </div>

          {/* Institutional Blueprint Link */}
          <div className="pt-2 border-t border-white/[0.06]">
            <a
              href="https://github.com/visheshsanghvi112/Analysis-tool/blob/main/docs/architecture/ARCHITECTURE.md"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-between p-3 rounded-xl bg-gradient-to-r from-blue-950/30 to-indigo-950/30 border border-blue-500/20 text-xs text-blue-200 hover:border-blue-500/40 transition-all group"
            >
              <div className="flex items-center gap-2.5">
                <BookOpen className="w-4 h-4 text-blue-400" />
                <div>
                  <div className="font-semibold text-blue-300">System Architecture Guide</div>
                  <div className="text-[10px] text-blue-400/80">Mathematical &amp; engineering manual</div>
                </div>
              </div>
              <ExternalLink className="w-3.5 h-3.5 text-blue-400 group-hover:translate-x-0.5 transition-transform" />
            </a>
          </div>
        </div>

        {/* ── Side Menu Footer ─────────────────────────────────────────── */}
        <div className="p-4 border-t border-white/[0.08] bg-black/50 text-[11px] text-slate-500 space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-emerald-400 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>7,954 Assets Loaded</span>
            </div>
            <span className="text-[10px] text-slate-600 font-mono">Sub-0.5ms Cache</span>
          </div>
          <p className="text-[10px] text-slate-600">FastAPI 2.4 + Next.js 16 Turbopack</p>
        </div>
      </div>
    </div>
  );
}
