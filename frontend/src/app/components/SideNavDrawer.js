'use client';

import React, { useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  X,
  TrendingUp,
  Activity,
  LayoutGrid,
  Briefcase,
  SlidersHorizontal,
  Search,
  FileText,
  Calculator,
  Shield,
  ChevronRight,
} from 'lucide-react';

export default function SideNavDrawer({ isOpen, onClose, onOpenSearch, currentTicker }) {
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

  // Lock body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const tickerParam = currentTicker ? `?ticker=${encodeURIComponent(currentTicker)}` : '';

  const NAV_LINKS = [
    {
      href: `/${tickerParam}`,
      label: 'Market Dashboard',
      description: 'Overview, live charts & fundamental analysis',
      icon: TrendingUp,
      active: pathname === '/',
      badge: null,
    },
    {
      href: `/intraday${tickerParam}`,
      label: 'Intraday Trading Desk',
      description: 'VWAP bands, volume profile & microstructure',
      icon: Activity,
      active: pathname === '/intraday',
      badge: 'Live Desk',
    },
    {
      href: '/browse',
      label: 'Browse & Screener',
      description: 'Screener across 7,950+ stocks, ETFs & sectors',
      icon: LayoutGrid,
      active: pathname === '/browse',
      badge: null,
    },
    {
      href: '/portfolio',
      label: 'Portfolio Tracker',
      description: 'Holdings, asset allocation & risk analysis',
      icon: Briefcase,
      active: pathname === '/portfolio',
      badge: null,
    },
    {
      href: '/features',
      label: 'Platform Models & Docs',
      description: 'Quantitative engines, formulas & user guide',
      icon: SlidersHorizontal,
      active: pathname === '/features',
      badge: null,
    },
  ];

  const handleSipClick = () => {
    onClose();
    if (pathname === '/') {
      const el = document.getElementById('sip-calculator') || document.querySelector('#sip-calculator-section');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
        return;
      }
    }
    window.location.assign(`/${tickerParam}#sip-calculator`);
  };

  const handleReportClick = () => {
    onClose();
    window.dispatchEvent(new CustomEvent('open-report-modal'));
  };

  const handleSearchClick = () => {
    onClose();
    if (onOpenSearch) onOpenSearch();
  };

  return (
    <div
      className="fixed inset-0 z-[130] flex justify-end bg-black/60 backdrop-blur-sm transition-opacity duration-200"
      onClick={onClose}
      aria-modal="true"
      role="dialog"
    >
      <div
        className="w-full max-w-sm bg-[#0c0d14] border-l border-white/[0.08] h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-200 select-none overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Header ───────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.08] bg-white/[0.01]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <TrendingUp className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-black text-white tracking-tight">StockIQ Pro</span>
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  PRO
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Market Intelligence &amp; Trading Desk</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition cursor-pointer"
            title="Close menu (Esc)"
            aria-label="Close menu"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* ── Active Ticker Pill (if available) ────────────────── */}
        {currentTicker && (
          <div className="px-5 py-2.5 bg-white/[0.02] border-b border-white/[0.06] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[11px] text-slate-400">Active Asset:</span>
              <span className="text-xs font-mono font-bold text-white">
                {currentTicker.replace('.NS', '').replace('.BO', '')}
              </span>
            </div>
            <Link
              href={`/intraday${tickerParam}`}
              onClick={onClose}
              className="text-[11px] text-emerald-400 hover:underline font-medium flex items-center gap-0.5"
            >
              Intraday Desk <ChevronRight className="w-3 h-3" />
            </Link>
          </div>
        )}

        {/* ── Navigation Links ─────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6">
          <div>
            <p className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Navigation
            </p>
            <nav className="space-y-1">
              {NAV_LINKS.map((link) => {
                const Icon = link.icon;
                return (
                  <Link
                    key={link.label}
                    href={link.href}
                    onClick={onClose}
                    className={`flex items-start gap-3 px-3 py-2.5 rounded-xl transition cursor-pointer ${
                      link.active
                        ? 'bg-white/[0.08] border border-white/[0.12] text-white shadow-sm'
                        : 'text-slate-300 hover:text-white hover:bg-white/[0.04] border border-transparent'
                    }`}
                  >
                    <div
                      className={`p-1.5 rounded-lg mt-0.5 ${
                        link.active
                          ? 'bg-indigo-500/20 text-indigo-300'
                          : 'bg-white/[0.04] text-slate-400'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold leading-tight">{link.label}</span>
                        {link.badge && (
                          <span className="text-[9px] font-bold px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 uppercase tracking-wider">
                            {link.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-500 leading-normal mt-0.5">
                        {link.description}
                      </p>
                    </div>
                    {link.active && (
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 mt-2 shrink-0" />
                    )}
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* ── Quick Tools ──────────────────────────────────────── */}
          <div>
            <p className="px-3 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
              Quick Tools
            </p>
            <div className="space-y-1">
              {/* Spotlight search */}
              <button
                onClick={handleSearchClick}
                className="w-full text-left flex items-center justify-between px-3 py-2.5 rounded-xl text-slate-300 hover:text-white hover:bg-white/[0.04] transition cursor-pointer group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-1.5 rounded-lg bg-white/[0.04] text-slate-400 group-hover:text-white">
                    <Search className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                      Search Instruments
                    </div>
                    <div className="text-[10px] text-slate-500">NSE, BSE, US Stocks &amp; ETFs</div>
                  </div>
                </div>
                <kbd className="text-[10px] px-2 py-0.5 rounded bg-white/[0.05] border border-white/10 text-slate-400 font-mono">
                  ⌘K
                </kbd>
              </button>

              {/* SIP Calculator Jump */}
              <button
                onClick={handleSipClick}
                className="w-full text-left flex items-center justify-between px-3 py-2.5 rounded-xl text-slate-300 hover:text-white hover:bg-white/[0.04] transition cursor-pointer group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-1.5 rounded-lg bg-white/[0.04] text-amber-400/80 group-hover:text-amber-300">
                    <Calculator className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                      SIP &amp; Wealth Planner
                    </div>
                    <div className="text-[10px] text-slate-500">Step-up compounding &amp; goal targets</div>
                  </div>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-transform" />
              </button>

              {/* Research Memo */}
              <button
                onClick={handleReportClick}
                className="w-full text-left flex items-center justify-between px-3 py-2.5 rounded-xl text-slate-300 hover:text-white hover:bg-white/[0.04] transition cursor-pointer group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-1.5 rounded-lg bg-white/[0.04] text-cyan-400/80 group-hover:text-cyan-300">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                      Export Research Memo
                    </div>
                    <div className="text-[10px] text-slate-500">Generate printable PDF analysis</div>
                  </div>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-transform" />
              </button>

              {/* Terms & Legal */}
              <Link
                href="/terms"
                onClick={onClose}
                className="w-full text-left flex items-center justify-between px-3 py-2.5 rounded-xl text-slate-300 hover:text-white hover:bg-white/[0.04] transition cursor-pointer group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-1.5 rounded-lg bg-white/[0.04] text-slate-400 group-hover:text-white">
                    <Shield className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                      Terms &amp; Disclaimers
                    </div>
                    <div className="text-[10px] text-slate-500">Risk disclosure &amp; policies</div>
                  </div>
                </div>
                <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-transform" />
              </Link>
            </div>
          </div>
        </div>

        {/* ── Footer ───────────────────────────────────────────── */}
        <div className="p-4 border-t border-white/[0.08] bg-black/40 text-[11px] text-slate-500 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-slate-400 font-medium">NSE · BSE · Global</span>
          </div>
          <span className="text-[10px] text-slate-600 font-mono">StockIQ Pro</span>
        </div>
      </div>
    </div>
  );
}
