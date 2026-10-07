'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import Header from './components/Header';
import LivePrice from './components/LivePrice';
import StockChart from './components/StockChart';
import MLPrediction from './components/MLPrediction';
import AdvancedNews from './components/AdvancedNews';
import InvestmentCommitteeDesk from './components/InvestmentCommitteeDesk';

// Viewport-aware dynamic imports to optimize bundle size and prevent network avalanche
const PortfolioMetrics = dynamic(() => import('./components/PortfolioMetrics'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[200px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading Risk Analytics...</div>
});
const Backtesting = dynamic(() => import('./components/Backtesting'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[220px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading Strategy Backtest...</div>
});
const MonteCarloSimulation = dynamic(() => import('./components/MonteCarloSimulation'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[220px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading Monte Carlo Simulation...</div>
});
const ETFLongTermPanel = dynamic(() => import('./components/ETFLongTermPanel'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[220px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading ETF Analytics...</div>
});
const LongTermAnalysis = dynamic(() => import('./components/LongTermAnalysis'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[220px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading Long-Term Trends...</div>
});
const FundamentalsAnalysis = dynamic(() => import('./components/FundamentalsAnalysis'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[220px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading Fundamentals...</div>
});
const SIPCalculator = dynamic(() => import('./components/SIPCalculator'), {
  ssr: false,
  loading: () => <div className="glass-card p-6 min-h-[180px] flex items-center justify-center text-slate-500 text-xs font-mono">Loading SIP Simulator...</div>
});
const ResearchReportModal = dynamic(() => import('./components/ResearchReportModal'), {
  ssr: false
});
import PeerComparison from './components/PeerComparison';
import SectorIntelligence from './components/SectorIntelligence';
import {
  TrendingUp, Brain, Newspaper, PieChart,
  Activity, ArrowRight, CheckCircle, Clock, AlertTriangle,
  LayoutGrid, BarChart2, Trophy, FileText, Star, Scale,
} from 'lucide-react';
import InfoBadge from './components/InfoBadge';
import { API_BASE_URL } from './config';

/* ── Lazy Section Viewport Wrapper ───────────────────────────────── */
const LazySection = ({ children, placeholderHeight = 220 }) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: '450px' }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={ref}>
      {isVisible ? children : (
        <div style={{ minHeight: placeholderHeight }} className="glass-card border border-white/[0.05] flex items-center justify-center text-slate-600 text-xs font-mono animate-pulse">
          Loading module...
        </div>
      )}
    </div>
  );
};

/* ── Status badge ─────────────────────────────────────────────────── */
const StatusBadge = ({ icon: Icon, title, subtitle, status, infoKey, targetId }) => {
  const isActive  = status === 'active';
  const isLoading = status === 'loading';

  return (
    <div
      onClick={() => {
        if (targetId) {
          const el = document.getElementById(targetId);
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }
      }}
      className={`glass-card p-3 sm:p-4 rounded-xl border transition-all duration-200 select-none ${
        isActive
          ? 'bg-emerald-500/[0.04] border-emerald-500/20 hover:border-emerald-500/40 hover:shadow-[0_0_20px_rgba(16,185,129,0.12)]'
          : isLoading
          ? 'bg-amber-500/[0.04] border-amber-500/20 hover:border-amber-500/40'
          : 'bg-white/[0.02] border-white/[0.06] hover:border-white/15'
      } ${targetId ? 'cursor-pointer group' : ''}`}
      title={targetId ? `Click to jump to ${title}` : undefined}
    >
      <div className="flex items-start gap-2.5 sm:gap-3">
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border transition-transform duration-200 group-hover:scale-105 ${
            isActive
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
              : isLoading
              ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
              : 'bg-white/[0.04] border-white/[0.08] text-slate-400'
          }`}
        >
          <Icon className="w-4 h-4" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 mb-0.5">
            <p className="text-xs sm:text-sm font-bold text-white group-hover:text-emerald-300 transition-colors truncate">
              {title}
            </p>
            {infoKey && <InfoBadge infoKey={infoKey} />}
          </div>
          <p className="text-[11px] text-slate-400 truncate leading-snug">
            {subtitle}
          </p>
        </div>
        {isActive && (
          <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
        )}
        {isLoading && (
          <Clock className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5 animate-spin" />
        )}
        {!isActive && !isLoading && (
          <AlertTriangle className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-0.5" />
        )}
      </div>
    </div>
  );
};

/* ── Hero / Welcome ───────────────────────────────────────────────── */
const WelcomeScreen = () => (
  <section className="flex flex-col items-center py-16 sm:py-24 px-4 relative overflow-hidden text-center max-w-6xl mx-auto">
    {/* Ambient Background Glow behind heading - strictly bounded within viewport */}
    <div
      aria-hidden
      className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-80 bg-radial from-blue-600/15 via-indigo-600/05 to-transparent blur-3xl pointer-events-none -z-10"
    />

    {/* Live Pill */}
    <div className="inline-flex items-center gap-2.5 px-3.5 py-1.5 rounded-full border border-white/[0.12] bg-white/[0.04] backdrop-blur-md mb-6 shadow-sm">
      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]" />
      <span className="text-xs font-semibold text-slate-300 tracking-wide">
        Live NSE &amp; BSE · Institutional Multi-Desk Architecture
      </span>
    </div>

    {/* Heading */}
    <h1 className="text-4xl sm:text-6xl md:text-7xl font-black tracking-tight max-w-4xl leading-[1.08] mb-6 bg-gradient-to-b from-white via-white/95 to-slate-400 bg-clip-text text-transparent px-2">
      Give your portfolio the<br className="hidden sm:inline" /> analysis it deserves
    </h1>

    {/* Subtitle */}
    <p className="text-sm sm:text-base md:text-lg text-slate-400 max-w-2xl leading-relaxed mb-8 px-2 font-normal">
      6-model ML forecasts, 15-day recency-weighted sentiment, deterministic investment committee deliberation, and institutional CRO risk gates — in one unified terminal.
    </p>

    {/* Feature Highlight Pills */}
    <div className="flex flex-wrap items-center justify-center gap-2 mb-8 max-w-3xl">
      {[
        { label: '⚡ Live Intraday Terminal', color: 'border-emerald-500/30 text-emerald-300 bg-emerald-500/10' },
        { label: '🧠 6-Model AI Forecast', color: 'border-blue-500/30 text-blue-300 bg-blue-500/10' },
        { label: '🏛️ Dialectical Red-Teaming', color: 'border-indigo-500/30 text-indigo-300 bg-indigo-500/10' },
        { label: '🛡️ Chief Risk Officer Gate', color: 'border-amber-500/30 text-amber-300 bg-amber-500/10' },
        { label: '📰 15-Day News Intelligence', color: 'border-cyan-500/30 text-cyan-300 bg-cyan-500/10' },
      ].map((chip) => (
        <span
          key={chip.label}
          className={`text-[11px] font-semibold px-2.5 py-1 rounded-full border backdrop-blur-sm ${chip.color}`}
        >
          {chip.label}
        </span>
      ))}
    </div>

    {/* Action Buttons */}
    <div className="flex flex-wrap items-center justify-center gap-3.5 mb-14 w-full max-w-md">
      <button
        onClick={() => { window.dispatchEvent(new CustomEvent('trigger-search-focus')); }}
        className="flex-1 sm:flex-initial flex items-center justify-center gap-2.5 px-6 py-3.5 text-sm font-bold rounded-xl bg-white text-slate-950 hover:bg-slate-100 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_0_24px_rgba(255,255,255,0.2)] cursor-pointer"
      >
        <span>Search any stock</span>
        <kbd className="hidden sm:inline-block px-1.5 py-0.5 text-[10px] bg-slate-900 text-white rounded font-mono font-semibold">⌘K</kbd>
        <ArrowRight className="w-4 h-4" />
      </button>

      <Link
        href="/browse"
        className="flex-1 sm:flex-initial flex items-center justify-center gap-2 px-6 py-3.5 text-sm font-semibold rounded-xl bg-white/[0.05] hover:bg-white/[0.1] text-white border border-white/[0.12] hover:border-white/25 transition-all text-decoration-none cursor-pointer"
      >
        <LayoutGrid className="w-4 h-4 text-slate-400" />
        <span>Browse 7,900+ assets</span>
      </Link>
    </div>

    {/* Dashboard preview showcase */}
    <div className="w-full max-w-4xl mx-auto relative mb-16 rounded-2xl p-1 bg-gradient-to-b from-white/10 to-transparent border border-white/[0.1] shadow-2xl overflow-hidden group">
      <div
        aria-hidden
        className="absolute -top-12 left-1/2 -translate-x-1/2 w-3/4 h-32 bg-blue-500/20 blur-3xl pointer-events-none rounded-full"
      />
      <div className="relative rounded-xl overflow-hidden bg-slate-950">
        <img
          src="/dashboard-preview.png"
          alt="StockIQ Pro dashboard"
          className="w-full h-auto block rounded-xl transform transition-transform duration-500 group-hover:scale-[1.01]"
          loading="eager"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#06070a] via-transparent to-transparent pointer-events-none" />
      </div>
    </div>

    {/* Quick access by sector */}
    <div className="w-full max-w-4xl mx-auto">
      <div className="flex items-center gap-3 mb-4">
        <div className="flex-1 h-px bg-white/[0.08]" />
        <span className="text-[11px] text-slate-500 tracking-wider uppercase font-bold">
          Quick Access by Sector
        </span>
        <div className="flex-1 h-px bg-white/[0.08]" />
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2.5">
        {[
          { emoji: '🪙', label: 'ETFs',     color: '#eab308' },
          { emoji: '🏦', label: 'Banking',  color: '#00c48c' },
          { emoji: '💻', label: 'IT',        color: '#3b82f6' },
          { emoji: '⚡', label: 'Energy',    color: '#f59e0b' },
          { emoji: '💊', label: 'Pharma',    color: '#8b5cf6' },
          { emoji: '🚗', label: 'Auto',      color: '#ef4444' },
          { emoji: '🛒', label: 'FMCG',      color: '#10b981' },
          { emoji: '📈', label: 'Finance',   color: '#06b6d4' },
        ].map((s) => (
          <Link
            key={s.label}
            href={`/browse?sector=${encodeURIComponent(s.label)}`}
            className="flex flex-col items-center gap-2 p-3.5 rounded-xl bg-white/[0.02] hover:bg-white/[0.06] border border-white/[0.06] hover:border-white/20 transition-all duration-200 group text-decoration-none cursor-pointer"
          >
            <span className="text-2xl transition-transform duration-200 group-hover:scale-110">
              {s.emoji}
            </span>
            <span className="text-xs font-semibold text-slate-300 group-hover:text-white transition-colors">
              {s.label}
            </span>
          </Link>
        ))}
      </div>

      <div className="text-center mt-5">
        <Link
          href="/browse"
          className="text-xs font-semibold text-slate-400 hover:text-white inline-flex items-center gap-1.5 transition-colors text-decoration-none"
        >
          <span>View all 7,900+ stocks, ETFs &amp; indices across NSE &amp; BSE</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>
    </div>
  </section>
);

/* ── Peer & Sector Intelligence Tabs ─────────────────────────── */
const PEER_TABS = [
  { id: 'compare',  label: 'Peer-to-Peer',       icon: BarChart2, desc: 'Compare vs a specific stock' },
  { id: 'sector',   label: 'Sector Intelligence', icon: Trophy,    desc: 'Rank all sector peers' },
];

const PeerSectorTabs = ({ ticker }) => {
  const [activeTab, setActiveTab] = useState('compare');
  const [comparePeer, setComparePeer] = useState(null);

  const handleSelectPeer = (peer) => {
    setComparePeer(peer);
    setActiveTab('compare');
  };

  return (
    <div>
      {/* Tab row */}
      <div className="flex gap-1.5 mb-3 p-1.5 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md">
        {PEER_TABS.map(tab => {
          const Icon = tab.icon;
          const isA = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 sm:px-4 rounded-lg text-xs font-bold cursor-pointer transition-all ${
                isA
                  ? 'bg-white/[0.08] text-white border border-white/[0.15] shadow-[0_0_15px_rgba(59,130,246,0.15)]'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.02] border border-transparent'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isA ? (tab.id === 'compare' ? 'text-blue-400' : 'text-amber-400') : 'text-slate-500'}`} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
      {/* Tab panels */}
      {activeTab === 'compare' && <PeerComparison ticker={ticker} initialPeer={comparePeer} />}
      {activeTab === 'sector'  && <SectorIntelligence ticker={ticker} onSelectPeer={handleSelectPeer} />}
    </div>
  );
};

/* ── Loading ──────────────────────────────────────────────────────── */
const LoadingState = ({ ticker }) => (
  <div className="flex flex-col items-center justify-center py-28 px-4 gap-4">
    <div className="relative">
      <div className="w-12 h-12 rounded-full border-2 border-indigo-500/20 border-t-indigo-400 animate-spin" />
      <div className="w-2.5 h-2.5 rounded-full bg-indigo-400 absolute inset-0 m-auto animate-pulse" />
    </div>
    <div className="text-center">
      <p className="text-sm font-semibold text-white">
        Loading <span className="font-mono text-indigo-400">{ticker.replace('.NS', '').replace('.BO', '')}</span>
      </p>
      <p className="text-xs text-slate-500 mt-1">
        Calibrating multi-desk quantitative models &amp; live market feeds...
      </p>
    </div>
  </div>
);

// Comprehensive ETF heuristic for Indian & global ETFs
const checkIsETF = (ticker) => {
  if (!ticker) return false;
  const t = ticker.toUpperCase().trim();
  const ETF_KEYWORDS = [
    'BEES', 'ETF', 'MON100', 'MONQ50', 'MAFANG', 'CPSE', 'GOLD', 'SILVER',
    'LIQUIDBEES', 'SILVERBEES', 'ITBEES', 'SETFNN50', 'KOTAKNV20',
    'MID150BEES', 'JUNIORBEES', 'HDFCNIFTY', 'ICICINIFTY', 'NIFTYETF',
    'Q50', 'NV20', 'MID150', 'BHARAT22', 'ICICIB22', 'SETF', 'NIFTYQLITY',
    'LOWVOL', 'ALPHA', 'MOMENTUM', 'COMMODITY', 'INVESCO', 'AXISNIFTY',
    'UTINIFT', 'KOTAKNIFTY', 'SBINIFTY', 'MASPTOP50', 'MIDSML400', 'HDFCSML250',
  ];
  const GLOBAL_ETFS = new Set([
    'SPY', 'QQQ', 'VOO', 'IVV', 'VTI', 'IWM', 'DIA', 'GLD', 'SLV',
    'ARKK', 'SMH', 'XLF', 'XLE', 'XLK', 'EEM', 'VEA', 'VWO', 'SCHD',
    'TLT', 'IEF', 'SHY', 'XBI', 'IBB', 'VNQ', 'VIG', 'VYM',
  ]);
  return (
    ETF_KEYWORDS.some(k => t.includes(k)) ||
    t.endsWith('ETF.NS') ||
    t.endsWith('ETF.BO') ||
    GLOBAL_ETFS.has(t)
  );
};

/* ── Dashboard ────────────────────────────────────────────────────── */
export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [companyName, setCompanyName]       = useState('');
  const [isLoading, setIsLoading]           = useState(false);
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [confirmedETF, setConfirmedETF]     = useState(null);

  const isETF = confirmedETF !== null ? confirmedETF : checkIsETF(selectedTicker);

  // Confirm ETF status from heuristic or LivePrice onDataLoaded
  useEffect(() => {
    if (!selectedTicker) {
      setConfirmedETF(null);
      return;
    }
    if (checkIsETF(selectedTicker)) {
      setConfirmedETF(true);
    }
  }, [selectedTicker]);

  const handleTickerSelect = useCallback((ticker) => {
    setIsLoading(true);
    setConfirmedETF(null);
    setSelectedTicker(ticker);
    window.scrollTo({ top: 0 });
    setTimeout(() => setIsLoading(false), 400);
  }, []);

  // Read ?ticker= param on mount (set by /browse page)
  const urlHandledRef = useRef(false);
  useEffect(() => {
    if (urlHandledRef.current) return;
    urlHandledRef.current = true;
    try {
      const params = new URLSearchParams(window.location.search);
      const t = params.get('ticker');
      if (t) {
        window.history.replaceState(window.history.state, '', '/');
        handleTickerSelect(t);
      }
    } catch (_) {}
  }, [handleTickerSelect]);

  // Listen to logo click event to go back to homepage welcome screen
  useEffect(() => {
    const handleReset = () => {
      setSelectedTicker('');
    };
    const handleOpenReport = () => {
      setReportModalOpen(true);
    };
    window.addEventListener('reset-selected-ticker', handleReset);
    window.addEventListener('open-report-modal', handleOpenReport);
    return () => {
      window.removeEventListener('reset-selected-ticker', handleReset);
      window.removeEventListener('open-report-modal', handleOpenReport);
    };
  }, []);

  // Track if current asset is pinned in Watchlist
  const [isWatchlisted, setIsWatchlisted] = useState(false);

  useEffect(() => {
    if (!selectedTicker) {
      setIsWatchlisted(false);
      return;
    }
    const checkWatchlist = () => {
      try {
        const saved = localStorage.getItem('stockiq_pro_watchlist');
        if (saved) {
          const list = JSON.parse(saved);
          if (Array.isArray(list)) {
            const cleanCur = selectedTicker.replace('.NS', '').replace('.BO', '').toUpperCase();
            const found = list.some(item => {
              const itemSym = (item.symbol || '').replace('.NS', '').replace('.BO', '').toUpperCase();
              return itemSym === cleanCur;
            });
            setIsWatchlisted(found);
            return;
          }
        }
      } catch (_) {}
      setIsWatchlisted(false);
    };

    checkWatchlist();
    window.addEventListener('stockiq-watchlist-changed', checkWatchlist);
    return () => window.removeEventListener('stockiq-watchlist-changed', checkWatchlist);
  }, [selectedTicker]);

  const handleToggleWatchlist = () => {
    if (!selectedTicker) return;
    try {
      const saved = localStorage.getItem('stockiq_pro_watchlist');
      let list = saved ? JSON.parse(saved) : [];
      if (!Array.isArray(list)) list = [];

      const cleanCur = selectedTicker.replace('.NS', '').replace('.BO', '').toUpperCase();
      const existingIdx = list.findIndex(item => {
        const itemSym = (item.symbol || '').replace('.NS', '').replace('.BO', '').toUpperCase();
        return itemSym === cleanCur;
      });

      if (existingIdx >= 0) {
        list.splice(existingIdx, 1);
        setIsWatchlisted(false);
      } else {
        list.push({
          symbol: selectedTicker.toUpperCase(),
          name: cleanCur,
          sector: 'Equities'
        });
        setIsWatchlisted(true);
      }
      localStorage.setItem('stockiq_pro_watchlist', JSON.stringify(list));
      window.dispatchEvent(new CustomEvent('stockiq-watchlist-changed'));
    } catch (e) {
      console.error('Error toggling watchlist:', e);
    }
  };

  const isIndianNSE = selectedTicker?.endsWith('.NS');
  const isIndianBSE = selectedTicker?.endsWith('.BO');
  const marketExchange = isIndianNSE ? 'NSE' : isIndianBSE ? 'BSE' : 'GLOBAL';

  return (
    <div className="min-h-screen flex flex-col bg-transparent text-white">
      <Header onTickerSelect={handleTickerSelect} currentTicker={selectedTicker} />

      <main className="flex-1 w-full max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 pb-16">
        {!selectedTicker ? (
          <WelcomeScreen />
        ) : isLoading ? (
          <LoadingState ticker={selectedTicker} />
        ) : (
          <div className="pt-4 sm:pt-6">
            {/* Status row with click-to-scroll */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 mb-4">
              <StatusBadge icon={Activity}  title="Live Prices"       subtitle="Real-time · 15 min delay" status="active" infoKey="live_prices" targetId="live-price-section" />
              <StatusBadge icon={Brain}     title="ML Predictions"    subtitle={isETF ? '30-Day ETF Outlook' : '5-Day Ensemble'} status="active" infoKey="ml_predictions" targetId="ml-prediction-section" />
              <StatusBadge icon={Newspaper} title="News Intelligence" subtitle="15-Day AI Sentiment & Directives" status="active" infoKey="news_intelligence" targetId="news-section" />
              <StatusBadge icon={PieChart}  title="Risk Analytics"    subtitle="VaR · Options · Portfolio" status="active" infoKey="risk_analytics" targetId="risk-analytics-section" />
            </div>

            {/* Quick Action & Active Asset Command Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5 p-3 sm:px-4 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-xl shadow-lg">
              <div className="flex items-center gap-2.5 flex-wrap min-w-0">
                <span className="text-[11px] uppercase tracking-wider text-slate-400 font-bold">Asset</span>
                <div className="flex items-center gap-1.5 bg-white/[0.06] border border-white/[0.12] px-2.5 py-1 rounded-lg">
                  <span className="text-[10px] font-black uppercase px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    {marketExchange}
                  </span>
                  <span className="text-sm font-bold font-mono text-white tracking-tight">
                    {selectedTicker.replace('.NS', '').replace('.BO', '')}
                  </span>
                </div>
                {companyName && (
                  <span className="text-xs text-slate-300 font-medium truncate max-w-[200px] sm:max-w-[280px]">
                    {companyName}
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2 flex-wrap">
                <button
                  onClick={handleToggleWatchlist}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all border ${
                    isWatchlisted
                      ? 'bg-amber-500/15 border-amber-500/40 text-amber-300 shadow-[0_0_12px_rgba(245,158,11,0.15)]'
                      : 'bg-white/[0.04] border-white/[0.08] text-slate-300 hover:text-white hover:bg-white/[0.08]'
                  }`}
                  title={isWatchlisted ? "Remove from personal watchlist" : "Pin to personal watchlist"}
                >
                  <Star className={`w-3.5 h-3.5 ${isWatchlisted ? 'fill-amber-400 text-amber-400' : 'text-slate-400'}`} />
                  <span>{isWatchlisted ? 'Pinned' : 'Watchlist'}</span>
                </button>
                <Link
                  href={`/intraday?ticker=${encodeURIComponent(selectedTicker)}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20 text-decoration-none shadow-[0_0_12px_rgba(16,185,129,0.1)]"
                  title="Open in High-Frequency Intraday Desk"
                >
                  <Activity className="w-3.5 h-3.5 text-emerald-400" />
                  <span>⚡ Intraday</span>
                </Link>
                <button
                  onClick={() => {
                    const el = document.getElementById('investment-committee-section');
                    if (el) el.scrollIntoView({ behavior: 'smooth' });
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-500/20 shadow-[0_0_12px_rgba(99,102,241,0.1)]"
                  title="Jump to Multi-Desk Investment Committee"
                >
                  <Scale className="w-3.5 h-3.5 text-indigo-400" />
                  <span>🏛️ Committee</span>
                </button>
                <button
                  onClick={() => setReportModalOpen(true)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all bg-blue-500/10 border border-blue-500/30 text-blue-300 hover:bg-blue-500/20 shadow-[0_0_12px_rgba(59,130,246,0.1)]"
                  title="Export Institutional Research Memo"
                >
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  <span>Export Memo</span>
                </button>
              </div>
            </div>

            {/* Main Application Workstation */}
            <div className="flex flex-col gap-6">
              {/* Level 1: Executive Price Ticker & Ranges (Full Width) */}
              <div id="live-price-section" className="w-full">
                <LivePrice 
                  ticker={selectedTicker} 
                  onDataLoaded={(quote) => {
                    if (quote?.name || quote?.longName || quote?.shortName) {
                      setCompanyName(quote.name || quote.longName || quote.shortName);
                    }
                    if (quote?.asset_type) {
                      setConfirmedETF(quote.asset_type === 'ETF' || quote.asset_type === 'MUTUALFUND');
                    } else if (quote?.longName && /\b(etf|bees)\b/i.test(quote.longName)) {
                      setConfirmedETF(true);
                    }
                  }}
                />
              </div>

              {/* Level 2: Interactive Technical Workstation (Full Width) */}
              <div id="stock-chart-section" className="w-full">
                <StockChart ticker={selectedTicker} />
              </div>

              {/* Level 3: Dual AI Intelligence Command Center (Side-by-side 50/50 Balanced Grid) */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start w-full">
                <div id="ml-prediction-section" className="min-w-0">
                  <MLPrediction ticker={selectedTicker} />
                </div>
                <div id="news-section" className="min-w-0">
                  <AdvancedNews ticker={selectedTicker} companyName={companyName} />
                </div>
              </div>

              {/* Level 4: Institutional Strategy & Governance Desk */}
              <div id="investment-committee-section" className="w-full">
                <InvestmentCommitteeDesk ticker={selectedTicker} />
              </div>
              
              {/* Level 5: Risk & Simulation Desks */}
              <div id="risk-analytics-section" className="w-full">
                <LazySection placeholderHeight={240}>
                  <PortfolioMetrics ticker={selectedTicker} />
                </LazySection>
              </div>

              <div id="backtesting-section" className="w-full">
                <LazySection placeholderHeight={280}>
                  <Backtesting ticker={selectedTicker} />
                </LazySection>
              </div>

              <div id="monte-carlo-section" className="w-full">
                <LazySection placeholderHeight={300}>
                  <MonteCarloSimulation ticker={selectedTicker} />
                </LazySection>
              </div>

              {/* Level 6: Long-Term: ETF deep-dive OR stock analysis */}
              {isETF ? (
                <div id="etf-analytics-section" className="w-full">
                  <LazySection placeholderHeight={350}>
                    <ETFLongTermPanel ticker={selectedTicker} />
                  </LazySection>
                </div>
              ) : (
                <>
                  <div id="long-term-trends-section" className="w-full">
                    <LazySection placeholderHeight={300}>
                      <LongTermAnalysis ticker={selectedTicker} />
                    </LazySection>
                  </div>
                  <div id="valuation-section" className="w-full">
                    <LazySection placeholderHeight={300}>
                      <FundamentalsAnalysis ticker={selectedTicker} />
                    </LazySection>
                  </div>
                </>
              )}

              {/* Level 7: Wealth Accumulation Simulator */}
              <div id="sip-simulator-section" className="w-full">
                <LazySection placeholderHeight={200}>
                  <SIPCalculator ticker={selectedTicker} />
                </LazySection>
              </div>

              {/* Level 8: Peer & Sector Intelligence Tabs */}
              {!isETF && (
                <div id="peer-sector-section" className="w-full">
                  <LazySection placeholderHeight={300}>
                    <PeerSectorTabs ticker={selectedTicker} />
                  </LazySection>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer style={{ borderTop: '1px solid rgba(255,255,255,0.08)', padding: '24px 16px', background: 'rgba(6,7,10,0.85)', backdropFilter: 'blur(16px)' }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '26px', height: '26px', background: '#fff', borderRadius: '5px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <TrendingUp style={{ width: '14px', height: '14px', color: '#000' }} />
            </div>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#fff' }}>StockIQ Pro</span>
            <span style={{ fontSize: '12px', color: '#444' }}>
              by{' '}
              <a 
                href="https://visheshsanghvi.qzz.io/" 
                target="_blank" 
                rel="noopener noreferrer" 
                style={{ color: '#aaa', textDecoration: 'underline', transition: 'color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.color = '#fff'}
                onMouseLeave={e => e.currentTarget.style.color = '#aaa'}
              >
                Vishesh Sanghvi
              </a>
            </span>
          </div>
          <p style={{ fontSize: '12px', color: '#666' }}>
            Data via Yahoo Finance (~15 min delay). Not financial advice.{' '}
            <Link href="/terms" style={{ color: '#aaa', textDecoration: 'underline', transition: 'color 0.15s' }}
              onMouseEnter={e => e.currentTarget.style.color = '#fff'}
              onMouseLeave={e => e.currentTarget.style.color = '#aaa'}
            >
              Terms &amp; Conditions
            </Link>
          </p>
        </div>
      </footer>

      {/* Research Memo Export Modal */}
      <ResearchReportModal
        isOpen={reportModalOpen}
        onClose={() => setReportModalOpen(false)}
        ticker={selectedTicker}
      />
    </div>
  );
}
