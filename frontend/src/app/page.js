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
  const color = status === 'active' ? '#00c48c' : status === 'loading' ? '#f5a623' : '#444';
  const bg    = status === 'active' ? 'rgba(0,196,140,0.06)' : 'transparent';
  return (
    <div
      onClick={() => {
        if (targetId) {
          const el = document.getElementById(targetId);
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }
      }}
      className={`v-card ${targetId ? 'cursor-pointer group hover:border-emerald-500/40 transition-all duration-200' : ''}`}
      style={{ padding: '14px 16px', background: bg, borderColor: status === 'active' ? 'rgba(0,196,140,0.2)' : undefined }}
      title={targetId ? `Click to jump to ${title}` : undefined}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
        <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: color + '18', border: `1px solid ${color}33`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Icon style={{ width: '16px', height: '16px', color }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
            <p className="group-hover:text-emerald-300 transition-colors" style={{ fontSize: '13px', fontWeight: 600, color: '#fff' }}>{title}</p>
            {infoKey && <InfoBadge infoKey={infoKey} />}
          </div>
          <p style={{ fontSize: '11px', color: '#94a3b8', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{subtitle}</p>
        </div>
        {status === 'active'   && <CheckCircle   style={{ width: '14px', height: '14px', color: '#00c48c', flexShrink: 0 }} />}
        {status === 'loading'  && <Clock         style={{ width: '14px', height: '14px', color: '#f5a623', flexShrink: 0, animation: 'spin 1s linear infinite' }} />}
        {status === 'inactive' && <AlertTriangle style={{ width: '14px', height: '14px', color: '#444', flexShrink: 0 }} />}
      </div>
    </div>
  );
};

/* ── Hero / Welcome ───────────────────────────────────────────────── */
const WelcomeScreen = () => (
  <section style={{
    fontFamily: 'var(--font-poppins), var(--font-inter), sans-serif',
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    padding: '96px 20px 96px', // Increased padding for grander layout
    animation: 'heroFadeIn 0.5s ease both',
    position: 'relative',
    overflow: 'hidden',
  }}>
    <style>{`
      @keyframes heroFadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
      @keyframes livePulse  { 0%,100% { opacity:1; } 50% { opacity:.4; } }
      @keyframes spin        { to { transform:rotate(360deg); } }
      .hero-cta:hover  { transform:scale(1.04); }
      .hero-cta:active { transform:scale(0.96); }
      .browse-card:hover { border-color:#333 !important; background:#0a0a0a !important; }
    `}</style>

    {/* Ambient Background Glow behind heading */}
    <div aria-hidden style={{
      position: 'absolute',
      top: '0%',
      left: '50%',
      transform: 'translateX(-50%)',
      width: '100%',
      height: '350px',
      background: 'radial-gradient(circle at 50% 30%, rgba(59,130,246,0.06) 0%, rgba(139,92,246,0.02) 50%, transparent 100%)',
      filter: 'blur(80px)',
      pointerEvents: 'none',
      zIndex: 0
    }} />

    {/* Pill */}
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '5px 16px', borderRadius: '999px', border: '1px solid #282828', background: 'rgba(255,255,255,0.03)', marginBottom: '32px', position: 'relative', zIndex: 1 }}>
      <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#00c48c', animation: 'livePulse 2s ease infinite', flexShrink: 0 }} />
      <span style={{ fontSize: '12px', color: '#888' }}>Live NSE &amp; BSE · Powered by ML</span>
    </div>

    {/* Heading */}
    <h1 style={{
      fontSize: 'clamp(38px, 7vw, 68px)', // Increased font size
      fontWeight: 700, textAlign: 'center',
      maxWidth: '850px', lineHeight: 1.1, letterSpacing: '-0.04em', marginBottom: '24px',
      background: 'linear-gradient(to bottom, #ffffff 0%, #ffffff 40%, rgba(255,255,255,0.3) 100%)',
      WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
      padding: '0 8px',
      position: 'relative',
      zIndex: 1
    }}>
      Give your portfolio the<br />analysis it deserves
    </h1>

    {/* Sub */}
    <p style={{ fontSize: 'clamp(14px, 2vw, 17px)', color: '#888888', textAlign: 'center', maxWidth: '520px', lineHeight: 1.7, marginBottom: '40px', padding: '0 8px', position: 'relative', zIndex: 1 }}>
      ML predictions, real-time sentiment &amp; institutional risk analytics for NSE and BSE — in one dashboard.
    </p>

    {/* CTAs */}
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center', marginBottom: '80px', position: 'relative', zIndex: 1 }}>
      <button
        className="hero-cta"
        onClick={() => { window.dispatchEvent(new CustomEvent('trigger-search-focus')); }}
        style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '14px 30px', fontSize: '14px', fontWeight: 600, borderRadius: '8px', border: 'none', cursor: 'pointer', background: '#ffffff', color: '#000000', transition: 'transform 0.2s ease', letterSpacing: '-0.01em' }}
      >
        Search a stock <ArrowRight style={{ width: '15px', height: '15px' }} />
      </button>
      <Link
        href="/browse"
        style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '14px 30px', fontSize: '14px', fontWeight: 500, borderRadius: '8px', border: '1px solid #282828', background: 'transparent', color: '#cccccc', textDecoration: 'none', transition: 'border-color 0.15s, color 0.15s', letterSpacing: '-0.01em' }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = '#444444'; e.currentTarget.style.color = '#ffffff'; }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = '#282828'; e.currentTarget.style.color = '#cccccc'; }}
      >
        <LayoutGrid style={{ width: '15px', height: '15px' }} /> Browse sectors
      </Link>
    </div>

    {/* Dashboard preview */}
    <div style={{ width: '100%', maxWidth: '960px', position: 'relative', marginBottom: '80px', zIndex: 1 }}>
      {/* Rich Multi-Layered Glow behind the dashboard image */}
      <div aria-hidden style={{
        position: 'absolute',
        top: '-10%',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '105%', // Wider than the dashboard to spill out
        height: '110%', // Taller to shine above and below
        background: 'radial-gradient(ellipse at 50% 40%, rgba(59,130,246,0.45) 0%, rgba(147,51,234,0.25) 30%, rgba(0,229,153,0.08) 60%, transparent 80%)',
        filter: 'blur(70px)', // Slightly reduced blur to maintain saturation
        pointerEvents: 'none',
        zIndex: 0
      }} />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <img src="/dashboard-preview.png" alt="StockIQ Pro dashboard" style={{ width: '100%', height: 'auto', borderRadius: '12px', display: 'block', boxShadow: '0 24px 64px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.06)' }} loading="eager" />
        <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '35%', background: 'linear-gradient(to bottom, transparent, #000000)', borderRadius: '0 0 12px 12px', pointerEvents: 'none' }} />
      </div>
    </div>

    {/* Sector teaser cards */}
    <div style={{ width: '100%', maxWidth: '960px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '20px' }}>
        <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)' }} />
        <span style={{ fontSize: '11px', color: '#64748b', letterSpacing: '0.08em', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Quick access by sector</span>
        <div style={{ flex: 1, height: '1px', background: 'rgba(255,255,255,0.08)' }} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }} className="teaser-grid">
        <style>{`
          @media (min-width: 480px)  { .teaser-grid { grid-template-columns: repeat(4, 1fr) !important; } }
          @media (min-width: 1024px) { .teaser-grid { grid-template-columns: repeat(8, 1fr) !important; } }
        `}</style>
        {[
          { emoji: '🪙', label: 'ETFs',     color: '#eab308' },
          { emoji: '🏦', label: 'Banking',  color: '#00c48c' },
          { emoji: '💻', label: 'IT',        color: '#3b82f6' },
          { emoji: '⚡', label: 'Energy',    color: '#f59e0b' },
          { emoji: '💊', label: 'Pharma',    color: '#8b5cf6' },
          { emoji: '🚗', label: 'Auto',      color: '#ef4444' },
          { emoji: '🛒', label: 'FMCG',      color: '#10b981' },
          { emoji: '📈', label: 'Finance',   color: '#06b6d4' },
        ].map(s => (
          <Link key={s.label} href="/browse" className="browse-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', padding: '16px 8px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '10px', textDecoration: 'none', transition: 'all 0.15s ease', cursor: 'pointer' }}>
            <span style={{ fontSize: '22px' }}>{s.emoji}</span>
            <span style={{ fontSize: '11px', fontWeight: 500, color: '#94a3b8', textAlign: 'center' }}>{s.label}</span>
          </Link>
        ))}
      </div>
      <div style={{ textAlign: 'center', marginTop: '16px' }}>
        <Link href="/browse" style={{ fontSize: '13px', color: '#64748b', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px', transition: 'color 0.15s' }}
          onMouseEnter={e => e.currentTarget.style.color = '#fff'}
          onMouseLeave={e => e.currentTarget.style.color = '#64748b'}
        >
          View all 7,900+ stocks, ETFs &amp; indices across NSE &amp; BSE <ArrowRight style={{ width: '13px', height: '13px' }} />
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
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '120px 16px', gap: '16px' }}>
    <div style={{ width: '40px', height: '40px', border: '2px solid #111', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
    <p style={{ fontSize: '14px', color: '#555' }}>Loading <span style={{ color: '#fff', fontWeight: 600 }}>{ticker.replace('.NS', '').replace('.BO', '')}</span>…</p>
    <style>{`@keyframes spin { to { transform:rotate(360deg); } }`}</style>
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

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: 'transparent' }}>
      <Header onTickerSelect={handleTickerSelect} currentTicker={selectedTicker} />

      <main style={{ flex: 1, width: '100%', maxWidth: '1360px', margin: '0 auto', padding: '0 16px 48px' }}>
        {!selectedTicker ? (
          <WelcomeScreen />
        ) : isLoading ? (
          <LoadingState ticker={selectedTicker} />
        ) : (
          <div style={{ paddingTop: '20px' }}>
            {/* Status row with click-to-scroll */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 mb-4">
              <StatusBadge icon={Activity}  title="Live Prices"       subtitle="Real-time · 15 min delay" status="active" infoKey="live_prices" targetId="live-price-section" />
              <StatusBadge icon={Brain}     title="ML Predictions"    subtitle={isETF ? '30-Day ETF Outlook' : '5-Day Ensemble'} status="active" infoKey="ml_predictions" targetId="ml-prediction-section" />
              <StatusBadge icon={Newspaper} title="News Intelligence" subtitle="15-Day AI Sentiment & Directives" status="active" infoKey="news_intelligence" targetId="news-section" />
              <StatusBadge icon={PieChart}  title="Risk Analytics"    subtitle="VaR · Options · Portfolio" status="active" infoKey="risk_analytics" targetId="risk-analytics-section" />
            </div>

            {/* Quick Action & Active Asset Command Bar */}
            <div className="flex items-center justify-between flex-wrap gap-2.5 mb-5 p-3 sm:px-4 rounded-xl bg-white/[0.03] border border-white/[0.08] backdrop-blur-md">
              <div className="flex items-center gap-2.5 flex-wrap">
                <span className="text-xs text-slate-400 font-medium">Active Asset:</span>
                <span className="text-sm font-bold font-mono text-white bg-white/[0.06] border border-white/[0.12] px-2.5 py-1 rounded-lg">
                  {selectedTicker}
                </span>
                {companyName && (
                  <span className="text-xs text-slate-300 font-semibold hidden sm:inline truncate max-w-[280px]">
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
                  <span>{isWatchlisted ? 'Pinned to Watchlist' : 'Add to Watchlist'}</span>
                </button>
                <Link
                  href={`/intraday?ticker=${encodeURIComponent(selectedTicker)}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20 text-decoration-none shadow-[0_0_12px_rgba(16,185,129,0.1)]"
                  title="Open in High-Frequency Intraday Desk"
                >
                  <Activity className="w-3.5 h-3.5 text-emerald-400" />
                  <span>⚡ Intraday Trading Desk</span>
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
                  <span>🏛️ Committee Desk</span>
                </button>
                <button
                  onClick={() => setReportModalOpen(true)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold cursor-pointer transition-all bg-blue-500/10 border border-blue-500/30 text-blue-300 hover:bg-blue-500/20 shadow-[0_0_12px_rgba(59,130,246,0.1)]"
                >
                  <FileText className="w-3.5 h-3.5 text-blue-400" />
                  <span>Export Research Memo</span>
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
