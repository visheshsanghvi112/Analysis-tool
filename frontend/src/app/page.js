'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Header from './components/Header';
import LivePrice from './components/LivePrice';
import StockChart from './components/StockChart';
import MLPrediction from './components/MLPrediction';
import AdvancedNews from './components/AdvancedNews';
import PortfolioMetrics from './components/PortfolioMetrics';
import PeerComparison from './components/PeerComparison';
import SectorIntelligence from './components/SectorIntelligence';
import Backtesting from './components/Backtesting';
import LongTermAnalysis from './components/LongTermAnalysis';
import {
  TrendingUp, Brain, Newspaper, PieChart,
  Activity, ArrowRight, CheckCircle, Clock, AlertTriangle,
  LayoutGrid, BarChart2, Trophy,
} from 'lucide-react';

/* ── Status badge ─────────────────────────────────────────────────── */
const StatusBadge = ({ icon: Icon, title, subtitle, status }) => {
  const color = status === 'active' ? '#00c48c' : status === 'loading' ? '#f5a623' : '#444';
  const bg    = status === 'active' ? 'rgba(0,196,140,0.06)' : 'transparent';
  return (
    <div className="v-card" style={{ padding: '14px 16px', background: bg, borderColor: status === 'active' ? 'rgba(0,196,140,0.2)' : undefined }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
        <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: color + '18', border: `1px solid ${color}33`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          <Icon style={{ width: '16px', height: '16px', color }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ fontSize: '13px', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>{title}</p>
          <p style={{ fontSize: '11px', color: '#555', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{subtitle}</p>
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
        <div style={{ flex: 1, height: '1px', background: '#111' }} />
        <span style={{ fontSize: '11px', color: '#444', letterSpacing: '0.08em', textTransform: 'uppercase', whiteSpace: 'nowrap' }}>Quick access by sector</span>
        <div style={{ flex: 1, height: '1px', background: '#111' }} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }} className="teaser-grid">
        <style>{`
          @media (min-width: 480px)  { .teaser-grid { grid-template-columns: repeat(4, 1fr) !important; } }
          @media (min-width: 1024px) { .teaser-grid { grid-template-columns: repeat(8, 1fr) !important; } }
        `}</style>
        {[
          { emoji: '🏦', label: 'Banking',  color: '#00c48c' },
          { emoji: '💻', label: 'IT',        color: '#3b82f6' },
          { emoji: '⚡', label: 'Energy',    color: '#f59e0b' },
          { emoji: '💊', label: 'Pharma',    color: '#8b5cf6' },
          { emoji: '🚗', label: 'Auto',      color: '#ef4444' },
          { emoji: '🛒', label: 'FMCG',      color: '#10b981' },
          { emoji: '📈', label: 'Finance',   color: '#06b6d4' },
          { emoji: '🏗️', label: 'Infra',    color: '#f97316' },
        ].map(s => (
          <Link key={s.label} href="/browse" className="browse-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', padding: '16px 8px', background: '#060606', border: '1px solid #141414', borderRadius: '10px', textDecoration: 'none', transition: 'border-color 0.15s, background 0.15s', cursor: 'pointer' }}>
            <span style={{ fontSize: '22px' }}>{s.emoji}</span>
            <span style={{ fontSize: '11px', fontWeight: 500, color: '#888', textAlign: 'center' }}>{s.label}</span>
          </Link>
        ))}
      </div>
      <div style={{ textAlign: 'center', marginTop: '16px' }}>
        <Link href="/browse" style={{ fontSize: '13px', color: '#555', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '4px', transition: 'color 0.15s' }}
          onMouseEnter={e => e.currentTarget.style.color = '#fff'}
          onMouseLeave={e => e.currentTarget.style.color = '#555'}
        >
          View all 48 stocks across 8 sectors <ArrowRight style={{ width: '13px', height: '13px' }} />
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
  return (
    <div>
      {/* Tab row */}
      <div style={{ display: 'flex', gap: '6px', marginBottom: '12px', background: '#0a0a0a', border: '1px solid #1c1c1c', borderRadius: '10px', padding: '6px' }}>
        {PEER_TABS.map(tab => {
          const Icon = tab.icon;
          const isA = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '7px',
                padding: '9px 14px',
                background: isA ? '#1a1a1a' : 'transparent',
                border: `1px solid ${isA ? '#2a2a2a' : 'transparent'}`,
                borderRadius: '8px',
                color: isA ? '#fff' : '#555',
                fontSize: '12px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s',
              }}
            >
              <Icon style={{ width: '13px', height: '13px', color: isA ? (tab.id === 'compare' ? '#3b82f6' : '#f59e0b') : '#555' }} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>
      {/* Tab panels */}
      {activeTab === 'compare' && <PeerComparison ticker={ticker} />}
      {activeTab === 'sector'  && <SectorIntelligence ticker={ticker} />}
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

/* ── Dashboard ────────────────────────────────────────────────────── */
export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [isLoading, setIsLoading]           = useState(false);

  // Read ?ticker= param on mount (set by /browse page)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const t = params.get('ticker');
    if (t) {
      window.history.replaceState({}, '', '/');
      handleTickerSelect(t);
    }
  }, []);

  // Listen to logo click event to go back to homepage welcome screen
  useEffect(() => {
    const handleReset = () => {
      setSelectedTicker('');
    };
    window.addEventListener('reset-selected-ticker', handleReset);
    return () => window.removeEventListener('reset-selected-ticker', handleReset);
  }, []);

  const handleTickerSelect = (ticker) => {
    setIsLoading(true);
    setSelectedTicker(ticker);
    window.scrollTo({ top: 0 });
    setTimeout(() => setIsLoading(false), 400);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#000' }}>
      <Header onTickerSelect={handleTickerSelect} currentTicker={selectedTicker} />

      <main style={{ flex: 1, width: '100%', maxWidth: '1280px', margin: '0 auto', padding: '0 16px 48px' }}>
        {!selectedTicker ? (
          <WelcomeScreen />
        ) : isLoading ? (
          <LoadingState ticker={selectedTicker} />
        ) : (
          <div style={{ paddingTop: '20px' }}>
            {/* Status row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', marginBottom: '16px' }} className="status-row">
              <style>{`
                @media (min-width: 1024px) {
                  .status-row        { grid-template-columns: repeat(4,1fr) !important; }
                  .dash-main-grid    { grid-template-columns: 2fr 1fr !important; }
                }
              `}</style>
              <StatusBadge icon={Activity}  title="Live Prices"       subtitle="Real-time · 15 min delay" status="active" />
              <StatusBadge icon={Brain}     title="ML Predictions"    subtitle="5-day Random Forest"       status="active" />
              <StatusBadge icon={Newspaper} title="News Intelligence" subtitle="AI sentiment · Alerts"     status="active" />
              <StatusBadge icon={PieChart}  title="Risk Analytics"    subtitle="VaR · Options · Portfolio" status="active" />
            </div>

            {/* Charts grid */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '16px' }} className="dash-main-grid">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <LivePrice ticker={selectedTicker} />
                  <StockChart ticker={selectedTicker} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <MLPrediction ticker={selectedTicker} />
                  <AdvancedNews ticker={selectedTicker} />
                </div>
              </div>
              <PortfolioMetrics ticker={selectedTicker} />
              <Backtesting ticker={selectedTicker} />
              <LongTermAnalysis ticker={selectedTicker} />

              {/* ── Peer & Sector Intelligence Tabs ───────────────────── */}
              <PeerSectorTabs ticker={selectedTicker} />
            </div>
          </div>
        )}
      </main>

      <footer style={{ borderTop: '1px solid #111', padding: '20px 24px' }}>
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
    </div>
  );
}
