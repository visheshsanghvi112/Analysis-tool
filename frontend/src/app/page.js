'use client';

import { useState } from 'react';
import Header from './components/Header';
import LivePrice from './components/LivePrice';
import StockChart from './components/StockChart';
import MLPrediction from './components/MLPrediction';
import AdvancedNews from './components/AdvancedNews';
import PortfolioMetrics from './components/PortfolioMetrics';
import {
  TrendingUp,
  Brain,
  Newspaper,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  Zap,
  Target,
  PieChart,
  Activity,
  BarChart3,
} from 'lucide-react';

/* ── Feature status card (shown after ticker selected) ───────────── */
const FeatureCard = ({ icon: Icon, title, description, status = 'active' }) => (
  <div className={`glass-card card-hover p-6 ${
    status === 'active' ? 'border-indigo-500/30' : 'border-slate-700/30'
  }`}>
    <div className="flex items-start justify-between mb-4">
      <div className={`p-3 rounded-xl transition-all duration-300 ${
        status === 'active'   ? 'bg-gradient-to-br from-indigo-500/20 to-purple-600/20 neon-glow' :
        status === 'loading'  ? 'bg-gradient-to-br from-yellow-500/20 to-orange-600/20' :
                                'bg-slate-700/40'
      }`}>
        <Icon className={`h-6 w-6 ${
          status === 'active'   ? 'text-indigo-400' :
          status === 'loading'  ? 'text-yellow-400' :
                                  'text-slate-500'
        }`} />
      </div>
      <div className={`flex items-center gap-2 text-xs font-bold uppercase tracking-wider px-3 py-1.5 rounded-full ${
        status === 'active'  ? 'status-badge-success text-emerald-400' :
        status === 'loading' ? 'status-badge-warning text-yellow-400'  : 
                               'bg-slate-800/60 text-slate-500'
      }`}>
        {status === 'active'   && <CheckCircle className="h-3.5 w-3.5" />}
        {status === 'loading'  && <Clock className="h-3.5 w-3.5 animate-spin" />}
        {status === 'inactive' && <AlertTriangle className="h-3.5 w-3.5" />}
        <span>{status === 'active' ? 'Live' : status === 'loading' ? 'Loading' : 'Offline'}</span>
      </div>
    </div>
    <h3 className="text-base font-bold text-white mb-2 group-hover:text-indigo-300 transition-colors">
      {title}
    </h3>
    <p className="text-sm text-slate-400 leading-relaxed">{description}</p>
  </div>
);

/* ── Welcome screen ──────────────────────────────────────────────── */
const WelcomeSection = () => (
  <div className="flex flex-col items-center justify-center min-h-[70vh] px-4 py-16 text-center relative">
    <div className="gradient-border mb-8 inline-block pulse-glow">
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 p-5 rounded-[14px]">
        <TrendingUp className="h-16 w-16 text-indigo-400" />
      </div>
    </div>

    <h1 className="text-6xl md:text-8xl font-black gradient-text mb-6 tracking-tight leading-none drop-shadow-2xl">
      StockIQ Pro
    </h1>

    <p className="text-xl text-slate-300 mb-12 max-w-2xl leading-relaxed">
      Professional-grade stock analysis with{' '}
      <span className="text-indigo-400 font-bold">ML predictions</span>,{' '}
      <span className="text-emerald-400 font-bold">real-time intelligence</span>, and{' '}
      <span className="text-purple-400 font-bold">institutional analytics</span>
    </p>

    <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-12 w-full max-w-3xl">
      <div className="glass-card card-hover p-6 flex flex-col items-center gap-4 border-indigo-500/20">
        <div className="p-3 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 neon-glow">
          <Brain className="h-8 w-8 text-indigo-400" />
        </div>
        <div>
          <p className="font-bold text-white text-base mb-1">AI Predictions</p>
          <p className="text-sm text-slate-400">Random Forest · 20+ indicators</p>
        </div>
      </div>
      <div className="glass-card card-hover p-6 flex flex-col items-center gap-4 border-emerald-500/20">
        <div className="p-3 rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-600/20 success-glow">
          <Shield className="h-8 w-8 text-emerald-400" />
        </div>
        <div>
          <p className="font-bold text-white text-base mb-1">Risk Management</p>
          <p className="text-sm text-slate-400">VaR · Sharpe · Portfolio metrics</p>
        </div>
      </div>
      <div className="glass-card card-hover p-6 flex flex-col items-center gap-4 border-purple-500/20">
        <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-600/20">
          <Zap className="h-8 w-8 text-purple-400" />
        </div>
        <div>
          <p className="font-bold text-white text-base mb-1">Real-Time Intel</p>
          <p className="text-sm text-slate-400">Live prices · News sentiment</p>
        </div>
      </div>
    </div>

    <div className="glass-card px-6 py-4 inline-flex items-center gap-3 border-indigo-500/30">
      <Target className="h-5 w-5 text-indigo-400" />
      <p className="text-slate-300 text-base font-medium">
        Search for any NSE / BSE stock above to begin
      </p>
    </div>
  </div>
);

/* ── Status bar shown after ticker selected ──────────────────────── */
const StatsOverview = ({ ticker }) => (
  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
    <FeatureCard icon={Activity}   title="Live Prices"      description="Real-time quotes · 15-min delay" status="active" />
    <FeatureCard icon={Brain}      title="ML Predictions"   description="5-day Random Forest forecasts"   status={ticker ? 'active' : 'inactive'} />
    <FeatureCard icon={Newspaper}  title="News Intelligence" description="AI sentiment · breaking alerts"  status={ticker ? 'active' : 'inactive'} />
    <FeatureCard icon={PieChart}   title="Risk Analytics"   description="VaR · options · portfolio grade"  status={ticker ? 'active' : 'inactive'} />
  </div>
);

/* ── Dashboard ───────────────────────────────────────────────────── */
export default function Dashboard() {
  const [selectedTicker, setSelectedTicker] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleTickerSelect = (ticker) => {
    setIsLoading(true);
    setSelectedTicker(ticker);
    setTimeout(() => setIsLoading(false), 400);
  };

  return (
    <div className="min-h-screen flex flex-col relative">
      <Header onTickerSelect={handleTickerSelect} currentTicker={selectedTicker} />

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 relative z-10">
        {!selectedTicker ? (
          <WelcomeSection />
        ) : (
          <div className="space-y-6">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-32 gap-6">
                <div className="relative">
                  <div className="h-16 w-16 border-4 border-indigo-600/30 border-t-indigo-500 rounded-full animate-spin" />
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-12 w-12 border-4 border-purple-600/30 border-b-purple-500 rounded-full animate-spin" style={{animationDirection: 'reverse', animationDuration: '1s'}} />
                </div>
                <p className="text-slate-300 text-base font-medium">
                  Loading analysis for <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 font-bold">{selectedTicker.replace('.NS', '').replace('.BO', '')}</span>…
                </p>
              </div>
            ) : (
              <>
                <StatsOverview ticker={selectedTicker} />

                {/* Main grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Left: price + chart */}
                  <div className="lg:col-span-2 flex flex-col gap-6">
                    <LivePrice ticker={selectedTicker} />
                    <StockChart ticker={selectedTicker} />
                  </div>

                  {/* Right: ML + news */}
                  <div className="flex flex-col gap-6">
                    <MLPrediction ticker={selectedTicker} />
                    <AdvancedNews ticker={selectedTicker} />
                  </div>
                </div>

                {/* Portfolio metrics full width */}
                <PortfolioMetrics ticker={selectedTicker} />
              </>
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-slate-800/60 mt-auto relative z-10 glass-card rounded-none">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="gradient-border">
                <div className="bg-slate-900 p-2 rounded-xl">
                  <TrendingUp className="h-5 w-5 text-indigo-400" />
                </div>
              </div>
              <div>
                <span className="font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400 text-base">
                  StockIQ Pro
                </span>
                <span className="text-sm text-slate-500 ml-2">by Vishesh Sanghvi</span>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm text-slate-400">
              <span className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                <Shield className="h-4 w-4 text-emerald-400" /> Secure
              </span>
              <span className="flex items-center gap-2 px-3 py-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
                <CheckCircle className="h-4 w-4 text-indigo-400" /> Real-time
              </span>
              <span className="flex items-center gap-2 px-3 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <Brain className="h-4 w-4 text-purple-400" /> AI-Powered
              </span>
            </div>
          </div>
          <p className="mt-4 text-center text-xs text-slate-600">
            Data via Yahoo Finance (~15 min delay). Not financial advice. Consult a qualified advisor.
          </p>
        </div>
      </footer>
    </div>
  );
}
