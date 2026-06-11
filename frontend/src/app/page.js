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
  <div className="glass-card p-5 hover:border-slate-700/80 transition-all duration-200 group"
       style={{ border: '1px solid rgba(148,163,184,0.1)' }}>
    <div className="flex items-start justify-between mb-3">
      <div className={`p-2.5 rounded-xl ${
        status === 'active'   ? 'bg-indigo-600/20 text-indigo-400' :
        status === 'loading'  ? 'bg-yellow-600/20 text-yellow-400' :
                                'bg-slate-700/40 text-slate-500'
      }`}>
        <Icon className="h-5 w-5" />
      </div>
      <div className={`flex items-center gap-1 text-xs font-semibold uppercase tracking-wider ${
        status === 'active'  ? 'text-emerald-400' :
        status === 'loading' ? 'text-yellow-400'  : 'text-slate-500'
      }`}>
        {status === 'active'   && <CheckCircle className="h-3 w-3" />}
        {status === 'loading'  && <Clock className="h-3 w-3 animate-spin" />}
        {status === 'inactive' && <AlertTriangle className="h-3 w-3" />}
        <span>{status === 'active' ? 'Live' : status === 'loading' ? 'Loading' : 'Offline'}</span>
      </div>
    </div>
    <h3 className="text-sm font-bold text-slate-200 mb-1 group-hover:text-white transition-colors">
      {title}
    </h3>
    <p className="text-xs text-slate-500 leading-relaxed">{description}</p>
  </div>
);

/* ── Welcome screen ──────────────────────────────────────────────── */
const WelcomeSection = () => (
  <div className="flex flex-col items-center justify-center min-h-[70vh] px-4 py-16 text-center">
    <div className="gradient-border mb-6 inline-block">
      <div className="bg-slate-900 p-4 rounded-[13px]">
        <TrendingUp className="h-12 w-12 text-indigo-400" />
      </div>
    </div>

    <h1 className="text-5xl md:text-7xl font-black gradient-text mb-5 tracking-tight leading-none">
      StockIQ Pro
    </h1>

    <p className="text-lg text-slate-400 mb-10 max-w-xl leading-relaxed">
      Professional-grade stock analysis with{' '}
      <span className="text-indigo-400 font-semibold">ML predictions</span>,{' '}
      <span className="text-emerald-400 font-semibold">real-time intelligence</span>, and{' '}
      <span className="text-purple-400 font-semibold">institutional analytics</span>
    </p>

    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-10 w-full max-w-2xl">
      <div className="glass-card p-5 flex flex-col items-center gap-3">
        <Brain className="h-7 w-7 text-indigo-400" />
        <div>
          <p className="font-bold text-slate-200 text-sm">AI Predictions</p>
          <p className="text-xs text-slate-500 mt-0.5">Random Forest · 20+ indicators</p>
        </div>
      </div>
      <div className="glass-card p-5 flex flex-col items-center gap-3">
        <Shield className="h-7 w-7 text-emerald-400" />
        <div>
          <p className="font-bold text-slate-200 text-sm">Risk Management</p>
          <p className="text-xs text-slate-500 mt-0.5">VaR · Sharpe · Portfolio metrics</p>
        </div>
      </div>
      <div className="glass-card p-5 flex flex-col items-center gap-3">
        <Zap className="h-7 w-7 text-purple-400" />
        <div>
          <p className="font-bold text-slate-200 text-sm">Real-Time Intel</p>
          <p className="text-xs text-slate-500 mt-0.5">Live prices · News sentiment</p>
        </div>
      </div>
    </div>

    <p className="text-slate-500 text-sm flex items-center gap-2">
      <Target className="h-4 w-4 text-indigo-400" />
      Search for any NSE / BSE stock above to begin
    </p>
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
    <div className="min-h-screen bg-slate-900 flex flex-col">
      <Header onTickerSelect={handleTickerSelect} currentTicker={selectedTicker} />

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {!selectedTicker ? (
          <WelcomeSection />
        ) : (
          <div className="space-y-6">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="h-10 w-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
                <p className="text-slate-400 text-sm">
                  Loading analysis for <span className="text-white font-semibold">{selectedTicker.replace('.NS', '').replace('.BO', '')}</span>…
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

      <footer className="border-t border-slate-800/60 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row justify-between items-center gap-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-indigo-400" />
              <span className="font-semibold text-slate-300 text-sm">StockIQ Pro</span>
              <span className="text-xs text-slate-500">by Vishesh Sanghvi</span>
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1"><Shield className="h-3.5 w-3.5" /> Secure</span>
              <span className="flex items-center gap-1"><CheckCircle className="h-3.5 w-3.5" /> Real-time</span>
              <span className="flex items-center gap-1"><Brain className="h-3.5 w-3.5" /> AI-Powered</span>
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
