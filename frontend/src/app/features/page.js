'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import Link from 'next/link';
import Header from '../components/Header';
import { 
  Brain, Cpu, LineChart as LineChartIcon, TrendingUp, Newspaper, ShieldAlert, 
  PieChart as PieChartIcon, Activity, ChevronRight, Info, BookOpen, Layers, 
  TrendingDown, Target, HelpCircle, ArrowLeft, ArrowRight, ActivitySquare,
  GitBranch, CheckSquare, BarChart2, ShieldCheck, Clipboard, Sliders, Play,
  Settings, Database, HelpCircle as QuestionIcon, RefreshCw, Terminal
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  LineChart,
  Line,
  ReferenceLine,
  PieChart,
  Pie,
  AreaChart,
  Area
} from 'recharts';

const PIPELINE_STEPS = [
  { id: 'ingestion', label: '01. Ingestion', subtitle: 'Market Prices & News' },
  { id: 'features', label: '02. Features', subtitle: '40+ Alpha Indicators' },
  { id: 'hmm', label: '03. HMM Regime', subtitle: 'Markov States' },
  { id: 'ensemble', label: '04. Stacked ML', subtitle: 'Ensemble & SHAP' },
  { id: 'risk', label: '05. Sizing & Risk', subtitle: 'GARCH & VaR Sizing' }
];

// Custom Premium Tooltip Component
function CustomTooltip({ active, payload, label, formatter }) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#09090b]/95 border border-white/[0.08] p-3 rounded-xl shadow-2xl backdrop-blur-md">
        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">{label}</p>
        <p className="text-sm font-mono font-bold text-white mt-1">
          {formatter ? formatter(payload[0].value) : `${payload[0].value}%`}
        </p>
      </div>
    );
  }
  return null;
}

export default function DeepdiveFeatures() {
  const [activeView, setActiveView] = useState('sandbox'); // 'sandbox', 'pipeline', 'faq'
  const [selectedPipelineStep, setSelectedPipelineStep] = useState('hmm');
  const [isMounted, setIsMounted] = useState(false);
  const [sandboxTab, setSandboxTab] = useState('tranches'); // 'tranches' or 'shap'

  // Simulator State
  const [simCapital, setSimCapital] = useState(100000);
  const [simRegime, setSimRegime] = useState('0'); // '0': Bullish, '1': Sideways, '2': Panic Bearish
  const [simMlExpectation, setSimMlExpectation] = useState(2.5); // % return expected
  const [simSentiment, setSimSentiment] = useState(0.4); // -1 to +1
  const [simTrendConfirm, setSimTrendConfirm] = useState(true);

  // Model-specific simulator parameters (Interactive Math Details)
  const [ridgeLambda, setRidgeLambda] = useState(1.0); // L2 penalty for meta-learner
  const [garchShock, setGarchShock] = useState(1.5); // Recent market shock coefficient

  // Live Backtest Simulator State
  const [isBacktesting, setIsBacktesting] = useState(false);
  const [backtestProgress, setBacktestProgress] = useState(0);
  const [backtestLogs, setBacktestLogs] = useState([]);
  const [showBacktestResult, setShowBacktestResult] = useState(false);
  const logIntervalRef = useRef(null);

  // Hydration Guard
  useEffect(() => {
    setIsMounted(true);
    return () => {
      if (logIntervalRef.current) clearInterval(logIntervalRef.current);
    };
  }, []);

  // 1. Memoized Simulator Calculation Flow
  const simResult = useMemo(() => {
    const sentimentBonus = simSentiment * 1.5; 
    const fused = Math.round((simMlExpectation * 0.8 + sentimentBonus * 0.2) * 100) / 100;

    let threshold = 1.8;
    let notes = '';
    if (simRegime === '1') {
      threshold = 2.2;
      notes = 'Consolidating market regime. Volatility buffer set to medium (+2.2%).';
    } else if (simRegime === '2') {
      threshold = 3.0;
      notes = 'HMM identifies high-volatility panic. Buy threshold scaled to 3.0% to protect capital.';
    } else {
      notes = 'Low volatility bullish regime. Standard 1.8% entry threshold active.';
    }

    let status = 'APPROVED ENTRY';
    let color = 'text-emerald-400 border-emerald-500/25 bg-emerald-500/[0.03] shadow-[0_0_15px_rgba(16,185,129,0.05)]';
    let pulseColor = 'bg-emerald-500';
    
    if (!simTrendConfirm) {
      status = 'BLOCKED: NO TREND CONFIRMATION';
      color = 'text-rose-400 border-rose-500/25 bg-rose-500/[0.03] shadow-[0_0_15px_rgba(244,63,94,0.05)]';
      pulseColor = 'bg-rose-500';
    } else if (fused < threshold) {
      status = 'HOLD: INSUFFICIENT FUSED RETURN';
      color = 'text-amber-400 border-amber-500/25 bg-amber-500/[0.03] shadow-[0_0_15px_rgba(245,158,11,0.05)]';
      pulseColor = 'bg-amber-500';
    } else if (simRegime === '2') {
      status = 'APPROVED (HIGH-RISK PANIC SECTOR)';
      color = 'text-sky-400 border-sky-500/25 bg-sky-500/[0.03] shadow-[0_0_15px_rgba(56,189,248,0.05)]';
      pulseColor = 'bg-sky-400';
    }

    let tranches = [];
    if (status.startsWith('APPROVED')) {
      if (simRegime === '2') {
        tranches = [
          { name: 'Tranche 1 (Initial Setup)', pct: 30, amount: simCapital * 0.3, trigger: 'Execute immediately at market close.', fill: '#818cf8' },
          { name: 'Tranche 2 (Dip Limit)', pct: 40, amount: simCapital * 0.4, trigger: 'Trigger only if stock pulls back 3%.', fill: '#6366f1' },
          { name: 'Tranche 3 (Confirmation)', pct: 30, amount: simCapital * 0.3, trigger: 'Trigger if price crosses above 20-period EMA.', fill: '#4f46e5' }
        ];
      } else {
        tranches = [
          { name: 'Tranche 1 (Initial Setup)', pct: 50, amount: simCapital * 0.5, trigger: 'Execute immediately.', fill: '#10b981' },
          { name: 'Tranche 2 (Dip Limit)', pct: 30, amount: simCapital * 0.3, trigger: 'Trigger on minor intraday pullbacks (-1.5%).', fill: '#059669' },
          { name: 'Tranche 3 (Momentum)', pct: 20, amount: simCapital * 0.2, trigger: 'Trigger on MACD bullish confirmation.', fill: '#047857' }
        ];
      }
    }

    return {
      status,
      color,
      pulseColor,
      fusedReturn: fused,
      requiredThreshold: threshold,
      tranches,
      mitigationNotes: notes
    };
  }, [simCapital, simRegime, simMlExpectation, simSentiment, simTrendConfirm]);

  // 2. Memoized SHAP Attribution Calculations
  const shapData = useMemo(() => {
    const mlContribution = Math.round(simMlExpectation * 0.8 * 100) / 100;
    const sentimentContribution = Math.round(simSentiment * 0.3 * 100) / 100;
    const regimeContribution = simRegime === '0' ? 0.25 : (simRegime === '1' ? -0.15 : -0.65);
    const trendContribution = simTrendConfirm ? 0.35 : -0.75;
    
    return [
      { name: 'Base Intercept', value: 1.10, fill: '#6366f1' },
      { name: 'Ensemble Consensus', value: mlContribution, fill: mlContribution >= 0 ? '#10b981' : '#f43f5e' },
      { name: 'Sentiment NLP', value: sentimentContribution, fill: sentimentContribution >= 0 ? '#10b981' : '#f43f5e' },
      { name: 'HMM Regime Friction', value: regimeContribution, fill: regimeContribution >= 0 ? '#10b981' : '#f43f5e' },
      { name: 'Trend Confirmation', value: trendContribution, fill: trendContribution >= 0 ? '#10b981' : '#f43f5e' }
    ];
  }, [simMlExpectation, simSentiment, simRegime, simTrendConfirm]);

  // 3. Memoized GARCH Volatility Decay Calculations
  const garchData = useMemo(() => {
    let baseVol = 12.5; // %
    let alpha = 0.08;
    let beta = 0.90;
    
    if (simRegime === '1') {
      baseVol = 20.0;
    } else if (simRegime === '2') {
      baseVol = 38.0;
    }

    // Amplify GARCH volatility spike based on the user's interactive shock multiplier
    baseVol = baseVol * (0.5 + garchShock / 2.0);

    const list = [];
    let currentVol = baseVol;
    const meanVol = 16.0;

    for (let day = 1; day <= 5; day++) {
      list.push({
        day: `T+${day}`,
        Volatility: Math.round(currentVol * 100) / 100
      });
      // GARCH conditional variance update formula
      currentVol = Math.sqrt(0.02 * (meanVol * meanVol) + alpha * (currentVol * currentVol * 0.9) + beta * (currentVol * currentVol));
    }
    return list;
  }, [simRegime, garchShock]);

  // 4. Memoized Ridge Regularization Weight Shrinkage Calculations
  const ridgeWeightsData = useMemo(() => {
    // Under low lambda, weights are highly dispersed due to base model correlation
    const baseWeights = {
      rf: 0.38,
      gbm: 0.12,
      xgb: 0.44,
      lgbm: 0.08,
      extra: -0.02
    };

    // Shrinkage towards equal weight consensus (0.20 each) as lambda L2 penalty increases
    const shrink = (val) => {
      const equalWeight = 0.20;
      const factor = 1 / (1 + ridgeLambda * 0.4);
      return Math.round((equalWeight + (val - equalWeight) * factor) * 1000) / 1000;
    };

    return [
      { name: 'Random Forest', Weight: shrink(baseWeights.rf) },
      { name: 'Gradient Boost', Weight: shrink(baseWeights.gbm) },
      { name: 'XGBoost', Weight: shrink(baseWeights.xgb) },
      { name: 'LightGBM', Weight: shrink(baseWeights.lgbm) },
      { name: 'Extra Trees', Weight: shrink(baseWeights.extra) }
    ];
  }, [ridgeLambda]);

  // 5. Simulated Walk-Forward Cumulative Return data
  const backtestChartData = [
    { year: 'Fold 1 start', 'Stacked ML Return': 100, 'Buy & Hold Return': 100 },
    { year: 'Fold 1 end', 'Stacked ML Return': 118, 'Buy & Hold Return': 109 },
    { year: 'Fold 2 end', 'Stacked ML Return': 136, 'Buy & Hold Return': 114 },
    { year: 'Fold 3 end', 'Stacked ML Return': 158, 'Buy & Hold Return': 122 },
    { year: 'Fold 4 end', 'Stacked ML Return': 189, 'Buy & Hold Return': 134 },
    { year: 'Out-of-Sample', 'Stacked ML Return': 214, 'Buy & Hold Return': 142 }
  ];

  // 6. Handle Triggering the Backtest
  const runBacktest = () => {
    if (isBacktesting) return;
    setIsBacktesting(true);
    setShowBacktestResult(false);
    setBacktestProgress(5);
    setBacktestLogs(['[LOG] Initializing walk-forward validation matrix...']);

    const messages = [
      { p: 15, msg: '[LOG] Ingesting 2 years price index dataset...' },
      { p: 30, msg: '[FOLD 1] Splitting chronological validation matrix. In-sample fit...' },
      { p: 45, msg: '[FOLD 2] Computing features (RSI, GARCH, News Sentiment Polarity)...' },
      { p: 60, msg: '[FOLD 3] Fitting Ridge regularized meta-regressor. Computing coefficients...' },
      { p: 75, msg: '[FOLD 4] Calculating Out-of-Sample predictions. Verifying data leak boundaries...' },
      { p: 90, msg: '[SUCCESS] Walk-forward optimization complete. Alpha generated: +72.0%.' },
      { p: 100, msg: '[SUCCESS] Backtest validation finished. Rendering performance comparison.' }
    ];

    let currentMsgIdx = 0;
    
    logIntervalRef.current = setInterval(() => {
      setBacktestProgress(prev => {
        const next = prev + 5;
        if (next >= 100) {
          clearInterval(logIntervalRef.current);
          setIsBacktesting(false);
          setShowBacktestResult(true);
          return 100;
        }

        // Add logs progressively
        const currentTarget = messages[currentMsgIdx];
        if (currentTarget && next >= currentTarget.p) {
          setBacktestLogs(logs => [...logs, currentTarget.msg]);
          currentMsgIdx++;
        }

        return next;
      });
    }, 150);
  };

  return (
    <div className="min-h-screen bg-[#030303] text-white flex flex-col font-sans selection:bg-indigo-500 selection:text-white relative">
      
      {/* Background Dot Grid Pattern */}
      <div 
        aria-hidden="true" 
        className="absolute inset-0 bg-[radial-gradient(#ffffff03_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none z-0" 
      />

      <style jsx global>{`
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: #ffffff;
          border: 2.5px solid #6366f1;
          cursor: pointer;
          transition: transform 0.15s ease, background-color 0.15s ease;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.25);
          background: #6366f1;
        }
        input[type="range"]::-moz-range-thumb {
          width: 16px;
          height: 16px;
          border: 2.5px solid #6366f1;
          border-radius: 50%;
          background: #ffffff;
          cursor: pointer;
          transition: transform 0.15s ease, background-color 0.15s ease;
        }
        input[type="range"]::-moz-range-thumb:hover {
          transform: scale(1.25);
          background: #6366f1;
        }
        /* Hide scrollbars */
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
        @keyframes pulse-glow {
          0%, 100% { opacity: 0.4; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.05); }
        }
        .pulse-effect {
          animation: pulse-glow 2s infinite ease-in-out;
        }
      `}</style>

      <Header />

      {/* Hero Section */}
      <section className="relative overflow-hidden py-14 sm:py-18 border-b border-white/[0.03] z-10 bg-[#040404]">
        {/* Glow */}
        <div aria-hidden className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-7xl h-[300px] bg-gradient-to-b from-indigo-500/10 via-transparent to-transparent blur-[120px] pointer-events-none" />

        <div className="max-w-5xl mx-auto px-4 text-center relative z-10">
          <Link 
            href="/"
            className="inline-flex items-center gap-1.5 text-xs uppercase tracking-widest font-black text-slate-400 hover:text-white transition mb-6 bg-white/[0.02] border border-white/[0.06] px-4 py-2 rounded-full"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to Dashboard
          </Link>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight leading-none text-white">
            Systematic Equity Engine
          </h1>
          <p className="text-sm sm:text-base text-slate-400 mt-4 max-w-2xl mx-auto leading-relaxed font-normal">
            A three-part framework designed for long-term quantitative stock analysis, risk budgeting, and algorithmic execution.
          </p>

          {/* Three View Tabs */}
          <div className="flex justify-center mt-10">
            <div className="bg-white/[0.01] border border-white/[0.06] p-1 rounded-2xl sm:rounded-full flex flex-col sm:flex-row gap-1 backdrop-blur-md w-full sm:w-auto">
              <button
                onClick={() => setActiveView('sandbox')}
                className={`flex items-center justify-center gap-2 px-5 py-3 sm:py-2.5 rounded-xl sm:rounded-full text-xs sm:text-sm font-bold transition duration-350 cursor-pointer ${
                  activeView === 'sandbox'
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/10'
                    : 'text-slate-400 hover:text-slate-200 bg-transparent'
                }`}
              >
                <Sliders className="h-4 w-4" />
                1. Interactive Sandbox
              </button>
              <button
                onClick={() => setActiveView('pipeline')}
                className={`flex items-center justify-center gap-2 px-5 py-3 sm:py-2.5 rounded-xl sm:rounded-full text-xs sm:text-sm font-bold transition duration-350 cursor-pointer ${
                  activeView === 'pipeline'
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/10'
                    : 'text-slate-400 hover:text-slate-200 bg-transparent'
                }`}
              >
                <GitBranch className="h-4 w-4" />
                2. Pipeline &amp; Models
              </button>
              <button
                onClick={() => setActiveView('faq')}
                className={`flex items-center justify-center gap-2 px-5 py-3 sm:py-2.5 rounded-xl sm:rounded-full text-xs sm:text-sm font-bold transition duration-350 cursor-pointer ${
                  activeView === 'faq'
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/10'
                    : 'text-slate-400 hover:text-slate-200 bg-transparent'
                }`}
              >
                <BookOpen className="h-4 w-4" />
                3. Engineering FAQs
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Main Content Area */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 py-10 relative z-10">
        
        {/* VIEW 1: INTERACTIVE SANDBOX */}
        {activeView === 'sandbox' && (
          <div className="space-y-12 sm:space-y-16 animate-fadeIn">
            
            {/* Explanatory intro */}
            <div className="max-w-2xl text-left space-y-3">
              <span className="text-xs uppercase tracking-wider font-extrabold text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded border border-indigo-500/25">Live Sandbox</span>
              <p className="text-sm sm:text-base text-slate-400 leading-relaxed">
                Use this simulator to see how the mathematical core of StockIQ Pro evaluates risk. Adjust the parameters below to trigger regime shifting, sentiment weighting, and tranche execution calculations.
              </p>
            </div>

            {/* Sandbox Container */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 bg-white/[0.01] border border-white/[0.05] p-5 sm:p-8 rounded-3xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-48 h-48 bg-indigo-500/[0.02] blur-[80px] rounded-full pointer-events-none" />

              {/* Controls */}
              <div className="lg:col-span-5 space-y-6">
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Capital to Deploy (INR)</label>
                  <input 
                    type="number"
                    value={simCapital}
                    onChange={e => setSimCapital(Math.max(1000, Number(e.target.value)))}
                    className="w-full bg-white/[0.02] border border-white/[0.08] rounded-xl px-4 py-3 text-sm focus:border-indigo-500 outline-none text-white font-mono"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Detected Market Regime</label>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { val: '0', label: 'Bullish' },
                      { val: '1', label: 'Sideways' },
                      { val: '2', label: 'Panic Sell' }
                    ].map(r => (
                      <button
                        key={r.val}
                        onClick={() => setSimRegime(r.val)}
                        className={`py-2 rounded-lg border text-xs font-bold transition cursor-pointer ${
                          simRegime === r.val
                            ? 'bg-white border-white text-black font-extrabold'
                            : 'bg-white/[0.01] border-white/[0.05] text-slate-400 hover:text-white'
                        }`}
                      >
                        {r.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block">ML Return Target: {simMlExpectation}%</label>
                    <span className="text-xs text-slate-500 font-mono">Consensus</span>
                  </div>
                  <input 
                    type="range"
                    min="-2.0"
                    max="5.0"
                    step="0.1"
                    value={simMlExpectation}
                    onChange={e => setSimMlExpectation(parseFloat(e.target.value))}
                    className="w-full h-1 bg-white/[0.08] rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block">News Sentiment: {simSentiment > 0 ? '+' : ''}{simSentiment}</label>
                    <span className="text-xs text-slate-500 font-mono">NLP Core</span>
                  </div>
                  <input 
                    type="range"
                    min="-1.0"
                    max="1.0"
                    step="0.1"
                    value={simSentiment}
                    onChange={e => setSimSentiment(parseFloat(e.target.value))}
                    className="w-full h-1 bg-white/[0.08] rounded-lg appearance-none cursor-pointer accent-indigo-500"
                  />
                </div>

                <div className="flex items-center justify-between p-4 bg-white/[0.01] border border-white/[0.05] rounded-xl">
                  <div>
                    <span className="text-xs sm:text-sm font-bold text-slate-300 block">Trend Confirmation Filter</span>
                    <span className="text-xs text-slate-500 block mt-0.5">RSI / 200 EMA check</span>
                  </div>
                  <input 
                    type="checkbox"
                    checked={simTrendConfirm}
                    onChange={e => setSimTrendConfirm(e.target.checked)}
                    className="w-5 h-5 rounded text-indigo-600 bg-black border-white/[0.1] accent-indigo-500 cursor-pointer"
                  />
                </div>
              </div>

              {/* Output Dashboard */}
              <div className="lg:col-span-7 flex flex-col justify-between bg-black/40 border border-white/[0.06] p-5 sm:p-6 rounded-2xl space-y-6">
                
                <div className="space-y-2">
                  <div className="text-xs font-mono text-slate-400 uppercase tracking-wider font-bold">Evaluation Status</div>
                  <div className={`border p-4 rounded-xl flex items-center justify-center gap-3 font-mono font-extrabold text-xs sm:text-sm tracking-wider ${simResult.color}`}>
                    <span className={`h-2.5 w-2.5 rounded-full ${simResult.pulseColor} pulse-effect shrink-0`} />
                    {simResult.status}
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="bg-white/[0.01] p-3.5 rounded-xl border border-white/[0.04]">
                    <span className="text-xs text-slate-500 uppercase font-bold tracking-wider block">Fused Expectation</span>
                    <span className="text-lg font-bold text-white block mt-1 font-mono">{simResult.fusedReturn > 0 ? '+' : ''}{simResult.fusedReturn}%</span>
                    <span className="text-xs text-slate-500 block mt-0.5">(80% ML / 20% News)</span>
                  </div>
                  <div className="bg-white/[0.01] p-3.5 rounded-xl border border-white/[0.04]">
                    <span className="text-xs text-slate-500 uppercase font-bold tracking-wider block">Regime Threshold</span>
                    <span className="text-lg font-bold text-white block mt-1 font-mono">+{simResult.requiredThreshold}%</span>
                    <span className="text-xs text-slate-500 block mt-0.5">(HMM state scale)</span>
                  </div>
                </div>

                {/* Sub Tab inside Output Area: Tranche vs SHAP */}
                <div className="border-t border-white/[0.06] pt-4">
                  <div className="flex gap-2 mb-4 bg-white/[0.02] border border-white/[0.06] p-1 rounded-xl w-max">
                    <button
                      onClick={() => setSandboxTab('tranches')}
                      className={`px-3.5 py-1.5 rounded-lg text-xs font-extrabold transition cursor-pointer ${
                        sandboxTab === 'tranches' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Tranche Allocation
                    </button>
                    <button
                      onClick={() => setSandboxTab('shap')}
                      className={`px-3.5 py-1.5 rounded-lg text-xs font-extrabold transition cursor-pointer ${
                        sandboxTab === 'shap' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      Live SHAP Waterfall Chart
                    </button>
                  </div>

                  {sandboxTab === 'tranches' ? (
                    <div className="space-y-4">
                      {simResult.tranches.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                          {/* Donut Chart visualizing softmax tranches */}
                          <div className="md:col-span-5 h-[130px] flex items-center justify-center relative">
                            {isMounted ? (
                              <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                  <Pie
                                    data={simResult.tranches}
                                    dataKey="pct"
                                    nameKey="name"
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={32}
                                    outerRadius={50}
                                    paddingAngle={3}
                                  >
                                    {simResult.tranches.map((entry, index) => (
                                      <Cell key={`cell-${index}`} fill={entry.fill} />
                                    ))}
                                  </Pie>
                                </PieChart>
                              </ResponsiveContainer>
                            ) : null}
                            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                              <span className="text-[10px] text-slate-500 font-mono uppercase font-bold">Total</span>
                              <span className="text-xs font-bold text-white font-mono">100%</span>
                            </div>
                          </div>

                          {/* Tranches Description list */}
                          <div className="md:col-span-7 space-y-2.5">
                            {simResult.tranches.map(t => (
                              <div key={t.name} className="flex justify-between items-start gap-4 p-2 bg-white/[0.01] border border-white/[0.04] rounded-lg text-xs">
                                <div className="flex gap-2">
                                  <span className="h-2 w-2 rounded-full mt-1.5 shrink-0" style={{ backgroundColor: t.fill }} />
                                  <div>
                                    <span className="font-bold text-slate-200 block text-[11px] sm:text-xs">{t.name}</span>
                                    <span className="text-slate-400 block mt-0.5 text-[10px]">{t.trigger}</span>
                                  </div>
                                </div>
                                <div className="text-right shrink-0">
                                  <span className="font-bold text-indigo-400 block font-mono text-[11px] sm:text-xs">₹{t.amount.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                                  <span className="text-[10px] text-slate-500 block mt-0.5">{t.pct}% weight</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="p-5 bg-white/[0.01] border border-dashed border-white/[0.06] rounded-lg text-center text-xs text-slate-500 font-mono">
                          Deployment disabled due to blocked/hold status.
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="h-[200px] w-full pt-1">
                      {isMounted ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={shapData}
                            margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                            <XAxis type="number" stroke="rgba(255,255,255,0.3)" fontSize={10} fontClassName="font-mono" />
                            <YAxis dataKey="name" type="category" stroke="rgba(255,255,255,0.3)" fontSize={10} width={110} />
                            <Tooltip content={<CustomTooltip formatter={(val) => `${val > 0 ? '+' : ''}${val}%`} />} />
                            <Bar dataKey="value">
                              {shapData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-full flex items-center justify-center text-xs text-slate-500 font-mono">
                          Loading SHAP Waterfall Canvas...
                        </div>
                      )}
                    </div>
                  )}
                </div>

              </div>
            </div>

            {/* Systematic Flow timeline (6 Steps) */}
            <div className="space-y-8">
              <div className="flex items-center gap-3">
                <div className="h-9 w-9 bg-white/[0.02] border border-white/[0.06] rounded-lg flex items-center justify-center">
                  <Target className="h-5 w-5 text-indigo-400" />
                </div>
                <div>
                  <h3 className="text-base sm:text-lg font-black text-white">Systematic Long-Term Execution Workflow</h3>
                  <p className="text-xs sm:text-sm text-slate-500 font-medium">How these calculated metrics map to long-term investment decisions.</p>
                </div>
              </div>

              <div className="divide-y divide-white/[0.04] border-t border-b border-white/[0.04]">
                {[
                  { step: '01', title: 'Macro Regime Validation', desc: 'The Gaussian HMM checks if the overall market state is safe. If the stock is classified in a panic sell-off regime, the entry threshold increases to 3.0% to prevent catch-falling-knife actions.' },
                  { step: '02', title: 'Consensus Fused return', desc: 'Fuses tree-based model targets with news NLP sentiment. A Ridge Meta-learner prevents feature duplication and calculates a net directional signal.' },
                  { step: '03', title: 'SHAP Attribution Check', desc: 'The investor validates the prediction by looking at the SHAP waterfall. Ensures predictions are backed by structural features rather than micro noise.' },
                  { step: '04', title: 'Downside Risk Budgeting', desc: 'Estimates 5-day conditional volatility using GARCH(1,1) alongside historical 95% and 99% Value-at-Risk (VaR) to size the position.' },
                  { step: '05', title: 'Smart Allocation Execution', desc: 'Determines the final tranche distribution weights (softmax) to execute the trade in stages, protecting capital reserves.' }
                ].map(item => (
                  <div key={item.step} className="py-6 flex flex-col md:flex-row gap-4 md:gap-8 px-2">
                    <span className="text-xs font-mono font-black text-indigo-400 bg-indigo-500/10 border border-indigo-500/25 px-3 py-1 rounded shrink-0 h-fit w-max">
                      {item.step}
                    </span>
                    <div className="space-y-1.5">
                      <span className="text-sm sm:text-base font-extrabold text-white block">{item.title}</span>
                      <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

        {/* VIEW 2: INTERACTIVE PIPELINE & MATHEMATICAL MODELS */}
        {activeView === 'pipeline' && (
          <div className="space-y-10 animate-fadeIn">
            
            <div className="max-w-2xl text-left space-y-2">
              <span className="text-xs uppercase tracking-wider font-extrabold text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded border border-indigo-500/25">Model Architecture</span>
              <p className="text-sm sm:text-base text-slate-400 leading-relaxed mt-2">
                Click on any step of the pipeline below to inspect its mathematical formula, configuration, and long-term strategic role.
              </p>
            </div>

            {/* Horizontal Swipeable Steps on Mobile, Grid on Desktop */}
            <div className="flex md:grid md:grid-cols-5 gap-3 overflow-x-auto no-scrollbar pb-3 border-b border-white/[0.04]">
              {PIPELINE_STEPS.map(step => {
                const isSelected = selectedPipelineStep === step.id;
                return (
                  <button
                    key={step.id}
                    onClick={() => setSelectedPipelineStep(step.id)}
                    className={`text-left p-4 rounded-xl border transition duration-300 cursor-pointer shrink-0 w-[160px] md:w-auto ${
                      isSelected 
                        ? 'bg-indigo-600/10 border-indigo-500/40 shadow-[0_0_15px_rgba(99,102,241,0.05)]'
                        : 'bg-white/[0.01] border-white/[0.05] hover:border-white/[0.1] hover:bg-white/[0.02]'
                    }`}
                  >
                    <span className={`text-xs font-mono font-bold block ${isSelected ? 'text-indigo-300' : 'text-slate-500'}`}>
                      {step.label.split('.')[0]}. {step.id.toUpperCase()}
                    </span>
                    <span className="text-xs sm:text-sm font-extrabold text-white block mt-2 leading-tight">{step.label.split('. ')[1]}</span>
                    <span className="text-[10px] sm:text-xs text-slate-500 block mt-1.5 font-medium">{step.subtitle}</span>
                  </button>
                );
              })}
            </div>

            {/* Dynamic Math / Spec Details Container */}
            <div className="bg-[#050508] border border-white/[0.05] p-6 sm:p-8 rounded-3xl min-h-[300px]">
              
              {selectedPipelineStep === 'ingestion' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <Database className="h-6 w-6 text-indigo-400" />
                    <h3 className="text-base sm:text-lg font-extrabold text-white">01. Data Ingestion Pipeline</h3>
                  </div>
                  <p className="text-xs sm:text-sm text-slate-400 leading-relaxed max-w-3xl">
                    Fetches raw inputs from public financial data providers and news networks. We maintain a local cache database to satisfy rate limits and reduce latency.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-white/[0.04]">
                    <div className="space-y-2">
                      <span className="text-xs font-bold text-slate-350 uppercase tracking-wider block font-mono">1. Market Price Series</span>
                      <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
                        Pulls 2 years of daily bar history (Open, High, Low, Close, Volume) using yfinance. Timestamps are formatted using local exchange timezone offsets (+19800 seconds for NSE) to display accurate quote dates next to LTP.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-bold text-slate-350 uppercase tracking-wider block font-mono">2. News Stream feeds</span>
                      <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
                        Monitors RSS feeds and articles. The text contents are extracted, stripped of HTML formatting, and queued for sentiment parsing.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {selectedPipelineStep === 'features' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <Layers className="h-6 w-6 text-indigo-400" />
                    <h3 className="text-base sm:text-lg font-extrabold text-white">02. Quantitative Feature Engineering</h3>
                  </div>
                  <p className="text-xs sm:text-sm text-slate-450 text-slate-400 leading-relaxed font-sans">
                    Raw market bars are transformed into a 40+ element feature vector representing mathematical momentum, volatility ratios, volume changes, and trend alignment.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-4 border-t border-white/[0.04]">
                    <div className="space-y-3">
                      <span className="text-xs font-bold uppercase text-slate-400 tracking-wider font-mono">Primary Features Engaged</span>
                      <ul className="grid grid-cols-2 gap-y-2 text-xs text-slate-400 font-mono">
                        <li>• RSI (14) Momentum</li>
                        <li>• MACD Signal / Hist</li>
                        <li>• Stochastic %K / %D</li>
                        <li>• EMA Cross (9/21d)</li>
                        <li>• Rolling Skewness</li>
                        <li>• ATR Volatility Ratio</li>
                        <li>• Bollinger Band Pos</li>
                        <li>• OBV Volume Ratio</li>
                      </ul>
                    </div>
                    <div className="space-y-3">
                      <span className="text-xs font-bold uppercase text-slate-400 tracking-wider font-mono">News NLP Sentiment Scoring</span>
                      <p className="text-xs sm:text-sm text-slate-450 text-slate-400 leading-relaxed font-sans">
                        Articles are analyzed using a text polarity engine. The sentiment score ranges from **-1.0** (extreme negative reputation risk) to **+1.0** (positive catalyst), providing a sentiment weight used in model adjustments.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {selectedPipelineStep === 'hmm' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <Activity className="h-6 w-6 text-indigo-400" />
                    <h3 className="text-base sm:text-lg font-extrabold text-white">03. HMM Market Regime Detection</h3>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <div className="lg:col-span-5 space-y-4">
                      <p className="text-xs sm:text-sm text-slate-450 text-slate-400 leading-relaxed font-sans">
                        To detect structural shifts in market volatility, we fit a **3-State Gaussian Hidden Markov Model (HMM)** directly to historical index returns.
                      </p>
                      <div className="space-y-3 text-xs sm:text-sm text-slate-400 font-sans pt-2">
                        <div className="flex justify-between border-b border-white/[0.04] pb-2">
                          <span className="font-bold text-slate-200">State 0 (Low Volatility)</span>
                          <span className="text-indigo-400 font-bold">Buy @ 1.8% threshold</span>
                        </div>
                        <div className="flex justify-between border-b border-white/[0.04] pb-2">
                          <span className="font-bold text-slate-200">State 1 (Sideways Market)</span>
                          <span className="text-indigo-400 font-bold">Buy @ 2.2% threshold</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="font-bold text-slate-200">State 2 (High Volatility Panic)</span>
                          <span className="text-indigo-400 font-bold">Buy @ 3.0% threshold</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* SVG Transition Network Diagram (Interactive click to switch active state) */}
                    <div className="lg:col-span-7 bg-black/40 p-5 rounded-xl border border-white/[0.05] flex flex-col justify-between">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-xs font-bold text-slate-350 uppercase tracking-widest font-mono block">Regime Transition Network (Click nodes to test)</span>
                        <span className="text-[10px] text-slate-500 font-mono">P_ii persistence</span>
                      </div>
                      
                      <div className="flex justify-around items-center h-[120px] relative border border-white/[0.03] rounded-lg bg-black/20">
                        {/* State 0 */}
                        <button 
                          onClick={() => setSimRegime('0')}
                          className={`w-14 h-14 rounded-full border-2 flex flex-col items-center justify-center font-mono text-[10px] cursor-pointer transition duration-300 ${
                            simRegime === '0' ? 'border-emerald-500 bg-emerald-500/10 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.2)]' : 'border-slate-700 bg-transparent text-slate-500 hover:border-slate-500 hover:text-slate-300'
                          }`}
                        >
                          <span>State 0</span>
                          <span className="font-bold">Bullish</span>
                        </button>

                        {/* State 1 */}
                        <button 
                          onClick={() => setSimRegime('1')}
                          className={`w-14 h-14 rounded-full border-2 flex flex-col items-center justify-center font-mono text-[10px] cursor-pointer transition duration-300 ${
                            simRegime === '1' ? 'border-amber-500 bg-amber-500/10 text-amber-300 shadow-[0_0_15px_rgba(245,158,11,0.2)]' : 'border-slate-700 bg-transparent text-slate-500 hover:border-slate-500 hover:text-slate-300'
                          }`}
                        >
                          <span>State 1</span>
                          <span className="font-bold">Sideways</span>
                        </button>

                        {/* State 2 */}
                        <button 
                          onClick={() => setSimRegime('2')}
                          className={`w-14 h-14 rounded-full border-2 flex flex-col items-center justify-center font-mono text-[10px] cursor-pointer transition duration-300 ${
                            simRegime === '2' ? 'border-rose-500 bg-rose-500/10 text-rose-300 shadow-[0_0_15px_rgba(244,63,94,0.2)]' : 'border-slate-700 bg-transparent text-slate-500 hover:border-slate-500 hover:text-slate-300'
                          }`}
                        >
                          <span>State 2</span>
                          <span className="font-bold">Panic</span>
                        </button>
                      </div>

                      <p className="text-xs text-slate-450 mt-3 font-sans leading-relaxed">
                        💡 **Interactive feature**: Click nodes directly to trigger regime transitions. Notice how it instantly shifts the evaluation thresholds and tranche weights in the sandbox.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {selectedPipelineStep === 'ensemble' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <Brain className="h-6 w-6 text-indigo-400" />
                    <h3 className="text-base sm:text-lg font-extrabold text-white">04. Ensemble Prediction Stack &amp; L2 regularized weight Shrinkage</h3>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <div className="lg:col-span-5 space-y-5">
                      <span className="text-xs font-bold text-slate-350 uppercase tracking-widest font-mono block">Ridge Meta-Learner Regularization</span>
                      <p className="text-xs sm:text-sm text-slate-450 text-slate-400 leading-relaxed font-sans">
                        Tree models are highly correlated. Ridge Meta-learning uses L2 penalty coefficients ($\lambda$) to control multicollinearity. Adjust the slider to see how L2 penalty shrinks base model weights towards equal weighting (0.20):
                      </p>
                      
                      {/* Ridge Lambda Slider */}
                      <div className="space-y-2 bg-black/40 p-4 border border-white/[0.05] rounded-xl">
                        <div className="flex justify-between items-center">
                          <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider font-mono">Ridge L2 Penalty (λ): {ridgeLambda}</label>
                          <span className="text-[10px] text-indigo-400 font-mono">Regularization</span>
                        </div>
                        <input 
                          type="range"
                          min="0.0"
                          max="10.0"
                          step="0.5"
                          value={ridgeLambda}
                          onChange={e => setRidgeLambda(parseFloat(e.target.value))}
                          className="w-full h-1 bg-white/[0.08] rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>
                    </div>

                    {/* Chart: Ensemble Stack output comparison */}
                    <div className="lg:col-span-7 bg-black/40 p-5 rounded-xl border border-white/[0.05]">
                      <span className="text-xs font-bold text-slate-350 uppercase tracking-widest font-mono block mb-4">Meta-Learner Weights Allocation (Closed-Form Shrinkage)</span>
                      <div className="h-[170px] w-full">
                        {isMounted ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={ridgeWeightsData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                              <XAxis dataKey="name" stroke="rgba(255,255,255,0.3)" fontSize={10} />
                              <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} domain={[-0.05, 0.5]} />
                              <Tooltip content={<CustomTooltip formatter={(val) => `Weight: ${val}`} />} />
                              <ReferenceLine y={0.20} stroke="rgba(99,102,241,0.3)" strokeDasharray="3 3" label={{ value: 'Equal consensus (1/N)', fill: 'rgba(99,102,241,0.5)', position: 'insideTopRight', fontSize: 9 }} />
                              <Bar dataKey="Weight" fill="#6366f1" radius={[4, 4, 0, 0]}>
                                {ridgeWeightsData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.Weight >= 0.20 ? '#10b981' : '#f59e0b'} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="h-full flex items-center justify-center text-xs text-slate-500 font-mono">
                            Loading Chart Canvas...
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {selectedPipelineStep === 'risk' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <LineChartIcon className="h-6 w-6 text-indigo-400" />
                    <h3 className="text-base sm:text-lg font-extrabold text-white">05. GARCH(1,1) Volatility Forecasting &amp; Mean Reversion</h3>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    <div className="lg:col-span-5 space-y-4">
                      <span className="text-xs font-bold text-slate-350 uppercase tracking-widest font-mono block">GARCH(1,1) Volatility Shock</span>
                      <p className="text-xs sm:text-sm text-slate-400 leading-relaxed font-sans">
                        Simulate a recent market price shock (ε_t-1) to watch how conditional variance immediately spikes and gradually decays back to its long-term baseline (16% mean).
                      </p>
                      
                      {/* GARCH Shock slider */}
                      <div className="space-y-2 bg-black/40 p-4 border border-white/[0.05] rounded-xl">
                        <div className="flex justify-between items-center">
                          <label className="text-[11px] font-bold text-slate-400 uppercase tracking-wider font-mono">Shock Multiplier: {garchShock}x</label>
                          <span className="text-[10px] text-indigo-400 font-mono">ε_(t-1) term</span>
                        </div>
                        <input 
                          type="range"
                          min="0.5"
                          max="3.0"
                          step="0.1"
                          value={garchShock}
                          onChange={e => setGarchShock(parseFloat(e.target.value))}
                          className="w-full h-1 bg-white/[0.08] rounded-lg appearance-none cursor-pointer accent-indigo-500"
                        />
                      </div>
                    </div>

                    {/* Chart: GARCH Volatility Decay Curve */}
                    <div className="lg:col-span-7 bg-black/40 p-5 rounded-xl border border-white/[0.05]">
                      <span className="text-xs font-bold text-slate-350 uppercase tracking-widest font-mono block mb-4">5-Day Forecast decay curve (σ_t)</span>
                      <div className="h-[170px] w-full">
                        {isMounted ? (
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={garchData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                              <XAxis dataKey="day" stroke="rgba(255,255,255,0.3)" fontSize={10} />
                              <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} unit="%" />
                              <Tooltip content={<CustomTooltip formatter={(val) => `${val}% forecasted volatility`} />} />
                              <ReferenceLine y={16.0} stroke="rgba(244,63,94,0.3)" strokeDasharray="3 3" label={{ value: 'Long-term Mean', fill: 'rgba(244,63,94,0.5)', position: 'insideBottomRight', fontSize: 9 }} />
                              <Line type="monotone" dataKey="Volatility" stroke="#818cf8" strokeWidth={2} dot={{ fill: '#818cf8', r: 4 }} />
                            </LineChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="h-full flex items-center justify-center text-xs text-slate-500 font-mono">
                            Loading Chart Canvas...
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              )}

            </div>
          </div>
        )}

        {/* VIEW 3: INTERVIEW FAQS & QUANT TRADE-OFFS */}
        {activeView === 'faq' && (
          <div className="space-y-12 sm:space-y-16 animate-fadeIn">
            
            <div className="max-w-2xl text-left space-y-2">
              <span className="text-xs uppercase tracking-wider font-extrabold text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded border border-indigo-500/25">Quant Interview prep</span>
              <p className="text-sm sm:text-base text-slate-450 text-slate-400 leading-relaxed mt-2">
                Key mathematical trade-offs and structural questions commonly asked during quantitative and machine learning engineering interviews.
              </p>
            </div>

            {/* Questions list */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-4">
              {[
                {
                  q: 'How did you prevent data leakage in the ML ensemble prediction?',
                  a: 'Data leakage was avoided by utilizing walk-forward time-series validation rather than standard random cross-validation. Since prices are highly path-dependent, random splits introduce future leakage. The models are trained on chronological folds, ensuring that at any point of prediction, only past indicators are used.'
                },
                {
                  q: 'Why use GARCH(1,1) over simple rolling historical volatility?',
                  a: 'Historical volatility assumes constant variance over time. However, financial markets exhibit volatility clustering (large changes tend to be followed by large changes). GARCH(1,1) models this conditional variance dynamically, allowing us to generate a forward-looking risk estimate for the next 5 days.'
                },
                {
                  q: 'How does the Ridge meta-learner address multicollinearity?',
                  a: 'The 5 base models (Random Forest, XGBoost, etc.) are trained on similar indicator spaces, making their forecasts highly correlated. If combined using simple averages, this creates multicollinearity. The Ridge meta-learner uses an L2 regularization penalty to bound the weights, preventing model inflation.'
                },
                {
                  q: 'How did you handle structural shifts in stock behavior?',
                  a: 'We implemented a Gaussian Hidden Markov Model (HMM) to classify market state into three distinct volatility regimes. The prediction engine uses these regimes to scale its target thresholds. During panic regimes (State 2), required expected return thresholds increase, filtering out weak signals.'
                }
              ].map((faq, idx) => (
                <div key={idx} className="border-l-2 border-white/[0.08] hover:border-indigo-500/40 pl-5 py-2 space-y-2 transition duration-300">
                  <span className="text-xs sm:text-sm font-black text-white uppercase tracking-wider block font-sans">Q: {faq.q}</span>
                  <p className="text-xs sm:text-sm text-slate-400 leading-relaxed font-sans mt-2">{faq.a}</p>
                </div>
              ))}
            </div>

            {/* NEW ADDITION: Walk-Forward Validation Backtest Simulator */}
            <div className="border border-white/[0.06] p-5 sm:p-8 rounded-3xl bg-white/[0.005] space-y-6 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-36 h-36 bg-emerald-500/[0.02] blur-[80px] rounded-full pointer-events-none" />
              
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Terminal className="h-5 w-5 text-indigo-400" />
                    <h3 className="text-sm sm:text-base font-extrabold text-white font-sans uppercase tracking-wider">Walk-Forward Backtest Simulator</h3>
                  </div>
                  <p className="text-xs text-slate-500">Run a chronological walk-forward simulation to evaluate strategy outperformance.</p>
                </div>
                <button
                  onClick={runBacktest}
                  disabled={isBacktesting}
                  className="px-5 py-2.5 rounded-full text-xs font-black bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center gap-2 cursor-pointer transition shrink-0 disabled:opacity-50"
                >
                  <RefreshCw className={`h-3.5 w-3.5 ${isBacktesting ? 'animate-spin' : ''}`} />
                  {isBacktesting ? 'Running Backtest...' : 'Run Backtest Matrix'}
                </button>
              </div>

              {/* Progress bar and logger console */}
              {isBacktesting && (
                <div className="space-y-3 animate-fadeIn">
                  <div className="h-1 w-full bg-white/[0.08] rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-500 transition-all duration-150" style={{ width: `${backtestProgress}%` }} />
                  </div>
                  <div className="bg-black/80 border border-white/[0.06] rounded-xl p-4 font-mono text-[10px] text-emerald-400 space-y-1 max-h-[140px] overflow-y-auto no-scrollbar">
                    {backtestLogs.map((log, index) => (
                      <div key={index} className="leading-relaxed">{log}</div>
                    ))}
                  </div>
                </div>
              )}

              {/* Backtest Result Area Chart */}
              {showBacktestResult && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 pt-4 border-t border-white/[0.06] animate-fadeIn">
                  <div className="lg:col-span-4 space-y-4">
                    <div className="bg-emerald-500/10 border border-emerald-500/20 p-4 rounded-2xl">
                      <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest font-mono block">Outperformance</span>
                      <span className="text-3xl font-black text-emerald-350 block mt-1 font-mono">+72.0%</span>
                      <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                        The Stacked ML strategy Outperformed Buy &amp; Hold index by 72% over 4 validation folds, mitigating drawdown during panic regimes.
                      </p>
                    </div>
                  </div>

                  <div className="lg:col-span-8 h-[220px] w-full">
                    {isMounted ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={backtestChartData}>
                          <defs>
                            <linearGradient id="colorMl" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#818cf8" stopOpacity={0.25} />
                              <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="colorIndex" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.1} />
                              <stop offset="95%" stopColor="#94a3b8" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                          <XAxis dataKey="year" stroke="rgba(255,255,255,0.3)" fontSize={10} />
                          <YAxis stroke="rgba(255,255,255,0.3)" fontSize={10} domain={[80, 240]} />
                          <Tooltip content={<CustomTooltip formatter={(val) => `₹${val.toFixed(0)} (Base ₹100)`} />} />
                          <Area type="monotone" dataKey="Stacked ML Return" stroke="#818cf8" strokeWidth={2.5} fillOpacity={1} fill="url(#colorMl)" />
                          <Area type="monotone" dataKey="Buy & Hold Return" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" fillOpacity={1} fill="url(#colorIndex)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    ) : null}
                  </div>
                </div>
              )}
            </div>

            {/* Multi-Factor Decisions & Metric Matrix Table */}
            <div className="space-y-6 pt-4">
              <h3 className="text-sm sm:text-base font-extrabold text-white font-sans uppercase tracking-wider">Multi-Factor decisions &amp; Metrics Summary</h3>
              <div className="overflow-x-auto border border-white/[0.04] rounded-2xl bg-white/[0.005]">
                <table className="w-full text-left text-xs sm:text-sm border-collapse min-w-[600px]">
                  <thead>
                    <tr className="border-b border-white/[0.04] bg-white/[0.01] text-slate-400">
                      <th className="p-4 font-bold uppercase tracking-wider text-xs">Factor Category</th>
                      <th className="p-4 font-bold uppercase tracking-wider text-xs">Metrics Used</th>
                      <th className="p-4 font-bold uppercase tracking-wider text-xs">Strategic Role</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.03] text-slate-350">
                    {[
                      { cat: 'Macro Regimes', metric: '3-State Gaussian HMM', role: 'Detects structural regime changes to scale expected return thresholds and mitigate catching falling knives.' },
                      { cat: 'Consensus Momentum', metric: 'Stacked Ensemble & Ridge meta-regressor', role: 'Synthesizes non-linear indicators to estimate trend direction and consensus confidence.' },
                      { cat: 'Feature Attribution', metric: 'SHAP Shapley Values', role: 'Decomposes model predictions to attribute them back to specific alpha features, preventing black-box risks.' },
                      { cat: 'Volatility Forecast', metric: 'GARCH(1,1) Annualized Variance', role: 'Forecasts 5-day conditional volatility clustering to calibrate stop-losses and option Greeks.' },
                      { cat: 'Downside Sizing', metric: '95%/99% Value-at-Risk (VaR)', role: 'Measures extreme daily loss boundaries to restrict absolute capital exposure.' }
                    ].map(row => (
                      <tr key={row.cat} className="hover:bg-white/[0.005] transition duration-150">
                        <td className="p-4 font-black text-slate-200 font-mono tracking-tight text-xs sm:text-sm">{row.cat}</td>
                        <td className="p-4 font-mono text-xs sm:text-sm text-indigo-300">{row.metric}</td>
                        <td className="p-4 text-xs sm:text-sm text-slate-400 leading-relaxed">{row.role}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-white/[0.03] py-12 bg-black/60 backdrop-blur-sm z-10">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="h-6 w-6 bg-white rounded-md flex items-center justify-center">
              <TrendingUp className="h-3.5 w-3.5 text-black" />
            </div>
            <span className="text-xs font-bold text-white">StockIQ Pro</span>
          </div>
          <p className="text-xs text-slate-600">
            © 2026 StockIQ Pro · Created by Vishesh Sanghvi. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
