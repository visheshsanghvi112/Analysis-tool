'use client';

import { useState, useCallback } from 'react';
import {
  Wallet, Clock, Zap, Target, TrendingDown, ArrowRight,
  ChevronDown, ChevronUp, AlertTriangle, CheckCircle2,
  BarChart3, ShieldCheck, Lightbulb, X, Sparkles, Info, Eye
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

const fmt    = (n, d = 2) => n == null ? 'N/A' : Number(n).toFixed(d);
const fmtINR = (n) => n == null ? 'N/A' : `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

// ── Horizon presets ────────────────────────────────────────────────────────
const HORIZON_PRESETS = [
  { label: '1 Month',   days: 30,  icon: '⚡', desc: 'Quick bounce play' },
  { label: '3 Months',  days: 90,  icon: '📅', desc: 'Short-term recovery' },
  { label: '6 Months',  days: 180, icon: '📊', desc: 'Medium-term hold' },
  { label: '1 Year',    days: 365, icon: '🎯', desc: 'Wealth building' },
  { label: '2+ Years',  days: 730, icon: '🌱', desc: 'Long-term SIP' },
];

// ── Signal chip ────────────────────────────────────────────────────────────
function SignalChip({ signal }) {
  const map = {
    STRONG_BUY:  { bg: 'bg-emerald-500/15 border-emerald-500/30', text: 'text-emerald-400', label: 'Strong Buy' },
    ACCUMULATE:  { bg: 'bg-indigo-500/15  border-indigo-500/30',  text: 'text-indigo-400',  label: 'Accumulate' },
    WAIT:        { bg: 'bg-amber-500/15   border-amber-500/30',   text: 'text-amber-400',   label: 'Wait' },
  };
  const s = map[signal] || map.WAIT;
  return (
    <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${s.bg} ${s.text} uppercase tracking-wider`}>
      {s.label}
    </span>
  );
}

// ── Confidence bar ─────────────────────────────────────────────────────────
function ConfidenceBar({ pct, label }) {
  const color = label === 'HIGH' ? '#10b981' : label === 'MODERATE' ? '#6366f1' : '#f59e0b';
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-bold" style={{ color }}>{pct}%</span>
    </div>
  );
}

// ── Suggestion card ────────────────────────────────────────────────────────
function SuggestionCard({ s, rank }) {
  const [expanded, setExpanded] = useState(rank === 1);
  const colorMap = {
    emerald: { border: 'border-emerald-500/25', bg: 'bg-emerald-500/8',  glow: 'shadow-emerald-500/10', badge: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25' },
    indigo:  { border: 'border-indigo-500/25',  bg: 'bg-indigo-500/8',   glow: 'shadow-indigo-500/10',  badge: 'bg-indigo-500/15  text-indigo-400  border-indigo-500/25' },
    amber:   { border: 'border-amber-500/25',   bg: 'bg-amber-500/8',    glow: 'shadow-amber-500/10',   badge: 'bg-amber-500/15   text-amber-400   border-amber-500/25' },
    slate:   { border: 'border-white/[0.08]',   bg: 'bg-white/[0.02]',   glow: '',                       badge: 'bg-white/[0.06]  text-slate-400   border-white/[0.10]' },
  };
  const c   = colorMap[s.priority_color] || colorMap.slate;
  const sym = s.ticker.replace('.NS', '').replace('.BO', '');

  return (
    <div className={`rounded-2xl border ${c.border} ${c.bg} shadow-lg ${c.glow} overflow-hidden transition-all duration-300`}>
      {/* Header */}
      <div
        className="flex items-center gap-3 px-4 py-3.5 cursor-pointer select-none"
        onClick={() => setExpanded(v => !v)}
      >
        {/* Rank badge */}
        <div className="h-9 w-9 rounded-xl bg-white/[0.06] border border-white/[0.08] flex items-center justify-center shrink-0">
          <span className="text-[11px] font-black text-white">{sym.slice(0, 3)}</span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <p className="text-sm font-black text-white">{sym}</p>
            <SignalChip signal={s.signal} />
            <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${c.badge}`}>
              {s.priority_label}
            </span>
            {s.sector && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-violet-500/10 border-violet-500/20 text-violet-300">
                📁 {s.sector}
              </span>
            )}
          </div>
          <p className={`text-[10px] font-bold mt-0.5 ${
            s.current_pnl_pct < 0 ? 'text-rose-400' : s.current_pnl_pct > 0 ? 'text-emerald-400' : 'text-slate-500'
          }`}>
            {s.current_pnl_pct === 0 ? 'Fresh Entry Option' : `${s.current_pnl_pct > 0 ? '+' : ''}${fmt(s.current_pnl_pct)}% current P&L`}
          </p>
        </div>

        <div className="text-right shrink-0">
          <p className="text-base font-black text-white">{fmtINR(s.allocated_amount)}</p>
          <p className="text-[9px] text-slate-500">{s.shares_to_buy} shares · {s.allocation_weight_pct}%</p>
        </div>

        <div className="shrink-0 text-slate-600">
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-white/[0.04]">

          {/* Action text */}
          <p className="text-[11px] text-slate-300 leading-relaxed pt-3">{s.action_text}</p>

          {/* Metrics grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {s.current_pnl_pct === 0 ? (
              [
                { label: 'Buy @ Price',     value: `₹${s.live_price.toLocaleString('en-IN')}`, sub: 'current LTP' },
                { label: '1M Momentum',     value: `${s.momentum_1m > 0 ? '+' : ''}${fmt(s.momentum_1m)}%`, sub: 'price change 30d' },
                { label: 'P/E Ratio',       value: s.pe_ratio ? `${fmt(s.pe_ratio, 1)}` : 'N/A', sub: s.sector || 'Sector benchmark' },
                { label: 'Sentiment',       value: s.sentiment > 0 ? 'Bullish' : s.sentiment < 0 ? 'Bearish' : 'Neutral', sub: `score: ${fmt(s.sentiment, 2)}` },
              ].map(m => (
                <div key={m.label} className="rounded-xl bg-white/[0.04] border border-white/[0.06] p-2.5 text-center">
                  <p className="text-[8px] text-slate-400 uppercase tracking-wider mb-1">{m.label}</p>
                  <p className="text-[12px] font-black text-white">{m.value}</p>
                  <p className="text-[9px] text-slate-400 mt-0.5">{m.sub}</p>
                </div>
              ))
            ) : (
              [
                { label: 'Buy @ Price',     value: `₹${s.live_price.toLocaleString('en-IN')}`, sub: 'current LTP' },
                { label: 'New Avg Cost',    value: `₹${s.new_avg_price.toLocaleString('en-IN')}`, sub: `was ₹${s.buy_price.toLocaleString('en-IN')}` },
                { label: 'Break-Even Now',  value: `+${fmt(s.new_gain_to_be)}%`, sub: `was +${fmt(s.current_gain_to_be)}%` },
                { label: 'Cost Reduced By', value: `${fmt(s.avg_cost_reduction_pct)}%`, sub: 'avg cost reduction' },
              ].map(m => (
                <div key={m.label} className="rounded-xl bg-white/[0.04] border border-white/[0.06] p-2.5 text-center">
                  <p className="text-[8px] text-slate-400 uppercase tracking-wider mb-1">{m.label}</p>
                  <p className="text-[12px] font-black text-white">{m.value}</p>
                  <p className="text-[9px] text-slate-400 mt-0.5">{m.sub}</p>
                </div>
              ))
            )}
          </div>

          {/* Signal badges row */}
          <div className="flex flex-wrap gap-1.5">
            {s.ml_propensity_score != null && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-violet-500/15 border-violet-500/30 text-violet-300 font-mono">
                AI Buy Probability: {s.ml_propensity_score}%
              </span>
            )}
            {s.rsi != null && (
              <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${
                s.rsi < 35 ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
                : s.rsi > 65 ? 'bg-rose-500/15 border-rose-500/30 text-rose-400'
                : 'bg-white/[0.05] border-white/[0.10] text-slate-400'
              }`}>RSI {s.rsi}</span>
            )}
            {s.macd_crossover && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-emerald-500/15 border-emerald-500/30 text-emerald-400">MACD ✓</span>
            )}
            {s.rsi_divergence && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-indigo-500/15 border-indigo-500/30 text-indigo-400">RSI Div ✓</span>
            )}
            {s.above_ema200 === true && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-emerald-500/15 border-emerald-500/30 text-emerald-400">Above 200 EMA ✓</span>
            )}
            {s.above_ema200 === false && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-rose-500/15 border-rose-500/30 text-rose-400">Below 200 EMA ⚠</span>
            )}
            {s.vol_spike && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-amber-500/15 border-amber-500/30 text-amber-400">Vol Spike ✓</span>
            )}
            {s.pe_ratio && (
              <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-white/[0.05] border-white/[0.10] text-slate-400">P/E {fmt(s.pe_ratio, 1)}</span>
            )}
            <span className="text-[9px] px-2 py-0.5 rounded-full border bg-white/[0.04] border-white/[0.08] text-slate-400">
               {s.current_pnl_pct === 0 ? 'Fresh Entry Pick' : `Recovery: ${s.estimated_recovery}`}
            </span>
          </div>

          {/* Tranche entry plan */}
          {s.tranches?.length > 0 && (
            <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-3">
              <p className="text-[9px] font-bold text-slate-300 uppercase tracking-wider mb-2">📋 Staggered Entry Plan (Industry Best Practice)</p>
              <div className="space-y-1.5">
                {s.tranches.map(t => (
                  <div key={t.tranche} className="flex items-center gap-2 text-[10px]">
                    <span className="h-5 w-5 rounded-full bg-violet-500/20 border border-violet-500/30 flex items-center justify-center text-[8px] font-black text-violet-400 shrink-0">{t.tranche}</span>
                    <span className="text-slate-400 flex-1">{t.condition}</span>
                    <span className="text-white font-bold shrink-0">{fmtINR(t.amount)} · {t.shares}sh</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Confidence bar */}
          <div>
            <div className="flex justify-between text-[9px] text-slate-400 mb-1">
              <span>Signal Confidence</span>
              <span className="uppercase font-bold">{s.confidence_label}</span>
            </div>
            <ConfidenceBar pct={s.confidence_pct} label={s.confidence_label} />
          </div>
        </div>
      )}
    </div>
  );
}

// ── Input panel (the "wizard") ─────────────────────────────────────────────
function InputPanel({ holdings, onResult, loading, setLoading }) {
  const [mode, setMode]         = useState('recovery'); // 'recovery' or 'market_buys'
  const [capital, setCapital]   = useState('');
  const [horizon, setHorizon]   = useState(null);
  const [customDays, setCustom] = useState('');
  const [maxPrice, setMaxPrice] = useState('Any');
  const [selectedSector, setSelectedSector] = useState('All');
  const [error, setError]       = useState(null);

  const selectedDays = horizon?.days || (customDays ? parseInt(customDays) : null);
  const hasHoldings = holdings && holdings.length > 0;

  const run = useCallback(async () => {
    if (!capital || !selectedDays) return;
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/capital-allocate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          holdings: mode === 'recovery' ? holdings.map(h => ({
            ticker: h.ticker.trim().toUpperCase(),
            qty: +h.qty,
            buy_price: +h.buy_price,
          })) : [],
          floating_capital: parseFloat(capital),
          horizon_days: selectedDays,
          mode: mode,
          max_stock_price: maxPrice === 'Any' ? null : parseFloat(maxPrice),
          sector: selectedSector === 'All' ? null : selectedSector,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Allocation failed');
      onResult(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [capital, selectedDays, holdings, mode, maxPrice, selectedSector, onResult, setLoading]);

  return (
    <div className="space-y-5">
      {/* Mode Toggle */}
      <div className="flex gap-2 p-1 bg-white/[0.02] border border-white/[0.08] rounded-xl shrink-0">
        <button
          onClick={() => setMode('recovery')}
          className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold transition flex items-center justify-center gap-1.5 cursor-pointer ${
            mode === 'recovery'
              ? 'bg-violet-500/20 border border-violet-500/30 text-violet-300'
              : 'text-slate-400 hover:text-slate-200 border border-transparent'
          }`}
        >
          <TrendingDown className="h-3.5 w-3.5" />
          Recover Positions
        </button>
        <button
          onClick={() => setMode('market_buys')}
          className={`flex-1 py-2 px-3 rounded-lg text-xs font-bold transition flex items-center justify-center gap-1.5 cursor-pointer ${
            mode === 'market_buys'
              ? 'bg-violet-500/20 border border-violet-500/30 text-violet-300'
              : 'text-slate-400 hover:text-slate-200 border border-transparent'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          Fresh Market Buys
        </button>
      </div>

      {mode === 'recovery' && !hasHoldings && (
        <div className="flex items-start gap-2 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p className="font-bold">No holdings detected</p>
            <p className="text-slate-400 mt-0.5">Please add stocks to your Portfolio Tracker first, or switch to <strong>Fresh Market Buys</strong> to find top market candidates.</p>
          </div>
        </div>
      )}

      {/* Capital input */}
      <div>
        <label className="block text-[10px] font-bold text-slate-200 uppercase tracking-wider mb-2">
          <Wallet className="inline h-3 w-3 mr-1" /> Available Floating Capital (₹)
        </label>
        <div className="flex items-center gap-2 px-3 py-2.5 bg-white/[0.04] border border-white/[0.10] rounded-xl focus-within:border-violet-500/50 transition">
          <span className="text-slate-400 text-sm font-bold">₹</span>
          <input
            type="number"
            min="1"
            value={capital}
            onChange={e => setCapital(e.target.value)}
            placeholder="e.g. 50000"
            className="bg-transparent text-white text-sm flex-1 outline-none placeholder-slate-500"
          />
        </div>
        {/* Quick amount chips */}
        <div className="flex flex-wrap gap-2 mt-2">
          {[10000, 25000, 50000, 100000, 250000].map(amt => (
            <button
              key={amt}
              onClick={() => setCapital(String(amt))}
              className={`text-[10px] px-2.5 py-1 rounded-lg border transition cursor-pointer ${
                capital === String(amt)
                  ? 'bg-violet-500/20 border-violet-500/40 text-violet-300'
                  : 'bg-white/[0.03] border-white/[0.08] text-slate-400 hover:text-slate-200'
              }`}
            >
              {fmtINR(amt)}
            </button>
          ))}
        </div>
      </div>

      {/* Filters Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Sector Filter */}
        <div>
          <label className="block text-[10px] font-bold text-slate-200 uppercase tracking-wider mb-2">
            📁 Category / Sector
          </label>
          <div className="relative">
            <select
              value={selectedSector}
              onChange={e => setSelectedSector(e.target.value)}
              className="w-full px-3 py-2 bg-white/[0.04] border border-white/[0.10] rounded-xl text-xs text-white outline-none focus:border-violet-500/50 appearance-none cursor-pointer"
            >
              {['All', 'Banking', 'IT', 'FMCG', 'Metals', 'Infrastructure', 'Auto', 'Pharma', 'Energy', 'Power', 'NBFC', 'Telecom'].map(sec => (
                <option key={sec} value={sec} className="bg-[#0a0a0f] text-white">
                  {sec === 'All' ? 'All Sectors' : sec}
                </option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-400">
              <ChevronDown className="h-3.5 w-3.5" />
            </div>
          </div>
        </div>

        {/* Price Limit Filter */}
        <div>
          <label className="block text-[10px] font-bold text-slate-200 uppercase tracking-wider mb-2">
            💰 Max Stock Price
          </label>
          <div className="relative">
            <select
              value={maxPrice}
              onChange={e => setMaxPrice(e.target.value)}
              className="w-full px-3 py-2 bg-white/[0.04] border border-white/[0.10] rounded-xl text-xs text-white outline-none focus:border-violet-500/50 appearance-none cursor-pointer"
            >
              <option value="Any" className="bg-[#0a0a0f] text-white">Any Price</option>
              <option value="100" className="bg-[#0a0a0f] text-white">Under ₹100</option>
              <option value="200" className="bg-[#0a0a0f] text-white">Under ₹200</option>
              <option value="500" className="bg-[#0a0a0f] text-white">Under ₹500</option>
              <option value="1000" className="bg-[#0a0a0f] text-white">Under ₹1,000</option>
              <option value="2000" className="bg-[#0a0a0f] text-white">Under ₹2,000</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-slate-400">
              <ChevronDown className="h-3.5 w-3.5" />
            </div>
          </div>
        </div>
      </div>

      {/* Horizon selector */}
      <div>
        <label className="block text-[10px] font-bold text-slate-200 uppercase tracking-wider mb-2">
          <Clock className="inline h-3 w-3 mr-1" /> Investment Horizon
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {HORIZON_PRESETS.map(h => (
            <button
              key={h.days}
              onClick={() => { setHorizon(h); setCustom(''); }}
              className={`flex flex-col items-center gap-0.5 py-2.5 px-2 rounded-xl border text-center transition cursor-pointer ${
                horizon?.days === h.days
                  ? 'bg-violet-600/20 border-violet-500/40 text-violet-300'
                  : 'bg-white/[0.03] border-white/[0.07] text-slate-350 hover:border-white/[0.15] hover:text-slate-200'
              }`}
            >
              <span className="text-lg leading-none">{h.icon}</span>
              <span className="text-[11px] font-bold mt-1">{h.label}</span>
              <span className="text-[9px] text-slate-400">{h.desc}</span>
            </button>
          ))}
        </div>
        {/* Custom days */}
        <div className="flex items-center gap-2 mt-2">
          <input
            type="number"
            min="1"
            value={customDays}
            onChange={e => { setCustom(e.target.value); setHorizon(null); }}
            placeholder="Or enter custom days…"
            className="flex-1 px-3 py-1.5 bg-white/[0.03] border border-white/[0.07] rounded-lg text-xs text-white placeholder-slate-500 outline-none focus:border-violet-500/40"
          />
          {customDays && <span className="text-[10px] text-slate-400">= {Math.round(parseInt(customDays)/30)} months</span>}
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs">
          <AlertTriangle className="h-4 w-4 shrink-0" />{error}
        </div>
      )}

      <button
        onClick={run}
        disabled={loading || !capital || !selectedDays || (mode === 'recovery' && !hasHoldings)}
        className="w-full flex items-center justify-center gap-2 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 disabled:opacity-40 text-white text-sm font-bold rounded-xl transition-all duration-200 cursor-pointer shadow-lg shadow-violet-500/20"
      >
        <Sparkles className={`h-4 w-4 ${loading ? 'animate-pulse' : ''}`} />
        {loading ? 'Analysing Portfolio & Markets…' : 'Generate Smart Allocation Plan'}
      </button>
    </div>
  );
}

// ── Results panel ──────────────────────────────────────────────────────────
function ResultsPanel({ data, onReset }) {
  const s = data.summary;

  const groupedSuggestions = (data.suggestions || []).reduce((acc, sug) => {
    const sector = sug.sector || 'Others';
    if (!acc[sector]) acc[sector] = [];
    acc[sector].push(sug);
    return acc;
  }, {});

  if (data.no_loss_positions) {
    return (
      <div className="text-center py-8 space-y-3">
        <CheckCircle2 className="h-12 w-12 text-emerald-400 mx-auto" />
        <p className="text-base font-bold text-white">All Positions in Profit!</p>
        <p className="text-sm text-slate-400">{data.message}</p>
        <button onClick={onReset} className="mt-3 text-xs text-violet-400 hover:text-violet-300 underline cursor-pointer">
          Recalculate
        </button>
      </div>
    );
  }

  const confColor = s.overall_confidence === 'HIGH' ? 'text-emerald-400' : s.overall_confidence === 'MODERATE' ? 'text-indigo-400' : 'text-amber-400';

  return (
    <div className="space-y-5">
      {/* Summary banner */}
      <div className="rounded-2xl bg-gradient-to-br from-violet-500/10 to-indigo-500/10 border border-violet-500/20 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div>
            <p className="text-[10px] text-slate-300 uppercase tracking-wider font-bold">Capital Allocation Plan</p>
            <p className="text-2xl font-black text-white mt-0.5">{fmtINR(s.total_allocated)} <span className="text-sm text-slate-300 font-normal">deployed</span></p>
          </div>
          <div className="text-right">
            <p className="text-[9px] text-slate-400 uppercase">Confidence</p>
            <p className={`text-lg font-black ${confColor}`}>{s.overall_confidence}</p>
          </div>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {[
            { label: 'Total Capital',    value: fmtINR(data.floating_capital) },
            { label: 'Reserve Buffer',   value: fmtINR(data.reserve_buffer), sub: data.buffer_reason?.split('(')[0]?.trim() },
            { label: 'Positions Funded', value: `${s.positions_addressed} stocks` },
            { label: 'Horizon',          value: s.horizon_label },
          ].map(m => (
            <div key={m.label} className="rounded-xl bg-white/[0.04] border border-white/[0.06] p-2.5 text-center">
              <p className="text-[8px] text-slate-400 uppercase tracking-wider mb-1">{m.label}</p>
              <p className="text-[11px] font-bold text-white">{m.value}</p>
              {m.sub && <p className="text-[8px] text-slate-500 mt-0.5">{m.sub}</p>}
            </div>
          ))}
        </div>
      </div>

      {/* Strategy note */}
      {data.strategy_note && (
        <div className="flex items-start gap-2 p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
          <Lightbulb className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
          <p className="text-[11px] text-indigo-200 leading-relaxed">{data.strategy_note}</p>
        </div>
      )}

      {/* Grouped Suggestion cards */}
      <div className="space-y-4">
        <p className="text-[10px] text-slate-300 uppercase tracking-wider font-bold flex items-center gap-1.5">
          <Target className="h-3 w-3" /> Category-Wise Allocation Plan
        </p>
        {Object.entries(groupedSuggestions).map(([sector, list]) => (
          <div key={sector} className="space-y-2">
            <p className="text-[10px] font-bold text-violet-400/80 uppercase tracking-wider pl-1 flex items-center gap-1">
              <span>📁</span> {sector} Sector
            </p>
            <div className="space-y-2.5">
              {list.map((sug, i) => (
                <SuggestionCard key={sug.ticker} s={sug} rank={i + 1} />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Skipped positions */}
      {data.skipped_positions?.length > 0 && (
        <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3">
          <p className="text-[10px] font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Eye className="h-3 w-3 text-indigo-400" /> Watchlist & Alternates under Monitor
          </p>
          {data.skipped_positions.map(sp => (
            <div key={sp.ticker} className="flex items-start gap-2 py-2 border-b border-white/[0.04] last:border-0">
              <div className="h-6 w-6 rounded-lg bg-indigo-500/10 flex items-center justify-center shrink-0">
                <span className="text-[8px] font-black text-indigo-400">{sp.ticker.replace('.NS','').slice(0,3)}</span>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-[11px] font-bold text-indigo-300">{sp.ticker.replace('.NS','').replace('.BO','')}</p>
                  {sp.ml_propensity_score != null && (
                    <span className="text-[8px] font-bold px-1.5 py-0.5 rounded bg-violet-500/10 border border-violet-500/20 text-violet-300">
                      AI Setup: {sp.ml_propensity_score}%
                    </span>
                  )}
                </div>
                <p className="text-[10px] text-slate-300 leading-relaxed mt-0.5">{sp.reason}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Disclaimer + reset */}
      <div className="flex items-center justify-between">
        <p className="text-[9px] text-slate-500 flex items-center gap-1">
          <Info className="h-3 w-3" /> For educational purposes only. Not financial advice.
        </p>
        <button onClick={onReset} className="text-[10px] text-violet-400 hover:text-violet-300 cursor-pointer">
          ← Recalculate
        </button>
      </div>
    </div>
  );
}

// ── Main exported component ────────────────────────────────────────────────
export default function SmartCapitalAdvisor({ holdings, onClose }) {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  const validHoldings = (holdings || []).filter(h => h.ticker && +h.qty > 0 && +h.buy_price > 0);

  return (
    <div className="fixed inset-0 z-[500] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}>

      <div className="w-full max-w-2xl max-h-[90vh] flex flex-col rounded-2xl border border-violet-500/20 bg-[#0a0a0f] shadow-2xl shadow-violet-500/10 overflow-hidden">

        {/* Modal header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-white/[0.06] shrink-0"
          style={{ background: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(99,102,241,0.05) 100%)' }}>
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-violet-500/25 to-indigo-500/25 border border-violet-500/30 flex items-center justify-center">
            <Sparkles className="h-4.5 w-4.5 text-violet-400" />
          </div>
          <div className="flex-1">
            <h2 className="text-base font-black text-white">Smart Capital Advisor</h2>
            <p className="text-[11px] text-slate-300">Tell us your floating money & horizon → we build your recovery plan</p>
          </div>
          {onClose && (
            <button onClick={onClose} className="h-7 w-7 flex items-center justify-center rounded-lg hover:bg-white/[0.06] text-slate-600 hover:text-slate-300 transition cursor-pointer">
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto p-5">
          {!result ? (
            <InputPanel
              holdings={validHoldings}
              onResult={setResult}
              loading={loading}
              setLoading={setLoading}
            />
          ) : (
            <ResultsPanel
              data={result}
              onReset={() => setResult(null)}
            />
          )}
        </div>
      </div>
    </div>
  );
}
