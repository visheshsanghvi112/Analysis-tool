'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ShieldAlert, ShieldCheck, Scale, TrendingUp, TrendingDown,
  AlertTriangle, CheckCircle2, Crosshair, Sliders, RefreshCw,
  Layers, Zap, Info, Lock, ArrowUpRight, ArrowDownRight,
  ChevronDown, ChevronUp, Sparkles, DollarSign, Percent
} from 'lucide-react';
import InfoBadge from './InfoBadge';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

export default function InvestmentCommitteeDesk({ ticker }) {
  const [horizon, setHorizon] = useState('swing');
  const [capital, setCapital] = useState(100000);
  const [riskPct, setRiskPct] = useState(1.0);
  const [direction, setDirection] = useState('LONG');
  const [showSettings, setShowSettings] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const isIndian = ticker?.endsWith('.NS') || ticker?.endsWith('.BO');
  const currSym = isIndian ? '₹' : '$';

  const fetchDeskData = useCallback(async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      const url = `${API_BASE_URL}/api/desk/evaluate/${encodeURIComponent(ticker)}?horizon=${horizon}&account_capital=${capital}&account_risk_pct=${riskPct}&direction=${direction}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Committee evaluation failed (${res.status})`);
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message || 'Failed to load Investment Committee desk data');
    } finally {
      setLoading(false);
    }
  }, [ticker, horizon, capital, riskPct, direction]);

  useEffect(() => {
    fetchDeskData();
  }, [fetchDeskData]);

  // Extract Bull vs Bear dialectical duel evidence
  const { bullPoints, bearPoints } = useMemo(() => {
    if (!data || !data.desks) return { bullPoints: [], bearPoints: [] };
    const allEvidence = [
      ...(data.desks.fundamental?.evidence || []),
      ...(data.desks.technical?.evidence || []),
      ...(data.desks.derivatives?.evidence || []),
    ];

    const bulls = allEvidence.filter(e => 
      e.status?.includes('BULL') || e.status === 'OK' || e.status === 'POSITIVE'
    );
    const bears = allEvidence.filter(e => 
      e.status?.includes('BEAR') || e.status === 'RISK' || e.status === 'EXTENDED' || e.status === 'ELEVATED' || e.status === 'HIGH_RISK'
    );

    return {
      bullPoints: bulls.slice(0, 3),
      bearPoints: bears.slice(0, 3)
    };
  }, [data]);

  const {
    committee_state,
    action_state,
    committee_score,
    committee_confidence,
    desks = {},
    risk_gate = {},
    active_conflicts = [],
    trade_geometry = {},
    weights = {}
  } = data || {};

  const getStanceColor = (s) => {
    if (s === 'BULLISH') return { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', glow: 'shadow-emerald-500/10' };
    if (s === 'BEARISH') return { bg: 'bg-rose-500/10', border: 'border-rose-500/30', text: 'text-rose-400', glow: 'shadow-rose-500/10' };
    return { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', glow: 'shadow-amber-500/10' };
  };

  const getActionBadge = (a) => {
    switch (a) {
      case 'LONG_BIAS':
        return { label: 'LONG EXECUTION BIAS', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' };
      case 'SHORT_BIAS':
        return { label: 'SHORT EXECUTION BIAS', color: 'bg-rose-500/20 text-rose-300 border-rose-500/40' };
      case 'WAIT':
        return { label: 'WAIT FOR TRIGGER / PULLBACK', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40' };
      case 'NO_TRADE':
        return { label: 'NO TRADE (CRO VETO)', color: 'bg-rose-900/40 text-rose-300 border-rose-600/50' };
      case 'CONFLICTED':
        return { label: 'CONFLICTED COMMITTEE', color: 'bg-purple-500/20 text-purple-300 border-purple-500/40' };
      default:
        return { label: a || 'INSUFFICIENT DATA', color: 'bg-slate-800 text-slate-300 border-slate-700' };
    }
  };

  const stanceStyle = getStanceColor(committee_state);
  const actionBadge = getActionBadge(action_state);

  return (
    <div className="glass-card p-4 sm:p-6 space-y-6">
      {/* ── Component Top Header ───────────────────────────────── */}
      <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center border border-indigo-500/30 shrink-0">
            <Scale className="h-5 w-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm sm:text-base font-bold text-white tracking-tight">
                Institutional Investment Committee
              </h3>
              <InfoBadge 
                title="Multi-Desk Investment Committee"
                what="A deterministic multi-agent deliberation framework inspired by institutional hedge funds. Evaluates Fundamental valuation, Technical momentum, and Derivatives positioning."
                why="Eliminates single-indicator bias and emotional trading by enforcing independent cross-examination and non-directional risk vetoes."
                interpretation="A Bullish verdict requires at least 2 desks in agreement with zero bearish dissent, subject to Chief Risk Officer (CRO) clearance."
              />
            </div>
            <p className="text-[10px] sm:text-xs text-slate-400">
              Deterministic Multi-Desk Deliberation, Conflict Matrix & CRO Veto
            </p>
          </div>
        </div>

        <button
          onClick={fetchDeskData}
          disabled={loading}
          className="p-2 rounded-xl bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.06] text-slate-400 hover:text-white transition disabled:opacity-40 cursor-pointer"
          title="Re-run Committee Evaluation"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin text-indigo-400' : ''}`} />
        </button>
      </div>

      {loading && !data ? (
        <div className="py-12 text-center">
          <div className="inline-flex items-center justify-center p-3.5 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-3 relative">
            <Scale className="w-7 h-7 text-indigo-300" />
            <RefreshCw className="w-3.5 h-3.5 animate-spin absolute -top-1 -right-1 text-indigo-400" />
          </div>
          <h4 className="text-sm font-bold text-white mb-1">Convening Multi-Desk Committee...</h4>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Cross-examining technical trend, DCF valuation, and derivatives risk across deterministic models.
          </p>
        </div>
      ) : error && !data ? (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button onClick={fetchDeskData} className="px-3 py-1 rounded-lg bg-rose-500/20 font-bold hover:bg-rose-500/30">
            Retry
          </button>
        </div>
      ) : (
        <>
          {/* ── Committee Executive Verdict Card ─────────────────── */}
          <div className={`rounded-2xl border ${stanceStyle.border} ${stanceStyle.bg} p-5 sm:p-6 backdrop-blur-xl relative overflow-hidden shadow-xl transition-all duration-300`}>
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-5 relative z-10">
              <div>
                <div className="flex items-center gap-2.5 mb-2 flex-wrap">
                  <span className="text-[10px] uppercase tracking-widest font-black text-slate-400 flex items-center gap-1.5">
                    <Scale className="w-3.5 h-3.5 text-indigo-400" />
                    Committee Stance
                  </span>
                  <span className={`text-[10px] uppercase tracking-wider font-extrabold px-2.5 py-0.5 rounded-full border ${actionBadge.color} shadow-sm`}>
                    {actionBadge.label}
                  </span>
                  <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-white/[0.06] border border-white/[0.08] text-slate-300">
                    Horizon: {horizon.toUpperCase()}
                  </span>
                </div>

                <div className="flex items-baseline gap-4">
                  <h2 className="text-2xl sm:text-3xl font-black text-white tracking-tight">
                    {committee_state}
                  </h2>
                  <div className="flex items-center gap-1.5">
                    <span className="text-xl font-black text-white tabular-nums">{committee_score}</span>
                    <span className="text-[11px] text-slate-400 font-bold">/ 100 Composite</span>
                  </div>
                </div>
              </div>

              {/* Right Controls: Horizon Selector & Interactive Capital Toggle */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="bg-black/40 border border-white/10 rounded-xl p-1 flex gap-1 shadow-inner">
                  {['intraday', 'swing', 'long_term'].map((h) => (
                    <button
                      key={h}
                      onClick={() => setHorizon(h)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold capitalize transition-all ${
                        horizon === h
                          ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {h.replace('_', ' ')}
                    </button>
                  ))}
                </div>

                <button
                  onClick={() => setShowSettings(v => !v)}
                  className={`p-2 rounded-xl border transition-all flex items-center gap-1.5 text-xs font-bold ${
                    showSettings 
                      ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300' 
                      : 'bg-black/30 border-white/10 text-slate-400 hover:text-white'
                  }`}
                  title="Adjust Capital & Risk Parameters"
                >
                  <Sliders className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Execution Params</span>
                </button>

                <div className="bg-black/30 border border-white/10 rounded-xl p-2.5 min-w-[120px]">
                  <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 mb-1">
                    <span>Confidence</span>
                    <span className="text-white tabular-nums">{committee_confidence}%</span>
                  </div>
                  <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                      style={{ width: `${committee_confidence}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* ── Interactive Capital & Risk Controls Drawer ──────── */}
            {showSettings && (
              <div className="mt-5 pt-5 border-t border-white/[0.08] grid grid-cols-1 sm:grid-cols-3 gap-4 animate-in fade-in duration-200">
                <div>
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                    Account Capital ({currSym})
                  </label>
                  <div className="flex items-center gap-1.5">
                    {[50000, 100000, 500000].map(amt => (
                      <button
                        key={amt}
                        onClick={() => setCapital(amt)}
                        className={`px-2 py-1 rounded-lg text-xs font-mono font-bold border transition-all ${
                          capital === amt ? 'bg-indigo-600/30 border-indigo-500/50 text-indigo-300' : 'bg-black/40 border-white/10 text-slate-400'
                        }`}
                      >
                        {(amt / 1000).toFixed(0)}k
                      </button>
                    ))}
                    <input
                      type="number"
                      value={capital}
                      onChange={(e) => setCapital(Number(e.target.value) || 10000)}
                      className="w-24 px-2.5 py-1 rounded-lg bg-black/50 border border-white/10 text-xs font-mono text-white outline-none focus:border-indigo-500/60 focus:ring-1 focus:ring-indigo-500/40 transition"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                    Risk Per Trade (%)
                  </label>
                  <div className="flex items-center gap-1.5">
                    {[0.5, 1.0, 2.0].map(r => (
                      <button
                        key={r}
                        onClick={() => setRiskPct(r)}
                        className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold border transition-all ${
                          riskPct === r ? 'bg-indigo-600/30 border-indigo-500/50 text-indigo-300' : 'bg-black/40 border-white/10 text-slate-400'
                        }`}
                      >
                        {r}%
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                    Trade Direction Bias
                  </label>
                  <div className="flex items-center gap-1.5">
                    {['LONG', 'SHORT'].map(d => (
                      <button
                        key={d}
                        onClick={() => setDirection(d)}
                        className={`px-3 py-1 rounded-lg text-xs font-bold border transition-all ${
                          direction === d 
                            ? (d === 'LONG' ? 'bg-emerald-600/30 border-emerald-500/50 text-emerald-300' : 'bg-rose-600/30 border-rose-500/50 text-rose-300')
                            : 'bg-black/40 border-white/10 text-slate-400'
                        }`}
                      >
                        {d}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── Dialectical Red-Teaming Duel (Bull vs Bear Clash) ─── */}
          <div className="rounded-2xl border border-white/[0.08] bg-[#0c1017]/90 p-5 shadow-lg relative overflow-hidden">
            <div className="flex items-center justify-between mb-3.5">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-indigo-400" />
                <h4 className="text-xs font-black uppercase tracking-wider text-white">
                  Dialectical Red-Teaming Duel (Bull Thesis vs. Bear Skeptic)
                </h4>
                <InfoBadge
                  title="Dialectical Red-Teaming"
                  what="Inspired by academic research from UCLA & MIT. Eliminates sycophancy and confirmation bias by pitting a dedicated Bull Researcher against a dedicated Bear Adversary."
                  why="Forces the platform to identify the strongest counter-arguments before capital is committed."
                  interpretation="If the Bear points reveal high debt or extreme valuation, the trade must be treated with caution regardless of technical momentum."
                />
              </div>
              <span className="text-[10px] font-mono text-slate-400 hidden sm:inline">
                Adversarial Cross-Examination
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Bull Thesis */}
              <div className="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/20">
                <div className="flex items-center gap-2 mb-2.5">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <h5 className="text-[11px] font-black text-emerald-300 uppercase tracking-wider">
                    The Long Thesis (Bull Case)
                  </h5>
                </div>
                <div className="space-y-2">
                  {bullPoints.length > 0 ? (
                    bullPoints.map((pt, idx) => (
                      <div key={idx} className="text-xs text-emerald-200/90 flex items-start gap-2">
                        <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-bold text-white">{pt.factor}: </span>
                          <span className="text-slate-300">{pt.detail}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-xs text-slate-500 italic">No strong bullish factors qualified under current thresholds.</p>
                  )}
                </div>
              </div>

              {/* Bear Thesis */}
              <div className="p-4 rounded-xl bg-rose-500/5 border border-rose-500/20">
                <div className="flex items-center gap-2 mb-2.5">
                  <div className="w-2 h-2 rounded-full bg-rose-400 animate-pulse" />
                  <h5 className="text-[11px] font-black text-rose-300 uppercase tracking-wider">
                    The Skeptic Counter-Case (Bear Red-Team)
                  </h5>
                </div>
                <div className="space-y-2">
                  {bearPoints.length > 0 ? (
                    bearPoints.map((pt, idx) => (
                      <div key={idx} className="text-xs text-rose-200/90 flex items-start gap-2">
                        <ArrowDownRight className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-bold text-white">{pt.factor}: </span>
                          <span className="text-slate-300">{pt.detail}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-xs text-slate-500 italic">No critical vulnerabilities or red flags flagged by the skeptic.</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ── Thesis Invalidation Triggers (Evidence-Driven) ─────── */}
          {data.thesis_invalidation_triggers && data.thesis_invalidation_triggers.length > 0 && (
            <div className="rounded-2xl border border-white/[0.08] bg-[#0c1017]/90 p-5 shadow-lg relative overflow-hidden">
              <div className="flex items-center justify-between mb-3.5">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  <h4 className="text-xs font-black uppercase tracking-wider text-white">
                    Thesis Invalidation Triggers (Evidence-Driven)
                  </h4>
                  <InfoBadge
                    title="Thesis Invalidation Triggers"
                    what="Deterministic mathematical thresholds that automatically invalidate the committee's thesis. Derived strictly from verified metrics (session VWAP, Supertrend, ORB range, CRO risk gate)."
                    why="Ensures capital protection by defining explicit stop criteria before market entry, rather than relying on discretionary exit rules."
                    interpretation="If any trigger is breached, the thesis is invalidated and trade risk must be eliminated or reduced."
                  />
                </div>
                <span className="text-[10px] font-mono text-slate-400 hidden sm:inline">
                  Deterministic Guardrails
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {data.thesis_invalidation_triggers.map((trig, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-xl border flex items-start gap-2.5 transition ${
                      trig.severity === 'CRITICAL'
                        ? 'bg-rose-500/10 border-rose-500/30 text-rose-200'
                        : trig.severity === 'HIGH'
                        ? 'bg-amber-500/10 border-amber-500/30 text-amber-200'
                        : 'bg-white/[0.02] border-white/[0.06] text-slate-300'
                    }`}
                  >
                    <div className={`p-1.5 rounded-lg shrink-0 ${
                      trig.severity === 'CRITICAL' ? 'bg-rose-500/20 text-rose-400' :
                      trig.severity === 'HIGH' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-white/[0.05] text-slate-400'
                    }`}>
                      <ShieldAlert className="w-3.5 h-3.5" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-2 mb-0.5">
                        <span className="text-[10px] font-mono uppercase font-bold tracking-wider opacity-80">
                          {trig.type?.replace(/_/g, ' ') || 'TRIGGER'}
                        </span>
                        <span className="text-[9px] font-extrabold uppercase px-1.5 py-0.2 rounded bg-black/40 border border-white/10">
                          {trig.severity}
                        </span>
                      </div>
                      <p className="text-xs font-semibold leading-snug">
                        {trig.condition}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Active Conflict Matrix Alerts ───────────────────── */}
          {active_conflicts.length > 0 ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs font-black uppercase tracking-wider text-amber-400">
                <AlertTriangle className="w-4 h-4" />
                Detected Model Conflicts ({active_conflicts.length})
                <InfoBadge
                  title="Conflict Matrix"
                  what="Detects contradictory signals between desks, such as Value Traps (cheap DCF but price dumping) or Parabolic Tops (RSI > 80 with 95th percentile P/E)."
                  why="Prevents the common retail mistake of averaging conflicting metrics into a misleading composite score."
                  interpretation="When HIGH severity conflicts occur, the action state becomes WAIT until price structure or valuation stabilizes."
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {active_conflicts.map((conflict, idx) => (
                  <div
                    key={idx}
                    className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4 sm:p-5 relative overflow-hidden shadow-lg"
                  >
                    <div className="flex items-start gap-3">
                      <div className="p-2 rounded-xl bg-amber-500/20 text-amber-400 shrink-0">
                        <ShieldAlert className="w-5 h-5" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <h4 className="text-sm font-black text-amber-200">{conflict.title}</h4>
                          <span className="text-[10px] font-extrabold uppercase px-2 py-0.5 rounded-full bg-amber-500/20 border border-amber-500/30 text-amber-300">
                            {conflict.severity} SEVERITY
                          </span>
                        </div>
                        <p className="text-xs text-amber-100/90 mb-2.5 leading-relaxed">
                          {conflict.diagnostic}
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {conflict.evidence?.map((ev, eIdx) => (
                            <span key={eIdx} className="text-[10px] px-2 py-0.5 rounded-md bg-black/40 border border-amber-500/20 text-amber-300 font-mono">
                              {ev}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between px-4 py-2.5 rounded-xl bg-white/[0.02] border border-white/[0.05] text-xs">
              <div className="flex items-center gap-2 text-slate-400">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span>Conflict Matrix: <strong>0 Model Conflicts</strong> Detected — Cross-desk signals aligned.</span>
              </div>
              <span className="text-[10px] font-mono text-emerald-400/80 uppercase font-semibold hidden sm:inline">
                Harmonious Deliberation
              </span>
            </div>
          )}

          {/* ── The 3 Desks Grid ───────────────────────────────── */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* Fundamental Desk */}
            {desks.fundamental && (
              <DeskCard
                title="Fundamental Desk"
                icon={<Layers className="w-4 h-4 text-emerald-400" />}
                weight={weights.fundamental}
                desk={desks.fundamental}
              />
            )}

            {/* Technical Desk */}
            {desks.technical && (
              <DeskCard
                title="Technical & Momentum"
                icon={<TrendingUp className="w-4 h-4 text-blue-400" />}
                weight={weights.technical}
                desk={desks.technical}
              />
            )}

            {/* Derivatives / Volatility Desk */}
            {desks.derivatives && (
              <DeskCard
                title="Derivatives & Volatility"
                icon={<Zap className="w-4 h-4 text-purple-400" />}
                weight={weights.derivatives}
                desk={desks.derivatives}
              />
            )}
          </div>

          {/* ── Chief Risk Officer (CRO) Gate & Trade Geometry ───── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* CRO Risk Gate */}
            <div className="rounded-2xl border border-white/[0.08] bg-[#0c1017]/90 p-5 sm:p-6 shadow-lg flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-indigo-400" />
                    <h4 className="text-xs sm:text-sm font-black uppercase tracking-wider text-white">
                      Chief Risk Officer (CRO) Gate
                    </h4>
                    <InfoBadge
                      title="CRO Risk Gate"
                      what="A non-directional risk management firewall that evaluates volatility spikes, historical drawdowns, earnings event proximity, and price freshness."
                      why="Protects capital by exercising veto authority over directional opinions."
                      interpretation="If risk penalty >= 65 or an earnings event is within 24 hours, the CRO issues a VETO, capping sizing at 0%."
                    />
                  </div>
                  <span className={`text-[10px] font-black uppercase px-2.5 py-1 rounded-full border ${
                    risk_gate.state === 'PASS' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' :
                    risk_gate.state === 'CAUTION' ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' :
                    risk_gate.state === 'CONDITIONAL' ? 'bg-blue-500/20 text-blue-300 border-blue-500/40' :
                    'bg-rose-500/20 text-rose-300 border-rose-500/40'
                  }`}>
                    Gate State: {risk_gate.state || 'PASS'}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <span className="text-[10px] font-bold text-slate-400 block mb-1">Risk Penalty Score</span>
                    <span className="text-base sm:text-lg font-black text-white tabular-nums">{risk_gate.risk_score || 0} / 100</span>
                  </div>
                  <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                    <span className="text-[10px] font-bold text-slate-400 block mb-1">Sizing Allocation Cap</span>
                    <span className="text-base sm:text-lg font-black text-indigo-300 tabular-nums">{((risk_gate.sizing_cap_pct ?? 1.0) * 100).toFixed(0)}%</span>
                  </div>
                </div>

                <div className="space-y-2 mb-4">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Risk Factors Evaluated:</span>
                  {risk_gate.risk_factors?.map((rf, rIdx) => (
                    <div key={rIdx} className="flex items-center justify-between text-xs py-1.5 px-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                      <span className="text-slate-300 font-medium">{rf.factor}</span>
                      <span className={`text-[10px] font-extrabold uppercase px-1.5 py-0.5 rounded ${
                        rf.severity === 'HIGH' ? 'bg-rose-500/20 text-rose-300' :
                        rf.severity === 'ELEVATED' ? 'bg-amber-500/20 text-amber-300' :
                        rf.severity === 'MISSING' ? 'bg-slate-800 text-slate-400' :
                        'bg-emerald-500/20 text-emerald-300'
                      }`}>
                        {rf.severity}: {String(rf.value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="pt-3 border-t border-white/[0.06] text-[10px] text-slate-500 flex justify-between">
                <span>Non-Directional Veto Gate</span>
                <span>Confidence: {risk_gate.confidence}%</span>
              </div>
            </div>

            {/* Trade Execution Geometry */}
            <div className="rounded-2xl border border-white/[0.08] bg-[#0c1017]/90 p-5 sm:p-6 shadow-lg flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Crosshair className="w-5 h-5 text-indigo-400" />
                    <h4 className="text-xs sm:text-sm font-black uppercase tracking-wider text-white">
                      Deterministic Trade Geometry
                    </h4>
                    <InfoBadge
                      title="Trade Geometry & Kelly Sizing"
                      what="Calculates mathematical entry, volatility-adjusted stop-loss (2x ATR), fixed 2R and 3R profit targets, and Half-Kelly position sizing."
                      why="Ensures trades have positive mathematical expectancy and account risk never exceeds the configured threshold."
                      interpretation="Half-Kelly sizing balances capital growth with drawdown safety by halving theoretical full-Kelly aggressiveness."
                    />
                  </div>
                  {trade_geometry.available && (
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      ATR 2.0x Sizing
                    </span>
                  )}
                </div>

                {trade_geometry.available ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                      <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                        <span className="text-[10px] font-bold text-slate-400 block mb-1">Entry Price</span>
                        <span className="text-xs sm:text-sm font-black text-white font-mono tabular-nums">{currSym}{trade_geometry.entry_price != null ? Number(trade_geometry.entry_price).toLocaleString(isIndian ? 'en-IN' : 'en-US') : '—'}</span>
                      </div>
                      <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/20">
                        <span className="text-[10px] font-bold text-rose-300 block mb-1">Stop Loss (ATR)</span>
                        <span className="text-xs sm:text-sm font-black text-rose-400 font-mono tabular-nums">{currSym}{trade_geometry.stop_loss != null ? Number(trade_geometry.stop_loss).toLocaleString(isIndian ? 'en-IN' : 'en-US') : '—'}</span>
                      </div>
                      <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                        <span className="text-[10px] font-bold text-emerald-300 block mb-1">Target 1 (2R)</span>
                        <span className="text-xs sm:text-sm font-black text-emerald-400 font-mono tabular-nums">{currSym}{trade_geometry.target1 != null ? Number(trade_geometry.target1).toLocaleString(isIndian ? 'en-IN' : 'en-US') : '—'}</span>
                      </div>
                      <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                        <span className="text-[10px] font-bold text-emerald-300 block mb-1">Target 2 (3R)</span>
                        <span className="text-xs sm:text-sm font-black text-emerald-400 font-mono tabular-nums">{currSym}{trade_geometry.target2 != null ? Number(trade_geometry.target2).toLocaleString(isIndian ? 'en-IN' : 'en-US') : '—'}</span>
                      </div>
                    </div>

                    {/* Sizing & Capital Controls */}
                    <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-2.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Half-Kelly Optimal Risk:</span>
                        <span className="font-bold text-white font-mono tabular-nums">{trade_geometry.half_kelly_pct ?? 'N/A'}%</span>
                      </div>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-400">Effective Account Risk:</span>
                        <span className="font-bold text-indigo-300 font-mono tabular-nums">{trade_geometry.effective_account_risk_pct ?? 'N/A'}%</span>
                      </div>
                      <div className="flex items-center justify-between text-xs pt-2 border-t border-white/[0.06]">
                        <span className="text-slate-300 font-semibold">Calculated Position Shares:</span>
                        <span className="font-black text-emerald-400 font-mono text-sm tabular-nums">
                          {trade_geometry.shares ?? 0} shares
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-6 text-center text-xs text-slate-400 italic">
                    {trade_geometry.reason || 'Trade geometry requires entry price and ATR.'}
                  </div>
                )}
              </div>

              <div className="pt-3 border-t border-white/[0.06] text-[10px] text-slate-500 flex justify-between">
                <span>Risk per share: {trade_geometry.risk_per_share ? `${currSym}${trade_geometry.risk_per_share}` : 'N/A'}</span>
                <span>Target R/R: {trade_geometry.target1_r ?? 2.0}x / {trade_geometry.target2_r ?? 3.0}x</span>
              </div>
            </div>
          </div>

          {/* ── Deterministic Audit Footer ─────────────────────── */}
          <div className="flex items-center justify-between text-[11px] text-slate-500 px-3 py-2 rounded-xl bg-white/[0.01] border border-white/[0.04] flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1 text-slate-400">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                100% Deterministic Engine
              </span>
              <span>•</span>
              <span className="text-slate-400">Zero LLM Hallucination</span>
              <span>•</span>
              <span className="text-slate-400">Audited Mathematical Invariants</span>
            </div>
            <div className="font-mono text-[10px] text-slate-500">
              v1.0.0 • {ticker}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function DeskCard({ title, icon, weight, desk }) {
  const stanceColor =
    desk.stance === 'BULLISH' ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' :
    desk.stance === 'BEARISH' ? 'text-rose-400 bg-rose-500/10 border-rose-500/30' :
    'text-amber-400 bg-amber-500/10 border-amber-500/30';

  return (
    <div className="rounded-2xl border border-white/[0.08] bg-[#0c1017]/90 p-4 sm:p-5 flex flex-col justify-between shadow-lg hover:border-white/20 transition-all duration-300">
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {icon}
            <h4 className="text-xs font-black uppercase tracking-wider text-white">{title}</h4>
          </div>
          {weight && (
            <span className="text-[10px] font-mono text-slate-400">
              {Math.round(weight * 100)}% wt
            </span>
          )}
        </div>

        <div className="flex items-baseline justify-between mb-3.5">
          <span className={`text-[10px] font-black uppercase px-2 py-0.5 rounded-full border ${stanceColor}`}>
            {desk.stance}
          </span>
          <div className="text-right">
            <span className="text-base sm:text-lg font-black text-white tabular-nums">{desk.score}</span>
            <span className="text-[10px] text-slate-400 block">Score / 100</span>
          </div>
        </div>

        {/* Evidence List */}
        <div className="space-y-2 mb-3.5">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Observed Evidence:</span>
          {desk.evidence?.slice(0, 4).map((ev, idx) => (
            <div key={idx} className="text-xs p-2 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:border-white/10 transition-colors">
              <div className="flex justify-between items-center text-[10px] mb-0.5">
                <span className="font-bold text-slate-300">{ev.factor}</span>
                <span className={`font-mono font-bold tabular-nums ${
                  ev.status?.includes('BULL') ? 'text-emerald-400' :
                  ev.status?.includes('BEAR') || ev.status === 'RISK' ? 'text-rose-400' :
                  'text-slate-400'
                }`}>
                  {ev.value !== undefined ? String(ev.value) : ev.status}
                </span>
              </div>
              <p className="text-[10px] text-slate-400 leading-normal">{ev.detail}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Confidence */}
      <div className="pt-2.5 border-t border-white/[0.06] flex items-center justify-between text-[10px] text-slate-400">
        <span>Confidence</span>
        <span className="font-bold text-white tabular-nums">{desk.confidence}%</span>
      </div>
    </div>
  );
}
