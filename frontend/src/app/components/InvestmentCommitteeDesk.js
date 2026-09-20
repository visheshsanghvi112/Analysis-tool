'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ShieldAlert, ShieldCheck, Scale, TrendingUp, TrendingDown,
  AlertTriangle, CheckCircle2, Crosshair, Sliders, RefreshCw,
  Layers, Zap, Info, Lock, ArrowUpRight, ArrowDownRight,
  ChevronDown, ChevronUp, Sparkles, DollarSign, Percent
} from 'lucide-react';

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

  if (loading && !data) {
    return (
      <div className="glass-card p-8 text-center rounded-3xl border border-white/[0.08] bg-[#0c1017]/80 backdrop-blur-xl">
        <div className="inline-flex items-center justify-center p-3.5 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-4 animate-pulse">
          <Scale className="w-8 h-8 animate-spin" />
        </div>
        <h3 className="text-lg font-bold text-white mb-1 tracking-tight">Convening Multi-Desk Committee...</h3>
        <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
          Deliberating technical structure, fundamental DCF valuation, and derivatives risk across deterministic quantitative models.
        </p>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="glass-card p-6 text-center rounded-3xl border border-rose-500/20 bg-rose-500/5 backdrop-blur-xl">
        <AlertTriangle className="w-8 h-8 text-rose-400 mx-auto mb-2" />
        <p className="text-sm font-semibold text-rose-300 mb-3">{error}</p>
        <button
          onClick={fetchDeskData}
          className="px-4 py-2 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 text-xs font-bold transition-all shadow-sm"
        >
          Retry Committee Evaluation
        </button>
      </div>
    );
  }

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
    <div className="space-y-6">
      {/* ── Committee Executive Header ───────────────────────── */}
      <div className={`rounded-3xl border ${stanceStyle.border} ${stanceStyle.bg} p-6 sm:p-8 backdrop-blur-2xl relative overflow-hidden shadow-2xl transition-all duration-300`}>
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 relative z-10">
          <div>
            <div className="flex items-center gap-3 mb-2 flex-wrap">
              <span className="text-xs uppercase tracking-widest font-black text-slate-400 flex items-center gap-1.5">
                <Scale className="w-4 h-4 text-indigo-400" />
                Deterministic Multi-Desk Committee
              </span>
              <span className={`text-[10px] uppercase tracking-wider font-extrabold px-3 py-1 rounded-full border ${actionBadge.color} shadow-sm`}>
                {actionBadge.label}
              </span>
              <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-white/[0.06] border border-white/[0.08] text-slate-300">
                Horizon: {horizon.toUpperCase()}
              </span>
            </div>

            <div className="flex items-baseline gap-4">
              <h2 className="text-3xl sm:text-4xl font-black text-white tracking-tight">
                {committee_state}
              </h2>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-black text-white tabular-nums">{committee_score}</span>
                <span className="text-xs text-slate-400 font-bold">/ 100 Composite</span>
              </div>
            </div>
          </div>

          {/* Right Controls: Horizon Selector & Interactive Capital Drawer Toggle */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="bg-black/40 border border-white/10 rounded-2xl p-1 flex gap-1 shadow-inner">
              {['intraday', 'swing', 'long_term'].map((h) => (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold capitalize transition-all ${
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
              className={`p-2 rounded-2xl border transition-all flex items-center gap-1.5 text-xs font-bold ${
                showSettings 
                  ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300' 
                  : 'bg-black/30 border-white/10 text-slate-400 hover:text-white'
              }`}
              title="Adjust Capital & Risk Parameters"
            >
              <Sliders className="w-4 h-4" />
              <span className="hidden sm:inline">Execution Params</span>
            </button>

            <div className="bg-black/30 border border-white/10 rounded-2xl p-3 min-w-[130px]">
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

        {/* ── Collapsible Interactive Capital & Risk Controls ────── */}
        {showSettings && (
          <div className="mt-6 pt-6 border-t border-white/[0.08] grid grid-cols-1 sm:grid-cols-3 gap-4 animate-in fade-in duration-200">
            <div>
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                Account Capital ({ticker?.endsWith('.NS') || ticker?.endsWith('.BO') ? '₹' : '$'})
              </label>
              <div className="flex items-center gap-2">
                {[50000, 100000, 500000].map(amt => (
                  <button
                    key={amt}
                    onClick={() => setCapital(amt)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold border transition-all ${
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
                  className="w-24 px-2 py-1 rounded-lg bg-black/50 border border-white/10 text-xs font-mono text-white"
                />
              </div>
            </div>

            <div>
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">
                Risk Per Trade (%)
              </label>
              <div className="flex items-center gap-2">
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
              <div className="flex items-center gap-2">
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

      {/* ── Dialectical Red-Teaming Duel (Bull vs Bear Clash) ───── */}
      <div className="rounded-3xl border border-white/[0.08] bg-[#0c1017] p-6 shadow-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <h3 className="text-xs font-black uppercase tracking-wider text-white">
              Dialectical Red-Teaming Duel (Bull Thesis vs. Bear Skeptic)
            </h3>
          </div>
          <span className="text-[10px] font-mono text-slate-400">
            Adversarial Consensus Filter
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Bull Thesis */}
          <div className="p-4 rounded-2xl bg-emerald-500/5 border border-emerald-500/20 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <h4 className="text-xs font-black text-emerald-300 uppercase tracking-wider">
                  The Long Thesis (Bull Case)
                </h4>
              </div>
              <div className="space-y-2">
                {bullPoints.length > 0 ? (
                  bullPoints.map((pt, idx) => (
                    <div key={idx} className="text-xs text-emerald-200/90 flex items-start gap-2">
                      <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold text-white">{pt.factor}: </span>
                        <span>{pt.detail}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-slate-500 italic">No strong bullish factors qualified under current thresholds.</p>
                )}
              </div>
            </div>
          </div>

          {/* Bear Thesis */}
          <div className="p-4 rounded-2xl bg-rose-500/5 border border-rose-500/20 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-2 h-2 rounded-full bg-rose-400 animate-pulse" />
                <h4 className="text-xs font-black text-rose-300 uppercase tracking-wider">
                  The Skeptic Counter-Case (Bear Red-Team)
                </h4>
              </div>
              <div className="space-y-2">
                {bearPoints.length > 0 ? (
                  bearPoints.map((pt, idx) => (
                    <div key={idx} className="text-xs text-rose-200/90 flex items-start gap-2">
                      <ArrowDownRight className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold text-white">{pt.factor}: </span>
                        <span>{pt.detail}</span>
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
      </div>

      {/* ── Active Conflict Matrix Alerts ─────────────────────── */}
      {active_conflicts.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-xs font-black uppercase tracking-wider text-amber-400">
            <AlertTriangle className="w-4 h-4" />
            Detected Model Conflicts ({active_conflicts.length})
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {active_conflicts.map((conflict, idx) => (
              <div
                key={idx}
                className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5 relative overflow-hidden shadow-lg"
              >
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-xl bg-amber-500/20 text-amber-400 shrink-0">
                    <ShieldAlert className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <h4 className="text-sm font-black text-amber-200">{conflict.title}</h4>
                      <span className="text-[10px] font-extrabold uppercase px-2.5 py-0.5 rounded-full bg-amber-500/20 border border-amber-500/30 text-amber-300">
                        {conflict.severity} SEVERITY
                      </span>
                    </div>
                    <p className="text-xs text-amber-100/90 mb-3 leading-relaxed">
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
      )}

      {/* ── The 3 Desks Grid ─────────────────────────────────── */}
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

      {/* ── Chief Risk Officer (CRO) Gate & Trade Geometry ─────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CRO Risk Gate */}
        <div className="rounded-3xl border border-white/[0.08] bg-[#0c1017] p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-black uppercase tracking-wider text-white">
                  Chief Risk Officer (CRO) Gate
                </h3>
              </div>
              <span className={`text-[11px] font-black uppercase px-3 py-1 rounded-full border ${
                risk_gate.state === 'PASS' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' :
                risk_gate.state === 'CAUTION' ? 'bg-amber-500/20 text-amber-300 border-amber-500/40' :
                risk_gate.state === 'CONDITIONAL' ? 'bg-blue-500/20 text-blue-300 border-blue-500/40' :
                'bg-rose-500/20 text-rose-300 border-rose-500/40'
              }`}>
                Gate State: {risk_gate.state || 'PASS'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
                <span className="text-[10px] font-bold text-slate-400 block mb-1">Risk Penalty Score</span>
                <span className="text-lg font-black text-white tabular-nums">{risk_gate.risk_score || 0} / 100</span>
              </div>
              <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
                <span className="text-[10px] font-bold text-slate-400 block mb-1">Sizing Allocation Cap</span>
                <span className="text-lg font-black text-indigo-300 tabular-nums">{((risk_gate.sizing_cap_pct ?? 1.0) * 100).toFixed(0)}%</span>
              </div>
            </div>

            <div className="space-y-2 mb-4">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Risk Factors Evaluated:</span>
              {risk_gate.risk_factors?.map((rf, rIdx) => (
                <div key={rIdx} className="flex items-center justify-between text-xs py-2 px-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
                  <span className="text-slate-300 font-medium">{rf.factor}</span>
                  <span className={`text-[10px] font-extrabold uppercase px-2 py-0.5 rounded ${
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
        <div className="rounded-3xl border border-white/[0.08] bg-[#0c1017] p-6 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Crosshair className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-black uppercase tracking-wider text-white">
                  Deterministic Trade Geometry
                </h3>
              </div>
              {trade_geometry.available && (
                <span className="text-[10px] font-mono px-2.5 py-1 rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  ATR 2.0x Sizing
                </span>
              )}
            </div>

            {trade_geometry.available ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3.5 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
                    <span className="text-[10px] font-bold text-slate-400 block mb-1">Entry Price</span>
                    <span className="text-sm font-black text-white font-mono tabular-nums">{trade_geometry.entry_price}</span>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-rose-500/10 border border-rose-500/20">
                    <span className="text-[10px] font-bold text-rose-300 block mb-1">Stop Loss (ATR)</span>
                    <span className="text-sm font-black text-rose-400 font-mono tabular-nums">{trade_geometry.stop_loss}</span>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-emerald-500/10 border border-emerald-500/20">
                    <span className="text-[10px] font-bold text-emerald-300 block mb-1">Target 1 (2R)</span>
                    <span className="text-sm font-black text-emerald-400 font-mono tabular-nums">{trade_geometry.target1}</span>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-emerald-500/10 border border-emerald-500/20">
                    <span className="text-[10px] font-bold text-emerald-300 block mb-1">Target 2 (3R)</span>
                    <span className="text-sm font-black text-emerald-400 font-mono tabular-nums">{trade_geometry.target2}</span>
                  </div>
                </div>

                {/* Sizing & Capital Controls */}
                <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.05] space-y-3">
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
              <div className="p-8 text-center text-xs text-slate-400 italic">
                {trade_geometry.reason || 'Trade geometry requires entry price and ATR.'}
              </div>
            )}
          </div>

          <div className="pt-3 border-t border-white/[0.06] text-[10px] text-slate-500 flex justify-between">
            <span>Risk per share: {trade_geometry.risk_per_share ?? 'N/A'}</span>
            <span>Target R/R: {trade_geometry.target1_r ?? 2.0}x / {trade_geometry.target2_r ?? 3.0}x</span>
          </div>
        </div>
      </div>

      {/* ── Deterministic Audit Footer ───────────────────────── */}
      <div className="flex items-center justify-between text-[11px] text-slate-500 px-3 py-2 rounded-2xl bg-white/[0.01] border border-white/[0.04] flex-wrap gap-2">
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
    </div>
  );
}

function DeskCard({ title, icon, weight, desk }) {
  const stanceColor =
    desk.stance === 'BULLISH' ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' :
    desk.stance === 'BEARISH' ? 'text-rose-400 bg-rose-500/10 border-rose-500/30' :
    'text-amber-400 bg-amber-500/10 border-amber-500/30';

  return (
    <div className="rounded-3xl border border-white/[0.08] bg-[#0c1017] p-5 flex flex-col justify-between shadow-xl hover:border-white/20 transition-all duration-300">
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

        <div className="flex items-baseline justify-between mb-4">
          <span className={`text-[10px] font-black uppercase px-2.5 py-0.5 rounded-full border ${stanceColor}`}>
            {desk.stance}
          </span>
          <div className="text-right">
            <span className="text-lg font-black text-white tabular-nums">{desk.score}</span>
            <span className="text-[10px] text-slate-400 block">Score / 100</span>
          </div>
        </div>

        {/* Evidence List */}
        <div className="space-y-2 mb-4">
          <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Observed Evidence:</span>
          {desk.evidence?.slice(0, 4).map((ev, idx) => (
            <div key={idx} className="text-xs p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04] hover:border-white/10 transition-colors">
              <div className="flex justify-between items-center text-[10px] mb-1">
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
      <div className="pt-3 border-t border-white/[0.06] flex items-center justify-between text-[10px] text-slate-400">
        <span>Confidence</span>
        <span className="font-bold text-white tabular-nums">{desk.confidence}%</span>
      </div>
    </div>
  );
}
