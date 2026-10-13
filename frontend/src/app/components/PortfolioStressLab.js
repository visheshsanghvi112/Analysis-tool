'use client';

import React, { useState } from 'react';
import {
  ShieldAlert, Activity, AlertTriangle, PieChart,
  TrendingDown, CheckCircle2, RefreshCw, BarChart2,
  Sliders, Info, Zap, Flame, Compass
} from 'lucide-react';

const fmt = (n, d = 2) => n == null ? 'N/A' : Number(n).toFixed(d);
const fmtINR = (n) => n == null ? 'N/A' : `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export default function PortfolioStressLab({
  macroStress = {},
  concentration = {},
  cashDeployer = {},
  loading = false,
  currencySymbol = '₹',
  onRefresh,
}) {
  const [selectedScenarioId, setSelectedScenarioId] = useState(
    macroStress?.scenarios?.[0]?.id || 'covid_2020'
  );

  if (loading) {
    return (
      <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-8 text-center space-y-3">
        <div className="h-8 w-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-bold text-white">Simulating Macro Crises & Auditing Concentration Fragility…</p>
        <p className="text-xs text-slate-400">Stress-testing against COVID 2020, GFC 2008, 2022 Rate Hikes, and Oil Shocks</p>
      </div>
    );
  }

  const scenarios = macroStress?.scenarios || [];
  const activeScenario = scenarios.find(s => s.id === selectedScenarioId) || scenarios[0];
  const resilienceScore = macroStress?.resilience_score || 75;

  const hhiScore = concentration?.hhi_score || 0;
  const hhiClass = concentration?.classification || 'DIVERSIFIED';
  const hhiColor = concentration?.color || 'emerald';

  const hhiBadgeClass = hhiColor === 'emerald'
    ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
    : hhiColor === 'amber'
    ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
    : 'bg-rose-500/10 border-rose-500/20 text-rose-400';

  return (
    <div className="space-y-6">
      {/* Overview Resilience Banner */}
      <div className="rounded-2xl bg-gradient-to-br from-indigo-500/10 via-purple-500/5 to-transparent border border-indigo-500/20 p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
              <p className="text-[10px] font-bold text-indigo-300 uppercase tracking-wider">Macro Stress Lab & Crisis Simulator</p>
            </div>
            <h2 className="text-xl font-black text-white">
              Portfolio Resilience Score: <span className={resilienceScore > 70 ? 'text-emerald-400' : resilienceScore > 50 ? 'text-amber-400' : 'text-rose-400'}>{resilienceScore}/100</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Simulating tail-risk shocks, sector contagion, and single-factor concentration exposure.
            </p>
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.08] text-white text-xs font-bold rounded-lg transition cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Re-run Stress Test
            </button>
          )}
        </div>

        {/* 3 Overview Metric Tiles */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-white/[0.06]">
          <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">HHI Concentration</p>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-sm font-black text-white">{hhiScore.toLocaleString()}</span>
              <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded border uppercase ${hhiBadgeClass}`}>
                {hhiClass.replace('_', ' ')}
              </span>
            </div>
            <p className="text-[9px] text-slate-500 mt-1">{concentration?.description}</p>
          </div>

          <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Top-3 Concentration</p>
            <p className="text-sm font-black text-white">{concentration?.top3_concentration_pct}% of total equity</p>
            <p className="text-[9px] text-slate-500 mt-1">
              {concentration?.top3_concentration_pct > 50 ? 'Overweight in top 3 names — high single-stock risk' : 'Healthy multi-stock risk distribution'}
            </p>
          </div>

          <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Market Cash Guidance</p>
            <p className="text-sm font-black text-indigo-400">Keep {cashDeployer?.recommended_cash_buffer_pct || 15}% Dry Powder</p>
            <p className="text-[9px] text-slate-500 mt-1">Regime: {cashDeployer?.regime?.replace('_', ' ')}</p>
          </div>
        </div>
      </div>

      {/* Sector Violations Warning (if any) */}
      {concentration?.sector_violations?.length > 0 && (
        <div className="rounded-xl bg-amber-500/10 border border-amber-500/25 p-3.5 flex items-start gap-3">
          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs space-y-1">
            <p className="font-bold text-white">Sector Concentration Limit Exceeded (&gt;30% Cap)</p>
            {concentration.sector_violations.map(v => (
              <p key={v.sector} className="text-slate-300 text-[11px]">
                • <strong className="text-amber-300">{v.sector}</strong> represents <strong className="text-white">{v.weight_pct}%</strong> of your portfolio (risk threshold: 30%). Consider diversifying into counter-cyclical sectors.
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Historical Crisis Simulation Interactive Desk */}
      <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-5 space-y-5">
        <div>
          <h3 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Flame className="h-4 w-4 text-rose-400" />
            Historical Crisis Simulator
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Select a crisis scenario below to stress-test your portfolio&apos;s weighted exposure against historic drawdowns.
          </p>
        </div>

        {/* Crisis Scenario Selector Tabs */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {scenarios.map((sc) => {
            const isSel = selectedScenarioId === sc.id;
            return (
              <button
                key={sc.id}
                onClick={() => setSelectedScenarioId(sc.id)}
                className={`p-3 rounded-xl border text-left transition cursor-pointer select-none ${
                  isSel
                    ? 'bg-rose-500/15 border-rose-500/40 shadow-lg shadow-rose-500/10'
                    : 'bg-white/[0.02] border-white/[0.05] hover:bg-white/[0.04]'
                }`}
              >
                <p className="text-xs font-bold text-white leading-tight">{sc.name}</p>
                <p className={`text-xs font-black mt-1 ${sc.portfolio_drawdown_pct < -25 ? 'text-rose-400' : 'text-amber-400'}`}>
                  {sc.portfolio_drawdown_pct}%
                </p>
              </button>
            );
          })}
        </div>

        {/* Selected Scenario Deep-Dive Card */}
        {activeScenario && (
          <div className="rounded-xl bg-black/40 border border-white/[0.06] p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-white/[0.06]">
              <div>
                <h4 className="text-base font-black text-white">{activeScenario.name}</h4>
                <p className="text-xs text-slate-400 mt-0.5">{activeScenario.desc}</p>
              </div>
              <div className="text-right">
                <p className="text-[10px] text-slate-400 uppercase font-bold">Simulated Portfolio Impact</p>
                <p className="text-xl font-black text-rose-400">
                  {activeScenario.portfolio_drawdown_pct}% ({currencySymbol}{Math.abs(activeScenario.portfolio_drawdown_rupees || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })})
                </p>
              </div>
            </div>

            {/* Weakest Links in This Crisis */}
            {activeScenario.weakest_links?.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                  <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
                  Primary Vulnerability Points (Weakest Links):
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {activeScenario.weakest_links.map((wl, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-rose-500/8 border border-rose-500/20 flex items-center justify-between">
                      <div>
                        <span className="font-black text-white text-xs">{wl.ticker.replace('.NS','')}</span>
                        <p className="text-[10px] text-slate-400">Idiosyncratic sector drag</p>
                      </div>
                      <div className="text-right">
                        <span className="text-xs font-black text-rose-400">{wl.simulated_pct}%</span>
                        <p className="text-[10px] text-slate-400">-{currencySymbol}{Math.abs(wl.simulated_rupees || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Dynamic Cash Sizing & Market Valuation Deployer Card */}
      <div className="rounded-2xl bg-gradient-to-r from-violet-500/10 via-indigo-500/10 to-transparent border border-violet-500/20 p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Compass className="h-4 w-4 text-violet-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Dynamic Dry Powder & Valuation Deployer</h3>
          </div>
          <span className="text-[9px] font-bold bg-violet-500/20 text-violet-300 border border-violet-500/30 px-2 py-0.5 rounded-full uppercase">
            Market Regime: {cashDeployer?.regime?.replace('_', ' ')}
          </span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          {cashDeployer?.advice}
        </p>
        <div className="flex flex-wrap items-center gap-6 pt-2 border-t border-violet-500/15 text-xs text-slate-400">
          <span>Target Dry Powder: <strong className="text-white">{cashDeployer?.recommended_cash_buffer_pct}%</strong></span>
          <span>Market P/E Estimate: <strong className="text-violet-300">{cashDeployer?.market_pe_est}x</strong></span>
          <span>Strategy: <strong className="text-emerald-400">Deploy fresh capital systematically on 200-EMA dips</strong></span>
        </div>
      </div>
    </div>
  );
}
