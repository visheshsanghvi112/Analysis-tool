'use client';

import React, { useState } from 'react';
import {
  PieChart, Activity, DollarSign, TrendingUp, TrendingDown,
  Scale, Sliders, CheckCircle2, ArrowRight, ShieldCheck,
  Calendar, Layers, Zap, AlertCircle
} from 'lucide-react';

const fmt = (n, d = 2) => n == null ? 'N/A' : Number(n).toFixed(d);
const fmtINR = (n) => n == null ? 'N/A' : `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export default function PortfolioXRayAnalytics({
  fundamentalsXray = {},
  dividendStream = {},
  benchmarkComparison = {},
  holdings = [],
  currencySymbol = '₹',
  onApplyDriftRebalance,
}) {
  // Target weights state for drift rebalancing
  const [targetWeights, setTargetWeights] = useState(() => {
    const initial = {};
    const equalWeight = holdings.length > 0 ? +(100 / holdings.length).toFixed(1) : 0;
    holdings.forEach(h => {
      initial[h.ticker] = equalWeight;
    });
    return initial;
  });

  const [activeSection, setActiveSection] = useState('xray'); // 'xray' | 'dividends' | 'drift'

  const xray = fundamentalsXray || {};
  const divs = dividendStream || {};
  const bench = benchmarkComparison || {};

  // Compute Drift
  const driftList = holdings.map(h => {
    const currWt = h.weight_pct || 0;
    const tgtWt = targetWeights[h.ticker] !== undefined ? targetWeights[h.ticker] : currWt;
    const drift = +(currWt - tgtWt).toFixed(1);
    return {
      ticker: h.ticker,
      company_name: h.company_name || h.ticker,
      live_price: h.live_price,
      curr_value: h.curr_value,
      qty: h.qty,
      current_wt: currWt,
      target_wt: tgtWt,
      drift_pct: drift,
    };
  });

  const totalTargetWt = Object.values(targetWeights).reduce((a, b) => a + Number(b || 0), 0);

  return (
    <div className="space-y-6">
      {/* Sub-navigation pills */}
      <div className="flex items-center gap-2 border-b border-white/[0.06] pb-3">
        <button
          onClick={() => setActiveSection('xray')}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
            activeSection === 'xray'
              ? 'bg-indigo-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          <Activity className="h-3.5 w-3.5" />
          Fundamental X-Ray & Alpha
        </button>

        <button
          onClick={() => setActiveSection('dividends')}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
            activeSection === 'dividends'
              ? 'bg-emerald-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          <DollarSign className="h-3.5 w-3.5" />
          Dividend Stream & Yield on Cost
        </button>

        <button
          onClick={() => setActiveSection('drift')}
          className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition cursor-pointer ${
            activeSection === 'drift'
              ? 'bg-purple-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-white/[0.04]'
          }`}
        >
          <Scale className="h-3.5 w-3.5" />
          Target Drift Rebalancer
        </button>
      </div>

      {/* SECTION 1: FUNDAMENTAL X-RAY & BENCHMARK ALPHA */}
      {activeSection === 'xray' && (
        <div className="space-y-5">
          {/* Top Banner: Valuation Tilt & Jensen's Alpha */}
          <div className="rounded-2xl bg-gradient-to-br from-indigo-500/10 via-purple-500/5 to-transparent border border-indigo-500/20 p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
                  <p className="text-[10px] font-bold text-indigo-300 uppercase tracking-wider">Portfolio Fundamental X-Ray</p>
                </div>
                <h3 className="text-lg font-black text-white">Synthesized Corporate Multiples</h3>
                <p className="text-xs text-slate-300 mt-0.5">{xray.valuation_description || "Evaluating portfolio valuation against Nifty 50."}</p>
              </div>

              {bench.jensen_alpha_annualized_pct !== undefined && (
                <div className="text-right p-3 rounded-xl bg-black/40 border border-white/[0.06]">
                  <p className="text-[9px] text-slate-400 uppercase font-bold">Annualized Jensen&apos;s Alpha (α)</p>
                  <p className={`text-xl font-black ${bench.jensen_alpha_annualized_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {bench.jensen_alpha_annualized_pct >= 0 ? '+' : ''}{bench.jensen_alpha_annualized_pct}%
                  </p>
                  <p className="text-[9px] text-slate-400">Beta (β): {bench.portfolio_beta || 1.0}</p>
                </div>
              )}
            </div>

            {/* 4 Multiples Comparison Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Weighted P/E Ratio</p>
                <p className="text-base font-black text-white mt-0.5">{xray.weighted_pe || 'N/A'}x</p>
                <p className="text-[9px] text-slate-500">Benchmark: {xray.benchmark_pe || 22.4}x</p>
              </div>

              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Weighted P/B Ratio</p>
                <p className="text-base font-black text-white mt-0.5">{xray.weighted_pb || 'N/A'}x</p>
                <p className="text-[9px] text-slate-500">Benchmark: {xray.benchmark_pb || 3.6}x</p>
              </div>

              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Weighted ROE</p>
                <p className="text-base font-black text-emerald-400 mt-0.5">{xray.weighted_roe_pct || 'N/A'}%</p>
                <p className="text-[9px] text-slate-500">Benchmark: {xray.benchmark_roe_pct || 15.2}%</p>
              </div>

              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Dividend Yield</p>
                <p className="text-base font-black text-indigo-400 mt-0.5">{xray.weighted_div_yield_pct || 'N/A'}%</p>
                <p className="text-[9px] text-slate-500">Benchmark: {xray.benchmark_div_yield_pct || 1.15}%</p>
              </div>
            </div>
          </div>

          {/* Benchmark Relative Performance Horizon Pills */}
          {bench.period_comparison && (
            <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <TrendingUp className="h-4 w-4 text-emerald-400" />
                  Relative Performance vs Benchmark ({bench.performance_verdict || "Outperformance tracking"})
                </h4>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(bench.period_comparison).map(([period, data]) => {
                  const isAlphaPos = data.alpha_pct >= 0;
                  return (
                    <div key={period} className="p-3 rounded-xl bg-black/40 border border-white/[0.04] space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-bold text-slate-300">{period} Horizon</span>
                        <span className={`text-[10px] font-black ${isAlphaPos ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {isAlphaPos ? '+' : ''}{data.alpha_pct}% α
                        </span>
                      </div>
                      <div className="flex justify-between items-center text-[10px] pt-1">
                        <span className="text-slate-400">Portfolio: <strong className="text-white">{data.portfolio_ret_pct}%</strong></span>
                        <span className="text-slate-400">Nifty: <strong className="text-slate-300">{data.benchmark_ret_pct}%</strong></span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SECTION 2: DIVIDEND STREAM & YIELD ON COST */}
      {activeSection === 'dividends' && (
        <div className="space-y-5">
          <div className="rounded-2xl bg-gradient-to-br from-emerald-500/10 via-teal-500/5 to-transparent border border-emerald-500/20 p-5 space-y-4">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <p className="text-[10px] font-bold text-emerald-300 uppercase tracking-wider">Passive Cash Flow Engine</p>
              </div>
              <h3 className="text-xl font-black text-white">
                +{currencySymbol}{divs.annual_projected_cash?.toLocaleString('en-IN', { maximumFractionDigits: 0 })} / Year
              </h3>
              <p className="text-xs text-slate-300 mt-0.5">
                Estimated passive dividend stream (~{currencySymbol}{divs.monthly_average_cash?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}/month).
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2">
              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Yield on Cost (YOC)</p>
                <p className="text-lg font-black text-emerald-400 mt-0.5">{divs.yield_on_cost_pct || 'N/A'}%</p>
                <p className="text-[9px] text-slate-500">Return on original buy cost</p>
              </div>

              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Current Market Yield</p>
                <p className="text-lg font-black text-white mt-0.5">{divs.portfolio_dividend_yield_pct || 'N/A'}%</p>
                <p className="text-[9px] text-slate-500">Yield at today&apos;s market value</p>
              </div>

              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-center">
                <p className="text-[9px] text-slate-400 uppercase font-bold">Monthly Average</p>
                <p className="text-lg font-black text-teal-300 mt-0.5">{currencySymbol}{divs.monthly_average_cash?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</p>
                <p className="text-[9px] text-slate-500">Predictable passive income</p>
              </div>
            </div>
          </div>

          {/* Top Dividend Contributors Table */}
          {divs.top_dividend_contributors?.length > 0 && (
            <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-4 space-y-3">
              <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                <Calendar className="h-4 w-4 text-emerald-400" />
                Primary Dividend Contributors
              </h4>
              <div className="divide-y divide-white/[0.04]">
                {divs.top_dividend_contributors.map(c => (
                  <div key={c.ticker} className="py-2.5 flex items-center justify-between text-xs">
                    <div>
                      <span className="font-bold text-white">{c.ticker.replace('.NS','')}</span>
                      <span className="text-[10px] text-slate-400 ml-2">Yield: {c.div_yield_pct}%</span>
                    </div>
                    <div className="font-black text-emerald-400">
                      +{currencySymbol}{c.annual_income?.toLocaleString('en-IN', { maximumFractionDigits: 0 })} / yr
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SECTION 3: TARGET DRIFT REBALANCER */}
      {activeSection === 'drift' && (
        <div className="space-y-4">
          <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Scale className="h-4 w-4 text-purple-400" />
                  Target Asset Allocation & Drift Rebalancer
                </h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Set target percentage allocations for each holding. The rebalancer identifies over- and under-allocated positions.
                </p>
              </div>
              <div className="text-right">
                <span className={`text-xs font-black px-2 py-0.5 rounded-full border ${
                  Math.abs(totalTargetWt - 100) < 0.5 ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400' : 'bg-amber-500/15 border-amber-500/30 text-amber-400'
                }`}>
                  Total Target: {totalTargetWt}%
                </span>
              </div>
            </div>

            {/* Drift Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs min-w-[500px]">
                <thead>
                  <tr className="border-b border-white/[0.06] text-slate-400 text-[10px] uppercase font-bold">
                    <th className="py-2 text-left">Stock</th>
                    <th className="py-2 text-right">Current Wt%</th>
                    <th className="py-2 text-right">Target Wt%</th>
                    <th className="py-2 text-right">Drift</th>
                    <th className="py-2 text-right">Action Needed</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/[0.03]">
                  {driftList.map(item => {
                    const isOver = item.drift_pct > 1.0;
                    const isUnder = item.drift_pct < -1.0;
                    return (
                      <tr key={item.ticker}>
                        <td className="py-2.5 font-bold text-white">{item.ticker.replace('.NS','')}</td>
                        <td className="py-2.5 text-right font-medium text-slate-300">{item.current_wt}%</td>
                        <td className="py-2.5 text-right">
                          <input
                            type="number"
                            min="0"
                            max="100"
                            step="0.5"
                            value={item.target_wt}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value) || 0;
                              setTargetWeights(prev => ({ ...prev, [item.ticker]: val }));
                            }}
                            className="w-16 px-1.5 py-0.5 bg-white/[0.06] border border-white/[0.1] rounded text-right text-xs text-white outline-none focus:border-purple-500"
                          />
                        </td>
                        <td className={`py-2.5 text-right font-black ${isOver ? 'text-rose-400' : isUnder ? 'text-emerald-400' : 'text-slate-400'}`}>
                          {item.drift_pct > 0 ? '+' : ''}{item.drift_pct}%
                        </td>
                        <td className="py-2.5 text-right text-[11px]">
                          {isOver ? (
                            <span className="text-rose-400 font-bold">Trim Overweight</span>
                          ) : isUnder ? (
                            <span className="text-emerald-400 font-bold">Add Underweight</span>
                          ) : (
                            <span className="text-slate-500">Balanced</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Apply button if callback provided */}
            {onApplyDriftRebalance && (
              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => onApplyDriftRebalance(targetWeights)}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-xs font-bold rounded-lg transition cursor-pointer flex items-center gap-1.5"
                >
                  <Scale className="h-3.5 w-3.5" />
                  Execute 1-Click Drift Rebalance
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
