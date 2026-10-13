'use client';

import React, { useState } from 'react';
import {
  TrendingUp, Sparkles, ShieldCheck, DollarSign, Target,
  ArrowRight, RefreshCw, Zap, CheckCircle2, ChevronDown, ChevronUp,
  Sliders, Award, Lock, Rocket
} from 'lucide-react';

const fmt = (n, d = 2) => n == null ? 'N/A' : Number(n).toFixed(d);
const fmtINR = (n) => n == null ? 'N/A' : `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export default function PortfolioProfitDesk({
  alphaHarvests = [],
  loading = false,
  currencySymbol = '₹',
  onRefresh,
}) {
  const [expandedTicker, setExpandedTicker] = useState(alphaHarvests[0]?.ticker || null);

  if (loading) {
    return (
      <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-8 text-center space-y-3">
        <div className="h-8 w-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-bold text-white">Running Alpha Harvest & Profit Maximization Models…</p>
        <p className="text-xs text-slate-400">Computing House Money free-rides, Chandelier trailing stops, and laddered profit targets</p>
      </div>
    );
  }

  if (!alphaHarvests || alphaHarvests.length === 0) {
    return (
      <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-8 text-center space-y-2">
        <Target className="h-10 w-10 text-slate-500 mx-auto mb-2" />
        <h3 className="text-base font-bold text-white">No Winning Positions Yet</h3>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Once your holdings move into positive territory (+0.5% or higher), the Alpha Desk activates systematic profit-harvesting, House Money zero-risk calculators, and trailing stops.
        </p>
      </div>
    );
  }

  const totalProfit = alphaHarvests.reduce((acc, h) => acc + (h.profit_amount || 0), 0);
  const eligibleHouseMoneyCount = alphaHarvests.filter(h => h.house_money?.eligible).length;
  const eligibleCapitalExtraction = alphaHarvests.reduce((acc, h) => acc + (h.house_money?.eligible ? (h.house_money?.capital_extracted || 0) : 0), 0);

  return (
    <div className="space-y-6">
      {/* Overview Metric Banner */}
      <div className="rounded-2xl bg-gradient-to-br from-emerald-500/10 via-indigo-500/5 to-transparent border border-emerald-500/20 p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <p className="text-[10px] font-bold text-emerald-300 uppercase tracking-wider">Alpha Harvest & Profit Maximization</p>
            </div>
            <h2 className="text-xl font-black text-white">
              +{currencySymbol}{totalProfit.toLocaleString('en-IN', { maximumFractionDigits: 0 })} Total Unrealized Gains
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Across {alphaHarvests.length} profitable {alphaHarvests.length === 1 ? 'position' : 'positions'}. Systematic profit taking eliminates round-tripping and lets winners run risk-free.
            </p>
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.08] text-white text-xs font-bold rounded-lg transition cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Re-audit Targets
            </button>
          )}
        </div>

        {/* 4 Quick Stat Tiles */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-white/[0.06]">
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">House Money Eligible</p>
            <p className="text-sm font-black text-emerald-400">{eligibleHouseMoneyCount} Positions</p>
            <p className="text-[9px] text-slate-500">100% principal extraction</p>
          </div>
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Extractable Capital</p>
            <p className="text-sm font-black text-indigo-400">{currencySymbol}{eligibleCapitalExtraction.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</p>
            <p className="text-[9px] text-slate-500">Free cash back in hand</p>
          </div>
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Trailing Stops Active</p>
            <p className="text-sm font-black text-white">{alphaHarvests.length} Positions</p>
            <p className="text-[9px] text-slate-500">Volatility-guarded exits</p>
          </div>
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Pyramid Scalable</p>
            <p className="text-sm font-black text-emerald-400">{alphaHarvests.filter(h => h.pyramiding?.is_recommended).length} Leaders</p>
            <p className="text-[9px] text-slate-500">Safe anti-martingale adds</p>
          </div>
        </div>
      </div>

      {/* Individual Alpha Holding Cards */}
      <div className="space-y-4">
        {alphaHarvests.map((h) => {
          const isExpanded = expandedTicker === h.ticker;
          const shortTicker = h.ticker.replace('.NS', '').replace('.BO', '');
          const isHouseEligible = h.house_money?.eligible;
          const stopStatus = h.trailing_stop?.status;

          const stopBadgeColor = stopStatus === 'SECURE_HOLD'
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : stopStatus === 'APPROACHING_STOP'
            ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
            : 'bg-rose-500/10 border-rose-500/20 text-rose-400';

          return (
            <div
              key={h.ticker}
              className="rounded-2xl border border-white/[0.08] bg-white/[0.02] transition-all duration-200 overflow-hidden hover:border-emerald-500/30"
            >
              {/* Card Header Accordion */}
              <div
                onClick={() => setExpandedTicker(isExpanded ? null : h.ticker)}
                className="p-4 flex items-center justify-between gap-3 cursor-pointer select-none hover:bg-white/[0.02] transition"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="h-10 w-10 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 flex items-center justify-center font-black text-xs shrink-0">
                    {shortTicker.slice(0, 3)}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-black text-sm text-white">{shortTicker}</span>
                      <span className="text-[10px] text-slate-400 font-mono">({h.sector})</span>
                      {isHouseEligible && (
                        <span className="text-[9px] font-bold px-2 py-0.5 rounded-full border bg-indigo-500/15 border-indigo-500/30 text-indigo-300 uppercase tracking-wider flex items-center gap-1">
                          <Lock className="h-2.5 w-2.5" /> House Money Ready
                        </span>
                      )}
                      <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider ${stopBadgeColor}`}>
                        {h.trailing_stop?.label}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 truncate mt-0.5">
                      Chandelier Stop at {currencySymbol}{h.trailing_stop?.stop_price?.toLocaleString('en-IN')} (+{h.trailing_stop?.protected_profit_pct}% protected profit)
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4 shrink-0">
                  <div className="text-right">
                    <p className="text-xs font-black text-emerald-400">
                      +{h.profit_pct}% (+{currencySymbol}{h.profit_amount?.toLocaleString('en-IN', { maximumFractionDigits: 0 })})
                    </p>
                    <p className="text-[10px] text-slate-400">
                      Val: <span className="font-bold text-slate-200">{currencySymbol}{h.current_value?.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                    </p>
                  </div>
                  {isExpanded ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
                </div>
              </div>

              {/* Expanded Diagnostic Body */}
              {isExpanded && (
                <div className="p-4 pt-0 border-t border-white/[0.05] space-y-4">
                  {/* House Money Free-Ride Card */}
                  <div className="mt-3 rounded-xl bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-transparent border border-indigo-500/25 p-4 space-y-2.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-indigo-300 text-xs font-bold">
                        <Sparkles className="h-4 w-4 text-indigo-400" />
                        <span>The &quot;House Money&quot; (Zero-Risk Free-Ride) Protocol</span>
                      </div>
                      <span className="text-[9px] bg-indigo-500/20 text-indigo-200 font-bold px-2 py-0.5 rounded-full uppercase">
                        Principal Protection
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {h.house_money?.action_text}
                    </p>
                    {isHouseEligible && (
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-indigo-500/20 text-center">
                        <div className="p-2 rounded-lg bg-black/40 border border-white/[0.04]">
                          <p className="text-[8px] text-slate-400 uppercase font-bold">Sell Shares</p>
                          <p className="text-xs font-black text-white">{h.house_money?.shares_to_sell} sh</p>
                        </div>
                        <div className="p-2 rounded-lg bg-black/40 border border-white/[0.04]">
                          <p className="text-[8px] text-slate-400 uppercase font-bold">Cash Extracted</p>
                          <p className="text-xs font-black text-indigo-400">+{currencySymbol}{h.house_money?.capital_extracted?.toLocaleString('en-IN')}</p>
                        </div>
                        <div className="p-2 rounded-lg bg-black/40 border border-white/[0.04]">
                          <p className="text-[8px] text-slate-400 uppercase font-bold">Free Shares Kept</p>
                          <p className="text-xs font-black text-emerald-400">{h.house_money?.remaining_free_shares} sh</p>
                        </div>
                        <div className="p-2 rounded-lg bg-black/40 border border-white/[0.04]">
                          <p className="text-[8px] text-slate-400 uppercase font-bold">Downside Capital Risk</p>
                          <p className="text-xs font-black text-emerald-400">0.00% (Zero)</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 2-Column Grid: 4-Tier Laddered Escalator vs Chandelier Stop */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* 4-Tier Laddered Profit Targets */}
                    <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3.5 space-y-2.5">
                      <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-bold">
                        <Target className="h-4 w-4" />
                        <span>4-Tier Laddered Profit Targets</span>
                      </div>
                      <div className="space-y-2">
                        {h.ladder_escalator?.tiers?.map((tier) => (
                          <div
                            key={tier.tier}
                            className={`p-2.5 rounded-lg border text-[11px] flex items-center justify-between gap-2 ${
                              tier.status === 'PASSED'
                                ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200'
                                : 'bg-black/40 border-white/[0.04] text-slate-400'
                            }`}
                          >
                            <div>
                              <div className="flex items-center gap-1.5">
                                <span className="font-bold text-white">{tier.name}</span>
                                {tier.status === 'PASSED' && (
                                  <span className="text-[8px] font-black bg-emerald-500/20 text-emerald-300 px-1.5 py-0.2 rounded">LOCKED</span>
                                )}
                              </div>
                              <p className="text-[9px] text-slate-400 mt-0.5">{tier.condition} · {tier.action}</p>
                            </div>
                            <div className="text-right shrink-0">
                              <span className="font-mono font-bold text-white">{currencySymbol}{tier.target_price?.toLocaleString('en-IN')}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Chandelier / ATR Trailing Stop Sentinel */}
                    <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3.5 space-y-2.5">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-amber-400 text-xs font-bold">
                          <ShieldCheck className="h-4 w-4" />
                          <span>Chandelier Trailing Stop Sentinel</span>
                        </div>
                        <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border uppercase ${stopBadgeColor}`}>
                          {h.trailing_stop?.label}
                        </span>
                      </div>
                      <div className="space-y-2 text-[11px]">
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Volatility Trailing Stop Price:</span>
                          <span className="font-black text-amber-400">{currencySymbol}{h.trailing_stop?.stop_price?.toLocaleString('en-IN')}</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Current Cushion to Stop:</span>
                          <span className="font-bold text-white">+{h.trailing_stop?.distance_pct}% ({currencySymbol}{h.trailing_stop?.distance_rupees?.toLocaleString('en-IN')})</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Protected Profit at Stop Exit:</span>
                          <span className="font-black text-emerald-400">+{h.trailing_stop?.protected_profit_pct}% above buy cost</span>
                        </div>
                        <div className="p-2 rounded-lg bg-black/40 border border-white/[0.04] text-[10px] text-slate-400 leading-relaxed">
                          🛡️ {h.trailing_stop?.recommendation}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Pyramiding Sizing Plan */}
                  <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3.5 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5 text-indigo-400 text-xs font-bold">
                        <Rocket className="h-4 w-4" />
                        <span>Winner Pyramiding Engine (Anti-Martingale Scaling)</span>
                      </div>
                      <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border uppercase ${
                        h.pyramiding?.is_recommended ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30' : 'bg-slate-500/10 text-slate-400 border-slate-500/20'
                      }`}>
                        {h.pyramiding?.is_recommended ? 'Pyramid Approved' : 'Hold Current Size'}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-300 leading-relaxed">
                      {h.pyramiding?.advice}
                    </p>
                    {h.pyramiding?.is_recommended && (
                      <div className="flex flex-wrap items-center gap-4 text-[10px] text-slate-400 pt-1">
                        <span>Recommended Add: <strong className="text-white">+{h.pyramiding?.pyramid_qty} shares</strong></span>
                        <span>Capital Required: <strong className="text-white">{currencySymbol}{h.pyramiding?.pyramid_capital?.toLocaleString('en-IN')}</strong></span>
                        <span>New Average Cost: <strong className="text-emerald-400">{currencySymbol}{h.pyramiding?.new_avg_cost}</strong></span>
                        <span>Stop Cushion: <strong className="text-emerald-400">+{h.pyramiding?.safety_margin_to_stop_pct}%</strong></span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
