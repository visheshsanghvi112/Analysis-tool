'use client';

import React, { useState } from 'react';
import {
  ShieldAlert, AlertTriangle, TrendingDown, Clock, ArrowRight,
  ShieldCheck, RefreshCw, Zap, ExternalLink, ChevronDown, ChevronUp,
  Scale, Layers, DollarSign, Ban
} from 'lucide-react';

const fmt = (n, d = 2) => n == null ? 'N/A' : Number(n).toFixed(d);
const fmtINR = (n) => n == null ? 'N/A' : `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;

export default function PortfolioLossRecoveryDesk({
  remediations = [],
  loading = false,
  currencySymbol = '₹',
  onRefresh,
}) {
  const [expandedTicker, setExpandedTicker] = useState(remediations[0]?.ticker || null);

  if (loading) {
    return (
      <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-8 text-center space-y-3">
        <div className="h-8 w-8 border-2 border-rose-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-bold text-white">Running Loss Remediation & The Great Filter Diagnostics…</p>
        <p className="text-xs text-slate-400">Auditing ROE, debt leverage, breakeven clocks, and tax-loss harvesting shields</p>
      </div>
    );
  }

  if (!remediations || remediations.length === 0) {
    return (
      <div className="rounded-2xl bg-white/[0.02] border border-white/[0.06] p-8 text-center space-y-2">
        <ShieldCheck className="h-10 w-10 text-emerald-400 mx-auto mb-2" />
        <h3 className="text-base font-bold text-white">Zero Underwater Positions Detected</h3>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          None of your active holdings are currently sitting at a significant loss. Your entire portfolio is performing at or above cost.
        </p>
      </div>
    );
  }

  const totalLoss = remediations.reduce((acc, r) => acc + (r.loss_amount || 0), 0);
  const totalTaxShield = remediations.reduce((acc, r) => acc + (r.tax_harvesting?.tax_shield_rupees || 0), 0);
  const trapsCount = remediations.filter(r => r.filter?.status === 'STRUCTURAL_VALUE_TRAP').length;
  const dcaApprovedCount = remediations.filter(r => r.filter?.directive === 'DCA_APPROVED').length;

  return (
    <div className="space-y-6">
      {/* Overview Metric Banner */}
      <div className="rounded-2xl bg-gradient-to-br from-rose-500/10 via-purple-500/5 to-transparent border border-rose-500/20 p-5">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="h-2 w-2 rounded-full bg-rose-400 animate-pulse" />
              <p className="text-[10px] font-bold text-rose-300 uppercase tracking-wider">Capital Defense & Loss Remediation</p>
            </div>
            <h2 className="text-xl font-black text-white">
              {currencySymbol}{totalLoss.toLocaleString('en-IN', { maximumFractionDigits: 0 })} Total Unrealized Loss
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Across {remediations.length} underwater {remediations.length === 1 ? 'position' : 'positions'}. The Great Filter separates quality compounders on sale from structural value traps.
            </p>
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.06] hover:bg-white/[0.1] border border-white/[0.08] text-white text-xs font-bold rounded-lg transition cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Re-audit Losses
            </button>
          )}
        </div>

        {/* 4 Quick Stat Tiles */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-white/[0.06]">
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Tax Shield Unlocked</p>
            <p className="text-sm font-black text-emerald-400">+{currencySymbol}{totalTaxShield.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</p>
            <p className="text-[9px] text-slate-500">Immediate fiscal savings</p>
          </div>
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">DCA Approved</p>
            <p className="text-sm font-black text-emerald-400">{dcaApprovedCount} Stocks</p>
            <p className="text-[9px] text-slate-500">Quality on sale</p>
          </div>
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Firewall Traps</p>
            <p className="text-sm font-black text-rose-400">{trapsCount} Stocks</p>
            <p className="text-[9px] text-slate-500">Do not average down</p>
          </div>
          <div className="p-2.5 rounded-xl bg-white/[0.02] border border-white/[0.04]">
            <p className="text-[9px] text-slate-400 uppercase font-bold">Rotation Candidates</p>
            <p className="text-sm font-black text-indigo-400">{remediations.filter(r => r.breakeven_horizon?.years_saved_by_rotation > 1.0).length} Stocks</p>
            <p className="text-[9px] text-slate-500">Fast recovery alternatives</p>
          </div>
        </div>
      </div>

      {/* Individual Loss Holding Cards */}
      <div className="space-y-4">
        {remediations.map((rem) => {
          const isExpanded = expandedTicker === rem.ticker;
          const shortTicker = rem.ticker.replace('.NS', '').replace('.BO', '');
          const isApproved = rem.filter?.directive === 'DCA_APPROVED';
          const isTrap = rem.filter?.status === 'STRUCTURAL_VALUE_TRAP';

          const badgeBg = isApproved
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : isTrap
            ? 'bg-rose-500/10 border-rose-500/20 text-rose-400'
            : 'bg-amber-500/10 border-amber-500/20 text-amber-400';

          return (
            <div
              key={rem.ticker}
              className={`rounded-2xl border transition-all duration-200 overflow-hidden ${
                isTrap
                  ? 'border-rose-500/25 bg-[#0e0a0d]'
                  : isApproved
                  ? 'border-emerald-500/25 bg-[#0a0f0d]'
                  : 'border-white/[0.08] bg-white/[0.02]'
              }`}
            >
              {/* Card Header Accordion */}
              <div
                onClick={() => setExpandedTicker(isExpanded ? null : rem.ticker)}
                className="p-4 flex items-center justify-between gap-3 cursor-pointer select-none hover:bg-white/[0.02] transition"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`h-10 w-10 rounded-xl flex items-center justify-center font-black text-xs shrink-0 border ${
                    isApproved ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300' :
                    isTrap ? 'bg-rose-500/15 border-rose-500/30 text-rose-300' :
                    'bg-amber-500/15 border-amber-500/30 text-amber-300'
                  }`}>
                    {shortTicker.slice(0, 3)}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-black text-sm text-white">{shortTicker}</span>
                      <span className="text-[10px] text-slate-400 font-mono">({rem.sector})</span>
                      <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider ${badgeBg}`}>
                        {rem.filter?.label}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 truncate mt-0.5">
                      {rem.filter?.directive === 'DCA_APPROVED' ? 'Approved for Staged Recovery Averaging' :
                       rem.filter?.directive === 'STRICT_FIREWALL_DO_NOT_AVERAGE' ? 'Firewall Engaged: High Risk Sunk-Cost Trap' :
                       'Wait for Price Base Formation'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4 shrink-0">
                  <div className="text-right">
                    <p className="text-xs font-black text-rose-400">
                      {rem.loss_pct}% ({currencySymbol}{rem.loss_amount?.toLocaleString('en-IN', { maximumFractionDigits: 0 })})
                    </p>
                    <p className="text-[10px] text-slate-400">
                      Req. Gain: <span className="font-bold text-amber-400">+{rem.req_gain_to_breakeven_pct}%</span>
                    </p>
                  </div>
                  {isExpanded ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
                </div>
              </div>

              {/* Expanded Deep Diagnostic Body */}
              {isExpanded && (
                <div className="p-4 pt-0 border-t border-white/[0.05] space-y-4">
                  {/* The Great Filter Advice Alert */}
                  <div className={`p-3.5 rounded-xl border mt-3 ${
                    isTrap ? 'bg-rose-500/8 border-rose-500/20 text-rose-200' :
                    isApproved ? 'bg-emerald-500/8 border-emerald-500/20 text-emerald-200' :
                    'bg-amber-500/8 border-amber-500/20 text-amber-200'
                  }`}>
                    <div className="flex items-start gap-2.5">
                      {isTrap ? <Ban className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" /> :
                       isApproved ? <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" /> :
                       <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />}
                      <div className="space-y-1">
                        <p className="text-xs font-bold text-white">The Great Filter Verdict: {rem.filter?.label} (Quality Score: {rem.filter?.quality_score}/100)</p>
                        <p className="text-[11px] leading-relaxed text-slate-300">{rem.filter?.advice}</p>
                        {rem.filter?.breakdown?.length > 0 && (
                          <div className="flex flex-wrap gap-1.5 mt-2">
                            {rem.filter.breakdown.map((item, idx) => (
                              <span key={idx} className="text-[9px] bg-black/40 border border-white/[0.08] text-slate-300 px-2 py-0.5 rounded-md">
                                {item}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 2-Column Grid: Breakeven Horizon Clock vs Tax-Loss Harvesting Shield */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Breakeven Horizon Clock */}
                    <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3.5 space-y-2.5">
                      <div className="flex items-center gap-1.5 text-indigo-400 text-xs font-bold">
                        <Clock className="h-4 w-4" />
                        <span>Opportunity Cost & Breakeven Clock</span>
                      </div>
                      <div className="space-y-2 text-[11px]">
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Stock Est. CAGR:</span>
                          <span className="font-bold text-white">{rem.breakeven_horizon?.stock_projected_cagr_pct}% p.a.</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Years to Breakeven in {shortTicker}:</span>
                          <span className="font-black text-rose-400">{rem.breakeven_horizon?.stock_years_to_breakeven} Years</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Years if Rotated to Leader (13.5% CAGR):</span>
                          <span className="font-black text-emerald-400">{rem.breakeven_horizon?.benchmark_years_to_breakeven} Years</span>
                        </div>
                        {rem.breakeven_horizon?.years_saved_by_rotation > 0 && (
                          <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-[10px] font-medium leading-relaxed">
                            💡 {rem.breakeven_horizon?.opportunity_cost_verdict}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Tax-Loss Harvesting Shield */}
                    <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] p-3.5 space-y-2.5">
                      <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-bold">
                        <Scale className="h-4 w-4" />
                        <span>Tax-Loss Harvesting Shield (Fiscal Value)</span>
                      </div>
                      <div className="space-y-2 text-[11px]">
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Immediate Tax Deduction Value:</span>
                          <span className="font-black text-emerald-400">+{currencySymbol}{rem.tax_harvesting?.tax_shield_rupees?.toLocaleString('en-IN')}</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Regime:</span>
                          <span className="font-mono text-slate-300">{rem.tax_harvesting?.tax_jurisdiction}</span>
                        </div>
                        <div className="flex justify-between items-center text-slate-300">
                          <span className="text-slate-400">Replacement Sector Proxy ETF:</span>
                          <span className="font-bold text-white bg-white/[0.06] px-1.5 py-0.5 rounded">{rem.tax_harvesting?.recommended_proxy_etf}</span>
                        </div>
                        <p className="text-[10px] text-slate-400 leading-relaxed pt-1">
                          {rem.tax_harvesting?.harvesting_note}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* 3-Tranche Staged DCA Execution Plan (if approved) OR Strict Firewall Warning (if trap) */}
                  {isApproved ? (
                    <div className="rounded-xl bg-white/[0.02] border border-emerald-500/20 p-3.5 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 text-emerald-400 text-xs font-bold">
                          <Layers className="h-4 w-4" />
                          <span>3-Tranche Volatility-Guided Recovery Plan</span>
                        </div>
                        <span className="text-[9px] bg-emerald-500/20 text-emerald-300 font-bold px-2 py-0.5 rounded-full uppercase">
                          Disciplined DCA
                        </span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                        {rem.staged_dca_plan?.tranches?.map((tr) => (
                          <div key={tr.tranche} className="p-2.5 rounded-lg bg-black/40 border border-white/[0.06] space-y-1">
                            <div className="flex items-center justify-between text-[10px]">
                              <span className="font-bold text-white">Tranche {tr.tranche}: {tr.name}</span>
                              <span className="text-emerald-400 font-bold">{tr.weight_pct}% size</span>
                            </div>
                            <p className="text-[9px] text-slate-400">{tr.trigger}</p>
                            <div className="flex justify-between items-center pt-1 text-[10px]">
                              <span className="text-slate-400">Qty: <strong className="text-white">{tr.qty} sh</strong></span>
                              <span className="text-slate-400">Capital: <strong className="text-white">{currencySymbol}{tr.capital?.toLocaleString('en-IN')}</strong></span>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04] flex flex-wrap items-center justify-between gap-3 text-[10px]">
                        <div>
                          <span className="text-slate-400">Projected New Average Price: </span>
                          <span className="font-bold text-emerald-400">{currencySymbol}{rem.staged_dca_plan?.projected_outcome?.new_avg_price}</span>
                          <span className="text-slate-400 ml-1.5">(-{rem.staged_dca_plan?.projected_outcome?.cost_reduction_pct}% cost basis)</span>
                        </div>
                        <div>
                          <span className="text-slate-400">Hard Invalidation Stop-Loss: </span>
                          <span className="font-bold text-rose-400">{currencySymbol}{rem.staged_dca_plan?.invalidation_stop}</span>
                        </div>
                      </div>
                    </div>
                  ) : isTrap ? (
                    <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 p-4 space-y-2">
                      <div className="flex items-center gap-2 text-rose-400 text-xs font-bold">
                        <Ban className="h-4 w-4" />
                        <span>Averaging Down Disabled by Sunk-Cost Firewall</span>
                      </div>
                      <p className="text-xs text-rose-200 leading-relaxed">
                        Our quantitative model strongly advises <strong>against</strong> allocating additional money to {shortTicker}.
                        High balance sheet leverage or declining return on equity indicate high likelihood of continued value erosion.
                        Recommended action: Harvest the tax loss of {currencySymbol}{rem.tax_harvesting?.tax_shield_rupees?.toLocaleString('en-IN')} and rotate remaining capital into {rem.tax_harvesting?.recommended_leader_proxy}.
                      </p>
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
