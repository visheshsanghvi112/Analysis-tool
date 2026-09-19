'use client';

import { useState, useEffect } from 'react';
import {
  TrendingUp, TrendingDown, CheckCircle2, XCircle,
  HelpCircle, BarChart3, AlertTriangle, Zap, Target,
  Coins, Activity,
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : 'https://stock-analysis-backend-seven.vercel.app');

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────
const fmt = (v, dec = 2) =>
  v == null || isNaN(Number(v)) ? '–' : Number(v).toFixed(dec);

const pct = (v, dec = 2) =>
  v == null || isNaN(Number(v)) ? '–' : `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(dec)}%`;

const formatMoney = (val, curr = '₹') => {
  if (val == null || isNaN(Number(val))) return '–';
  const num = Number(val);
  if (curr === '₹') {
    if (num >= 1e7) return `${curr}${(num / 1e7).toFixed(2)} Cr`;
    if (num >= 1e5) return `${curr}${(num / 1e5).toFixed(2)} L`;
    return `${curr}${num.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  }
  if (num >= 1e6) return `${curr}${(num / 1e6).toFixed(2)} M`;
  return `${curr}${num.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
};

// CAGR comparison bar — ETF vs benchmark
function CAGRBar({ etf, bench, label }) {
  if (etf == null) return null;
  const max = Math.max(Math.abs(etf || 0), Math.abs(bench || 0), 0.01);
  const etfW  = Math.min(100, (Math.abs(etf) / max) * 100);
  const benchW = bench != null ? Math.min(100, (Math.abs(bench) / max) * 100) : 0;
  const etfPos  = etf  >= 0;
  const benchPos = bench != null ? bench >= 0 : true;

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between text-[9px] text-slate-500 mb-1">
        <span className="font-semibold text-slate-300">{label}</span>
        <div className="flex items-center gap-3">
          <span className={etfPos ? 'text-emerald-400' : 'text-rose-400'}>{pct(etf)}</span>
          {bench != null && (
            <span className="text-slate-500">{pct(bench)} bench</span>
          )}
        </div>
      </div>
      <div className="flex items-center gap-1.5 h-2">
        <div className="flex-1 h-full rounded-full bg-white/[0.06] overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${etfPos ? 'bg-emerald-500' : 'bg-rose-500'}`}
            style={{ width: `${etfW}%` }}
          />
        </div>
        {bench != null && (
          <div className="flex-1 h-full rounded-full bg-white/[0.06] overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${benchPos ? 'bg-sky-500/60' : 'bg-orange-500/60'}`}
              style={{ width: `${benchW}%` }}
            />
          </div>
        )}
      </div>
      {bench != null && (
        <div className="flex items-center gap-3 mt-0.5">
          <div className="flex items-center gap-1 text-[8px] text-emerald-400">
            <div className="h-1.5 w-1.5 rounded-full bg-emerald-500" />ETF
          </div>
          <div className="flex items-center gap-1 text-[8px] text-sky-400">
            <div className="h-1.5 w-1.5 rounded-full bg-sky-500/60" />Benchmark
          </div>
        </div>
      )}
    </div>
  );
}

// 6-Point Health Checklist Item
function ChecklistItem({ item }) {
  const { metric, value, condition, passed, note } = item;
  const icon =
    passed === true  ? <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" /> :
    passed === false ? <XCircle      className="h-4 w-4 text-rose-400    shrink-0 mt-0.5" /> :
                       <HelpCircle   className="h-4 w-4 text-amber-400   shrink-0 mt-0.5" />;

  const pillColor =
    passed === true  ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' :
    passed === false ? 'bg-rose-500/10    border-rose-500/20    text-rose-400'    :
                       'bg-amber-500/10   border-amber-500/20   text-amber-400';

  return (
    <div className="flex gap-2.5 py-2.5 border-b border-white/[0.04] last:border-0">
      {icon}
      <div className="min-w-0 flex-1">
        <div className="flex items-start justify-between gap-2 flex-wrap">
          <span className="text-[11px] font-semibold text-slate-200">{metric}</span>
          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${pillColor}`}>
            {value}
          </span>
        </div>
        <p className="text-[9px] text-slate-500 mt-0.5">{note}</p>
        <p className="text-[9px] text-slate-600 mt-0.5">Condition: {condition}</p>
      </div>
    </div>
  );
}

// Yearly returns sparkline-style row chart
function YearlyReturns({ data }) {
  if (!data || data.length === 0) return null;
  const max = Math.max(...data.map(d => Math.abs(d.return_pct)), 0.01);

  return (
    <div>
      <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-3">
        Year-by-Year Returns
      </p>
      <div className="space-y-1.5">
        {data.map((row) => {
          const w = Math.min(100, (Math.abs(row.return_pct) / max) * 100);
          const pos = row.return_pct >= 0;
          return (
            <div key={row.year} className="flex items-center gap-2">
              <span className="text-[9px] text-slate-500 w-9 shrink-0 text-right">{row.year}</span>
              <div className="flex-1 h-3 bg-white/[0.04] rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${pos ? 'bg-emerald-500/70' : 'bg-rose-500/70'}`}
                  style={{ width: `${w}%` }}
                />
              </div>
              <span className={`text-[9px] font-bold w-14 text-right shrink-0 ${pos ? 'text-emerald-400' : 'text-rose-400'}`}>
                {pct(row.return_pct, 1)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// SIP Score meter (0–10)
function SIPMeter({ score, label }) {
  const pct = (score / 10) * 100;
  const color =
    score >= 8 ? 'text-emerald-400' :
    score >= 6 ? 'text-sky-400'     :
    score >= 4 ? 'text-amber-400'   : 'text-rose-400';

  const barColor =
    score >= 8 ? 'from-emerald-500 to-teal-400'    :
    score >= 6 ? 'from-sky-500 to-indigo-400'       :
    score >= 4 ? 'from-amber-500 to-orange-400'     : 'from-rose-500 to-rose-400';

  return (
    <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06] mb-4">
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">
        SIP Suitability Score
      </p>
      <div className="flex items-end justify-between mb-2">
        <span className={`text-3xl font-black tabular-nums ${color}`}>{score}</span>
        <span className="text-slate-500 text-xs">/10</span>
      </div>
      <div className="h-2 rounded-full bg-white/[0.06] overflow-hidden mb-2">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-700`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className={`text-[11px] font-bold ${color}`}>{label}</p>
    </div>
  );
}


// 3-Year Rolling Returns (Consistency & Capital Preservation)
function RollingReturnsCard({ data }) {
  if (!data || data.total_periods == null || data.median_cagr == null) return null;

  const { window_years, median_cagr, min_cagr, max_cagr, current_cagr, positive_periods_pct, total_periods } = data;

  const isProfitable = positive_periods_pct >= 90;
  const isModerate   = positive_periods_pct >= 70;

  // Visual position of median and current within [min, max] range
  const span = Math.max(max_cagr - min_cagr, 0.01);
  const medianPct = Math.min(100, Math.max(0, ((median_cagr - min_cagr) / span) * 100));
  const currentPct = Math.min(100, Math.max(0, ((current_cagr - min_cagr) / span) * 100));

  return (
    <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
      <div className="flex items-center justify-between mb-3">
        <div>
          <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
            {window_years}-Year Rolling Returns
          </p>
          <p className="text-[9px] text-slate-500 mt-0.5">
            Evaluated across {total_periods} rolling 3-year holding periods
          </p>
        </div>
        <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${
          isProfitable
            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
            : isModerate
            ? 'bg-sky-500/10 border-sky-500/30 text-sky-400'
            : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
        }`}>
          {positive_periods_pct}% Profitable
        </span>
      </div>

      {/* Grid of Key Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Median 3Y CAGR</p>
          <p className={`text-xs sm:text-sm font-bold mt-0.5 ${median_cagr >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {pct(median_cagr)}
          </p>
        </div>
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Worst 3Y Period</p>
          <p className={`text-xs sm:text-sm font-bold mt-0.5 ${min_cagr >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {pct(min_cagr)}
          </p>
        </div>
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Best 3Y Period</p>
          <p className="text-xs sm:text-sm font-bold mt-0.5 text-emerald-400">
            {pct(max_cagr)}
          </p>
        </div>
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Current 3Y CAGR</p>
          <p className={`text-xs sm:text-sm font-bold mt-0.5 ${current_cagr >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {pct(current_cagr)}
          </p>
        </div>
      </div>

      {/* Visual Min -> Median -> Max Range Track */}
      <div className="pt-2 border-t border-white/[0.04]">
        <div className="flex items-center justify-between text-[9px] text-slate-500 mb-1.5">
          <span>Worst: <strong className="text-rose-400">{pct(min_cagr)}</strong></span>
          <span>Median: <strong className="text-emerald-400">{pct(median_cagr)}</strong></span>
          <span>Best: <strong className="text-emerald-400">{pct(max_cagr)}</strong></span>
        </div>
        <div className="relative h-2 w-full bg-white/[0.06] rounded-full overflow-visible">
          <div
            className="absolute top-0 bottom-0 rounded-full bg-gradient-to-r from-rose-500/30 via-amber-500/30 to-emerald-500/40"
            style={{ left: '0%', width: '100%' }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-3.5 w-1 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"
            style={{ left: `${medianPct}%` }}
            title={`Median: ${pct(median_cagr)}`}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-2.5 w-2.5 rounded-full bg-sky-400 border border-white shadow-[0_0_8px_rgba(56,189,248,0.8)]"
            style={{ left: `${currentPct}%` }}
            title={`Current: ${pct(current_cagr)}`}
          />
        </div>
        <div className="flex items-center justify-between text-[8px] text-slate-500 mt-1.5">
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 inline-block" /> Green line = Median
          </span>
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-sky-400 inline-block" /> Blue dot = Current
          </span>
        </div>
      </div>
    </div>
  );
}

// Top Holdings and Sector Exposure Breakdown
function HoldingsAndSectorsCard({ holdings, sectors }) {
  const hasHoldings = holdings && holdings.length > 0;
  const hasSectors  = sectors && sectors.length > 0;
  const [activeTab, setActiveTab] = useState(hasHoldings ? 'holdings' : 'sectors');

  if (!hasHoldings && !hasSectors) return null;

  const currentTab = (!hasHoldings && hasSectors) ? 'sectors' : (hasHoldings && !hasSectors) ? 'holdings' : activeTab;

  return (
    <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
      <div className="flex items-center justify-between mb-4">
        <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
          Underlying Portfolio Breakdown
        </p>
        <div className="flex items-center rounded-lg bg-white/[0.04] p-0.5 border border-white/[0.06]">
          {hasHoldings && (
            <button
              onClick={() => setActiveTab('holdings')}
              className={`px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all ${
                currentTab === 'holdings'
                  ? 'bg-violet-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Top Holdings ({holdings.length})
            </button>
          )}
          {hasSectors && (
            <button
              onClick={() => setActiveTab('sectors')}
              className={`px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all ${
                currentTab === 'sectors'
                  ? 'bg-violet-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Sectors ({sectors.length})
            </button>
          )}
        </div>
      </div>

      {/* Holdings Tab */}
      {currentTab === 'holdings' && hasHoldings && (
        <div className="space-y-2">
          {holdings.map((h, i) => {
            const w = Math.min(100, (h.weight_pct || 0) * 2.5);
            return (
              <div key={h.symbol || h.name || i} className="group">
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <div className="flex items-center gap-2 min-w-0 pr-2">
                    <span className="text-[9px] font-mono text-slate-500 w-4 shrink-0 text-right">
                      {i + 1}.
                    </span>
                    <span className="font-semibold text-slate-200 truncate group-hover:text-white transition-colors">
                      {h.name}
                    </span>
                    {h.symbol && (
                      <span className="text-[9px] font-mono text-slate-500 shrink-0">
                        {h.symbol}
                      </span>
                    )}
                  </div>
                  <span className="font-bold text-emerald-400 tabular-nums shrink-0">
                    {h.weight_pct != null ? `${Number(h.weight_pct).toFixed(1)}%` : '–'}
                  </span>
                </div>
                <div className="ml-6 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-500 to-indigo-400 transition-all duration-500"
                    style={{ width: `${w}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Sectors Tab */}
      {currentTab === 'sectors' && hasSectors && (
        <div className="space-y-2">
          {sectors.map((s, i) => {
            const w = Math.min(100, (s.weight_pct || 0) * 1.5);
            return (
              <div key={s.sector || i} className="group">
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="font-semibold text-slate-200 truncate group-hover:text-white transition-colors">
                    {s.sector}
                  </span>
                  <span className="font-bold text-sky-400 tabular-nums shrink-0">
                    {s.weight_pct != null ? `${Number(s.weight_pct).toFixed(1)}%` : '–'}
                  </span>
                </div>
                <div className="h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-sky-500 to-teal-400 transition-all duration-500"
                    style={{ width: `${w}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


// Realized Historical SIP vs Lump Sum Simulator
function HistoricalSIPCard({ data, curr }) {
  if (!data || Object.keys(data).length === 0) return null;

  const availableHorizons = Object.keys(data);
  const [selectedHorizon, setSelectedHorizon] = useState(
    availableHorizons.includes('3Y') ? '3Y' : availableHorizons[0]
  );

  const curData = data[selectedHorizon];
  if (!curData) return null;

  const {
    months_count,
    monthly_investment,
    total_invested,
    sip_value,
    sip_gain_pct,
    sip_xirr_pct,
    lump_value,
    lump_gain_pct,
    lump_cagr,
  } = curData;

  const sipProfit = sip_value - total_invested;
  const maxVal = Math.max(sip_value, lump_value, total_invested * 1.2, 1);
  const investedW = Math.min(100, (total_invested / maxVal) * 100);
  const sipW = Math.min(100, (sip_value / maxVal) * 100);
  const lumpW = Math.min(100, (lump_value / maxVal) * 100);

  return (
    <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
      <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
            <Coins className="h-3.5 w-3.5 text-violet-400" />
          </div>
          <div>
            <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
              Historical SIP vs Lump Sum
            </p>
            <p className="text-[9px] text-slate-500">
              Realized returns investing {curr}{monthly_investment?.toLocaleString()}/mo on the 1st of each month
            </p>
          </div>
        </div>

        {/* Horizon Toggle */}
        <div className="flex items-center rounded-lg bg-white/[0.04] p-0.5 border border-white/[0.06]">
          {availableHorizons.map((h) => (
            <button
              key={h}
              onClick={() => setSelectedHorizon(h)}
              className={`px-2.5 py-1 rounded-md text-[10px] font-semibold transition-all ${
                selectedHorizon === h
                  ? 'bg-violet-500 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {h} ({data[h]?.months_count}m)
            </button>
          ))}
        </div>
      </div>

      {/* Grid of SIP Performance */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Total Invested</p>
          <p className="text-xs sm:text-sm font-bold mt-0.5 text-slate-200">
            {formatMoney(total_invested, curr)}
          </p>
          <p className="text-[8px] text-slate-500 mt-0.5">{months_count} installments</p>
        </div>

        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">SIP Current Value</p>
          <p className="text-xs sm:text-sm font-bold mt-0.5 text-emerald-400">
            {formatMoney(sip_value, curr)}
          </p>
          <p className="text-[8px] text-emerald-400/80 mt-0.5">+{formatMoney(sipProfit, curr)}</p>
        </div>

        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">SIP Total Gain</p>
          <p className={`text-xs sm:text-sm font-bold mt-0.5 ${sip_gain_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {pct(sip_gain_pct)}
          </p>
          <p className="text-[8px] text-slate-500 mt-0.5">Absolute return</p>
        </div>

        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04] text-center">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Annualized XIRR</p>
          <p className={`text-xs sm:text-sm font-bold mt-0.5 ${sip_xirr_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
            {sip_xirr_pct != null ? `${sip_xirr_pct}%` : '–'}
          </p>
          <p className="text-[8px] text-slate-500 mt-0.5">Money-weighted rate</p>
        </div>
      </div>

      {/* Lump Sum Comparison Visualizer */}
      <div className="pt-3 border-t border-white/[0.04] space-y-2">
        <div className="flex items-center justify-between text-[9px] text-slate-400 font-semibold mb-1">
          <span>Wealth Accumulation Comparison</span>
          <span className="text-[8px] text-slate-500">Invested {formatMoney(total_invested, curr)}</span>
        </div>

        {/* Invested bar */}
        <div>
          <div className="flex items-center justify-between text-[8px] text-slate-500 mb-0.5">
            <span>Capital Invested</span>
            <span>{formatMoney(total_invested, curr)}</span>
          </div>
          <div className="h-1.5 w-full bg-white/[0.04] rounded-full overflow-hidden">
            <div className="h-full bg-slate-500/60 rounded-full" style={{ width: `${investedW}%` }} />
          </div>
        </div>

        {/* SIP Value bar */}
        <div>
          <div className="flex items-center justify-between text-[8px] text-emerald-400 mb-0.5">
            <span className="font-semibold">Monthly SIP Result</span>
            <span className="font-bold">{formatMoney(sip_value, curr)} ({pct(sip_gain_pct)})</span>
          </div>
          <div className="h-1.5 w-full bg-white/[0.04] rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full" style={{ width: `${sipW}%` }} />
          </div>
        </div>

        {/* Lump Sum Value bar */}
        <div>
          <div className="flex items-center justify-between text-[8px] text-sky-400 mb-0.5">
            <span className="font-semibold">Day-1 Lump Sum Result</span>
            <span className="font-bold">{formatMoney(lump_value, curr)} ({pct(lump_gain_pct)} | {lump_cagr != null ? `${lump_cagr}% CAGR` : ''})</span>
          </div>
          <div className="h-1.5 w-full bg-white/[0.04] rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-sky-500 to-indigo-400 rounded-full" style={{ width: `${lumpW}%` }} />
          </div>
        </div>
      </div>

      <p className="text-[8px] text-slate-500 mt-2 leading-relaxed">
        💡 <strong>Key Takeaway:</strong> SIP averages out volatility and protects against market peaks. Lump sum yields higher returns during persistent bull markets, but carries higher timing risk.
      </p>
    </div>
  );
}


// Liquidity & Execution Quality Card
function LiquidityCard({ data, curr }) {
  if (!data || !data.adv_30d) return null;

  const {
    adv_30d,
    daily_turnover,
    daily_turnover_display,
    liquidity_grade,
    liquidity_advice,
    liquidity_color,
  } = data;

  const badgeStyle =
    liquidity_color === 'emerald'
      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
      : liquidity_color === 'amber'
      ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
      : 'bg-rose-500/10 border-rose-500/30 text-rose-400';

  return (
    <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <div className="h-7 w-7 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center">
            <Activity className="h-3.5 w-3.5 text-sky-400" />
          </div>
          <div>
            <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">
              Liquidity & Execution Quality
            </p>
            <p className="text-[9px] text-slate-500">
              30-day trading activity and slippage assessment
            </p>
          </div>
        </div>
        <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full border ${badgeStyle}`}>
          {liquidity_grade}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04]">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">30-Day ADV (Volume)</p>
          <p className="text-xs sm:text-sm font-bold mt-0.5 text-slate-200">
            {Number(adv_30d).toLocaleString()} <span className="text-[9px] text-slate-500 font-normal">shares/day</span>
          </p>
        </div>
        <div className="rounded-lg p-2.5 bg-white/[0.02] border border-white/[0.04]">
          <p className="text-[8px] text-slate-500 uppercase tracking-wide">Daily Turnover</p>
          <p className="text-xs sm:text-sm font-bold mt-0.5 text-slate-200">
            {daily_turnover_display || formatMoney(daily_turnover, curr)} <span className="text-[9px] text-slate-500 font-normal">/day</span>
          </p>
        </div>
      </div>

      <div className="rounded-lg p-2 bg-white/[0.02] border border-white/[0.04] flex items-start gap-2">
        <Zap className="h-3.5 w-3.5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-[9px] text-slate-400 leading-relaxed">
          <strong className="text-slate-300">Execution Guidance: </strong>
          {liquidity_advice}
        </div>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────
// Main ETF Panel
// ─────────────────────────────────────────────────────────
export default function ETFLongTermPanel({ ticker }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!ticker) return;
    setData(null);
    setError(null);
    setLoading(true);

    fetch(`${API_BASE_URL}/api/etf-analysis?ticker=${ticker}`)
      .then(r => r.json().then(j => ({ ok: r.ok, body: j })))
      .then(({ ok, body }) => {
        if (!ok) throw new Error(body.detail || 'ETF analysis failed');
        setData(body);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading) {
    return (
      <div className="glass-card p-6 flex flex-col items-center justify-center gap-3 py-14 text-slate-500">
        <BarChart3 className="h-6 w-6 animate-pulse text-violet-400" />
        <span className="text-xs">Analysing ETF long-term data…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card p-6 flex items-center gap-3 text-rose-400 text-sm">
        <AlertTriangle className="h-5 w-5 shrink-0" />
        <span>{error}</span>
      </div>
    );
  }

  if (!data) return null;

  const { long_name, etf_meta, performance, health_checklist,
          health_score_label, sip_score, sip_label,
          yearly_returns, currency_symbol,
          top_holdings, sector_exposure, rolling_returns,
          historical_sip, liquidity,
          market_structure_status } = data;

  const curr = currency_symbol || '₹';

  return (
    <div className="glass-card p-4 sm:p-6 space-y-5">

      {/* Header */}
      <div className="flex items-start gap-3">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-violet-500/20 to-indigo-500/20 border border-violet-500/20 flex items-center justify-center shrink-0">
          <TrendingUp className="h-5 w-5 text-violet-400" />
        </div>
        <div className="min-w-0">
          <h3 className="text-sm sm:text-base font-bold text-white leading-tight">{long_name}</h3>
          <p className="text-[10px] text-slate-400 mt-0.5">ETF Long-Term Analysis</p>
          {etf_meta?.benchmark_ticker && (
            <div className="flex flex-col gap-1 mt-1">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-[9px] text-slate-400">
                  Official Benchmark: <span className="font-semibold text-slate-200">{etf_meta?.benchmark_info?.primary_benchmark_name || etf_meta.benchmark_ticker}</span>
                  {etf_meta?.benchmark_info?.benchmark_type && (
                    <span className="ml-1 text-[8px] px-1.5 py-0.2 rounded bg-white/[0.06] text-slate-300 font-medium">
                      {etf_meta.benchmark_info.benchmark_type === 'TOTAL_RETURN_INDEX' ? 'Total Return (TRI)' : etf_meta.benchmark_info.benchmark_type}
                    </span>
                  )}
                </p>
              </div>
              {etf_meta?.benchmark_info?.is_proxy_used && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <p className="text-[9px] text-amber-400/90">
                    Calculation Series: <span className="font-semibold text-amber-300">{etf_meta?.benchmark_info?.active_benchmark_name || etf_meta?.benchmark_info?.active_benchmark_ticker} — Price Return Proxy</span>
                  </p>
                  <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[8px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">
                    ⚠️ Proxy diagnostic only
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="ml-auto text-right shrink-0">
          <p className="text-xs text-slate-500">Current</p>
          <p className="text-base font-bold text-white">
            {curr}{fmt(data.current_price)}
          </p>
        </div>
      </div>

      {/* Benchmark Proxy Notice Callout if active */}
      {etf_meta?.benchmark_info?.is_proxy_used && (
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-500/[0.06] border border-amber-500/20 text-[10px] text-amber-300">
          <HelpCircle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-amber-200">
              Benchmark Proxy Diagnostic Notice: {etf_meta?.benchmark_info?.primary_benchmark_name || etf_meta?.benchmark_ticker} (TRI) vs {etf_meta?.benchmark_info?.active_benchmark_name || etf_meta?.benchmark_info?.active_benchmark_ticker} (Price Return)
            </p>
            <p className="text-amber-300/85 leading-relaxed">
              {etf_meta.benchmark_info.proxy_reason || "Official scheme benchmark time-series unavailable on current data feed (<30 observations)."}
              {" "}This calculation is performed against a <strong>Price Return proxy series</strong> for secondary-market diagnostic purposes only. It is <strong>not</strong> the scheme&apos;s official Total Return Index performance or regulatory NAV tracking error.
            </p>
          </div>
        </div>
      )}

      {/* Market Structure Dislocation Notice Callout if active */}
      {(market_structure_status?.secondary_market_dislocation === 'HIGH' || performance?.secondary_market_dislocation?.is_dislocated) && (
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-500/[0.08] border border-amber-500/25 text-[10px] text-amber-300">
          <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
          <div className="space-y-1 w-full">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <p className="font-semibold text-amber-200">
                Secondary Market Price Dislocation — Premium/(Discount) to iNAV
              </p>
              {performance?.secondary_market_dislocation?.premium_discount_pct != null && (
                <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-amber-500/20 border border-amber-500/30 text-amber-200">
                  {performance.secondary_market_dislocation.premium_discount_pct > 0 ? '+' : ''}
                  {performance.secondary_market_dislocation.premium_discount_pct}% to iNAV
                </span>
              )}
            </div>
            <p className="text-amber-200/90 leading-relaxed">
              <strong>Observation: </strong>
              {market_structure_status?.observation || "Market price is substantially above indicative iNAV."}
            </p>
            <p className="text-amber-300/80 leading-relaxed">
              <strong>Attribution: </strong>
              {market_structure_status?.attribution || "Potential contributors include overseas investment quota limits, trading constraints, retail circuit limits, and secondary-market liquidity dynamics."}
            </p>
          </div>
        </div>
      )}

      {/* ETF Identity Card */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {[
          {
            label: 'AUM',
            value: etf_meta?.aum_display || '–',
          },
          {
            label: 'Expense Ratio',
            value: etf_meta?.expense_ratio_pct != null
              ? `${etf_meta.expense_ratio_pct}%` : '–',
          },
          {
            label: 'YTD Return',
            value: performance?.ytd_return != null
              ? pct(performance.ytd_return) : '–',
            positive: (performance?.ytd_return ?? 0) >= 0,
          },
          {
            label: 'Max Drawdown',
            value: performance?.max_drawdown != null
              ? `${performance.max_drawdown}%` : '–',
            positive: false,
          },
        ].map(({ label, value, positive }) => (
          <div key={label} className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06] text-center">
            <p className="text-[9px] text-slate-500 uppercase tracking-wide">{label}</p>
            <p className={`text-xs font-bold mt-0.5 ${
              positive === undefined ? 'text-slate-200' :
              positive ? 'text-emerald-400' : 'text-rose-400'
            }`}>
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* CAGR Comparison */}
      <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
        <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-4">
          CAGR — ETF vs Benchmark
        </p>
        <CAGRBar label="1-Year CAGR"  etf={performance?.cagr_1y} bench={performance?.bench_cagr_1y} />
        <CAGRBar label="3-Year CAGR"  etf={performance?.cagr_3y} bench={performance?.bench_cagr_3y} />
        <CAGRBar label="5-Year CAGR"  etf={performance?.cagr_5y} bench={performance?.bench_cagr_5y} />

        {/* Metric Segregation: Secondary-Market Divergence & Regulatory NAV TE */}
        {(performance?.secondary_market_divergence?.divergence_annual != null || performance?.tracking_error_annual != null) && (
          <div className="pt-2.5 border-t border-white/[0.05] mt-2.5 space-y-2">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-slate-400 font-medium">
                Secondary-Market Return Divergence (Annual)
              </span>
              <span className={`font-bold ${
                (performance?.secondary_market_divergence?.divergence_annual ?? performance.tracking_error_annual) < 0.5 ? 'text-emerald-400' :
                (performance?.secondary_market_divergence?.divergence_annual ?? performance.tracking_error_annual) < 2.5 ? 'text-sky-400' : 'text-amber-400'
              }`}>
                {performance?.secondary_market_divergence?.divergence_annual ?? performance.tracking_error_annual}%
              </span>
            </div>

            {/* Multi-Horizon Secondary-Market Divergence breakdown */}
            {(performance?.tracking_error_30d != null || performance?.tracking_error_90d != null || performance?.tracking_error_1y != null) && (
              <div className="grid grid-cols-3 gap-1 pt-1 pb-0.5 text-center bg-white/[0.02] rounded border border-white/[0.04]">
                <div>
                  <p className="text-[8px] text-slate-500 uppercase">30D Divergence</p>
                  <p className="text-[9px] font-semibold text-slate-300">
                    {performance.tracking_error_30d != null ? `${performance.tracking_error_30d}%` : '–'}
                  </p>
                </div>
                <div>
                  <p className="text-[8px] text-slate-500 uppercase">90D Divergence</p>
                  <p className="text-[9px] font-semibold text-slate-300">
                    {performance.tracking_error_90d != null ? `${performance.tracking_error_90d}%` : '–'}
                  </p>
                </div>
                <div>
                  <p className="text-[8px] text-slate-500 uppercase">1Y Divergence</p>
                  <p className="text-[9px] font-semibold text-slate-300">
                    {performance.tracking_error_1y != null ? `${performance.tracking_error_1y}%` : '–'}
                  </p>
                </div>
              </div>
            )}

            {/* Regulatory NAV Tracking Error disclosure */}
            <div className="flex items-center justify-between text-[10px] pt-1 text-slate-500">
              <span>Regulatory NAV Tracking Error (SEBI Definition)</span>
              <span className="font-semibold text-slate-400 italic">
                {performance?.regulatory_nav_tracking_error != null
                  ? `${performance.regulatory_nav_tracking_error}%`
                  : 'Not calculated (scheme NAV series unavailable)'}
              </span>
            </div>

            {/* Premium/(Discount) to iNAV */}
            {performance?.secondary_market_dislocation?.premium_discount_pct != null && (
              <div className="flex items-center justify-between text-[10px] pt-0.5">
                <span className="text-slate-500">Premium/(Discount) to iNAV</span>
                <span className={`font-bold ${
                  Math.abs(performance.secondary_market_dislocation.premium_discount_pct) < 0.5 ? 'text-emerald-400' :
                  Math.abs(performance.secondary_market_dislocation.premium_discount_pct) < 2.0 ? 'text-sky-400' : 'text-amber-400'
                }`}>
                  {performance.secondary_market_dislocation.premium_discount_pct > 0 ? '+' : ''}
                  {performance.secondary_market_dislocation.premium_discount_pct}%
                </span>
              </div>
            )}
          </div>
        )}

        {performance?.sharpe_3y != null && (
          <div className="flex items-center justify-between text-[10px] pt-1">
            <span className="text-slate-500">Sharpe Ratio (3yr)</span>
            <span className={`font-bold ${
              performance.sharpe_3y >= 1 ? 'text-emerald-400' :
              performance.sharpe_3y >= 0.5 ? 'text-sky-400' : 'text-amber-400'
            }`}>
              {fmt(performance.sharpe_3y)}
            </span>
          </div>
        )}
      </div>

      {/* Liquidity & Execution Quality */}
      <LiquidityCard data={liquidity} curr={curr} />

      {/* 3-Year Rolling Returns Distribution */}
      <RollingReturnsCard data={rolling_returns} />

      {/* SIP Score */}
      <SIPMeter score={sip_score ?? 0} label={sip_label ?? ''} />

      {/* Realized Historical SIP vs Lump Sum Backtester */}
      <HistoricalSIPCard data={historical_sip} curr={curr} />

      {/* 6-Point Health Checklist */}
      <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
        <div className="flex items-center justify-between mb-1">
          <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">
            ETF Health Checklist
          </p>
          <span className="text-[9px] font-bold text-slate-500 bg-white/[0.04] border border-white/[0.08] px-2 py-0.5 rounded-full">
            {health_score_label}
          </span>
        </div>
        <div className="divide-y divide-white/[0.04]">
          {(health_checklist || []).map((item, i) => (
            <ChecklistItem key={i} item={item} />
          ))}
        </div>
      </div>

      {/* Underlying Portfolio Breakdown (Top Holdings & Sectors) */}
      <HoldingsAndSectorsCard holdings={top_holdings} sectors={sector_exposure} />

      {/* Yearly Returns */}
      <div className="rounded-xl p-4 bg-white/[0.03] border border-white/[0.06]">
        <YearlyReturns data={yearly_returns} />
      </div>

      {/* Disclaimer */}
      <p className="text-[9px] text-slate-600 text-center leading-relaxed">
        StockIQ Pro is a personal quantitative stock and ETF analytics research platform. Calculations reference external regulatory definitions (e.g. SEBI ETF tracking error) for methodological consistency. For educational and analytical purposes only. Past performance does not guarantee future returns.
      </p>

    </div>
  );
}
