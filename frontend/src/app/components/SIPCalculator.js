'use client';

import { useState, useMemo, useEffect, useRef } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ReferenceLine, Legend,
} from 'recharts';
import {
  Calculator, TrendingUp, IndianRupee, Target, Calendar,
  Sparkles, BarChart3, Info, ChevronDown, ChevronUp,
} from 'lucide-react';
import InfoBadge from './InfoBadge';

// ── Chart Container Wrapper for Responsive Recharts ──────────────────────────
function ChartContainer({ height = 280, children }) {
  const ref = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 0, height });

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 10) {
          setDimensions({
            width: entry.contentRect.width,
            height: entry.contentRect.height || height
          });
        }
      }
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [height]);

  return (
    <div ref={ref} className="w-full chart-container" style={{ height, minHeight: height }}>
      {dimensions.width > 0 ? children(dimensions.width, dimensions.height) : null}
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmtVal = (n, currSym = '₹', isUS = false) => {
  if (n == null) return 'N/A';
  if (isUS) {
    if (n >= 1e9) return `${currSym}${(n / 1e9).toFixed(2)} B`;
    if (n >= 1e6) return `${currSym}${(n / 1e6).toFixed(2)} M`;
    if (n >= 1e3) return `${currSym}${(n / 1e3).toFixed(2)} K`;
    return `${currSym}${Number(n).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  }
  if (n >= 1e7) return `${currSym}${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `${currSym}${(n / 1e5).toFixed(2)} L`;
  return `${currSym}${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
};
const pct = (n, d = 1) => n == null ? 'N/A' : `${n >= 0 ? '+' : ''}${Number(n).toFixed(d)}%`;

// ── XIRR approximation via Newton-Raphson (for SIP) ──────────────────────────
function calcXIRR(cashflows, dates) {
  // cashflows: array of numbers (negative = outflow, positive = inflow)
  // dates: array of Date objects aligned with cashflows
  const DAYS_IN_YEAR = 365.0;
  const refDate = dates[0];
  const t = dates.map(d => (d - refDate) / (1000 * 60 * 60 * 24) / DAYS_IN_YEAR);

  let rate = 0.1;
  for (let iter = 0; iter < 100; iter++) {
    let f = 0, df = 0;
    for (let i = 0; i < cashflows.length; i++) {
      const factor = Math.pow(1 + rate, t[i]);
      f  += cashflows[i] / factor;
      df -= t[i] * cashflows[i] / (factor * (1 + rate));
    }
    const delta = f / df;
    rate -= delta;
    if (Math.abs(delta) < 1e-7) break;
  }
  return isFinite(rate) ? rate * 100 : null;
}

// ── SIP Math ──────────────────────────────────────────────────────────────────
function computeSIP({ monthly, years, annualReturn }) {
  const r = annualReturn / 100 / 12;       // monthly rate
  const n = years * 12;                    // total months
  const invested = monthly * n;

  const corpus = r > 0
    ? monthly * (Math.pow(1 + r, n) - 1) / r * (1 + r)
    : invested;

  const wealth = corpus - invested;
  const returns_pct = invested > 0 ? (wealth / invested) * 100 : 0;

  // Build yearly series for chart
  const series = [];
  for (let yr = 1; yr <= years; yr++) {
    const months = yr * 12;
    const inv = monthly * months;
    const val = r > 0
      ? monthly * (Math.pow(1 + r, months) - 1) / r * (1 + r)
      : inv;
    series.push({ year: `Y${yr}`, invested: Math.round(inv), value: Math.round(val) });
  }

  // XIRR — all outflows are SIP instalments, final inflow is corpus
  let xirrRate = null;
  try {
    const flows = [];
    const dts = [];
    const start = new Date();
    for (let m = 0; m < n; m++) {
      flows.push(-monthly);
      const d = new Date(start);
      d.setMonth(d.getMonth() + m);
      dts.push(d);
    }
    flows.push(corpus);
    const endDate = new Date(start);
    endDate.setMonth(endDate.getMonth() + n);
    dts.push(endDate);
    xirrRate = calcXIRR(flows, dts);
  } catch { xirrRate = null; }

  return { corpus, invested, wealth, returns_pct, series, xirrRate };
}

// ── Quick preset buttons ──────────────────────────────────────────────────────
const MONTHLY_PRESETS = [1000, 5000, 10000, 25000, 50000];
const YEAR_PRESETS    = [5, 10, 15, 20, 30];
const RETURN_PRESETS  = [
  { label: 'Conservative', r: 8, color: 'text-amber-400' },
  { label: 'Moderate',     r: 12, color: 'text-indigo-400' },
  { label: 'Aggressive',   r: 15, color: 'text-emerald-400' },
  { label: 'Market Leader',r: 18, color: 'text-violet-400' },
];

// ── Tooltip ───────────────────────────────────────────────────────────────────
function SIPTooltip({ active, payload, label, currSym = '₹', isUS = false }) {
  if (!active || !payload?.length) return null;
  const invested = payload.find(p => p.dataKey === 'invested')?.value;
  const value    = payload.find(p => p.dataKey === 'value')?.value;
  const gain     = value - invested;
  return (
    <div className="bg-[#0d0d14] border border-white/10 rounded-xl px-3 py-2.5 text-xs shadow-xl min-w-[160px]">
      <p className="text-slate-400 font-bold mb-1.5 border-b border-white/[0.06] pb-1">{label}</p>
      <p className="text-slate-300">Invested: <span className="font-bold text-white">{fmtVal(invested, currSym, isUS)}</span></p>
      <p className="text-emerald-400">Value: <span className="font-bold">{fmtVal(value, currSym, isUS)}</span></p>
      <p className="text-violet-400">Gains: <span className="font-bold">+{fmtVal(gain, currSym, isUS)}</span></p>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function SIPCalculator({ ticker }) {
  const isUS = ticker ? (!ticker.endsWith('.NS') && !ticker.endsWith('.BO') && !ticker.startsWith('^')) : false;
  const currSym = isUS ? '$' : '₹';
  const benchName = isUS ? 'S&P 500' : 'Nifty 50';
  const benchReturn = isUS ? 10 : 12;
  const fdReturn = isUS ? 5 : 7;

  const [monthly, setMonthly]   = useState(isUS ? 500 : 10000);
  const [years, setYears]       = useState(10);
  const [annualReturn, setReturn] = useState(12);
  const [showCompare, setShowCompare] = useState(false);

  // Current ticker CAGR passed via prop (from FundamentalsAnalysis parent)
  const result  = useMemo(() => computeSIP({ monthly, years, annualReturn }), [monthly, years, annualReturn]);
  const fdResult = useMemo(() => computeSIP({ monthly, years, annualReturn: fdReturn }), [monthly, years, fdReturn]);   // FD comparison
  const niftyResult = useMemo(() => computeSIP({ monthly, years, annualReturn: benchReturn }), [monthly, years, benchReturn]); // Benchmark historical avg

  return (
    <div className="glass-card p-5 space-y-5">

      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Calculator className="h-5 w-5 text-emerald-400" />
          <h3 className="text-sm font-bold text-white">SIP Return Calculator</h3>
          <InfoBadge infoKey="sip_calculator" />
        </div>
        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full border bg-emerald-500/10 border-emerald-500/25 text-emerald-300">
          Compounding Power
        </span>
      </div>

      {/* Inputs */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">

        {/* Monthly Amount */}
        <div className="space-y-2">
          <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            Monthly Investment
          </label>
          <div className="flex items-center gap-2 px-3 py-2 bg-white/[0.03] border border-white/[0.08] rounded-xl focus-within:border-emerald-500/50 transition">
            <span className="text-slate-400 font-bold text-sm">{currSym}</span>
            <input
              type="number"
              min={10}
              value={monthly}
              onChange={e => setMonthly(Math.max(10, parseInt(e.target.value) || 0))}
              className="bg-transparent text-white text-sm flex-1 outline-none font-bold"
            />
          </div>
          <div className="flex flex-wrap gap-1.5">
            {(isUS ? [100, 250, 500, 1000, 2500] : MONTHLY_PRESETS).map(p => (
              <button key={p} onClick={() => setMonthly(p)}
                className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer ${
                  monthly === p
                    ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                    : 'bg-white/[0.03] border-white/[0.06] text-slate-400 hover:text-slate-200'
                }`}>
                {currSym}{p.toLocaleString(isUS ? 'en-US' : 'en-IN')}
              </button>
            ))}
          </div>
        </div>

        {/* Duration */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Time Horizon
            </label>
            <span className="text-lg font-black text-white">{years} <span className="text-xs font-normal text-slate-400">years</span></span>
          </div>
          <input
            type="range"
            min={1}
            max={40}
            value={years}
            onChange={e => setYears(Math.max(1, Math.min(40, parseInt(e.target.value) || 1)))}
            className="w-full h-1.5 accent-emerald-500 bg-slate-800 rounded-lg cursor-pointer"
          />
          <div className="flex flex-wrap gap-1.5">
            {YEAR_PRESETS.map(y => (
              <button key={y} onClick={() => setYears(y)}
                className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer ${
                  years === y
                    ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                    : 'bg-white/[0.03] border-white/[0.06] text-slate-400 hover:text-slate-200'
                }`}>
                {y}Y
              </button>
            ))}
          </div>
        </div>

        {/* Expected Return */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Expected Annual Return
            </label>
            <span className="text-lg font-black text-emerald-400">{annualReturn}%</span>
          </div>
          <input
            type="range"
            min={4}
            max={30}
            step={0.5}
            value={annualReturn}
            onChange={e => setReturn(parseFloat(e.target.value))}
            className="w-full h-1.5 accent-emerald-500 bg-slate-800 rounded-lg cursor-pointer"
          />
          <div className="flex flex-wrap gap-1.5">
            {RETURN_PRESETS.map(({ label, r, color }) => (
              <button key={r} onClick={() => setReturn(r)}
                className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer ${
                  annualReturn === r
                    ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
                    : 'bg-white/[0.03] border-white/[0.06] text-slate-400 hover:text-slate-200'
                }`}>
                {r}% {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Result Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Total Invested',   val: fmtVal(result.invested, currSym, isUS),    color: 'text-slate-300', icon: '💰' },
          { label: 'Estimated Corpus', val: fmtVal(result.corpus, currSym, isUS),      color: 'text-emerald-400', icon: '🏆' },
          { label: 'Wealth Gained',    val: fmtVal(result.wealth, currSym, isUS),      color: 'text-violet-400', icon: '📈' },
          { label: 'Returns on Investment', val: pct(result.returns_pct, 0), color: result.returns_pct >= 0 ? 'text-emerald-400' : 'text-rose-400', icon: '⚡' },
        ].map(({ label, val, color, icon }) => (
          <div key={label} className="p-3 rounded-xl border bg-white/[0.02] border-white/[0.06] text-center space-y-1">
            <p className="text-base">{icon}</p>
            <p className={`text-sm font-black ${color}`}>{val}</p>
            <p className="text-[9px] text-slate-400">{label}</p>
          </div>
        ))}
      </div>

      {/* The Rule of 72 insight */}
      <div className="flex items-start gap-2 p-3 bg-indigo-500/5 border border-indigo-500/15 rounded-xl text-[11px] text-slate-300 leading-relaxed">
        <Info className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-slate-200">Rule of 72: </span>
          At {annualReturn}% p.a., your money doubles every{' '}
          <strong className="text-indigo-400">{(72 / annualReturn).toFixed(1)} years</strong>.
          {' '}Over {years} years, {currSym}1 becomes <strong className="text-emerald-400">
            {currSym}{Math.pow(1 + annualReturn / 100, years).toFixed(1)}
          </strong>.
          {result.xirrRate != null && (
            <> | XIRR (SIP): <strong className="text-violet-400">{result.xirrRate.toFixed(2)}%</strong></>
          )}
        </div>
      </div>

      {/* Corpus Growth Chart */}
      <div>
        <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-2">
          Corpus Growth — Invested vs. Projected Value
        </p>
        <div className="bg-white/[0.01] rounded-xl p-3 border border-white/[0.04]">
          <ChartContainer height={180}>
            {(width, height) => (
              <AreaChart width={width} height={height} data={result.series} margin={{ top: 4, right: 4, left: -25, bottom: 0 }}>
                <defs>
                  <linearGradient id="sipGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#10b981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="invGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff07" />
                <XAxis dataKey="year" tick={{ fill: '#888', fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#888', fontSize: 9 }} axisLine={false} tickLine={false}
                  tickFormatter={v => fmtVal(v, currSym, isUS).replace(currSym, '')} width={45} tickCount={4} />
                <Tooltip content={<SIPTooltip currSym={currSym} isUS={isUS} />} />
                <Area type="monotone" dataKey="invested" name="invested" stroke="#6366f1" strokeWidth={1.5}
                  fill="url(#invGrad)" dot={false} />
                <Area type="monotone" dataKey="value" name="value" stroke="#10b981" strokeWidth={2}
                  fill="url(#sipGrad)" dot={false} />
              </AreaChart>
            )}
          </ChartContainer>
        </div>
        <div className="flex gap-4 mt-1 text-[10px] text-slate-400">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-indigo-500 rounded inline-block" /> Invested</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-500 rounded inline-block" /> Projected Value</span>
          <span className="flex-1 text-right text-slate-400">*Assumes constant {annualReturn}% p.a. compounding · not financial advice</span>
        </div>
      </div>

      {/* Comparison toggle */}
      <div>
        <button
          onClick={() => setShowCompare(v => !v)}
          className="flex items-center gap-1.5 text-[11px] text-indigo-400 hover:text-indigo-300 transition cursor-pointer"
        >
          {showCompare ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          Compare with FD ({fdReturn}%) and {benchName} avg ({benchReturn}%)
        </button>

        {showCompare && (
          <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { label: `This scenario (${annualReturn}%)`, corpus: result.corpus, color: 'text-emerald-400', icon: '⭐' },
              { label: `${benchName} avg (${benchReturn}% p.a.)`, corpus: niftyResult.corpus, color: 'text-indigo-400', icon: '📊' },
              { label: `Fixed Deposit (${fdReturn}% p.a.)`,           corpus: fdResult.corpus,    color: 'text-amber-400',  icon: '🏦' },
            ].map(({ label, corpus, color, icon }) => (
              <div key={label} className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.06] text-center space-y-1">
                <p className="text-base">{icon}</p>
                <p className={`text-sm font-black ${color}`}>{fmtVal(corpus, currSym, isUS)}</p>
                <p className="text-[9px] text-slate-500 leading-relaxed">{label}</p>
                <p className="text-[9px] text-slate-600">{years}Y · {fmtVal(monthly, currSym, isUS)}/mo</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
