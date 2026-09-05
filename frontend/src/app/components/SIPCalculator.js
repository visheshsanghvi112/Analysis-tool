'use client';

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import {
  Calculator, TrendingUp, Target, Calendar,
  Sparkles, BarChart3, Info, ChevronDown, ChevronUp,
  Copy, Check, Flame, ShieldCheck, ArrowUpRight, Percent, Clock, Layers
} from 'lucide-react';
import InfoBadge from './InfoBadge';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (
  typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : 'https://stock-analysis-backend-seven.vercel.app'
);

// ── Chart Container Wrapper for Responsive Recharts ──────────────────────────
function ChartContainer({ height = 240, children }) {
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
  if (n == null || isNaN(n)) return 'N/A';
  const val = Math.abs(n);
  const sign = n < 0 ? '-' : '';
  if (isUS) {
    if (val >= 1e9) return `${sign}${currSym}${(val / 1e9).toFixed(2)} B`;
    if (val >= 1e6) return `${sign}${currSym}${(val / 1e6).toFixed(2)} M`;
    if (val >= 1e3) return `${sign}${currSym}${(val / 1e3).toFixed(2)} K`;
    return `${sign}${currSym}${Number(val).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
  }
  if (val >= 1e7) return `${sign}${currSym}${(val / 1e7).toFixed(2)} Cr`;
  if (val >= 1e5) return `${sign}${currSym}${(val / 1e5).toFixed(2)} L`;
  return `${sign}${currSym}${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
};

const pct = (n, d = 1) => (n == null || isNaN(n)) ? 'N/A' : `${n >= 0 ? '+' : ''}${Number(n).toFixed(d)}%`;

// ── XIRR approximation via Newton-Raphson ────────────────────────────────────
function calcXIRR(cashflows, dates) {
  if (!cashflows.length || cashflows.length !== dates.length) return null;
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

// ── Core Financial Math Engines ───────────────────────────────────────────────

// 1. Regular Monthly SIP
function computeRegularSIP({ monthly, years, annualReturn }) {
  const r = annualReturn / 100 / 12;
  const n = years * 12;
  const invested = monthly * n;

  const corpus = r > 0
    ? monthly * ((Math.pow(1 + r, n) - 1) / r) * (1 + r)
    : invested;

  const wealth = corpus - invested;
  const returns_pct = invested > 0 ? (wealth / invested) * 100 : 0;

  const series = [];
  for (let yr = 1; yr <= years; yr++) {
    const months = yr * 12;
    const inv = monthly * months;
    const val = r > 0
      ? monthly * ((Math.pow(1 + r, months) - 1) / r) * (1 + r)
      : inv;
    series.push({
      year: `Y${yr}`,
      yearNum: yr,
      annualDeposit: monthly * 12,
      invested: Math.round(inv),
      value: Math.round(val),
      yearlyGains: Math.round(val - inv),
    });
  }

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
  } catch (_) { xirrRate = null; }

  return { corpus, invested, wealth, returns_pct, series, xirrRate, finalMonthly: monthly };
}

// 2. Step-Up (Top-Up) SIP
function computeStepUpSIP({ initialMonthly, annualStepUpPct, years, annualReturn }) {
  const r = annualReturn / 100 / 12;
  const n = years * 12;
  let balance = 0;
  let totalInvested = 0;
  const series = [];

  for (let yr = 1; yr <= years; yr++) {
    const stepFactor = Math.pow(1 + annualStepUpPct / 100, yr - 1);
    const monthlyForYear = initialMonthly * stepFactor;
    let yearDeposit = 0;

    for (let m = 0; m < 12; m++) {
      totalInvested += monthlyForYear;
      yearDeposit += monthlyForYear;
      balance = (balance + monthlyForYear) * (1 + r);
    }

    series.push({
      year: `Y${yr}`,
      yearNum: yr,
      monthlyInstallment: Math.round(monthlyForYear),
      annualDeposit: Math.round(yearDeposit),
      invested: Math.round(totalInvested),
      value: Math.round(balance),
      yearlyGains: Math.round(balance - totalInvested),
    });
  }

  const wealth = balance - totalInvested;
  const returns_pct = totalInvested > 0 ? (wealth / totalInvested) * 100 : 0;
  const finalMonthly = initialMonthly * Math.pow(1 + annualStepUpPct / 100, years - 1);

  return { corpus: balance, invested: totalInvested, wealth, returns_pct, series, xirrRate: annualReturn, finalMonthly };
}

// 3. Lump Sum Investment
function computeLumpSum({ principal, years, annualReturn }) {
  const r = annualReturn / 100;
  const corpus = principal * Math.pow(1 + r, years);
  const invested = principal;
  const wealth = corpus - invested;
  const returns_pct = invested > 0 ? (wealth / invested) * 100 : 0;

  const series = [];
  for (let yr = 1; yr <= years; yr++) {
    const val = principal * Math.pow(1 + r, yr);
    series.push({
      year: `Y${yr}`,
      yearNum: yr,
      annualDeposit: yr === 1 ? principal : 0,
      invested: Math.round(principal),
      value: Math.round(val),
      yearlyGains: Math.round(val - principal),
    });
  }

  return { corpus, invested, wealth, returns_pct, series, xirrRate: annualReturn, finalMonthly: 0 };
}

// 4. Goal-Based Reverse SIP
function computeGoalSIP({ targetCorpus, years, annualReturn }) {
  const r = annualReturn / 100 / 12;
  const n = years * 12;
  const monthly = r > 0
    ? (targetCorpus * r) / ((Math.pow(1 + r, n) - 1) * (1 + r))
    : targetCorpus / n;

  // Cost of 5-year delay
  let delayedMonthly = null;
  if (years > 5) {
    const delayedN = (years - 5) * 12;
    delayedMonthly = r > 0
      ? (targetCorpus * r) / ((Math.pow(1 + r, delayedN) - 1) * (1 + r))
      : targetCorpus / delayedN;
  }

  const regular = computeRegularSIP({ monthly, years, annualReturn });
  return {
    ...regular,
    requiredMonthly: monthly,
    delayedMonthly,
    targetCorpus,
  };
}

// ── Tooltip Component ────────────────────────────────────────────────────────
function SIPTooltip({ active, payload, label, currSym = '₹', isUS = false }) {
  if (!active || !payload?.length) return null;
  const invested = payload.find(p => p.dataKey === 'invested')?.value;
  const value    = payload.find(p => p.dataKey === 'value')?.value;
  const gain     = value - invested;
  return (
    <div className="bg-slate-950/95 backdrop-blur-md border border-slate-700/80 rounded-xl px-3 py-2.5 text-xs shadow-2xl min-w-[170px]">
      <p className="text-slate-400 font-bold mb-1.5 border-b border-slate-800 pb-1 font-mono">{label}</p>
      <p className="text-slate-300 flex justify-between gap-2">
        <span>Invested:</span>
        <span className="font-bold font-mono text-white">{fmtVal(invested, currSym, isUS)}</span>
      </p>
      <p className="text-emerald-400 flex justify-between gap-2">
        <span>Portfolio Value:</span>
        <span className="font-bold font-mono">{fmtVal(value, currSym, isUS)}</span>
      </p>
      <p className="text-indigo-400 flex justify-between gap-2 pt-1 border-t border-slate-800/80">
        <span>Total Gains:</span>
        <span className="font-bold font-mono">+{fmtVal(gain, currSym, isUS)}</span>
      </p>
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

  // Active Mode: 'sip' | 'stepup' | 'lumpsum' | 'goal'
  const [calcMode, setCalcMode] = useState('sip');

  // Input states
  const [monthly, setMonthly] = useState(isUS ? 500 : 10000);
  const [stepUpPct, setStepUpPct] = useState(10);
  const [lumpSumAmt, setLumpSumAmt] = useState(isUS ? 10000 : 250000);
  const [goalTarget, setGoalTarget] = useState(isUS ? 1000000 : 10000000); // 1 Cr or $1M
  const [years, setYears] = useState(10);
  const [annualReturn, setReturn] = useState(12);

  // Advanced toggles
  const [showCompare, setShowCompare] = useState(false);
  const [showSchedule, setShowSchedule] = useState(false);
  const [adjustInflation, setAdjustInflation] = useState(false);
  const [inflationRate, setInflationRate] = useState(isUS ? 3 : 6);
  const [copiedSchedule, setCopiedSchedule] = useState(false);

  // Active Ticker CAGR state
  const [tickerCAGR, setTickerCAGR] = useState(null);

  // Fetch Ticker Historical CAGR
  useEffect(() => {
    if (!ticker) return;
    let isCancelled = false;
    const fetchCAGR = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/fundamentals?ticker=${encodeURIComponent(ticker)}`);
        if (res.ok && !isCancelled) {
          const json = await res.json();
          if (json?.price_cagr) {
            setTickerCAGR(json.price_cagr);
          }
        }
      } catch (_) {}
    };
    fetchCAGR();
    return () => { isCancelled = true; };
  }, [ticker]);

  // Main Computation based on mode
  const result = useMemo(() => {
    switch (calcMode) {
      case 'stepup':
        return computeStepUpSIP({
          initialMonthly: monthly,
          annualStepUpPct: stepUpPct,
          years,
          annualReturn,
        });
      case 'lumpsum':
        return computeLumpSum({
          principal: lumpSumAmt,
          years,
          annualReturn,
        });
      case 'goal':
        return computeGoalSIP({
          targetCorpus: goalTarget,
          years,
          annualReturn,
        });
      case 'sip':
      default:
        return computeRegularSIP({
          monthly,
          years,
          annualReturn,
        });
    }
  }, [calcMode, monthly, stepUpPct, lumpSumAmt, goalTarget, years, annualReturn]);

  // Benchmark comparisons
  const fdResult = useMemo(() => {
    if (calcMode === 'lumpsum') {
      return computeLumpSum({ principal: lumpSumAmt, years, annualReturn: fdReturn });
    }
    return computeRegularSIP({ monthly: calcMode === 'goal' ? (result.requiredMonthly || monthly) : monthly, years, annualReturn: fdReturn });
  }, [calcMode, lumpSumAmt, monthly, result.requiredMonthly, years, fdReturn]);

  const benchResult = useMemo(() => {
    if (calcMode === 'lumpsum') {
      return computeLumpSum({ principal: lumpSumAmt, years, annualReturn: benchReturn });
    }
    return computeRegularSIP({ monthly: calcMode === 'goal' ? (result.requiredMonthly || monthly) : monthly, years, annualReturn: benchReturn });
  }, [calcMode, lumpSumAmt, monthly, result.requiredMonthly, years, benchReturn]);

  // Inflation adjusted values
  const inflationFactor = Math.pow(1 + inflationRate / 100, years);
  const realCorpus = Math.round(result.corpus / inflationFactor);
  const realWealth = Math.round(realCorpus - result.invested);

  // FIRE 4% Rule Monthly Passive Income
  const monthlyPassiveIncome = Math.round((result.corpus * 0.04) / 12);

  // Estimated Post-Tax Corpus (12.5% on LTCG > 1.25L for India; 15% flat for US)
  const estTax = useMemo(() => {
    if (result.wealth <= 0) return 0;
    if (isUS) {
      return Math.round(result.wealth * 0.15);
    }
    const taxableGains = Math.max(0, result.wealth - 125000);
    return Math.round(taxableGains * 0.125);
  }, [result.wealth, isUS]);
  const postTaxCorpus = Math.round(result.corpus - estTax);

  // Copy Schedule to Clipboard
  const handleCopySchedule = useCallback(() => {
    if (!result?.series?.length) return;
    const header = ['Year', 'Annual Deposit', 'Cumulative Invested', 'Yearly Gains', 'Portfolio Balance'].join('\t');
    const rows = result.series.map(s => [
      s.year,
      fmtVal(s.annualDeposit, currSym, isUS),
      fmtVal(s.invested, currSym, isUS),
      fmtVal(s.yearlyGains, currSym, isUS),
      fmtVal(s.value, currSym, isUS)
    ].join('\t')).join('\n');
    const content = `${header}\n${rows}`;
    navigator.clipboard.writeText(content);
    setCopiedSchedule(true);
    setTimeout(() => setCopiedSchedule(false), 2000);
  }, [result.series, currSym, isUS]);

  // Preset arrays
  const MONTHLY_PRESETS = isUS ? [100, 250, 500, 1000, 2500] : [2500, 5000, 10000, 25000, 50000];
  const LUMPSUM_PRESETS = isUS ? [5000, 10000, 25000, 50000, 100000] : [50000, 100000, 250000, 500000, 1000000];
  const GOAL_PRESETS = isUS
    ? [{ label: '$250K', val: 250000 }, { label: '$500K', val: 500000 }, { label: '$1M', val: 1000000 }, { label: '$2.5M', val: 2500000 }]
    : [{ label: '₹25 Lakh', val: 2500000 }, { label: '₹50 Lakh', val: 5000000 }, { label: '₹1 Crore', val: 10000000 }, { label: '₹5 Crore', val: 50000000 }];
  const YEAR_PRESETS = [5, 10, 15, 20, 25, 30];
  const STEPUP_PRESETS = [5, 10, 15, 20];
  const RETURN_PRESETS = [
    { label: 'Conservative', r: isUS ? 6 : 8 },
    { label: 'Moderate',     r: isUS ? 10 : 12 },
    { label: 'Aggressive',   r: isUS ? 12 : 15 },
    { label: 'Market Leader',r: isUS ? 15 : 18 },
  ];

  return (
    <div className="glass-card p-5 sm:p-6 space-y-6">

      {/* ── Header & Mode Switcher ────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Calculator className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-base font-bold text-white tracking-tight">SIP & Compounding Wealth Desk</h3>
              <InfoBadge infoKey={calcMode === 'stepup' ? 'step_up_sip' : calcMode === 'goal' ? 'target_goal_planner' : 'sip_calculator'} />
            </div>
            <p className="text-xs text-slate-400">Institutional geometric compounding, Newton-Raphson XIRR & FIRE projections</p>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-semibold">
          {[
            { id: 'sip', label: 'Regular SIP' },
            { id: 'stepup', label: 'Step-Up SIP' },
            { id: 'lumpsum', label: 'Lump Sum' },
            { id: 'goal', label: 'Goal Planner' },
          ].map((m) => (
            <button
              key={m.id}
              onClick={() => setCalcMode(m.id)}
              className={`px-3 py-1.5 rounded-lg transition cursor-pointer ${
                calcMode === m.id
                  ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Inputs Grid ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">

        {/* Column 1: Contribution / Target Input */}
        {calcMode === 'goal' ? (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Target Wealth Goal
              </label>
              <span className="text-xs font-mono font-bold text-emerald-400">{fmtVal(goalTarget, currSym, isUS)}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 border border-slate-800 rounded-xl focus-within:border-emerald-500/50 transition">
              <span className="text-slate-400 font-bold text-sm">{currSym}</span>
              <input
                type="number"
                min={1000}
                value={goalTarget}
                onChange={e => setGoalTarget(Math.max(1000, parseInt(e.target.value) || 0))}
                className="bg-transparent text-white text-sm flex-1 outline-none font-bold font-mono"
              />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {GOAL_PRESETS.map(p => (
                <button
                  key={p.label}
                  onClick={() => setGoalTarget(p.val)}
                  className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer font-mono ${
                    goalTarget === p.val
                      ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300 font-bold'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
        ) : calcMode === 'lumpsum' ? (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                Initial Lump Sum Investment
              </label>
              <span className="text-xs font-mono font-bold text-emerald-400">{fmtVal(lumpSumAmt, currSym, isUS)}</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 border border-slate-800 rounded-xl focus-within:border-emerald-500/50 transition">
              <span className="text-slate-400 font-bold text-sm">{currSym}</span>
              <input
                type="number"
                min={100}
                value={lumpSumAmt}
                onChange={e => setLumpSumAmt(Math.max(100, parseInt(e.target.value) || 0))}
                className="bg-transparent text-white text-sm flex-1 outline-none font-bold font-mono"
              />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {LUMPSUM_PRESETS.map(p => (
                <button
                  key={p}
                  onClick={() => setLumpSumAmt(p)}
                  className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer font-mono ${
                    lumpSumAmt === p
                      ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300 font-bold'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {fmtVal(p, currSym, isUS)}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                {calcMode === 'stepup' ? 'Starting Monthly SIP' : 'Monthly Investment'}
              </label>
              <span className="text-xs font-mono font-bold text-emerald-400">{fmtVal(monthly, currSym, isUS)}/mo</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/80 border border-slate-800 rounded-xl focus-within:border-emerald-500/50 transition">
              <span className="text-slate-400 font-bold text-sm">{currSym}</span>
              <input
                type="number"
                min={100}
                value={monthly}
                onChange={e => setMonthly(Math.max(100, parseInt(e.target.value) || 0))}
                className="bg-transparent text-white text-sm flex-1 outline-none font-bold font-mono"
              />
            </div>
            <div className="flex flex-wrap gap-1.5">
              {MONTHLY_PRESETS.map(p => (
                <button
                  key={p}
                  onClick={() => setMonthly(p)}
                  className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer font-mono ${
                    monthly === p
                      ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300 font-bold'
                      : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {fmtVal(p, currSym, isUS)}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Column 2: Duration / Step-Up Control */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Time Horizon
            </label>
            <span className="text-base font-black text-white font-mono">{years} <span className="text-xs font-normal text-slate-400">years</span></span>
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
              <button
                key={y}
                onClick={() => setYears(y)}
                className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer font-mono ${
                  years === y
                    ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300 font-bold'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {y}Y
              </button>
            ))}
          </div>

          {/* Step-Up Sub-Control when in Step-Up mode */}
          {calcMode === 'stepup' && (
            <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-1.5">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Annual Step-Up Increment</span>
                <span className="text-xs font-mono font-bold text-cyan-400">+{stepUpPct}% / yr</span>
              </div>
              <input
                type="range"
                min={1}
                max={30}
                value={stepUpPct}
                onChange={e => setStepUpPct(Math.max(1, Math.min(30, parseInt(e.target.value) || 1)))}
                className="w-full h-1.5 accent-cyan-500 bg-slate-800 rounded-lg cursor-pointer"
              />
              <div className="flex flex-wrap gap-1">
                {STEPUP_PRESETS.map(s => (
                  <button
                    key={s}
                    onClick={() => setStepUpPct(s)}
                    className={`text-[9px] px-1.5 py-0.5 rounded border transition cursor-pointer font-mono ${
                      stepUpPct === s ? 'bg-cyan-500/20 border-cyan-500/30 text-cyan-300' : 'bg-slate-900/60 border-slate-800 text-slate-400'
                    }`}
                  >
                    +{s}%
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-400 font-mono mt-1">
                Ends at: <strong className="text-cyan-300">{fmtVal(result.finalMonthly, currSym, isUS)}/mo</strong> in Y{years}
              </p>
            </div>
          )}
        </div>

        {/* Column 3: Expected Annual Return */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
              Expected Annual Return
            </label>
            <span className="text-base font-black text-emerald-400 font-mono">{annualReturn}%</span>
          </div>
          <input
            type="range"
            min={4}
            max={35}
            step={0.5}
            value={annualReturn}
            onChange={e => setReturn(parseFloat(e.target.value))}
            className="w-full h-1.5 accent-emerald-500 bg-slate-800 rounded-lg cursor-pointer"
          />
          <div className="flex flex-wrap gap-1.5">
            {RETURN_PRESETS.map(({ label, r }) => (
              <button
                key={r}
                onClick={() => setReturn(r)}
                className={`text-[10px] px-2 py-0.5 rounded-lg border transition cursor-pointer font-mono ${
                  annualReturn === r
                    ? 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300 font-bold'
                    : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {r}% {label}
              </button>
            ))}
          </div>

          {/* Active Ticker CAGR Quick Preset Button */}
          {tickerCAGR && (tickerCAGR['3y'] > 0 || tickerCAGR['5y'] > 0) && (
            <div className="mt-2 pt-2 border-t border-slate-800/80 flex flex-wrap gap-1.5 items-center">
              <span className="text-[9px] text-slate-400 uppercase font-bold tracking-wider">Active Ticker:</span>
              {tickerCAGR['3y'] > 0 && (
                <button
                  onClick={() => setReturn(Math.min(35, Math.max(4, Math.round(tickerCAGR['3y']))))}
                  className="text-[10px] px-2 py-0.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 flex items-center gap-1 font-mono transition"
                >
                  <Sparkles className="w-3 h-3" />
                  {ticker.split('.')[0]} 3Y ({Math.round(tickerCAGR['3y'])}%)
                </button>
              )}
              {tickerCAGR['5y'] > 0 && (
                <button
                  onClick={() => setReturn(Math.min(35, Math.max(4, Math.round(tickerCAGR['5y']))))}
                  className="text-[10px] px-2 py-0.5 rounded-lg bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 flex items-center gap-1 font-mono transition"
                >
                  <Sparkles className="w-3 h-3" />
                  {ticker.split('.')[0]} 5Y ({Math.round(tickerCAGR['5y'])}%)
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Goal Planner Highlight Banner (when active) ───────────────────── */}
      {calcMode === 'goal' && (
        <div className="p-4 rounded-xl bg-gradient-to-r from-emerald-950/40 via-slate-900 to-indigo-950/40 border border-emerald-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-emerald-400" />
              <span className="text-xs font-bold text-white uppercase tracking-wider">Required Monthly SIP to reach {fmtVal(goalTarget, currSym, isUS)}</span>
            </div>
            <p className="text-2xl font-black font-mono text-emerald-400 mt-1">
              {fmtVal(result.requiredMonthly, currSym, isUS)} <span className="text-xs font-normal text-slate-400 font-sans">per month for {years} years</span>
            </p>
          </div>

          {result.delayedMonthly && (
            <div className="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-right">
              <span className="text-[10px] font-bold text-rose-300 uppercase tracking-wider block">⚠️ Cost of 5-Year Delay</span>
              <span className="text-sm font-black font-mono text-rose-400">
                {fmtVal(result.delayedMonthly, currSym, isUS)}/mo (+{Math.round(((result.delayedMonthly - result.requiredMonthly) / result.requiredMonthly) * 100)}%)
              </span>
              <span className="text-[9px] text-slate-400 block mt-0.5">Starting 5 years later requires 2× more monthly capital!</span>
            </div>
          )}
        </div>
      )}

      {/* ── Result Metrics Cards ──────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {[
          {
            label: calcMode === 'lumpsum' ? 'Initial Capital' : 'Total Deposited',
            val: fmtVal(result.invested, currSym, isUS),
            sub: `${years} Years`,
            color: 'text-slate-200',
            icon: '💰'
          },
          {
            label: adjustInflation ? 'Real Purchasing Power' : 'Estimated Corpus',
            val: fmtVal(adjustInflation ? realCorpus : result.corpus, currSym, isUS),
            sub: adjustInflation ? `Discounted @ ${inflationRate}% inflation` : `Nominal value @ ${annualReturn}%`,
            color: 'text-emerald-400',
            icon: '🏆'
          },
          {
            label: adjustInflation ? 'Real Wealth Gained' : 'Wealth Gained',
            val: fmtVal(adjustInflation ? realWealth : result.wealth, currSym, isUS),
            sub: `${result.returns_pct.toFixed(0)}% ROI`,
            color: 'text-indigo-400',
            icon: '📈'
          },
          {
            label: 'FIRE Monthly Passive Income',
            val: `${fmtVal(monthlyPassiveIncome, currSym, isUS)}/mo`,
            sub: '4% Safe Withdrawal Rate (SWR)',
            color: 'text-cyan-400',
            icon: '🏖️'
          },
        ].map(({ label, val, sub, color, icon }) => (
          <div key={label} className="p-3.5 rounded-xl border bg-slate-900/60 border-slate-800 text-center space-y-1">
            <p className="text-lg">{icon}</p>
            <p className={`text-base sm:text-lg font-black font-mono ${color}`}>{val}</p>
            <p className="text-[10px] font-bold text-slate-300">{label}</p>
            <p className="text-[9px] text-slate-500 font-mono">{sub}</p>
          </div>
        ))}
      </div>

      {/* ── Advanced Insight & Tax Card ────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-3.5 bg-slate-950/70 border border-slate-800 rounded-xl text-xs">
        <div className="flex items-start gap-2.5">
          <Info className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
          <div>
            <div className="text-slate-200 font-medium">
              <strong className="text-white font-bold">Rule of 72:</strong> Your money doubles every{' '}
              <span className="text-indigo-400 font-mono font-bold">{(72 / annualReturn).toFixed(1)} years</span>.
              {' '}Over {years} years, {currSym}1 compounding at {annualReturn}% multiplies to{' '}
              <span className="text-emerald-400 font-mono font-bold">{currSym}{Math.pow(1 + annualReturn / 100, years).toFixed(1)}</span>.
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">
              Est. Post-Tax Corpus: <span className="font-mono text-slate-200 font-bold">{fmtVal(postTaxCorpus, currSym, isUS)}</span> (after ~{fmtVal(estTax, currSym, isUS)} estimated long-term capital gains tax).
            </div>
          </div>
        </div>

        {/* Inflation Toggle */}
        <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
          <button
            onClick={() => setAdjustInflation(v => !v)}
            className={`px-2.5 py-1.5 rounded-lg border text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer ${
              adjustInflation
                ? 'bg-amber-500/20 border-amber-500/30 text-amber-300 font-bold'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Flame className="w-3.5 h-3.5" />
            Adjust for {inflationRate}% Inflation
          </button>
          <InfoBadge infoKey="inflation_adjusted_corpus" />
        </div>
      </div>

      {/* ── Corpus Growth AreaChart ───────────────────────────────────────── */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">
            Compounding Growth Trajectory — Principal Invested vs. Portfolio Value
          </p>
          <span className="text-[10px] text-slate-400 font-mono">
            Terminal Value: <strong className="text-emerald-400">{fmtVal(result.corpus, currSym, isUS)}</strong>
          </span>
        </div>

        <div className="bg-slate-950/60 rounded-xl p-3 border border-slate-800/80">
          <ChartContainer height={210}>
            {(width, height) => (
              <AreaChart width={width} height={height} data={result.series} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="sipGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#10b981" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="invGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#6366f1" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.3} />
                <XAxis dataKey="year" tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                <YAxis
                  tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => fmtVal(v, currSym, isUS).replace(currSym, '')}
                  width={50}
                  tickCount={5}
                />
                <Tooltip content={<SIPTooltip currSym={currSym} isUS={isUS} />} />
                <Area
                  type="monotone"
                  dataKey="invested"
                  name="invested"
                  stroke="#6366f1"
                  strokeWidth={1.8}
                  fill="url(#invGrad)"
                  dot={false}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  name="value"
                  stroke="#10b981"
                  strokeWidth={2.2}
                  fill="url(#sipGrad)"
                  dot={false}
                />
              </AreaChart>
            )}
          </ChartContainer>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 mt-2 text-[10px] text-slate-400">
          <div className="flex gap-4">
            <span className="flex items-center gap-1.5"><span className="w-3 h-1 bg-indigo-500 rounded inline-block" /> Cumulative Invested</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-1 bg-emerald-500 rounded inline-block" /> Portfolio Value</span>
          </div>
          <span className="text-slate-400 italic">*Geometric compounding model · past returns do not guarantee future gains</span>
        </div>
      </div>

      {/* ── Comparison with Benchmark & Fixed Deposit ─────────────────────── */}
      <div className="space-y-3">
        <button
          onClick={() => setShowCompare(v => !v)}
          className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 transition cursor-pointer font-semibold"
        >
          {showCompare ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          Compare Scenario vs. Fixed Deposit ({fdReturn}%) and {benchName} historical avg ({benchReturn}%)
        </button>

        {showCompare && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              {
                label: `This Scenario (${annualReturn}%)`,
                corpus: result.corpus,
                wealth: result.wealth,
                color: 'text-emerald-400',
                border: 'border-emerald-500/30',
                icon: '⭐'
              },
              {
                label: `${benchName} Index (${benchReturn}% p.a.)`,
                corpus: benchResult.corpus,
                wealth: benchResult.wealth,
                color: 'text-indigo-400',
                border: 'border-slate-800',
                icon: '📊'
              },
              {
                label: `Fixed Deposit (${fdReturn}% p.a.)`,
                corpus: fdResult.corpus,
                wealth: fdResult.wealth,
                color: 'text-amber-400',
                border: 'border-slate-800',
                icon: '🏦'
              },
            ].map(({ label, corpus, wealth, color, border, icon }) => (
              <div key={label} className={`p-3.5 rounded-xl bg-slate-950/60 border ${border} text-center space-y-1`}>
                <p className="text-lg">{icon}</p>
                <p className={`text-base font-black font-mono ${color}`}>{fmtVal(corpus, currSym, isUS)}</p>
                <p className="text-[10px] font-bold text-slate-300">{label}</p>
                <p className="text-[9px] text-slate-500 font-mono">Gains: +{fmtVal(wealth, currSym, isUS)}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Year-by-Year Schedule (Expandable Table) ───────────────────────── */}
      <div className="pt-2 border-t border-slate-800">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowSchedule(v => !v)}
            className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white transition cursor-pointer font-semibold"
          >
            <Layers className="h-3.5 w-3.5 text-emerald-400" />
            {showSchedule ? 'Hide Yearly Compounding Schedule' : 'View Year-by-Year Amortization Schedule'}
            {showSchedule ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>

          {showSchedule && (
            <button
              onClick={handleCopySchedule}
              className="flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition cursor-pointer font-mono"
            >
              {copiedSchedule ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copiedSchedule ? 'Copied to Clipboard' : 'Copy Schedule'}
            </button>
          )}
        </div>

        {showSchedule && (
          <div className="mt-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/80">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800 uppercase text-[10px]">
                <tr>
                  <th className="py-2.5 px-3">Timeline</th>
                  <th className="py-2.5 px-3 text-right">Annual Deposit</th>
                  <th className="py-2.5 px-3 text-right">Cumulative Invested</th>
                  <th className="py-2.5 px-3 text-right">Interest / Gains</th>
                  <th className="py-2.5 px-3 text-right text-emerald-400">Closing Balance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {result.series.map((row) => (
                  <tr key={row.year} className="hover:bg-slate-900/40 transition">
                    <td className="py-2 px-3 font-bold text-slate-200">{row.year} (Year {row.yearNum})</td>
                    <td className="py-2 px-3 text-right text-slate-400">{fmtVal(row.annualDeposit, currSym, isUS)}</td>
                    <td className="py-2 px-3 text-right text-slate-300">{fmtVal(row.invested, currSym, isUS)}</td>
                    <td className="py-2 px-3 text-right text-indigo-400">+{fmtVal(row.yearlyGains, currSym, isUS)}</td>
                    <td className="py-2 px-3 text-right font-bold text-emerald-400">{fmtVal(row.value, currSym, isUS)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
}
