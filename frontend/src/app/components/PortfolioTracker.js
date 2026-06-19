'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import {
  PieChart, Pie, Cell, Tooltip as RTooltip, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts';
import {
  Plus, Trash2, RefreshCw, TrendingUp, TrendingDown,
  AlertCircle, Briefcase, ShieldAlert, Search, X,
  Lightbulb, Zap, Target, ArrowDownRight, Sparkles, Wallet,
} from 'lucide-react';
import SmartCapitalAdvisor from './SmartCapitalAdvisor';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

const PORTFOLIO_KEY = 'stockiq_portfolio_v1';

const CHART_COLORS = [
  '#6366f1', '#10b981', '#f59e0b', '#ef4444', '#06b6d4',
  '#8b5cf6', '#f97316', '#84cc16', '#ec4899', '#14b8a6',
];

// ── helpers ──────────────────────────────────────────────────────────────────
const fmt    = (n, d = 2) => n == null ? 'N/A' : Number(n).toFixed(d);
const fmtINR = (n) => n == null ? 'N/A' : `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
const pSign  = (v) => v >= 0 ? '+' : '';
const pColor = (v) => v > 0 ? 'text-emerald-400' : v < 0 ? 'text-rose-400' : 'text-slate-400';
const pBg    = (v) => v > 0 ? 'bg-emerald-500/10 border-emerald-500/20' : v < 0 ? 'bg-rose-500/10 border-rose-500/20' : 'bg-white/[0.03] border-white/[0.06]';

// ── Ticker autocomplete search ────────────────────────────────────────────────
function TickerSearch({ value, onChange }) {
  const [query, setQuery]   = useState(value || '');
  const [results, setResults] = useState([]);
  const [open, setOpen]     = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    if (query.length < 1) { setResults([]); return; }
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`${API_BASE_URL}/api/tickers?q=${encodeURIComponent(query)}&limit=8`);
        const j = await r.json();
        setResults(j.tickers || []);
        setOpen(true);
      } catch { setResults([]); }
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  const pick = (t) => {
    setQuery(t.symbol);
    onChange(t.symbol);
    setResults([]);
    setOpen(false);
  };

  return (
    <div ref={ref} className="relative w-full">
      <div className="flex items-center gap-2 px-2.5 py-1.5 bg-white/[0.04] border border-white/[0.08] rounded-lg focus-within:border-indigo-500/50 transition">
        <Search className="h-3.5 w-3.5 text-slate-500 shrink-0" />
        <input
          value={query}
          onChange={e => { setQuery(e.target.value); onChange(e.target.value); }}
          onBlur={() => {
            // auto-append .NS if no suffix given
            if (query && !query.includes('.')) {
              const sym = query.toUpperCase() + '.NS';
              setQuery(sym);
              onChange(sym);
            }
          }}
          placeholder="Search stock…"
          className="bg-transparent text-xs text-white placeholder-slate-600 outline-none flex-1 min-w-0"
        />
        {query && (
          <button onClick={() => { setQuery(''); onChange(''); setResults([]); }} className="shrink-0">
            <X className="h-3 w-3 text-slate-600 hover:text-slate-300" />
          </button>
        )}
      </div>
      {open && results.length > 0 && (
        <div className="absolute z-[999] top-full mt-1 left-0 w-72 bg-[#111] border border-white/[0.12] rounded-xl shadow-2xl" style={{boxShadow:'0 20px 60px rgba(0,0,0,0.8)'}}>
          {results.map((t, i) => (
            <button
              key={t.symbol}
              onMouseDown={e => { e.preventDefault(); pick(t); }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 hover:bg-white/[0.07] text-left transition ${i < results.length-1 ? 'border-b border-white/[0.05]' : ''}`}
            >
              <div className="h-7 w-7 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center shrink-0">
                <span className="text-[8px] font-black text-indigo-400">{t.symbol.replace('.NS','').replace('.BO','').slice(0,3)}</span>
              </div>
              <div className="min-w-0">
                <p className="text-[11px] font-bold text-white">{t.symbol.replace('.NS','').replace('.BO','')}</p>
                <p className="text-[9px] text-slate-500 truncate">{t.name}</p>
              </div>
              {t.sector && <span className="ml-auto text-[9px] text-indigo-400 shrink-0 bg-indigo-500/10 px-1.5 py-0.5 rounded">{t.sector}</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Allocation pie ────────────────────────────────────────────────────────────
function AllocationPie({ holdings }) {
  const data = holdings.map((h, i) => ({
    name: h.ticker.replace('.NS','').replace('.BO',''),
    value: h.curr_value,
    color: CHART_COLORS[i % CHART_COLORS.length],
  }));

  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4">
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Allocation</p>
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={50} outerRadius={80}
            paddingAngle={2} dataKey="value">
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Pie>
          <RTooltip
            formatter={(v, n) => [fmtINR(v), n]}
            contentStyle={{ background: '#0d0d0d', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', fontSize: '11px' }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className="space-y-1 mt-1">
        {data.map((d, i) => (
          <div key={i} className="flex items-center gap-2 text-[10px]">
            <span className="h-2 w-2 rounded-full shrink-0" style={{ background: d.color }} />
            <span className="text-slate-300 flex-1">{d.name}</span>
            <span className="text-slate-400 font-bold">{holdings[i]?.weight_pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Correlation heatmap ───────────────────────────────────────────────────────
function CorrelationHeatmap({ pairs, tickers }) {
  if (!pairs?.length || tickers.length < 2) return null;
  const unique = [...new Set(pairs.map(p => p.a))];

  const corrOf = (a, b) => pairs.find(p => p.a === a && p.b === b)?.corr ?? 0;
  const cellColor = (v) => {
    if (v >= 0.7)  return 'rgba(239,68,68,0.5)';
    if (v >= 0.4)  return 'rgba(245,158,11,0.4)';
    if (v >= 0)    return 'rgba(99,102,241,0.3)';
    return 'rgba(16,185,129,0.3)';
  };
  const short = (t) => t.replace('.NS','').replace('.BO','');

  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4 overflow-x-auto">
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">Correlation Matrix (1Y Returns)</p>
      <table className="text-[9px] border-collapse">
        <thead>
          <tr>
            <th className="w-16" />
            {unique.map(t => (
              <th key={t} className="px-2 py-1 text-slate-500 font-bold text-center w-14">{short(t)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {unique.map(a => (
            <tr key={a}>
              <td className="pr-2 text-slate-500 font-bold text-right">{short(a)}</td>
              {unique.map(b => {
                const v = corrOf(a, b);
                return (
                  <td key={b} className="px-1 py-1 text-center font-bold rounded"
                    style={{ background: cellColor(v), color: '#fff', minWidth: '48px' }}>
                    {fmt(v, 2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-4 mt-3 text-[9px] text-slate-500">
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{background:'rgba(239,68,68,0.5)'}} /> High (≥0.7)</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{background:'rgba(245,158,11,0.4)'}} /> Medium</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{background:'rgba(99,102,241,0.3)'}} /> Low</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded" style={{background:'rgba(16,185,129,0.3)'}} /> Negative</span>
      </div>
    </div>
  );
}

// ── Price history chart ───────────────────────────────────────────────────────
function PriceHistoryChart({ history, holdings }) {
  const tickers = Object.keys(history);
  if (!tickers.length) return null;

  // Merge all dates, normalize to % return from start
  const dateSet = new Set();
  tickers.forEach(t => history[t].forEach(p => dateSet.add(p.date)));
  const dates = [...dateSet].sort();

  const priceAt = (t, date) => {
    const arr = history[t];
    const entry = arr.find(p => p.date === date);
    return entry?.price ?? null;
  };

  // First valid price per ticker (for normalization)
  const base = {};
  tickers.forEach(t => { base[t] = history[t][0]?.price || 1; });

  const chartData = dates.map(d => {
    const row = { date: d.slice(2, 7) };
    tickers.forEach((t, i) => {
      const p = priceAt(t, d);
      row[t.replace('.NS','').replace('.BO','')] = p ? +((p / base[t] - 1) * 100).toFixed(2) : undefined;
    });
    return row;
  });

  return (
    <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4">
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">1-Year Return (Normalised %)</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#555' }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 9, fill: '#555' }} tickLine={false} axisLine={false} tickFormatter={v => `${v}%`} />
          <RTooltip
            formatter={(v, n) => [`${v}%`, n]}
            contentStyle={{ background: '#0d0d0d', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', fontSize: '10px' }}
          />
          <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '8px' }} iconType="circle" iconSize={6} />
          {tickers.map((t, i) => (
            <Line key={t} type="monotone" dataKey={t.replace('.NS','').replace('.BO','')}
              stroke={CHART_COLORS[i % CHART_COLORS.length]} strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} connectNulls />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function PortfolioTracker() {
  const emptyRow = () => ({ id: Date.now(), ticker: '', qty: '', buy_price: '' });

  const [rows, setRows]           = useState([emptyRow()]);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [advice, setAdvice]       = useState(null);
  const [advLoading, setAdvLoading] = useState(false);
  const [showAdvisor, setShowAdvisor] = useState(false);

  // Persist to localStorage
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(PORTFOLIO_KEY));
      if (saved?.length) setRows(saved);
    } catch {}
  }, []);

  useEffect(() => {
    try { localStorage.setItem(PORTFOLIO_KEY, JSON.stringify(rows)); } catch {}
  }, [rows]);

  const updateRow = (id, field, val) =>
    setRows(r => r.map(row => row.id === id ? { ...row, [field]: val } : row));

  const removeRow = (id) => setRows(r => r.filter(row => row.id !== id));

  const addRow = () => setRows(r => [...r, emptyRow()]);

  const validRows = rows.filter(r => r.ticker && +r.qty > 0 && +r.buy_price > 0);

  const analyze = useCallback(async () => {
    if (!validRows.length) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/portfolio-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          holdings: validRows.map(r => ({
            ticker: r.ticker.trim().toUpperCase(),
            qty: +r.qty,
            buy_price: +r.buy_price,
          })),
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Analysis failed');
      setResult(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [validRows]);

  const getAdvice = useCallback(async () => {
    if (!validRows.length) return;
    setAdvLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/portfolio-insight`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          holdings: validRows.map(r => ({
            ticker: r.ticker.trim().toUpperCase(),
            qty: +r.qty,
            buy_price: +r.buy_price,
          })),
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Failed');
      setAdvice(json);
    } catch (e) {
      console.error(e);
    } finally {
      setAdvLoading(false);
    }
  }, [validRows]);

  const s   = result?.summary;
  const risk = result?.risk;

  return (
    <>
    <div className="min-h-screen bg-black text-white" style={{ fontFamily: 'var(--font-inter, Inter, sans-serif)' }}>
      {/* Header */}
      <div className="border-b border-white/[0.06] px-6 py-4 flex items-center gap-3">
        <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/20 flex items-center justify-center">
          <Briefcase className="h-4 w-4 text-indigo-400" />
        </div>
        <div>
          <h1 className="text-base font-bold text-white">Portfolio Tracker</h1>
          <p className="text-[11px] text-slate-500">Track holdings · Live P&L · Risk analytics</p>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 py-6 space-y-6">

        {/* Holdings input table */}
        <div className="rounded-xl bg-white/[0.02] border border-white/[0.06]">
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.05]">
            <p className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">Your Holdings</p>
            <p className="text-[10px] text-slate-600">{validRows.length} valid · max 15</p>
          </div>

          {/* Column headers */}
          <div className="grid grid-cols-[2fr_90px_120px_36px] gap-3 px-4 py-2 border-b border-white/[0.04]">
            <p className="text-[9px] text-slate-600 font-bold uppercase tracking-wider">Stock</p>
            <p className="text-[9px] text-slate-600 font-bold uppercase tracking-wider">Qty</p>
            <p className="text-[9px] text-slate-600 font-bold uppercase tracking-wider">Buy Price (₹)</p>
            <div />
          </div>

          {/* Rows */}
          <div className="divide-y divide-white/[0.03]">
            {rows.map((row) => (
              <div key={row.id} className="grid grid-cols-[2fr_90px_120px_36px] gap-3 px-4 py-3 items-center">
                <TickerSearch value={row.ticker} onChange={v => updateRow(row.id, 'ticker', v)} />
                <input
                  type="number" min="0" value={row.qty}
                  onChange={e => updateRow(row.id, 'qty', e.target.value)}
                  placeholder="10"
                  className="px-2.5 py-1.5 bg-white/[0.04] border border-white/[0.08] rounded-lg text-xs text-white placeholder-slate-600 outline-none focus:border-indigo-500/50 w-full"
                />
                <input
                  type="number" min="0" value={row.buy_price}
                  onChange={e => updateRow(row.id, 'buy_price', e.target.value)}
                  placeholder="1500"
                  className="px-2.5 py-1.5 bg-white/[0.04] border border-white/[0.08] rounded-lg text-xs text-white placeholder-slate-600 outline-none focus:border-indigo-500/50 w-full"
                />
                <button onClick={() => removeRow(row.id)}
                  className="h-7 w-7 flex items-center justify-center rounded-lg hover:bg-rose-500/10 text-slate-600 hover:text-rose-400 transition cursor-pointer">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 px-4 py-3 border-t border-white/[0.05]">
            <button onClick={addRow}
              className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-white transition cursor-pointer">
              <Plus className="h-3.5 w-3.5" /> Add stock
            </button>
            <button
              onClick={analyze}
              disabled={loading || !validRows.length}
              className="ml-auto flex items-center gap-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-bold rounded-lg transition cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
              {loading ? 'Analysing…' : 'Analyse Portfolio'}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-start gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm">
            <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Summary banner */}
            <div className={`rounded-xl border p-4 sm:p-5 ${pBg(s.total_pnl_pct)}`}>
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-1">Total Portfolio Value</p>
                  <p className="text-3xl font-black text-white">{fmtINR(s.total_value)}</p>
                  <div className="flex items-center gap-2 mt-1">
                    {s.total_pnl >= 0 ? <TrendingUp className="h-4 w-4 text-emerald-400" /> : <TrendingDown className="h-4 w-4 text-rose-400" />}
                    <span className={`text-sm font-bold ${pColor(s.total_pnl)}`}>
                      {pSign(s.total_pnl)}{fmtINR(s.total_pnl)} ({pSign(s.total_pnl_pct)}{fmt(s.total_pnl_pct)}%)
                    </span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 text-right">
                  <div>
                    <p className="text-[9px] text-slate-500 uppercase tracking-wider">Invested</p>
                    <p className="text-sm font-bold text-slate-200">{fmtINR(s.total_cost)}</p>
                  </div>
                  <div>
                    <p className="text-[9px] text-slate-500 uppercase tracking-wider">Holdings</p>
                    <p className="text-sm font-bold text-slate-200">{s.num_holdings}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Holdings table */}
            <div className="rounded-xl bg-white/[0.02] border border-white/[0.06] overflow-x-auto">
              <div className="px-4 py-3 border-b border-white/[0.05]">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Position Breakdown</p>
              </div>
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="border-b border-white/[0.04]">
                    {['Stock','Qty','Buy @','LTP','Invested','Value','P&L','Return','Wt%'].map(h => (
                      <th key={h} className="px-3 py-2 text-left text-[9px] text-slate-600 font-bold uppercase tracking-wider whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.holdings.map((h, i) => (
                    <tr key={h.ticker} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition">
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full" style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                          <span className="font-bold text-white">{h.ticker.replace('.NS','').replace('.BO','')}</span>
                        </div>
                      </td>
                      <td className="px-3 py-2.5 text-slate-300">{h.qty}</td>
                      <td className="px-3 py-2.5 text-slate-300">₹{h.buy_price.toLocaleString('en-IN')}</td>
                      <td className="px-3 py-2.5 text-white font-medium">₹{h.live_price.toLocaleString('en-IN')}</td>
                      <td className="px-3 py-2.5 text-slate-400">{fmtINR(h.cost)}</td>
                      <td className="px-3 py-2.5 text-slate-200 font-medium">{fmtINR(h.curr_value)}</td>
                      <td className={`px-3 py-2.5 font-bold ${pColor(h.pnl)}`}>
                        {pSign(h.pnl)}{fmtINR(h.pnl)}
                      </td>
                      <td className={`px-3 py-2.5 font-bold ${pColor(h.pnl_pct)}`}>
                        {pSign(h.pnl_pct)}{fmt(h.pnl_pct)}%
                      </td>
                      <td className="px-3 py-2.5 text-slate-400">{h.weight_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Charts row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <AllocationPie holdings={result.holdings} />
              {risk && !risk.error && (
                <div className="rounded-xl bg-white/[0.03] border border-white/[0.06] p-4 space-y-3">
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <ShieldAlert className="h-3.5 w-3.5 text-amber-400" /> Portfolio Risk
                  </p>
                  {[
                    { label: 'Annual Volatility', value: risk.ann_volatility_pct != null ? `${fmt(risk.ann_volatility_pct)}%` : 'N/A' },
                    { label: 'Sharpe Ratio', value: fmt(risk.sharpe_ratio, 3), accent: risk.sharpe_ratio >= 1 ? 'text-emerald-400' : 'text-amber-400' },
                    { label: 'VaR 95% (1-day)', value: risk.var_95_rupees != null ? fmtINR(Math.abs(risk.var_95_rupees)) : 'N/A', sub: 'max expected daily loss', accent: 'text-rose-400' },
                    { label: 'VaR 99% (1-day)', value: risk.var_99_rupees != null ? fmtINR(Math.abs(risk.var_99_rupees)) : 'N/A', sub: 'worst-case daily loss', accent: 'text-rose-500' },
                    { label: 'Max Drawdown', value: risk.max_drawdown_pct != null ? `${fmt(risk.max_drawdown_pct)}%` : 'N/A', accent: 'text-rose-400' },
                  ].map(item => (
                    <div key={item.label} className="flex items-center justify-between gap-2">
                      <div>
                        <p className="text-[10px] text-slate-500">{item.label}</p>
                        {item.sub && <p className="text-[9px] text-slate-700">{item.sub}</p>}
                      </div>
                      <span className={`text-[11px] font-bold ${item.accent || 'text-slate-200'}`}>{item.value}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 1Y price chart */}
            {Object.keys(result.price_history || {}).length > 0 && (
              <PriceHistoryChart history={result.price_history} holdings={result.holdings} />
            )}

            {/* Correlation heatmap */}
            {risk?.correlation_pairs?.length > 0 && (
              <CorrelationHeatmap
                pairs={risk.correlation_pairs}
                tickers={result.holdings.map(h => h.ticker)}
              />
            )}

            {/* Smart Capital Advisor CTA — show when any holding is in loss */}
            {result.holdings.some(h => h.pnl_pct < -0.5) && (
              <div className="rounded-xl border border-violet-500/25 bg-gradient-to-br from-violet-500/8 to-indigo-500/5 p-4 flex items-center gap-4">
                <div className="h-10 w-10 rounded-xl bg-violet-500/15 border border-violet-500/25 flex items-center justify-center shrink-0">
                  <Wallet className="h-5 w-5 text-violet-400" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-bold text-white">Got spare cash sitting idle?</p>
                  <p className="text-[11px] text-slate-400">Tell us how much floating money you have and your time horizon — we'll build a precision recovery plan.</p>
                </div>
                <button
                  onClick={() => setShowAdvisor(true)}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white text-xs font-bold rounded-xl transition-all cursor-pointer shadow-lg shadow-violet-500/20 shrink-0"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Get Plan
                </button>
              </div>
            )}

            {/* Recovery Advisor */}
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 overflow-hidden">
              <div className="flex items-center justify-between px-4 py-3 border-b border-amber-500/10">
                <div className="flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-amber-400" />
                  <p className="text-sm font-bold text-white">Recovery Advisor</p>
                  <span className="text-[9px] text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full font-bold uppercase tracking-wider">AI</span>
                </div>
                <button
                  onClick={getAdvice}
                  disabled={advLoading}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-500 hover:bg-amber-400 disabled:opacity-50 text-black text-[11px] font-bold rounded-lg transition cursor-pointer"
                >
                  <Zap className={`h-3 w-3 ${advLoading ? 'animate-pulse' : ''}`} />
                  {advLoading ? 'Analysing…' : advice ? 'Refresh' : 'Get Advice'}
                </button>
              </div>

              {!advice && !advLoading && (
                <div className="px-4 py-8 text-center">
                  <Lightbulb className="h-8 w-8 mx-auto mb-2 text-amber-500/30" />
                  <p className="text-sm text-slate-400 mb-1">Smart recovery analysis for your portfolio</p>
                  <p className="text-[11px] text-slate-600">RSI signals · News sentiment · Averaging down calculator</p>
                </div>
              )}

              {advLoading && (
                <div className="px-4 py-8 text-center text-slate-500">
                  <div className="h-6 w-6 border-2 border-amber-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
                  <p className="text-xs">Fetching signals and sentiment for each holding…</p>
                </div>
              )}

              {advice && !advLoading && (() => {
                const ps = advice.portfolio_summary;
                const sentColor = ps.sentiment_label === 'Positive' ? 'text-emerald-400' : ps.sentiment_label === 'Negative' ? 'text-rose-400' : 'text-amber-400';
                return (
                  <div className="p-4 space-y-4">
                    {/* Portfolio sentiment banner */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      {[{
                        label: 'In Loss', value: ps.in_loss, color: ps.in_loss > 0 ? 'text-rose-400' : 'text-slate-400',
                      }, {
                        label: 'In Profit', value: ps.in_profit, color: 'text-emerald-400',
                      }, {
                        label: 'Market Mood', value: ps.sentiment_label, color: sentColor,
                      }, {
                        label: 'Capital to Avg Down', value: fmtINR(ps.total_capital_to_avg_down), color: 'text-amber-400',
                      }].map(item => (
                        <div key={item.label} className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-2.5 text-center">
                          <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">{item.label}</p>
                          <p className={`text-sm font-bold ${item.color}`}>{item.value}</p>
                        </div>
                      ))}
                    </div>

                    {/* Per-holding cards */}
                    <div className="space-y-3">
                      {advice.insights.map(ins => {
                        const colors = {
                          emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', text: 'text-emerald-400', badge: 'bg-emerald-500/20' },
                          rose:    { bg: 'bg-rose-500/10',    border: 'border-rose-500/20',    text: 'text-rose-400',    badge: 'bg-rose-500/20' },
                          amber:   { bg: 'bg-amber-500/10',   border: 'border-amber-500/20',   text: 'text-amber-400',   badge: 'bg-amber-500/20' },
                          indigo:  { bg: 'bg-indigo-500/10',  border: 'border-indigo-500/20',  text: 'text-indigo-400',  badge: 'bg-indigo-500/20' },
                        };
                        const c = colors[ins.rec_color] || colors.amber;
                        const short = ins.ticker.replace('.NS','').replace('.BO','');
                        return (
                          <div key={ins.ticker} className={`rounded-xl border ${c.border} ${c.bg} p-3`}>
                            <div className="flex items-start justify-between gap-3 mb-2">
                              <div className="flex items-center gap-2">
                                <div className="h-8 w-8 rounded-lg bg-white/[0.05] border border-white/[0.08] flex items-center justify-center shrink-0">
                                  <span className="text-[9px] font-black text-white">{short.slice(0,3)}</span>
                                </div>
                                <div>
                                  <p className="text-[12px] font-bold text-white">{short}</p>
                                  <div className="flex items-center gap-1.5 mt-0.5">
                                    <span className={`text-[10px] font-bold ${ins.pnl_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                      {ins.pnl_pct >= 0 ? '+' : ''}{fmt(ins.pnl_pct)}%
                                    </span>
                                    {ins.rsi != null && (
                                      <span className="text-[9px] text-slate-500">· RSI {ins.rsi}</span>
                                    )}
                                    <span className={`text-[9px] ${
                                      ins.signal === 'OVERSOLD' ? 'text-emerald-400' :
                                      ins.signal === 'OVERBOUGHT' ? 'text-rose-400' : 'text-slate-500'
                                    }`}>
                                      · {ins.signal}
                                    </span>
                                  </div>
                                </div>
                              </div>
                              <span className={`shrink-0 text-[10px] font-bold px-2.5 py-1 rounded-full ${c.badge} ${c.text} border ${c.border}`}>
                                {ins.rec_label}
                              </span>
                            </div>

                            <p className="text-[10px] text-slate-400 leading-relaxed mb-2">{ins.rec_reason}</p>

                            {/* Metrics row */}
                            <div className="flex flex-wrap gap-2">
                              <div className="flex items-center gap-1 text-[9px] text-slate-500">
                                <Target className="h-3 w-3" />
                                Sentiment: <span className={`font-bold ml-0.5 ${
                                  ins.news_sentiment > 0.1 ? 'text-emerald-400' :
                                  ins.news_sentiment < -0.1 ? 'text-rose-400' : 'text-slate-400'
                                }`}>{ins.news_sentiment > 0 ? '+' : ''}{ins.news_sentiment}</span>
                              </div>
                              {ins.in_loss && (
                                <div className="flex items-center gap-1 text-[9px] text-slate-500">
                                  <ArrowDownRight className="h-3 w-3" />
                                  Need <span className="font-bold text-rose-400 ml-0.5">+{fmt(ins.gain_to_breakeven_pct)}%</span> to break even
                                </div>
                              )}
                            </div>

                            {/* Avg down box */}
                            {ins.avg_down && (
                              <div className="mt-2.5 p-2.5 rounded-lg bg-white/[0.04] border border-white/[0.06]">
                                <p className="text-[9px] text-slate-500 font-bold uppercase tracking-wider mb-1.5">Averaging Down Calculator</p>
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
                                  {[{
                                    label: 'Add Qty', value: ins.avg_down.add_qty,
                                  }, {
                                    label: 'Capital Needed', value: fmtINR(ins.avg_down.add_cost),
                                  }, {
                                    label: 'New Avg Cost', value: `₹${ins.avg_down.new_avg_price.toLocaleString('en-IN')}`,
                                  }, {
                                    label: 'Need to Break Even', value: `+${fmt(ins.avg_down.new_gain_to_breakeven_pct)}%`,
                                  }].map(m => (
                                    <div key={m.label} className="rounded bg-white/[0.03] p-1.5">
                                      <p className="text-[8px] text-slate-600 mb-0.5">{m.label}</p>
                                      <p className="text-[10px] font-bold text-white">{m.value}</p>
                                    </div>
                                  ))}
                                </div>
                                <p className="text-[9px] text-slate-600 mt-1.5">
                                  Buying {ins.avg_down.add_qty} more shares reduces your avg cost by <span className="text-amber-400 font-bold">{ins.avg_down.avg_cost_reduction_pct}%</span>
                                </p>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    <p className="text-[9px] text-slate-700 text-center">Not financial advice. For educational purposes only.</p>
                  </div>
                );
              })()}
            </div>

            {/* As of */}
            <p className="text-[10px] text-slate-700 text-center">
              Last updated: {new Date(result.as_of).toLocaleString('en-IN')} · Data ~15 min delayed · Not financial advice
            </p>
          </>
        )}
      </div>
    </div>

    {/* Smart Capital Advisor Modal */}
    {showAdvisor && (
      <SmartCapitalAdvisor
        holdings={validRows}
        onClose={() => setShowAdvisor(false)}
      />
    )}
    </>
  );
}
