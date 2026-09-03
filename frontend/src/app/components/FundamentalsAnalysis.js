'use client';

import { useState, useEffect, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell,
  LineChart, Line, Legend,
} from 'recharts';
import {
  TrendingUp, TrendingDown, DollarSign, Users, PieChart,
  Calendar, AlertTriangle, RefreshCw, ChevronDown, ChevronUp,
  Award, Landmark, ArrowUpRight, ArrowDownRight, Minus,
} from 'lucide-react';
import InfoBadge from './InfoBadge';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

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

// ── Formatters ────────────────────────────────────────────────────────────────
const fmtCr = (v) => {
  if (v == null) return 'N/A';
  const abs = Math.abs(v);
  if (abs >= 1e12) return `₹${(v / 1e12).toFixed(2)}T`;
  if (abs >= 1e9)  return `₹${(v / 1e9).toFixed(2)}B`;
  if (abs >= 1e7)  return `₹${(v / 1e7).toFixed(1)}Cr`;
  if (abs >= 1e5)  return `₹${(v / 1e5).toFixed(1)}L`;
  return `₹${v.toFixed(0)}`;
};
const fmtPct = (v, d = 1) => v == null ? 'N/A' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(d)}%`;
const fmtNum = (v, d = 2) => v == null ? 'N/A' : Number(v).toFixed(d);

// ── Custom tooltip for bar charts ─────────────────────────────────────────────
function ChartTooltip({ active, payload, label, formatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#0d0d14] border border-white/10 rounded-xl px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1 font-bold">{label}</p>
      {payload.map((p) => (
        <p key={p.name} style={{ color: p.color }} className="font-semibold">
          {p.name}: {formatter ? formatter(p.value) : p.value}
        </p>
      ))}
    </div>
  );
}

// ── CAGR pill ─────────────────────────────────────────────────────────────────
function CagrPill({ label, value }) {
  if (value == null) return null;
  const pos = value >= 0;
  return (
    <div className="flex flex-col items-center gap-1 px-4 py-3 rounded-xl border bg-white/[0.02] border-white/[0.07]">
      <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">{label} CAGR</span>
      <span className={`text-lg font-black ${pos ? 'text-emerald-400' : 'text-rose-400'}`}>
        {pos ? '+' : ''}{value}%
      </span>
    </div>
  );
}

// ── Section header ────────────────────────────────────────────────────────────
function SectionHeader({ icon: Icon, title, badge, color = '#6366f1', infoProps }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
      <div className="flex items-center gap-2">
        <Icon className="h-5 w-5" style={{ color }} />
        <h3 className="text-sm font-bold text-white">{title}</h3>
        {infoProps && <InfoBadge {...infoProps} />}
      </div>
      {badge && (
        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full border"
          style={{ color, borderColor: color + '40', background: color + '15' }}>
          {badge}
        </span>
      )}
    </div>
  );
}

// ── Earnings & Revenue Panel ──────────────────────────────────────────────────
function EarningsPanel({ annual, quarterly, ratios, price_cagr }) {
  const [view, setView] = useState('annual'); // 'annual' | 'quarterly'
  const data = view === 'annual' ? annual : quarterly;
  const xKey = view === 'annual' ? 'year' : 'quarter';

  // Compute revenue growth for annual
  const revGrowth = ratios?.revenue_growth != null
    ? `${(ratios.revenue_growth * 100).toFixed(1)}% YoY`
    : null;
  const earningsGrowth = ratios?.earnings_growth != null
    ? `${(ratios.earnings_growth * 100).toFixed(1)}% YoY`
    : null;

  return (
    <div className="glass-card p-5 space-y-4">
      <SectionHeader
        icon={TrendingUp}
        title="Revenue & Earnings Trend"
        badge={revGrowth ? `Rev ${revGrowth}` : 'Annual Financials'}
        color="#6366f1"
        infoProps={{
          title: "Revenue & Earnings Growth",
          what: "Historical progression of Topline Revenue and Bottomline Net Profit on annual and quarterly cadences.",
          why: "Consistent double-digit revenue and earnings compounding is the primary fundamental driver of long-term share price appreciation.",
          interpretation: "Look for Net Profit growing faster than Revenue, indicating expanding operating leverage."
        }}
      />

      {/* Toggle */}
      <div className="flex gap-2 p-1 bg-white/[0.02] border border-white/[0.06] rounded-xl w-fit">
        {['annual', 'quarterly'].map(v => (
          <button key={v} onClick={() => setView(v)}
            className={`px-3 py-1.5 rounded-lg text-[11px] font-bold transition cursor-pointer ${
              view === v
                ? 'bg-indigo-500/20 border border-indigo-500/30 text-indigo-300'
                : 'text-slate-500 hover:text-slate-300'
            }`}>
            {v === 'annual' ? '📅 Annual' : '📊 Quarterly'}
          </button>
        ))}
      </div>

      {/* CAGR Pills */}
      {view === 'annual' && (
        <div className="flex flex-wrap gap-2">
          <CagrPill label="1Y Price" value={price_cagr?.['1y']} />
          <CagrPill label="3Y Price" value={price_cagr?.['3y']} />
          <CagrPill label="5Y Price" value={price_cagr?.['5y']} />
          {ratios?.roe != null && (
            <div className="flex flex-col items-center gap-1 px-4 py-3 rounded-xl border bg-white/[0.02] border-white/[0.07]">
              <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">ROE</span>
              <span className="text-lg font-black text-amber-400">{fmtPct(ratios.roe * 100)}</span>
            </div>
          )}
          {ratios?.profit_margins != null && (
            <div className="flex flex-col items-center gap-1 px-4 py-3 rounded-xl border bg-white/[0.02] border-white/[0.07]">
              <span className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">Net Margin</span>
              <span className="text-lg font-black text-violet-400">{fmtPct(ratios.profit_margins * 100)}</span>
            </div>
          )}
        </div>
      )}

      {/* Bar Chart */}
      {data?.length > 0 ? (
        <div className="bg-white/[0.01] rounded-xl p-3 border border-white/[0.04]">
          <ChartContainer height={220}>
            {(width, height) => (
              <BarChart width={width} height={height} data={data} barCategoryGap="25%" margin={{ top: 4, right: 4, left: -25, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff08" />
                <XAxis dataKey={xKey} tick={{ fill: '#555', fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#555', fontSize: 9 }} axisLine={false} tickLine={false}
                  tickFormatter={v => fmtCr(v).replace('₹', '')} width={45} tickCount={4} />
                <Tooltip content={<ChartTooltip formatter={fmtCr} />} />
                <Bar dataKey="revenue" name="Revenue" fill="#6366f1" radius={[4, 4, 0, 0]}>
                  {data.map((_, i) => <Cell key={i} fill={i === data.length - 1 ? '#818cf8' : '#6366f155'} />)}
                </Bar>
                <Bar dataKey="net_income" name="Net Income" fill="#10b981" radius={[4, 4, 0, 0]}>
                  {data.map((entry, i) => (
                    <Cell key={i} fill={
                      entry.net_income < 0 ? '#f43f5e88'
                      : i === data.length - 1 ? '#10b981' : '#10b98155'
                    } />
                  ))}
                </Bar>
              </BarChart>
            )}
          </ChartContainer>
        </div>
      ) : (
        <div className="h-32 flex items-center justify-center text-slate-600 text-sm">
          Financial history not available for this ticker.
        </div>
      )}

      {/* Ratio row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
        {[
          { label: 'P/E Ratio',  val: fmtNum(ratios?.pe_ratio, 1), color: 'text-indigo-400' },
          { label: 'P/B Ratio',  val: fmtNum(ratios?.pb_ratio, 2), color: 'text-violet-400' },
          { label: 'D/E Ratio',  val: fmtNum(ratios?.debt_to_equity, 2), color: 'text-amber-400' },
          { label: 'Cur. Ratio', val: fmtNum(ratios?.current_ratio, 2), color: 'text-emerald-400' },
        ].map(({ label, val, color }) => (
          <div key={label} className="text-center p-2 rounded-lg bg-white/[0.02] border border-white/[0.05]">
            <p className="text-[9px] text-slate-500 uppercase tracking-wider mb-0.5">{label}</p>
            <p className={`text-sm font-bold ${color}`}>{val}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Dividend Panel ────────────────────────────────────────────────────────────
function DividendPanel({ dividend }) {
  const [showHistory, setShowHistory] = useState(false);
  const { yield_pct, rate, payout_ratio_pct, five_yr_avg_yield_pct,
          ex_dividend_date, annual_totals, history,
          last_split_factor, last_split_date } = dividend || {};

  const hasDividends = annual_totals?.length > 0 || rate > 0;

  return (
    <div className="glass-card p-5 space-y-4">
      <SectionHeader
        icon={DollarSign}
        title="Dividend Analysis"
        badge={hasDividends ? `${yield_pct ?? 0}% Yield` : 'No Dividends'}
        color="#10b981"
        infoProps={{
          title: "Dividend Yield & Payout Health",
          what: "Cash distributions returned to shareholders as a percentage of share price and net profit.",
          why: "Provides direct shareholder yield and signals management confidence in ongoing cash generation.",
          interpretation: "Healthy payout ratios typically range between 20%–50%; >80% may threaten dividend sustainability."
        }}
      />

      {!hasDividends ? (
        <div className="flex items-center gap-3 p-4 bg-slate-800/40 border border-slate-700/40 rounded-xl text-slate-400 text-sm">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-500" />
          <div>
            <p className="font-bold text-slate-300">No dividend history found</p>
            <p className="text-xs text-slate-500 mt-0.5">This company has not paid dividends recently. Typical for high-growth companies that reinvest profits.</p>
          </div>
        </div>
      ) : (
        <>
          {/* Key metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: 'Current Yield',  val: yield_pct != null ? `${yield_pct}%` : 'N/A', color: 'text-emerald-400', icon: '💰' },
              { label: 'Annual Dividend', val: rate != null ? `₹${rate}` : 'N/A', color: 'text-white', icon: '📅' },
              { label: 'Payout Ratio', val: payout_ratio_pct != null ? `${payout_ratio_pct}%` : 'N/A', color: payout_ratio_pct > 80 ? 'text-rose-400' : 'text-indigo-400', icon: '📊' },
              { label: '5Y Avg Yield', val: five_yr_avg_yield_pct != null ? `${five_yr_avg_yield_pct}%` : 'N/A', color: 'text-violet-400', icon: '📈' },
            ].map(({ label, val, color, icon }) => (
              <div key={label} className="text-center p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
                <p className="text-base">{icon}</p>
                <p className={`text-sm font-black mt-1 ${color}`}>{val}</p>
                <p className="text-[9px] text-slate-500 mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          {/* Ex-div date & split */}
          <div className="flex flex-wrap gap-2">
            {ex_dividend_date && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-xs text-emerald-300">
                <Calendar className="h-3.5 w-3.5" />
                <span>Ex-Div Date: <strong>{ex_dividend_date}</strong></span>
              </div>
            )}
            {last_split_factor && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-500/10 border border-indigo-500/20 rounded-lg text-xs text-indigo-300">
                <Award className="h-3.5 w-3.5" />
                <span>Last Split: <strong>{last_split_factor}</strong> on {last_split_date}</span>
              </div>
            )}
          </div>

          {/* Annual dividends bar chart */}
          {annual_totals?.length > 0 && (
            <div className="bg-white/[0.01] rounded-xl p-3 border border-white/[0.04]">
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-2">Annual Dividend Per Share (₹)</p>
              <ChartContainer height={150}>
                {(width, height) => (
                  <BarChart width={width} height={height} data={annual_totals} margin={{ top: 0, right: 4, left: -25, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
                    <XAxis dataKey="year" tick={{ fill: '#555', fontSize: 9 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: '#555', fontSize: 9 }} axisLine={false} tickLine={false} width={30} tickCount={4} />
                    <Tooltip content={<ChartTooltip formatter={v => `₹${v}`} />} />
                    <Bar dataKey="dividend" name="Dividend" radius={[4, 4, 0, 0]}>
                      {annual_totals.map((_, i) => (
                        <Cell key={i} fill={i === annual_totals.length - 1 ? '#10b981' : '#10b98155'} />
                      ))}
                    </Bar>
                  </BarChart>
                )}
              </ChartContainer>
            </div>
          )}

          {/* Payment history toggle */}
          {history?.length > 0 && (
            <div>
              <button
                onClick={() => setShowHistory(v => !v)}
                className="flex items-center gap-1.5 text-[11px] text-slate-400 hover:text-slate-200 transition cursor-pointer"
              >
                {showHistory ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                {showHistory ? 'Hide' : 'Show'} payment history ({history.length} payments)
              </button>
              {showHistory && (
                <div className="mt-2 max-h-44 overflow-y-auto space-y-1 pr-1">
                  {[...history].reverse().map((d, i) => (
                    <div key={i} className="flex items-center justify-between py-1.5 px-3 rounded-lg bg-white/[0.02] border border-white/[0.04] text-xs">
                      <span className="text-slate-400">{d.date}</span>
                      <span className="text-emerald-400 font-bold">₹{d.amount}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Ownership / Shareholding Panel ────────────────────────────────────────────
function OwnershipPanel({ ownership }) {
  const { promoter_pct, institutions_pct, retail_pct, top_insiders } = ownership || {};

  const segments = [
    { label: 'Promoter / Insider', pct: promoter_pct, color: '#6366f1', bg: 'bg-indigo-500', desc: 'Founders, promoter group' },
    { label: 'Institutions (FII+DII)', pct: institutions_pct, color: '#10b981', bg: 'bg-emerald-500', desc: 'Foreign & domestic funds' },
    { label: 'Public / Retail', pct: retail_pct, color: '#f59e0b', bg: 'bg-amber-500', desc: 'General public' },
  ].filter(s => s.pct != null);

  const hasData = segments.length > 0;

  return (
    <div className="glass-card p-5 space-y-4">
      <SectionHeader
        icon={Users}
        title="Shareholding Pattern"
        badge="Ownership Breakdown"
        color="#6366f1"
        infoProps={{
          title: "Shareholding & Insider Ownership",
          what: "Equity breakdown between Promoters/Founders, Institutions (FII + DII), and Retail public.",
          why: "High promoter skin-in-the-game aligns founder interests with minority investors; institutional holding provides valuation support.",
          interpretation: "Promoter holding > 50% with low or zero shares pledged indicates strong alignment and governance."
        }}
      />

      {!hasData ? (
        <div className="flex items-center gap-3 p-4 bg-slate-800/40 border border-slate-700/40 rounded-xl text-slate-400 text-sm">
          <AlertTriangle className="h-5 w-5 shrink-0 text-amber-500" />
          <div>
            <p className="font-bold text-slate-300">Ownership data unavailable</p>
            <p className="text-xs text-slate-500 mt-0.5">Shareholding pattern not available for this ticker via Yahoo Finance.</p>
          </div>
        </div>
      ) : (
        <>
          {/* Stacked progress bar */}
          <div className="h-3 w-full rounded-full overflow-hidden flex gap-0.5">
            {segments.map(({ pct, bg }) => (
              <div key={bg} className={`${bg} h-full transition-all duration-700`}
                style={{ width: `${pct}%` }} />
            ))}
          </div>

          {/* Segment cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {segments.map(({ label, pct, color, desc }) => {
              const isHighPromoter = label.startsWith('Promoter') && pct >= 50;
              const isLowPromoter  = label.startsWith('Promoter') && pct < 30;
              return (
                <div key={label} className="p-3 rounded-xl border bg-white/[0.02] border-white/[0.06] text-center space-y-1">
                  <p className="text-[8px] text-slate-500 uppercase tracking-widest font-bold">{label}</p>
                  <p className="text-2xl font-black" style={{ color }}>{pct}%</p>
                  <p className="text-[9px] text-slate-400">{desc}</p>
                  {isHighPromoter && (
                    <span className="inline-block text-[8px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/25 text-emerald-400">
                      ✓ Strong Promoter Confidence
                    </span>
                  )}
                  {isLowPromoter && (
                    <span className="inline-block text-[8px] font-bold px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/25 text-amber-400">
                      ⚠ Low Promoter Holding
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Interpretation note */}
          <div className="p-3 bg-indigo-500/5 border border-indigo-500/15 rounded-xl text-[11px] text-slate-300 leading-relaxed">
            <span className="font-bold text-slate-300">How to read: </span>
            High promoter holding (&gt;50%) signals management confidence.
            {institutions_pct > 30 ? ` Strong institutional interest (${institutions_pct}%) — FIIs and DIIs have conviction in the business.` : ''}
            {promoter_pct < 30 ? ' ⚠ Low promoter stake — watch for dilution risk or lack of skin-in-the-game.' : ''}
          </div>

          {/* Insider transactions */}
          {top_insiders?.length > 0 && (
            <div>
              <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-2">Recent Insider Activity</p>
              <div className="space-y-1.5">
                {top_insiders.map((h, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04] text-xs">
                    <div>
                      <p className="text-slate-300 font-semibold">{h.name}</p>
                      <p className="text-slate-400 text-[10px]">{h.relation}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-400">{h.transaction || '—'}</p>
                      {h.shares && <p className="text-slate-400 text-[10px]">{Number(h.shares).toLocaleString('en-IN')} shares</p>}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Main exported component ───────────────────────────────────────────────────
export default function FundamentalsAnalysis({ ticker }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!ticker) return;
    setData(null);
    setError(null);
    setLoading(true);
    fetch(`${API_BASE_URL}/api/fundamentals?ticker=${encodeURIComponent(ticker)}`)
      .then(r => r.json().then(j => ({ ok: r.ok, j })))
      .then(({ ok, j }) => {
        if (!ok) throw new Error(j.detail || 'Failed to load fundamentals');
        setData(j);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [ticker]);

  if (loading) return (
    <div className="glass-card p-8 flex flex-col items-center justify-center gap-3 min-h-[200px]">
      <div className="h-9 w-9 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin" />
      <p className="text-slate-400 text-sm animate-pulse">Loading fundamentals — revenue, dividends, ownership…</p>
    </div>
  );

  if (error) return (
    <div className="glass-card p-5 flex items-center gap-3 border border-rose-500/20 bg-rose-500/5 text-rose-400 text-sm">
      <AlertTriangle className="h-5 w-5 shrink-0" />
      <div>
        <p className="font-bold">Fundamentals unavailable</p>
        <p className="text-xs text-slate-500 mt-0.5">{error}</p>
      </div>
    </div>
  );

  if (!data) return null;

  return (
    <div className="space-y-4">
      {/* Section title */}
      <div className="flex items-center gap-3 px-1">
        <div className="h-8 w-8 rounded-lg bg-indigo-500/15 border border-indigo-500/25 flex items-center justify-center">
          <Landmark className="h-4 w-4 text-indigo-400" />
        </div>
        <div>
          <h2 className="text-base font-black text-white">Fundamental Intelligence</h2>
          <p className="text-[11px] text-slate-500">Revenue trend · Dividends · Ownership pattern</p>
        </div>
      </div>

      {/* Earnings Panel (full width) */}
      <EarningsPanel
        annual={data.income_annual}
        quarterly={data.income_quarterly}
        ratios={data.ratios}
        price_cagr={data.price_cagr}
      />

      {/* Dividend + Ownership (side-by-side on large screens) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <DividendPanel dividend={data.dividend} />
        <OwnershipPanel ownership={data.ownership} />
      </div>
    </div>
  );
}
