'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, CartesianGrid,
} from 'recharts';
import {
  FlaskConical, TrendingUp, TrendingDown, RefreshCw,
  AlertCircle, Trophy, Target, Activity, BarChart2,
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

// ─── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (n, decimals = 2) =>
  n == null ? 'N/A' : Number(n).toFixed(decimals);

const pctColor = (v) =>
  v > 0 ? 'text-emerald-400' : v < 0 ? 'text-rose-400' : 'text-slate-400';

const pctSign = (v) => (v > 0 ? '+' : '');

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg bg-[#0d0d0d] border border-white/[0.08] p-2.5 text-xs shadow-xl">
      <p className="text-slate-400 mb-1.5 font-mono">{label}</p>
      {payload.map((p) => (
        <div key={p.dataKey} className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: p.color }} />
          <span className="text-slate-300 w-20">{p.name}:</span>
          <span className="font-bold text-white">₹{Number(p.value).toLocaleString('en-IN')}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Stat Card ────────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] p-3 flex flex-col gap-0.5">
      <p className="text-[9px] uppercase tracking-wider font-bold text-slate-500">{label}</p>
      <p className={`text-sm font-bold ${accent || 'text-white'}`}>{value}</p>
      {sub && <p className="text-[9px] text-slate-600">{sub}</p>}
    </div>
  );
}

// ─── Trade Log Row ────────────────────────────────────────────────────────────
function TradeRow({ trade, idx }) {
  const isWin = trade.result === 'WIN';
  return (
    <div className={`flex items-center gap-2 py-1.5 px-2 rounded text-[10px] ${
      isWin ? 'bg-emerald-500/5' : 'bg-rose-500/5'
    }`}>
      <span className="text-slate-500 w-5 text-right shrink-0">{idx + 1}</span>
      <span className={`w-2 h-2 rounded-full shrink-0 ${isWin ? 'bg-emerald-500' : 'bg-rose-500'}`} />
      <span className="text-slate-400 w-20 shrink-0">{trade.entry_date}</span>
      <span className="text-slate-600">→</span>
      <span className="text-slate-400 w-20 shrink-0">{trade.exit_date}</span>
      <span className="text-slate-300 w-20 shrink-0 text-right">
        ₹{trade.entry_price.toLocaleString('en-IN')}
      </span>
      <span className="text-slate-600">→</span>
      <span className="text-slate-300 w-20 shrink-0 text-right">
        ₹{trade.exit_price.toLocaleString('en-IN')}
      </span>
      <span className={`ml-auto font-bold shrink-0 ${isWin ? 'text-emerald-400' : 'text-rose-400'}`}>
        {pctSign(trade.return_pct)}{trade.return_pct}%
      </span>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function Backtesting({ ticker }) {
  const [data, setData]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [period, setPeriod]     = useState('2y');
  const [showTrades, setShowTrades] = useState(false);

  const fetchBacktest = useCallback(async (p = period) => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/backtest?ticker=${ticker}&period=${p}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Backtest failed');
      setData(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, period]);

  // Reset state when ticker changes (don't auto-fetch — backtest is expensive)
  useEffect(() => {
    setData(null);
    setError(null);
    setLoading(false);
    setShowTrades(false);
  }, [ticker]);

  // Merge equity curves into one array keyed by date
  const chartData = (() => {
    if (!data?.equity_curves) return [];
    const map = {};
    const add = (arr, key) =>
      arr.forEach(({ date, value }) => {
        if (!map[date]) map[date] = { date };
        map[date][key] = value;
      });
    add(data.equity_curves.strategy,     'Strategy');
    add(data.equity_curves.buy_and_hold, 'Buy & Hold');
    if (data.equity_curves.nifty?.length)
      add(data.equity_curves.nifty, 'Nifty 50');
    return Object.values(map).sort((a, b) => a.date.localeCompare(b.date));
  })();

  const s = data?.stats;

  return (
    <div className="glass-card p-4 sm:p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-500/20 flex items-center justify-center">
            <FlaskConical className="h-4 w-4 text-amber-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white">Signal Backtesting</h3>
            <p className="text-[10px] text-slate-400">RSI(14) + MACD Crossover + ATR Stop-Loss</p>
          </div>
        </div>
        <button
          onClick={() => fetchBacktest(period)}
          disabled={loading}
          className="p-2 rounded-lg text-slate-500 active:text-slate-300 transition disabled:opacity-40 cursor-pointer"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Period selector */}
      <div className="flex gap-1 p-0.5 bg-white/[0.03] rounded-lg border border-white/[0.06] mb-4">
        {['1y', '2y', '5y'].map((p) => (
          <button
            key={p}
            onClick={() => { setPeriod(p); fetchBacktest(p); }}
            disabled={loading}
            className={`flex-1 py-1.5 text-[11px] font-semibold rounded-md transition ${
              period === p ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-900'
            } disabled:opacity-50`}
          >
            {p.toUpperCase()}
          </button>
        ))}
      </div>

      {/* States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-slate-500">
          <FlaskConical className="h-6 w-6 animate-pulse text-amber-400" />
          <span className="text-xs text-slate-400">Running backtest on {ticker}…</span>
        </div>
      )}

      {error && (
        <div className={`flex items-start gap-3 p-4 rounded-lg text-sm border ${
          error.includes('404') || error.includes('Not Found')
            ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
            : 'bg-rose-500/10 border-rose-500/20 text-rose-400'
        }`}>
          <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
          <div>
            {error.includes('404') || error.includes('Not Found') ? (
              <>
                <p className="font-bold mb-0.5">Backend deploying…</p>
                <p className="text-[11px] opacity-80">
                  The backtest endpoint is being deployed to Vercel. Try again in ~2 minutes.
                </p>
              </>
            ) : (
              <span>{error}</span>
            )}
          </div>
        </div>
      )}

      {!loading && !error && !data && (
        <div className="text-center py-10">
          <FlaskConical className="h-8 w-8 mx-auto mb-3 text-amber-500/40" />
          <p className="text-sm text-slate-400 mb-1 font-medium">Backtest not run yet</p>
          <p className="text-[11px] text-slate-600 mb-4">Simulate an RSI+MACD strategy on {ticker?.replace('.NS','')}</p>
          <button
            onClick={() => fetchBacktest(period)}
            className="px-5 py-2 bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold rounded-lg transition cursor-pointer"
          >
            Run Backtest
          </button>
        </div>
      )}

      {!loading && data && s && (
        <>
          {/* Alpha badge */}
          <div className={`flex items-center justify-between mb-4 p-3 rounded-xl border ${
            s.alpha >= 0
              ? 'bg-emerald-500/10 border-emerald-500/30'
              : 'bg-rose-500/10 border-rose-500/30'
          }`}>
            <div>
              <p className="text-[10px] text-slate-400 uppercase tracking-wider font-bold mb-0.5">
                Strategy Alpha vs Buy &amp; Hold
              </p>
              <p className={`text-2xl font-black ${s.alpha >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {pctSign(s.alpha)}{fmt(s.alpha)}%
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                Strategy: {pctSign(s.total_return_pct)}{fmt(s.total_return_pct)}% vs B&amp;H: {pctSign(s.bh_return_pct)}{fmt(s.bh_return_pct)}%
              </p>
            </div>
            <div className="text-right">
              <p className="text-[10px] text-slate-400 mb-0.5">Final Value</p>
              <p className="text-base font-bold text-white">
                ₹{Number(s.final_value).toLocaleString('en-IN')}
              </p>
              <p className="text-[10px] text-slate-500">
                from ₹{Number(s.initial_capital).toLocaleString('en-IN')}
              </p>
            </div>
          </div>

          {/* Equity curve chart */}
          {chartData.length > 0 && (
            <div className="mb-4 rounded-xl bg-white/[0.02] border border-white/[0.05] p-3">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-3">
                Equity Curve — Strategy vs Benchmarks
              </p>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.04)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 9, fill: '#555' }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => v.slice(2, 7)}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: '#555' }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend
                    wrapperStyle={{ fontSize: '10px', paddingTop: '8px' }}
                    iconType="circle"
                    iconSize={6}
                  />
                  <ReferenceLine
                    y={s.initial_capital}
                    stroke="rgba(255,255,255,0.1)"
                    strokeDasharray="3 3"
                  />
                  <Line
                    type="monotone" dataKey="Strategy"
                    stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 3 }}
                  />
                  <Line
                    type="monotone" dataKey="Buy & Hold"
                    stroke="#6366f1" strokeWidth={1.5} dot={false} strokeDasharray="4 2"
                    activeDot={{ r: 3 }}
                  />
                  {chartData[0]?.['Nifty 50'] !== undefined && (
                    <Line
                      type="monotone" dataKey="Nifty 50"
                      stroke="#22d3ee" strokeWidth={1.5} dot={false} strokeDasharray="2 3"
                      activeDot={{ r: 3 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Stats grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
            <StatCard
              label="Sharpe Ratio"
              value={fmt(s.sharpe_ratio, 3)}
              sub="Risk-adjusted return"
              accent={s.sharpe_ratio >= 1 ? 'text-emerald-400' : s.sharpe_ratio >= 0 ? 'text-amber-400' : 'text-rose-400'}
            />
            <StatCard
              label="Max Drawdown"
              value={`${fmt(s.max_drawdown_pct)}%`}
              sub="Peak-to-trough"
              accent="text-rose-400"
            />
            <StatCard
              label="Calmar Ratio"
              value={fmt(s.calmar_ratio, 3)}
              sub="Return / Max DD"
              accent={s.calmar_ratio >= 1 ? 'text-emerald-400' : 'text-amber-400'}
            />
            <StatCard
              label="Ann. Volatility"
              value={`${fmt(s.annualized_vol)}%`}
              sub="Strategy vol"
            />
          </div>

          {/* Trade stats */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
            <StatCard
              label="Total Trades"
              value={s.total_trades}
              sub={`${s.win_rate_pct}% win rate`}
            />
            <StatCard
              label="Win Rate"
              value={`${fmt(s.win_rate_pct, 1)}%`}
              sub={`${data.trades.filter(t => t.result === 'WIN').length}W / ${data.trades.filter(t => t.result === 'LOSS').length}L shown`}
              accent={s.win_rate_pct >= 55 ? 'text-emerald-400' : 'text-amber-400'}
            />
            <StatCard
              label="Avg Win"
              value={`+${fmt(s.avg_win_pct)}%`}
              accent="text-emerald-400"
              sub="Per winning trade"
            />
            <StatCard
              label="Profit Factor"
              value={s.profit_factor >= 99 ? '∞' : fmt(s.profit_factor, 2)}
              sub="Gross win / gross loss"
              accent={s.profit_factor >= 1.5 ? 'text-emerald-400' : 'text-amber-400'}
            />
          </div>

          {/* Trade log toggle */}
          {data.trades?.length > 0 && (
            <div>
              <button
                onClick={() => setShowTrades((p) => !p)}
                className="flex items-center gap-2 text-[10px] font-bold text-slate-400 hover:text-white transition mb-2 cursor-pointer"
              >
                <BarChart2 className="h-3.5 w-3.5" />
                {showTrades ? 'Hide' : 'Show'} Trade Log ({data.trades.length} most recent)
                <span className="ml-auto text-slate-600">{showTrades ? '▲' : '▼'}</span>
              </button>

              {showTrades && (
                <div className="rounded-lg bg-white/[0.02] border border-white/[0.05] p-2 space-y-1 max-h-64 overflow-y-auto">
                  {/* Header */}
                  <div className="flex items-center gap-2 px-2 text-[9px] text-slate-600 font-bold uppercase tracking-wider pb-1 border-b border-white/[0.04]">
                    <span className="w-5" />
                    <span className="w-2" />
                    <span className="w-20">Entry</span>
                    <span className="w-4" />
                    <span className="w-20">Exit</span>
                    <span className="w-20 text-right">Buy @</span>
                    <span className="w-4" />
                    <span className="w-20 text-right">Sell @</span>
                    <span className="ml-auto">P&amp;L</span>
                  </div>
                  {data.trades.map((t, i) => <TradeRow key={i} trade={t} idx={i} />)}
                </div>
              )}
            </div>
          )}

          {/* Disclaimer */}
          <div className="mt-4 p-2.5 bg-amber-500/5 border border-amber-500/15 rounded-lg">
            <p className="text-[9px] text-amber-200/70 leading-relaxed">
              Backtest uses historical data only. Past performance doesn't guarantee future results.
              Strategy: RSI(14) entry above 35 oversold + MACD confirmation · Exit at RSI 65 / MACD cross / 2×ATR stop.
              No transaction costs, slippage, or taxes applied.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
