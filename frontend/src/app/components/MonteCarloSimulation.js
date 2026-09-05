'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, ReferenceLine
} from 'recharts';
import {
  Compass, TrendingUp, TrendingDown, RefreshCw, AlertCircle,
  Percent, Activity, ArrowRightLeft, Info
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

// Helpers
const fmt = (n, decimals = 2) =>
  n == null ? 'N/A' : Number(n).toFixed(decimals);

function StatCard({ label, value, sub, accent }) {
  return (
    <div className="rounded-lg bg-white/[0.02] border border-white/[0.05] p-3 flex flex-col gap-0.5">
      <p className="text-[9px] uppercase tracking-wider font-bold text-slate-400">{label}</p>
      <p className={`text-sm font-bold ${accent || 'text-white'}`}>{value}</p>
      {sub && <p className="text-[9px] text-slate-400 leading-tight">{sub}</p>}
    </div>
  );
}

function ProbabilityRow({ label, value, isLoss }) {
  const barColor = isLoss ? 'bg-rose-500/20' : 'bg-emerald-500/20';
  const textColor = isLoss ? 'text-rose-400' : 'text-emerald-400';
  return (
    <div className="flex flex-col gap-1 py-1.5 px-2 rounded bg-white/[0.01] border border-white/[0.03]">
      <div className="flex justify-between items-center text-[11px]">
        <span className="text-slate-300 font-medium">{label}</span>
        <span className={`font-mono font-bold ${textColor}`}>{value}%</span>
      </div>
      <div className="w-full bg-white/[0.04] h-1 rounded-full overflow-hidden">
        <div 
          className={`h-full ${isLoss ? 'bg-rose-500' : 'bg-emerald-500'} rounded-full`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

function CustomTooltip({ active, payload, label, currSym = '₹' }) {
  if (!active || !payload?.length) return null;
  const isSimulated = payload[0]?.payload?.is_simulated;

  return (
    <div className="rounded-lg bg-[#0d0d0d] border border-white/[0.08] p-3 text-xs shadow-2xl space-y-1.5 font-sans min-w-[160px]">
      <p className="text-slate-500 font-mono border-b border-white/[0.06] pb-1 mb-1 font-bold">
        {label} {isSimulated ? '(Projected)' : '(Historical)'}
      </p>
      {isSimulated ? (
        <div className="space-y-1">
          <div className="flex justify-between items-center gap-4">
            <span className="text-slate-400 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Median (50%):
            </span>
            <span className="font-bold text-emerald-400">{currSym}{fmt(payload.find(p => p.dataKey === 'p500')?.value)}</span>
          </div>
          <div className="flex justify-between items-center gap-4 text-[10px]">
            <span className="text-slate-500 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-[#3b82f6]/40" /> 50% Range:
            </span>
            <span className="text-slate-300">
              {currSym}{fmt(payload.find(p => p.dataKey === 'p250')?.value)} - {currSym}{fmt(payload.find(p => p.dataKey === 'p750')?.value)}
            </span>
          </div>
          <div className="flex justify-between items-center gap-4 text-[10px]">
            <span className="text-slate-500 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-[#3b82f6]/20" /> 95% Range:
            </span>
            <span className="text-slate-400">
              {currSym}{fmt(payload.find(p => p.dataKey === 'p025')?.value)} - {currSym}{fmt(payload.find(p => p.dataKey === 'p975')?.value)}
            </span>
          </div>
        </div>
      ) : (
        <div className="flex justify-between items-center gap-4">
          <span className="text-slate-400 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-[#3b82f6]" /> Close Price:
          </span>
          <span className="font-bold text-white">{currSym}{fmt(payload.find(p => p.dataKey === 'close')?.value)}</span>
        </div>
      )}
    </div>
  );
}

export default function MonteCarloSimulation({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [horizon, setHorizon] = useState(30);
  const [mode, setMode] = useState('gbm'); // 'gbm' or 'bootstrap'

  const isUS = ticker && !ticker.endsWith('.NS') && !ticker.endsWith('.BO');
  const currSym = data?.currency_symbol || (isUS ? '$' : '₹');

  const fetchSimulation = useCallback(async (h = horizon, m = mode) => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/monte-carlo?ticker=${ticker}&horizon_days=${h}&simulation_mode=${m}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail || 'Failed to fetch simulation');
      setData(json);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [ticker, horizon, mode]);

  useEffect(() => {
    fetchSimulation(horizon, mode);
  }, [ticker, horizon, mode, fetchSimulation]);

  const chartData = (() => {
    if (!data) return [];
    const merged = [];
    
    // Add historical
    if (data.historical) {
      data.historical.forEach(h => {
        merged.push({
          date: h.date,
          close: h.close,
          is_simulated: false
        });
      });
    }

    // Add simulated
    if (data.simulated) {
      data.simulated.forEach((s, idx) => {
        const item = {
          date: s.date,
          p025: s.p025,
          p250: s.p250,
          p500: s.p500,
          p750: s.p750,
          p975: s.p975,
          is_simulated: true
        };

        // Merge sample paths
        if (data.sample_paths) {
          data.sample_paths.forEach((path, pathIdx) => {
            if (path[idx]) {
              item[`path_${pathIdx}`] = path[idx].price;
            }
          });
        }

        merged.push(item);
      });
    }

    return merged;
  })();

  const stats = data?.stats;

  return (
    <div className="glass-card p-4 sm:p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500/20 to-blue-500/20 border border-indigo-500/20 flex items-center justify-center">
            <Compass className="h-4 w-4 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h3 className="text-sm font-bold text-white">Institutional Monte Carlo Projections</h3>
              <InfoBadge infoKey="monte_carlo_var" />
            </div>
            <p className="text-[10px] text-slate-400">
              {mode === 'bootstrap' 
                ? 'Non-parametric historical return bootstrapping (empirical fat tails & jumps)' 
                : 'Geometric Brownian Motion with Student-t fat tails & Bayesian drift shrinkage'}
            </p>
          </div>
        </div>
        <button
          onClick={() => fetchSimulation(horizon, mode)}
          disabled={loading}
          className="p-2 rounded-lg text-slate-500 hover:text-slate-300 active:scale-95 transition disabled:opacity-40 cursor-pointer"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Control Bar: Mode Toggle & Horizon Selector */}
      <div className="flex flex-col sm:flex-row gap-2 mb-4 items-stretch sm:items-center justify-between">
        {/* Simulation Mode Toggle */}
        <div className="flex items-center gap-1 p-0.5 bg-white/[0.02] rounded-lg border border-white/[0.05]">
          <button
            onClick={() => setMode('gbm')}
            disabled={loading}
            className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition flex items-center gap-1.5 cursor-pointer ${
              mode === 'gbm'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <span>GBM (Fat Tails)</span>
            <InfoBadge infoKey="monte_carlo_var" />
          </button>
          <button
            onClick={() => setMode('bootstrap')}
            disabled={loading}
            className={`px-3 py-1.5 text-[11px] font-semibold rounded-md transition flex items-center gap-1.5 cursor-pointer ${
              mode === 'bootstrap'
                ? 'bg-purple-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            <span>Historical Bootstrap</span>
            <InfoBadge infoKey="historical_bootstrap" />
          </button>
        </div>

        {/* Horizon selector */}
        <div className="flex flex-1 sm:max-w-md gap-1 p-0.5 bg-white/[0.02] rounded-lg border border-white/[0.05]">
          {[
            { label: '30D', days: 30 },
            { label: '60D', days: 60 },
            { label: '90D', days: 90 },
            { label: '1Y',  days: 252 },
            { label: '3Y',  days: 756 },
            { label: '5Y',  days: 1260 },
          ].map(({ label, days }) => (
            <button
              key={days}
              onClick={() => setHorizon(days)}
              disabled={loading}
              className={`flex-1 py-1.5 text-[11px] font-semibold rounded-md transition cursor-pointer ${
                horizon === days
                  ? days >= 252 ? 'bg-emerald-600 text-white' : 'bg-indigo-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              } disabled:opacity-50`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Model Diagnostic Badges */}
      {stats && (
        <div className="flex flex-wrap items-center gap-2 mb-4 text-[10px]">
          {mode === 'gbm' ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-indigo-500/10 border border-indigo-500/20 text-indigo-300">
              <Activity className="h-3 w-3" />
              <span>Student-t shocks (ν=5) · Double volatility drag eliminated</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-purple-500/10 border border-purple-500/20 text-purple-300">
              <ArrowRightLeft className="h-3 w-3" />
              <span>Non-parametric empirical returns resampled with replacement</span>
            </div>
          )}

          {mode === 'gbm' && stats.shrinkage_alpha > 0.15 && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
              <span>Bayesian Drift Shrinkage (α={stats.shrinkage_alpha}): Raw {stats.raw_drift_pct}% → Calibrated {stats.ann_drift_pct}%</span>
              <InfoBadge infoKey="drift_shrinkage" />
            </div>
          )}
        </div>
      )}

      {/* States */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-3 text-slate-500">
          <Compass className="h-7 w-7 animate-spin text-indigo-400" />
          <span className="text-xs text-slate-400 font-medium">
            Running 1,000 {mode === 'bootstrap' ? 'historical bootstrap' : 'Student-t GBM'} simulations for {ticker}…
          </span>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 p-4 rounded-lg text-sm border bg-rose-500/10 border-rose-500/20 text-rose-400">
          <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
          <div>
            <p className="font-bold">Simulation Error</p>
            <p className="text-[11px] opacity-80">{error}</p>
          </div>
        </div>
      )}

      {!loading && !error && data && stats && (
        <div className="space-y-5">
          {/* Main Visual Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* Chart Area */}
            <div className="lg:col-span-2 rounded-xl bg-white/[0.01] border border-white/[0.04] p-3">
              <div className="flex justify-between items-center mb-3">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                  Simulation Fan Chart ({horizon}-Day Forecast Horizon · {mode.toUpperCase()})
                </p>
                <div className="flex gap-3 text-[9px] text-slate-400 font-medium">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-emerald-400" /> Median</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#3b82f6]/40" /> 50% Conf</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-[#3b82f6]/10" /> 95% Conf</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded bg-purple-400/40" /> Sample Paths</span>
                </div>
              </div>
              
              <ChartContainer height={260}>
                {(width, height) => (
                  <AreaChart width={width} height={height} data={chartData} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.03)" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 9, fill: '#888' }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => v.slice(5)}
                    />
                    <YAxis
                      domain={['auto', 'auto']}
                      tick={{ fontSize: 9, fill: '#888' }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `${currSym}${v}`}
                    />
                    <Tooltip content={<CustomTooltip currSym={currSym} />} />
                    
                    {/* Outer 95% Confidence Interval (2.5% to 97.5%) */}
                    <Area
                      type="monotone"
                      dataKey="p975"
                      stroke="none"
                      fill="#3b82f6"
                      fillOpacity={0.05}
                      connectNulls={true}
                    />
                    <Area
                      type="monotone"
                      dataKey="p025"
                      stroke="none"
                      fill="#3b82f6"
                      fillOpacity={0}
                      connectNulls={true}
                    />

                    {/* Inner 50% Confidence Interval (25% to 75%) */}
                    <Area
                      type="monotone"
                      dataKey="p750"
                      stroke="none"
                      fill="#3b82f6"
                      fillOpacity={0.15}
                      connectNulls={true}
                    />
                    <Area
                      type="monotone"
                      dataKey="p250"
                      stroke="none"
                      fill="#3b82f6"
                      fillOpacity={0}
                      connectNulls={true}
                    />

                    {/* Sample Stochastic Paths */}
                    <Line type="monotone" dataKey="path_0" stroke="#c084fc" strokeWidth={1} dot={false} opacity={0.3} connectNulls={true} />
                    <Line type="monotone" dataKey="path_1" stroke="#c084fc" strokeWidth={1} dot={false} opacity={0.3} connectNulls={true} />
                    <Line type="monotone" dataKey="path_2" stroke="#c084fc" strokeWidth={1} dot={false} opacity={0.3} connectNulls={true} />
                    <Line type="monotone" dataKey="path_3" stroke="#c084fc" strokeWidth={1} dot={false} opacity={0.3} connectNulls={true} />
                    <Line type="monotone" dataKey="path_4" stroke="#c084fc" strokeWidth={1} dot={false} opacity={0.3} connectNulls={true} />

                    {/* Historical Line */}
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="#3b82f6"
                      strokeWidth={2.5}
                      dot={false}
                      connectNulls={true}
                    />

                    {/* Median Projected Line */}
                    <Line
                      type="monotone"
                      dataKey="p500"
                      stroke="#10b981"
                      strokeWidth={2}
                      strokeDasharray="4 3"
                      dot={false}
                      connectNulls={true}
                    />

                    {/* Reference line showing current price */}
                    <ReferenceLine y={stats.current_price} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                  </AreaChart>
                )}
              </ChartContainer>
            </div>

            {/* Probability Breakdown Column */}
            <div className="flex flex-col gap-2 rounded-xl bg-white/[0.01] border border-white/[0.04] p-3 justify-center">
              <div className="flex items-center justify-between mb-1">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                  Horizon Price Probabilities
                </p>
                <InfoBadge infoKey="horizon_probabilities" />
              </div>
              <div className="space-y-1.5">
                <ProbabilityRow label="Probability of Stock finishing UP" value={stats.prob_up} isLoss={false} />
                <ProbabilityRow label="Probability of Gain ≥ 5%" value={stats.prob_gain_5} isLoss={false} />
                <ProbabilityRow label="Probability of Gain ≥ 10%" value={stats.prob_gain_10} isLoss={false} />
                <ProbabilityRow label="Probability of Gain ≥ 20%" value={stats.prob_gain_20} isLoss={false} />
                <ProbabilityRow label="Probability of Loss ≥ 5%" value={stats.prob_loss_5} isLoss={true} />
                <ProbabilityRow label="Probability of Loss ≥ 10%" value={stats.prob_loss_10} isLoss={true} />
                <ProbabilityRow label="Probability of Loss ≥ 20%" value={stats.prob_loss_20} isLoss={true} />
              </div>
            </div>
          </div>

          {/* Institutional Tail Risk & VaR Panel */}
          <div className="rounded-xl bg-gradient-to-br from-white/[0.02] to-white/[0.005] border border-white/[0.06] p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-rose-400" />
                <h4 className="text-xs font-bold text-white uppercase tracking-wider">Institutional Tail Risk & VaR (FRTB / Basel III)</h4>
              </div>
              <div className="flex items-center gap-1.5">
                <InfoBadge infoKey="monte_carlo_var" />
                <InfoBadge infoKey="expected_shortfall" />
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard
                label="95% Horizon VaR"
                value={`${stats.var_95_pct >= 0 ? '-' : '+'}${fmt(Math.abs(stats.var_95_pct))}%`}
                accent={stats.var_95_pct > 20 ? 'text-rose-400' : 'text-amber-400'}
                sub="Max loss with 95% confidence"
              />
              <StatCard
                label="99% Stress VaR"
                value={`${stats.var_99_pct >= 0 ? '-' : '+'}${fmt(Math.abs(stats.var_99_pct))}%`}
                accent="text-rose-400"
                sub="1-in-100 extreme tail loss"
              />
              <StatCard
                label="Expected Shortfall (CVaR)"
                value={`${stats.cvar_95_pct >= 0 ? '-' : '+'}${fmt(Math.abs(stats.cvar_95_pct))}%`}
                accent="text-rose-500"
                sub="Avg loss in worst 5% crash scenarios"
              />
              <StatCard
                label="Simulated Max Drawdown"
                value={`${fmt(stats.avg_max_drawdown_pct)}%`}
                accent="text-rose-400"
                sub={`Worst path drop: ${fmt(stats.worst_drawdown_pct)}%`}
              />
            </div>
          </div>

          {/* General Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            <StatCard
              label="Expected Price"
              value={`${currSym}${fmt(stats.expected_price_horizon)}`}
              accent="text-emerald-400"
              sub={`Mean path at T+${horizon}`}
            />
            <StatCard
              label="Expected Return"
              value={`${stats.expected_return_pct >= 0 ? '+' : ''}${fmt(stats.expected_return_pct)}%`}
              accent={stats.expected_return_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}
              sub="Expected Compound Return"
            />
            <StatCard
              label="Annualized Volatility"
              value={`${fmt(stats.ann_volatility_pct)}%`}
              sub="Historical standard deviation"
            />
            <StatCard
              label="Calibrated Drift (μ)"
              value={`${stats.ann_drift_pct >= 0 ? '+' : ''}${fmt(stats.ann_drift_pct)}%`}
              accent={stats.ann_drift_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}
              sub={stats.shrinkage_alpha > 0.15 ? `Shrunk from ${fmt(stats.raw_drift_pct)}%` : 'Annualized drift'}
            />
            <StatCard
              label="Extreme Bounds (Max / Min)"
              value={`${currSym}${fmt(stats.max_simulated_price)} / ${currSym}${fmt(stats.min_simulated_price)}`}
              sub="Simulated range"
            />
          </div>

          {/* Educational Note */}
          <div className="p-3 bg-white/[0.01] border border-white/[0.04] rounded-lg flex items-start gap-2.5">
            <Info className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" />
            <div className="text-[10px] text-slate-400 leading-relaxed">
              <span className="font-bold text-white">Institutional Methodology:</span> {mode === 'bootstrap' ? (
                <span>
                  <strong>Historical Bootstrapping (Non-Parametric)</strong> draws random daily returns directly from {ticker}&apos;s empirical trading history with replacement. 
                  This inherently preserves real-world fat tails, flash crashes, earnings gaps, and skewness without imposing an artificial Gaussian normal distribution.
                </span>
              ) : (
                <span>
                  <strong>Geometric Brownian Motion with Student-t Fat Tails (ν=5)</strong> simulates stochastic price paths using standardized Student-t shocks. 
                  The simulation uses direct log-space stepping to eliminate double volatility drag. To protect against Merton estimation risk, Bayesian shrinkage scales the 
                  annualized drift (μ = {fmt(stats.ann_drift_pct)}%) toward long-term market equilibrium prior (12%) as the horizon expands.
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
