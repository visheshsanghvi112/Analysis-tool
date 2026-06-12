'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell
} from 'recharts';
import { RefreshCw, AlertTriangle } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="bg-[#1e293b] border border-slate-600 rounded-lg p-3 text-xs shadow-xl max-w-[200px]">
      <p className="text-slate-400 font-bold mb-1.5">{label}</p>
      {payload.map((p) =>
        p.value !== null && p.value !== undefined ? (
          <div key={p.dataKey} className="flex items-center gap-2 py-0.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ background: p.color }} />
            <span className="text-slate-400 truncate">{p.name}:</span>
            <span className="text-white font-semibold ml-auto pl-2">
              {typeof p.value === 'number'
                ? p.value.toLocaleString(undefined, { maximumFractionDigits: 2 })
                : p.value}
            </span>
          </div>
        ) : null
      )}
    </div>
  );
};

// Wrapper that only renders children once it has real dimensions
function ChartContainer({ height = 280, children }) {
  const ref = useRef(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!ref.current) return;
    // Use ResizeObserver to wait for real width
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentRect.width > 10) {
          setReady(true);
          ro.disconnect();
          break;
        }
      }
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  return (
    <div ref={ref} style={{ height, width: '100%', minHeight: height }}>
      {ready ? children : null}
    </div>
  );
}

export default function StockChart({ ticker }) {
  const [mounted, setMounted] = useState(false);
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => { setMounted(true); }, []);

  useEffect(() => {
    if (!ticker) return;
    const fetchChart = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/api/analyze?ticker=${encodeURIComponent(ticker)}`);
        const json = await res.json();
        if (!res.ok) throw new Error(json.detail || 'Failed to fetch chart data');
        setData(json.chartData || []);
      } catch (err) {
        setError(err.message);
        setData([]);
      } finally {
        setLoading(false);
      }
    };
    fetchChart();
  }, [ticker]);

  if (!mounted) return null;

  if (loading) {
    return (
      <div className="h-96 w-full flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-800">
        <div className="flex items-center gap-2 text-slate-400">
          <RefreshCw className="h-5 w-5 animate-spin" />
          <span className="text-sm">Loading chart data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-32 w-full flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-800">
        <div className="flex items-center gap-2 text-rose-400 text-sm">
          <AlertTriangle className="h-5 w-5" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="h-32 w-full flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-800">
        <span className="text-slate-400 text-sm">No chart data available</span>
      </div>
    );
  }

  // Thin out x-axis labels — max 8 labels regardless of data length
  const tickInterval = Math.max(1, Math.floor(data.length / 8));
  const xTicks = data.filter((_, i) => i % tickInterval === 0).map(d => d.date);

  // Average volume for colour coding
  const avgVol = data.reduce((s, d) => s + d.volume, 0) / data.length;

  const axisProps = {
    stroke: '#475569',
    fontSize: 10,
    tickLine: false,
    tick: { fill: '#64748b' },
  };

  return (
    <div className="space-y-4 w-full">

      {/* Price + MAs + Bollinger Bands */}
      <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300 mb-3">
          Price History · Moving Averages · Bollinger Bands
        </h3>
        <ChartContainer height={288}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 5, right: 8, left: -15, bottom: 0 }}>
              <XAxis dataKey="date" {...axisProps} ticks={xTicks} />
              <YAxis
                domain={['auto', 'auto']}
                {...axisProps}
                tickFormatter={(v) => `Rs.${v.toLocaleString()}`}
                width={70}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend verticalAlign="top" height={32} iconType="circle" iconSize={7}
                wrapperStyle={{ fontSize: '10px', color: '#94a3b8' }} />
              <Line type="monotone" dataKey="upperBand" stroke="#38bdf8" strokeWidth={1} strokeDasharray="3 3" dot={false} name="BB Upper" />
              <Line type="monotone" dataKey="lowerBand" stroke="#38bdf8" strokeWidth={1} strokeDasharray="3 3" dot={false} name="BB Lower" />
              <Line type="monotone" dataKey="close"  stroke="#10b981" strokeWidth={2}   dot={false} name="Close" />
              <Line type="monotone" dataKey="ma20"   stroke="#3b82f6" strokeWidth={1.2} dot={false} name="MA 20" />
              <Line type="monotone" dataKey="ma50"   stroke="#f97316" strokeWidth={1.2} dot={false} name="MA 50" />
              <Line type="monotone" dataKey="ma100"  stroke="#8b5cf6" strokeWidth={1.2} dot={false} name="MA 100" />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>

      {/* Volume */}
      <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300 mb-1">
          Volume
          <span className="text-slate-500 font-normal ml-2 text-xs">green = above avg · red = below avg</span>
        </h3>
        <ChartContainer height={128}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 5, right: 8, left: -15, bottom: 0 }}>
              <XAxis dataKey="date" {...axisProps} ticks={xTicks} />
              <YAxis {...axisProps} tickFormatter={(v) => `${(v / 1e6).toFixed(0)}M`} width={45} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="volume" name="Volume" maxBarSize={6} radius={[2, 2, 0, 0]}>
                {data.map((entry, i) => (
                  <Cell key={`v-${i}`} fill={entry.volume >= avgVol ? '#10b981' : '#ef4444'} opacity={0.7} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="volumeMa20" stroke="#facc15" strokeWidth={1.5} dot={false} name="Vol MA20" />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>

      {/* RSI + MACD */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ minWidth: 0 }}>

        {/* RSI */}
        <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">RSI (14)</h3>
          <ChartContainer height={176}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 5, right: 8, left: -15, bottom: 0 }}>
                <XAxis dataKey="date" {...axisProps} ticks={xTicks} />
                <YAxis domain={[0, 100]} {...axisProps} width={30} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 3"
                  label={{ value: '70', fill: '#ef4444', fontSize: 9, position: 'insideTopRight' }} />
                <ReferenceLine y={30} stroke="#3b82f6" strokeDasharray="4 3"
                  label={{ value: '30', fill: '#3b82f6', fontSize: 9, position: 'insideBottomRight' }} />
                <ReferenceLine y={50} stroke="#475569" strokeDasharray="2 4" />
                <Line type="monotone" dataKey="rsi" stroke="#818cf8" strokeWidth={2} dot={false} name="RSI" />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>

        {/* MACD + Histogram */}
        <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">MACD · Signal · Histogram</h3>
          <ChartContainer height={176}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={data} margin={{ top: 5, right: 8, left: -15, bottom: 0 }}>
                <XAxis dataKey="date" {...axisProps} ticks={xTicks} />
                <YAxis {...axisProps} width={45} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={0} stroke="#475569" />
                <Bar dataKey="macdHist" name="Histogram" maxBarSize={4} radius={[2, 2, 0, 0]}>
                  {data.map((entry, i) => (
                    <Cell key={`h-${i}`} fill={entry.macdHist >= 0 ? '#10b981' : '#ef4444'} opacity={0.6} />
                  ))}
                </Bar>
                <Line type="monotone" dataKey="macd"       stroke="#f43f5e" strokeWidth={1.5} dot={false} name="MACD" />
                <Line type="monotone" dataKey="macdSignal" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="Signal" />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartContainer>
        </div>
      </div>

      {/* ADX */}
      <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800">
        <h3 className="text-sm font-semibold text-slate-300 mb-2">
          ADX — Trend Strength
          <span className="text-slate-500 font-normal ml-2 text-xs">
            &lt;15 no trend · 15-25 developing · &gt;25 strong
          </span>
        </h3>
        <ChartContainer height={128}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 5, right: 8, left: -15, bottom: 0 }}>
              <XAxis dataKey="date" {...axisProps} ticks={xTicks} />
              <YAxis domain={[0, 60]} {...axisProps} width={30} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={25} stroke="#f97316" strokeDasharray="4 3"
                label={{ value: '25', fill: '#f97316', fontSize: 9, position: 'insideTopRight' }} />
              <ReferenceLine y={15} stroke="#64748b" strokeDasharray="4 3"
                label={{ value: '15', fill: '#94a3b8', fontSize: 9, position: 'insideTopRight' }} />
              <Line type="monotone" dataKey="adx" stroke="#a78bfa" strokeWidth={2} dot={false} name="ADX" />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartContainer>
      </div>

    </div>
  );
}
