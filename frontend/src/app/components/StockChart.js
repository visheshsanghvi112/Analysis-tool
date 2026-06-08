'use client';

import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine
} from 'recharts';

export default function StockChart({ data }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted || !data || data.length === 0) {
    return (
      <div className="h-96 w-full flex items-center justify-center bg-slate-900/40 rounded-xl border border-slate-800">
        <span className="text-slate-400 text-sm">No chart data available</span>
      </div>
    );
  }

  // Format data for chart display
  const chartData = data.map((d) => ({
    ...d,
    dateStr: d.date,
    Price: d.close,
    MA20: d.ma20,
    MA50: d.ma50,
    MA100: d.ma100,
    VWAP: d.vwap,
    RSI: d.rsi,
    MACD: d.macd,
    Signal: d.macdSignal,
    UpperBand: d.upperBand,
    LowerBand: d.lowerBand,
  }));

  return (
    <div className="space-y-6 w-full">
      {/* Primary Price Chart */}
      <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800 backdrop-blur-md">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Price History, MAs & Bollinger Bands</h3>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <XAxis dataKey="dateStr" stroke="#64748b" fontSize={11} tickLine={false} />
              <YAxis domain={['auto', 'auto']} stroke="#64748b" fontSize={11} tickLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                labelClassName="text-slate-400 font-bold text-xs"
                itemStyle={{ color: '#fff', fontSize: '12px' }}
              />
              <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: '#cbd5e1' }} />
              
              {/* Bollinger Bands Shading */}
              <Line type="monotone" dataKey="UpperBand" stroke="#38bdf8" strokeWidth={1} strokeDasharray="3 3" dot={false} name="BB Upper" />
              <Line type="monotone" dataKey="LowerBand" stroke="#38bdf8" strokeWidth={1} strokeDasharray="3 3" dot={false} name="BB Lower" />

              {/* Price & Moving Averages */}
              <Line type="monotone" dataKey="Price" stroke="#10b981" strokeWidth={2.5} dot={false} name="Close Price" />
              <Line type="monotone" dataKey="VWAP" stroke="#eab308" strokeWidth={1.5} dot={false} name="VWAP" />
              <Line type="monotone" dataKey="MA20" stroke="#3b82f6" strokeWidth={1.2} dot={false} name="20 MA" />
              <Line type="monotone" dataKey="MA50" stroke="#f97316" strokeWidth={1.2} dot={false} name="50 MA" />
              <Line type="monotone" dataKey="MA100" stroke="#8b5cf6" strokeWidth={1.2} dot={false} name="100 MA" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sub-Charts (RSI & MACD) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* RSI Indicator Chart */}
        <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800 backdrop-blur-md">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">RSI (Relative Strength Index)</h3>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="dateStr" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis domain={[0, 100]} stroke="#64748b" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff', fontSize: '11px' }}
                />
                <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Overbought', fill: '#ef4444', fontSize: 9, position: 'insideTopRight' }} />
                <ReferenceLine y={30} stroke="#3b82f6" strokeDasharray="3 3" label={{ value: 'Oversold', fill: '#3b82f6', fontSize: 9, position: 'insideBottomRight' }} />
                <Line type="monotone" dataKey="RSI" stroke="#818cf8" strokeWidth={1.8} dot={false} name="RSI" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* MACD Chart */}
        <div className="bg-[#111827]/80 rounded-xl p-4 border border-slate-800 backdrop-blur-md">
          <h3 className="text-sm font-semibold text-slate-300 mb-2">MACD & Signal</h3>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="dateStr" stroke="#64748b" fontSize={10} tickLine={false} />
                <YAxis stroke="#64748b" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff', fontSize: '11px' }}
                />
                <Legend verticalAlign="top" iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                <ReferenceLine y={0} stroke="#475569" />
                <Line type="monotone" dataKey="MACD" stroke="#f43f5e" strokeWidth={1.5} dot={false} name="MACD" />
                <Line type="monotone" dataKey="Signal" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="Signal Line" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
