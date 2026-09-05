'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Link from 'next/link';
import {
  Activity, ArrowUpRight, ArrowDownRight, RefreshCw, Layers,
  Compass, Calculator, ShieldAlert, Sparkles, Sliders, ChevronDown,
  Search, TrendingUp, TrendingDown, Target, Zap, Clock, ShieldCheck,
  BarChart2, Flame, Eye, ArrowRight, CheckCircle2, XCircle, AlertCircle
} from 'lucide-react';
import InfoBadge from './InfoBadge';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (
  typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : 'https://stock-analysis-backend-seven.vercel.app'
);

const QUICK_TICKERS = [
  { symbol: 'RELIANCE.NS', name: 'Reliance Ind.', market: 'IN' },
  { symbol: 'TCS.NS',      name: 'TCS',           market: 'IN' },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank',     market: 'IN' },
  { symbol: 'INFY.NS',     name: 'Infosys',       market: 'IN' },
  { symbol: 'TATAMOTORS.NS', name: 'Tata Motors', market: 'IN' },
  { symbol: 'NVDA',        name: 'Nvidia Corp',   market: 'US' },
  { symbol: 'AAPL',        name: 'Apple Inc',     market: 'US' },
  { symbol: 'TSLA',        name: 'Tesla Inc',     market: 'US' },
  { symbol: 'MSFT',        name: 'Microsoft',     market: 'US' },
];

const TIMEFRAMES = [
  { label: '1m', interval: '1m', period: '1d' },
  { label: '3m', interval: '3m', period: '1d' },
  { label: '5m', interval: '5m', period: '1d' },
  { label: '15m', interval: '15m', period: '1d' },
  { label: '30m', interval: '30m', period: '1d' },
  { label: '1h', interval: '1h', period: '5d' },
];

export default function IntradayTerminal() {
  const [ticker, setTicker] = useState('RELIANCE.NS');
  const [searchInput, setSearchInput] = useState('');
  const [interval, setInterval] = useState('5m');
  const [period, setPeriod] = useState('1d');

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Auto-refresh state
  const [autoRefreshSecs, setAutoRefreshSecs] = useState(30);
  const [refreshCountdown, setRefreshCountdown] = useState(30);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Chart Overlays
  const [showVWAP, setShowVWAP] = useState(true);
  const [showVWAPBands, setShowVWAPBands] = useState(true);
  const [showSupertrend, setShowSupertrend] = useState(true);
  const [showEMA, setShowEMA] = useState(true);
  const [showORB, setShowORB] = useState(true);
  const [showCamarilla, setShowCamarilla] = useState(false);

  // Sub-chart selector
  const [activeSubChart, setActiveSubChart] = useState('volume'); // 'volume' | 'rsi' | 'cvd'

  // Hovered candle for inspection
  const [hoveredCandle, setHoveredCandle] = useState(null);

  // Position Sizing Calculator state
  const [calcCapital, setCalcCapital] = useState(100000);
  const [calcRiskPct, setCalcRiskPct] = useState(1.0);
  const [calcLeverage, setCalcLeverage] = useState(5); // MIS 5x
  const [calcEntry, setCalcEntry] = useState('');
  const [calcStop, setCalcStop] = useState('');

  // Scanner state
  const [scannerMarket, setScannerMarket] = useState('IN');
  const [scannerData, setScannerData] = useState([]);
  const [scannerLoading, setScannerLoading] = useState(false);

  const isUS = ticker && !ticker.endsWith('.NS') && !ticker.endsWith('.BO');
  const currSym = data?.currency_symbol || (isUS ? '$' : '₹');

  // Fetch Main Intraday Data
  const fetchData = useCallback(async (isSilent = false) => {
    if (!isSilent) setLoading(true);
    setIsRefreshing(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/intraday/analysis?ticker=${encodeURIComponent(ticker)}&interval=${interval}&period=${period}`);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned status ${res.status}`);
      }
      const json = await res.json();
      setData(json);
      // Pre-fill calculator entry if empty or changed
      if (json.current_price) {
        setCalcEntry(json.current_price.toString());
        // Default stop loss at supertrend or 1% below
        const defaultStop = json.supertrend && json.supertrend !== json.current_price
          ? json.supertrend
          : Math.round(json.current_price * 0.99 * 100) / 100;
        setCalcStop(defaultStop.toString());
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
      setRefreshCountdown(autoRefreshSecs);
    }
  }, [ticker, interval, period, autoRefreshSecs]);

  // Initial and param-change load
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh countdown timer
  useEffect(() => {
    if (autoRefreshSecs <= 0) return;
    const timer = setInterval(() => {
      setRefreshCountdown(prev => {
        if (prev <= 1) {
          fetchData(true);
          return autoRefreshSecs;
        }
        return prev - 1;
      });
    }, 1000);
    return () => clearInterval(timer);
  }, [autoRefreshSecs, fetchData]);

  // Fetch Scanner Data
  const fetchScanner = useCallback(async () => {
    setScannerLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/intraday/scanner?market=${scannerMarket}`);
      if (res.ok) {
        const json = await res.json();
        setScannerData(json.results || []);
      }
    } catch (_) {}
    finally {
      setScannerLoading(false);
    }
  }, [scannerMarket]);

  useEffect(() => {
    fetchScanner();
  }, [fetchScanner]);

  // Position Sizing calculations
  const sizingResults = useMemo(() => {
    const entry = parseFloat(calcEntry) || 0;
    const stop = parseFloat(calcStop) || 0;
    const capital = parseFloat(calcCapital) || 0;
    const riskPct = parseFloat(calcRiskPct) || 1.0;
    const leverage = parseFloat(calcLeverage) || 1;

    if (entry <= 0 || stop <= 0 || capital <= 0 || entry === stop) {
      return null;
    }

    const isLong = entry > stop;
    const riskPerShare = Math.abs(entry - stop);
    const maxRiskAmount = (capital * riskPct) / 100.0;
    const sharesByRisk = Math.floor(maxRiskAmount / riskPerShare);

    // Max shares allowed by leverage capital constraint
    const maxLeveragedCapital = capital * leverage;
    const sharesByCapital = Math.floor(maxLeveragedCapital / entry);

    const exactShares = Math.min(sharesByRisk, sharesByCapital);
    const effectiveExposure = exactShares * entry;
    const marginRequired = effectiveExposure / leverage;
    const actualRiskAmount = exactShares * riskPerShare;

    // R:R Targets
    const t1Dist = riskPerShare * 1.5;
    const t2Dist = riskPerShare * 2.5;
    const t3Dist = riskPerShare * 3.5;

    const target1 = isLong ? entry + t1Dist : entry - t1Dist;
    const target2 = isLong ? entry + t2Dist : entry - t2Dist;
    const target3 = isLong ? entry + t3Dist : entry - t3Dist;

    return {
      isLong,
      exactShares,
      marginRequired: Math.round(marginRequired),
      effectiveExposure: Math.round(effectiveExposure),
      actualRiskAmount: Math.round(actualRiskAmount),
      riskRewardTargets: [
        { label: 'T1 (1:1.5 R:R)', price: Math.round(target1 * 100) / 100, profit: Math.round(exactShares * t1Dist) },
        { label: 'T2 (1:2.5 R:R)', price: Math.round(target2 * 100) / 100, profit: Math.round(exactShares * t2Dist) },
        { label: 'T3 (1:3.5 R:R)', price: Math.round(target3 * 100) / 100, profit: Math.round(exactShares * t3Dist) },
      ]
    };
  }, [calcEntry, calcStop, calcCapital, calcRiskPct, calcLeverage]);

  // Handle manual search
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    if (searchInput.trim()) {
      let sym = searchInput.trim().toUpperCase();
      if (!sym.includes('.') && scannerMarket === 'IN') {
        sym = `${sym}.NS`;
      }
      setTicker(sym);
      setSearchInput('');
    }
  };

  // SVG Candlestick Chart calculations
  const candles = data?.candles || [];
  const chartHeight = 360;
  const chartWidth = 720;
  const padding = { top: 20, right: 65, bottom: 30, left: 10 };

  const { priceMin, priceMax, xScale, yScale, candleWidth } = useMemo(() => {
    if (!candles.length) {
      return { priceMin: 0, priceMax: 1, xScale: () => 0, yScale: () => 0, candleWidth: 5 };
    }
    let min = Infinity;
    let max = -Infinity;

    candles.forEach(c => {
      if (c.low < min) min = c.low;
      if (c.high > max) max = c.high;
      if (showVWAPBands) {
        if (c.lower_band_2 < min && c.lower_band_2 > 0) min = c.lower_band_2;
        if (c.upper_band_2 > max) max = c.upper_band_2;
      }
      if (showSupertrend && c.supertrend > 0) {
        if (c.supertrend < min) min = c.supertrend;
        if (c.supertrend > max) max = c.supertrend;
      }
    });

    const buffer = (max - min) * 0.05 || 1;
    min -= buffer;
    max += buffer;

    const innerW = chartWidth - padding.left - padding.right;
    const innerH = chartHeight - padding.top - padding.bottom;

    const xs = (idx) => padding.left + (idx / Math.max(candles.length - 1, 1)) * innerW;
    const ys = (val) => padding.top + innerH - ((val - min) / (max - min)) * innerH;
    const cw = Math.max(2, Math.min(14, (innerW / candles.length) * 0.7));

    return { priceMin: min, priceMax: max, xScale: xs, yScale: ys, candleWidth: cw };
  }, [candles, showVWAPBands, showSupertrend]);

  return (
    <div className="w-full min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-6 lg:p-8 font-sans">
      {/* ── TOP TERMINAL BAR ────────────────────────────────────────────── */}
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-tr from-emerald-500/20 to-cyan-500/20 border border-emerald-500/40 rounded-xl">
                <Activity className="w-6 h-6 text-emerald-400 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-100 to-slate-400">
                    Intraday Quantitative Desk
                  </h1>
                  <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 rounded-full">
                    High-Frequency
                  </span>
                </div>
                <p className="text-xs sm:text-sm text-slate-400 mt-0.5">
                  Real-time VWAP bands, Volume Profile (VPVR), Camarilla Pivots, ORB & Order Flow Delta
                </p>
              </div>
            </div>
          </div>

          {/* Quick controls: Market toggle, Auto-refresh & Search */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Market identifier pill */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs font-semibold">
              <button
                onClick={() => { setScannerMarket('IN'); setTicker('RELIANCE.NS'); }}
                className={`px-2.5 py-1 rounded-md transition ${scannerMarket === 'IN' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-slate-400 hover:text-white'}`}
              >
                🇮🇳 NSE / BSE
              </button>
              <button
                onClick={() => { setScannerMarket('US'); setTicker('NVDA'); }}
                className={`px-2.5 py-1 rounded-md transition ${scannerMarket === 'US' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' : 'text-slate-400 hover:text-white'}`}
              >
                🇺🇸 NYSE / NASDAQ
              </button>
            </div>

            {/* Auto-refresh control */}
            <div className="flex items-center gap-2 bg-slate-900/90 border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
              <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${isRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
              <span className="text-slate-400">Refresh:</span>
              <select
                value={autoRefreshSecs}
                onChange={(e) => setAutoRefreshSecs(Number(e.target.value))}
                className="bg-transparent text-white font-mono text-xs focus:outline-none cursor-pointer"
              >
                <option value={15} className="bg-slate-900">15s</option>
                <option value={30} className="bg-slate-900">30s</option>
                <option value={60} className="bg-slate-900">60s</option>
                <option value={0} className="bg-slate-900">Paused</option>
              </select>
              {autoRefreshSecs > 0 && (
                <span className="text-[10px] font-mono text-cyan-400 w-4 text-right">
                  {refreshCountdown}s
                </span>
              )}
            </div>

            {/* Manual Refresh button */}
            <button
              onClick={() => fetchData(false)}
              disabled={loading}
              className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-300 transition"
              title="Force Refresh Data"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
            </button>
          </div>
        </header>

        {/* ── TICKER COMMAND BAR & POPULAR SHORTCUTS ───────────────────────── */}
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 bg-slate-900/60 border border-slate-800/80 rounded-2xl p-3 backdrop-blur-md">
          {/* Quick ticker pills */}
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 lg:pb-0 scrollbar-none">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider pl-1 pr-1 shrink-0">
              Quick Switch:
            </span>
            {QUICK_TICKERS.filter(t => t.market === scannerMarket).map(t => (
              <button
                key={t.symbol}
                onClick={() => setTicker(t.symbol)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold shrink-0 transition flex items-center gap-1.5 ${
                  ticker === t.symbol
                    ? 'bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/10'
                    : 'bg-slate-800/60 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700/40'
                }`}
              >
                <span>{t.name}</span>
                <span className="text-[10px] text-slate-500 font-mono">({t.symbol.split('.')[0]})</span>
              </button>
            ))}
          </div>

          {/* Search box */}
          <form onSubmit={handleSearchSubmit} className="relative min-w-[240px]">
            <input
              type="text"
              placeholder={`Search ${scannerMarket === 'IN' ? 'NSE stock (e.g. SBIN)' : 'US stock (e.g. AMD)'}...`}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700/80 rounded-xl pl-9 pr-8 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition"
            />
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
            {searchInput && (
              <button
                type="submit"
                className="absolute right-2 top-1.5 px-2 py-0.5 text-[10px] font-bold bg-cyan-500/20 text-cyan-300 rounded hover:bg-cyan-500/30 transition"
              >
                Load
              </button>
            )}
          </form>
        </div>

        {/* ── ACTIVE TICKER HEADLINE BAR ──────────────────────────────────── */}
        {data && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {/* Price & Change */}
            <div className="col-span-2 bg-gradient-to-br from-slate-900/90 to-slate-900/50 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-xs font-semibold text-slate-400">{data.company_name}</span>
                  <div className="flex items-baseline gap-2">
                    <h2 className="text-2xl sm:text-3xl font-black font-mono tracking-tight text-white">
                      {currSym}{data.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </h2>
                    <span className={`text-xs sm:text-sm font-bold font-mono flex items-center ${data.change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {data.change >= 0 ? <ArrowUpRight className="w-4 h-4 mr-0.5" /> : <ArrowDownRight className="w-4 h-4 mr-0.5" />}
                      {data.change >= 0 ? '+' : ''}{data.change} ({data.change >= 0 ? '+' : ''}{data.change_pct}%)
                    </span>
                  </div>
                </div>
                <InfoBadge infoKey="live_prices" />
              </div>
              <div className="flex items-center gap-3 text-[11px] text-slate-400 mt-2 font-mono">
                <span>Open: <strong className="text-slate-200">{currSym}{data.open}</strong></span>
                <span>High: <strong className="text-emerald-400">{currSym}{data.high}</strong></span>
                <span>Low: <strong className="text-rose-400">{currSym}{data.low}</strong></span>
              </div>
            </div>

            {/* VWAP Benchmark */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  Session VWAP
                  <InfoBadge infoKey="vwap" />
                </span>
                <span className="text-[10px] font-mono text-cyan-400 font-bold px-1.5 py-0.5 bg-cyan-500/10 rounded">
                  {data.current_price > data.vwap ? 'ABOVE' : 'BELOW'}
                </span>
              </div>
              <div className="mt-1">
                <p className="text-xl font-bold font-mono text-cyan-300">
                  {currSym}{data.vwap?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
                <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
                  Diff: <span className={data.current_price >= data.vwap ? 'text-emerald-400' : 'text-rose-400'}>
                    {data.current_price >= data.vwap ? '+' : ''}
                    {(((data.current_price - data.vwap) / data.vwap) * 100).toFixed(2)}%
                  </span>
                </p>
              </div>
            </div>

            {/* Supertrend Level */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  Supertrend (10, 3)
                  <InfoBadge infoKey="supertrend" />
                </span>
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${data.supertrend_dir === 1 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                  {data.supertrend_dir === 1 ? 'LONG' : 'SHORT'}
                </span>
              </div>
              <div className="mt-1">
                <p className={`text-xl font-bold font-mono ${data.supertrend_dir === 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {currSym}{data.supertrend?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
                <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
                  Stop: <span className="text-slate-300">{currSym}{data.supertrend}</span>
                </p>
              </div>
            </div>

            {/* ORB 15m State */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  ORB (15 Min)
                  <InfoBadge infoKey="orb_strategy" />
                </span>
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                  data.orb.status === 'BULLISH_BREAKOUT'
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : data.orb.status === 'BEARISH_BREAKDOWN'
                    ? 'bg-rose-500/10 text-rose-400'
                    : 'bg-amber-500/10 text-amber-400'
                }`}>
                  {data.orb.status === 'BULLISH_BREAKOUT' ? 'BREAKOUT' : data.orb.status === 'BEARISH_BREAKDOWN' ? 'BREAKDOWN' : 'RANGE'}
                </span>
              </div>
              <div className="mt-1">
                <p className="text-xs font-mono text-slate-300">
                  H: <strong className="text-emerald-400">{currSym}{data.orb.high_15m}</strong>
                </p>
                <p className="text-xs font-mono text-slate-300 mt-0.5">
                  L: <strong className="text-rose-400">{currSym}{data.orb.low_15m}</strong>
                </p>
              </div>
            </div>

            {/* Composite Bias Score */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-900/50 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  Quant Bias
                  <InfoBadge infoKey="intraday_quant_score" />
                </span>
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                  data.signals.overall_bias.includes('BUY')
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : data.signals.overall_bias.includes('SELL')
                    ? 'bg-rose-500/10 text-rose-400'
                    : 'bg-amber-500/10 text-amber-400'
                }`}>
                  {data.signals.overall_bias}
                </span>
              </div>
              <div className="mt-1">
                <div className="flex items-baseline gap-1">
                  <span className={`text-xl font-black font-mono ${data.signals.quant_score >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {data.signals.quant_score >= 0 ? '+' : ''}{data.signals.quant_score}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">/ 100</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1.5">
                  <div
                    className={`h-full transition-all duration-500 ${data.signals.quant_score >= 0 ? 'bg-emerald-400' : 'bg-rose-400'}`}
                    style={{ width: `${Math.abs(data.signals.quant_score)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── MAIN CHART & SIDEBAR SECTION ─────────────────────────────────── */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Main Chart Column (3 spans) */}
          <div className="xl:col-span-3 space-y-4">
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-4 sm:p-6 backdrop-blur-md">
              {/* Chart Toolbar: Timeframe & Overlays */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
                {/* Timeframes */}
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
                  {TIMEFRAMES.map((tf) => (
                    <button
                      key={tf.label}
                      onClick={() => { setInterval(tf.interval); setPeriod(tf.period); }}
                      className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition ${
                        interval === tf.interval
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {tf.label}
                    </button>
                  ))}
                </div>

                {/* Overlay Toggles */}
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  <button
                    onClick={() => setShowVWAP(!showVWAP)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showVWAP ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-cyan-400 rounded-full" />
                    VWAP
                  </button>

                  <button
                    onClick={() => setShowVWAPBands(!showVWAPBands)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showVWAPBands ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-cyan-300/40 rounded-full" />
                    ±2σ Bands
                  </button>

                  <button
                    onClick={() => setShowSupertrend(!showSupertrend)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showSupertrend ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-emerald-400 rounded-full" />
                    Supertrend
                  </button>

                  <button
                    onClick={() => setShowEMA(!showEMA)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showEMA ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-purple-400 rounded-full" />
                    EMA 9/21
                  </button>

                  <button
                    onClick={() => setShowORB(!showORB)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showORB ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-amber-400 rounded-full" />
                    ORB 15m
                  </button>

                  <button
                    onClick={() => setShowCamarilla(!showCamarilla)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showCamarilla ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-rose-400 rounded-full" />
                    Camarilla
                  </button>
                </div>
              </div>

              {/* Hover Inspection Bar */}
              <div className="h-6 flex items-center justify-between text-[11px] font-mono text-slate-400 mt-2 px-1">
                {hoveredCandle ? (
                  <div className="flex flex-wrap items-center gap-3">
                    <span>Time: <strong className="text-white">{hoveredCandle.time}</strong></span>
                    <span>O: <strong className="text-slate-200">{hoveredCandle.open}</strong></span>
                    <span>H: <strong className="text-emerald-400">{hoveredCandle.high}</strong></span>
                    <span>L: <strong className="text-rose-400">{hoveredCandle.low}</strong></span>
                    <span>C: <strong className={hoveredCandle.close >= hoveredCandle.open ? 'text-emerald-400' : 'text-rose-400'}>{hoveredCandle.close}</strong></span>
                    <span>Vol: <strong className="text-cyan-300">{hoveredCandle.volume?.toLocaleString()}</strong></span>
                    <span>VWAP: <strong className="text-cyan-400">{hoveredCandle.vwap}</strong></span>
                  </div>
                ) : (
                  <span className="text-slate-500 italic">Hover over candles to inspect high-frequency price & indicator metrics</span>
                )}
              </div>

              {/* High-Resolution SVG Candlestick Rendering */}
              <div className="relative w-full overflow-hidden bg-slate-950/60 rounded-2xl border border-slate-800/60 mt-1">
                {loading && (
                  <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center z-20">
                    <div className="flex items-center gap-2 text-cyan-400 text-sm font-semibold">
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      Loading High-Frequency Feed...
                    </div>
                  </div>
                )}

                {error && (
                  <div className="h-72 flex items-center justify-center p-6 text-rose-400 text-sm">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    {error}
                  </div>
                )}

                {!loading && !error && candles.length > 0 && (
                  <svg
                    viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                    className="w-full h-auto cursor-crosshair select-none"
                    onMouseLeave={() => setHoveredCandle(null)}
                  >
                    <defs>
                      <linearGradient id="vwapBandFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.08" />
                        <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.02" />
                      </linearGradient>
                    </defs>

                    {/* Horizontal Price Grid Lines */}
                    {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => {
                      const p = priceMin + (priceMax - priceMin) * (1 - pct);
                      const y = yScale(p);
                      return (
                        <g key={i}>
                          <line
                            x1={padding.left}
                            y1={y}
                            x2={chartWidth - padding.right}
                            y2={y}
                            stroke="#334155"
                            strokeDasharray="3 3"
                            strokeOpacity={0.4}
                          />
                          <text
                            x={chartWidth - padding.right + 6}
                            y={y + 3}
                            fill="#64748b"
                            fontSize="9"
                            fontFamily="monospace"
                          >
                            {p.toFixed(2)}
                          </text>
                        </g>
                      );
                    })}

                    {/* ORB Range Box Overlay */}
                    {showORB && data?.orb && (
                      <g>
                        <line
                          x1={padding.left}
                          y1={yScale(data.orb.high_15m)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.orb.high_15m)}
                          stroke="#f59e0b"
                          strokeDasharray="4 4"
                          strokeWidth="1.2"
                          strokeOpacity={0.8}
                        />
                        <text
                          x={padding.left + 6}
                          y={yScale(data.orb.high_15m) - 4}
                          fill="#f59e0b"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          ORB 15m HIGH ({data.orb.high_15m})
                        </text>

                        <line
                          x1={padding.left}
                          y1={yScale(data.orb.low_15m)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.orb.low_15m)}
                          stroke="#f59e0b"
                          strokeDasharray="4 4"
                          strokeWidth="1.2"
                          strokeOpacity={0.8}
                        />
                        <text
                          x={padding.left + 6}
                          y={yScale(data.orb.low_15m) + 10}
                          fill="#f59e0b"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          ORB 15m LOW ({data.orb.low_15m})
                        </text>
                      </g>
                    )}

                    {/* Camarilla Inflection Levels */}
                    {showCamarilla && data?.pivots?.camarilla && (
                      <g>
                        {data.pivots.camarilla.h4 && (
                          <line
                            x1={padding.left}
                            y1={yScale(data.pivots.camarilla.h4)}
                            x2={chartWidth - padding.right}
                            y2={yScale(data.pivots.camarilla.h4)}
                            stroke="#10b981"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                          />
                        )}
                        {data.pivots.camarilla.h3 && (
                          <line
                            x1={padding.left}
                            y1={yScale(data.pivots.camarilla.h3)}
                            x2={chartWidth - padding.right}
                            y2={yScale(data.pivots.camarilla.h3)}
                            stroke="#f43f5e"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                          />
                        )}
                        {data.pivots.camarilla.l3 && (
                          <line
                            x1={padding.left}
                            y1={yScale(data.pivots.camarilla.l3)}
                            x2={chartWidth - padding.right}
                            y2={yScale(data.pivots.camarilla.l3)}
                            stroke="#10b981"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                          />
                        )}
                        {data.pivots.camarilla.l4 && (
                          <line
                            x1={padding.left}
                            y1={yScale(data.pivots.camarilla.l4)}
                            x2={chartWidth - padding.right}
                            y2={yScale(data.pivots.camarilla.l4)}
                            stroke="#f43f5e"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                          />
                        )}
                      </g>
                    )}

                    {/* VWAP ±2σ Bands Envelope Path */}
                    {showVWAPBands && (
                      <g>
                        <path
                          d={candles.reduce((acc, c, i) => {
                            const x = xScale(i);
                            const y = yScale(c.upper_band_2);
                            return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                          }, '')}
                          fill="none"
                          stroke="#06b6d4"
                          strokeOpacity={0.35}
                          strokeWidth="1"
                          strokeDasharray="2 2"
                        />
                        <path
                          d={candles.reduce((acc, c, i) => {
                            const x = xScale(i);
                            const y = yScale(c.lower_band_2);
                            return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                          }, '')}
                          fill="none"
                          stroke="#06b6d4"
                          strokeOpacity={0.35}
                          strokeWidth="1"
                          strokeDasharray="2 2"
                        />
                      </g>
                    )}

                    {/* VWAP Main Benchmark Line */}
                    {showVWAP && (
                      <path
                        d={candles.reduce((acc, c, i) => {
                          const x = xScale(i);
                          const y = yScale(c.vwap);
                          return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }, '')}
                        fill="none"
                        stroke="#06b6d4"
                        strokeWidth="1.8"
                      />
                    )}

                    {/* EMA 9 and EMA 21 Ribbon Lines */}
                    {showEMA && (
                      <g>
                        <path
                          d={candles.reduce((acc, c, i) => {
                            const x = xScale(i);
                            const y = yScale(c.ema9);
                            return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                          }, '')}
                          fill="none"
                          stroke="#a855f7"
                          strokeWidth="1.2"
                        />
                        <path
                          d={candles.reduce((acc, c, i) => {
                            const x = xScale(i);
                            const y = yScale(c.ema21);
                            return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                          }, '')}
                          fill="none"
                          stroke="#ec4899"
                          strokeWidth="1.2"
                          strokeOpacity={0.8}
                        />
                      </g>
                    )}

                    {/* Supertrend Stop Line */}
                    {showSupertrend && (
                      <g>
                        {candles.map((c, i) => {
                          if (i === 0) return null;
                          const x1 = xScale(i - 1);
                          const y1 = yScale(candles[i - 1].supertrend);
                          const x2 = xScale(i);
                          const y2 = yScale(c.supertrend);
                          const color = c.supertrend_dir === 1 ? '#10b981' : '#f43f5e';
                          return (
                            <line
                              key={`st-${i}`}
                              x1={x1}
                              y1={y1}
                              x2={x2}
                              y2={y2}
                              stroke={color}
                              strokeWidth="2"
                            />
                          );
                        })}
                      </g>
                    )}

                    {/* Candlesticks & Wicks */}
                    {candles.map((c, i) => {
                      const x = xScale(i);
                      const isUp = c.close >= c.open;
                      const candleColor = isUp ? '#10b981' : '#f43f5e';
                      const yOpen = yScale(c.open);
                      const yClose = yScale(c.close);
                      const yHigh = yScale(c.high);
                      const yLow = yScale(c.low);
                      const bodyY = Math.min(yOpen, yClose);
                      const bodyHeight = Math.max(Math.abs(yClose - yOpen), 1.5);

                      return (
                        <g
                          key={i}
                          onMouseEnter={() => setHoveredCandle(c)}
                          className="cursor-pointer"
                        >
                          {/* Upper & Lower Wick */}
                          <line
                            x1={x}
                            y1={yHigh}
                            x2={x}
                            y2={yLow}
                            stroke={candleColor}
                            strokeWidth="1"
                          />
                          {/* Candle Body */}
                          <rect
                            x={x - candleWidth / 2}
                            y={bodyY}
                            width={candleWidth}
                            height={bodyHeight}
                            fill={candleColor}
                            rx={1}
                          />
                        </g>
                      );
                    })}

                    {/* X-axis Time Labels */}
                    {candles.map((c, i) => {
                      if (i % Math.ceil(candles.length / 6) !== 0) return null;
                      const x = xScale(i);
                      return (
                        <text
                          key={`x-${i}`}
                          x={x}
                          y={chartHeight - 8}
                          fill="#64748b"
                          fontSize="9"
                          fontFamily="monospace"
                          textAnchor="middle"
                        >
                          {c.time}
                        </text>
                      );
                    })}
                  </svg>
                )}
              </div>

              {/* Sub-Chart Selector & Sub-Chart Panel */}
              <div className="mt-4 pt-4 border-t border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400">Sub-Indicator:</span>
                    <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs">
                      <button
                        onClick={() => setActiveSubChart('volume')}
                        className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'volume' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}
                      >
                        Volume & Delta
                      </button>
                      <button
                        onClick={() => setActiveSubChart('rsi')}
                        className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'rsi' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}
                      >
                        RSI (14)
                      </button>
                      <button
                        onClick={() => setActiveSubChart('cvd')}
                        className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'cvd' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}
                      >
                        Order Flow CVD
                      </button>
                    </div>
                  </div>

                  {activeSubChart === 'volume' && (
                    <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-emerald-400" /> Buyer Volume
                      </span>
                      <span className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-rose-400" /> Seller Volume
                      </span>
                    </div>
                  )}

                  {activeSubChart === 'rsi' && (
                    <div className="flex items-center gap-2 text-[11px] font-mono">
                      <span className="text-slate-400">Overbought: 70</span>
                      <span className="text-slate-400">Oversold: 30</span>
                    </div>
                  )}
                </div>

                {/* Sub-Chart SVG Container */}
                <div className="h-28 w-full bg-slate-950/60 rounded-xl border border-slate-800/60 p-2 overflow-hidden">
                  {activeSubChart === 'volume' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const maxVol = Math.max(...candles.map(c => c.volume || 1), 1);
                        return candles.map((c, i) => {
                          const x = xScale(i);
                          const bH = ((c.buyer_vol || 0) / maxVol) * 85;
                          const sH = ((c.seller_vol || 0) / maxVol) * 85;
                          return (
                            <g key={i}>
                              <rect
                                x={x - candleWidth / 2}
                                y={95 - bH}
                                width={candleWidth / 2}
                                height={bH}
                                fill="#10b981"
                                fillOpacity={0.8}
                              />
                              <rect
                                x={x}
                                y={95 - sH}
                                width={candleWidth / 2}
                                height={sH}
                                fill="#f43f5e"
                                fillOpacity={0.8}
                              />
                            </g>
                          );
                        });
                      })()}
                    </svg>
                  )}

                  {activeSubChart === 'rsi' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      <line x1={padding.left} y1={30} x2={chartWidth - padding.right} y2={30} stroke="#f43f5e" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <line x1={padding.left} y1={70} x2={chartWidth - padding.right} y2={70} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <path
                        d={candles.reduce((acc, c, i) => {
                          const x = xScale(i);
                          const y = 100 - (c.rsi || 50);
                          return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }, '')}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="1.5"
                      />
                    </svg>
                  )}

                  {activeSubChart === 'cvd' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const cvdVals = candles.map(c => c.cum_delta || 0);
                        const minCvd = Math.min(...cvdVals, 0);
                        const maxCvd = Math.max(...cvdVals, 1);
                        const cvdRange = (maxCvd - minCvd) || 1;
                        return (
                          <path
                            d={candles.reduce((acc, c, i) => {
                              const x = xScale(i);
                              const y = 90 - (((c.cum_delta || 0) - minCvd) / cvdRange) * 80;
                              return `${acc} ${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                            }, '')}
                            fill="none"
                            stroke="#eab308"
                            strokeWidth="1.8"
                          />
                        );
                      })()}
                    </svg>
                  )}
                </div>
              </div>
            </div>

            {/* ── TACTICAL SIGNALS CHECKLIST & ORDER FLOW GAUGE ──────────────── */}
            {data && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Quant Signals Matrix */}
                <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 sm:p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Target className="w-4 h-4 text-emerald-400" />
                      Algorithmic Execution Checklist
                    </h3>
                    <InfoBadge infoKey="intraday_quant_score" />
                  </div>
                  <div className="space-y-2">
                    {data.signals.checklist.map((item, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 rounded-xl bg-slate-950/60 border border-slate-800/60 text-xs"
                      >
                        <div className="flex items-center gap-2">
                          {item.status === 'BULLISH' ? (
                            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                          ) : item.status === 'BEARISH' ? (
                            <XCircle className="w-4 h-4 text-rose-400 shrink-0" />
                          ) : (
                            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
                          )}
                          <div>
                            <span className="font-semibold text-slate-200">{item.factor}</span>
                            <p className="text-[10px] text-slate-400">{item.desc}</p>
                          </div>
                        </div>
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono ${
                          item.status === 'BULLISH' ? 'bg-emerald-500/10 text-emerald-400' :
                          item.status === 'BEARISH' ? 'bg-rose-500/10 text-rose-400' :
                          'bg-amber-500/10 text-amber-400'
                        }`}>
                          {item.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Order Flow & Pressure Gauge */}
                <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 sm:p-5 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <Flame className="w-4 h-4 text-cyan-400" />
                        Microstructure Order Pressure
                      </h3>
                      <InfoBadge infoKey="order_flow_delta" />
                    </div>

                    <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/60 space-y-3">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-emerald-400 font-bold">
                          Buyers: {data.order_flow.buy_pressure_pct}%
                        </span>
                        <span className="text-rose-400 font-bold">
                          Sellers: {data.order_flow.sell_pressure_pct}%
                        </span>
                      </div>
                      {/* Pressure Bar */}
                      <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden flex">
                        <div
                          className="bg-emerald-500 h-full transition-all duration-500"
                          style={{ width: `${data.order_flow.buy_pressure_pct}%` }}
                        />
                        <div
                          className="bg-rose-500 h-full transition-all duration-500"
                          style={{ width: `${data.order_flow.sell_pressure_pct}%` }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                        <span>Net Delta: <strong className={data.order_flow.net_delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {data.order_flow.net_delta >= 0 ? '+' : ''}{data.order_flow.net_delta?.toLocaleString()} shares
                        </strong></span>
                        <span>Total Vol: <strong className="text-white">{data.volume?.toLocaleString()}</strong></span>
                      </div>
                    </div>
                  </div>

                  {/* Proximity to Day Extremes */}
                  <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-400">
                    <span>Session Range:</span>
                    <span className="font-mono text-slate-200">
                      {currSym}{data.low} — {currSym}{data.high} (Spread: {((data.high - data.low) / data.low * 100).toFixed(2)}%)
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Volume Profile (VPVR) & Camarilla Pivots (1 span) */}
          <div className="space-y-6">
            {/* Volume Profile (VPVR) Card */}
            {data?.volume_profile && (
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 backdrop-blur-md">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <BarChart2 className="w-4 h-4 text-amber-400" />
                    Volume Profile (VPVR)
                  </h3>
                  <InfoBadge infoKey="volume_profile" />
                </div>

                <div className="flex items-center justify-between text-xs font-mono mb-2 p-2 bg-slate-950/80 rounded-xl border border-slate-800/60">
                  <div>
                    <span className="text-[10px] text-amber-400 block font-bold">POC PRICE</span>
                    <span className="text-white font-bold">{currSym}{data.volume_profile.poc_price}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-cyan-400 block font-bold">VAL (70%)</span>
                    <span className="text-slate-300 font-bold">{currSym}{data.volume_profile.val_price}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-purple-400 block font-bold">VAH (70%)</span>
                    <span className="text-slate-300 font-bold">{currSym}{data.volume_profile.vah_price}</span>
                  </div>
                </div>

                {/* Horizontal Volume Profile Bars */}
                <div className="space-y-1 max-h-56 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-800">
                  {data.volume_profile.profile.map((b, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center gap-2 text-[10px] font-mono py-0.5 px-1.5 rounded transition ${
                        b.is_poc
                          ? 'bg-amber-500/20 border border-amber-500/40 text-amber-300 font-bold'
                          : b.in_value_area
                          ? 'bg-slate-950/80 text-slate-300'
                          : 'text-slate-500'
                      }`}
                    >
                      <span className="w-12 shrink-0">{b.price.toFixed(2)}</span>
                      <div className="flex-1 bg-slate-800/60 h-2 rounded-full overflow-hidden flex">
                        <div
                          className={`h-full ${b.is_poc ? 'bg-amber-400' : 'bg-cyan-500/70'}`}
                          style={{ width: `${Math.min(b.pct_of_total * 4, 100)}%` }}
                        />
                      </div>
                      {b.is_poc && <span className="text-[9px] text-amber-400 shrink-0">POC</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Camarilla & Floor Pivots Card */}
            {data?.pivots?.camarilla && (
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 backdrop-blur-md">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <Compass className="w-4 h-4 text-cyan-400" />
                    Camarilla Inflection Levels
                  </h3>
                  <InfoBadge infoKey="camarilla_pivots" />
                </div>

                <div className="space-y-2 text-xs font-mono">
                  {/* H4 Breakout */}
                  <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                    <div>
                      <span className="font-bold text-emerald-400">H4 Breakout Target</span>
                      <p className="text-[10px] text-slate-400">Bullish acceleration level</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.h4}</span>
                  </div>

                  {/* H3 Short Resistance */}
                  <div className="flex items-center justify-between p-2 rounded-xl bg-rose-500/10 border border-rose-500/20">
                    <div>
                      <span className="font-bold text-rose-400">H3 Short Resistance</span>
                      <p className="text-[10px] text-slate-400">Mean-reversion ceiling</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.h3}</span>
                  </div>

                  {/* Central Floor Pivot */}
                  <div className="flex items-center justify-between p-2 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div>
                      <span className="font-bold text-slate-300">Central Floor Pivot (P)</span>
                      <p className="text-[10px] text-slate-400">Session equilibrium</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.floor.p}</span>
                  </div>

                  {/* L3 Long Support */}
                  <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                    <div>
                      <span className="font-bold text-emerald-400">L3 Long Support</span>
                      <p className="text-[10px] text-slate-400">Mean-reversion floor</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.l3}</span>
                  </div>

                  {/* L4 Breakdown */}
                  <div className="flex items-center justify-between p-2 rounded-xl bg-rose-500/10 border border-rose-500/20">
                    <div>
                      <span className="font-bold text-rose-400">L4 Breakdown Target</span>
                      <p className="text-[10px] text-slate-400">Bearish acceleration level</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.l4}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── INSTITUTIONAL POSITION SIZER & REAL-TIME SCANNER ─────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Position Sizing & MIS Scalp Calculator */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Calculator className="w-5 h-5 text-cyan-400" />
                  Intraday Position Sizing & MIS Calculator
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Calculate strict share sizing via the 1% risk rule and 5× MIS leverage margin
                </p>
              </div>
              <InfoBadge infoKey="mis_leverage" />
            </div>

            {/* Input Controls */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs mb-4">
              <div>
                <label className="text-slate-400 block mb-1">Trading Capital ({currSym})</label>
                <input
                  type="number"
                  value={calcCapital}
                  onChange={(e) => setCalcCapital(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Risk % Per Trade</label>
                <select
                  value={calcRiskPct}
                  onChange={(e) => setCalcRiskPct(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value={0.5}>0.5% (Conservative)</option>
                  <option value={1.0}>1.0% (Institutional Standard)</option>
                  <option value={2.0}>2.0% (Aggressive)</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Margin Leverage</label>
                <select
                  value={calcLeverage}
                  onChange={(e) => setCalcLeverage(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value={1}>1× (Cash CNC)</option>
                  <option value={3}>3× (Conservative Margin)</option>
                  <option value={5}>5× (MIS Intraday)</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Entry Price ({currSym})</label>
                <input
                  type="number"
                  step="0.05"
                  value={calcEntry}
                  onChange={(e) => setCalcEntry(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Stop Loss ({currSym})</label>
                <input
                  type="number"
                  step="0.05"
                  value={calcStop}
                  onChange={(e) => setCalcStop(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="flex items-end">
                <button
                  type="button"
                  onClick={() => {
                    if (data?.current_price) {
                      setCalcEntry(data.current_price.toString());
                      setCalcStop(data.supertrend?.toString() || (data.current_price * 0.99).toFixed(2));
                    }
                  }}
                  className="w-full py-2 px-2 bg-slate-800 hover:bg-slate-700 text-cyan-300 font-semibold rounded-xl text-xs transition border border-slate-700/60"
                >
                  Sync to Current
                </button>
              </div>
            </div>

            {/* Results Grid */}
            {sizingResults ? (
              <div className="p-4 bg-slate-950/80 rounded-2xl border border-slate-800 space-y-3">
                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">EXACT SHARES</span>
                    <span className="text-lg font-bold text-cyan-300">{sizingResults.exactShares}</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">MARGIN REQUIRED</span>
                    <span className="text-lg font-bold text-white">{currSym}{sizingResults.marginRequired?.toLocaleString()}</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">MAX RUPEE RISK</span>
                    <span className="text-lg font-bold text-rose-400">{currSym}{sizingResults.actualRiskAmount?.toLocaleString()}</span>
                  </div>
                </div>

                {/* Risk-Reward Targets */}
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-xs font-mono">
                  {sizingResults.riskRewardTargets.map((t, idx) => (
                    <div key={idx} className="p-2 bg-emerald-500/5 border border-emerald-500/20 rounded-xl">
                      <span className="text-[10px] text-emerald-400 block font-bold">{t.label}</span>
                      <p className="text-white font-bold">{currSym}{t.price}</p>
                      <p className="text-[10px] text-emerald-300 mt-0.5">+{currSym}{t.profit?.toLocaleString()}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-slate-500 text-xs bg-slate-950/40 rounded-xl border border-slate-800/60">
                Enter valid Entry and Stop Loss prices above to view position sizing and targets.
              </div>
            )}
          </div>

          {/* Real-Time Intraday Radar Scanner */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-400" />
                    Intraday Radar Scanner ({scannerMarket})
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Real-time detection of high momentum, ORB breakouts, and VWAP deviations
                  </p>
                </div>
                <InfoBadge infoKey="intraday_rvol" />
              </div>

              {scannerLoading ? (
                <div className="h-48 flex items-center justify-center text-xs text-slate-400">
                  <RefreshCw className="w-4 h-4 animate-spin mr-2 text-cyan-400" />
                  Scanning high-liquidity universe...
                </div>
              ) : (
                <div className="space-y-2 max-h-72 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-800">
                  {scannerData.map((item) => (
                    <div
                      key={item.ticker}
                      onClick={() => setTicker(item.ticker)}
                      className={`flex items-center justify-between p-2.5 rounded-2xl border transition cursor-pointer ${
                        ticker === item.ticker
                          ? 'bg-cyan-500/10 border-cyan-500/40'
                          : 'bg-slate-950/60 hover:bg-slate-950 border-slate-800/60'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <div className={`p-1.5 rounded-xl ${item.change_pct >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                          {item.change_pct >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-xs text-white">{item.ticker.split('.')[0]}</span>
                            <span className={`text-[10px] font-bold font-mono px-1.5 py-0.2 rounded ${
                              item.orb_status === 'BREAKOUT' ? 'bg-emerald-500/20 text-emerald-400' :
                              item.orb_status === 'BREAKDOWN' ? 'bg-rose-500/20 text-rose-400' :
                              'bg-slate-800 text-slate-400'
                            }`}>
                              {item.orb_status}
                            </span>
                          </div>
                          <p className="text-[10px] text-slate-400 font-mono">
                            VWAP Dist: <strong className={item.vwap_dist_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                              {item.vwap_dist_pct >= 0 ? '+' : ''}{item.vwap_dist_pct}%
                            </strong>
                          </p>
                        </div>
                      </div>

                      <div className="text-right font-mono">
                        <span className="text-xs font-bold text-white">
                          {item.currency_symbol}{item.price}
                        </span>
                        <span className={`block text-[11px] font-semibold ${item.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {item.change_pct >= 0 ? '+' : ''}{item.change_pct}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <span>Click any scanner opportunity to load chart</span>
              <button
                onClick={fetchScanner}
                className="text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1 text-xs"
              >
                <RefreshCw className="w-3 h-3" /> Rescan Now
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
