'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Link from 'next/link';
import {
  Activity, ArrowUpRight, ArrowDownRight, RefreshCw, Layers,
  Compass, Calculator, ShieldAlert, Sparkles, Sliders, ChevronDown,
  Search, TrendingUp, TrendingDown, Target, Zap, Clock, ShieldCheck,
  BarChart2, Flame, Eye, ArrowRight, CheckCircle2, XCircle, AlertCircle,
  Copy, Check, Scale, AlertTriangle, Play, HelpCircle
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

  // Market Pulse state
  const [marketPulse, setMarketPulse] = useState(null);

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

  // Copy plan state
  const [planCopied, setPlanCopied] = useState(false);

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

      if (json.current_price) {
        setCalcEntry(json.current_price.toString());
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

  // Fetch Market Pulse
  const fetchPulse = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/intraday/market-pulse?market=${scannerMarket}`);
      if (res.ok) {
        const json = await res.json();
        setMarketPulse(json);
      }
    } catch (_) {}
  }, [scannerMarket]);

  useEffect(() => {
    fetchPulse();
    const intervalId = setInterval(fetchPulse, 30000);
    return () => clearInterval(intervalId);
  }, [fetchPulse]);

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

  // Position Sizing & Friction Breakeven Calculations
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

    const maxLeveragedCapital = capital * leverage;
    const sharesByCapital = Math.floor(maxLeveragedCapital / entry);

    const exactShares = Math.max(1, Math.min(sharesByRisk, sharesByCapital));
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

    // Real-Life Friction (Brokerage, STT, GST, NSE Turnover)
    let totalCharges = 0;
    let brokerage = 0;
    let stt = 0;
    let nseTxn = 0;
    let gst = 0;

    const buyTurnover = entry * exactShares;
    const sellTurnover = target1 * exactShares;
    const totalTurnover = buyTurnover + sellTurnover;

    if (!isUS) {
      // Indian NSE Intraday Equities standard (Zerodha/Groww)
      brokerage = Math.min(20.0, 0.0005 * buyTurnover) + Math.min(20.0, 0.0005 * sellTurnover);
      stt = 0.00025 * sellTurnover; // 0.025% on sell side
      nseTxn = 0.0000297 * totalTurnover; // 0.00297%
      const sebi = 0.000001 * totalTurnover;
      const stampDuty = 0.00003 * buyTurnover;
      gst = 0.18 * (brokerage + nseTxn + sebi);
      totalCharges = Math.round((brokerage + stt + nseTxn + sebi + stampDuty + gst) * 100) / 100;
    } else {
      // US standard $0 commission with nominal regulatory fee
      totalCharges = Math.round((0.0000278 * sellTurnover + 0.000166 * exactShares) * 100) / 100;
    }

    const breakevenMovePts = Math.round((totalCharges / exactShares) * 100) / 100;
    const breakevenMovePct = Math.round(((breakevenMovePts / entry) * 100) * 1000) / 1000;

    const grossT1 = Math.round(exactShares * t1Dist);
    const grossT2 = Math.round(exactShares * t2Dist);
    const grossT3 = Math.round(exactShares * t3Dist);

    return {
      isLong,
      exactShares,
      marginRequired: Math.round(marginRequired),
      effectiveExposure: Math.round(effectiveExposure),
      actualRiskAmount: Math.round(actualRiskAmount),
      totalCharges,
      breakevenMovePts,
      breakevenMovePct,
      riskRewardTargets: [
        { label: 'Target 1 (1.5R)', price: Math.round(target1 * 100) / 100, gross: grossT1, net: Math.round(grossT1 - totalCharges) },
        { label: 'Target 2 (2.5R)', price: Math.round(target2 * 100) / 100, gross: grossT2, net: Math.round(grossT2 - totalCharges) },
        { label: 'Target 3 (3.5R)', price: Math.round(target3 * 100) / 100, gross: grossT3, net: Math.round(grossT3 - totalCharges) },
      ]
    };
  }, [calcEntry, calcStop, calcCapital, calcRiskPct, calcLeverage, isUS]);

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

  // Copy Trade Plan to Clipboard
  const handleCopyPlan = () => {
    if (data?.battle_plan?.formatted_card) {
      navigator.clipboard.writeText(data.battle_plan.formatted_card);
      setPlanCopied(true);
      setTimeout(() => setPlanCopied(false), 2500);
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
      <div className="max-w-7xl mx-auto space-y-6">

        {/* ── REAL-TIME MARKET SESSION CLOCK & PHASE BANNER ─────────────────── */}
        {marketPulse && (
          <div className="bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800/80 rounded-2xl p-3.5 backdrop-blur-md flex flex-col md:flex-row items-start md:items-center justify-between gap-3 shadow-lg shadow-black/40">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-xl flex items-center justify-center ${marketPulse.is_open ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' : 'bg-amber-500/15 text-amber-400 border border-amber-500/30'}`}>
                <Clock className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    {marketPulse.market === 'IN' ? 'Dalal Street Session' : 'Wall Street Session'} ({marketPulse.local_time})
                  </span>
                  <span className={`px-2 py-0.2 text-[10px] font-bold rounded-full uppercase ${marketPulse.is_open ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}`}>
                    {marketPulse.is_open ? 'LIVE SESSION' : 'CLOSED'}
                  </span>
                  <InfoBadge infoKey="session_phase_clock" />
                </div>
                <p className="text-xs font-bold text-white mt-0.5 flex items-center gap-1.5">
                  <span>{marketPulse.phase_name}</span>
                  <span className="text-slate-500">—</span>
                  <span className="text-slate-300 font-normal">{marketPulse.directive}</span>
                </p>
              </div>
            </div>

            {/* Benchmark Indices Pills */}
            <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0 scrollbar-none">
              {marketPulse.indices?.map((idx) => (
                <div key={idx.symbol} className="px-2.5 py-1 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono shrink-0">
                  <span className="text-slate-400 font-semibold mr-1.5">{idx.name}:</span>
                  <span className="text-white font-bold">{idx.price?.toLocaleString()}</span>
                  <span className={`ml-1 text-[11px] font-bold ${idx.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {idx.change_pct >= 0 ? '+' : ''}{idx.change_pct}%
                  </span>
                </div>
              ))}

              {marketPulse.mins_to_mis_squareoff > 0 && (
                <div className="px-3 py-1 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono shrink-0 flex items-center gap-1.5 font-bold">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Auto-Square-Off in: {marketPulse.mins_to_mis_squareoff}m
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── TOP TERMINAL BAR ────────────────────────────────────────────── */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
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
                  Real-world session clocks, gap intelligence, trap detectors & institutional friction calculators
                </p>
              </div>
            </div>
          </div>

          {/* Quick controls: Market toggle, Auto-refresh & Search */}
          <div className="flex flex-wrap items-center gap-2.5">
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
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 lg:pb-0 scrollbar-none">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider pl-1 pr-1 shrink-0">
              Active Tickers:
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

        {/* ── INSTITUTIONAL TRAP ALERT BANNER (If Active) ─────────────────── */}
        {data?.trap_detection && data.trap_detection.status !== 'NONE' && (
          <div className={`p-3.5 rounded-2xl border flex items-center gap-3 backdrop-blur-md shadow-lg ${
            data.trap_detection.status === 'BULL_TRAP'
              ? 'bg-rose-950/40 border-rose-500/40 text-rose-200'
              : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200'
          }`}>
            <AlertTriangle className={`w-5 h-5 shrink-0 ${data.trap_detection.status === 'BULL_TRAP' ? 'text-rose-400' : 'text-emerald-400'}`} />
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="font-bold text-xs">{data.trap_detection.title}</span>
                <InfoBadge infoKey="institutional_trap_detector" />
              </div>
              <p className="text-xs text-slate-300 mt-0.5">{data.trap_detection.desc}</p>
            </div>
          </div>
        )}

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

            {/* Session VWAP */}
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

            {/* Relative Strength vs Benchmark */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  Relative Strength
                  <InfoBadge infoKey="benchmark_relative_strength" />
                </span>
                <span className="text-[10px] font-mono text-slate-400 font-bold px-1.5 py-0.5 bg-slate-800 rounded">
                  vs {data.relative_strength?.benchmark_name}
                </span>
              </div>
              <div className="mt-1">
                <p className={`text-xl font-bold font-mono ${data.relative_strength?.alpha_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {data.relative_strength?.alpha_pct >= 0 ? '+' : ''}{data.relative_strength?.alpha_pct}% Alpha
                </p>
                <p className="text-[11px] text-slate-400 mt-0.5 truncate">
                  {data.relative_strength?.status}
                </p>
              </div>
            </div>

            {/* Pre-Market Gap Intelligence */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  Pre-Market Gap
                  <InfoBadge infoKey="pre_market_gap" />
                </span>
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                  data.gap_analysis?.gap_pct >= 0 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                }`}>
                  {data.gap_analysis?.gap_type?.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="mt-1">
                <p className={`text-lg font-bold font-mono ${data.gap_analysis?.gap_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {data.gap_analysis?.gap_pct >= 0 ? '+' : ''}{data.gap_analysis?.gap_pct}% ({currSym}{data.gap_analysis?.gap_pts})
                </p>
                <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
                  Fill: <span className={data.gap_analysis?.gap_filled ? 'text-emerald-400' : 'text-amber-400'}>
                    {data.gap_analysis?.gap_filled ? 'FILLED' : `OPEN (${currSym}${data.gap_analysis?.gap_fill_dist})`}
                  </span>
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
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
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
                  <span className="text-slate-500 italic">Hover over candles to inspect high-frequency metrics</span>
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
                          <line x1={padding.left} y1={yScale(data.pivots.camarilla.h4)} x2={chartWidth - padding.right} y2={yScale(data.pivots.camarilla.h4)} stroke="#10b981" strokeWidth="1" strokeDasharray="2 2" />
                        )}
                        {data.pivots.camarilla.h3 && (
                          <line x1={padding.left} y1={yScale(data.pivots.camarilla.h3)} x2={chartWidth - padding.right} y2={yScale(data.pivots.camarilla.h3)} stroke="#f43f5e" strokeWidth="1" strokeDasharray="2 2" />
                        )}
                        {data.pivots.camarilla.l3 && (
                          <line x1={padding.left} y1={yScale(data.pivots.camarilla.l3)} x2={chartWidth - padding.right} y2={yScale(data.pivots.camarilla.l3)} stroke="#10b981" strokeWidth="1" strokeDasharray="2 2" />
                        )}
                        {data.pivots.camarilla.l4 && (
                          <line x1={padding.left} y1={yScale(data.pivots.camarilla.l4)} x2={chartWidth - padding.right} y2={yScale(data.pivots.camarilla.l4)} stroke="#f43f5e" strokeWidth="1" strokeDasharray="2 2" />
                        )}
                      </g>
                    )}

                    {/* VWAP ±2σ Bands */}
                    {showVWAPBands && (
                      <g>
                        <path
                          d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.upper_band_2)}`, '')}
                          fill="none"
                          stroke="#06b6d4"
                          strokeOpacity={0.35}
                          strokeWidth="1"
                          strokeDasharray="2 2"
                        />
                        <path
                          d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.lower_band_2)}`, '')}
                          fill="none"
                          stroke="#06b6d4"
                          strokeOpacity={0.35}
                          strokeWidth="1"
                          strokeDasharray="2 2"
                        />
                      </g>
                    )}

                    {/* VWAP Main Line */}
                    {showVWAP && (
                      <path
                        d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.vwap)}`, '')}
                        fill="none"
                        stroke="#06b6d4"
                        strokeWidth="1.8"
                      />
                    )}

                    {/* EMA 9 and 21 */}
                    {showEMA && (
                      <g>
                        <path d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.ema9)}`, '')} fill="none" stroke="#a855f7" strokeWidth="1.2" />
                        <path d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.ema21)}`, '')} fill="none" stroke="#ec4899" strokeWidth="1.2" strokeOpacity={0.8} />
                      </g>
                    )}

                    {/* Supertrend Stop Line */}
                    {showSupertrend && (
                      <g>
                        {candles.map((c, i) => {
                          if (i === 0) return null;
                          return (
                            <line
                              key={`st-${i}`}
                              x1={xScale(i - 1)}
                              y1={yScale(candles[i - 1].supertrend)}
                              x2={xScale(i)}
                              y2={yScale(c.supertrend)}
                              stroke={c.supertrend_dir === 1 ? '#10b981' : '#f43f5e'}
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
                        <g key={i} onMouseEnter={() => setHoveredCandle(c)} className="cursor-pointer">
                          <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={candleColor} strokeWidth="1" />
                          <rect x={x - candleWidth / 2} y={bodyY} width={candleWidth} height={bodyHeight} fill={candleColor} rx={1} />
                        </g>
                      );
                    })}

                    {/* X-axis Labels */}
                    {candles.map((c, i) => {
                      if (i % Math.ceil(candles.length / 6) !== 0) return null;
                      return (
                        <text key={`x-${i}`} x={xScale(i)} y={chartHeight - 8} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">
                          {c.time}
                        </text>
                      );
                    })}
                  </svg>
                )}
              </div>

              {/* Sub-Chart Selector */}
              <div className="mt-4 pt-4 border-t border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400">Sub-Indicator:</span>
                    <div className="flex items-center bg-slate-950 p-0.5 rounded-lg border border-slate-800 text-xs">
                      <button onClick={() => setActiveSubChart('volume')} className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'volume' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}>Volume & Delta</button>
                      <button onClick={() => setActiveSubChart('rsi')} className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'rsi' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}>RSI (14)</button>
                      <button onClick={() => setActiveSubChart('cvd')} className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'cvd' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}>Order Flow CVD</button>
                    </div>
                  </div>
                </div>

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
                              <rect x={x - candleWidth / 2} y={95 - bH} width={candleWidth / 2} height={bH} fill="#10b981" fillOpacity={0.8} />
                              <rect x={x} y={95 - sH} width={candleWidth / 2} height={sH} fill="#f43f5e" fillOpacity={0.8} />
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
                        d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${100 - (c.rsi || 50)}`, '')}
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
                            d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.cum_delta || 0) - minCvd) / cvdRange) * 80}`, '')}
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

            {/* ── MULTI-TIMEFRAME CONFLUENCE & TACTICAL SIGNALS ──────────────── */}
            {data && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Triple-Screen Confluence Matrix */}
                <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 sm:p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Layers className="w-4 h-4 text-cyan-400" />
                      Triple-Screen Confluence Matrix
                    </h3>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-bold text-cyan-300">
                        {data.multi_timeframe?.confluence_score}%
                      </span>
                      <InfoBadge infoKey="triple_screen_confluence" />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center mb-3 font-mono text-xs">
                    {data.multi_timeframe?.screens?.map((s, idx) => (
                      <div key={idx} className="p-2.5 bg-slate-950/70 border border-slate-800 rounded-xl">
                        <span className="text-[10px] text-slate-400 block font-sans">{s.timeframe}</span>
                        <span className={`text-xs font-bold block my-1 ${s.trend === 'BULLISH' ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {s.trend}
                        </span>
                        <span className="text-[10px] text-slate-500 block">RSI: {s.rsi}</span>
                      </div>
                    ))}
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-950/90 border border-slate-800/80 flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-medium">Confluence Verdict:</span>
                    <span className={`font-bold font-mono ${data.multi_timeframe?.confluence_score >= 70 ? 'text-emerald-400' : data.multi_timeframe?.confluence_score <= 30 ? 'text-rose-400' : 'text-amber-400'}`}>
                      {data.multi_timeframe?.confluence_bias}
                    </span>
                  </div>
                </div>

                {/* Microstructure Order Pressure */}
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
                        <span className="text-emerald-400 font-bold">Buyers: {data.order_flow.buy_pressure_pct}%</span>
                        <span className="text-rose-400 font-bold">Sellers: {data.order_flow.sell_pressure_pct}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-3 rounded-full overflow-hidden flex">
                        <div className="bg-emerald-500 h-full transition-all duration-500" style={{ width: `${data.order_flow.buy_pressure_pct}%` }} />
                        <div className="bg-rose-500 h-full transition-all duration-500" style={{ width: `${data.order_flow.sell_pressure_pct}%` }} />
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                        <span>Net Delta: <strong className={data.order_flow.net_delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {data.order_flow.net_delta >= 0 ? '+' : ''}{data.order_flow.net_delta?.toLocaleString()} shares
                        </strong></span>
                        <span>Total Vol: <strong className="text-white">{data.volume?.toLocaleString()}</strong></span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-xs text-slate-400">
                    <span>Gap Strategy:</span>
                    <span className="font-mono text-cyan-300 font-semibold">{data.gap_analysis?.directive}</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Volume Profile (VPVR) & Camarilla Pivots (1 span) */}
          <div className="space-y-6">
            {/* Volume Profile (VPVR) */}
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

            {/* Camarilla & Floor Pivots */}
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
                  <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                    <div>
                      <span className="font-bold text-emerald-400">H4 Breakout Target</span>
                      <p className="text-[10px] text-slate-400">Bullish acceleration level</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.h4}</span>
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-xl bg-rose-500/10 border border-rose-500/20">
                    <div>
                      <span className="font-bold text-rose-400">H3 Short Resistance</span>
                      <p className="text-[10px] text-slate-400">Mean-reversion ceiling</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.h3}</span>
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-xl bg-slate-950/80 border border-slate-800">
                    <div>
                      <span className="font-bold text-slate-300">Central Floor Pivot (P)</span>
                      <p className="text-[10px] text-slate-400">Session equilibrium</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.floor.p}</span>
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                    <div>
                      <span className="font-bold text-emerald-400">L3 Long Support</span>
                      <p className="text-[10px] text-slate-400">Mean-reversion floor</p>
                    </div>
                    <span className="font-bold text-white">{currSym}{data.pivots.camarilla.l3}</span>
                  </div>

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

        {/* ── REAL-WORLD INTRADAY BATTLE PLAN CARD & EXECUTION ────────────── */}
        {data?.battle_plan && (
          <div className="bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div className="flex items-center gap-2.5">
                <div className="p-2 bg-gradient-to-tr from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 rounded-xl">
                  <Target className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-white">
                      Actionable Intraday Battle Plan: {data.battle_plan.setup_name}
                    </h3>
                    <InfoBadge infoKey="intraday_battle_plan" />
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Pre-calculated institutional entry, hard stop-loss, and multi-tier profit targets
                  </p>
                </div>
              </div>

              <button
                onClick={handleCopyPlan}
                className="px-4 py-2 bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/30 rounded-xl text-xs font-semibold transition flex items-center gap-2 shrink-0 self-start sm:self-auto"
              >
                {planCopied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                {planCopied ? 'Copied to Clipboard!' : 'Copy Plan for Broker / Journal'}
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs font-mono">
              <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-xl">
                <span className="text-[10px] text-slate-400 block font-sans">ENTRY TRIGGER</span>
                <span className="text-base font-bold text-white">{currSym}{data.battle_plan.entry_price}</span>
                <p className="text-[10px] text-slate-500 mt-1 font-sans truncate">{data.battle_plan.trigger_rule}</p>
              </div>

              <div className="p-3 bg-slate-950/80 border border-rose-500/30 rounded-xl">
                <span className="text-[10px] text-rose-400 block font-sans">HARD STOP LOSS</span>
                <span className="text-base font-bold text-rose-400">{currSym}{data.battle_plan.stop_loss}</span>
                <p className="text-[10px] text-slate-500 mt-1 font-sans">Risk: {currSym}{data.battle_plan.risk_per_share} / share</p>
              </div>

              <div className="p-3 bg-slate-950/80 border border-emerald-500/30 rounded-xl">
                <span className="text-[10px] text-emerald-400 block font-sans">TARGET 1 (1.5R)</span>
                <span className="text-base font-bold text-emerald-400">{currSym}{data.battle_plan.target_1}</span>
                <p className="text-[10px] text-slate-500 mt-1 font-sans">Scale out 50% & trail stop</p>
              </div>

              <div className="p-3 bg-slate-950/80 border border-cyan-500/30 rounded-xl">
                <span className="text-[10px] text-cyan-400 block font-sans">TARGET 2 (2.5R)</span>
                <span className="text-base font-bold text-cyan-300">{currSym}{data.battle_plan.target_2}</span>
                <p className="text-[10px] text-slate-500 mt-1 font-sans">Full runner exit target</p>
              </div>
            </div>
          </div>
        )}

        {/* ── REAL-LIFE FRICTION & SEBI BREAKEVEN CALCULATOR ────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Position Sizing & Friction Calculator */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Scale className="w-5 h-5 text-cyan-400" />
                  Real-Life Brokerage, STT & Friction Calculator
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Calculates exact statutory charges & breakeven spread (The SEBI Reality Check)
                </p>
              </div>
              <InfoBadge infoKey="brokerage_friction_breakeven" />
            </div>

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

            {sizingResults ? (
              <div className="p-4 bg-slate-950/80 rounded-2xl border border-slate-800 space-y-3">
                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block font-sans">EXACT SHARES</span>
                    <span className="text-lg font-bold text-cyan-300">{sizingResults.exactShares}</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block font-sans">MARGIN NEEDED</span>
                    <span className="text-lg font-bold text-white">{currSym}{sizingResults.marginRequired?.toLocaleString()}</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block font-sans">TOTAL FRICTION</span>
                    <span className="text-lg font-bold text-amber-400">{currSym}{sizingResults.totalCharges}</span>
                  </div>
                </div>

                {/* Breakeven Spread Alert */}
                <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-between text-xs font-mono">
                  <span className="text-slate-300 font-sans">Breakeven Tick Spread Needed:</span>
                  <span className="text-amber-400 font-bold">
                    +{currSym}{sizingResults.breakevenMovePts} (+{sizingResults.breakevenMovePct}%)
                  </span>
                </div>

                {/* Net Profit Table */}
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-slate-800 text-xs font-mono">
                  {sizingResults.riskRewardTargets.map((t, idx) => (
                    <div key={idx} className="p-2 bg-emerald-500/5 border border-emerald-500/20 rounded-xl">
                      <span className="text-[10px] text-emerald-400 block font-bold font-sans">{t.label}</span>
                      <p className="text-white font-bold">{currSym}{t.price}</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">Gross: +{currSym}{t.gross?.toLocaleString()}</p>
                      <p className="text-[10px] text-emerald-300 font-bold">Net: +{currSym}{t.net?.toLocaleString()}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-slate-500 text-xs bg-slate-950/40 rounded-xl border border-slate-800/60">
                Enter valid Entry and Stop Loss prices above to view net in-pocket profit after brokerage & STT.
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
              <span>Click any opportunity to load into terminal</span>
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
