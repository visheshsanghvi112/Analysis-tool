'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import {
  Activity, ArrowUpRight, ArrowDownRight, RefreshCw, Layers,
  Compass, Calculator, ShieldAlert, Sparkles, Sliders, ChevronDown,
  Search, TrendingUp, TrendingDown, Target, Zap, Clock, ShieldCheck,
  BarChart2, Flame, Eye, ArrowRight, CheckCircle2, XCircle, AlertCircle,
  Copy, Check, Scale, AlertTriangle, Play, HelpCircle,
  Volume2, VolumeX, Edit3, Trash2, Maximize2, Minimize2, Bell, BellOff
} from 'lucide-react';
import InfoBadge from './InfoBadge';
import Header from './Header';

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
  const searchParams = useSearchParams();
  const urlTicker = searchParams ? searchParams.get('ticker') : null;

  const [ticker, setTicker] = useState(urlTicker ? urlTicker.trim().toUpperCase() : 'RELIANCE.NS');
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

  // Audio Alerts state
  const [soundAlerts, setSoundAlerts] = useState(false);

  // Chart Overlays
  const [showVWAP, setShowVWAP] = useState(true);
  const [showVWAPBands, setShowVWAPBands] = useState(true);
  const [showSupertrend, setShowSupertrend] = useState(true);
  const [showEMA, setShowEMA] = useState(true);
  const [showORB, setShowORB] = useState(true);
  const [showCamarilla, setShowCamarilla] = useState(false);
  const [showPDH, setShowPDH] = useState(true);

  // Viewport Zoom: 'all' | '60' | '30'
  const [candleSlice, setCandleSlice] = useState('all');

  // Sub-chart selector
  const [activeSubChart, setActiveSubChart] = useState('volume'); // 'volume' | 'rsi' | 'cvd' | 'macd'

  // Hovered candle for inspection
  const [hoveredCandle, setHoveredCandle] = useState(null);

  // Position Sizing Calculator state
  const [calcCapital, setCalcCapital] = useState(100000);
  const [calcRiskPct, setCalcRiskPct] = useState(1.0);
  const [calcLeverage, setCalcLeverage] = useState(5); // MIS 5x
  const [calcEntry, setCalcEntry] = useState('');
  const [calcStop, setCalcStop] = useState('');

  // Trader's Scratchpad & Journal state
  const [scratchpadOpen, setScratchpadOpen] = useState(false);
  const [notes, setNotes] = useState('');
  const [notesSaved, setNotesSaved] = useState(false);
  const [notesCopied, setNotesCopied] = useState(false);
  const [clearNotesConfirm, setClearNotesConfirm] = useState(false);

  // Chart crosshair
  const [hoveredX, setHoveredX] = useState(null);
  const [hoveredY, setHoveredY] = useState(null); // for Y-axis price label

  // Fullscreen chart mode
  const [fullscreenChart, setFullscreenChart] = useState(false);

  // Price flash animation (green/red on tick update)
  const [priceFlash, setPriceFlash] = useState(null); // 'up' | 'down' | null
  const prevPriceRef = useRef(null);

  // Price alert system
  const [alertPrice, setAlertPrice] = useState('');
  const [alertTriggered, setAlertTriggered] = useState(false);
  const [alertAbove, setAlertAbove] = useState(true); // alert when price goes above/below

  // Scanner state
  const [scannerMarket, setScannerMarket] = useState('IN');
  const [scannerData, setScannerData] = useState([]);
  const [scannerLoading, setScannerLoading] = useState(false);

  // Copy plan state
  const [planCopied, setPlanCopied] = useState(false);

  // Options PCR state
  const [pcrData, setPcrData] = useState(null);
  const [pcrLoading, setPcrLoading] = useState(false);

  // Block / Bulk Deals state
  const [blockDeals, setBlockDeals] = useState(null);
  const [blockDealsLoading, setBlockDealsLoading] = useState(false);

  // Trade Log state (persisted in localStorage)
  const [tradeLog, setTradeLog] = useState([]);
  const [tradeLogOpen, setTradeLogOpen] = useState(false);
  const [newTrade, setNewTrade] = useState({
    ticker: '',
    direction: 'LONG',
    entry: '',
    exit: '',
    qty: '',
    note: '',
  });

  const isUS = ticker && !ticker.endsWith('.NS') && !ticker.endsWith('.BO');
  const currSym = data?.currency_symbol || (isUS ? '$' : '₹');

  // Sync URL query when urlTicker changes
  useEffect(() => {
    if (urlTicker && urlTicker.trim()) {
      const clean = urlTicker.trim().toUpperCase();
      if (clean !== ticker) {
        setTicker(clean);
      }
    }
  }, [urlTicker, ticker]);

  // Sync URL in browser history when ticker changes
  useEffect(() => {
    if (typeof window !== 'undefined' && ticker) {
      try {
        const url = new URL(window.location.href);
        if (url.searchParams.get('ticker') !== ticker) {
          url.searchParams.set('ticker', ticker);
          window.history.replaceState({}, '', url.toString());
        }
      } catch (_) {}
    }
  }, [ticker]);

  // Load Trader's Scratchpad notes for active ticker
  useEffect(() => {
    if (typeof window !== 'undefined' && ticker) {
      try {
        const saved = localStorage.getItem('stockiq_intraday_notes_' + ticker);
        setNotes(saved || '');
      } catch (_) {}
    }
  }, [ticker]);

  const handleNotesChange = (e) => {
    const val = e.target.value;
    setNotes(val);
    if (typeof window !== 'undefined' && ticker) {
      try {
        localStorage.setItem('stockiq_intraday_notes_' + ticker, val);
        setNotesSaved(true);
        setTimeout(() => setNotesSaved(false), 1200);
      } catch (_) {}
    }
  };

  const addTimestampToNotes = () => {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const insertion = notes ? `\n[${timeStr}] ` : `[${timeStr}] `;
    const updated = notes + insertion;
    setNotes(updated);
    if (typeof window !== 'undefined' && ticker) {
      try {
        localStorage.setItem('stockiq_intraday_notes_' + ticker, updated);
      } catch (_) {}
    }
  };

  const addTemplateTag = (tag) => {
    const updated = notes ? `${notes} ${tag} ` : `${tag} `;
    setNotes(updated);
    if (typeof window !== 'undefined' && ticker) {
      try {
        localStorage.setItem('stockiq_intraday_notes_' + ticker, updated);
      } catch (_) {}
    }
  };

  // Synthesizer Chime via Native Web Audio API
  const playChime = useCallback((type = 'notification') => {
    if (!soundAlerts || typeof window === 'undefined') return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const now = ctx.currentTime;
      if (type === 'warning') {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.setValueAtTime(580, now + 0.12);
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.35);
      } else if (type === 'breakout') {
        [523.25, 659.25, 783.99].forEach((freq, i) => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, now + i * 0.08);
          gain.gain.setValueAtTime(0.1, now + i * 0.08);
          gain.gain.exponentialRampToValueAtTime(0.001, now + (i + 1) * 0.14);
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.start(now + i * 0.08);
          osc.stop(now + (i + 1) * 0.14);
        });
      } else {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(659.25, now);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.2);
      }
    } catch (_) {}
  }, [soundAlerts]);

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

  // Audio Alert trigger on trap or breakout detection
  useEffect(() => {
    if (data && soundAlerts) {
      if (data.trap_alert?.detected) {
        playChime('warning');
      } else if (data.orb?.status && data.orb.status !== 'INSIDE_RANGE') {
        playChime('breakout');
      }
    }
  }, [data, soundAlerts, playChime]);

  // Price flash animation on tick update
  useEffect(() => {
    if (!data?.current_price) return;
    const prev = prevPriceRef.current;
    if (prev !== null && prev !== data.current_price) {
      setPriceFlash(data.current_price > prev ? 'up' : 'down');
      const t = setTimeout(() => setPriceFlash(null), 800);
      return () => clearTimeout(t);
    }
    prevPriceRef.current = data.current_price;
  }, [data?.current_price]);

  // Price alert trigger check
  useEffect(() => {
    if (!data?.current_price || !alertPrice) return;
    const ap = parseFloat(alertPrice);
    if (!ap) return;
    const triggered = alertAbove
      ? data.current_price >= ap
      : data.current_price <= ap;
    setAlertTriggered(triggered);
    if (triggered && soundAlerts) playChime('breakout');
  }, [data?.current_price, alertPrice, alertAbove, soundAlerts, playChime]);

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

  // Fetch Options PCR (on ticker change, refreshed every 5min)
  const fetchPCR = useCallback(async () => {
    if (!ticker) return;
    setPcrLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/intraday/options-pcr?ticker=${encodeURIComponent(ticker)}&market=${scannerMarket === 'IN' ? 'IN' : 'US'}`);
      if (res.ok) {
        const json = await res.json();
        setPcrData(json);
      } else {
        setPcrData(null);
      }
    } catch (_) { setPcrData(null); }
    finally { setPcrLoading(false); }
  }, [ticker, scannerMarket]);

  useEffect(() => {
    fetchPCR();
    const id = setInterval(fetchPCR, 300000); // refresh every 5 min
    return () => clearInterval(id);
  }, [fetchPCR]);

  // Fetch NSE Block / Bulk Deals (IN market only, refreshed every 5min)
  const fetchBlockDeals = useCallback(async () => {
    if (scannerMarket !== 'IN') { setBlockDeals(null); return; }
    setBlockDealsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/intraday/block-deals`);
      if (res.ok) {
        const json = await res.json();
        setBlockDeals(json);
      }
    } catch (_) {}
    finally { setBlockDealsLoading(false); }
  }, [scannerMarket]);

  useEffect(() => {
    fetchBlockDeals();
    const id = setInterval(fetchBlockDeals, 300000);
    return () => clearInterval(id);
  }, [fetchBlockDeals]);

  // Load Trade Log from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const saved = localStorage.getItem('stockiq_trade_log');
        if (saved) setTradeLog(JSON.parse(saved));
      } catch (_) {}
    }
  }, []);

  const addTradeEntry = () => {
    const entry = parseFloat(newTrade.entry) || 0;
    const exit = parseFloat(newTrade.exit) || 0;
    const qty = parseInt(newTrade.qty) || 0;
    if (!newTrade.ticker || entry <= 0 || qty <= 0) return;

    const pnlPerShare = newTrade.direction === 'LONG' ? (exit - entry) : (entry - exit);
    const grossPnl = exit > 0 ? Math.round(pnlPerShare * qty * 100) / 100 : null;
    const status = exit > 0 ? (grossPnl >= 0 ? 'WIN' : 'LOSS') : 'OPEN';

    const trade = {
      id: Date.now(),
      ticker: newTrade.ticker.toUpperCase(),
      direction: newTrade.direction,
      entry,
      exit: exit || null,
      qty,
      note: newTrade.note,
      grossPnl,
      status,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      date: new Date().toLocaleDateString(),
    };
    const updated = [trade, ...tradeLog].slice(0, 50); // keep last 50
    setTradeLog(updated);
    try { localStorage.setItem('stockiq_trade_log', JSON.stringify(updated)); } catch (_) {}
    setNewTrade({ ticker: ticker.split('.')[0], direction: 'LONG', entry: '', exit: '', qty: '', note: '' });
  };

  const removeTrade = (id) => {
    const updated = tradeLog.filter(t => t.id !== id);
    setTradeLog(updated);
    try { localStorage.setItem('stockiq_trade_log', JSON.stringify(updated)); } catch (_) {}
  };

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
  const rawCandles = data?.candles || [];
  const candles = useMemo(() => {
    if (!rawCandles.length) return [];
    if (candleSlice === '30') return rawCandles.slice(-30);
    if (candleSlice === '60') return rawCandles.slice(-60);
    return rawCandles;
  }, [rawCandles, candleSlice]);

  const chartHeight = 360;
  const chartWidth = 720;
  const padding = { top: 20, right: 65, bottom: 40, left: 10 };

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

    if (showPDH && data?.pivots?.daily_levels) {
      const { pdh, pdl } = data.pivots.daily_levels;
      if (pdh > 0 && pdh > max) max = pdh;
      if (pdl > 0 && pdl < min) min = pdl;
    }

    const buffer = (max - min) * 0.05 || 1;
    min -= buffer;
    max += buffer;

    const innerW = chartWidth - padding.left - padding.right;
    const innerH = chartHeight - padding.top - padding.bottom;

    const xs = (idx) => padding.left + (idx / Math.max(candles.length - 1, 1)) * innerW;
    const ys = (val) => padding.top + innerH - ((val - min) / (max - min)) * innerH;
    const cw = Math.max(2, Math.min(22, (innerW / candles.length) * 0.7));

    return { priceMin: min, priceMax: max, xScale: xs, yScale: ys, candleWidth: cw };
  }, [candles, showVWAPBands, showSupertrend, showPDH, data]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col justify-between">
      <div>
        <Header
          currentTicker={ticker}
          onTickerSelect={(sym) => {
            if (!sym) return;
            const clean = sym.trim().toUpperCase();
            setTicker(clean);
            try {
              window.history.replaceState({}, '', `/intraday?ticker=${encodeURIComponent(clean)}`);
            } catch (_) {}
          }}
        />

        <main className="w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-6">

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
                  <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full uppercase ${marketPulse.is_open ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-slate-800 text-slate-400'}`}>
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
                  {scannerMarket === 'IN' ? 'Auto-Square-Off' : 'Market Close'} in: {marketPulse.mins_to_mis_squareoff}m
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

            {/* Audio Alerts Toggle */}
            <button
              onClick={() => {
                const next = !soundAlerts;
                setSoundAlerts(next);
                if (next) {
                  try {
                    const AudioCtx = window.AudioContext || window.webkitAudioContext;
                    if (AudioCtx) {
                      const ctx = new AudioCtx();
                      const now = ctx.currentTime;
                      const osc = ctx.createOscillator();
                      const gain = ctx.createGain();
                      osc.type = 'sine';
                      osc.frequency.setValueAtTime(659.25, now);
                      gain.gain.setValueAtTime(0.08, now);
                      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
                      osc.connect(gain);
                      gain.connect(ctx.destination);
                      osc.start(now);
                      osc.stop(now + 0.2);
                    }
                  } catch (_) {}
                }
              }}
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg border text-xs font-semibold transition ${
                soundAlerts
                  ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
                  : 'bg-slate-900/90 text-slate-500 border-slate-800 hover:text-slate-300'
              }`}
              title={soundAlerts ? 'Audio Alerts: ACTIVE (Click to Mute)' : 'Audio Alerts: MUTED (Click to Enable Synthesizer Chimes)'}
            >
              {soundAlerts ? <Volume2 className="w-3.5 h-3.5 text-cyan-400" /> : <VolumeX className="w-3.5 h-3.5" />}
              <span className="hidden sm:inline">{soundAlerts ? 'Sound ON' : 'Muted'}</span>
            </button>

            {/* Trader's Scratchpad Toggle */}
            <button
              onClick={() => setScratchpadOpen(!scratchpadOpen)}
              className={`flex items-center gap-1 px-2.5 py-1.5 rounded-lg border text-xs font-semibold transition ${
                scratchpadOpen
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                  : 'bg-slate-900/90 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Open Trader's Real-Time Execution Notepad & Journal"
            >
              <Edit3 className="w-3.5 h-3.5 text-amber-400" />
              <span className="hidden sm:inline">Trader&apos;s Journal</span>
            </button>

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

        {/* Loading skeleton for headline cards */}
        {loading && !data && (
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-slate-900/60 border border-slate-800/40 rounded-2xl p-4 animate-pulse">
                <div className="h-2.5 w-20 bg-slate-800 rounded mb-3" />
                <div className="h-8 w-28 bg-slate-800 rounded mb-2" />
                <div className="h-2 w-16 bg-slate-800/60 rounded" />
              </div>
            ))}
          </div>
        )}

        {/* ── ACTIVE TICKER HEADLINE BAR ──────────────────────────────────── */}
        {data && (
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
            {/* Price & Change — with flash animation on tick update */}
            <div className={`bg-gradient-to-br from-slate-900/90 to-slate-900/50 rounded-2xl p-4 flex flex-col justify-between transition-all duration-300 ${
              priceFlash === 'up'
                ? 'border border-emerald-400/60 shadow-md shadow-emerald-500/20'
                : priceFlash === 'down'
                ? 'border border-rose-400/60 shadow-md shadow-rose-500/20'
                : 'border border-slate-800/80'
            }`}>
              <div>
                <div className="flex items-center justify-between gap-1">
                  <span className="text-xs font-semibold text-slate-400 truncate max-w-[110px]" title={data.company_name}>{data.company_name}</span>
                  <div className="flex items-center gap-1 shrink-0">
                    <InfoBadge infoKey="live_prices" />
                  </div>
                </div>
                <div className="mt-1">
                  <h2 className={`text-xl sm:text-2xl font-black font-mono tracking-tight transition-colors duration-300 ${
                    priceFlash === 'up' ? 'text-emerald-300' : priceFlash === 'down' ? 'text-rose-300' : 'text-white'
                  }`}>
                    {currSym}{data.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </h2>
                  <div className="flex items-center justify-between gap-1 mt-0.5">
                    <span className={`text-xs font-bold font-mono flex items-center ${data.change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {data.change >= 0 ? <ArrowUpRight className="w-3.5 h-3.5 mr-0.5 shrink-0" /> : <ArrowDownRight className="w-3.5 h-3.5 mr-0.5 shrink-0" />}
                      {data.change >= 0 ? '+' : ''}{data.change} ({data.change >= 0 ? '+' : ''}{data.change_pct}%)
                    </span>
                    {/* Live open P&L badge */}
                    {(() => {
                      const open = tradeLog.filter(t => t.status === 'OPEN' && t.ticker === ticker.split('.')[0]);
                      if (!open.length || !data.current_price) return null;
                      const unrealized = open.reduce((sum, t) => {
                        const pnl = t.direction === 'LONG'
                          ? (data.current_price - t.entry) * t.qty
                          : (t.entry - data.current_price) * t.qty;
                        return sum + pnl;
                      }, 0);
                      return (
                        <span className={`text-[9px] font-bold font-mono px-1 py-0.5 rounded border ${
                          unrealized >= 0 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                        }`}>
                          {unrealized >= 0 ? '+' : ''}{currSym}{Math.round(unrealized)}
                        </span>
                      );
                    })()}
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                <span>O: <strong className="text-slate-200">{currSym}{data.open}</strong></span>
                <span>H: <strong className="text-emerald-400">{currSym}{data.high}</strong></span>
                <span>L: <strong className="text-rose-400">{currSym}{data.low}</strong></span>
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

            {/* Supertrend Signal — 6th headline card */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                  Supertrend
                  <InfoBadge infoKey="supertrend" />
                </span>
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                  data.supertrend_dir === 1 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'
                }`}>
                  {data.supertrend_dir === 1 ? '▲ BULL' : '▼ BEAR'}
                </span>
              </div>
              <div className="mt-1">
                <p className={`text-xl font-bold font-mono ${data.supertrend_dir === 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {currSym}{data.supertrend?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </p>
                <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
                  RSI{' '}
                  <span className={`font-bold ${data.rsi >= 70 ? 'text-rose-400' : data.rsi <= 30 ? 'text-emerald-400' : 'text-slate-200'}`}>
                    {data.rsi?.toFixed(1)}
                  </span>
                  {data.rsi >= 70 && <span className="text-rose-400 ml-1 text-[9px]">OB</span>}
                  {data.rsi <= 30 && <span className="text-emerald-400 ml-1 text-[9px]">OS</span>}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ── MAIN CHART & SIDEBAR SECTION ─────────────────────────────────── */}
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Main Chart Column (3 spans) */}
          <div className={`xl:col-span-3 space-y-4 ${
            fullscreenChart ? 'fixed inset-0 z-[150] bg-slate-950 p-4 overflow-y-auto' : ''
          }`}>
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-4 sm:p-6 backdrop-blur-md">
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
                <div className="flex flex-wrap items-center gap-2">
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

                  {/* Viewport Zoom */}
                  <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800">
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 px-1.5">Zoom</span>
                    {[
                      { id: 'all', label: 'All Day' },
                      { id: '60', label: '60 Bars' },
                      { id: '30', label: '30 Bars' },
                    ].map((z) => (
                      <button
                        key={z.id}
                        onClick={() => setCandleSlice(z.id)}
                        className={`px-2 py-1 text-[11px] font-semibold rounded-lg transition ${
                          candleSlice === z.id
                            ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {z.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Fullscreen + Alert button row */}
                <div className="flex items-center gap-2">
                  {/* Price Alert mini widget */}
                  {data && (
                    <div className={`flex items-center gap-1 px-2 py-1 rounded-lg border text-xs ${
                      alertTriggered
                        ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
                        : 'bg-slate-950 border-slate-800 text-slate-500'
                    }`}>
                      {alertTriggered
                        ? <Bell className="w-3 h-3 text-amber-400 animate-bounce" />
                        : <BellOff className="w-3 h-3" />}
                      <select
                        value={alertAbove ? 'above' : 'below'}
                        onChange={e => { setAlertAbove(e.target.value === 'above'); setAlertTriggered(false); }}
                        className="bg-transparent text-[10px] font-mono focus:outline-none cursor-pointer"
                      >
                        <option value="above" className="bg-slate-900">Alert ≥</option>
                        <option value="below" className="bg-slate-900">Alert ≤</option>
                      </select>
                      <input
                        type="number"
                        placeholder={data.current_price?.toFixed(0)}
                        value={alertPrice}
                        onChange={e => { setAlertPrice(e.target.value); setAlertTriggered(false); }}
                        className="w-16 bg-transparent font-mono text-[10px] text-white placeholder-slate-600 focus:outline-none"
                      />
                    </div>
                  )}
                  <button
                    onClick={() => setFullscreenChart(!fullscreenChart)}
                    className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-400 hover:text-white transition"
                    title={fullscreenChart ? 'Exit Fullscreen' : 'Fullscreen Chart'}
                  >
                    {fullscreenChart ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  {/* ── Price Overlays ── */}
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

                  {/* Divider between Price Overlays and Key Levels */}
                  <div className="w-px h-4 bg-slate-700/60 mx-0.5 self-center" />
                  <span className="text-[9px] text-slate-600 uppercase tracking-wider font-mono">Levels</span>

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

                  <button
                    onClick={() => setShowPDH(!showPDH)}
                    className={`px-2.5 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                      showPDH ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40' : 'bg-slate-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    <span className="w-2 h-0.5 bg-amber-400 rounded-full" />
                    PDH / PDL
                  </button>
                </div>

              {/* Hover Inspection Bar */}
              <div className="min-h-6 flex items-center justify-between text-[11px] font-mono text-slate-400 mt-2 px-1 overflow-x-auto">
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
                    onMouseLeave={() => { setHoveredCandle(null); setHoveredX(null); setHoveredY(null); }}
                    onMouseMove={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setHoveredX(((e.clientX - rect.left) / rect.width) * chartWidth);
                      setHoveredY(((e.clientY - rect.top) / rect.height) * chartHeight);
                    }}
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

                    {/* Previous Day Benchmark Levels (PDH, PDL, PDC) */}
                    {showPDH && data?.pivots?.daily_levels && data.pivots.daily_levels.pdh > 0 && (
                      <g>
                        {/* PDH Line & Tag */}
                        <line
                          x1={padding.left}
                          y1={yScale(data.pivots.daily_levels.pdh)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.pivots.daily_levels.pdh)}
                          stroke="#f59e0b"
                          strokeDasharray="4 3"
                          strokeWidth="1.2"
                          strokeOpacity={0.85}
                        />
                        <text
                          x={chartWidth - padding.right + 4}
                          y={yScale(data.pivots.daily_levels.pdh) + 3}
                          fill="#f59e0b"
                          fontSize="8"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          PDH
                        </text>

                        {/* PDC Line & Tag */}
                        {data.pivots.daily_levels.pdc > 0 && (
                          <>
                            <line
                              x1={padding.left}
                              y1={yScale(data.pivots.daily_levels.pdc)}
                              x2={chartWidth - padding.right}
                              y2={yScale(data.pivots.daily_levels.pdc)}
                              stroke="#94a3b8"
                              strokeDasharray="2 2"
                              strokeWidth="1"
                              strokeOpacity={0.6}
                            />
                            <text
                              x={chartWidth - padding.right + 4}
                              y={yScale(data.pivots.daily_levels.pdc) + 3}
                              fill="#94a3b8"
                              fontSize="8"
                              fontFamily="monospace"
                            >
                              PDC
                            </text>
                          </>
                        )}

                        {/* PDL Line & Tag */}
                        <line
                          x1={padding.left}
                          y1={yScale(data.pivots.daily_levels.pdl)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.pivots.daily_levels.pdl)}
                          stroke="#06b6d4"
                          strokeDasharray="4 3"
                          strokeWidth="1.2"
                          strokeOpacity={0.85}
                        />
                        <text
                          x={chartWidth - padding.right + 4}
                          y={yScale(data.pivots.daily_levels.pdl) + 3}
                          fill="#06b6d4"
                          fontSize="8"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          PDL
                        </text>
                      </g>
                    )}

                    {/* VWAP ±2σ Bands */}
                    {showVWAPBands && (
                      <g>
                        <path
                          d={candles.reduce((acc, c, i) => !c.upper_band_2 ? acc : `${acc}${acc ? ' L' : 'M'} ${xScale(i)} ${yScale(c.upper_band_2)}`, '')}
                          fill="none"
                          stroke="#06b6d4"
                          strokeOpacity={0.35}
                          strokeWidth="1"
                          strokeDasharray="2 2"
                        />
                        <path
                          d={candles.reduce((acc, c, i) => !c.lower_band_2 ? acc : `${acc}${acc ? ' L' : 'M'} ${xScale(i)} ${yScale(c.lower_band_2)}`, '')}
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
                        d={candles.reduce((acc, c, i) => !c.vwap ? acc : `${acc}${acc ? ' L' : 'M'} ${xScale(i)} ${yScale(c.vwap)}`, '')}
                        fill="none"
                        stroke="#06b6d4"
                        strokeWidth="1.8"
                      />
                    )}

                    {/* EMA 9 and 21 — null-guarded for early candles */}
                    {showEMA && (
                      <g>
                        <path d={candles.reduce((acc, c, i) => !c.ema9 ? acc : `${acc}${acc ? ' L' : 'M'} ${xScale(i)} ${yScale(c.ema9)}`, '')} fill="none" stroke="#a855f7" strokeWidth="1.2" />
                        <path d={candles.reduce((acc, c, i) => !c.ema21 ? acc : `${acc}${acc ? ' L' : 'M'} ${xScale(i)} ${yScale(c.ema21)}`, '')} fill="none" stroke="#ec4899" strokeWidth="1.2" strokeOpacity={0.8} />
                      </g>
                    )}

                    {/* Supertrend Stop Line */}
                    {showSupertrend && (
                      <g>
                        {candles.map((c, i) => {
                          if (i === 0 || !candles[i - 1].supertrend || !c.supertrend) return null;
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

                    {/* X-axis Labels — max 4 to prevent overlap */}
                    {candles.map((c, i) => {
                      if (i % Math.ceil(candles.length / 4) !== 0 && i !== candles.length - 1) return null;
                      return (
                        <text key={`x-${i}`} x={xScale(i)} y={chartHeight - 10} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">
                          {c.time}
                        </text>
                      );
                    })}

                    {/* Crosshair vertical line */}
                    {hoveredX !== null && (
                      <line
                        x1={hoveredX} y1={padding.top}
                        x2={hoveredX} y2={chartHeight - padding.bottom}
                        stroke="#06b6d4" strokeWidth="0.7"
                        strokeDasharray="3 3" strokeOpacity={0.5}
                        pointerEvents="none"
                      />
                    )}

                    {/* Crosshair horizontal line & dynamic Y-axis price label */}
                    {hoveredY !== null && hoveredY >= padding.top && hoveredY <= chartHeight - padding.bottom && (() => {
                      const dynamicPrice = priceMax - ((hoveredY - padding.top) / (chartHeight - padding.top - padding.bottom)) * (priceMax - priceMin);
                      return (
                        <g pointerEvents="none">
                          <line
                            x1={padding.left}
                            y1={hoveredY}
                            x2={chartWidth - padding.right}
                            y2={hoveredY}
                            stroke="#06b6d4"
                            strokeWidth="0.7"
                            strokeDasharray="3 3"
                            strokeOpacity={0.5}
                          />
                          <rect
                            x={chartWidth - padding.right + 2}
                            y={hoveredY - 8}
                            width={padding.right - 4}
                            height={16}
                            rx={3}
                            fill="#0f172a"
                            stroke="#06b6d4"
                            strokeWidth="1"
                          />
                          <text
                            x={chartWidth - padding.right + 5}
                            y={hoveredY + 3.5}
                            fill="#38bdf8"
                            fontSize="8"
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            {currSym}{dynamicPrice.toFixed(2)}
                          </text>
                        </g>
                      );
                    })()}
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
                      <button onClick={() => setActiveSubChart('macd')} className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'macd' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}>MACD (12,26,9)</button>
                      <button onClick={() => setActiveSubChart('cvd')} className={`px-2.5 py-1 rounded-md transition ${activeSubChart === 'cvd' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400'}`}>Order Flow CVD</button>
                    </div>
                    <InfoBadge infoKey={activeSubChart === 'cvd' ? 'order_flow_delta' : activeSubChart === 'volume' ? 'order_flow_delta' : 'rsi'} />
                  </div>
                </div>

                <div className="h-36 w-full bg-slate-950/60 rounded-xl border border-slate-800/60 p-2 overflow-hidden">
                  {activeSubChart === 'volume' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const maxVol = Math.max(...candles.map(c => c.volume || 1), 1);
                        const hasDelta = candles.some(c => (c.buyer_vol || 0) + (c.seller_vol || 0) > 0);
                        const lastCandle = candles[candles.length - 1];
                        return (
                          <>
                            <text x={padding.left + 4} y={14} fill="#64748b" fontSize="8" fontFamily="monospace">
                              Vol: {lastCandle?.volume?.toLocaleString() || 0}
                              {hasDelta && ` | Buyers: ${lastCandle?.buyer_vol?.toLocaleString() || 0} | Sellers: ${lastCandle?.seller_vol?.toLocaleString() || 0}`}
                            </text>
                            <text x={chartWidth - padding.right + 4} y={14} fill="#475569" fontSize="8" fontFamily="monospace">
                              Max {(maxVol / 1000).toFixed(0)}K
                            </text>
                            {candles.map((c, i) => {
                              const x = xScale(i);
                              if (hasDelta) {
                                const bH = ((c.buyer_vol || 0) / maxVol) * 85;
                                const sH = ((c.seller_vol || 0) / maxVol) * 85;
                                return (
                                  <g key={i}>
                                    <rect x={x - candleWidth / 2} y={95 - bH} width={candleWidth / 2} height={bH} fill="#10b981" fillOpacity={0.8} />
                                    <rect x={x} y={95 - sH} width={candleWidth / 2} height={sH} fill="#f43f5e" fillOpacity={0.8} />
                                  </g>
                                );
                              }
                              const totalH = ((c.volume || 0) / maxVol) * 85;
                              const isUp = c.close >= c.open;
                              return <rect key={i} x={x - candleWidth / 2} y={95 - totalH} width={candleWidth} height={totalH} fill={isUp ? '#10b981' : '#f43f5e'} fillOpacity={0.6} />;
                            })}
                          </>
                        );
                      })()}
                    </svg>
                  )}

                  {activeSubChart === 'rsi' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      <line x1={padding.left} y1={30} x2={chartWidth - padding.right} y2={30} stroke="#f43f5e" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <text x={padding.left + 4} y={28} fill="#f43f5e" fontSize="8" fontFamily="monospace" fillOpacity={0.8}>OB 70</text>
                      <line x1={padding.left} y1={70} x2={chartWidth - padding.right} y2={70} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.5} />
                      <text x={padding.left + 4} y={83} fill="#10b981" fontSize="8" fontFamily="monospace" fillOpacity={0.8}>OS 30</text>
                      <text x={chartWidth - padding.right - 10} y={16} fill="#94a3b8" fontSize="8" fontFamily="monospace" textAnchor="end">
                        RSI (14): <tspan fill={(candles[candles.length - 1]?.rsi || 50) >= 70 ? '#f43f5e' : (candles[candles.length - 1]?.rsi || 50) <= 30 ? '#10b981' : '#38bdf8'} fontWeight="bold">{(candles[candles.length - 1]?.rsi || 50).toFixed(1)}</tspan>
                      </text>
                      <path
                        d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${100 - (c.rsi || 50)}`, '')}
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="1.5"
                      />
                    </svg>
                  )}

                  {activeSubChart === 'macd' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const histVals = candles.map(c => c.macd_histogram || 0);
                        const macdLine = candles.map(c => c.macd || 0);
                        const signalLine = candles.map(c => c.macd_signal || 0);
                        const allVals = [...histVals, ...macdLine, ...signalLine];
                        const minV = Math.min(...allVals);
                        const maxV = Math.max(...allVals);
                        const range = (maxV - minV) || 0.001;
                        const norm = (v) => 90 - ((v - minV) / range) * 80;
                        const zeroY = norm(0);
                        const lastCandle = candles[candles.length - 1];
                        return (
                          <>
                            {/* Top info badge */}
                            <text x={padding.left + 4} y={14} fill="#64748b" fontSize="8" fontFamily="monospace">
                              MACD: <tspan fill="#38bdf8">{(lastCandle?.macd || 0).toFixed(2)}</tspan> | Sig: <tspan fill="#f59e0b">{(lastCandle?.macd_signal || 0).toFixed(2)}</tspan> | Hist: <tspan fill={(lastCandle?.macd_histogram || 0) >= 0 ? '#10b981' : '#f43f5e'}>{(lastCandle?.macd_histogram || 0).toFixed(2)}</tspan>
                            </text>
                            {/* Zero line */}
                            <line x1={padding.left} y1={zeroY} x2={chartWidth - padding.right} y2={zeroY} stroke="#475569" strokeOpacity={0.6} strokeDasharray="2 2" />
                            {/* 4-color Histogram bars */}
                            {candles.map((c, i) => {
                              const h = c.macd_histogram || 0;
                              const prevH = i > 0 ? (candles[i-1].macd_histogram || 0) : 0;
                              const isPos = h >= 0;
                              const isGrowing = isPos ? h >= prevH : h <= prevH;
                              const barColor = isPos ? (isGrowing ? '#10b981' : '#34d399') : (isGrowing ? '#f43f5e' : '#fb7185');
                              const barOpacity = isGrowing ? 0.85 : 0.45;
                              const y1 = norm(h);
                              const y2 = zeroY;
                              return (
                                <rect
                                  key={i}
                                  x={xScale(i) - candleWidth / 2}
                                  y={Math.min(y1, y2)}
                                  width={candleWidth}
                                  height={Math.max(Math.abs(y1 - y2), 0.5)}
                                  fill={barColor}
                                  fillOpacity={barOpacity}
                                />
                              );
                            })}
                            {/* MACD Line */}
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${norm(c.macd || 0)}`, '')}
                              fill="none" stroke="#38bdf8" strokeWidth="1.5"
                            />
                            {/* Signal Line */}
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${norm(c.macd_signal || 0)}`, '')}
                              fill="none" stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="3 2"
                            />
                          </>
                        );
                      })()}
                    </svg>
                  )}

                  {activeSubChart === 'cvd' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const cvdVals = candles.map(c => c.cum_delta || 0);
                        const minCvd = Math.min(...cvdVals, 0);
                        const maxCvd = Math.max(...cvdVals, 1);
                        const cvdRange = (maxCvd - minCvd) || 1;
                        const zeroY = 90 - ((0 - minCvd) / cvdRange) * 80;
                        const lastCandle = candles[candles.length - 1];
                        return (
                          <>
                            <text x={padding.left + 4} y={14} fill="#64748b" fontSize="8" fontFamily="monospace">
                              CVD Net Cumulative Delta: <tspan fill={(lastCandle?.cum_delta || 0) >= 0 ? '#eab308' : '#f43f5e'} fontWeight="bold">{(lastCandle?.cum_delta || 0) >= 0 ? '+' : ''}{(lastCandle?.cum_delta || 0).toLocaleString()} shares</tspan>
                            </text>
                            <line x1={padding.left} y1={zeroY} x2={chartWidth - padding.right} y2={zeroY} stroke="#475569" strokeOpacity={0.6} strokeDasharray="2 2" />
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.cum_delta || 0) - minCvd) / cvdRange) * 80}`, '')}
                              fill="none"
                              stroke="#eab308"
                              strokeWidth="1.8"
                            />
                          </>
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
        {data?.battle_plan?.entry_price && (
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

        {/* ── TRADER'S EXECUTION SCRATCHPAD & JOURNAL ─────────────────────── */}
        {scratchpadOpen && (
          <div className="bg-slate-900/90 border border-amber-500/30 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl shadow-amber-950/10">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center">
                  <Edit3 className="w-4 h-4 text-amber-400" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-bold text-white tracking-wide">
                      Trader&apos;s Execution Scratchpad &amp; Mental Discipline Journal
                    </h3>
                    <InfoBadge infoKey="traders_scratchpad" />
                  </div>
                  <p className="text-[11px] text-slate-400">
                    Live trade thesis, mental stops &amp; execution notes for <strong>{ticker}</strong> — auto-saved locally
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {notesSaved && (
                  <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                    <CheckCircle2 className="w-3 h-3" /> Saved
                  </span>
                )}
                <button
                  onClick={addTimestampToNotes}
                  className="px-2.5 py-1 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 flex items-center gap-1 transition"
                  title="Insert current local time into journal"
                >
                  <Clock className="w-3 h-3 text-cyan-400" />
                  <span>+ Timestamp</span>
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard?.writeText(notes || '');
                    setNotesCopied(true);
                    setTimeout(() => setNotesCopied(false), 2500);
                  }}
                  className={`px-2.5 py-1 text-xs font-semibold rounded-lg border flex items-center gap-1 transition ${notesCopied ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'}`}
                >
                  {notesCopied ? <CheckCircle2 className="w-3 h-3" /> : <Copy className="w-3 h-3 text-slate-400" />}
                  <span>{notesCopied ? 'Copied ✓' : 'Copy'}</span>
                </button>
                {clearNotesConfirm ? (
                  <button
                    onClick={() => {
                      setNotes('');
                      try { localStorage.removeItem('stockiq_intraday_notes_' + ticker); } catch (_) {}
                      setClearNotesConfirm(false);
                    }}
                    className="px-2.5 py-1 text-xs font-bold bg-rose-500/15 text-rose-400 rounded-lg border border-rose-500/30 transition"
                  >
                    Confirm?
                  </button>
                ) : (
                  <button
                    onClick={() => setClearNotesConfirm(true)}
                    onBlur={() => setTimeout(() => setClearNotesConfirm(false), 300)}
                    className="p-1.5 text-slate-500 hover:text-rose-400 transition"
                    title="Clear notes"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>

            {/* Quick Discipline Tags */}
            <div className="flex flex-wrap items-center gap-1.5 pt-3 pb-2 text-[11px]">
              <span className="text-slate-500 font-mono text-[10px] uppercase">Discipline Tags:</span>
              {[
                { tag: '📌 [VWAP Retest Entry]', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20' },
                { tag: '🛑 [Hard Stop Violation Risk]', color: 'text-rose-400 bg-rose-500/10 border-rose-500/20' },
                { tag: '🎯 [Camarilla Target Achieved]', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' },
                { tag: '⚠️ [Lunch Chop Slump - No Trade]', color: 'text-amber-400 bg-amber-500/10 border-amber-500/20' },
                { tag: '⚡ [MIS Square-Off Approaching]', color: 'text-purple-400 bg-purple-500/10 border-purple-500/20' },
              ].map((chip) => (
                <button
                  key={chip.tag}
                  onClick={() => addTemplateTag(chip.tag)}
                  className={`px-2 py-0.5 rounded border text-[11px] font-mono transition hover:scale-105 active:scale-95 ${chip.color}`}
                >
                  {chip.tag}
                </button>
              ))}
            </div>

            {/* Notepad Textarea */}
            <textarea
              value={notes}
              onChange={handleNotesChange}
              placeholder={`Write your trade hypothesis for ${ticker}...\nExample:\n- 10:15 AM: Bullish reclaim of VWAP with positive CVD (+15,000 delta).\n- Entry: At VWAP retest.\n- Stop Loss: 5m close below Supertrend.\n- Target: Camarilla H3 resistance.`}
              className="w-full h-28 sm:h-32 bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs sm:text-sm font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-amber-500/50 resize-y"
            />
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
                      <p className={`text-[10px] font-bold ${t.net >= 0 ? 'text-emerald-300' : 'text-rose-400'}`}>Net: {t.net >= 0 ? '+' : ''}{currSym}{t.net?.toLocaleString()}</p>
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
                            <span className={`text-[10px] font-bold font-mono px-1.5 py-0.5 rounded ${
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
                            {item.rvol && <span className={`ml-2 ${item.rvol >= 2 ? 'text-amber-400' : item.rvol >= 1.5 ? 'text-cyan-400' : 'text-slate-500'}`}>
                              RVOL {item.rvol}×
                            </span>}
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
                  {scannerData.length === 0 && (
                    <div className="h-28 flex flex-col items-center justify-center text-xs text-slate-500 gap-2">
                      <Zap className="w-5 h-5 text-slate-700" />
                      No high-momentum setups detected in current session.
                    </div>
                  )}
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

        {/* ── OPTIONS PCR + BLOCK DEALS + TRADE LOG ROW ──────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* Options Put-Call Ratio Widget */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 backdrop-blur-md">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Scale className="w-4 h-4 text-purple-400" />
                  Options Put-Call Ratio
                </h3>
                <InfoBadge infoKey="options_pcr" />
              </div>
              <div className="flex items-center gap-2">
                {pcrLoading && <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />}
                <button onClick={fetchPCR} className="text-slate-500 hover:text-slate-300 transition">
                  <RefreshCw className="w-3 h-3" />
                </button>
              </div>
            </div>

            {pcrData ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <span className={`text-2xl font-black font-mono ${
                      pcrData.color === 'bearish' ? 'text-rose-400' :
                      pcrData.color === 'bullish' ? 'text-emerald-400' : 'text-amber-400'
                    }`}>{pcrData.pcr_oi}</span>
                    <span className="text-slate-500 text-xs ml-1">PCR OI</span>
                  </div>
                  <div className="text-right">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                      pcrData.color === 'bearish' ? 'bg-rose-500/20 text-rose-400' :
                      pcrData.color === 'bullish' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
                    }`}>{pcrData.sentiment?.replace('_', ' ')}</span>
                    <p className="text-[10px] text-slate-400 mt-0.5">{pcrData.expiry_date}</p>
                  </div>
                </div>

                {/* Put vs Call OI bar */}
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] font-mono">
                    <span className="text-emerald-400">Calls: {(pcrData.call_oi / 1000).toFixed(0)}K OI</span>
                    <span className="text-rose-400">Puts: {(pcrData.put_oi / 1000).toFixed(0)}K OI</span>
                  </div>
                  <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden flex">
                    {(() => {
                      const total = (pcrData.call_oi || 0) + (pcrData.put_oi || 0) || 1;
                      const callPct = Math.round((pcrData.call_oi / total) * 100);
                      return (
                        <>
                          <div className="bg-emerald-500 h-full transition-all" style={{ width: `${callPct}%` }} />
                          <div className="bg-rose-500 h-full transition-all" style={{ width: `${100 - callPct}%` }} />
                        </>
                      );
                    })()}
                  </div>
                </div>

                {pcrData.max_pain_strike && (
                  <div className="p-2 bg-slate-950/60 rounded-xl border border-slate-800 flex justify-between text-xs">
                    <span className="text-slate-400">Max Pain Strike:</span>
                    <span className="font-bold font-mono text-amber-400">{currSym}{pcrData.max_pain_strike}</span>
                  </div>
                )}

                <p className="text-[10px] text-slate-500 leading-snug">{pcrData.sentiment_label}</p>
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-xs text-slate-500">
                {pcrLoading ? 'Loading options chain...' : 'No options data available for this ticker'}
              </div>
            )}
          </div>

          {/* NSE Block / Bulk Deals */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 backdrop-blur-md">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Flame className="w-4 h-4 text-orange-400" />
                  NSE Block & Bulk Deals
                </h3>
                <InfoBadge infoKey="block_deals" />
              </div>
              {blockDealsLoading && <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />}
            </div>

            {scannerMarket !== 'IN' ? (
              <div className="h-32 flex items-center justify-center text-xs text-slate-500 text-center">
                Block/Bulk deal feed is available for NSE (Indian market) only.
              </div>
            ) : blockDeals ? (
              <div className="space-y-2 max-h-52 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-800">
                {[...(blockDeals.block_deals || []).map(d => ({...d, type: 'BLOCK'})),
                   ...(blockDeals.bulk_deals || []).map(d => ({...d, type: 'BULK'}))].slice(0, 15).map((deal, idx) => (
                  <div
                    key={idx}
                    onClick={() => deal.symbol && setTicker(deal.symbol + '.NS')}
                    className="flex items-center justify-between p-2 rounded-xl bg-slate-950/60 border border-slate-800/60 hover:border-slate-700 cursor-pointer transition text-xs"
                  >
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-white">{deal.symbol}</span>
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${
                          deal.type === 'BLOCK' ? 'bg-purple-500/20 text-purple-400' : 'bg-orange-500/20 text-orange-400'
                        }`}>{deal.type}</span>
                        <span className={`text-[9px] font-semibold ${
                          deal.trade_type === 'B' || deal.trade_type === 'BUY' ? 'text-emerald-400' : 'text-rose-400'
                        }`}>{deal.trade_type === 'B' || deal.trade_type === 'BUY' ? 'BUY' : 'SELL'}</span>
                      </div>
                      <p className="text-[10px] text-slate-500 mt-0.5 truncate max-w-[120px]">{deal.client || 'Undisclosed'}</p>
                    </div>
                    <div className="text-right font-mono">
                      <span className="text-slate-300 font-semibold">{deal.quantity?.toLocaleString() || '—'}</span>
                      <p className="text-[10px] text-slate-500">@ ₹{deal.price || deal.avg_price || '—'}</p>
                    </div>
                  </div>
                ))}
                {blockDeals.block_count === 0 && blockDeals.bulk_count === 0 && (
                  <div className="h-24 flex items-center justify-center text-xs text-slate-500">
                    No block/bulk deals recorded today yet.
                  </div>
                )}
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-xs text-slate-500">
                Loading NSE deal feed...
              </div>
            )}
          </div>

          {/* Trade Log with Live P&L */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 backdrop-blur-md">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-1.5">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-cyan-400" />
                  Trade Log & P&L Tracker
                </h3>
                <InfoBadge infoKey="trade_log" />
              </div>
              <button
                onClick={() => setTradeLogOpen(!tradeLogOpen)}
                className={`text-xs font-semibold px-2 py-1 rounded-lg border transition ${
                  tradeLogOpen ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30' : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}
              >
                {tradeLogOpen ? 'Hide Form' : '+ Log Trade'}
              </button>
            </div>

            {tradeLogOpen && (
              <div className="mb-3 p-3 bg-slate-950/70 rounded-2xl border border-slate-800 space-y-2">
                <div className="grid grid-cols-2 gap-2">
                  <input
                    value={newTrade.ticker}
                    onChange={e => setNewTrade(p => ({...p, ticker: e.target.value}))}
                    placeholder="Ticker (e.g. SBIN)"
                    className="col-span-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  />
                  <select
                    value={newTrade.direction}
                    onChange={e => setNewTrade(p => ({...p, direction: e.target.value}))}
                    className="col-span-1 bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none"
                  >
                    <option value="LONG">LONG</option>
                    <option value="SHORT">SHORT</option>
                  </select>
                  <input
                    type="number" step="0.05"
                    value={newTrade.entry}
                    onChange={e => setNewTrade(p => ({...p, entry: e.target.value}))}
                    placeholder="Entry price"
                    className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  />
                  <input
                    type="number" step="0.05"
                    value={newTrade.exit}
                    onChange={e => setNewTrade(p => ({...p, exit: e.target.value}))}
                    placeholder="Exit price (optional)"
                    className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  />
                  <input
                    type="number"
                    value={newTrade.qty}
                    onChange={e => setNewTrade(p => ({...p, qty: e.target.value}))}
                    placeholder="Qty / Shares"
                    className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none"
                  />
                  <button
                    onClick={() => setNewTrade(p => ({...p, ticker: ticker.split('.')[0], entry: data?.current_price?.toString() || ''}))}
                    className="bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-lg px-2 py-1.5 text-xs font-semibold transition"
                  >
                    Sync Live
                  </button>
                </div>
                <button
                  onClick={addTradeEntry}
                  className="w-full py-2 bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 hover:from-cyan-500/30 border border-cyan-500/30 text-cyan-300 font-bold rounded-xl text-xs transition"
                >
                  Add to Trade Log
                </button>
              </div>
            )}

            {/* Trade Log Summary */}
            {tradeLog.length > 0 && (() => {
              const closed = tradeLog.filter(t => t.status !== 'OPEN');
              const totalPnl = closed.reduce((s, t) => s + (t.grossPnl || 0), 0);
              const wins = closed.filter(t => t.status === 'WIN').length;
              const winRate = closed.length ? Math.round((wins / closed.length) * 100) : 0;
              return (
                <div className="flex items-center justify-between mb-2 px-1">
                  <span className="text-[10px] text-slate-400">{closed.length} closed · {winRate}% W/R</span>
                  <span className={`text-xs font-bold font-mono ${totalPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    Net P&L: {totalPnl >= 0 ? '+' : ''}{currSym}{totalPnl.toLocaleString()}
                  </span>
                </div>
              );
            })()}

            <div className="space-y-1.5 max-h-44 overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-slate-800">
              {tradeLog.slice(0, 20).map(t => (
                <div key={t.id} className={`flex items-center justify-between p-2 rounded-xl border text-xs ${
                  t.status === 'WIN' ? 'bg-emerald-500/5 border-emerald-500/20' :
                  t.status === 'LOSS' ? 'bg-rose-500/5 border-rose-500/20' :
                  'bg-slate-950/60 border-slate-800'
                }`}>
                  <div className="flex items-center gap-2">
                    <span className={`text-[9px] font-bold px-1 py-0.5 rounded ${
                      t.direction === 'LONG' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                    }`}>{t.direction}</span>
                    <div>
                      <span className="font-bold text-white">{t.ticker}</span>
                      <span className="text-slate-500 ml-1 font-mono">{t.time}</span>
                      {t.note && <p className="text-[10px] text-slate-500 mt-0.5 truncate max-w-[110px]">{t.note}</p>}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`font-bold font-mono ${
                      t.status === 'WIN' ? 'text-emerald-400' :
                      t.status === 'LOSS' ? 'text-rose-400' : 'text-slate-400'
                    }`}>
                      {t.grossPnl !== null ? `${t.grossPnl >= 0 ? '+' : ''}${currSym}${t.grossPnl}` : 'OPEN'}
                    </span>
                    <button onClick={() => removeTrade(t.id)} className="text-slate-600 hover:text-rose-400 transition">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
              {tradeLog.length === 0 && !tradeLogOpen && (
                <div className="h-24 flex items-center justify-center text-xs text-slate-500">
                  No trades logged yet. Click &quot;+ Log Trade&quot; above to start tracking positions.
                </div>
              )}
            </div>
          </div>
        </div>

        </main>
      </div>

      {/* ── GLOBAL SITE FOOTER ── */}
      <footer className="w-full border-t border-slate-900 bg-black/90 backdrop-blur-md py-6 px-4 sm:px-6 lg:px-8 mt-12 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2 text-white font-bold no-underline hover:opacity-85 transition">
              <div className="w-6 h-6 bg-white rounded-md flex items-center justify-center">
                <TrendingUp className="w-3.5 h-3.5 text-black" />
              </div>
              <span className="text-sm font-bold text-white">StockIQ Pro</span>
            </Link>
            <span className="text-slate-700">|</span>
            <span className="text-slate-400 text-xs">
              by{' '}
              <a
                href="https://visheshsanghvi.qzz.io/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-300 hover:text-white underline transition"
              >
                Vishesh Sanghvi
              </a>
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 text-xs text-slate-400">
            <Link href="/" className="hover:text-white transition no-underline">Dashboard</Link>
            <Link href="/browse" className="hover:text-white transition no-underline">Browse</Link>
            <Link href="/portfolio" className="hover:text-white transition no-underline">Portfolio Tracker</Link>
            <Link href="/features" className="hover:text-white transition no-underline">Features &amp; Docs</Link>
            <Link href="/terms" className="hover:text-white transition no-underline">Terms &amp; Disclaimer</Link>
          </div>

          <p className="text-[11px] text-slate-500 text-center sm:text-right">
            Data via Yahoo Finance &amp; NSE/BSE proxies (~15m delay). Not financial advice.
          </p>
        </div>
      </footer>
    </div>
  );
}
