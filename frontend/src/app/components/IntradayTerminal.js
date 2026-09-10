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
  Volume2, VolumeX, Edit3, Trash2, Maximize2, Minimize2, Bell, BellOff,
  Star, Keyboard, X, Download
} from 'lucide-react';
import InfoBadge from './InfoBadge';
import Header from './Header';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (
  typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : 'https://stock-analysis-backend-seven.vercel.app'
);

const QUICK_TICKERS = [
  { symbol: 'RELIANCE.NS',   name: 'Reliance Ind.', market: 'IN' },
  { symbol: 'TCS.NS',        name: 'TCS',           market: 'IN' },
  { symbol: 'HDFCBANK.NS',   name: 'HDFC Bank',     market: 'IN' },
  { symbol: 'INFY.NS',       name: 'Infosys',       market: 'IN' },
  { symbol: 'TATAMOTORS.NS', name: 'Tata Motors',   market: 'IN' },
  { symbol: 'SBIN.NS',       name: 'SBI',           market: 'IN' },
  { symbol: 'ICICIBANK.NS',  name: 'ICICI Bank',    market: 'IN' },
  { symbol: 'BHARTIARTL.NS', name: 'Airtel',        market: 'IN' },
  { symbol: 'BAJFINANCE.NS', name: 'Bajaj Finance', market: 'IN' },
  { symbol: 'MARUTI.NS',     name: 'Maruti Suzuki', market: 'IN' },
  { symbol: 'TATASTEEL.NS',  name: 'Tata Steel',    market: 'IN' },
  { symbol: 'SUNPHARMA.NS',  name: 'Sun Pharma',    market: 'IN' },
  { symbol: 'ADANIENT.NS',   name: 'Adani Ent',     market: 'IN' },
  { symbol: 'TITAN.NS',      name: 'Titan Co',      market: 'IN' },
  { symbol: 'NVDA',          name: 'Nvidia Corp',   market: 'US' },
  { symbol: 'AAPL',          name: 'Apple Inc',     market: 'US' },
  { symbol: 'TSLA',          name: 'Tesla Inc',     market: 'US' },
  { symbol: 'MSFT',          name: 'Microsoft',     market: 'US' },
  { symbol: 'AMD',           name: 'AMD Inc',       market: 'US' },
  { symbol: 'AMZN',          name: 'Amazon',        market: 'US' },
  { symbol: 'SPY',           name: 'S&P 500 ETF',   market: 'US' },
  { symbol: 'QQQ',           name: 'Invesco QQQ',   market: 'US' },
];

const TIMEFRAMES = [
  { label: '1m', interval: '1m', period: '1d' },
  { label: '2m', interval: '2m', period: '1d' },
  { label: '3m', interval: '3m', period: '1d' },
  { label: '5m', interval: '5m', period: '1d' },
  { label: '15m', interval: '15m', period: '1d' },
  { label: '30m', interval: '30m', period: '1d' },
  { label: '1h', interval: '1h', period: '5d' },
];

export default function IntradayTerminal() {
  const searchParams = useSearchParams();
  const urlTicker = searchParams ? searchParams.get('ticker') : null;

  // Initialize ticker safely from searchParams or window.location, defaulting to RELIANCE.NS
  const [ticker, setTicker] = useState(() => {
    if (urlTicker && urlTicker.trim()) return urlTicker.trim().toUpperCase();
    if (typeof window !== 'undefined') {
      try {
        const p = new URLSearchParams(window.location.search).get('ticker');
        if (p && p.trim()) return p.trim().toUpperCase();
      } catch (_) {}
    }
    return 'RELIANCE.NS';
  });
  const [searchInput, setSearchInput] = useState('');
  const [candleInterval, setCandleInterval] = useState('5m');
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
  const [showCPR, setShowCPR] = useState(true);
  const [showEMA200, setShowEMA200] = useState(false);
  const [candleMode, setCandleMode] = useState('regular'); // 'regular' | 'heikin_ashi'

  // Pinned Watchlist & Hotkeys Modal
  const [pinnedTickers, setPinnedTickers] = useState([]);
  const [showHotkeysModal, setShowHotkeysModal] = useState(false);
  const searchInputRef = useRef(null);

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

  // Track ticker synced to URL to avoid ping-pong loops
  const lastSyncedTickerRef = useRef(urlTicker ? urlTicker.trim().toUpperCase() : null);

  // Helper to switch active ticker and sync to URL without creating infinite history loops
  const changeTicker = useCallback((newSym) => {
    if (!newSym) return;
    const clean = newSym.trim().toUpperCase();
    if (clean === ticker) return;

    lastSyncedTickerRef.current = clean;
    setTicker(clean);

    if (typeof window !== 'undefined') {
      try {
        const currentUrl = new URL(window.location.href);
        if (currentUrl.searchParams.get('ticker') !== clean) {
          currentUrl.searchParams.set('ticker', clean);
          window.history.replaceState(window.history.state, '', `${currentUrl.pathname}?${currentUrl.searchParams.toString()}`);
        }
      } catch (_) {}
    }
  }, [ticker]);

  // Sync URL query when urlTicker changes externally (e.g., browser back/forward buttons)
  useEffect(() => {
    if (urlTicker && urlTicker.trim()) {
      const clean = urlTicker.trim().toUpperCase();
      if (clean !== ticker && clean !== lastSyncedTickerRef.current) {
        lastSyncedTickerRef.current = clean;
        setTicker(clean);
      }
    }
  }, [urlTicker, ticker]);

  // Load Trader's Scratchpad notes for active ticker
  useEffect(() => {
    if (typeof window !== 'undefined' && ticker) {
      try {
        const saved = localStorage.getItem('stockiq_intraday_notes_' + ticker);
        setNotes(saved || '');
      } catch (_) {}
    }
  }, [ticker]);

  // Load Pinned Watchlist from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const saved = localStorage.getItem('stockiq_pinned_tickers');
        if (saved) {
          setPinnedTickers(JSON.parse(saved));
        } else {
          const defaults = ['RELIANCE.NS', 'HDFCBANK.NS', 'TCS.NS', 'INFY.NS', 'TATASTEEL.NS'];
          setPinnedTickers(defaults);
          localStorage.setItem('stockiq_pinned_tickers', JSON.stringify(defaults));
        }
      } catch (_) {}
    }
  }, []);

  const togglePinTicker = useCallback((sym) => {
    const target = (sym || ticker).toUpperCase();
    setPinnedTickers(prev => {
      const exists = prev.includes(target);
      const updated = exists ? prev.filter(t => t !== target) : [...prev, target];
      if (typeof window !== 'undefined') {
        try {
          localStorage.setItem('stockiq_pinned_tickers', JSON.stringify(updated));
        } catch (_) {}
      }
      return updated;
    });
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

  // Synthesizer Chime via Native Web Audio API with shared singleton context
  const audioCtxRef = useRef(null);

  const getAudioContext = useCallback(() => {
    if (typeof window === 'undefined') return null;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return null;
      if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
        audioCtxRef.current = new AudioCtx();
      }
      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume().catch(() => {});
      }
      return audioCtxRef.current;
    } catch (_) {
      return null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') {
        try {
          audioCtxRef.current.close().catch(() => {});
        } catch (_) {}
      }
    };
  }, []);

  const playChime = useCallback((type = 'notification') => {
    if (!soundAlerts || typeof window === 'undefined') return;
    try {
      const ctx = getAudioContext();
      if (!ctx || ctx.state !== 'running') return;
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
  }, [soundAlerts, getAudioContext]);

  // Fetch Main Intraday Data
  const fetchData = useCallback(async (isSilent = false) => {
    if (!isSilent) setLoading(true);
    setIsRefreshing(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/intraday/analysis?ticker=${encodeURIComponent(ticker)}&interval=${candleInterval}&period=${period}`);
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
  }, [ticker, candleInterval, period, autoRefreshSecs]);

  // Export Intraday Candles & Technical Indicators to CSV
  const handleExportIntradayCSV = useCallback(() => {
    if (!data?.candles?.length) return;
    const cleanTicker = (ticker || 'INTRADAY').replace(/[^a-zA-Z0-9_-]/g, '_');
    const headers = [
      'Timestamp',
      'Open',
      'High',
      'Low',
      'Close',
      'Volume',
      'VWAP',
      'VWAP_Upper_1',
      'VWAP_Lower_1',
      'Supertrend',
      'Supertrend_Signal',
      'EMA9',
      'EMA21',
      'EMA50',
      'EMA200',
      'RSI',
      'MACD',
      'MACD_Signal',
      'ATR'
    ];
    const rows = data.candles.map(c => [
      `"${c.timestamp || ''}"`,
      c.open ?? '',
      c.high ?? '',
      c.low ?? '',
      c.close ?? '',
      c.volume ?? '',
      c.vwap ?? '',
      c.upper_1 ?? '',
      c.lower_1 ?? '',
      c.supertrend ?? '',
      c.supertrend_dir === 1 ? 'BULLISH' : 'BEARISH',
      c.ema9 ?? '',
      c.ema21 ?? '',
      c.ema50 ?? '',
      c.ema200 ?? '',
      c.rsi ?? '',
      c.macd ?? '',
      c.macd_signal ?? '',
      c.atr ?? ''
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${cleanTicker}_Intraday_${candleInterval}_${period}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [data?.candles, ticker, candleInterval, period]);

  // Initial and param-change load
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Pro Trading Hotkeys global keyboard listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') {
        if (e.key === 'Escape') {
          document.activeElement?.blur();
        }
        return;
      }

      if (e.key === '/') {
        e.preventDefault();
        searchInputRef.current?.focus();
      } else if (e.key === '1') {
        setCandleInterval('1m'); setPeriod('1d');
      } else if (e.key === '2') {
        setCandleInterval('2m'); setPeriod('1d');
      } else if (e.key === '3') {
        setCandleInterval('3m'); setPeriod('1d');
      } else if (e.key === '5') {
        setCandleInterval('5m'); setPeriod('1d');
      } else if (e.key === '4') {
        setCandleInterval('15m'); setPeriod('1d');
      } else if (e.key === '6') {
        setCandleInterval('30m'); setPeriod('1d');
      } else if (e.key.toLowerCase() === 'h' && !e.ctrlKey && !e.metaKey) {
        setCandleInterval('1h'); setPeriod('5d');
      } else if (e.key.toLowerCase() === 'v') {
        setShowVWAP(prev => !prev);
      } else if (e.key.toLowerCase() === 's') {
        setShowSupertrend(prev => !prev);
      } else if (e.key.toLowerCase() === 'c') {
        setShowCPR(prev => !prev);
      } else if (e.key.toLowerCase() === 'k') {
        setCandleMode(prev => prev === 'regular' ? 'heikin_ashi' : 'regular');
      } else if (e.key.toLowerCase() === 'r') {
        fetchData();
      } else if (e.key.toLowerCase() === 'f') {
        setFullscreenChart(prev => !prev);
      } else if (e.key === '?') {
        setShowHotkeysModal(prev => !prev);
      } else if (e.key === 'Escape') {
        setShowHotkeysModal(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
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

  const exportTradeLogCSV = () => {
    if (!tradeLog.length) return;
    const headers = ['Date', 'Time', 'Ticker', 'Direction', 'Entry', 'Exit', 'Qty', 'Gross_PnL', 'Status', 'Note'];
    const rows = tradeLog.map(t => [
      `"${t.date || ''}"`,
      `"${t.time || ''}"`,
      `"${t.ticker || ''}"`,
      `"${t.direction || ''}"`,
      t.entry || '',
      t.exit || '',
      t.qty || '',
      t.grossPnl !== null ? t.grossPnl : '',
      `"${t.status || ''}"`,
      `"${(t.note || '').replace(/"/g, '""')}"`
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `trade_log_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportBlockDealsCSV = () => {
    if (!blockDeals) return;
    const deals = [
      ...(blockDeals.block_deals || []).map(d => ({ ...d, type: 'BLOCK' })),
      ...(blockDeals.bulk_deals || []).map(d => ({ ...d, type: 'BULK' }))
    ];
    if (!deals.length) return;

    const headers = ['Symbol', 'Deal_Type', 'Trade_Type', 'Client', 'Quantity', 'Price'];
    const rows = deals.map(d => [
      `"${d.symbol || ''}"`,
      `"${d.type || ''}"`,
      `"${d.trade_type === 'B' || d.trade_type === 'BUY' ? 'BUY' : 'SELL'}"`,
      `"${(d.client || 'Undisclosed').replace(/"/g, '""')}"`,
      d.quantity ?? '',
      d.price ?? d.avg_price ?? ''
    ]);

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `nse_block_bulk_deals_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
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
      changeTicker(sym);
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
  const slicedCandles = useMemo(() => {
    if (!rawCandles.length) return [];
    if (candleSlice === '30') return rawCandles.slice(-30);
    if (candleSlice === '60') return rawCandles.slice(-60);
    return rawCandles;
  }, [rawCandles, candleSlice]);

  const candles = useMemo(() => {
    if (!slicedCandles.length) return [];
    if (candleMode !== 'heikin_ashi') return slicedCandles;

    let prevHaOpen = null;
    let prevHaClose = null;
    return slicedCandles.map((c, idx) => {
      const haClose = (c.open + c.high + c.low + c.close) / 4;
      const haOpen = idx === 0 ? (c.open + c.close) / 2 : (prevHaOpen + prevHaClose) / 2;
      const haHigh = Math.max(c.high, haOpen, haClose);
      const haLow = Math.min(c.low, haOpen, haClose);
      prevHaOpen = haOpen;
      prevHaClose = haClose;

      return {
        ...c,
        open: Number(haOpen.toFixed(2)),
        high: Number(haHigh.toFixed(2)),
        low: Number(haLow.toFixed(2)),
        close: Number(haClose.toFixed(2)),
        isHeikinAshi: true,
        realOpen: c.open,
        realHigh: c.high,
        realLow: c.low,
        realClose: c.close,
      };
    });
  }, [slicedCandles, candleMode]);

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
      if (c.low > 0 && c.low < min) min = c.low;
      if (c.high > 0 && c.high > max) max = c.high;
      if (showVWAPBands) {
        if (c.lower_band_2 > 0 && c.lower_band_2 < min) min = c.lower_band_2;
        if (c.upper_band_2 > 0 && c.upper_band_2 > max) max = c.upper_band_2;
      }
      if (showSupertrend && c.supertrend > 0) {
        if (c.supertrend < min) min = c.supertrend;
        if (c.supertrend > max) max = c.supertrend;
      }
      if (showEMA200 && c.ema200 > 0) {
        if (c.ema200 < min) min = c.ema200;
        if (c.ema200 > max) max = c.ema200;
      }
    });

    if (data?.current_price > 0) {
      if (data.current_price < min) min = data.current_price;
      if (data.current_price > max) max = data.current_price;
    }

    if (showPDH && data?.pivots?.daily_levels) {
      const { pdh, pdl } = data.pivots.daily_levels;
      if (pdh > 0 && pdh > max) max = pdh;
      if (pdl > 0 && pdl < min) min = pdl;
    }

    if (showORB && data?.orb) {
      const { high_15m, low_15m } = data.orb;
      if (high_15m > 0 && high_15m > max) max = high_15m;
      if (low_15m > 0 && low_15m < min) min = low_15m;
    }

    if (showCamarilla && data?.pivots?.camarilla) {
      const { h4, l4 } = data.pivots.camarilla;
      if (h4 > 0 && h4 > max) max = h4;
      if (l4 > 0 && l4 < min) min = l4;
    }

    if (!isFinite(min) || !isFinite(max) || min <= 0) {
      min = candles[0]?.open || 100;
      max = min * 1.05;
    }

    const buffer = (max - min) * 0.05 || 1;
    min -= buffer;
    max += buffer;

    const innerW = chartWidth - padding.left - padding.right;
    const innerH = chartHeight - padding.top - padding.bottom;

    const xs = (idx) => padding.left + (idx / Math.max(candles.length - 1, 1)) * innerW;
    const ys = (val) => padding.top + innerH - ((val - min) / Math.max(max - min, 1e-6)) * innerH;
    const cw = Math.max(2, Math.min(22, (innerW / candles.length) * 0.7));

    return { priceMin: min, priceMax: max, xScale: xs, yScale: ys, candleWidth: cw };
  }, [candles, showVWAPBands, showSupertrend, showPDH, showORB, showCamarilla, data]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex flex-col justify-between">
      <div>
        <Header
          currentTicker={ticker}
          onTickerSelect={(sym) => {
            if (!sym) return;
            changeTicker(sym);
          }}
        />

        <main className="w-full max-w-7xl mx-auto p-4 sm:p-6 lg:p-8 space-y-6">

        {/* ── REAL-TIME MARKET SESSION CLOCK & PHASE BANNER ─────────────────── */}
        {marketPulse && (
          <div className="bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800/80 rounded-2xl p-3 sm:p-4 backdrop-blur-md flex flex-col gap-3 shadow-xl shadow-black/40">
            {/* Top row: Clock on left, Benchmark indices on right */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 w-full">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-xl flex items-center justify-center shrink-0 ${marketPulse.is_open ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' : 'bg-amber-500/15 text-amber-400 border border-amber-500/30'}`}>
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
                  <p className="text-xs font-bold text-white mt-0.5 flex items-center gap-1.5 flex-wrap">
                    <span>{marketPulse.phase_name}</span>
                    <span className="text-slate-500">—</span>
                    <span className="text-slate-300 font-normal">{marketPulse.directive}</span>
                  </p>
                </div>
              </div>

              {/* Benchmark Indices Pills */}
              <div className="flex items-center gap-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0 scrollbar-none shrink-0">
                {marketPulse.indices?.map((idx) => {
                  const isVix = idx.name.includes('VIX');
                  return (
                    <div
                      key={idx.symbol}
                      className={`px-2.5 py-1 rounded-xl text-xs font-mono shrink-0 flex items-center gap-1.5 ${
                        isVix
                          ? 'bg-purple-950/40 border border-purple-500/40 text-purple-300'
                          : 'bg-slate-950/80 border border-slate-800 text-white'
                      }`}
                    >
                      <span className="text-slate-400 font-semibold">{idx.name}:</span>
                      <span className="font-bold">{idx.price?.toLocaleString()}</span>
                      <span className={`text-[11px] font-bold ${idx.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {idx.change_pct >= 0 ? '+' : ''}{idx.change_pct}%
                      </span>
                      {isVix && marketPulse.vix?.regime && (
                        <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded uppercase ${
                          marketPulse.vix.regime === 'LOW' ? 'bg-emerald-500/20 text-emerald-300' :
                          marketPulse.vix.regime === 'NORMAL' ? 'bg-cyan-500/20 text-cyan-300' : 'bg-rose-500/20 text-rose-300'
                        }`}>
                          {marketPulse.vix.regime}
                        </span>
                      )}
                    </div>
                  );
                })}

                {marketPulse.mins_to_mis_squareoff > 0 && (
                  <div className="px-3 py-1 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono shrink-0 flex items-center gap-1.5 font-bold">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    {scannerMarket === 'IN' ? 'Auto-Square-Off' : 'Market Close'} in: {marketPulse.mins_to_mis_squareoff}m
                  </div>
                )}
              </div>
            </div>

            {/* Live Sectoral Heatmap Flow Strip (Spanning full width) */}
            {marketPulse.sectors && marketPulse.sectors.length > 0 && (
              <div className="flex items-center gap-2 overflow-x-auto w-full pt-2.5 border-t border-slate-800/60 scrollbar-none text-[11px] font-mono">
                <div className="flex items-center gap-1.5 text-slate-500 uppercase tracking-wider font-sans font-bold text-[10px] shrink-0 pr-1">
                  <Flame className="w-3.5 h-3.5 text-amber-400" />
                  <span>Sector Flow:</span>
                </div>
                {marketPulse.sectors.map((sec) => (
                  <div
                    key={sec.symbol}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-950/70 border border-slate-800/80 shrink-0"
                  >
                    <span className="text-slate-300 font-medium font-sans">{sec.name.replace('NIFTY ', '')}</span>
                    <span className={`font-bold ${sec.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {sec.change_pct >= 0 ? '+' : ''}{sec.change_pct}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── TOP TERMINAL BAR ────────────────────────────────────────────── */}
        <header className="flex flex-col xl:flex-row xl:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div>
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-tr from-emerald-500/20 to-cyan-500/20 border border-emerald-500/40 rounded-2xl shadow-sm shadow-emerald-500/10">
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
                  Real-world session clocks, gap intelligence, trap detectors &amp; institutional friction calculators
                </p>
              </div>
            </div>
          </div>

          {/* Quick controls: Grouped Segmented Control Pods */}
          <div className="flex flex-wrap items-center gap-2">
            {/* Market Switcher */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs font-semibold">
              <button
                onClick={() => { setScannerMarket('IN'); changeTicker('RELIANCE.NS'); }}
                className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1 ${scannerMarket === 'IN' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 shadow-sm shadow-emerald-500/10 font-bold' : 'text-slate-400 hover:text-white'}`}
              >
                <span>🇮🇳</span>
                <span>NSE / BSE</span>
              </button>
              <button
                onClick={() => { setScannerMarket('US'); changeTicker('NVDA'); }}
                className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1 ${scannerMarket === 'US' ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10 font-bold' : 'text-slate-400 hover:text-white'}`}
              >
                <span>🇺🇸</span>
                <span>NYSE / NASDAQ</span>
              </button>
            </div>

            {/* Auto-Refresh Control Pod */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs">
              <div className="flex items-center gap-1.5 px-2 py-1 text-slate-400">
                <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : 'text-slate-400'}`} />
                <span className="text-[11px] font-semibold text-slate-400">Auto:</span>
                <select
                  value={autoRefreshSecs}
                  onChange={(e) => setAutoRefreshSecs(Number(e.target.value))}
                  className="bg-transparent text-white font-mono text-xs focus:outline-none cursor-pointer"
                >
                  <option value={10} className="bg-slate-900">10s (Fast)</option>
                  <option value={15} className="bg-slate-900">15s</option>
                  <option value={30} className="bg-slate-900">30s</option>
                  <option value={60} className="bg-slate-900">60s</option>
                  <option value={0} className="bg-slate-900">Paused</option>
                </select>
                {autoRefreshSecs > 0 && (
                  <span className="text-[10px] font-mono font-bold text-cyan-400 w-5 text-center bg-cyan-500/10 rounded px-1">
                    {refreshCountdown}s
                  </span>
                )}
              </div>
              <button
                onClick={() => fetchData(false)}
                disabled={loading}
                className="p-1.5 bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 hover:text-white rounded-lg transition"
                title="Force Refresh Data Now"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
              </button>
            </div>

            {/* Desk Utilities Toolbar */}
            <div className="flex items-center bg-slate-900 border border-slate-800 rounded-xl p-1 text-xs gap-1">
              {/* Audio Alerts Toggle */}
              <button
                onClick={() => {
                  const next = !soundAlerts;
                  setSoundAlerts(next);
                  if (next) {
                    try {
                      const ctx = getAudioContext();
                      if (ctx && ctx.state === 'running') {
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
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition font-medium ${
                  soundAlerts
                    ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title={soundAlerts ? 'Audio Alerts: ACTIVE (Click to Mute)' : 'Audio Alerts: MUTED (Click to Enable Synthesizer Chimes)'}
              >
                {soundAlerts ? <Volume2 className="w-3.5 h-3.5 text-cyan-400" /> : <VolumeX className="w-3.5 h-3.5" />}
                <span className="hidden sm:inline">{soundAlerts ? 'Audio' : 'Muted'}</span>
              </button>

              {/* Trader's Scratchpad Toggle */}
              <button
                onClick={() => setScratchpadOpen(!scratchpadOpen)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition font-medium ${
                  scratchpadOpen
                    ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
                title="Open Trader's Real-Time Execution Notepad & Journal"
              >
                <Edit3 className="w-3.5 h-3.5 text-amber-400" />
                <span className="hidden sm:inline">Journal</span>
              </button>

              {/* Pro Hotkeys Modal Button */}
              <button
                onClick={() => setShowHotkeysModal(true)}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition font-medium"
                title="Pro Keyboard Shortcuts (Press '?')"
              >
                <Keyboard className="w-3.5 h-3.5 text-cyan-400" />
                <kbd className="px-1 py-0.2 rounded bg-slate-800 text-[10px] text-cyan-300 font-mono">?</kbd>
              </button>

              {/* Export CSV */}
              <button
                onClick={handleExportIntradayCSV}
                disabled={!data?.candles?.length}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-cyan-300 disabled:opacity-40 transition font-medium cursor-pointer"
                title="Download Intraday Candles & Technical Indicators as CSV"
              >
                <Download className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">CSV</span>
              </button>
            </div>
          </div>
        </header>

        {/* ── TICKER COMMAND BAR & POPULAR SHORTCUTS ───────────────────────── */}
        <div className="space-y-2.5 bg-slate-900/60 border border-slate-800/80 rounded-2xl p-3 backdrop-blur-md shadow-md shadow-black/20">
          {/* Pinned Watchlist Strip (If Available) */}
          {pinnedTickers.length > 0 && (
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs border-b border-slate-800/60 scrollbar-none">
              <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider pl-1 pr-1 shrink-0 flex items-center gap-1">
                <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                Pinned Desk:
              </span>
              {pinnedTickers.map(sym => (
                <div
                  key={sym}
                  className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold shrink-0 transition border ${
                    ticker === sym
                      ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-sm shadow-amber-500/10'
                      : 'bg-slate-950/70 text-slate-300 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <button
                    onClick={() => changeTicker(sym)}
                    className="cursor-pointer font-mono text-[11px]"
                  >
                    {sym.split('.')[0]}
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); togglePinTicker(sym); }}
                    className="text-slate-500 hover:text-rose-400 ml-1"
                    title="Unpin from desk"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0 scrollbar-none">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider pl-1 pr-1 shrink-0 font-mono">
                Active Desk:
              </span>
              {QUICK_TICKERS.filter(t => t.market === scannerMarket).map(t => (
                <button
                  key={t.symbol}
                  onClick={() => changeTicker(t.symbol)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-semibold shrink-0 transition flex items-center gap-1.5 ${
                    ticker === t.symbol
                      ? 'bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/10'
                      : 'bg-slate-950/70 hover:bg-slate-800/80 text-slate-400 hover:text-slate-200 border border-slate-800/80'
                  }`}
                >
                  <span>{t.name}</span>
                  <span className="text-[10px] text-slate-500 font-mono">({t.symbol.split('.')[0]})</span>
                </button>
              ))}
            </div>

            <form onSubmit={handleSearchSubmit} className="relative min-w-[260px]">
              <input
                ref={searchInputRef}
                type="text"
                placeholder={`Search ${scannerMarket === 'IN' ? 'NSE stock (e.g. SBIN)' : 'US stock (e.g. AMD)'}... (Press '/')`}
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700/80 rounded-xl pl-9 pr-14 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition font-mono"
              />
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <kbd className="absolute right-3 top-2 px-1.5 py-0.5 rounded bg-slate-800/80 text-[10px] text-slate-400 font-mono border border-slate-700">
                /
              </kbd>
            </form>
          </div>
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
            {/* 1. Price & Change — with flash animation on tick update */}
            <div className={`bg-gradient-to-br from-slate-900/90 to-slate-900/50 rounded-2xl p-3.5 sm:p-4 flex flex-col justify-between transition-all duration-300 shadow-md ${
              priceFlash === 'up'
                ? 'border border-emerald-400/60 shadow-emerald-500/20'
                : priceFlash === 'down'
                ? 'border border-rose-400/60 shadow-rose-500/20'
                : 'border border-slate-800/80 shadow-black/30'
            }`}>
              <div>
                <div className="flex items-center justify-between gap-1">
                  <div className="flex items-center gap-1.5 truncate max-w-[125px]">
                    <button
                      onClick={() => togglePinTicker(ticker)}
                      className={`p-0.5 rounded transition ${
                        pinnedTickers.includes(ticker)
                          ? 'text-amber-400 hover:text-amber-300'
                          : 'text-slate-600 hover:text-slate-400'
                      }`}
                      title={pinnedTickers.includes(ticker) ? 'Unpin from desk' : 'Pin to my desk'}
                    >
                      <Star className={`w-3.5 h-3.5 ${pinnedTickers.includes(ticker) ? 'fill-amber-400 text-amber-400' : ''}`} />
                    </button>
                    <span className="text-xs font-semibold text-slate-300 truncate" title={data.company_name}>{data.company_name}</span>
                  </div>
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
                        <span className={`text-[9px] font-bold font-mono px-1.5 py-0.5 rounded border ${
                          unrealized >= 0 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                        }`}>
                          {unrealized >= 0 ? '+' : ''}{currSym}{Math.round(unrealized)}
                        </span>
                      );
                    })()}
                  </div>
                </div>
              </div>

              {/* Bottom Telemetry: O/H/L & Day Range */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                  <span>O: <strong className="text-slate-200">{currSym}{data.open}</strong></span>
                  <span>H: <strong className="text-emerald-400">{currSym}{data.high}</strong></span>
                  <span>L: <strong className="text-rose-400">{currSym}{data.low}</strong></span>
                </div>
                {data.high > data.low && data.current_price ? (
                  <div className="mt-1.5 space-y-0.5">
                    <div className="w-full bg-slate-800/80 h-1 rounded-full overflow-hidden relative">
                      <div className="h-full bg-gradient-to-r from-rose-500 via-amber-400 to-emerald-500 w-full" />
                      <div
                        className="absolute top-0 bottom-0 w-1.5 bg-white rounded-full shadow-sm ring-1 ring-white/60"
                        style={{
                          left: `${Math.max(0, Math.min(97, ((data.current_price - data.low) / (data.high - data.low)) * 100))}%`
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[8px] text-slate-500 font-mono">
                      <span>LOD</span>
                      <span className="text-slate-400 font-semibold">
                        {Math.round(((data.current_price - data.low) / (data.high - data.low)) * 100)}% of Range
                      </span>
                      <span>HOD</span>
                    </div>
                  </div>
                ) : (
                  <div className="mt-1.5 flex justify-between text-[8px] text-slate-500 font-mono">
                    <span>Session Open</span>
                    <span className="text-slate-400">Regular Trading</span>
                    <span>Close</span>
                  </div>
                )}
              </div>
            </div>

            {/* 2. Session VWAP */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-3.5 sm:p-4 flex flex-col justify-between shadow-md shadow-black/30">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                    Session VWAP
                    <InfoBadge infoKey="vwap" />
                  </span>
                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                    data.current_price >= data.vwap
                      ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
                      : 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                  }`}>
                    {data.current_price >= data.vwap ? 'ABOVE' : 'BELOW'}
                  </span>
                </div>
                <div className="mt-1">
                  <p className="text-xl sm:text-2xl font-black font-mono tracking-tight text-cyan-300">
                    {currSym}{data.vwap?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5 font-mono">
                    Dev: <span className={`font-bold ${data.current_price >= data.vwap ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {data.current_price >= data.vwap ? '+' : ''}
                      {(((data.current_price - data.vwap) / data.vwap) * 100).toFixed(2)}%
                    </span>
                  </p>
                </div>
              </div>

              {/* Bottom Telemetry: VWAP ±2σ Bands */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                  <span>-2σ: <strong className="text-cyan-400">{currSym}{data.vwap_bands?.lower_2 ? Number(data.vwap_bands.lower_2).toFixed(1) : '—'}</strong></span>
                  <span>+2σ: <strong className="text-cyan-400">{currSym}{data.vwap_bands?.upper_2 ? Number(data.vwap_bands.upper_2).toFixed(1) : '—'}</strong></span>
                </div>
                <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                  <span>Lower Vol Band</span>
                  <span className="text-cyan-400/80 font-semibold">Institutional Mean</span>
                  <span>Upper Vol Band</span>
                </div>
              </div>
            </div>

            {/* 3. Relative Strength vs Benchmark */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-3.5 sm:p-4 flex flex-col justify-between shadow-md shadow-black/30">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                    Relative Strength
                    <InfoBadge infoKey="benchmark_relative_strength" />
                  </span>
                  <span className="text-[10px] font-mono text-slate-400 font-bold px-1.5 py-0.5 bg-slate-950/80 border border-slate-800 rounded">
                    vs {data.relative_strength?.benchmark_name?.replace('NIFTY ', '') || 'Index'}
                  </span>
                </div>
                <div className="mt-1">
                  <p className={`text-xl sm:text-2xl font-black font-mono tracking-tight ${data.relative_strength?.alpha_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {data.relative_strength?.alpha_pct >= 0 ? '+' : ''}{data.relative_strength?.alpha_pct}%
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5 truncate font-mono">
                    Alpha: <span className="font-semibold text-slate-200">{data.relative_strength?.status || 'Neutral'}</span>
                  </p>
                </div>
              </div>

              {/* Bottom Telemetry: Benchmark Index Move & Regime */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                  <span>Idx: <strong className={data.relative_strength?.benchmark_change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{data.relative_strength?.benchmark_change_pct >= 0 ? '+' : ''}{data.relative_strength?.benchmark_change_pct}%</strong></span>
                  <span className="text-slate-300 truncate max-w-[85px]">{data.relative_strength?.regime || 'Tracking'}</span>
                </div>
                <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                  <span>Lagging</span>
                  <span className={data.relative_strength?.alpha_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                    {data.relative_strength?.alpha_pct >= 0 ? 'Outperforming' : 'Underperforming'}
                  </span>
                  <span>Leading</span>
                </div>
              </div>
            </div>

            {/* 4. Pre-Market Gap Intelligence */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-3.5 sm:p-4 flex flex-col justify-between shadow-md shadow-black/30">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                    Pre-Market Gap
                    <InfoBadge infoKey="pre_market_gap" />
                  </span>
                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                    data.gap_analysis?.gap_pct >= 0 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                  }`}>
                    {data.gap_analysis?.gap_type?.replace(/_/g, ' ') || 'FLAT'}
                  </span>
                </div>
                <div className="mt-1">
                  <p className={`text-xl sm:text-2xl font-black font-mono tracking-tight ${data.gap_analysis?.gap_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {data.gap_analysis?.gap_pct >= 0 ? '+' : ''}{data.gap_analysis?.gap_pct}%
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5 font-mono">
                    Points: <span className="font-semibold text-slate-200">{currSym}{data.gap_analysis?.gap_pts}</span>
                  </p>
                </div>
              </div>

              {/* Bottom Telemetry: Prev Close & Gap Fill Status */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                  <span>Prev: <strong className="text-slate-300">{currSym}{data.gap_analysis?.prev_close}</strong></span>
                  <span>Fill: <strong className={data.gap_analysis?.gap_filled ? 'text-emerald-400' : 'text-amber-400'}>
                    {data.gap_analysis?.gap_filled ? 'FILLED' : `OPEN (${currSym}${data.gap_analysis?.gap_fill_dist})`}
                  </strong></span>
                </div>
                <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                  <span>Gap Origin</span>
                  <span className="text-amber-400 truncate max-w-[110px]">{data.gap_analysis?.directive?.split('—')[0] || 'Gap Setup'}</span>
                  <span>PDC</span>
                </div>
              </div>
            </div>

            {/* 5. Composite Quant Bias Score */}
            <div className="bg-gradient-to-br from-slate-900/90 to-slate-900/50 border border-slate-800/80 rounded-2xl p-3.5 sm:p-4 flex flex-col justify-between shadow-md shadow-black/30">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                    Quant Bias
                    <InfoBadge infoKey="intraday_quant_score" />
                  </span>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${
                    data.signals.overall_bias.includes('BUY')
                      ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                      : data.signals.overall_bias.includes('SELL')
                      ? 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                      : 'bg-amber-500/15 text-amber-400 border-amber-500/30'
                  }`}>
                    {data.signals.overall_bias}
                  </span>
                </div>
                <div className="mt-1">
                  <div className="flex items-baseline gap-1">
                    <span className={`text-xl sm:text-2xl font-black font-mono tracking-tight ${data.signals.quant_score >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {data.signals.quant_score >= 0 ? '+' : ''}{data.signals.quant_score}
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono">/ 100</span>
                  </div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1.5">
                    <div
                      className={`h-full transition-all duration-500 ${data.signals.quant_score >= 0 ? 'bg-emerald-400' : 'bg-rose-400'}`}
                      style={{ width: `${Math.abs(data.signals.quant_score)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Bottom Telemetry: Signal Confluence Counter */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                  <span>Bull: <strong className="text-emerald-400">{data.signals?.bullish_count || 0}</strong></span>
                  <span>Bear: <strong className="text-rose-400">{data.signals?.bearish_count || 0}</strong></span>
                  <span>Conf: <strong className="text-cyan-300">{Math.round((data.signals?.bullish_count || 0) / Math.max(1, (data.signals?.bullish_count || 0) + (data.signals?.bearish_count || 0)) * 100)}%</strong></span>
                </div>
                <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                  <span>Bearish</span>
                  <span className="text-slate-400">Multi-Model Engine</span>
                  <span>Bullish</span>
                </div>
              </div>
            </div>

            {/* 6. Supertrend & Momentum Signal */}
            <div className="bg-slate-900/80 border border-slate-800/80 rounded-2xl p-3.5 sm:p-4 flex flex-col justify-between shadow-md shadow-black/30">
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                    Supertrend
                    <InfoBadge infoKey="supertrend" />
                  </span>
                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${
                    data.supertrend_dir === 1 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                  }`}>
                    {data.supertrend_dir === 1 ? '▲ BULL' : '▼ BEAR'}
                  </span>
                </div>
                <div className="mt-1">
                  <p className={`text-xl sm:text-2xl font-black font-mono tracking-tight ${data.supertrend_dir === 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {currSym}{data.supertrend?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5 font-mono">
                    RSI:{' '}
                    <span className={`font-bold ${data.rsi >= 70 ? 'text-rose-400' : data.rsi <= 30 ? 'text-emerald-400' : 'text-slate-200'}`}>
                      {data.rsi?.toFixed(1)}
                    </span>
                    {data.rsi >= 70 && <span className="text-rose-400 ml-1 text-[9px] font-bold">OB</span>}
                    {data.rsi <= 30 && <span className="text-emerald-400 ml-1 text-[9px] font-bold">OS</span>}
                  </p>
                </div>
              </div>

              {/* Bottom Telemetry: ATR & Moving Average Regime */}
              <div>
                <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-slate-800/60 font-mono">
                  <span>ATR: <strong className="text-amber-400">{currSym}{data.atr ? Number(data.atr).toFixed(1) : '—'}</strong></span>
                  <span>EMA: <strong className={data.ema9 > data.ema21 ? 'text-emerald-400' : 'text-rose-400'}>{data.ema9 > data.ema21 ? '9>21 Bull' : '9<21 Bear'}</strong></span>
                </div>
                <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                  <span>Volatility Anchor</span>
                  <span className="text-slate-400">Trailing Level</span>
                </div>
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
              {/* Upper Chart Control Ribbon */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3.5 border-b border-slate-800">
                <div className="flex flex-wrap items-center gap-2">
                  {/* Timeframe Selector Pills */}
                  <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800/80">
                    {TIMEFRAMES.map((tf) => (
                      <button
                        key={tf.label}
                        onClick={() => { setCandleInterval(tf.interval); setPeriod(tf.period); }}
                        className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition ${
                          candleInterval === tf.interval
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {tf.label}
                      </button>
                    ))}
                  </div>

                  {/* Viewport Zoom */}
                  <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800/80">
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500 px-1.5 font-mono">Zoom</span>
                    {[
                      { id: 'all', label: 'All Day' },
                      { id: '60', label: '60b' },
                      { id: '30', label: '30b' },
                    ].map((z) => (
                      <button
                        key={z.id}
                        onClick={() => setCandleSlice(z.id)}
                        className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg transition ${
                          candleSlice === z.id
                            ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm shadow-purple-500/10'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {z.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Right controls: Price Alert & Fullscreen */}
                <div className="flex items-center gap-2">
                  {/* Price Alert mini widget */}
                  {data && (
                    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-xl border text-xs ${
                      alertTriggered
                        ? 'bg-amber-500/15 border-amber-500/40 text-amber-300 shadow-sm shadow-amber-500/20'
                        : 'bg-slate-950 border-slate-800 text-slate-400'
                    }`}>
                      {alertTriggered
                        ? <Bell className="w-3.5 h-3.5 text-amber-400 animate-bounce" />
                        : <BellOff className="w-3.5 h-3.5 text-slate-500" />}
                      <select
                        value={alertAbove ? 'above' : 'below'}
                        onChange={e => { setAlertAbove(e.target.value === 'above'); setAlertTriggered(false); }}
                        className="bg-transparent text-[11px] font-mono focus:outline-none cursor-pointer text-slate-300"
                      >
                        <option value="above" className="bg-slate-900">Alert ≥</option>
                        <option value="below" className="bg-slate-900">Alert ≤</option>
                      </select>
                      <input
                        type="number"
                        placeholder={data.current_price?.toFixed(0)}
                        value={alertPrice}
                        onChange={e => { setAlertPrice(e.target.value); setAlertTriggered(false); }}
                        className="w-16 bg-transparent font-mono text-xs text-white placeholder-slate-600 focus:outline-none"
                      />
                    </div>
                  )}

                  <button
                    onClick={() => setFullscreenChart(!fullscreenChart)}
                    className="p-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-400 hover:text-white transition"
                    title={fullscreenChart ? 'Exit Fullscreen' : 'Fullscreen Chart'}
                  >
                    {fullscreenChart ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* Dedicated Overlays & Indicators Ribbon */}
              <div className="my-3 p-2 bg-slate-950/70 border border-slate-800/80 rounded-2xl flex flex-wrap items-center justify-between gap-2.5">
                {/* Left Group: Technical Indicators */}
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="text-[10px] uppercase font-mono font-bold tracking-wider text-slate-500 pl-1 pr-1">
                    Overlays:
                  </span>
                  <button
                    onClick={() => setShowVWAP(!showVWAP)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showVWAP ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full" />
                    VWAP
                  </button>

                  <button
                    onClick={() => setShowVWAPBands(!showVWAPBands)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showVWAPBands ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-cyan-300/60 rounded-full" />
                    ±2σ Bands
                  </button>

                  <button
                    onClick={() => setShowSupertrend(!showSupertrend)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showSupertrend ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm shadow-emerald-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                    Supertrend
                  </button>

                  <button
                    onClick={() => setShowEMA(!showEMA)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showEMA ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm shadow-purple-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-purple-400 rounded-full" />
                    EMA 9/21
                  </button>

                  <button
                    onClick={() => setShowEMA200(!showEMA200)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showEMA200 ? 'bg-amber-400/20 text-amber-300 border border-amber-400/40 shadow-sm shadow-amber-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                    title="200-period Exponential Moving Average (Institutional Anchor)"
                  >
                    <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                    200 EMA
                  </button>
                </div>

                {/* Right Group: Key Levels & Heikin-Ashi */}
                <div className="flex flex-wrap items-center gap-1.5 text-xs">
                  <span className="text-[10px] uppercase font-mono font-bold tracking-wider text-slate-500 pl-1 pr-1">
                    Levels:
                  </span>
                  <button
                    onClick={() => setShowORB(!showORB)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showORB ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm shadow-amber-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                    ORB 15m
                  </button>

                  <button
                    onClick={() => setShowCamarilla(!showCamarilla)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showCamarilla ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-sm shadow-rose-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-rose-400 rounded-full" />
                    Camarilla
                  </button>

                  <button
                    onClick={() => setShowPDH(!showPDH)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showPDH ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm shadow-amber-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-amber-400 rounded-full" />
                    PDH / PDL
                  </button>

                  <button
                    onClick={() => setShowCPR(!showCPR)}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      showCPR ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 shadow-sm shadow-indigo-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <span className="w-1.5 h-1.5 bg-indigo-400 rounded-full" />
                    CPR
                  </button>

                  <div className="w-px h-4 bg-slate-800 mx-1 hidden sm:block" />

                  <button
                    onClick={() => setCandleMode(candleMode === 'regular' ? 'heikin_ashi' : 'regular')}
                    className={`px-2.5 py-1 rounded-lg font-semibold transition flex items-center gap-1.5 ${
                      candleMode === 'heikin_ashi' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10' : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                    }`}
                    title="Toggle Heikin-Ashi Trend-Smoothing Candlesticks (HotKey: 'K')"
                  >
                    <span className="text-[11px]">🥢</span>
                    {candleMode === 'heikin_ashi' ? 'Heikin-Ashi' : 'Candles'}
                  </button>
                </div>
              </div>

              {/* Hover Inspection Bar — with fixed height and live default to prevent jitter */}
              <div className="h-7 flex items-center justify-between text-[11px] font-mono text-slate-400 px-2.5 bg-slate-950/40 rounded-xl border border-slate-800/40 overflow-x-auto scrollbar-none">
                {hoveredCandle || (candles.length > 0 ? candles[candles.length - 1] : null) ? (() => {
                  const c = hoveredCandle || candles[candles.length - 1];
                  const isLive = !hoveredCandle;
                  return (
                    <div className="flex items-center gap-3 w-full justify-between shrink-0">
                      <div className="flex items-center gap-3">
                        {isLive ? (
                          <span className="px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-bold flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> LIVE
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-300 text-[10px] font-bold">
                            INSPECT
                          </span>
                        )}
                        {candleMode === 'heikin_ashi' && (
                          <span className="px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-300 text-[10px] font-bold">
                            HA SMOOTHED
                          </span>
                        )}
                        <span>Time: <strong className="text-white">{c.time}</strong></span>
                        <span>O: <strong className="text-slate-200">{c.open}</strong></span>
                        <span>H: <strong className="text-emerald-400">{c.high}</strong></span>
                        <span>L: <strong className="text-rose-400">{c.low}</strong></span>
                        <span>C: <strong className={c.close >= c.open ? 'text-emerald-400' : 'text-rose-400'}>{c.close}</strong></span>
                        <span>Vol: <strong className="text-cyan-300">{c.volume?.toLocaleString()}</strong></span>
                        {c.vwap && <span>VWAP: <strong className="text-cyan-400">{c.vwap}</strong></span>}
                        {c.ema200 > 0 && <span>EMA200: <strong className="text-amber-400">{c.ema200}</strong></span>}
                        {c.atr > 0 && <span>ATR: <strong className="text-amber-300">{c.atr}</strong></span>}
                      </div>
                      <div className="hidden md:flex items-center text-[10px] text-slate-500">
                        {isLive ? 'Hover candles to inspect' : 'Crosshair active'}
                      </div>
                    </div>
                  );
                })() : (
                  <span className="text-slate-500 italic text-[10px]">Awaiting high-frequency market stream...</span>
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
                  <div className="h-72 flex flex-col items-center justify-center p-6 text-center space-y-3">
                    <div className="flex items-center gap-2 text-rose-400 text-sm font-semibold">
                      <AlertCircle className="w-5 h-5 shrink-0" />
                      <span>{error}</span>
                    </div>
                    <p className="text-xs text-slate-400 max-w-md">
                      Intraday chart data may be temporarily unavailable for {ticker} (e.g. market closed, corporate restructuring, or no trades). Select an active liquid stock to continue:
                    </p>
                    <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                      {['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'TATASTEEL.NS', 'NVDA', 'AAPL'].map(sym => (
                        <button
                          key={sym}
                          onClick={() => changeTicker(sym)}
                          className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-700 hover:border-cyan-500/50 text-xs font-mono font-bold text-slate-200 hover:text-cyan-400 transition"
                        >
                          {sym}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {!loading && !error && candles.length > 0 && (
                  <svg
                    viewBox={`0 0 ${chartWidth} ${chartHeight}`}
                    className="w-full h-auto cursor-crosshair select-none"
                    onMouseLeave={() => { setHoveredCandle(null); setHoveredX(null); setHoveredY(null); }}
                    onMouseMove={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const currentX = ((e.clientX - rect.left) / rect.width) * chartWidth;
                      const currentY = ((e.clientY - rect.top) / rect.height) * chartHeight;
                      setHoveredX(currentX);
                      setHoveredY(currentY);

                      const innerW = chartWidth - padding.left - padding.right;
                      const relX = currentX - padding.left;
                      const candleIdx = Math.round((relX / Math.max(innerW, 1)) * (candles.length - 1));
                      if (candleIdx >= 0 && candleIdx < candles.length) {
                        setHoveredCandle(candles[candleIdx]);
                      }
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

                    {/* Central Pivot Range (CPR: TC, Pivot, BC) */}
                    {showCPR && data?.pivots?.cpr && data.pivots.cpr.pivot > 0 && (
                      <g key="cpr-overlay">
                        {/* Shaded CPR Range Cloud */}
                        <rect
                          x={padding.left}
                          y={Math.min(yScale(data.pivots.cpr.tc), yScale(data.pivots.cpr.bc))}
                          width={chartWidth - padding.left - padding.right}
                          height={Math.max(1, Math.abs(yScale(data.pivots.cpr.tc) - yScale(data.pivots.cpr.bc)))}
                          fill="#6366f1"
                          fillOpacity={0.08}
                        />

                        {/* TC Line (Top Central) */}
                        <line
                          x1={padding.left}
                          y1={yScale(data.pivots.cpr.tc)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.pivots.cpr.tc)}
                          stroke="#818cf8"
                          strokeDasharray="3 3"
                          strokeWidth="1.1"
                          strokeOpacity={0.8}
                        />
                        <text
                          x={chartWidth - padding.right + 4}
                          y={yScale(data.pivots.cpr.tc) + 3}
                          fill="#818cf8"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          TC
                        </text>

                        {/* Central Pivot (P) */}
                        <line
                          x1={padding.left}
                          y1={yScale(data.pivots.cpr.pivot)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.pivots.cpr.pivot)}
                          stroke="#6366f1"
                          strokeWidth="1.4"
                          strokeOpacity={0.9}
                        />
                        <text
                          x={chartWidth - padding.right + 4}
                          y={yScale(data.pivots.cpr.pivot) + 3}
                          fill="#6366f1"
                          fontSize="8"
                          fontFamily="monospace"
                          fontWeight="bold"
                        >
                          CPR-P
                        </text>

                        {/* BC Line (Bottom Central) */}
                        <line
                          x1={padding.left}
                          y1={yScale(data.pivots.cpr.bc)}
                          x2={chartWidth - padding.right}
                          y2={yScale(data.pivots.cpr.bc)}
                          stroke="#a855f7"
                          strokeDasharray="3 3"
                          strokeWidth="1.1"
                          strokeOpacity={0.8}
                        />
                        <text
                          x={chartWidth - padding.right + 4}
                          y={yScale(data.pivots.cpr.bc) + 3}
                          fill="#a855f7"
                          fontSize="8"
                          fontFamily="monospace"
                        >
                          BC
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

                    {/* EMA 200 Institutional Anchor */}
                    {showEMA200 && (
                      <path
                        d={candles.reduce((acc, c, i) => !c.ema200 ? acc : `${acc}${acc ? ' L' : 'M'} ${xScale(i)} ${yScale(c.ema200)}`, '')}
                        fill="none"
                        stroke="#f59e0b"
                        strokeWidth="1.8"
                        strokeDasharray="4 3"
                        strokeOpacity={0.9}
                      />
                    )}

                    {/* Supertrend Stop Line */}
                    {showSupertrend && (
                      <g>
                        {candles.map((c, i) => {
                          if (i === 0 || !candles[i - 1].supertrend || !c.supertrend || candles[i - 1].supertrend_dir !== c.supertrend_dir) return null;
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

                    {/* Live Market Price Horizontal Line & Right Axis Badge */}
                    {data?.current_price && (() => {
                      const liveY = yScale(data.current_price);
                      if (liveY < padding.top || liveY > chartHeight - padding.bottom) return null;
                      const lastX = candles.length > 0 ? xScale(candles.length - 1) : chartWidth - padding.right;
                      const isUpDay = data.current_price >= (data.prev_close || data.candles?.[0]?.open || data.current_price);
                      const liveColor = isUpDay ? '#10b981' : '#f43f5e';
                      return (
                        <g pointerEvents="none" key="live-price-indicator">
                          {/* Pulsating radar ping beacon at the active candle tick */}
                          <circle cx={lastX} cy={liveY} r="7" fill={liveColor} fillOpacity="0.2">
                            <animate attributeName="r" values="3;9;3" dur="2s" repeatCount="indefinite" />
                            <animate attributeName="fill-opacity" values="0.6;0.1;0.6" dur="2s" repeatCount="indefinite" />
                          </circle>
                          <circle cx={lastX} cy={liveY} r="3" fill={liveColor} />

                          {/* Horizontal dashed price level line across the chart */}
                          <line
                            x1={padding.left}
                            y1={liveY}
                            x2={chartWidth - padding.right}
                            y2={liveY}
                            stroke={liveColor}
                            strokeWidth="1.2"
                            strokeDasharray="4 3"
                            strokeOpacity={0.85}
                          />

                          {/* Right Y-Axis Illuminated Price Badge */}
                          <rect
                            x={chartWidth - padding.right + 2}
                            y={liveY - 9}
                            width={padding.right - 4}
                            height={18}
                            rx={3}
                            fill={liveColor}
                          />
                          <text
                            x={chartWidth - padding.right + 5}
                            y={liveY + 3.5}
                            fill="#ffffff"
                            fontSize="8.5"
                            fontFamily="monospace"
                            fontWeight="bold"
                          >
                            {currSym}{data.current_price.toFixed(2)}
                          </text>
                        </g>
                      );
                    })()}

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
              <div className="mt-4 pt-4 border-t border-slate-800/80">
                <div className="flex flex-wrap items-center justify-between gap-2.5 mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Sub-Indicator:</span>
                    <div className="flex flex-wrap items-center bg-slate-950/90 p-1 rounded-xl border border-slate-800/80 text-xs shadow-inner gap-1">
                      <button
                        onClick={() => setActiveSubChart('volume')}
                        className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                          activeSubChart === 'volume'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-[0_0_12px_rgba(16,185,129,0.2)] font-bold'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        Volume &amp; Delta
                      </button>
                      <button
                        onClick={() => setActiveSubChart('rsi')}
                        className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                          activeSubChart === 'rsi'
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_12px_rgba(56,189,248,0.2)] font-bold'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        RSI (14)
                      </button>
                      <button
                        onClick={() => setActiveSubChart('macd')}
                        className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                          activeSubChart === 'macd'
                            ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-[0_0_12px_rgba(168,85,247,0.2)] font-bold'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        MACD (12,26,9)
                      </button>
                      <button
                        onClick={() => setActiveSubChart('cvd')}
                        className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                          activeSubChart === 'cvd'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-[0_0_12px_rgba(245,158,11,0.2)] font-bold'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        Order Flow CVD
                      </button>
                      <button
                        onClick={() => setActiveSubChart('atr')}
                        className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                          activeSubChart === 'atr'
                            ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40 shadow-[0_0_12px_rgba(249,115,22,0.2)] font-bold'
                            : 'text-slate-400 hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        ATR Volatility (14)
                      </button>
                    </div>
                    <InfoBadge infoKey={activeSubChart === 'cvd' ? 'order_flow_delta' : activeSubChart === 'volume' ? 'order_flow_delta' : activeSubChart === 'macd' ? 'macd_cross' : 'rsi'} />
                  </div>

                  <div className="hidden sm:flex items-center gap-2 text-[11px] font-mono text-slate-400">
                    <span className="px-2 py-0.5 rounded-md bg-slate-950 border border-slate-800/80 text-slate-300">
                      Active: <strong className="text-white uppercase">{activeSubChart}</strong>
                    </span>
                  </div>
                </div>

                <div className="h-44 sm:h-48 w-full bg-gradient-to-b from-slate-950/90 via-slate-950/70 to-slate-950/90 rounded-2xl border border-slate-800/80 p-2.5 overflow-hidden shadow-inner relative">
                  {activeSubChart === 'volume' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const maxVol = Math.max(...candles.map(c => c.volume || 1), 1);
                        const hasDelta = candles.some(c => (c.buyer_vol || 0) + (c.seller_vol || 0) > 0);
                        const lastCandle = candles[candles.length - 1];
                        const activeCandle = hoveredCandle || lastCandle;
                        return (
                          <>
                            {/* Horizontal guideline */}
                            <line x1={padding.left} y1={50} x2={chartWidth - padding.right} y2={50} stroke="#334155" strokeDasharray="3 3" strokeOpacity={0.25} />
                            <text x={padding.left + 4} y={14} fill="#94a3b8" fontSize="8" fontFamily="monospace" fontWeight="bold">
                              Vol: <tspan fill="#ffffff">{activeCandle?.volume?.toLocaleString() || 0}</tspan>
                              {hasDelta && (
                                <>
                                  <tspan fill="#64748b"> | </tspan>
                                  <tspan fill="#10b981">Buyers: {activeCandle?.buyer_vol?.toLocaleString() || 0}</tspan>
                                  <tspan fill="#64748b"> | </tspan>
                                  <tspan fill="#f43f5e">Sellers: {activeCandle?.seller_vol?.toLocaleString() || 0}</tspan>
                                </>
                              )}
                              {hoveredCandle && <tspan fill="#38bdf8">{` (${activeCandle?.time})`}</tspan>}
                            </text>
                            <text x={chartWidth - padding.right + 4} y={14} fill="#64748b" fontSize="8" fontFamily="monospace">
                              Max {(maxVol / 1000).toFixed(0)}K
                            </text>
                            {candles.map((c, i) => {
                              const x = xScale(i);
                              if (hasDelta) {
                                const bH = ((c.buyer_vol || 0) / maxVol) * 82;
                                const sH = ((c.seller_vol || 0) / maxVol) * 82;
                                return (
                                  <g key={i}>
                                    <rect x={x - candleWidth / 2} y={96 - bH} width={candleWidth / 2} height={bH} fill="#10b981" fillOpacity={0.85} rx={0.5} />
                                    <rect x={x} y={96 - sH} width={candleWidth / 2} height={sH} fill="#f43f5e" fillOpacity={0.85} rx={0.5} />
                                  </g>
                                );
                              }
                              const totalH = ((c.volume || 0) / maxVol) * 82;
                              const isUp = c.close >= c.open;
                              return <rect key={i} x={x - candleWidth / 2} y={96 - totalH} width={candleWidth} height={totalH} fill={isUp ? '#10b981' : '#f43f5e'} fillOpacity={0.7} rx={0.5} />;
                            })}
                          </>
                        );
                      })()}
                    </svg>
                  )}

                  {activeSubChart === 'rsi' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const lastCandle = candles[candles.length - 1];
                        const activeCandle = hoveredCandle || lastCandle;
                        const activeRsi = (activeCandle?.rsi !== undefined && activeCandle?.rsi !== null) ? activeCandle.rsi : 50;
                        return (
                          <>
                            {/* Overbought / Oversold Zones */}
                            <rect x={padding.left} y={10} width={chartWidth - padding.left - padding.right} height={20} fill="#f43f5e" fillOpacity={0.04} />
                            <line x1={padding.left} y1={30} x2={chartWidth - padding.right} y2={30} stroke="#f43f5e" strokeDasharray="3 3" strokeOpacity={0.6} />
                            <text x={padding.left + 4} y={26} fill="#f43f5e" fontSize="8" fontFamily="monospace" fontWeight="bold">OB 70</text>

                            <line x1={padding.left} y1={50} x2={chartWidth - padding.right} y2={50} stroke="#475569" strokeDasharray="2 2" strokeOpacity={0.4} />
                            <text x={padding.left + 4} y={48} fill="#64748b" fontSize="7" fontFamily="monospace">Mid 50</text>

                            <rect x={padding.left} y={70} width={chartWidth - padding.left - padding.right} height={25} fill="#10b981" fillOpacity={0.04} />
                            <line x1={padding.left} y1={70} x2={chartWidth - padding.right} y2={70} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.6} />
                            <text x={padding.left + 4} y={82} fill="#10b981" fontSize="8" fontFamily="monospace" fontWeight="bold">OS 30</text>

                            <text x={chartWidth - padding.right - 10} y={16} fill="#94a3b8" fontSize="8" fontFamily="monospace" textAnchor="end">
                              RSI (14): <tspan fill={activeRsi >= 70 ? '#f43f5e' : activeRsi <= 30 ? '#10b981' : '#38bdf8'} fontWeight="bold">{activeRsi.toFixed(1)}</tspan>
                              {activeRsi >= 70 ? ' (Overbought)' : activeRsi <= 30 ? ' (Oversold)' : ' (Neutral)'}
                              {hoveredCandle && ` (${activeCandle?.time})`}
                            </text>
                            <path
                              d={candles.reduce((acc, c, i) => {
                                const r = (c.rsi !== undefined && c.rsi !== null) ? c.rsi : 50;
                                return `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${100 - r}`;
                              }, '')}
                              fill="none"
                              stroke="#38bdf8"
                              strokeWidth="1.8"
                            />
                          </>
                        );
                      })()}
                    </svg>
                  )}

                  {activeSubChart === 'macd' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const histVals = candles.map(c => c.macd_histogram || 0);
                        const macdLine = candles.map(c => c.macd || 0);
                        const signalLine = candles.map(c => c.macd_signal || 0);
                        const allVals = [...histVals, ...macdLine, ...signalLine];
                        const minV = Math.min(...allVals, 0);
                        const maxV = Math.max(...allVals, 0);
                        const range = (maxV - minV) || 0.001;
                        const norm = (v) => 90 - ((v - minV) / range) * 80;
                        const zeroY = Math.max(10, Math.min(90, norm(0)));
                        const lastCandle = candles[candles.length - 1];
                        const activeCandle = hoveredCandle || lastCandle;
                        return (
                          <>
                            {/* Top info badge */}
                            <text x={padding.left + 4} y={14} fill="#94a3b8" fontSize="8" fontFamily="monospace" fontWeight="bold">
                              MACD: <tspan fill="#38bdf8">{(activeCandle?.macd || 0).toFixed(2)}</tspan> | Sig: <tspan fill="#f59e0b">{(activeCandle?.macd_signal || 0).toFixed(2)}</tspan> | Hist: <tspan fill={(activeCandle?.macd_histogram || 0) >= 0 ? '#10b981' : '#f43f5e'}>{(activeCandle?.macd_histogram || 0).toFixed(2)}</tspan>
                              {hoveredCandle && ` (${activeCandle?.time})`}
                            </text>
                            {/* Zero line */}
                            <line x1={padding.left} y1={zeroY} x2={chartWidth - padding.right} y2={zeroY} stroke="#64748b" strokeOpacity={0.6} strokeDasharray="2 2" />
                            {/* 4-color Histogram bars */}
                            {candles.map((c, i) => {
                              const h = c.macd_histogram || 0;
                              const prevH = i > 0 ? (candles[i-1].macd_histogram || 0) : 0;
                              const isPos = h >= 0;
                              const isGrowing = isPos ? h >= prevH : h <= prevH;
                              const barColor = isPos ? (isGrowing ? '#10b981' : '#34d399') : (isGrowing ? '#f43f5e' : '#fb7185');
                              const barOpacity = isGrowing ? 0.9 : 0.45;
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
                                  rx={0.5}
                                />
                              );
                            })}
                            {/* MACD Line */}
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${norm(c.macd || 0)}`, '')}
                              fill="none" stroke="#38bdf8" strokeWidth="1.8"
                            />
                            {/* Signal Line */}
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${norm(c.macd_signal || 0)}`, '')}
                              fill="none" stroke="#f59e0b" strokeWidth="1.4" strokeDasharray="3 2"
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
                        const zeroY = Math.max(10, Math.min(90, 90 - ((0 - minCvd) / cvdRange) * 80));
                        const lastCandle = candles[candles.length - 1];
                        const activeCandle = hoveredCandle || lastCandle;
                        return (
                          <>
                            <text x={padding.left + 4} y={14} fill="#94a3b8" fontSize="8" fontFamily="monospace" fontWeight="bold">
                              CVD Net Cumulative Delta: <tspan fill={(activeCandle?.cum_delta || 0) >= 0 ? '#eab308' : '#f43f5e'} fontWeight="bold">{(activeCandle?.cum_delta || 0) >= 0 ? '+' : ''}{(activeCandle?.cum_delta || 0).toLocaleString()} shares</tspan>
                              {hoveredCandle && ` (${activeCandle?.time})`}
                            </text>
                            <line x1={padding.left} y1={zeroY} x2={chartWidth - padding.right} y2={zeroY} stroke="#64748b" strokeOpacity={0.6} strokeDasharray="2 2" />
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.cum_delta || 0) - minCvd) / cvdRange) * 80}`, '')}
                              fill="none"
                              stroke="#eab308"
                              strokeWidth="2"
                            />
                          </>
                        );
                      })()}
                    </svg>
                  )}

                  {activeSubChart === 'atr' && (
                    <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full">
                      {(() => {
                        const atrVals = candles.map(c => c.atr || 0).filter(v => v > 0);
                        const minAtr = atrVals.length ? Math.min(...atrVals) * 0.85 : 0;
                        const maxAtr = atrVals.length ? Math.max(...atrVals) * 1.15 : 1;
                        const atrRange = (maxAtr - minAtr) || 1;
                        const lastCandle = candles[candles.length - 1];
                        const activeCandle = hoveredCandle || lastCandle;
                        const currentAtr = activeCandle?.atr || data?.atr || 0;
                        const atrPct = data?.current_price > 0 ? ((currentAtr / data.current_price) * 100).toFixed(2) : '0';
                        return (
                          <>
                            <text x={padding.left + 4} y={14} fill="#94a3b8" fontSize="8" fontFamily="monospace" fontWeight="bold">
                              Average True Range (14): <tspan fill="#f59e0b" fontWeight="bold">{currSym}{currentAtr} ({atrPct}% Volatility)</tspan>
                              <tspan fill="#cbd5e1" dx={8}>Dynamic 1.5× Stop Buffer: ±{currSym}{(currentAtr * 1.5).toFixed(2)}</tspan>
                              {hoveredCandle && ` (${activeCandle?.time})`}
                            </text>
                            {/* ATR Area */}
                            <path
                              d={`${candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.atr || 0) - minAtr) / atrRange) * 70}`, '')} L ${xScale(candles.length - 1)} 90 L ${xScale(0)} 90 Z`}
                              fill="rgba(245, 158, 11, 0.12)"
                            />
                            {/* ATR Line */}
                            <path
                              d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.atr || 0) - minAtr) / atrRange) * 70}`, '')}
                              fill="none"
                              stroke="#f59e0b"
                              strokeWidth="2"
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
                <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2.5">
                        <div className="p-2 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 shadow-sm">
                          <Layers className="w-4 h-4" />
                        </div>
                        <div>
                          <h3 className="text-sm font-bold text-white tracking-wide">
                            Triple-Screen Confluence Matrix
                          </h3>
                          <p className="text-[11px] text-slate-400">Elder 3-tier trend & momentum validation</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="px-2.5 py-1 rounded-xl bg-cyan-500/10 border border-cyan-500/30 font-mono font-bold text-xs text-cyan-300">
                          {data.multi_timeframe?.confluence_score}% Fit
                        </div>
                        <InfoBadge infoKey="triple_screen_confluence" />
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2.5 text-center mb-4 font-mono text-xs">
                      {data.multi_timeframe?.screens?.map((s, idx) => (
                        <div key={idx} className="p-3 bg-slate-950/80 border border-slate-800/80 rounded-2xl hover:border-slate-700 transition">
                          <span className="text-[10px] text-slate-400 block font-sans uppercase font-bold tracking-wider mb-1">
                            {s.timeframe}
                          </span>
                          <span className={`inline-flex items-center gap-1 text-[11px] font-bold px-2 py-0.5 rounded-full my-1 ${
                            s.trend === 'BULLISH'
                              ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                              : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                          }`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${s.trend === 'BULLISH' ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400 animate-pulse'}`} />
                            {s.trend}
                          </span>
                          <span className="text-[10px] text-slate-400 block font-mono mt-1">
                            RSI: <strong className="text-slate-200">{s.rsi}</strong>
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="p-3 rounded-2xl bg-slate-950/90 border border-slate-800/80 flex items-center justify-between text-xs">
                    <span className="text-slate-400 font-medium flex items-center gap-1.5">
                      <Target className="w-3.5 h-3.5 text-cyan-400" />
                      Tactical Verdict:
                    </span>
                    <span className={`font-bold font-mono px-2.5 py-0.5 rounded-lg border text-xs ${
                      data.multi_timeframe?.confluence_score >= 70
                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40'
                        : data.multi_timeframe?.confluence_score <= 30
                        ? 'bg-rose-500/15 text-rose-300 border-rose-500/40'
                        : 'bg-amber-500/15 text-amber-300 border-amber-500/40'
                    }`}>
                      {data.multi_timeframe?.confluence_bias}
                    </span>
                  </div>
                </div>

                {/* Microstructure Order Pressure */}
                <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2.5">
                        <div className="p-2 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shadow-sm">
                          <Flame className="w-4 h-4" />
                        </div>
                        <div>
                          <h3 className="text-sm font-bold text-white tracking-wide">
                            Microstructure Order Pressure
                          </h3>
                          <p className="text-[11px] text-slate-400">Bid-ask tick volume aggression delta</p>
                        </div>
                      </div>
                      <InfoBadge infoKey="order_flow_delta" />
                    </div>

                    <div className="p-3.5 bg-slate-950/80 rounded-2xl border border-slate-800/80 space-y-3">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="px-2 py-0.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-bold">
                          Buyers: {data.order_flow.buy_pressure_pct}%
                        </span>
                        <span className="px-2 py-0.5 rounded-md bg-rose-500/10 border border-rose-500/20 text-rose-400 font-bold">
                          Sellers: {data.order_flow.sell_pressure_pct}%
                        </span>
                      </div>

                      <div className="w-full bg-slate-900 h-3 rounded-full overflow-hidden flex p-0.5 border border-slate-800">
                        <div
                          className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full rounded-l-full transition-all duration-500"
                          style={{ width: `${data.order_flow.buy_pressure_pct}%` }}
                        />
                        <div
                          className="bg-gradient-to-r from-rose-500 to-rose-600 h-full rounded-r-full transition-all duration-500"
                          style={{ width: `${data.order_flow.sell_pressure_pct}%` }}
                        />
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono pt-1">
                        <span>Net Delta: <strong className={data.order_flow.net_delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {data.order_flow.net_delta >= 0 ? '+' : ''}{data.order_flow.net_delta?.toLocaleString()} shares
                        </strong></span>
                        <span>Total Vol: <strong className="text-white">{data.volume?.toLocaleString()}</strong></span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3.5 pt-2.5 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                    <span className="flex items-center gap-1.5">
                      <Zap className="w-3.5 h-3.5 text-amber-400" />
                      Gap Directive:
                    </span>
                    <span className="font-mono text-cyan-300 font-bold bg-cyan-500/10 px-2 py-0.5 rounded-lg border border-cyan-500/25">
                      {data.gap_analysis?.directive}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Volume Profile (VPVR) & Camarilla Pivots (1 span) */}
          <div className="space-y-6">
            {/* Volume Profile (VPVR) */}
            {data?.volume_profile && (
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shadow-sm">
                      <BarChart2 className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white tracking-wide">
                        Volume Profile (VPVR)
                      </h3>
                      <p className="text-[11px] text-slate-400">Horizontal liquidity & Value Area distribution</p>
                    </div>
                  </div>
                  <InfoBadge infoKey="volume_profile" />
                </div>

                <div className="grid grid-cols-3 gap-2 text-xs font-mono mb-3">
                  <div className="p-2.5 rounded-2xl bg-amber-500/10 border border-amber-500/30">
                    <span className="text-[9px] text-amber-400 block font-bold font-sans uppercase tracking-wider">POC Price</span>
                    <span className="text-sm font-bold text-white block mt-0.5">{currSym}{data.volume_profile.poc_price}</span>
                    <span className="text-[9px] text-amber-400/80 font-sans block mt-0.5">High Volume Node</span>
                  </div>
                  <div className="p-2.5 rounded-2xl bg-cyan-500/10 border border-cyan-500/30">
                    <span className="text-[9px] text-cyan-400 block font-bold font-sans uppercase tracking-wider">VAL (70%)</span>
                    <span className="text-sm font-bold text-slate-200 block mt-0.5">{currSym}{data.volume_profile.val_price}</span>
                    <span className="text-[9px] text-slate-400 font-sans block mt-0.5">Value Area Floor</span>
                  </div>
                  <div className="p-2.5 rounded-2xl bg-purple-500/10 border border-purple-500/30">
                    <span className="text-[9px] text-purple-400 block font-bold font-sans uppercase tracking-wider">VAH (70%)</span>
                    <span className="text-sm font-bold text-slate-200 block mt-0.5">{currSym}{data.volume_profile.vah_price}</span>
                    <span className="text-[9px] text-slate-400 font-sans block mt-0.5">Value Area Ceiling</span>
                  </div>
                </div>

                <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1.5 scrollbar-thin scrollbar-thumb-slate-800">
                  {data.volume_profile.profile.map((b, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center gap-2 text-[10px] font-mono py-1 px-2 rounded-xl transition border ${
                        b.is_poc
                          ? 'bg-amber-500/20 border-amber-500/40 text-amber-300 font-bold shadow-[0_0_10px_rgba(245,158,11,0.15)]'
                          : b.in_value_area
                          ? 'bg-slate-950/70 border-slate-800/80 text-slate-300 hover:border-slate-700'
                          : 'bg-transparent border-transparent text-slate-500 opacity-60'
                      }`}
                    >
                      <span className="w-14 shrink-0 font-bold">{b.price.toFixed(2)}</span>
                      <div className="flex-1 bg-slate-900/80 h-2 rounded-full overflow-hidden flex p-0.5 border border-slate-800/40">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            b.is_poc
                              ? 'bg-gradient-to-r from-amber-500 to-amber-300 shadow-[0_0_6px_rgba(245,158,11,0.5)]'
                              : b.in_value_area
                              ? 'bg-gradient-to-r from-cyan-600 to-cyan-400'
                              : 'bg-slate-700'
                          }`}
                          style={{ width: `${Math.min(b.pct_of_total * 4, 100)}%` }}
                        />
                      </div>
                      {b.is_poc ? (
                        <span className="text-[9px] bg-amber-400 text-black px-1.5 py-0.2 rounded font-black shrink-0 tracking-wider">
                          POC
                        </span>
                      ) : b.in_value_area ? (
                        <span className="text-[8px] text-cyan-400/70 shrink-0 font-sans">
                          VA
                        </span>
                      ) : (
                        <span className="w-5 shrink-0" />
                      )}
                    </div>
                  ))}
                </div>

                <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400">
                  <span>Value Area Volume:</span>
                  <span className="font-mono text-cyan-300 font-semibold">70% Standard Deviation</span>
                </div>
              </div>
            )}

            {/* CPR & Camarilla Inflection Levels */}
            {(data?.pivots?.camarilla || data?.pivots?.cpr) && (
              <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 shadow-sm">
                      <Compass className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white tracking-wide">
                        CPR &amp; Institutional Pivots
                      </h3>
                      <p className="text-[11px] text-slate-400">Floor equilibrium &amp; mean-reversion boundaries</p>
                    </div>
                  </div>
                  <InfoBadge infoKey="camarilla_pivots" />
                </div>

                {/* Central Pivot Range (CPR) Box */}
                {data.pivots?.cpr && (
                  <div className="mb-4 p-3.5 rounded-2xl bg-gradient-to-br from-indigo-950/40 via-indigo-950/20 to-slate-950/60 border border-indigo-500/30 space-y-2.5 shadow-inner">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-indigo-300 font-sans flex items-center gap-1.5">
                        <Layers className="w-3.5 h-3.5 text-indigo-400" />
                        Central Pivot Range (CPR)
                      </span>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full font-mono uppercase border ${
                        data.pivots.cpr.classification === 'NARROW'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-[0_0_8px_rgba(16,185,129,0.2)]'
                          : data.pivots.cpr.classification === 'WIDE'
                          ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-[0_0_8px_rgba(245,158,11,0.2)]'
                          : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                      }`}>
                        {data.pivots.cpr.classification} CPR ({data.pivots.cpr.width_pct}%)
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                      <div className="p-2 rounded-xl bg-slate-950/80 border border-slate-800">
                        <span className="text-[9px] text-slate-400 block font-sans uppercase font-bold">TC (Top)</span>
                        <span className="font-bold text-indigo-300 mt-0.5 block">{currSym}{data.pivots.cpr.tc}</span>
                      </div>
                      <div className="p-2 rounded-xl bg-slate-950/80 border border-slate-800">
                        <span className="text-[9px] text-slate-400 block font-sans uppercase font-bold">Pivot (P)</span>
                        <span className="font-bold text-white mt-0.5 block">{currSym}{data.pivots.cpr.pivot}</span>
                      </div>
                      <div className="p-2 rounded-xl bg-slate-950/80 border border-slate-800">
                        <span className="text-[9px] text-slate-400 block font-sans uppercase font-bold">BC (Bottom)</span>
                        <span className="font-bold text-purple-300 mt-0.5 block">{currSym}{data.pivots.cpr.bc}</span>
                      </div>
                    </div>

                    <p className="text-[11px] text-slate-300 leading-snug bg-slate-950/60 p-2 rounded-xl border border-slate-800/60">
                      💡 {data.pivots.cpr.description}
                    </p>
                  </div>
                )}

                {/* Camarilla Pivots Stack */}
                <div className="space-y-2 text-xs font-mono">
                  {/* H4 Breakout */}
                  <div className="flex items-center justify-between p-2.5 rounded-2xl bg-emerald-500/10 border border-emerald-500/25 hover:border-emerald-500/40 transition">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-emerald-400">H4 Breakout Target</span>
                        <span className="text-[9px] font-sans px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-bold uppercase">Acceleration</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">Bullish continuation trigger</p>
                    </div>
                    <span className="text-sm font-bold text-white">{currSym}{data.pivots.camarilla.h4}</span>
                  </div>

                  {/* H3 Resistance */}
                  <div className="flex items-center justify-between p-2.5 rounded-2xl bg-rose-500/10 border border-rose-500/25 hover:border-rose-500/40 transition">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-rose-400">H3 Short Resistance</span>
                        <span className="text-[9px] font-sans px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 font-bold uppercase">Reversal</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">Mean-reversion ceiling</p>
                    </div>
                    <span className="text-sm font-bold text-white">{currSym}{data.pivots.camarilla.h3}</span>
                  </div>

                  {/* Central Floor Pivot (P) */}
                  <div className="flex items-center justify-between p-2.5 rounded-2xl bg-slate-950/90 border border-slate-800 hover:border-slate-700 transition">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-slate-200">Central Floor Pivot (P)</span>
                        <span className="text-[9px] font-sans px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 font-bold uppercase">Equilibrium</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">Session baseline balance point</p>
                    </div>
                    <span className="text-sm font-bold text-cyan-300">{currSym}{data.pivots.floor.p}</span>
                  </div>

                  {/* L3 Support */}
                  <div className="flex items-center justify-between p-2.5 rounded-2xl bg-emerald-500/10 border border-emerald-500/25 hover:border-emerald-500/40 transition">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-emerald-400">L3 Long Support</span>
                        <span className="text-[9px] font-sans px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 font-bold uppercase">Reversal</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">Mean-reversion floor</p>
                    </div>
                    <span className="text-sm font-bold text-white">{currSym}{data.pivots.camarilla.l3}</span>
                  </div>

                  {/* L4 Breakdown */}
                  <div className="flex items-center justify-between p-2.5 rounded-2xl bg-rose-500/10 border border-rose-500/25 hover:border-rose-500/40 transition">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-rose-400">L4 Breakdown Target</span>
                        <span className="text-[9px] font-sans px-1.5 py-0.2 rounded bg-rose-500/20 text-rose-300 font-bold uppercase">Acceleration</span>
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">Bearish expansion trigger</p>
                    </div>
                    <span className="text-sm font-bold text-white">{currSym}{data.pivots.camarilla.l4}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── REAL-WORLD INTRADAY BATTLE PLAN CARD & EXECUTION ────────────── */}
        {data?.battle_plan?.entry_price && (
          <div className="bg-gradient-to-r from-slate-900 via-slate-900/95 to-slate-950 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-gradient-to-tr from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 rounded-2xl shadow-sm">
                  <Target className="w-5 h-5 text-cyan-400" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-base font-bold text-white tracking-wide">
                      Actionable Intraday Battle Plan: <span className="text-cyan-300">{data.battle_plan.setup_name}</span>
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
                className="px-4 py-2 bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/30 rounded-xl text-xs font-semibold transition flex items-center gap-2 shrink-0 self-start sm:self-auto shadow-sm"
              >
                {planCopied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                {planCopied ? 'Copied to Clipboard!' : 'Copy Plan for Broker / Journal'}
              </button>
            </div>

            {/* 4 Core Execution Pods */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 mt-4 text-xs font-mono">
              {/* Entry */}
              <div className="p-3.5 bg-gradient-to-b from-slate-950 to-slate-950/80 border border-cyan-500/30 rounded-2xl shadow-[0_0_12px_rgba(56,189,248,0.06)] hover:border-cyan-500/50 transition">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-cyan-400 font-bold font-sans uppercase tracking-wider">ENTRY TRIGGER</span>
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                </div>
                <span className="text-lg font-bold text-white block mt-1">{currSym}{data.battle_plan.entry_price}</span>
                <p className="text-[10px] text-slate-400 mt-1 font-sans truncate">{data.battle_plan.trigger_rule}</p>
              </div>

              {/* Stop Loss */}
              <div className="p-3.5 bg-gradient-to-b from-slate-950 to-slate-950/80 border border-rose-500/40 rounded-2xl shadow-[0_0_12px_rgba(244,63,94,0.06)] hover:border-rose-500/60 transition">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-rose-400 font-bold font-sans uppercase tracking-wider">HARD STOP LOSS</span>
                  <span className="w-2 h-2 rounded-full bg-rose-400" />
                </div>
                <span className="text-lg font-bold text-rose-400 block mt-1">{currSym}{data.battle_plan.stop_loss}</span>
                <p className="text-[10px] text-slate-400 mt-1 font-sans">Risk: {currSym}{data.battle_plan.risk_per_share} / share</p>
              </div>

              {/* Target 1 */}
              <div className="p-3.5 bg-gradient-to-b from-slate-950 to-slate-950/80 border border-emerald-500/40 rounded-2xl shadow-[0_0_12px_rgba(16,185,129,0.06)] hover:border-emerald-500/60 transition">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-emerald-400 font-bold font-sans uppercase tracking-wider">TARGET 1 (1.5R)</span>
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                </div>
                <span className="text-lg font-bold text-emerald-400 block mt-1">{currSym}{data.battle_plan.target_1}</span>
                <p className="text-[10px] text-slate-400 mt-1 font-sans">Scale out 50% & trail stop</p>
              </div>

              {/* Target 2 */}
              <div className="p-3.5 bg-gradient-to-b from-slate-950 to-slate-950/80 border border-purple-500/40 rounded-2xl shadow-[0_0_12px_rgba(168,85,247,0.06)] hover:border-purple-500/60 transition">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-purple-400 font-bold font-sans uppercase tracking-wider">TARGET 2 (2.5R)</span>
                  <span className="w-2 h-2 rounded-full bg-purple-400" />
                </div>
                <span className="text-lg font-bold text-purple-300 block mt-1">{currSym}{data.battle_plan.target_2}</span>
                <p className="text-[10px] text-slate-400 mt-1 font-sans">Full runner exit target</p>
              </div>
            </div>

            {/* Visual Risk:Reward Road Map Strip */}
            <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span className="text-rose-400 font-semibold">🛑 Stop: {currSym}{data.battle_plan.stop_loss}</span>
              <div className="flex-1 mx-4 h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                <div className="w-1/4 bg-rose-500/60" />
                <div className="w-2/4 bg-emerald-500/60" />
                <div className="w-1/4 bg-purple-500/60" />
              </div>
              <span className="text-emerald-400 font-semibold">🎯 Target: {currSym}{data.battle_plan.target_2}</span>
            </div>
          </div>
        )}

        {/* ── TRADER'S EXECUTION SCRATCHPAD & JOURNAL ─────────────────────── */}
        {scratchpadOpen && (
          <div className="bg-slate-900/90 border border-amber-500/30 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl shadow-amber-950/10">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center shadow-sm">
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
                  <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1 bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">
                    <CheckCircle2 className="w-3 h-3" /> Saved
                  </span>
                )}
                <button
                  onClick={addTimestampToNotes}
                  className="px-2.5 py-1 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 flex items-center gap-1 transition shadow-sm"
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
                  className={`px-2.5 py-1 text-xs font-semibold rounded-lg border flex items-center gap-1 transition shadow-sm ${notesCopied ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' : 'bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700'}`}
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
                    className="px-2.5 py-1 text-xs font-bold bg-rose-500/15 text-rose-400 rounded-lg border border-rose-500/30 transition shadow-sm"
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
              <span className="text-slate-500 font-mono text-[10px] uppercase font-bold">Discipline Tags:</span>
              {[
                { tag: '📌 [VWAP Retest Entry]', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/25' },
                { tag: '🛑 [Hard Stop Violation Risk]', color: 'text-rose-400 bg-rose-500/10 border-rose-500/25' },
                { tag: '🎯 [Camarilla Target Achieved]', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/25' },
                { tag: '⚠️ [Lunch Chop Slump - No Trade]', color: 'text-amber-400 bg-amber-500/10 border-amber-500/25' },
                { tag: '⚡ [MIS Square-Off Approaching]', color: 'text-purple-400 bg-purple-500/10 border-purple-500/25' },
              ].map((chip) => (
                <button
                  key={chip.tag}
                  onClick={() => addTemplateTag(chip.tag)}
                  className={`px-2.5 py-1 rounded-lg border text-[11px] font-mono transition hover:scale-105 active:scale-95 ${chip.color}`}
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
              className="w-full h-28 sm:h-32 bg-slate-950/90 border border-slate-800 rounded-2xl p-3.5 text-xs sm:text-sm font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-amber-500/60 shadow-inner resize-y leading-relaxed"
            />
          </div>
        )}

        {/* ── REAL-LIFE FRICTION & SEBI BREAKEVEN CALCULATOR ────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Position Sizing & Friction Calculator */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 shadow-sm">
                    <Scale className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white tracking-wide">
                      Real-Life Brokerage, STT &amp; Friction Calculator
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Statutory charges, exact position sizing &amp; SEBI breakeven spread check
                    </p>
                  </div>
                </div>
                <InfoBadge infoKey="brokerage_friction_breakeven" />
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs mb-4">
                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Trading Capital ({currSym})</label>
                  <input
                    type="number"
                    value={calcCapital}
                    onChange={(e) => setCalcCapital(e.target.value)}
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500 transition shadow-inner"
                  />
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Risk % Per Trade</label>
                  <select
                    value={calcRiskPct}
                    onChange={(e) => setCalcRiskPct(Number(e.target.value))}
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500 transition shadow-inner"
                  >
                    <option value={0.5}>0.5% (Conservative)</option>
                    <option value={1.0}>1.0% (Institutional Standard)</option>
                    <option value={2.0}>2.0% (Aggressive)</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Margin Leverage</label>
                  <select
                    value={calcLeverage}
                    onChange={(e) => setCalcLeverage(Number(e.target.value))}
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500 transition shadow-inner"
                  >
                    <option value={1}>1× (Cash CNC)</option>
                    <option value={3}>3× (Conservative Margin)</option>
                    <option value={5}>5× (MIS Intraday)</option>
                  </select>
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Entry Price ({currSym})</label>
                  <input
                    type="number"
                    step="0.05"
                    value={calcEntry}
                    onChange={(e) => setCalcEntry(e.target.value)}
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500 transition shadow-inner"
                  />
                </div>

                <div>
                  <label className="text-slate-400 block mb-1 font-medium">Stop Loss ({currSym})</label>
                  <input
                    type="number"
                    step="0.05"
                    value={calcStop}
                    onChange={(e) => setCalcStop(e.target.value)}
                    className="w-full bg-slate-950/90 border border-slate-800 rounded-xl px-3 py-2 font-mono text-white focus:outline-none focus:border-cyan-500 transition shadow-inner"
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
                    className="w-full py-2 px-2 bg-slate-800 hover:bg-slate-700 text-cyan-300 font-semibold rounded-xl text-xs transition border border-slate-700 shadow-sm flex items-center justify-center gap-1.5"
                  >
                    <RefreshCw className="w-3 h-3" />
                    <span>Sync Live Price</span>
                  </button>
                </div>
              </div>

              {sizingResults ? (
                <div className="p-4 bg-slate-950/90 rounded-2xl border border-slate-800 space-y-3.5 shadow-inner">
                  <div className="grid grid-cols-3 gap-2.5 text-center font-mono">
                    <div className="p-3 rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 hover:border-slate-700 transition">
                      <span className="text-[10px] text-slate-400 block font-sans uppercase font-bold tracking-wider">EXACT SHARES</span>
                      <span className="text-xl font-bold text-cyan-300 block mt-1">{sizingResults.exactShares}</span>
                    </div>
                    <div className="p-3 rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 hover:border-slate-700 transition">
                      <span className="text-[10px] text-slate-400 block font-sans uppercase font-bold tracking-wider">MARGIN NEEDED</span>
                      <span className="text-xl font-bold text-white block mt-1">{currSym}{sizingResults.marginRequired?.toLocaleString()}</span>
                    </div>
                    <div className="p-3 rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 hover:border-slate-700 transition">
                      <span className="text-[10px] text-slate-400 block font-sans uppercase font-bold tracking-wider">TOTAL FRICTION</span>
                      <span className="text-xl font-bold text-amber-400 block mt-1">{currSym}{sizingResults.totalCharges}</span>
                    </div>
                  </div>

                  {/* Breakeven Spread Alert */}
                  <div className="p-3 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-300 font-sans flex items-center gap-1.5 font-medium">
                      ⚠️ Breakeven Spread Needed:
                    </span>
                    <span className="text-amber-400 font-bold bg-amber-500/15 px-2.5 py-0.5 rounded-lg border border-amber-500/30">
                      +{currSym}{sizingResults.breakevenMovePts} (+{sizingResults.breakevenMovePct}%)
                    </span>
                  </div>

                  {/* Net Profit Table */}
                  <div className="grid grid-cols-3 gap-2.5 pt-1 border-t border-slate-800/80 text-xs font-mono">
                    {sizingResults.riskRewardTargets.map((t, idx) => (
                      <div key={idx} className="p-2.5 bg-emerald-500/5 border border-emerald-500/20 rounded-2xl hover:border-emerald-500/40 transition">
                        <span className="text-[10px] text-emerald-400 block font-bold font-sans uppercase tracking-wider">{t.label}</span>
                        <p className="text-sm font-bold text-white mt-0.5">{currSym}{t.price}</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">Gross: +{currSym}{t.gross?.toLocaleString()}</p>
                        <p className={`text-[10px] font-bold mt-0.5 ${t.net >= 0 ? 'text-emerald-300' : 'text-rose-400'}`}>
                          Net: {t.net >= 0 ? '+' : ''}{currSym}{t.net?.toLocaleString()}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-slate-400 text-xs bg-slate-950/50 rounded-2xl border border-slate-800/80 leading-relaxed">
                  Enter valid Entry and Stop Loss prices above to calculate exact position sizing, SEBI statutory friction, and in-pocket net profit.
                </div>
              )}
            </div>
          </div>

          {/* Real-Time Intraday Radar Scanner */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shadow-sm">
                    <Zap className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white tracking-wide">
                      Intraday Radar Scanner ({scannerMarket})
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Real-time momentum, ORB breakouts, and VWAP deviations
                    </p>
                  </div>
                </div>
                <InfoBadge infoKey="intraday_rvol" />
              </div>

              {scannerLoading ? (
                <div className="h-48 flex flex-col items-center justify-center text-xs text-slate-400 gap-2">
                  <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
                  <span>Scanning high-liquidity universe...</span>
                </div>
              ) : (
                <div className="space-y-2 max-h-72 overflow-y-auto pr-1.5 scrollbar-thin scrollbar-thumb-slate-800">
                  {scannerData.map((item) => (
                    <div
                      key={item.ticker}
                      onClick={() => changeTicker(item.ticker)}
                      className={`flex items-center justify-between p-3 rounded-2xl border transition cursor-pointer ${
                        ticker === item.ticker
                          ? 'bg-cyan-500/15 border-cyan-500/50 shadow-[0_0_12px_rgba(56,189,248,0.15)]'
                          : 'bg-slate-950/70 hover:bg-slate-950 border-slate-800/80 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-xl ${item.change_pct >= 0 ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'}`}>
                          {item.change_pct >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-sm text-white">{item.ticker.split('.')[0]}</span>
                            <span className={`text-[10px] font-bold font-mono px-2 py-0.5 rounded-full border ${
                              item.orb_status === 'BREAKOUT' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' :
                              item.orb_status === 'BREAKDOWN' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' :
                              'bg-slate-800/80 text-slate-400 border-slate-700'
                            }`}>
                              {item.orb_status}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                            VWAP Dist: <strong className={item.vwap_dist_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                              {item.vwap_dist_pct >= 0 ? '+' : ''}{item.vwap_dist_pct}%
                            </strong>
                            {item.rvol && (
                              <span className={`ml-2 px-1.5 py-0.2 rounded text-[10px] font-bold ${
                                item.rvol >= 2
                                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                                  : item.rvol >= 1.5
                                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                                  : 'text-slate-500'
                              }`}>
                                RVOL {item.rvol}×
                              </span>
                            )}
                          </p>
                        </div>
                      </div>

                      <div className="text-right font-mono">
                        <span className="text-sm font-bold text-white block">
                          {item.currency_symbol}{item.price}
                        </span>
                        <span className={`inline-block text-[11px] font-bold px-2 py-0.5 rounded-md mt-0.5 ${
                          item.change_pct >= 0 ? 'bg-emerald-500/15 text-emerald-400' : 'bg-rose-500/15 text-rose-400'
                        }`}>
                          {item.change_pct >= 0 ? '+' : ''}{item.change_pct}%
                        </span>
                      </div>
                    </div>
                  ))}
                  {scannerData.length === 0 && (
                    <div className="h-32 flex flex-col items-center justify-center text-xs text-slate-500 gap-2 bg-slate-950/40 rounded-2xl border border-slate-800/60">
                      <Zap className="w-5 h-5 text-slate-700" />
                      <span>No high-momentum setups detected in current session.</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="mt-4 pt-3.5 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
              <span>Click any opportunity to load into terminal</span>
              <button
                onClick={fetchScanner}
                className="text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1.5 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 px-3 py-1 rounded-xl border border-cyan-500/30 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" /> Rescan Now
              </button>
            </div>
          </div>
        </div>

        {/* ── OPTIONS PCR + BLOCK DEALS + TRADE LOG ROW ──────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Options Put-Call Ratio Widget */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-purple-500/15 border border-purple-500/30 text-purple-400 shadow-sm">
                    <Scale className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-wide">
                      Options Put-Call Ratio
                    </h3>
                    <p className="text-[11px] text-slate-400">Macro derivatives sentiment &amp; pain</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <InfoBadge infoKey="options_pcr" />
                  <button
                    onClick={fetchPCR}
                    className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-white transition border border-slate-700/60"
                    title="Refresh PCR data"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${pcrLoading ? 'animate-spin text-cyan-400' : ''}`} />
                  </button>
                </div>
              </div>

              {pcrData ? (
                pcrData.available === false ? (
                  <div className="h-44 flex flex-col items-center justify-center p-4 text-center bg-slate-950/50 rounded-2xl border border-slate-800/80">
                    <Scale className="w-7 h-7 text-slate-600 mb-2" />
                    <span className="text-xs font-bold text-slate-300 mb-1">Derivatives Unavailable</span>
                    <p className="text-[11px] text-slate-400 max-w-[220px] leading-relaxed">
                      {pcrData.message || 'Options chain data is available for US securities and select NSE F&O listings.'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3.5">
                    {pcrData.is_index_benchmark && (
                      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl bg-purple-500/10 border border-purple-500/30 text-[10px] text-purple-300 font-mono">
                        <Scale className="w-3 h-3 text-purple-400" />
                        <span>{pcrData.benchmark_name} Macro Derivatives Sentiment</span>
                      </div>
                    )}

                    <div className="p-3.5 rounded-2xl bg-slate-950/80 border border-slate-800/80 flex items-center justify-between">
                      <div>
                        <span className="text-[10px] text-slate-400 font-sans uppercase font-bold tracking-wider block">PCR OI Ratio</span>
                        <div className="flex items-baseline gap-1.5 mt-0.5">
                          <span className={`text-2xl font-black font-mono ${
                            pcrData.color === 'bearish' ? 'text-rose-400' :
                            pcrData.color === 'bullish' ? 'text-emerald-400' : 'text-amber-400'
                          }`}>
                            {pcrData.pcr_oi}
                          </span>
                          <span className="text-[10px] text-slate-500 font-mono">OI</span>
                        </div>
                      </div>

                      <div className="text-right">
                        <span className={`inline-block text-[10px] font-bold px-2.5 py-0.5 rounded-full border uppercase ${
                          pcrData.color === 'bearish' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 shadow-[0_0_8px_rgba(244,63,94,0.2)]' :
                          pcrData.color === 'bullish' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-[0_0_8px_rgba(16,185,129,0.2)]' :
                          'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-[0_0_8px_rgba(245,158,11,0.2)]'
                        }`}>
                          {pcrData.sentiment?.replace('_', ' ')}
                        </span>
                        <p className="text-[10px] text-slate-400 mt-1 font-mono">{pcrData.expiry_date}</p>
                      </div>
                    </div>

                    {/* Put vs Call OI distribution bar */}
                    <div className="space-y-1.5 p-3 rounded-2xl bg-slate-950/60 border border-slate-800/60">
                      <div className="flex justify-between text-[10px] font-mono">
                        <span className="text-emerald-400 font-bold">Calls: {(pcrData.call_oi / 1000).toFixed(0)}K OI</span>
                        <span className="text-rose-400 font-bold">Puts: {(pcrData.put_oi / 1000).toFixed(0)}K OI</span>
                      </div>
                      <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden flex p-0.5 border border-slate-800">
                        {(() => {
                          const total = (pcrData.call_oi || 0) + (pcrData.put_oi || 0) || 1;
                          const callPct = Math.round((pcrData.call_oi / total) * 100);
                          return (
                            <>
                              <div className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full rounded-l-full transition-all duration-500" style={{ width: `${callPct}%` }} />
                              <div className="bg-gradient-to-r from-rose-500 to-rose-600 h-full rounded-r-full transition-all duration-500" style={{ width: `${100 - callPct}%` }} />
                            </>
                          );
                        })()}
                      </div>
                    </div>

                    {pcrData.max_pain_strike && (
                      <div className="p-2.5 bg-slate-950/80 rounded-xl border border-slate-800 flex justify-between items-center text-xs">
                        <span className="text-slate-400 font-medium">Max Pain Strike:</span>
                        <span className="font-bold font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/25">
                          {currSym}{pcrData.max_pain_strike}
                        </span>
                      </div>
                    )}

                    <p className="text-[11px] text-slate-400 leading-snug pt-1">
                      💡 {pcrData.sentiment_label}
                    </p>
                  </div>
                )
              ) : (
                <div className="h-44 flex items-center justify-center text-xs text-slate-500">
                  {pcrLoading ? 'Loading options chain...' : 'No options data available for this ticker'}
                </div>
              )}
            </div>
          </div>

          {/* NSE Block / Bulk Deals */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-orange-500/15 border border-orange-500/30 text-orange-400 shadow-sm">
                    <Flame className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-wide">
                      NSE Block &amp; Bulk Deals
                    </h3>
                    <p className="text-[11px] text-slate-400">Institutional smart-money prints</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <InfoBadge infoKey="block_deals" />
                  {blockDeals && ((blockDeals.block_deals?.length || 0) > 0 || (blockDeals.bulk_deals?.length || 0) > 0) && (
                    <button
                      onClick={exportBlockDealsCSV}
                      title="Export NSE Block & Bulk Deals to CSV"
                      className="text-xs font-semibold px-2.5 py-1 rounded-lg border bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700 hover:text-orange-400 hover:border-orange-500/40 transition flex items-center gap-1 cursor-pointer shadow-sm"
                    >
                      <Download className="w-3 h-3" />
                      <span className="hidden sm:inline">CSV</span>
                    </button>
                  )}
                  {blockDealsLoading && <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />}
                </div>
              </div>

              {scannerMarket !== 'IN' ? (
                <div className="h-44 flex items-center justify-center text-xs text-slate-500 text-center p-4 bg-slate-950/40 rounded-2xl border border-slate-800/60">
                  Block/Bulk deal feed is available for NSE (Indian market) securities only.
                </div>
              ) : blockDeals ? (
                <div className="space-y-2 max-h-60 overflow-y-auto pr-1.5 scrollbar-thin scrollbar-thumb-slate-800">
                  {[...(blockDeals.block_deals || []).map(d => ({...d, type: 'BLOCK'})),
                     ...(blockDeals.bulk_deals || []).map(d => ({...d, type: 'BULK'}))].slice(0, 15).map((deal, idx) => (
                    <div
                      key={idx}
                      onClick={() => deal.symbol && changeTicker(deal.symbol + '.NS')}
                      className="flex items-center justify-between p-2.5 rounded-2xl bg-slate-950/70 border border-slate-800/70 hover:border-slate-700 cursor-pointer transition text-xs hover:bg-slate-950"
                    >
                      <div>
                        <div className="flex items-center gap-1.5">
                          <span className="font-bold text-white text-xs">{deal.symbol}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border uppercase ${
                            deal.type === 'BLOCK'
                              ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                              : 'bg-orange-500/20 text-orange-300 border-orange-500/40'
                          }`}>
                            {deal.type}
                          </span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border ${
                            deal.trade_type === 'B' || deal.trade_type === 'BUY'
                              ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                              : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                          }`}>
                            {deal.trade_type === 'B' || deal.trade_type === 'BUY' ? 'BUY' : 'SELL'}
                          </span>
                        </div>
                        <p className="text-[10px] text-slate-400 mt-1 truncate max-w-[130px] font-sans">
                          {deal.client || 'Undisclosed'}
                        </p>
                      </div>

                      <div className="text-right font-mono">
                        <span className="text-slate-200 font-bold block">{deal.quantity?.toLocaleString() || '—'}</span>
                        <p className="text-[10px] text-slate-400 mt-0.5">@ ₹{deal.price || deal.avg_price || '—'}</p>
                      </div>
                    </div>
                  ))}
                  {blockDeals.block_count === 0 && blockDeals.bulk_count === 0 && (
                    <div className="h-44 flex flex-col items-center justify-center text-xs text-slate-400 p-4 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-center gap-2">
                      <span className="font-bold text-slate-300">Dedicated Window Schedule</span>
                      <p className="text-[11px] text-slate-400 max-w-[230px] leading-relaxed">
                        {blockDeals.next_window || 'Block Deals execute in two windows (08:45 AM & 02:05 PM IST). Bulk deals report at EOD.'}
                      </p>
                      <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-slate-900 border border-slate-800 text-cyan-400 font-semibold mt-0.5">
                        Status: {blockDeals.window_status || 'STANDBY'}
                      </span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-44 flex items-center justify-center text-xs text-slate-500">
                  Loading NSE deal feed...
                </div>
              )}
            </div>
          </div>

          {/* Trade Log with Live P&L */}
          <div className="bg-slate-900/80 border border-slate-800/80 rounded-3xl p-5 sm:p-6 backdrop-blur-md shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 shadow-sm">
                    <BarChart2 className="w-4 h-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white tracking-wide">
                      Trade Log &amp; P&amp;L Tracker
                    </h3>
                    <p className="text-[11px] text-slate-400">Position ledger &amp; win rate telemetry</p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <InfoBadge infoKey="trade_log" />
                  {tradeLog.length > 0 && (
                    <button
                      onClick={exportTradeLogCSV}
                      className="text-xs font-semibold px-2.5 py-1 rounded-lg border bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700 hover:text-emerald-400 hover:border-emerald-500/40 transition flex items-center gap-1 shadow-sm"
                      title="Export logged trades as CSV spreadsheet"
                    >
                      <Download className="w-3 h-3" />
                      <span className="hidden sm:inline">CSV</span>
                    </button>
                  )}
                  <button
                    onClick={() => setTradeLogOpen(!tradeLogOpen)}
                    className={`text-xs font-semibold px-3 py-1 rounded-xl border transition shadow-sm ${
                      tradeLogOpen
                        ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-[0_0_8px_rgba(56,189,248,0.2)] font-bold'
                        : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border-slate-700'
                    }`}
                  >
                    {tradeLogOpen ? 'Hide Form' : '+ Log Trade'}
                  </button>
                </div>
              </div>

              {tradeLogOpen && (
                <div className="mb-3.5 p-3.5 bg-slate-950/90 rounded-2xl border border-slate-800 space-y-2.5 shadow-inner">
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      value={newTrade.ticker}
                      onChange={e => setNewTrade(p => ({...p, ticker: e.target.value}))}
                      placeholder="Ticker (e.g. SBIN)"
                      className="col-span-1 bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
                    />
                    <select
                      value={newTrade.direction}
                      onChange={e => setNewTrade(p => ({...p, direction: e.target.value}))}
                      className="col-span-1 bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500 shadow-inner"
                    >
                      <option value="LONG">LONG</option>
                      <option value="SHORT">SHORT</option>
                    </select>
                    <input
                      type="number" step="0.05"
                      value={newTrade.entry}
                      onChange={e => setNewTrade(p => ({...p, entry: e.target.value}))}
                      placeholder="Entry price"
                      className="bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
                    />
                    <input
                      type="number" step="0.05"
                      value={newTrade.exit}
                      onChange={e => setNewTrade(p => ({...p, exit: e.target.value}))}
                      placeholder="Exit price (optional)"
                      className="bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
                    />
                    <input
                      type="number"
                      value={newTrade.qty}
                      onChange={e => setNewTrade(p => ({...p, qty: e.target.value}))}
                      placeholder="Qty / Shares"
                      className="bg-slate-900 border border-slate-700 rounded-xl px-2.5 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 shadow-inner"
                    />
                    <button
                      onClick={() => setNewTrade(p => ({...p, ticker: ticker.split('.')[0], entry: data?.current_price?.toString() || ''}))}
                      className="bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-xl px-2 py-1.5 text-xs font-semibold transition border border-slate-700 shadow-sm"
                    >
                      Sync Live
                    </button>
                  </div>
                  <button
                    onClick={addTradeEntry}
                    className="w-full py-2 bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 hover:from-cyan-500/30 border border-cyan-500/40 text-cyan-300 font-bold rounded-xl text-xs transition shadow-sm"
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
                  <div className="flex items-center justify-between mb-3 px-2 py-1.5 rounded-xl bg-slate-950/80 border border-slate-800/80">
                    <span className="text-[11px] font-mono text-slate-400">
                      {closed.length} closed · <strong className="text-white">{winRate}%</strong> W/R
                    </span>
                    <span className={`text-xs font-bold font-mono px-2 py-0.5 rounded-lg border ${
                      totalPnl >= 0
                        ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                        : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                    }`}>
                      Net P&amp;L: {totalPnl >= 0 ? '+' : ''}{currSym}{totalPnl.toLocaleString()}
                    </span>
                  </div>
                );
              })()}

              <div className="space-y-2 max-h-52 overflow-y-auto pr-1.5 scrollbar-thin scrollbar-thumb-slate-800">
                {tradeLog.slice(0, 20).map(t => (
                  <div key={t.id} className={`flex items-center justify-between p-2.5 rounded-2xl border text-xs transition ${
                    t.status === 'WIN' ? 'bg-emerald-500/10 border-emerald-500/30' :
                    t.status === 'LOSS' ? 'bg-rose-500/10 border-rose-500/30' :
                    'bg-slate-950/70 border-slate-800/80'
                  }`}>
                    <div className="flex items-center gap-2">
                      <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded border uppercase ${
                        t.direction === 'LONG'
                          ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                      }`}>{t.direction}</span>
                      <div>
                        <span className="font-bold text-white text-xs">{t.ticker}</span>
                        <span className="text-slate-500 ml-1.5 font-mono text-[10px]">{t.time}</span>
                        {t.note && <p className="text-[10px] text-slate-400 mt-0.5 truncate max-w-[110px] font-sans">{t.note}</p>}
                      </div>
                    </div>

                    <div className="flex items-center gap-2.5">
                      <span className={`font-bold font-mono text-xs ${
                        t.status === 'WIN' ? 'text-emerald-400' :
                        t.status === 'LOSS' ? 'text-rose-400' : 'text-slate-400'
                      }`}>
                        {t.grossPnl !== null ? `${t.grossPnl >= 0 ? '+' : ''}${currSym}{t.grossPnl}` : 'OPEN'}
                      </span>
                      <button onClick={() => removeTrade(t.id)} className="text-slate-600 hover:text-rose-400 transition p-1">
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
                {tradeLog.length === 0 && !tradeLogOpen && (
                  <div className="h-32 flex flex-col items-center justify-center text-xs text-slate-400 p-4 bg-slate-950/40 rounded-2xl border border-slate-800/60 text-center gap-1">
                    <BarChart2 className="w-6 h-6 text-slate-600 mb-1" />
                    <span>No trades logged in current session.</span>
                    <span className="text-[11px] text-slate-500">Click &quot;+ Log Trade&quot; to track your live setups.</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* ── PRO KEYBOARD SHORTCUTS MODAL ─────────────────────────────────── */}
        {showHotkeysModal && (
          <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-slate-900 border border-slate-700/80 rounded-3xl p-6 max-w-lg w-full shadow-2xl space-y-4 animate-in fade-in zoom-in-95 duration-200">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 bg-cyan-500/15 border border-cyan-500/30 rounded-2xl shadow-sm">
                    <Keyboard className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white">Pro Keyboard Shortcuts</h3>
                    <p className="text-xs text-slate-400">Institutional desk navigation without touching the mouse</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowHotkeysModal(false)}
                  className="p-2 rounded-xl text-slate-400 hover:text-white bg-slate-800/80 hover:bg-slate-700 transition border border-slate-700/60"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs max-h-[60vh] overflow-y-auto pr-1.5 scrollbar-thin scrollbar-thumb-slate-800">
                {[
                  { key: '/', desc: 'Focus Ticker Search' },
                  { key: '1, 2, 3, 5', desc: '1m, 2m, 3m, 5m Timeframe' },
                  { key: '4', desc: '15m Timeframe' },
                  { key: '6', desc: '30m Timeframe' },
                  { key: 'H', desc: '1h Timeframe' },
                  { key: 'V', desc: 'Toggle VWAP & Bands' },
                  { key: 'S', desc: 'Toggle Supertrend' },
                  { key: 'C', desc: 'Toggle CPR Range' },
                  { key: 'K', desc: 'Toggle Heikin-Ashi' },
                  { key: 'R', desc: 'Force Refresh Data' },
                  { key: 'F', desc: 'Toggle Fullscreen Chart' },
                  { key: '?', desc: 'Open this Shortcuts HUD' },
                  { key: 'ESC', desc: 'Close Modals & Blur Input' },
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2.5 rounded-2xl bg-slate-950/80 border border-slate-800/80 hover:border-slate-700 transition">
                    <span className="text-slate-300 font-medium">{item.desc}</span>
                    <kbd className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700 text-cyan-300 font-mono font-bold text-[11px] shadow-sm">
                      {item.key}
                    </kbd>
                  </div>
                ))}
              </div>

              <div className="pt-3 border-t border-slate-800 flex justify-end">
                <button
                  onClick={() => setShowHotkeysModal(false)}
                  className="px-5 py-2 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 font-bold text-xs transition border border-cyan-500/40 shadow-sm"
                >
                  Got it (Esc)
                </button>
              </div>
            </div>
          </div>
        )}

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
