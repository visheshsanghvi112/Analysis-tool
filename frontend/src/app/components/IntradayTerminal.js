'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import {
  Activity, ArrowUpRight, ArrowDownRight, RefreshCw, Layers,
  Compass, Calculator, ShieldAlert, Sparkles, Sliders, ChevronDown,
  Search, TrendingUp, TrendingDown, Target, Zap, Clock, ShieldCheck,
  BarChart2, Flame, Eye, ArrowRight, CheckCircle2, XCircle, AlertCircle, MinusCircle,
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
  const [showOverlaysMenu, setShowOverlaysMenu] = useState(false);

  // Pinned Watchlist & Hotkeys Modal
  const [pinnedTickers, setPinnedTickers] = useState([]);
  const [showHotkeysModal, setShowHotkeysModal] = useState(false);
  const searchInputRef = useRef(null);

  // Viewport Zoom: 'all' | '60' | '30'
  const [candleSlice, setCandleSlice] = useState('all');

  // Microstructure Sidebar Active Tab: 'pivots' | 'flow' | 'options' | 'calc' | 'confluence' | 'log'
  const [sidebarTab, setSidebarTab] = useState('pivots');

  const activeOverlaysCount = useMemo(() => {
    let count = 0;
    if (showVWAP) count++;
    if (showVWAPBands) count++;
    if (showSupertrend) count++;
    if (showEMA) count++;
    if (showEMA200) count++;
    if (showORB) count++;
    if (showCamarilla) count++;
    if (showPDH) count++;
    if (showCPR) count++;
    if (candleMode === 'heikin_ashi') count++;
    return count;
  }, [showVWAP, showVWAPBands, showSupertrend, showEMA, showEMA200, showORB, showCamarilla, showPDH, showCPR, candleMode]);

  // Sub-chart selector
  const [activeSubChart, setActiveSubChart] = useState('volume'); // 'volume' | 'rsi' | 'cvd' | 'macd' | 'atr'

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
  const [hoveredY, setHoveredY] = useState(null);

  // Fullscreen chart mode
  const [fullscreenChart, setFullscreenChart] = useState(false);

  // Price flash animation (green/red on tick update)
  const [priceFlash, setPriceFlash] = useState(null); // 'up' | 'down' | null
  const prevPriceRef = useRef(null);

  // Price alert system
  const [alertPrice, setAlertPrice] = useState('');
  const [alertTriggered, setAlertTriggered] = useState(false);
  const [alertAbove, setAlertAbove] = useState(true);

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

  useEffect(() => {
    if (urlTicker && urlTicker.trim()) {
      const clean = urlTicker.trim().toUpperCase();
      if (clean !== ticker && clean !== lastSyncedTickerRef.current) {
        lastSyncedTickerRef.current = clean;
        setTicker(clean);
      }
    }
  }, [urlTicker, ticker]);

  useEffect(() => {
    if (typeof window !== 'undefined' && ticker) {
      try {
        const saved = localStorage.getItem('stockiq_intraday_notes_' + ticker);
        setNotes(saved || '');
      } catch (_) {}
    }
  }, [ticker]);

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
      'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
      'VWAP', 'VWAP_Upper_1', 'VWAP_Lower_1', 'Supertrend',
      'Supertrend_Signal', 'EMA9', 'EMA21', 'EMA50', 'EMA200',
      'RSI', 'MACD', 'MACD_Signal', 'ATR'
    ];
    const rows = data.candles.map(c => [
      `"${c.timestamp || ''}"`,
      c.open ?? '', c.high ?? '', c.low ?? '', c.close ?? '', c.volume ?? '',
      c.vwap ?? '', c.upper_1 ?? '', c.lower_1 ?? '',
      c.supertrend ?? '', c.supertrend_dir === 1 ? 'BULLISH' : 'BEARISH',
      c.ema9 ?? '', c.ema21 ?? '', c.ema50 ?? '', c.ema200 ?? '',
      c.rsi ?? '', c.macd ?? '', c.macd_signal ?? '', c.atr ?? ''
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
  }, [data, ticker, candleInterval, period]);

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
        setShowOverlaysMenu(false);
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

  // Fetch Options PCR
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
    const id = setInterval(fetchPCR, 300000);
    return () => clearInterval(id);
  }, [fetchPCR]);

  // Fetch NSE Block / Bulk Deals
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
    const updated = [trade, ...tradeLog].slice(0, 50);
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

    let totalCharges = 0;
    let brokerage = 0;
    let stt = 0;
    let nseTxn = 0;
    let gst = 0;

    const buyTurnover = entry * exactShares;
    const sellTurnover = target1 * exactShares;
    const totalTurnover = buyTurnover + sellTurnover;

    if (!isUS) {
      brokerage = Math.min(20.0, 0.0005 * buyTurnover) + Math.min(20.0, 0.0005 * sellTurnover);
      stt = 0.00025 * sellTurnover;
      nseTxn = 0.0000297 * totalTurnover;
      const sebi = 0.000001 * totalTurnover;
      const stampDuty = 0.00003 * buyTurnover;
      gst = 0.18 * (brokerage + nseTxn + sebi);
      totalCharges = Math.round((brokerage + stt + nseTxn + sebi + stampDuty + gst) * 100) / 100;
    } else {
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
  }, [candles, showVWAPBands, showSupertrend, showPDH, showORB, showCamarilla, showEMA200, data]);

  // Touch and Mouse crosshair handlers
  const handleChartTouch = useCallback((e) => {
    if (!e.touches || !e.touches[0] || !candles.length) return;
    const touch = e.touches[0];
    const rect = e.currentTarget.getBoundingClientRect();
    const currentX = ((touch.clientX - rect.left) / rect.width) * chartWidth;
    const currentY = ((touch.clientY - rect.top) / rect.height) * chartHeight;
    setHoveredX(currentX);
    setHoveredY(currentY);

    const innerW = chartWidth - padding.left - padding.right;
    const relX = currentX - padding.left;
    const candleIdx = Math.round((relX / Math.max(innerW, 1)) * (candles.length - 1));
    if (candleIdx >= 0 && candleIdx < candles.length) {
      setHoveredCandle(candles[candleIdx]);
    }
  }, [candles, chartWidth, chartHeight, padding]);

  const handleSubChartTouch = useCallback((e) => {
    if (!e.touches || !e.touches[0] || !candles.length) return;
    const touch = e.touches[0];
    const rect = e.currentTarget.getBoundingClientRect();
    const currentX = ((touch.clientX - rect.left) / rect.width) * chartWidth;
    setHoveredX(currentX);

    const innerW = chartWidth - padding.left - padding.right;
    const relX = currentX - padding.left;
    const candleIdx = Math.round((relX / Math.max(innerW, 1)) * (candles.length - 1));
    if (candleIdx >= 0 && candleIdx < candles.length) {
      setHoveredCandle(candles[candleIdx]);
    }
  }, [candles, chartWidth, padding]);

  const handleChartTouchEnd = useCallback(() => {
    setHoveredCandle(null);
    setHoveredX(null);
    setHoveredY(null);
  }, []);

  return (
    <div className="min-h-screen bg-[#06080d] text-slate-100 font-sans flex flex-col justify-between selection:bg-cyan-500/20 selection:text-cyan-200">
      <div>
        <Header
          currentTicker={ticker}
          onTickerSelect={(sym) => {
            if (!sym) return;
            changeTicker(sym);
          }}
        />

        <main className="w-full max-w-[1600px] mx-auto p-3 sm:p-5 lg:p-6 space-y-4 sm:space-y-5">

          {/* ── REAL-TIME MARKET SESSION CLOCK & PHASE BANNER ── */}
          {marketPulse && (
            <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-4 backdrop-blur-md shadow-lg shadow-black/40 space-y-3">
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 w-full">
                <div className="flex items-center gap-3">
                  <div className={`p-2.5 rounded-xl flex items-center justify-center shrink-0 border ${
                    marketPulse.is_open 
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25' 
                      : 'bg-amber-500/10 text-amber-400 border-amber-500/25'
                  }`}>
                    <Clock className="w-4 h-4 animate-pulse" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                        {marketPulse.market === 'IN' ? 'Dalal Street Session' : 'Wall Street Session'} ({marketPulse.local_time})
                      </span>
                      <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded-full uppercase border ${
                        marketPulse.is_open 
                          ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30' 
                          : 'bg-slate-800/80 text-slate-400 border-slate-700'
                      }`}>
                        {marketPulse.is_open ? 'LIVE SESSION' : 'CLOSED'}
                      </span>
                      <InfoBadge infoKey="session_phase_clock" />
                    </div>
                    <p className="text-xs font-semibold text-slate-200 mt-0.5 flex items-center gap-1.5 flex-wrap">
                      <span className="text-white font-bold">{marketPulse.phase_name}</span>
                      <span className="text-slate-600">—</span>
                      <span className="text-slate-400 font-normal">{marketPulse.directive}</span>
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
                        className={`px-2.5 py-1.5 rounded-xl text-xs font-mono shrink-0 flex items-center gap-2 border ${
                          isVix
                            ? 'bg-purple-950/30 border-purple-500/30 text-purple-300'
                            : 'bg-[#070a10] border-white/[0.06] text-white'
                        }`}
                      >
                        <span className="text-slate-400 font-sans font-medium text-[11px]">{idx.name}:</span>
                        <span className="font-bold tabular-nums">{idx.price?.toLocaleString()}</span>
                        <span className={`text-[11px] font-bold tabular-nums ${idx.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {idx.change_pct >= 0 ? '+' : ''}{idx.change_pct}%
                        </span>
                        {isVix && marketPulse.vix?.regime && (
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded uppercase border ${
                            marketPulse.vix.regime === 'LOW' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30' :
                            marketPulse.vix.regime === 'NORMAL' ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30' : 
                            'bg-rose-500/20 text-rose-300 border-rose-500/30'
                          }`}>
                            {marketPulse.vix.regime}
                          </span>
                        )}
                      </div>
                    );
                  })}

                  {marketPulse.mins_to_mis_squareoff > 0 && (
                    <div className="px-3 py-1.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs font-mono shrink-0 flex items-center gap-1.5 font-bold">
                      <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                      {scannerMarket === 'IN' ? 'Auto-Square-Off' : 'Market Close'} in {marketPulse.mins_to_mis_squareoff}m
                    </div>
                  )}
                </div>
              </div>

              {/* Live Sectoral Heatmap Flow Strip */}
              {marketPulse.sectors && marketPulse.sectors.length > 0 && (
                <div className="flex items-center gap-2 overflow-x-auto w-full pt-2.5 border-t border-white/[0.06] scrollbar-none text-[11px] font-mono">
                  <div className="flex items-center gap-1.5 text-slate-400 uppercase tracking-wider font-sans font-bold text-[10px] shrink-0 pr-1">
                    <Flame className="w-3.5 h-3.5 text-amber-400" />
                    <span>Sector Flow:</span>
                  </div>
                  {marketPulse.sectors.map((sec) => (
                    <div
                      key={sec.symbol}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#070a10] border border-white/[0.06] shrink-0"
                    >
                      <span className="text-slate-300 font-sans font-medium">{sec.name.replace('NIFTY ', '')}</span>
                      <span className={`font-bold tabular-nums ${sec.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {sec.change_pct >= 0 ? '+' : ''}{sec.change_pct}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── TOP TERMINAL BAR & CONTROLS ── */}
          <header className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-2 border-b border-white/[0.06]">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-tr from-emerald-500/15 to-cyan-500/15 border border-cyan-500/30 rounded-2xl shadow-sm">
                <Activity className="w-5 h-5 text-cyan-400 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl sm:text-2xl font-black tracking-tight text-white">
                    Intraday Quantitative Desk
                  </h1>
                  <span className="px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 rounded-full">
                    High-Frequency
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Institutional session anchors, volume delta, Camarilla levels &amp; friction engine
                </p>
              </div>
            </div>

            {/* Quick Controls Ribbon */}
            <div className="flex flex-wrap items-center gap-2">
              {/* Market Switcher */}
              <div className="flex items-center bg-[#0a0e16] border border-white/[0.08] rounded-xl p-1 text-xs font-semibold">
                <button
                  onClick={() => { setScannerMarket('IN'); changeTicker('RELIANCE.NS'); }}
                  className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 ${
                    scannerMarket === 'IN' 
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-bold shadow-sm' 
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <span>🇮🇳</span>
                  <span>NSE / BSE</span>
                </button>
                <button
                  onClick={() => { setScannerMarket('US'); changeTicker('NVDA'); }}
                  className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 ${
                    scannerMarket === 'US' 
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold shadow-sm' 
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <span>🇺🇸</span>
                  <span>US Markets</span>
                </button>
              </div>

              {/* Auto-Refresh Control */}
              <div className="flex items-center bg-[#0a0e16] border border-white/[0.08] rounded-xl p-1 text-xs">
                <div className="flex items-center gap-1.5 px-2 py-1 text-slate-400">
                  <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-cyan-400' : 'text-slate-400'}`} />
                  <span className="text-[11px] font-medium text-slate-400">Auto:</span>
                  <select
                    value={autoRefreshSecs}
                    onChange={(e) => setAutoRefreshSecs(Number(e.target.value))}
                    className="bg-transparent text-white font-mono text-xs focus:outline-none cursor-pointer"
                  >
                    <option value={10} className="bg-slate-900">10s</option>
                    <option value={15} className="bg-slate-900">15s</option>
                    <option value={30} className="bg-slate-900">30s</option>
                    <option value={60} className="bg-slate-900">60s</option>
                    <option value={0} className="bg-slate-900">Paused</option>
                  </select>
                  {autoRefreshSecs > 0 && (
                    <span className="text-[10px] font-mono font-bold text-cyan-400 px-1 bg-cyan-500/10 rounded">
                      {refreshCountdown}s
                    </span>
                  )}
                </div>
                <button
                  onClick={() => fetchData(false)}
                  disabled={loading}
                  className="p-1.5 bg-slate-800/80 hover:bg-slate-700 text-slate-300 hover:text-white rounded-lg transition"
                  title="Force Refresh Data Now"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-cyan-400' : ''}`} />
                </button>
              </div>

              {/* Desk Utilities Toolbar */}
              <div className="flex items-center bg-[#0a0e16] border border-white/[0.08] rounded-xl p-1 text-xs gap-1">
                {/* Audio Alerts */}
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
                  title={soundAlerts ? 'Audio Alerts Active' : 'Audio Alerts Muted'}
                >
                  {soundAlerts ? <Volume2 className="w-3.5 h-3.5 text-cyan-400" /> : <VolumeX className="w-3.5 h-3.5" />}
                  <span className="hidden sm:inline">{soundAlerts ? 'Audio' : 'Mute'}</span>
                </button>

                {/* Trader's Scratchpad Toggle */}
                <button
                  onClick={() => setScratchpadOpen(!scratchpadOpen)}
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition font-medium ${
                    scratchpadOpen
                      ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  title="Open Trader Journal & Execution Notes"
                >
                  <Edit3 className="w-3.5 h-3.5 text-amber-400" />
                  <span className="hidden sm:inline">Journal</span>
                </button>

                {/* Keyboard Shortcuts */}
                <button
                  onClick={() => setShowHotkeysModal(true)}
                  className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 transition font-medium"
                  title="Keyboard Shortcuts (Press '?')"
                >
                  <Keyboard className="w-3.5 h-3.5 text-cyan-400" />
                  <kbd className="px-1 rounded bg-slate-800 text-[10px] text-cyan-300 font-mono">?</kbd>
                </button>

                {/* Export CSV */}
                <button
                  onClick={handleExportIntradayCSV}
                  disabled={!data?.candles?.length}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-slate-400 hover:text-cyan-300 disabled:opacity-40 transition font-medium cursor-pointer"
                  title="Export Intraday Candle Data to CSV"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">CSV</span>
                </button>
              </div>
            </div>
          </header>

          {/* ── TICKER COMMAND BAR & SEARCH ── */}
          <div className="space-y-2 bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-2.5 sm:p-3 backdrop-blur-md shadow-md">
            {/* Pinned Watchlist */}
            {pinnedTickers.length > 0 && (
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs border-b border-white/[0.06] scrollbar-none">
                <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider pl-1 pr-1 shrink-0 flex items-center gap-1 font-mono">
                  <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                  Pinned:
                </span>
                {pinnedTickers.map(sym => (
                  <div
                    key={sym}
                    className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold shrink-0 transition border ${
                      ticker === sym
                        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 shadow-sm'
                        : 'bg-[#070a10] text-slate-300 border-white/[0.06] hover:border-slate-700'
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
                      title="Unpin ticker"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 md:pb-0 scrollbar-none">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider pl-1 pr-1 shrink-0 font-mono">
                  Active Desk:
                </span>
                {QUICK_TICKERS.filter(t => t.market === scannerMarket).map(t => (
                  <button
                    key={t.symbol}
                    onClick={() => changeTicker(t.symbol)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-semibold shrink-0 transition flex items-center gap-1.5 ${
                      ticker === t.symbol
                        ? 'bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-emerald-300 border border-emerald-500/40 font-bold shadow-sm'
                        : 'bg-[#070a10] hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-white/[0.06]'
                    }`}
                  >
                    <span>{t.name}</span>
                    <span className="text-[10px] text-slate-500 font-mono">({t.symbol.split('.')[0]})</span>
                  </button>
                ))}
              </div>

              <form onSubmit={handleSearchSubmit} className="relative w-full md:w-auto md:min-w-[280px]">
                <input
                  ref={searchInputRef}
                  type="text"
                  placeholder={`Search ${scannerMarket === 'IN' ? 'NSE stock (e.g. SBIN)' : 'US stock (e.g. AMD)'}... (Press '/')`}
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  className="w-full bg-[#070a10] border border-white/[0.1] rounded-xl pl-9 pr-14 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition font-mono"
                />
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                <kbd className="absolute right-3 top-2 px-1.5 py-0.5 rounded bg-slate-800 text-[10px] text-slate-400 font-mono border border-slate-700">
                  /
                </kbd>
              </form>
            </div>
          </div>

          {/* ── INSTITUTIONAL TRAP ALERT BANNER ── */}
          {data?.trap_detection && data.trap_detection.status !== 'NONE' && (
            <div className={`p-3.5 rounded-2xl border flex items-center gap-3 backdrop-blur-md shadow-lg ${
              data.trap_detection.status === 'BULL_TRAP'
                ? 'bg-rose-950/40 border-rose-500/40 text-rose-200'
                : 'bg-emerald-950/40 border-emerald-500/40 text-emerald-200'
            }`}>
              <AlertTriangle className={`w-5 h-5 shrink-0 ${data.trap_detection.status === 'BULL_TRAP' ? 'text-rose-400' : 'text-emerald-400'}`} />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-xs uppercase tracking-wide">{data.trap_detection.title}</span>
                  <InfoBadge infoKey="institutional_trap_detector" />
                </div>
                <p className="text-xs text-slate-300 mt-0.5">{data.trap_detection.desc}</p>
              </div>
            </div>
          )}

          {/* ── LOADING SKELETON ── */}
          {loading && !data && (
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2.5 sm:gap-3">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="bg-[#0b0f17]/80 border border-white/[0.06] rounded-2xl p-4 animate-pulse">
                  <div className="h-3 w-20 bg-slate-800 rounded mb-3" />
                  <div className="h-7 w-28 bg-slate-800 rounded mb-2" />
                  <div className="h-2 w-16 bg-slate-800/60 rounded" />
                </div>
              ))}
            </div>
          )}

          {/* ── ACTIVE TICKER HEADLINE TELEMETRY (6 Responsive Cards) ── */}
          {data && (
            <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2.5 sm:gap-3">
              {/* Card 1: LTP & Session Range */}
              <div className={`bg-[#0b0f17]/90 rounded-2xl p-3 sm:p-3.5 flex flex-col justify-between transition-all duration-300 shadow-md ${
                priceFlash === 'up'
                  ? 'border border-emerald-400/60 shadow-emerald-500/20'
                  : priceFlash === 'down'
                  ? 'border border-rose-400/60 shadow-rose-500/20'
                  : 'border border-white/[0.08] shadow-black/30'
              }`}>
                <div>
                  <div className="flex items-center justify-between gap-1">
                    <div className="flex items-center gap-1.5 truncate max-w-[130px]">
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
                    <InfoBadge infoKey="live_prices" />
                  </div>
                  <div className="mt-1">
                    <h2 className={`text-xl sm:text-2xl font-black font-mono tracking-tight tabular-nums transition-colors duration-300 ${
                      priceFlash === 'up' ? 'text-emerald-300' : priceFlash === 'down' ? 'text-rose-300' : 'text-white'
                    }`}>
                      {currSym}{data.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </h2>
                    <div className="flex items-center justify-between gap-1 mt-0.5">
                      <span className={`text-xs font-bold font-mono tabular-nums flex items-center ${data.change >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {data.change >= 0 ? <ArrowUpRight className="w-3.5 h-3.5 mr-0.5 shrink-0" /> : <ArrowDownRight className="w-3.5 h-3.5 mr-0.5 shrink-0" />}
                        {data.change >= 0 ? '+' : ''}{data.change} ({data.change >= 0 ? '+' : ''}{data.change_pct}%)
                      </span>
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
                          <span className={`text-[9px] font-bold font-mono px-1.5 py-0.5 rounded border tabular-nums ${
                            unrealized >= 0 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                          }`}>
                            {unrealized >= 0 ? '+' : ''}{currSym}{Math.round(unrealized)}
                          </span>
                        );
                      })()}
                    </div>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-white/[0.06] font-mono tabular-nums">
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
                            left: `${Math.max(0, Math.min(100, ((data.current_price - data.low) / (data.high - data.low)) * 100))}%`
                          }}
                        />
                      </div>
                      <div className="flex items-center justify-between text-[8px] text-slate-500 font-mono">
                        <span>Day Low</span>
                        <span>Day High</span>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>

              {/* Card 2: Intraday VWAP & Bands */}
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-3.5 flex flex-col justify-between shadow-md shadow-black/30">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                      Session VWAP
                      <InfoBadge infoKey="vwap" />
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border tabular-nums ${
                      data.vwap_dist_pct >= 0 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                    }`}>
                      {data.vwap_dist_pct >= 0 ? '+' : ''}{data.vwap_dist_pct}%
                    </span>
                  </div>
                  <div className="mt-1">
                    <p className="text-xl sm:text-2xl font-black font-mono tracking-tight text-cyan-400 tabular-nums">
                      {currSym}{data.vwap?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Bias:{' '}
                      <span className={`font-semibold ${data.vwap_bias.includes('BULL') ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {data.vwap_bias}
                      </span>
                    </p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-white/[0.06] font-mono tabular-nums">
                    <span>+2σ: <strong className="text-cyan-300">{currSym}{data.upper_band_2}</strong></span>
                    <span>-2σ: <strong className="text-cyan-300">{currSym}{data.lower_band_2}</strong></span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                    <span>Oversold Floor</span>
                    <span className="text-cyan-400">±2σ Envelope</span>
                    <span>Overbought Cap</span>
                  </div>
                </div>
              </div>

              {/* Card 3: Relative Performance vs Benchmark */}
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-3.5 flex flex-col justify-between shadow-md shadow-black/30">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                      Relative Strength
                      <InfoBadge infoKey="benchmark_relative_strength" />
                    </span>
                    <span className="text-[10px] font-mono text-slate-400 font-bold px-1.5 py-0.5 bg-[#070a10] border border-white/[0.06] rounded">
                      vs {data.relative_strength?.benchmark_name?.replace('NIFTY ', '') || 'Index'}
                    </span>
                  </div>
                  <div className="mt-1">
                    <p className={`text-xl sm:text-2xl font-black font-mono tracking-tight tabular-nums ${(data.relative_strength?.relative_perf_pct ?? data.relative_strength?.alpha_pct) >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {(data.relative_strength?.relative_perf_pct ?? data.relative_strength?.alpha_pct) >= 0 ? '+' : ''}{data.relative_strength?.relative_perf_pct ?? data.relative_strength?.alpha_pct}%
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5 truncate font-mono">
                      Spread: <span className="font-semibold text-slate-200">{data.relative_strength?.status || 'Neutral'}</span>
                    </p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-white/[0.06] font-mono tabular-nums">
                    <span>Idx: <strong className={data.relative_strength?.benchmark_change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{data.relative_strength?.benchmark_change_pct >= 0 ? '+' : ''}{data.relative_strength?.benchmark_change_pct}%</strong></span>
                    <span className="text-slate-300 truncate max-w-[130px]">{data.relative_strength?.status || 'Tracking'}</span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                    <span>Lagging</span>
                    <span className={(data.relative_strength?.relative_perf_pct ?? data.relative_strength?.alpha_pct) >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                      {(data.relative_strength?.relative_perf_pct ?? data.relative_strength?.alpha_pct) >= 0 ? 'Outperforming' : 'Underperforming'}
                    </span>
                    <span>Leading</span>
                  </div>
                </div>
              </div>

              {/* Card 4: Pre-Market Gap Intelligence */}
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-3.5 flex flex-col justify-between shadow-md shadow-black/30">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                      Pre-Market Gap
                      <InfoBadge infoKey="pre_market_gap" />
                    </span>
                    <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border uppercase tabular-nums ${
                      data.gap_analysis?.gap_pct >= 0 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'
                    }`}>
                      {data.gap_analysis?.gap_type?.replace(/_/g, ' ') || 'FLAT'}
                    </span>
                  </div>
                  <div className="mt-1">
                    <p className={`text-xl sm:text-2xl font-black font-mono tracking-tight tabular-nums ${data.gap_analysis?.gap_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {data.gap_analysis?.gap_pct >= 0 ? '+' : ''}{data.gap_analysis?.gap_pct}%
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5 font-mono">
                      Points: <span className="font-semibold text-slate-200">{currSym}{data.gap_analysis?.gap_pts}</span>
                    </p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-white/[0.06] font-mono tabular-nums">
                    <span>Prev: <strong className="text-slate-300">{currSym}{data.gap_analysis?.prev_close}</strong></span>
                    <span>Fill: <strong className={data.gap_analysis?.gap_filled ? 'text-emerald-400' : 'text-amber-400'}>
                      {data.gap_analysis?.gap_filled ? 'FILLED' : `OPEN (${currSym}${data.gap_analysis?.gap_fill_dist})`}
                    </strong></span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                    <span>Gap Open</span>
                    <span className="text-amber-400 truncate max-w-[130px]">{data.gap_analysis?.directive?.split('—')[0] || 'Gap Setup'}</span>
                    <span>Fill Level</span>
                  </div>
                </div>
              </div>

              {/* Card 5: Composite Quant Bias Score */}
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-3.5 flex flex-col justify-between shadow-md shadow-black/30">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 flex items-center gap-1">
                      Quant Bias
                      <InfoBadge infoKey="intraday_quant_score" />
                    </span>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${
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
                      <span className={`text-xl sm:text-2xl font-black font-mono tracking-tight tabular-nums ${data.signals.quant_score >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
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

                <div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-white/[0.06] font-mono tabular-nums">
                    <span>Bull: <strong className="text-emerald-400">{data.signals?.bullish_count || 0}</strong></span>
                    <span>Bear: <strong className="text-rose-400">{data.signals?.bearish_count || 0}</strong></span>
                    <span>Conf: <strong className="text-cyan-300">{Math.round((data.signals?.bullish_count || 0) / Math.max(1, (data.signals?.bullish_count || 0) + (data.signals?.bearish_count || 0)) * 100)}%</strong></span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                    <span>Bearish</span>
                    <span className="text-slate-400">{data.signals?.risk_regime || 'Multi-Model Engine'}</span>
                    <span>Bullish</span>
                  </div>
                </div>
              </div>

              {/* Card 6: Supertrend & Momentum Signal */}
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-3.5 flex flex-col justify-between shadow-md shadow-black/30">
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
                    <p className={`text-xl sm:text-2xl font-black font-mono tracking-tight tabular-nums ${data.supertrend_dir === 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {currSym}{data.supertrend?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                    </p>
                    <p className="text-xs text-slate-400 mt-0.5 font-mono tabular-nums">
                      RSI:{' '}
                      <span className={`font-bold ${data.rsi >= 70 ? 'text-rose-400' : data.rsi <= 30 ? 'text-emerald-400' : 'text-slate-200'}`}>
                        {data.rsi?.toFixed(1)}
                      </span>
                      {data.rsi >= 70 && <span className="text-rose-400 ml-1 text-[9px] font-bold">OB</span>}
                      {data.rsi <= 30 && <span className="text-emerald-400 ml-1 text-[9px] font-bold">OS</span>}
                    </p>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 mt-2 pt-2 border-t border-white/[0.06] font-mono tabular-nums">
                    <span>ATR: <strong className="text-amber-400">{currSym}{data.atr ? Number(data.atr).toFixed(1) : '—'}</strong></span>
                    <span>EMA: <strong className={data.ema9 > data.ema21 ? 'text-emerald-400' : 'text-rose-400'}>{data.ema9 > data.ema21 ? '9>21 Bull' : '9<21 Bear'}</strong></span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[8px] text-slate-500 font-mono">
                    <span>Stop Level</span>
                    <span className="text-slate-400">Trailing Anchor</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── CIRCUIT LIMIT CORRIDOR & AUCTION FREEZE MONITOR ── */}
          {data?.circuit_bands && (
            <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-4 backdrop-blur-md shadow-lg shadow-black/30 space-y-2.5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div className={`p-1.5 rounded-lg border ${
                    data.circuit_bands.circuit_status === 'APPROACHING_UPPER_CIRCUIT'
                      ? 'bg-amber-500/15 border-amber-500/30 text-amber-400'
                      : data.circuit_bands.circuit_status === 'APPROACHING_LOWER_CIRCUIT'
                      ? 'bg-rose-500/15 border-rose-500/30 text-rose-400'
                      : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                  }`}>
                    {data.circuit_bands.circuit_status !== 'NORMAL' ? (
                      <AlertTriangle className="w-4 h-4 shrink-0" />
                    ) : (
                      <ShieldCheck className="w-4 h-4 shrink-0" />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-xs font-bold text-white tracking-wide uppercase font-mono">
                        Circuit Limits &amp; Volatility Corridor
                      </h3>
                      <InfoBadge
                        title="Regulatory Circuit Corridor"
                        what="The mandatory price collar (±10% for Indian securities, ±20% for US) set from previous close. Orders beyond these limits freeze or halt."
                        why="Crucial for intraday traders to avoid having capital locked in limit-up / limit-down circuit freezes."
                        interpretation="Approaching (<1%) indicates extreme directional pressure with imminent auction freeze risk."
                      />
                    </div>
                    <p className="text-[11px] text-slate-400 font-mono">
                      Band: <strong className="text-slate-200">±{data.circuit_bands.band_pct}%</strong> from Prev Close ({currSym}{data.prev_close || data.open})
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2 self-start sm:self-auto">
                  {data.circuit_bands.circuit_status === 'APPROACHING_UPPER_CIRCUIT' ? (
                    <span className="px-2.5 py-1 rounded-full text-xs font-bold font-mono bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> ⚠️ UPPER CIRCUIT IMMINENT (&lt;1% away)
                    </span>
                  ) : data.circuit_bands.circuit_status === 'APPROACHING_LOWER_CIRCUIT' ? (
                    <span className="px-2.5 py-1 rounded-full text-xs font-bold font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse flex items-center gap-1.5">
                      <AlertTriangle className="w-3.5 h-3.5" /> ⚠️ LOWER CIRCUIT IMMINENT (&lt;1% away)
                    </span>
                  ) : (
                    <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5" /> Normal Trading Band (±{data.circuit_bands.band_pct}%)
                    </span>
                  )}
                </div>
              </div>

              {/* Visual Corridor Bar & Boundaries */}
              {(() => {
                const cb = data.circuit_bands;
                const lower = cb.lower_band || (data.current_price * 0.9);
                const upper = cb.upper_band || (data.current_price * 1.1);
                const bandRange = Math.max(0.01, upper - lower);
                const currPos = Math.max(0, Math.min(100, ((data.current_price - lower) / bandRange) * 100));
                const prevClosePos = data.prev_close ? Math.max(0, Math.min(100, ((data.prev_close - lower) / bandRange) * 100)) : 50;
                const lowPos = data.low ? Math.max(0, Math.min(100, ((data.low - lower) / bandRange) * 100)) : currPos;
                const highPos = data.high ? Math.max(0, Math.min(100, ((data.high - lower) / bandRange) * 100)) : currPos;
                const sessionRangeWidth = Math.max(1.5, highPos - lowPos);

                return (
                  <div className="space-y-1.5 pt-1">
                    {/* Flank prices and central track */}
                    <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-center">
                      {/* Left Flank: Lower Circuit */}
                      <div className="md:col-span-3 p-2.5 rounded-xl bg-[#070a10] border border-rose-500/20 flex items-center justify-between">
                        <div>
                          <span className="text-[10px] text-slate-400 uppercase font-mono font-bold block">
                            Lower Circuit (LC)
                          </span>
                          <span className="text-base font-black font-mono text-rose-400 tabular-nums">
                            {currSym}{lower.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30">
                            -{cb.dist_to_lower_pct}% cushion
                          </span>
                          <span className="text-[9px] text-slate-500 block font-mono mt-0.5">Floor Freeze</span>
                        </div>
                      </div>

                      {/* Center Track: Full allowable band representation */}
                      <div className="md:col-span-6 px-1 space-y-1">
                        <div className="relative w-full h-4 bg-slate-950 rounded-full border border-slate-800 overflow-visible">
                          {/* Active day session range shaded bar */}
                          <div
                            className="absolute top-0 bottom-0 bg-gradient-to-r from-rose-500/25 via-cyan-500/25 to-emerald-500/25 rounded-full border-t border-b border-cyan-400/30"
                            style={{
                              left: `${lowPos}%`,
                              width: `${sessionRangeWidth}%`,
                            }}
                            title={`Session Range: ${currSym}${data.low} - ${currSym}${data.high}`}
                          />

                          {/* Prev Close Anchor Marker */}
                          <div
                            className="absolute top-[-2px] bottom-[-2px] w-0.5 bg-slate-400 z-10"
                            style={{ left: `${prevClosePos}%` }}
                            title={`Previous Close: ${currSym}${data.prev_close}`}
                          >
                            <span className="absolute -top-3.5 -translate-x-1/2 text-[8px] font-mono text-slate-400 font-bold whitespace-nowrap">
                              PC
                            </span>
                          </div>

                          {/* Current Price Pin */}
                          <div
                            className="absolute top-[-4px] bottom-[-4px] w-2 -ml-1 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] z-20 transition-all duration-300"
                            style={{ left: `${currPos}%` }}
                            title={`Current Price: ${currSym}${data.current_price}`}
                          />
                        </div>

                        {/* Track Legend */}
                        <div className="flex items-center justify-between text-[9px] text-slate-500 font-mono pt-1">
                          <span>LC ({currSym}{lower.toFixed(0)})</span>
                          <span className="text-slate-400">
                            LTP: <strong className="text-cyan-300">{currSym}{data.current_price}</strong> ({currPos.toFixed(0)}% of band)
                          </span>
                          <span>UC ({currSym}{upper.toFixed(0)})</span>
                        </div>
                      </div>

                      {/* Right Flank: Upper Circuit */}
                      <div className="md:col-span-3 p-2.5 rounded-xl bg-[#070a10] border border-emerald-500/20 flex items-center justify-between">
                        <div>
                          <span className="text-[10px] text-slate-400 uppercase font-mono font-bold block">
                            Upper Circuit (UC)
                          </span>
                          <span className="text-base font-black font-mono text-emerald-400 tabular-nums">
                            {currSym}{upper.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                            +{cb.dist_to_upper_pct}% headroom
                          </span>
                          <span className="text-[9px] text-slate-500 block font-mono mt-0.5">Ceiling Freeze</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* ── MAIN WORKSTATION GRID (Chart & Plan Left, Microstructure Sidebar Right) ── */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">

            {/* ── LEFT DESK: CHART + ACTIONABLE BATTLE PLAN + JOURNAL (8 Spans) ── */}
            <div className={`xl:col-span-8 space-y-4 ${
              fullscreenChart ? 'fixed inset-0 z-[150] bg-[#06080d] p-4 sm:p-6 overflow-y-auto' : ''
            }`}>
              
              {/* Candlestick & Technical Canvas Card */}
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-5 backdrop-blur-md shadow-xl space-y-3">
                
                {/* Horizontal Tool Ribbon */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-3 border-b border-white/[0.06]">
                  {/* Left: Timeframe pills & Zoom */}
                  <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0 scrollbar-none">
                    {/* Timeframe Selector */}
                    <div className="flex items-center gap-1 bg-[#070a10] p-1 rounded-xl border border-white/[0.06] shrink-0">
                      {TIMEFRAMES.map((tf) => (
                        <button
                          key={tf.label}
                          onClick={() => { setCandleInterval(tf.interval); setPeriod(tf.period); }}
                          className={`px-2.5 py-1 text-xs font-semibold rounded-lg transition ${
                            candleInterval === tf.interval
                              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm font-bold'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {tf.label}
                        </button>
                      ))}
                    </div>

                    {/* Viewport Zoom */}
                    <div className="flex items-center gap-1 bg-[#070a10] p-1 rounded-xl border border-white/[0.06] shrink-0">
                      {[
                        { id: 'all', label: 'All' },
                        { id: '60', label: '60b' },
                        { id: '30', label: '30b' },
                      ].map((z) => (
                        <button
                          key={z.id}
                          onClick={() => setCandleSlice(z.id)}
                          className={`px-2 py-1 text-[11px] font-semibold rounded-lg transition ${
                            candleSlice === z.id
                              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 font-bold shadow-sm'
                              : 'text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {z.label}
                        </button>
                      ))}
                    </div>

                    {/* Candle Mode */}
                    <button
                      onClick={() => setCandleMode(candleMode === 'regular' ? 'heikin_ashi' : 'regular')}
                      className={`px-2.5 py-1 text-xs font-semibold rounded-xl transition border shrink-0 flex items-center gap-1 ${
                        candleMode === 'heikin_ashi'
                          ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 font-bold'
                          : 'bg-[#070a10] text-slate-400 border-white/[0.06] hover:text-white'
                      }`}
                      title="Toggle Heikin-Ashi (HotKey: 'K')"
                    >
                      <span>🥢</span>
                      <span>{candleMode === 'heikin_ashi' ? 'Heikin-Ashi' : 'Candles'}</span>
                    </button>
                  </div>

                  {/* Right: Overlays Dropdown, Price Alert, Fullscreen */}
                  <div className="flex items-center gap-2 shrink-0">
                    {/* Overlays Popover Toggle */}
                    <div className="relative">
                      <button
                        onClick={() => setShowOverlaysMenu(!showOverlaysMenu)}
                        className={`px-2.5 py-1 text-xs font-semibold rounded-xl transition border flex items-center gap-1.5 ${
                          showOverlaysMenu || activeOverlaysCount > 0
                            ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/40 font-bold shadow-sm'
                            : 'bg-[#070a10] text-slate-400 border-white/[0.06] hover:text-white'
                        }`}
                      >
                        <Layers className="w-3.5 h-3.5" />
                        <span>Overlays</span>
                        <span className="px-1.5 py-0.2 rounded-full bg-cyan-400 text-slate-950 font-black text-[9px]">
                          {activeOverlaysCount}
                        </span>
                        <ChevronDown className={`w-3 h-3 transition-transform ${showOverlaysMenu ? 'rotate-180' : ''}`} />
                      </button>

                      {/* Dropdown Menu */}
                      {showOverlaysMenu && (
                        <div className="absolute right-0 mt-2 z-50 w-64 bg-[#0a0e16] border border-white/10 rounded-2xl p-2.5 shadow-2xl backdrop-blur-xl space-y-1">
                          <div className="flex items-center justify-between pb-2 border-b border-white/[0.06] px-1 text-xs font-bold text-slate-300">
                            <span>Chart Indicators &amp; Levels</span>
                            <button onClick={() => setShowOverlaysMenu(false)} className="text-slate-500 hover:text-white">
                              <X className="w-3.5 h-3.5" />
                            </button>
                          </div>
                          <div className="space-y-1 max-h-72 overflow-y-auto scrollbar-thin">
                            {[
                              { label: 'Session VWAP', active: showVWAP, toggle: () => setShowVWAP(!showVWAP), color: 'bg-cyan-400' },
                              { label: '±2σ VWAP Bands', active: showVWAPBands, toggle: () => setShowVWAPBands(!showVWAPBands), color: 'bg-cyan-300' },
                              { label: 'Supertrend Line', active: showSupertrend, toggle: () => setShowSupertrend(!showSupertrend), color: 'bg-emerald-400' },
                              { label: 'EMA 9 / 21 Crossover', active: showEMA, toggle: () => setShowEMA(!showEMA), color: 'bg-purple-400' },
                              { label: '200 EMA Anchor', active: showEMA200, toggle: () => setShowEMA200(!showEMA200), color: 'bg-amber-400' },
                              { label: 'ORB 15m Box', active: showORB, toggle: () => setShowORB(!showORB), color: 'bg-amber-500' },
                              { label: 'Camarilla H3/H4 & L3/L4', active: showCamarilla, toggle: () => setShowCamarilla(!showCamarilla), color: 'bg-rose-400' },
                              { label: 'Previous Day High/Low', active: showPDH, toggle: () => setShowPDH(!showPDH), color: 'bg-amber-300' },
                              { label: 'Central Pivot Range (CPR)', active: showCPR, toggle: () => setShowCPR(!showCPR), color: 'bg-indigo-400' },
                            ].map((item, idx) => (
                              <button
                                key={idx}
                                onClick={item.toggle}
                                className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs transition ${
                                  item.active ? 'bg-cyan-500/10 text-cyan-200 font-semibold' : 'text-slate-400 hover:bg-slate-900'
                                }`}
                              >
                                <div className="flex items-center gap-2">
                                  <span className={`w-2 h-2 rounded-full ${item.color}`} />
                                  <span>{item.label}</span>
                                </div>
                                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${item.active ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'bg-slate-800 text-slate-500'}`}>
                                  {item.active ? 'ON' : 'OFF'}
                                </span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Price Alert Mini */}
                    {data && (
                      <div className={`hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-xl border text-xs ${
                        alertTriggered
                          ? 'bg-amber-500/15 border-amber-500/40 text-amber-300 shadow-sm'
                          : 'bg-[#070a10] border-white/[0.06] text-slate-400'
                      }`}>
                        {alertTriggered ? <Bell className="w-3.5 h-3.5 text-amber-400 animate-bounce" /> : <BellOff className="w-3.5 h-3.5 text-slate-500" />}
                        <select
                          value={alertAbove ? 'above' : 'below'}
                          onChange={e => { setAlertAbove(e.target.value === 'above'); setAlertTriggered(false); }}
                          className="bg-transparent text-[11px] font-mono focus:outline-none cursor-pointer text-slate-300"
                        >
                          <option value="above" className="bg-slate-900">≥</option>
                          <option value="below" className="bg-slate-900">≤</option>
                        </select>
                        <input
                          type="number"
                          placeholder={data.current_price?.toFixed(0)}
                          value={alertPrice}
                          onChange={e => { setAlertPrice(e.target.value); setAlertTriggered(false); }}
                          className="w-14 bg-transparent font-mono text-xs text-white placeholder-slate-600 focus:outline-none tabular-nums"
                        />
                        {alertPrice && (
                          <button
                            onClick={() => { setAlertPrice(''); setAlertTriggered(false); }}
                            className="text-slate-400 hover:text-rose-400 transition p-0.5"
                            title="Clear price alert"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        )}
                      </div>
                    )}

                    {/* Fullscreen Button */}
                    <button
                      onClick={() => setFullscreenChart(!fullscreenChart)}
                      className="p-1.5 rounded-xl bg-[#070a10] border border-white/[0.06] text-slate-400 hover:text-white transition"
                      title={fullscreenChart ? 'Exit Fullscreen' : 'Fullscreen Chart (Press F)'}
                    >
                      {fullscreenChart ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                {/* Live / Crosshair Inspection HUD Strip */}
                <div className="min-h-[28px] py-1 flex items-center justify-between text-[11px] font-mono text-slate-400 px-2.5 bg-[#070a10] rounded-xl border border-white/[0.06] overflow-x-auto scrollbar-none">
                  {hoveredCandle || (candles.length > 0 ? candles[candles.length - 1] : null) ? (() => {
                    const c = hoveredCandle || candles[candles.length - 1];
                    const isLive = !hoveredCandle;
                    return (
                      <div className="flex items-center gap-3 w-full justify-between shrink-0 tabular-nums">
                        <div className="flex items-center gap-3">
                          {isLive ? (
                            <span className="px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-bold flex items-center gap-1 font-sans">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> LIVE
                            </span>
                          ) : (
                            <span className="px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-300 text-[10px] font-bold font-sans">
                              INSPECT
                            </span>
                          )}
                          {candleMode === 'heikin_ashi' && (
                            <span className="px-1.5 py-0.2 rounded bg-purple-500/20 text-purple-300 text-[10px] font-bold font-sans">
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
                        <div className="hidden md:flex items-center text-[10px] text-slate-500 font-sans">
                          {isLive ? 'Hover chart to inspect ticks' : 'Crosshair active'}
                        </div>
                      </div>
                    );
                  })() : (
                    <span className="text-slate-500 italic text-[10px]">Awaiting high-frequency market stream...</span>
                  )}
                </div>

                {/* High-Resolution Candlestick SVG Canvas */}
                <div className="relative w-full overflow-hidden bg-[#070a10] rounded-2xl border border-white/[0.06]">
                  {loading && (
                    <div className="absolute inset-0 bg-[#070a10]/80 backdrop-blur-sm flex items-center justify-center z-20">
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
                        Intraday chart data may be temporarily unavailable for {ticker}. Select an active stock to continue:
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
                      className="w-full h-auto cursor-crosshair select-none touch-none"
                      onTouchStart={handleChartTouch}
                      onTouchMove={handleChartTouch}
                      onTouchEnd={handleChartTouchEnd}
                      onTouchCancel={handleChartTouchEnd}
                      style={{ touchAction: "none" }}
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
                              strokeOpacity={0.25}
                            />
                            <text
                              x={chartWidth - padding.right + 6}
                              y={y + 3}
                              fill="#64748b"
                              fontSize="9"
                              fontFamily="monospace"
                              textAnchor="start"
                            >
                              {p.toFixed(2)}
                            </text>
                          </g>
                        );
                      })}

                      {/* Previous Day High / Low Lines */}
                      {showPDH && data?.pivots?.daily_levels && (() => {
                        const { pdh, pdl } = data.pivots.daily_levels;
                        return (
                          <>
                            {pdh > 0 && (
                              <g>
                                <line x1={padding.left} y1={yScale(pdh)} x2={chartWidth - padding.right} y2={yScale(pdh)} stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 3" strokeOpacity={0.75} />
                                <text x={chartWidth - padding.right + 6} y={yScale(pdh) + 3} fill="#f59e0b" fontSize="8" fontFamily="monospace" fontWeight="bold">PDH</text>
                              </g>
                            )}
                            {pdl > 0 && (
                              <g>
                                <line x1={padding.left} y1={yScale(pdl)} x2={chartWidth - padding.right} y2={yScale(pdl)} stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 3" strokeOpacity={0.75} />
                                <text x={chartWidth - padding.right + 6} y={yScale(pdl) + 3} fill="#f59e0b" fontSize="8" fontFamily="monospace" fontWeight="bold">PDL</text>
                              </g>
                            )}
                          </>
                        );
                      })()}

                      {/* Opening Range Breakout (ORB 15m) Box */}
                      {showORB && data?.orb && (() => {
                        const { high_15m, low_15m } = data.orb;
                        if (!high_15m || !low_15m) return null;
                        const yH = yScale(high_15m);
                        const yL = yScale(low_15m);
                        const rectW = chartWidth - padding.left - padding.right;
                        return (
                          <g>
                            <rect x={padding.left} y={Math.min(yH, yL)} width={rectW} height={Math.max(Math.abs(yL - yH), 2)} fill="#f59e0b" fillOpacity={0.04} stroke="#f59e0b" strokeWidth="0.8" strokeDasharray="3 3" />
                            <text x={padding.left + 6} y={yH - 4} fill="#f59e0b" fontSize="8" fontFamily="monospace" fontWeight="bold">ORB 15m High ({high_15m})</text>
                            <text x={padding.left + 6} y={yL + 10} fill="#f59e0b" fontSize="8" fontFamily="monospace" fontWeight="bold">ORB 15m Low ({low_15m})</text>
                          </g>
                        );
                      })()}

                      {/* CPR Support/Resistance Cloud */}
                      {showCPR && data?.pivots?.cpr && (() => {
                        const { tc, bc, pivot } = data.pivots.cpr;
                        if (!tc || !bc) return null;
                        const yTc = yScale(tc);
                        const yBc = yScale(bc);
                        const yP = yScale(pivot);
                        return (
                          <g>
                            <rect x={padding.left} y={Math.min(yTc, yBc)} width={chartWidth - padding.left - padding.right} height={Math.max(Math.abs(yBc - yTc), 2)} fill="#6366f1" fillOpacity={0.06} />
                            <line x1={padding.left} y1={yP} x2={chartWidth - padding.right} y2={yP} stroke="#6366f1" strokeWidth="1" strokeDasharray="3 2" strokeOpacity={0.7} />
                            <text x={padding.left + 6} y={yP - 3} fill="#818cf8" fontSize="8" fontFamily="monospace">CPR Pivot ({pivot})</text>
                          </g>
                        );
                      })()}

                      {/* Camarilla H3/H4 and L3/L4 Lines */}
                      {showCamarilla && data?.pivots?.camarilla && (() => {
                        const { h4, h3, l3, l4 } = data.pivots.camarilla;
                        return (
                          <>
                            {h4 > 0 && (
                              <g>
                                <line x1={padding.left} y1={yScale(h4)} x2={chartWidth - padding.right} y2={yScale(h4)} stroke="#10b981" strokeWidth="1" strokeDasharray="2 2" strokeOpacity={0.8} />
                                <text x={padding.left + 6} y={yScale(h4) - 3} fill="#10b981" fontSize="7" fontFamily="monospace">Cam H4 Breakout</text>
                              </g>
                            )}
                            {h3 > 0 && (
                              <g>
                                <line x1={padding.left} y1={yScale(h3)} x2={chartWidth - padding.right} y2={yScale(h3)} stroke="#f43f5e" strokeWidth="1" strokeDasharray="2 2" strokeOpacity={0.7} />
                                <text x={padding.left + 6} y={yScale(h3) - 3} fill="#f43f5e" fontSize="7" fontFamily="monospace">Cam H3 Resistance</text>
                              </g>
                            )}
                            {l3 > 0 && (
                              <g>
                                <line x1={padding.left} y1={yScale(l3)} x2={chartWidth - padding.right} y2={yScale(l3)} stroke="#10b981" strokeWidth="1" strokeDasharray="2 2" strokeOpacity={0.7} />
                                <text x={padding.left + 6} y={yScale(l3) + 9} fill="#10b981" fontSize="7" fontFamily="monospace">Cam L3 Support</text>
                              </g>
                            )}
                            {l4 > 0 && (
                              <g>
                                <line x1={padding.left} y1={yScale(l4)} x2={chartWidth - padding.right} y2={yScale(l4)} stroke="#f43f5e" strokeWidth="1" strokeDasharray="2 2" strokeOpacity={0.8} />
                                <text x={padding.left + 6} y={yScale(l4) + 9} fill="#f43f5e" fontSize="7" fontFamily="monospace">Cam L4 Breakdown</text>
                              </g>
                            )}
                          </>
                        );
                      })()}

                      {/* VWAP Volatility Bands Area */}
                      {showVWAPBands && (
                        <path
                          d={`${candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.upper_band_2 || c.vwap)}`, '')} ${candles.slice().reverse().reduce((acc, c, i) => `${acc} L ${xScale(candles.length - 1 - i)} ${yScale(c.lower_band_2 || c.vwap)}`, '')} Z`}
                          fill="rgba(56, 189, 248, 0.04)"
                        />
                      )}

                      {/* VWAP ±2σ Band Lines */}
                      {showVWAPBands && (
                        <>
                          <path
                            d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.upper_band_2 || c.vwap)}`, '')}
                            fill="none" stroke="#38bdf8" strokeWidth="0.8" strokeDasharray="2 2" strokeOpacity={0.5}
                          />
                          <path
                            d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.lower_band_2 || c.vwap)}`, '')}
                            fill="none" stroke="#38bdf8" strokeWidth="0.8" strokeDasharray="2 2" strokeOpacity={0.5}
                          />
                        </>
                      )}

                      {/* Session VWAP Line */}
                      {showVWAP && (
                        <path
                          d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.vwap || c.close)}`, '')}
                          fill="none" stroke="#0284c7" strokeWidth="2" strokeOpacity={0.9}
                        />
                      )}

                      {/* EMA 9 Line */}
                      {showEMA && (
                        <path
                          d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.ema9 || c.close)}`, '')}
                          fill="none" stroke="#a855f7" strokeWidth="1.2" strokeOpacity={0.8}
                        />
                      )}

                      {/* EMA 21 Line */}
                      {showEMA && (
                        <path
                          d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.ema21 || c.close)}`, '')}
                          fill="none" stroke="#ec4899" strokeWidth="1.2" strokeOpacity={0.8}
                        />
                      )}

                      {/* 200 EMA Line */}
                      {showEMA200 && (
                        <path
                          d={candles.filter(c => c.ema200 > 0).reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(candles.indexOf(c))} ${yScale(c.ema200)}`, '')}
                          fill="none" stroke="#f59e0b" strokeWidth="1.8" strokeOpacity={0.9}
                        />
                      )}

                      {/* Supertrend Stepped Line */}
                      {showSupertrend && (
                        <path
                          d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(c.supertrend || c.close)}`, '')}
                          fill="none"
                          stroke={data.supertrend_dir === 1 ? '#10b981' : '#f43f5e'}
                          strokeWidth="1.8"
                          strokeDasharray="4 2"
                        />
                      )}

                      {/* Candlestick Bars */}
                      {candles.map((c, i) => {
                        const isUp = c.close >= c.open;
                        const x = xScale(i);
                        const yOpen = yScale(c.open);
                        const yClose = yScale(c.close);
                        const yHigh = yScale(c.high);
                        const yLow = yScale(c.low);
                        const top = Math.min(yOpen, yClose);
                        const height = Math.max(Math.abs(yOpen - yClose), 1);
                        const color = isUp ? '#10b981' : '#f43f5e';

                        return (
                          <g key={i}>
                            {/* Candle Wick */}
                            <line x1={x} y1={yHigh} x2={x} y2={yLow} stroke={color} strokeWidth="1.2" strokeOpacity={0.9} />
                            {/* Candle Body */}
                            <rect
                              x={x - candleWidth / 2}
                              y={top}
                              width={candleWidth}
                              height={height}
                              fill={color}
                              rx={0.5}
                            />
                          </g>
                        );
                      })}

                      {/* Interactive Price Alert Target Overlay */}
                      {alertPrice && !isNaN(parseFloat(alertPrice)) && (() => {
                        const ap = parseFloat(alertPrice);
                        if (ap < priceMin || ap > priceMax) return null;
                        const y = yScale(ap);
                        const isTriggered = alertTriggered;
                        const strokeColor = isTriggered ? '#ef4444' : '#f59e0b';
                        const bgColor = isTriggered ? '#7f1d1d' : '#78350f';
                        const textColor = isTriggered ? '#fca5a5' : '#fef3c7';
                        return (
                          <g key="chart-alert-overlay" className="transition-all duration-300">
                            {/* Halo glow when triggered */}
                            {isTriggered && (
                              <line
                                x1={padding.left}
                                y1={y}
                                x2={chartWidth - padding.right}
                                y2={y}
                                stroke="#ef4444"
                                strokeWidth="5"
                                strokeOpacity="0.25"
                              />
                            )}
                            {/* Alert dashed line */}
                            <line
                              x1={padding.left}
                              y1={y}
                              x2={chartWidth - padding.right}
                              y2={y}
                              stroke={strokeColor}
                              strokeWidth="1.5"
                              strokeDasharray="5 3"
                              strokeOpacity={isTriggered ? 1 : 0.85}
                            />
                            {/* Left Banner on chart canvas */}
                            <rect
                              x={padding.left + 6}
                              y={y - 14}
                              width={isTriggered ? 112 : 98}
                              height={13}
                              fill={bgColor}
                              stroke={strokeColor}
                              strokeWidth="0.8"
                              rx="2"
                            />
                            <text
                              x={padding.left + 10}
                              y={y - 4}
                              fill={textColor}
                              fontSize="7.5"
                              fontFamily="monospace"
                              fontWeight="bold"
                            >
                              {isTriggered ? '⚡ ALERT FIRED' : `ALERT ${alertAbove ? '≥' : '≤'} ${currSym}${ap.toFixed(2)}`}
                            </text>
                            {/* Right Y-Axis Badge */}
                            <rect
                              x={chartWidth - padding.right + 2}
                              y={y - 8}
                              width={padding.right - 4}
                              height={16}
                              fill={strokeColor}
                              rx="3"
                            />
                            <text
                              x={chartWidth - padding.right + 5}
                              y={y + 3}
                              fill="#000000"
                              fontSize="8"
                              fontFamily="monospace"
                              fontWeight="bold"
                            >
                              🔔{alertAbove ? '≥' : '≤'}{ap.toFixed(1)}
                            </text>
                          </g>
                        );
                      })()}

                      {/* Interactive Crosshair */}
                      {hoveredX !== null && hoveredX >= padding.left && hoveredX <= chartWidth - padding.right && (
                        <g>
                          {/* Vertical Time Line */}
                          <line
                            x1={hoveredX}
                            y1={padding.top}
                            x2={hoveredX}
                            y2={chartHeight - padding.bottom}
                            stroke="#38bdf8"
                            strokeWidth="1"
                            strokeDasharray="2 2"
                            strokeOpacity={0.6}
                          />
                          {/* Horizontal Price Line */}
                          {hoveredY !== null && hoveredY >= padding.top && hoveredY <= chartHeight - padding.bottom && (
                            <>
                              <line
                                x1={padding.left}
                                y1={hoveredY}
                                x2={chartWidth - padding.right}
                                y2={hoveredY}
                                stroke="#38bdf8"
                                strokeWidth="1"
                                strokeDasharray="2 2"
                                strokeOpacity={0.6}
                              />
                              {/* Price Label Badge */}
                              {(() => {
                                const innerH = chartHeight - padding.top - padding.bottom;
                                const pVal = priceMax - ((hoveredY - padding.top) / innerH) * (priceMax - priceMin);
                                return (
                                  <g>
                                    <rect
                                      x={chartWidth - padding.right + 2}
                                      y={hoveredY - 8}
                                      width={padding.right - 4}
                                      height={16}
                                      fill="#0284c7"
                                      rx={3}
                                    />
                                    <text
                                      x={chartWidth - padding.right + 6}
                                      y={hoveredY + 3}
                                      fill="#ffffff"
                                      fontSize="8"
                                      fontFamily="monospace"
                                      fontWeight="bold"
                                    >
                                      {pVal.toFixed(2)}
                                    </text>
                                  </g>
                                );
                              })()}
                            </>
                          )}
                        </g>
                      )}

                      {/* X-Axis Time Labels */}
                      {candles.filter((_, i) => i % Math.max(Math.floor(candles.length / 6), 1) === 0).map((c, i, arr) => {
                        const originalIdx = candles.indexOf(c);
                        return (
                          <text
                            key={i}
                            x={xScale(originalIdx)}
                            y={chartHeight - padding.bottom + 16}
                            fill="#64748b"
                            fontSize="8"
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

                {/* Sub-Chart Selector & Canvas */}
                <div className="pt-2 border-t border-white/[0.06] space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-none text-xs">
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider font-mono pr-1">Sub-Chart:</span>
                      {[
                        { id: 'volume', label: 'Volume + MA' },
                        { id: 'rsi', label: 'RSI (14)' },
                        { id: 'macd', label: 'MACD (12,26,9)' },
                        { id: 'cvd', label: 'CVD Proxy' },
                        { id: 'atr', label: 'ATR (14)' },
                      ].map(sc => (
                        <button
                          key={sc.id}
                          onClick={() => setActiveSubChart(sc.id)}
                          className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition shrink-0 ${
                            activeSubChart === sc.id
                              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold shadow-sm'
                              : 'bg-[#070a10] text-slate-400 hover:text-white border border-white/[0.06]'
                          }`}
                        >
                          {sc.label}
                        </button>
                      ))}
                    </div>

                    <InfoBadge infoKey={activeSubChart === 'cvd' ? 'order_flow_delta' : activeSubChart === 'volume' ? 'order_flow_delta' : activeSubChart === 'macd' ? 'macd_cross' : 'rsi'} />
                  </div>

                  {/* Sub-Chart SVG Container */}
                  <div className="h-28 w-full bg-[#070a10] rounded-xl border border-white/[0.06] overflow-hidden p-1">
                    {activeSubChart === 'volume' && (
                      <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full cursor-crosshair select-none touch-none" onTouchStart={handleSubChartTouch} onTouchMove={handleSubChartTouch} onTouchEnd={handleChartTouchEnd} style={{ touchAction: "none" }}>
                        {(() => {
                          const maxVol = Math.max(...candles.map(c => c.volume || 0), 1);
                          return candles.map((c, i) => {
                            const isUp = c.close >= c.open;
                            const h = Math.max(((c.volume || 0) / maxVol) * 85, 2);
                            return (
                              <rect
                                key={i}
                                x={xScale(i) - candleWidth / 2}
                                y={100 - h}
                                width={candleWidth}
                                height={h}
                                fill={isUp ? '#10b981' : '#f43f5e'}
                                fillOpacity={0.65}
                                rx={0.5}
                              />
                            );
                          });
                        })()}
                      </svg>
                    )}

                    {activeSubChart === 'rsi' && (
                      <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full cursor-crosshair select-none touch-none" onTouchStart={handleSubChartTouch} onTouchMove={handleSubChartTouch} onTouchEnd={handleChartTouchEnd} style={{ touchAction: "none" }}>
                        {(() => {
                          const lastCandle = candles[candles.length - 1];
                          const activeCandle = hoveredCandle || lastCandle;
                          const activeRsi = (activeCandle?.rsi !== undefined && activeCandle?.rsi !== null) ? activeCandle.rsi : 50;
                          return (
                            <>
                              <rect x={padding.left} y={10} width={chartWidth - padding.left - padding.right} height={20} fill="#f43f5e" fillOpacity={0.04} />
                              <line x1={padding.left} y1={30} x2={chartWidth - padding.right} y2={30} stroke="#f43f5e" strokeDasharray="3 3" strokeOpacity={0.6} />
                              <text x={padding.left + 4} y={26} fill="#f43f5e" fontSize="8" fontFamily="monospace" fontWeight="bold">OB 70</text>
                              <line x1={padding.left} y1={50} x2={chartWidth - padding.right} y2={50} stroke="#475569" strokeDasharray="2 2" strokeOpacity={0.4} />
                              <line x1={padding.left} y1={70} x2={chartWidth - padding.right} y2={70} stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.6} />
                              <text x={padding.left + 4} y={82} fill="#10b981" fontSize="8" fontFamily="monospace" fontWeight="bold">OS 30</text>

                              <text x={chartWidth - padding.right - 10} y={16} fill="#94a3b8" fontSize="8" fontFamily="monospace" textAnchor="end">
                                RSI (14): <tspan fill={activeRsi >= 70 ? '#f43f5e' : activeRsi <= 30 ? '#10b981' : '#38bdf8'} fontWeight="bold">{activeRsi.toFixed(1)}</tspan>
                                {activeRsi >= 70 ? ' (Overbought)' : activeRsi <= 30 ? ' (Oversold)' : ' (Neutral)'}
                              </text>
                              <path
                                d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${100 - ((c.rsi ?? 50))}`, '')}
                                fill="none" stroke="#38bdf8" strokeWidth="1.8"
                              />
                            </>
                          );
                        })()}
                      </svg>
                    )}

                    {activeSubChart === 'macd' && (
                      <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full cursor-crosshair select-none touch-none" onTouchStart={handleSubChartTouch} onTouchMove={handleSubChartTouch} onTouchEnd={handleChartTouchEnd} style={{ touchAction: "none" }}>
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
                              <text x={padding.left + 4} y={14} fill="#94a3b8" fontSize="8" fontFamily="monospace" fontWeight="bold">
                                MACD: <tspan fill="#38bdf8">{(activeCandle?.macd || 0).toFixed(2)}</tspan> | Sig: <tspan fill="#f59e0b">{(activeCandle?.macd_signal || 0).toFixed(2)}</tspan> | Hist: <tspan fill={(activeCandle?.macd_histogram || 0) >= 0 ? '#10b981' : '#f43f5e'}>{(activeCandle?.macd_histogram || 0).toFixed(2)}</tspan>
                              </text>
                              <line x1={padding.left} y1={zeroY} x2={chartWidth - padding.right} y2={zeroY} stroke="#64748b" strokeOpacity={0.5} strokeDasharray="2 2" />
                              {candles.map((c, i) => {
                                const h = c.macd_histogram || 0;
                                const prevH = i > 0 ? (candles[i-1].macd_histogram || 0) : 0;
                                const isPos = h >= 0;
                                const isGrowing = isPos ? h >= prevH : h <= prevH;
                                const barColor = isPos ? (isGrowing ? '#10b981' : '#34d399') : (isGrowing ? '#f43f5e' : '#fb7185');
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
                                    fillOpacity={0.8}
                                    rx={0.5}
                                  />
                                );
                              })}
                              <path
                                d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${norm(c.macd || 0)}`, '')}
                                fill="none" stroke="#38bdf8" strokeWidth="1.6"
                              />
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
                      <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full cursor-crosshair select-none touch-none" onTouchStart={handleSubChartTouch} onTouchMove={handleSubChartTouch} onTouchEnd={handleChartTouchEnd} style={{ touchAction: "none" }}>
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
                                Cumulative Volume Delta: <tspan fill={(activeCandle?.cum_delta || 0) >= 0 ? '#eab308' : '#f43f5e'} fontWeight="bold">{(activeCandle?.cum_delta || 0) >= 0 ? '+' : ''}{(activeCandle?.cum_delta || 0).toLocaleString()} shares</tspan>
                              </text>
                              <line x1={padding.left} y1={zeroY} x2={chartWidth - padding.right} y2={zeroY} stroke="#64748b" strokeOpacity={0.5} strokeDasharray="2 2" />
                              <path
                                d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.cum_delta || 0) - minCvd) / cvdRange) * 80}`, '')}
                                fill="none" stroke="#eab308" strokeWidth="2"
                              />
                            </>
                          );
                        })()}
                      </svg>
                    )}

                    {activeSubChart === 'atr' && (
                      <svg viewBox={`0 0 ${chartWidth} 100`} className="w-full h-full cursor-crosshair select-none touch-none" onTouchStart={handleSubChartTouch} onTouchMove={handleSubChartTouch} onTouchEnd={handleChartTouchEnd} style={{ touchAction: "none" }}>
                        {(() => {
                          const atrVals = candles.map(c => c.atr || 0).filter(v => v > 0);
                          const minAtr = atrVals.length ? Math.min(...atrVals) * 0.85 : 0;
                          const maxAtr = atrVals.length ? Math.max(...atrVals) * 1.15 : 1;
                          const atrRange = (maxAtr - minAtr) || 1;
                          const lastCandle = candles[candles.length - 1];
                          const activeCandle = hoveredCandle || lastCandle;
                          const currentAtr = activeCandle?.atr || data?.atr || 0;
                          return (
                            <>
                              <text x={padding.left + 4} y={14} fill="#94a3b8" fontSize="8" fontFamily="monospace" fontWeight="bold">
                                ATR (14): <tspan fill="#f59e0b" fontWeight="bold">{currSym}{currentAtr}</tspan> | Dynamic 1.5× Stop: ±{currSym}{(currentAtr * 1.5).toFixed(2)}
                              </text>
                              <path
                                d={candles.reduce((acc, c, i) => `${acc} ${i === 0 ? 'M' : 'L'} ${xScale(i)} ${90 - (((c.atr || 0) - minAtr) / atrRange) * 70}`, '')}
                                fill="none" stroke="#f59e0b" strokeWidth="2"
                              />
                            </>
                          );
                        })()}
                      </svg>
                    )}
                  </div>
                </div>
              </div>

              {/* ── ACTIONABLE INTRADAY BATTLE PLAN CARD (Right beneath Chart) ── */}
              {data?.battle_plan?.entry_price && (
                <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-4 sm:p-5 backdrop-blur-md shadow-xl space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-white/[0.06]">
                    <div className="flex items-center gap-3">
                      <div className="p-2.5 bg-gradient-to-tr from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 rounded-2xl shadow-sm">
                        <Target className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm sm:text-base font-bold text-white tracking-wide">
                            Intraday Battle Plan: <span className="text-cyan-300">{data.battle_plan.setup_name}</span>
                          </h3>
                          <InfoBadge infoKey="intraday_battle_plan" />
                        </div>
                        <p className="text-xs text-slate-400 mt-0.5">
                          Institutional trigger, hard stop loss, and multi-tier profit targets
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleCopyPlan}
                        className="px-3 py-1.5 bg-cyan-500/15 hover:bg-cyan-500/25 text-cyan-300 border border-cyan-500/30 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-sm"
                      >
                        {planCopied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>{planCopied ? 'Copied!' : 'Copy Plan'}</span>
                      </button>

                      <button
                        onClick={() => {
                          setCalcEntry(data.battle_plan.entry_price?.toString() || '');
                          setCalcStop(data.battle_plan.stop_loss?.toString() || '');
                          setSidebarTab('calc');
                        }}
                        className="px-3 py-1.5 bg-purple-500/15 hover:bg-purple-500/25 text-purple-300 border border-purple-500/30 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-sm"
                      >
                        <Scale className="w-3.5 h-3.5" />
                        <span>Calc Risk</span>
                      </button>
                    </div>
                  </div>

                  {/* 4 Execution Pods */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3 text-xs font-mono">
                    {/* Entry */}
                    <div className="p-3 bg-[#070a10] border border-cyan-500/30 rounded-xl shadow-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-cyan-400 font-bold font-sans uppercase tracking-wider">ENTRY TRIGGER</span>
                        <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                      </div>
                      <span className="text-lg font-bold text-white block mt-1 tabular-nums">{currSym}{data.battle_plan.entry_price}</span>
                      <p className="text-[10px] text-slate-400 mt-0.5 font-sans truncate">{data.battle_plan.trigger_rule}</p>
                    </div>

                    {/* Stop Loss */}
                    <div className="p-3 bg-[#070a10] border border-rose-500/40 rounded-xl shadow-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-rose-400 font-bold font-sans uppercase tracking-wider">HARD STOP</span>
                        <span className="w-2 h-2 rounded-full bg-rose-400" />
                      </div>
                      <span className="text-lg font-bold text-rose-400 block mt-1 tabular-nums">{currSym}{data.battle_plan.stop_loss}</span>
                      <p className="text-[10px] text-slate-400 mt-0.5 font-sans">Risk: {currSym}{data.battle_plan.risk_per_share}/sh</p>
                    </div>

                    {/* Target 1 */}
                    <div className="p-3 bg-[#070a10] border border-emerald-500/40 rounded-xl shadow-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-emerald-400 font-bold font-sans uppercase tracking-wider">TARGET 1 (1.5R)</span>
                        <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      </div>
                      <span className="text-lg font-bold text-emerald-400 block mt-1 tabular-nums">{currSym}{data.battle_plan.target_1}</span>
                      <p className="text-[10px] text-slate-400 mt-0.5 font-sans">Scale out 50% & trail</p>
                    </div>

                    {/* Target 2 */}
                    <div className="p-3 bg-[#070a10] border border-purple-500/40 rounded-xl shadow-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-purple-400 font-bold font-sans uppercase tracking-wider">TARGET 2 (2.5R)</span>
                        <span className="w-2 h-2 rounded-full bg-purple-400" />
                      </div>
                      <span className="text-lg font-bold text-purple-300 block mt-1 tabular-nums">{currSym}{data.battle_plan.target_2}</span>
                      <p className="text-[10px] text-slate-400 mt-0.5 font-sans">Full runner target</p>
                    </div>
                  </div>

                  {/* R:R Roadmap Strip */}
                  <div className="pt-2 border-t border-white/[0.06] flex items-center justify-between text-[11px] font-mono text-slate-400">
                    <span className="text-rose-400 font-semibold">🛑 Stop: {currSym}{data.battle_plan.stop_loss}</span>
                    <div className="flex-1 mx-3 h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                      <div className="w-1/4 bg-rose-500/60" />
                      <div className="w-2/4 bg-emerald-500/60" />
                      <div className="w-1/4 bg-purple-500/60" />
                    </div>
                    <span className="text-purple-300 font-semibold">⚖️ {data.battle_plan.rr_ratio || '1:2.0'}</span>
                    <span className="text-emerald-400 font-semibold ml-3">🎯 T2: {currSym}{data.battle_plan.target_2}</span>
                  </div>
                </div>
              )}

              {/* ── TRADER'S SCRATCHPAD DRAWER (Collapsible) ── */}
              {scratchpadOpen && (
                <div className="bg-[#0b0f17]/90 border border-amber-500/30 rounded-2xl p-4 sm:p-5 backdrop-blur-md shadow-xl space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-white/[0.06]">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-lg bg-amber-500/15 border border-amber-500/30 flex items-center justify-center">
                        <Edit3 className="w-3.5 h-3.5 text-amber-400" />
                      </div>
                      <div>
                        <h4 className="text-xs sm:text-sm font-bold text-white">Execution Scratchpad &amp; Journal</h4>
                        <p className="text-[10px] text-slate-400">Notes for {ticker} (persisted locally)</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {notesSaved && (
                        <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                          <CheckCircle2 className="w-3 h-3" /> Saved
                        </span>
                      )}
                      <button
                        onClick={addTimestampToNotes}
                        className="px-2.5 py-1 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 flex items-center gap-1 transition"
                      >
                        <Clock className="w-3 h-3 text-cyan-400" />
                        <span>+ Time</span>
                      </button>
                      <button
                        onClick={() => {
                          navigator.clipboard?.writeText(notes || '');
                          setNotesCopied(true);
                          setTimeout(() => setNotesCopied(false), 2000);
                        }}
                        className="px-2.5 py-1 text-xs font-semibold rounded-lg border bg-slate-800 hover:bg-slate-700 text-slate-200 border-slate-700 transition"
                      >
                        {notesCopied ? 'Copied ✓' : 'Copy'}
                      </button>
                      <button
                        onClick={() => {
                          setNotes('');
                          try { localStorage.removeItem('stockiq_intraday_notes_' + ticker); } catch (_) {}
                        }}
                        className="p-1.5 text-slate-500 hover:text-rose-400 transition"
                        title="Clear notes"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Discipline Chips */}
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                    {[
                      { tag: '📌 [VWAP Retest]', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20' },
                      { tag: '🛑 [Stop Violation Risk]', color: 'text-rose-400 bg-rose-500/10 border-rose-500/20' },
                      { tag: '🎯 [Target Achieved]', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20' },
                      { tag: '⚡ [MIS Square-Off]', color: 'text-purple-400 bg-purple-500/10 border-purple-500/20' },
                    ].map(chip => (
                      <button
                        key={chip.tag}
                        onClick={() => addTemplateTag(chip.tag)}
                        className={`px-2 py-0.5 rounded-lg border text-[10px] font-mono transition hover:scale-105 ${chip.color}`}
                      >
                        {chip.tag}
                      </button>
                    ))}
                  </div>

                  <textarea
                    value={notes}
                    onChange={handleNotesChange}
                    placeholder={`Trade thesis, entry logic & mental stop notes for ${ticker}...`}
                    className="w-full h-24 bg-[#070a10] border border-white/[0.08] rounded-xl p-3 text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-amber-500/60 shadow-inner resize-y leading-relaxed"
                  />
                </div>
              )}
            </div>

            {/* ── RIGHT DESK: TABBED MICROSTRUCTURE DECK (4 Spans) ── */}
            <div className="xl:col-span-4 space-y-4">
              <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-3 sm:p-4 backdrop-blur-md shadow-xl flex flex-col justify-between">
                
                {/* Microstructure Tabs Navigation */}
                <div className="flex items-center gap-1 overflow-x-auto pb-2 border-b border-white/[0.06] scrollbar-none text-xs">
                  {[
                    { id: 'pivots', label: 'Pivots & CPR' },
                    { id: 'signals', label: 'Quant Checklist' },
                    { id: 'flow', label: 'Order Flow & DOM' },
                    { id: 'options', label: 'Options & OI' },
                    { id: 'calc', label: 'Sizing & Risk' },
                    { id: 'confluence', label: 'MTF Matrix' },
                    { id: 'log', label: `Trades (${tradeLog.length})` },
                  ].map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setSidebarTab(tab.id)}
                      className={`px-2.5 py-1.5 rounded-xl font-semibold text-xs whitespace-nowrap transition ${
                        sidebarTab === tab.id
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold shadow-sm'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                      }`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Tab 1: Pivots & CPR */}
                {sidebarTab === 'pivots' && (
                  <div className="pt-3 space-y-3">
                    {data?.pivots?.cpr && (
                      <div className="p-3 rounded-xl bg-[#070a10] border border-indigo-500/30 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-indigo-300 flex items-center gap-1.5">
                            <Layers className="w-3.5 h-3.5 text-indigo-400" />
                            Central Pivot Range (CPR)
                          </span>
                          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full font-mono uppercase border ${
                            data.pivots.cpr.classification === 'NARROW'
                              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                              : data.pivots.cpr.classification === 'WIDE'
                              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                              : 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                          }`}>
                            {data.pivots.cpr.classification} ({data.pivots.cpr.width_pct}%)
                          </span>
                        </div>

                        <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs tabular-nums">
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-[9px] text-slate-400 block uppercase font-bold">TC</span>
                            <span className="font-bold text-indigo-300 mt-0.5 block">{currSym}{data.pivots.cpr.tc}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-[9px] text-slate-400 block uppercase font-bold">Pivot</span>
                            <span className="font-bold text-white mt-0.5 block">{currSym}{data.pivots.cpr.pivot}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-[9px] text-slate-400 block uppercase font-bold">BC</span>
                            <span className="font-bold text-purple-300 mt-0.5 block">{currSym}{data.pivots.cpr.bc}</span>
                          </div>
                        </div>

                        <p className="text-[11px] text-slate-300 leading-snug bg-slate-950/60 p-2 rounded-lg border border-slate-800/60">
                          💡 {data.pivots.cpr.description}
                        </p>
                      </div>
                    )}

                    {data?.pivots?.camarilla && (
                      <div className="space-y-1.5 text-xs font-mono tabular-nums">
                        {/* H4 Breakout */}
                        <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/25">
                          <span className="font-bold text-emerald-400">H4 Breakout Target</span>
                          <span className="font-bold text-white">{currSym}{data.pivots.camarilla.h4}</span>
                        </div>
                        {/* H3 Resistance */}
                        <div className="flex items-center justify-between p-2 rounded-xl bg-rose-500/10 border border-rose-500/25">
                          <span className="font-bold text-rose-400">H3 Short Resistance</span>
                          <span className="font-bold text-white">{currSym}{data.pivots.camarilla.h3}</span>
                        </div>
                        {/* Floor Pivot */}
                        <div className="flex items-center justify-between p-2 rounded-xl bg-[#070a10] border border-white/[0.08]">
                          <span className="font-bold text-slate-300">Central Floor Pivot (P)</span>
                          <span className="font-bold text-cyan-300">{currSym}{data.pivots.floor.p}</span>
                        </div>
                        {/* L3 Support */}
                        <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/25">
                          <span className="font-bold text-emerald-400">L3 Long Support</span>
                          <span className="font-bold text-white">{currSym}{data.pivots.camarilla.l3}</span>
                        </div>
                        {/* L4 Breakdown */}
                        <div className="flex items-center justify-between p-2 rounded-xl bg-rose-500/10 border border-rose-500/25">
                          <span className="font-bold text-rose-400">L4 Breakdown Target</span>
                          <span className="font-bold text-white">{currSym}{data.pivots.camarilla.l4}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Tab: Quant Signals Audit */}
                {sidebarTab === 'signals' && (
                  <div className="pt-3 space-y-3">
                    {/* Top Quant Bias Overview Card */}
                    <div className="p-3 rounded-xl bg-[#070a10] border border-white/[0.08] space-y-2.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-slate-400 uppercase font-mono font-bold tracking-wider">
                          Composite Quant Bias
                        </span>
                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${
                          data.signals?.overall_bias?.includes('BUY')
                            ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                            : data.signals?.overall_bias?.includes('SELL')
                            ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                            : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                        }`}>
                          {data.signals?.overall_bias || 'NEUTRAL'}
                        </span>
                      </div>

                      <div className="flex items-baseline justify-between">
                        <div className="flex items-baseline gap-1">
                          <span className={`text-2xl font-black font-mono tabular-nums ${
                            (data.signals?.quant_score || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                          }`}>
                            {(data.signals?.quant_score || 0) >= 0 ? '+' : ''}{data.signals?.quant_score || 0}
                          </span>
                          <span className="text-xs text-slate-500 font-mono">/ 100</span>
                        </div>
                        <div className="text-right text-[11px] font-mono">
                          <span className="text-slate-400">Regime: </span>
                          <span className="text-cyan-300 font-bold">{data.signals?.risk_regime || 'Equilibrium'}</span>
                        </div>
                      </div>

                      {/* Visual Score Bar */}
                      <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden flex border border-slate-800">
                        {(() => {
                          const score = data.signals?.quant_score || 0;
                          const norm = Math.max(0, Math.min(100, (score + 100) / 2));
                          return (
                            <div
                              className={`h-full transition-all duration-500 ${
                                score >= 20 ? 'bg-emerald-400' : score <= -20 ? 'bg-rose-500' : 'bg-amber-400'
                              }`}
                              style={{ width: `${norm}%` }}
                            />
                          );
                        })()}
                      </div>

                      {/* Extension state note */}
                      {data.signals?.extension_desc && (
                        <p className="text-[11px] text-slate-300 leading-snug bg-slate-950/60 p-2 rounded-lg border border-slate-800/60 font-mono">
                          💡 {data.signals.extension_desc}
                        </p>
                      )}
                    </div>

                    {/* Factor-by-Factor Validations Checklist */}
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between px-1 text-[10px] text-slate-400 uppercase font-mono font-bold">
                        <span>Institutional Checklist</span>
                        <span>{data.signals?.checklist?.filter(c => c.status === 'BULLISH').length || 0} Bull · {data.signals?.checklist?.filter(c => c.status === 'BEARISH').length || 0} Bear</span>
                      </div>

                      {data.signals?.checklist && data.signals.checklist.length > 0 ? (
                        data.signals.checklist.map((item, idx) => (
                          <div
                            key={idx}
                            className={`p-2.5 rounded-xl border transition-all text-xs font-mono ${
                              item.status === 'BULLISH'
                                ? 'bg-emerald-950/20 border-emerald-500/25'
                                : item.status === 'BEARISH'
                                ? 'bg-rose-950/20 border-rose-500/25'
                                : 'bg-[#070a10] border-white/[0.08]'
                            }`}
                          >
                            <div className="flex items-center justify-between mb-1">
                              <div className="flex items-center gap-1.5 font-bold">
                                {item.status === 'BULLISH' ? (
                                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                                ) : item.status === 'BEARISH' ? (
                                  <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                                ) : (
                                  <MinusCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                                )}
                                <span className={
                                  item.status === 'BULLISH' ? 'text-emerald-300' :
                                  item.status === 'BEARISH' ? 'text-rose-300' : 'text-slate-300'
                                }>
                                  {item.factor}
                                </span>
                              </div>
                              <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded uppercase border ${
                                item.status === 'BULLISH'
                                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                                  : item.status === 'BEARISH'
                                  ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                                  : 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                              }`}>
                                {item.status}
                              </span>
                            </div>
                            <p className="text-[11px] text-slate-400 leading-snug pl-5 font-sans">
                              {item.desc}
                            </p>
                          </div>
                        ))
                      ) : (
                        <div className="p-4 text-center text-xs text-slate-500 bg-[#070a10] rounded-xl border border-white/[0.06]">
                          Signals checklist evaluating session rules...
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Tab 2: Order Flow & VPVR */}
                {sidebarTab === 'flow' && (
                  <div className="pt-3 space-y-3">
                    {/* Pressure Meter */}
                    <div className="p-3 bg-[#070a10] rounded-xl border border-white/[0.08] space-y-2.5">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-bold">
                          Buyers: {data?.order_flow?.buy_pressure_pct ?? data?.volume_delta_proxy?.buy_pressure_pct}%
                        </span>
                        <span className="px-2 py-0.5 rounded bg-rose-500/15 text-rose-400 font-bold">
                          Sellers: {data?.order_flow?.sell_pressure_pct ?? data?.volume_delta_proxy?.sell_pressure_pct}%
                        </span>
                      </div>

                      <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden flex border border-slate-800">
                        <div
                          className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-full transition-all duration-500"
                          style={{ width: `${data?.order_flow?.buy_pressure_pct ?? data?.volume_delta_proxy?.buy_pressure_pct}%` }}
                        />
                        <div
                          className="bg-gradient-to-r from-rose-500 to-rose-600 h-full transition-all duration-500"
                          style={{ width: `${data?.order_flow?.sell_pressure_pct ?? data?.volume_delta_proxy?.sell_pressure_pct}%` }}
                        />
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono tabular-nums">
                        <span>Net Delta: <strong className={(data?.order_flow?.net_delta ?? data?.volume_delta_proxy?.net_delta) >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                          {(data?.order_flow?.net_delta ?? data?.volume_delta_proxy?.net_delta) >= 0 ? '+' : ''}{(data?.order_flow?.net_delta ?? data?.volume_delta_proxy?.net_delta)?.toLocaleString()}
                        </strong></span>
                        <span>Vol: <strong className="text-white">{data?.volume?.toLocaleString()}</strong></span>
                      </div>
                    </div>

                    {/* Level 2 Market Depth (DOM) */}
                    <div className="p-3 bg-[#070a10] rounded-xl border border-white/[0.08] space-y-2">
                      <div className="flex items-center justify-between text-[10px] font-mono">
                        <span className="text-slate-400 uppercase font-bold tracking-wider">
                          Level 2 Market Depth (DOM)
                        </span>
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/25">
                          MODELED L2 BOOK
                        </span>
                      </div>

                      {(() => {
                        const price = data.current_price || 100;
                        const tick = Math.max(0.05, Math.round(price * 0.00025 * 20) / 20);
                        const buyPct = (data?.order_flow?.buy_pressure_pct ?? 50) / 100;
                        const sellPct = (data?.order_flow?.sell_pressure_pct ?? 50) / 100;
                        const baseVol = Math.max(100, Math.round((data?.volume || 50000) / 180));

                        const bids = [
                          { orders: Math.max(1, Math.round(buyPct * 16)), qty: Math.round(baseVol * buyPct * 1.5), price: price - tick },
                          { orders: Math.max(1, Math.round(buyPct * 28)), qty: Math.round(baseVol * buyPct * 2.4), price: price - tick * 2 },
                          { orders: Math.max(1, Math.round(buyPct * 42)), qty: Math.round(baseVol * buyPct * 3.1), price: price - tick * 3 },
                          { orders: Math.max(1, Math.round(buyPct * 22)), qty: Math.round(baseVol * buyPct * 1.9), price: price - tick * 4 },
                          { orders: Math.max(1, Math.round(buyPct * 10)), qty: Math.round(baseVol * buyPct * 1.1), price: price - tick * 5 },
                        ];
                        const asks = [
                          { price: price + tick, qty: Math.round(baseVol * sellPct * 1.4), orders: Math.max(1, Math.round(sellPct * 14)) },
                          { price: price + tick * 2, qty: Math.round(baseVol * sellPct * 2.2), orders: Math.max(1, Math.round(sellPct * 25)) },
                          { price: price + tick * 3, qty: Math.round(baseVol * sellPct * 2.9), orders: Math.max(1, Math.round(sellPct * 38)) },
                          { price: price + tick * 4, qty: Math.round(baseVol * sellPct * 1.8), orders: Math.max(1, Math.round(sellPct * 20)) },
                          { price: price + tick * 5, qty: Math.round(baseVol * sellPct * 1.0), orders: Math.max(1, Math.round(sellPct * 9)) },
                        ];

                        const maxQty = Math.max(...bids.map(b => b.qty), ...asks.map(a => a.qty), 1);
                        const totalBidQty = bids.reduce((s, b) => s + b.qty, 0);
                        const totalAskQty = asks.reduce((s, a) => s + a.qty, 0);
                        const totalBookQty = totalBidQty + totalAskQty || 1;
                        const spread = (asks[0].price - bids[0].price).toFixed(2);
                        const spreadPct = ((spread / price) * 100).toFixed(3);

                        return (
                          <div className="space-y-1 font-mono text-[10px] tabular-nums">
                            {/* Column headers */}
                            <div className="grid grid-cols-6 text-[8.5px] text-slate-500 uppercase pb-1 border-b border-white/[0.06] font-bold">
                              <span className="text-left">Orders</span>
                              <span className="text-right">Qty</span>
                              <span className="text-right text-emerald-400">Bid</span>
                              <span className="text-left text-rose-400 pl-2">Ask</span>
                              <span className="text-right">Qty</span>
                              <span className="text-right">Orders</span>
                            </div>

                            {/* 5 depth rows */}
                            {bids.map((b, idx) => {
                              const a = asks[idx];
                              const bDepth = Math.min(100, Math.round((b.qty / maxQty) * 100));
                              const aDepth = Math.min(100, Math.round((a.qty / maxQty) * 100));
                              return (
                                <div key={idx} className="grid grid-cols-6 items-center py-0.5 relative text-[9.5px]">
                                  {/* Bid depth background bar */}
                                  <div
                                    className="absolute left-0 top-0 bottom-0 bg-emerald-500/10 pointer-events-none rounded-l"
                                    style={{ width: `${bDepth / 2}%` }}
                                  />
                                  {/* Ask depth background bar */}
                                  <div
                                    className="absolute right-0 top-0 bottom-0 bg-rose-500/10 pointer-events-none rounded-r"
                                    style={{ width: `${aDepth / 2}%` }}
                                  />

                                  {/* Bid info */}
                                  <span className="text-slate-500 text-left relative z-10">{b.orders}</span>
                                  <span className="text-slate-300 text-right relative z-10">{b.qty.toLocaleString()}</span>
                                  <span className="text-emerald-400 font-bold text-right relative z-10">{b.price.toFixed(2)}</span>

                                  {/* Ask info */}
                                  <span className="text-rose-400 font-bold text-left pl-2 relative z-10">{a.price.toFixed(2)}</span>
                                  <span className="text-slate-300 text-right relative z-10">{a.qty.toLocaleString()}</span>
                                  <span className="text-slate-500 text-right relative z-10">{a.orders}</span>
                                </div>
                              );
                            })}

                            {/* Footer totals */}
                            <div className="pt-1.5 border-t border-white/[0.06] flex items-center justify-between text-[10px]">
                              <div className="flex items-center gap-1">
                                <span className="text-emerald-400 font-bold">{totalBidQty.toLocaleString()}</span>
                                <span className="text-slate-500 text-[9px]">({Math.round((totalBidQty / totalBookQty) * 100)}%)</span>
                              </div>
                              <div className="text-slate-400 text-[9px]">
                                Spread: <strong className="text-amber-400">{currSym}{spread}</strong> ({spreadPct}%)
                              </div>
                              <div className="flex items-center gap-1">
                                <span className="text-rose-400 font-bold">{totalAskQty.toLocaleString()}</span>
                                <span className="text-slate-500 text-[9px]">({Math.round((totalAskQty / totalBookQty) * 100)}%)</span>
                              </div>
                            </div>
                          </div>
                        );
                      })()}
                    </div>

                    {/* VPVR Distribution */}
                    {data?.volume_profile && (
                      <div className="space-y-2">
                        <div className="grid grid-cols-3 gap-2 text-xs font-mono tabular-nums text-center">
                          <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/25">
                            <span className="text-[9px] text-amber-400 block font-bold uppercase">POC Price</span>
                            <span className="font-bold text-white mt-0.5 block">{currSym}{data.volume_profile.poc_price}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/25">
                            <span className="text-[9px] text-cyan-400 block font-bold uppercase">VAL (70%)</span>
                            <span className="font-bold text-slate-200 mt-0.5 block">{currSym}{data.volume_profile.val_price}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/25">
                            <span className="text-[9px] text-purple-400 block font-bold uppercase">VAH (70%)</span>
                            <span className="font-bold text-slate-200 mt-0.5 block">{currSym}{data.volume_profile.vah_price}</span>
                          </div>
                        </div>

                        <div className="space-y-1 max-h-48 overflow-y-auto pr-1 scrollbar-thin">
                          {data.volume_profile.profile?.slice(0, 10).map((b, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-[10px] font-mono py-0.5 px-2 rounded-lg bg-[#070a10]">
                              <span className="w-12 font-bold tabular-nums">{b.price.toFixed(2)}</span>
                              <div className="flex-1 bg-slate-900 h-1.5 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${b.is_poc ? 'bg-amber-400' : b.in_value_area ? 'bg-cyan-500' : 'bg-slate-700'}`}
                                  style={{ width: `${Math.min(b.pct_of_total * 4, 100)}%` }}
                                />
                              </div>
                              {b.is_poc && <span className="text-[8px] bg-amber-400 text-black px-1 rounded font-bold">POC</span>}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Tab 3: Options & PCR */}
                {sidebarTab === 'options' && (
                  <div className="pt-3 space-y-3">
                    {pcrData && pcrData.available !== false ? (
                      <div className="space-y-2.5">
                        <div className="p-3 rounded-xl bg-[#070a10] border border-white/[0.08] flex items-center justify-between">
                          <div>
                            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block">PCR OI Ratio</span>
                            <div className="flex items-baseline gap-1 mt-0.5">
                              <span className={`text-2xl font-black font-mono tabular-nums ${
                                pcrData.color === 'bearish' ? 'text-rose-400' :
                                pcrData.color === 'bullish' ? 'text-emerald-400' : 'text-amber-400'
                              }`}>
                                {pcrData.pcr_oi}
                              </span>
                              <span className="text-[10px] text-slate-500 font-mono">OI</span>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full border uppercase ${
                              pcrData.color === 'bearish' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' :
                              pcrData.color === 'bullish' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' :
                              'bg-amber-500/20 text-amber-300 border-amber-500/40'
                            }`}>
                              {pcrData.sentiment?.replace('_', ' ')}
                            </span>
                            <p className="text-[10px] text-slate-400 mt-1 font-mono">{pcrData.expiry_date}</p>
                          </div>
                        </div>

                        {/* Open Interest Bar */}
                        <div className="space-y-1 p-2.5 rounded-xl bg-[#070a10] border border-white/[0.08]">
                          <div className="flex justify-between text-[10px] font-mono tabular-nums">
                            <span className="text-emerald-400 font-bold">Calls: {(pcrData.call_oi / 1000).toFixed(0)}K</span>
                            <span className="text-rose-400 font-bold">Puts: {(pcrData.put_oi / 1000).toFixed(0)}K</span>
                          </div>
                          <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden flex">
                            {(() => {
                              const total = (pcrData.call_oi || 0) + (pcrData.put_oi || 0) || 1;
                              const callPct = Math.round((pcrData.call_oi / total) * 100);
                              return (
                                <>
                                  <div className="bg-emerald-500 h-full" style={{ width: `${callPct}%` }} />
                                  <div className="bg-rose-500 h-full" style={{ width: `${100 - callPct}%` }} />
                                </>
                              );
                            })()}
                          </div>
                        </div>

                        {pcrData.max_pain_strike && (
                          <div className="p-2.5 rounded-xl bg-[#070a10] border border-white/[0.08] space-y-1">
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-slate-400 font-medium">Max Pain Strike:</span>
                              <div className="flex items-center gap-1.5">
                                <span className="font-bold font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/25">
                                  {currSym}{pcrData.max_pain_strike}
                                </span>
                                {data?.current_price && (
                                  <span className={`text-[10px] font-mono font-bold ${
                                    pcrData.max_pain_strike >= data.current_price ? 'text-emerald-400' : 'text-rose-400'
                                  }`}>
                                    {pcrData.max_pain_strike >= data.current_price ? '+' : ''}
                                    {((pcrData.max_pain_strike - data.current_price) / data.current_price * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            </div>
                            <p className="text-[10px] text-slate-500 font-mono">
                              Dealers experience minimum option payout at {currSym}{pcrData.max_pain_strike}. Expiry gravity anchor.
                            </p>
                          </div>
                        )}

                        {/* Top Call Strikes (Resistance Walls) */}
                        {pcrData.top_call_strikes && pcrData.top_call_strikes.length > 0 && (
                          <div className="p-2.5 rounded-xl bg-[#070a10] border border-white/[0.08] space-y-1.5">
                            <div className="flex items-center justify-between text-[10px] font-mono">
                              <span className="text-cyan-400 uppercase font-bold tracking-wider">Top Call Walls (Resistance)</span>
                              <span className="text-slate-500">Call OI</span>
                            </div>
                            {(() => {
                              const maxCall = Math.max(...pcrData.top_call_strikes.map(s => s.oi), 1);
                              return pcrData.top_call_strikes.slice(0, 4).map((item, idx) => (
                                <div key={idx} className="space-y-0.5">
                                  <div className="flex items-center justify-between text-[10px] font-mono tabular-nums">
                                    <div className="flex items-center gap-1.5">
                                      <span className="font-bold text-white">{currSym}{item.strike}</span>
                                      {idx === 0 && <span className="text-[8px] bg-cyan-500/20 text-cyan-300 px-1 rounded font-bold">L1 WALL</span>}
                                    </div>
                                    <span className="text-cyan-400 font-bold">{(item.oi / 1000).toFixed(1)}K</span>
                                  </div>
                                  <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-gradient-to-r from-cyan-600 to-cyan-400 h-full rounded-full transition-all duration-300" style={{ width: `${(item.oi / maxCall) * 100}%` }} />
                                  </div>
                                </div>
                              ));
                            })()}
                          </div>
                        )}

                        {/* Top Put Strikes (Support Walls) */}
                        {pcrData.top_put_strikes && pcrData.top_put_strikes.length > 0 && (
                          <div className="p-2.5 rounded-xl bg-[#070a10] border border-white/[0.08] space-y-1.5">
                            <div className="flex items-center justify-between text-[10px] font-mono">
                              <span className="text-rose-400 uppercase font-bold tracking-wider">Top Put Walls (Support)</span>
                              <span className="text-slate-500">Put OI</span>
                            </div>
                            {(() => {
                              const maxPut = Math.max(...pcrData.top_put_strikes.map(s => s.oi), 1);
                              return pcrData.top_put_strikes.slice(0, 4).map((item, idx) => (
                                <div key={idx} className="space-y-0.5">
                                  <div className="flex items-center justify-between text-[10px] font-mono tabular-nums">
                                    <div className="flex items-center gap-1.5">
                                      <span className="font-bold text-white">{currSym}{item.strike}</span>
                                      {idx === 0 && <span className="text-[8px] bg-rose-500/20 text-rose-300 px-1 rounded font-bold">L1 FLOOR</span>}
                                    </div>
                                    <span className="text-rose-400 font-bold">{(item.oi / 1000).toFixed(1)}K</span>
                                  </div>
                                  <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                                    <div className="bg-gradient-to-r from-rose-600 to-rose-400 h-full rounded-full transition-all duration-300" style={{ width: `${(item.oi / maxPut) * 100}%` }} />
                                  </div>
                                </div>
                              ));
                            })()}
                          </div>
                        )}

                        <p className="text-[11px] text-slate-400 leading-snug">💡 {pcrData.sentiment_label}</p>
                      </div>
                    ) : (
                      <div className="p-6 text-center text-xs text-slate-400 bg-[#070a10] rounded-xl border border-white/[0.06]">
                        {pcrLoading ? 'Loading live options chain...' : 'No options chain available for this ticker.'}
                      </div>
                    )}
                  </div>
                )}

                {/* Tab 4: Position Sizing & Real Friction */}
                {sidebarTab === 'calc' && (
                  <div className="pt-3 space-y-3">
                    {/* Capital Quick Presets */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                        <span>Account Capital:</span>
                        <span className="text-white font-bold">{currSym}{Number(calcCapital || 0).toLocaleString()}</span>
                      </div>
                      <div className="grid grid-cols-4 sm:grid-cols-5 gap-1">
                        {(currSym === '₹' ? [
                          { label: '₹50K', val: 50000 },
                          { label: '₹1L', val: 100000 },
                          { label: '₹2.5L', val: 250000 },
                          { label: '₹5L', val: 500000 },
                          { label: '₹10L', val: 1000000 },
                        ] : [
                          { label: '$10K', val: 10000 },
                          { label: '$25K', val: 25000 },
                          { label: '$50K', val: 50000 },
                          { label: '$100K', val: 100000 },
                        ]).map(preset => (
                          <button
                            key={preset.label}
                            type="button"
                            onClick={() => setCalcCapital(preset.val.toString())}
                            className={`py-1 px-1 text-[10px] font-mono font-bold rounded-lg border transition ${
                              Number(calcCapital) === preset.val
                                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                                : 'bg-[#070a10] text-slate-400 border-white/[0.06] hover:text-white hover:border-slate-700'
                            }`}
                          >
                            {preset.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Risk % Quick Tiers */}
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                        <span>Max Risk Per Trade:</span>
                        <span className="text-amber-400 font-bold">{calcRiskPct}% ({currSym}{Math.round((Number(calcCapital) || 0) * (calcRiskPct / 100)).toLocaleString()})</span>
                      </div>
                      <div className="grid grid-cols-4 gap-1">
                        {[
                          { label: '0.5% Cons.', val: 0.5 },
                          { label: '1.0% Std.', val: 1.0 },
                          { label: '1.5% Active', val: 1.5 },
                          { label: '2.0% Agg.', val: 2.0 },
                        ].map(tier => (
                          <button
                            key={tier.val}
                            type="button"
                            onClick={() => setCalcRiskPct(tier.val)}
                            className={`py-1 px-1 text-[10px] font-mono font-bold rounded-lg border transition ${
                              calcRiskPct === tier.val
                                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                                : 'bg-[#070a10] text-slate-400 border-white/[0.06] hover:text-white hover:border-slate-700'
                            }`}
                          >
                            {tier.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <label className="text-slate-400 block mb-1 font-medium text-[11px]">Capital ({currSym})</label>
                        <input
                          type="number"
                          value={calcCapital}
                          onChange={(e) => setCalcCapital(e.target.value)}
                          className="w-full bg-[#070a10] border border-white/[0.1] rounded-lg px-2.5 py-1.5 font-mono text-xs text-white"
                        />
                      </div>
                      <div>
                        <label className="text-slate-400 block mb-1 font-medium text-[11px]">Risk %</label>
                        <select
                          value={calcRiskPct}
                          onChange={(e) => setCalcRiskPct(Number(e.target.value))}
                          className="w-full bg-[#070a10] border border-white/[0.1] rounded-lg px-2.5 py-1.5 font-mono text-xs text-white"
                        >
                          <option value={0.5}>0.5%</option>
                          <option value={1.0}>1.0%</option>
                          <option value={1.5}>1.5%</option>
                          <option value={2.0}>2.0%</option>
                        </select>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-slate-400 font-medium text-[11px]">Entry ({currSym})</label>
                          {data?.current_price && (
                            <button
                              type="button"
                              onClick={() => setCalcEntry(data.current_price.toString())}
                              className="text-[9px] text-cyan-400 hover:text-cyan-300 font-mono font-bold"
                            >
                              Fill LTP
                            </button>
                          )}
                        </div>
                        <input
                          type="number" step="0.05"
                          value={calcEntry}
                          onChange={(e) => setCalcEntry(e.target.value)}
                          className="w-full bg-[#070a10] border border-white/[0.1] rounded-lg px-2.5 py-1.5 font-mono text-xs text-white"
                        />
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-slate-400 font-medium text-[11px]">Stop ({currSym})</label>
                          {data?.supertrend ? (
                            <button
                              type="button"
                              onClick={() => setCalcStop(data.supertrend.toFixed(2))}
                              className="text-[9px] text-rose-400 hover:text-rose-300 font-mono font-bold"
                            >
                              Fill ST
                            </button>
                          ) : data?.low ? (
                            <button
                              type="button"
                              onClick={() => setCalcStop(data.low.toString())}
                              className="text-[9px] text-rose-400 hover:text-rose-300 font-mono font-bold"
                            >
                              Fill Low
                            </button>
                          ) : null}
                        </div>
                        <input
                          type="number" step="0.05"
                          value={calcStop}
                          onChange={(e) => setCalcStop(e.target.value)}
                          className="w-full bg-[#070a10] border border-white/[0.1] rounded-lg px-2.5 py-1.5 font-mono text-xs text-white"
                        />
                      </div>
                    </div>

                    {sizingResults && (
                      <div className="p-3 bg-[#070a10] rounded-xl border border-white/[0.08] space-y-2.5 text-xs font-mono tabular-nums">
                        <div className="grid grid-cols-3 gap-2 text-center">
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-[9px] text-slate-400 block font-sans uppercase font-bold">SHARES</span>
                            <span className="text-base font-bold text-cyan-300 block">{sizingResults.exactShares}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-[9px] text-slate-400 block font-sans uppercase font-bold">MARGIN</span>
                            <span className="text-base font-bold text-white block">{currSym}{sizingResults.marginRequired?.toLocaleString()}</span>
                          </div>
                          <div className="p-2 rounded-lg bg-slate-950 border border-slate-800">
                            <span className="text-[9px] text-slate-400 block font-sans uppercase font-bold">FEES</span>
                            <span className="text-base font-bold text-amber-400 block">{currSym}{sizingResults.totalCharges}</span>
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-[11px] p-2 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300">
                          <span>Breakeven Move:</span>
                          <span className="font-bold">+{currSym}{sizingResults.breakevenMovePts} ({sizingResults.breakevenMovePct}%)</span>
                        </div>

                        <button
                          onClick={() => {
                            setNewTrade({
                              ticker: ticker.split('.')[0],
                              direction: sizingResults.isLong ? 'LONG' : 'SHORT',
                              entry: calcEntry,
                              exit: '',
                              qty: sizingResults.exactShares.toString(),
                              note: 'From position sizing calculator'
                            });
                            setSidebarTab('log');
                            setTradeLogOpen(true);
                          }}
                          className="w-full py-2 bg-gradient-to-r from-cyan-500/20 to-emerald-500/20 hover:from-cyan-500/30 border border-cyan-500/40 text-cyan-300 font-bold rounded-xl text-xs transition"
                        >
                          + Push to Trade Log
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Tab 5: Multi-TF Confluence */}
                {sidebarTab === 'confluence' && (
                  <div className="pt-3 space-y-3">
                    <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                      {data?.multi_timeframe?.screens?.map((s, idx) => (
                        <div key={idx} className="p-2.5 bg-[#070a10] border border-white/[0.08] rounded-xl">
                          <span className="text-[10px] text-slate-400 block font-sans uppercase font-bold mb-1">
                            {s.timeframe}
                          </span>
                          <span className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full my-0.5 ${
                            s.trend === 'BULLISH' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
                          }`}>
                            {s.trend}
                          </span>
                          <span className="text-[10px] text-slate-400 block mt-0.5">RSI: {s.rsi}</span>
                        </div>
                      ))}
                    </div>

                    <div className="p-2.5 rounded-xl bg-[#070a10] border border-white/[0.08] flex items-center justify-between text-xs">
                      <span className="text-slate-400 font-medium">Consensus Bias:</span>
                      <span className="font-bold font-mono text-cyan-300 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/30">
                        {data?.multi_timeframe?.confluence_bias} ({data?.multi_timeframe?.confluence_score}%)
                      </span>
                    </div>
                  </div>
                )}

                {/* Tab 6: Trade Log */}
                {sidebarTab === 'log' && (
                  <div className="pt-3 space-y-3">
                    <div className="flex items-center justify-between">
                      {(() => {
                        const closed = tradeLog.filter(t => t.status !== 'OPEN');
                        const totalPnl = closed.reduce((s, t) => s + (t.grossPnl || 0), 0);
                        const wins = closed.filter(t => t.status === 'WIN').length;
                        const winRate = closed.length ? Math.round((wins / closed.length) * 100) : 0;
                        return (
                          <div className="flex items-center gap-2 text-xs font-mono tabular-nums">
                            <span className="text-slate-400">{closed.length} cl · <strong className="text-white">{winRate}%</strong> W/R</span>
                            <span className={`px-2 py-0.5 rounded border font-bold ${totalPnl >= 0 ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/15 text-rose-400 border-rose-500/30'}`}>
                              {totalPnl >= 0 ? '+' : ''}{currSym}{totalPnl.toLocaleString()}
                            </span>
                          </div>
                        );
                      })()}
                      <div className="flex items-center gap-1.5">
                        {tradeLog.length > 0 && (
                          <button onClick={exportTradeLogCSV} className="p-1 rounded-lg bg-slate-800 text-slate-300 hover:text-white" title="Export CSV">
                            <Download className="w-3.5 h-3.5" />
                          </button>
                        )}
                        <button
                          onClick={() => setTradeLogOpen(!tradeLogOpen)}
                          className="px-2.5 py-1 rounded-lg bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 text-xs font-semibold"
                        >
                          {tradeLogOpen ? 'Close' : '+ Add'}
                        </button>
                      </div>
                    </div>

                    {/* Trade Entry Form */}
                    {tradeLogOpen && (
                      <div className="p-3 bg-[#070a10] rounded-xl border border-white/[0.08] space-y-2">
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <input
                            value={newTrade.ticker}
                            onChange={e => setNewTrade(p => ({...p, ticker: e.target.value}))}
                            placeholder="Ticker (e.g. SBIN)"
                            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                          />
                          <select
                            value={newTrade.direction}
                            onChange={e => setNewTrade(p => ({...p, direction: e.target.value}))}
                            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                          >
                            <option value="LONG">LONG</option>
                            <option value="SHORT">SHORT</option>
                          </select>
                          <input
                            type="number" step="0.05"
                            value={newTrade.entry}
                            onChange={e => setNewTrade(p => ({...p, entry: e.target.value}))}
                            placeholder="Entry price"
                            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                          />
                          <input
                            type="number" step="0.05"
                            value={newTrade.exit}
                            onChange={e => setNewTrade(p => ({...p, exit: e.target.value}))}
                            placeholder="Exit price"
                            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                          />
                          <input
                            type="number"
                            value={newTrade.qty}
                            onChange={e => setNewTrade(p => ({...p, qty: e.target.value}))}
                            placeholder="Quantity"
                            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-white"
                          />
                          <button
                            onClick={() => setNewTrade(p => ({...p, ticker: ticker.split('.')[0], entry: data?.current_price?.toString() || ''}))}
                            className="bg-slate-800 hover:bg-slate-700 text-cyan-300 rounded-lg px-2 py-1 text-xs font-semibold"
                          >
                            Sync Live
                          </button>
                        </div>
                        <button
                          onClick={addTradeEntry}
                          className="w-full py-1.5 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 font-bold rounded-lg text-xs border border-cyan-500/40"
                        >
                          Save Position
                        </button>
                      </div>
                    )}

                    {/* Trades List */}
                    <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1 scrollbar-thin">
                      {tradeLog.slice(0, 15).map(t => (
                        <div key={t.id} className="flex items-center justify-between p-2 rounded-xl bg-[#070a10] border border-white/[0.06] text-xs">
                          <div className="flex items-center gap-2">
                            <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border ${t.direction === 'LONG' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' : 'bg-rose-500/20 text-rose-300 border-rose-500/40'}`}>
                              {t.direction}
                            </span>
                            <span className="font-bold text-white">{t.ticker}</span>
                            <span className="text-[10px] text-slate-500 font-mono">{t.time}</span>
                          </div>
                          <div className="flex items-center gap-2 font-mono tabular-nums">
                            <span className={`font-bold ${t.status === 'WIN' ? 'text-emerald-400' : t.status === 'LOSS' ? 'text-rose-400' : 'text-slate-400'}`}>
                              {t.grossPnl !== null ? `${t.grossPnl >= 0 ? '+' : ''}${currSym}${t.grossPnl}` : 'OPEN'}
                            </span>
                            <button onClick={() => removeTrade(t.id)} className="text-slate-600 hover:text-rose-400">
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>
                      ))}
                      {tradeLog.length === 0 && (
                        <p className="text-center text-xs text-slate-500 py-6">No trades logged in current session.</p>
                      )}
                    </div>
                  </div>
                )}

              </div>
            </div>

          </div>

          {/* ── LOWER INTELLIGENCE GRID: RADAR SCANNER & DEALS SCANNER ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {/* Intraday Radar Scanner */}
            <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-4 sm:p-5 backdrop-blur-md shadow-xl flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-400 shadow-sm">
                      <Zap className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white tracking-wide">
                        Intraday Radar Scanner ({scannerMarket})
                      </h3>
                      <p className="text-[11px] text-slate-400">Real-time breakouts &amp; momentum deviations</p>
                    </div>
                  </div>
                  <InfoBadge infoKey="intraday_rvol" />
                </div>

                <div className="mt-3 space-y-1.5 max-h-60 overflow-y-auto pr-1 scrollbar-thin">
                  {scannerLoading ? (
                    <div className="h-36 flex items-center justify-center text-xs text-slate-400 gap-2">
                      <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                      <span>Scanning high-liquidity stocks...</span>
                    </div>
                  ) : (
                    scannerData.map(item => (
                      <div
                        key={item.ticker}
                        onClick={() => changeTicker(item.ticker)}
                        className={`flex items-center justify-between p-2.5 rounded-xl border transition cursor-pointer ${
                          ticker === item.ticker
                            ? 'bg-cyan-500/15 border-cyan-500/50 shadow-sm'
                            : 'bg-[#070a10] border-white/[0.06] hover:border-slate-700'
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="font-bold text-xs text-white">{item.ticker.split('.')[0]}</span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border ${
                            item.orb_status === 'BREAKOUT' ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' :
                            item.orb_status === 'BREAKDOWN' ? 'bg-rose-500/20 text-rose-300 border-rose-500/40' :
                            'bg-slate-800 text-slate-400 border-slate-700'
                          }`}>
                            {item.orb_status}
                          </span>
                          {item.rvol && (
                            <span className="text-[9px] text-amber-300 font-mono">
                              RVOL {item.rvol}×
                            </span>
                          )}
                        </div>

                        <div className="text-right font-mono text-xs tabular-nums">
                          <span className="text-white font-bold">{item.currency_symbol}{item.price}</span>
                          <span className={`ml-2 font-bold ${item.change_pct >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {item.change_pct >= 0 ? '+' : ''}{item.change_pct}%
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="pt-2 border-t border-white/[0.06] flex items-center justify-between text-xs text-slate-400">
                <span>Click opportunity to load into terminal</span>
                <button onClick={fetchScanner} className="text-cyan-400 hover:text-cyan-300 font-semibold flex items-center gap-1">
                  <RefreshCw className="w-3 h-3" /> Rescan
                </button>
              </div>
            </div>

            {/* NSE Block & Bulk Deals Scanner */}
            <div className="bg-[#0b0f17]/90 border border-white/[0.08] rounded-2xl p-4 sm:p-5 backdrop-blur-md shadow-xl flex flex-col justify-between space-y-3">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
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
                    {blockDeals && (
                      <button onClick={exportBlockDealsCSV} className="text-xs px-2 py-1 rounded bg-slate-800 text-slate-300 hover:text-white" title="Export CSV">
                        <Download className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>

                <div className="mt-3 space-y-1.5 max-h-60 overflow-y-auto pr-1 scrollbar-thin">
                  {scannerMarket !== 'IN' ? (
                    <div className="h-36 flex items-center justify-center text-xs text-slate-500 text-center p-4 bg-[#070a10] rounded-xl border border-white/[0.06]">
                      Block &amp; Bulk deals feed is available for NSE (Indian market) securities.
                    </div>
                  ) : blockDeals ? (
                    [...(blockDeals.block_deals || []).map(d => ({...d, type: 'BLOCK'})),
                     ...(blockDeals.bulk_deals || []).map(d => ({...d, type: 'BULK'}))].slice(0, 10).map((deal, idx) => (
                      <div
                        key={idx}
                        onClick={() => deal.symbol && changeTicker(deal.symbol + '.NS')}
                        className="flex items-center justify-between p-2 rounded-xl bg-[#070a10] border border-white/[0.06] hover:border-slate-700 cursor-pointer transition text-xs"
                      >
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-bold text-white">{deal.symbol}</span>
                            <span className={`text-[8px] font-bold px-1 rounded uppercase ${deal.type === 'BLOCK' ? 'bg-purple-500/20 text-purple-300' : 'bg-orange-500/20 text-orange-300'}`}>
                              {deal.type}
                            </span>
                            <span className={`text-[8px] font-bold px-1 rounded ${deal.trade_type === 'B' || deal.trade_type === 'BUY' ? 'text-emerald-400' : 'text-rose-400'}`}>
                              {deal.trade_type === 'B' || deal.trade_type === 'BUY' ? 'BUY' : 'SELL'}
                            </span>
                          </div>
                          <p className="text-[10px] text-slate-400 truncate max-w-[140px] mt-0.5">{deal.client || 'Undisclosed'}</p>
                        </div>
                        <div className="text-right font-mono tabular-nums">
                          <span className="text-slate-200 font-bold block">{deal.quantity?.toLocaleString() || '—'}</span>
                          <span className="text-[10px] text-slate-400">@ ₹{deal.price || deal.avg_price || '—'}</span>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="h-36 flex items-center justify-center text-xs text-slate-500">
                      Loading NSE block &amp; bulk deal stream...
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-2 border-t border-white/[0.06] text-[11px] text-slate-500 text-right">
                {blockDeals?.window_status ? `Window: ${blockDeals.window_status}` : 'Updated every 5 mins'}
              </div>
            </div>
          </div>

          {/* ── PRO HOTKEYS HUD MODAL ── */}
          {showHotkeysModal && (
            <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
              <div className="bg-[#0b0f17] border border-white/10 rounded-2xl p-5 max-w-md w-full shadow-2xl space-y-4 animate-in fade-in duration-150">
                <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-cyan-500/15 border border-cyan-500/30 rounded-xl text-cyan-400">
                      <Keyboard className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">Pro Keyboard Shortcuts</h3>
                      <p className="text-[11px] text-slate-400">High-frequency navigation HUD</p>
                    </div>
                  </div>
                  <button onClick={() => setShowHotkeysModal(false)} className="p-1 rounded-lg text-slate-400 hover:text-white">
                    <X className="w-4 h-4" />
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    { key: '/', desc: 'Search Ticker' },
                    { key: '1, 2, 3, 5', desc: '1m to 5m TFs' },
                    { key: '4, 6', desc: '15m, 30m TFs' },
                    { key: 'H', desc: '1h Timeframe' },
                    { key: 'V', desc: 'Toggle VWAP' },
                    { key: 'S', desc: 'Toggle Supertrend' },
                    { key: 'C', desc: 'Toggle CPR' },
                    { key: 'K', desc: 'Toggle Heikin-Ashi' },
                    { key: 'R', desc: 'Refresh Data' },
                    { key: 'F', desc: 'Fullscreen Chart' },
                    { key: '?', desc: 'Hotkeys HUD' },
                    { key: 'ESC', desc: 'Dismiss Modal' },
                  ].map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 rounded-xl bg-[#070a10] border border-white/[0.06]">
                      <span className="text-slate-300 text-[11px]">{item.desc}</span>
                      <kbd className="px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 font-mono text-[10px] font-bold">
                        {item.key}
                      </kbd>
                    </div>
                  ))}
                </div>

                <div className="pt-2 border-t border-white/[0.06] flex justify-end">
                  <button
                    onClick={() => setShowHotkeysModal(false)}
                    className="px-4 py-1.5 rounded-xl bg-cyan-500/20 text-cyan-300 font-bold text-xs border border-cyan-500/40"
                  >
                    Got it (Esc)
                  </button>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>

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
