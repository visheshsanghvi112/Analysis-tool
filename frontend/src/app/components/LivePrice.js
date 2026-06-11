'use client';

import { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  RefreshCw, 
  WifiOff, 
  Activity,
  AlertTriangle,
  Clock,
  BarChart3
} from 'lucide-react';

const POLL_INTERVAL_MS = 30_000; // 30 seconds
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-backend.vercel.app' 
  : 'http://127.0.0.1:8000';

function fmt(v, dec = 2) {
  if (v === null || v === undefined) return '—';
  return typeof v === 'number' ? v.toLocaleString('en-IN', { maximumFractionDigits: dec }) : v;
}

function formatVolume(volume) {
  if (!volume) return '—';
  if (volume >= 1e7) return `${(volume / 1e7).toFixed(2)} Cr`;
  if (volume >= 1e5) return `${(volume / 1e5).toFixed(2)} L`;
  return fmt(volume, 0);
}

const StatusIndicator = ({ error, loading }) => {
  if (error) {
    return (
      <div className="flex items-center gap-1.5">
        <AlertTriangle className="h-3 w-3 text-red-400" />
        <span className="text-xs font-semibold text-red-400 uppercase tracking-wider">Offline</span>
      </div>
    );
  }
  
  if (loading) {
    return (
      <div className="flex items-center gap-1.5">
        <div className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
        <span className="text-xs font-semibold text-yellow-400 uppercase tracking-wider">Updating</span>
      </div>
    );
  }
  
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
      <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Live</span>
      <span className="text-xs text-slate-500 bg-slate-800/60 px-2 py-0.5 rounded-md font-mono">
        ~15min delay
      </span>
    </div>
  );
};

const PriceChangeIndicator = ({ change, changePct }) => {
  const isPositive = change >= 0;
  const Icon = isPositive ? TrendingUp : TrendingDown;
  
  return (
    <div className={`flex items-center gap-1.5 font-bold text-lg ${
      isPositive ? 'text-emerald-400' : 'text-red-400'
    }`}>
      <Icon className="h-5 w-5" />
      <span>
        {isPositive ? '+' : ''}{fmt(changePct, 2)}%
      </span>
    </div>
  );
};

const MetricCard = ({ label, value, subtitle, color = 'text-slate-200', icon: Icon }) => (
  <div className="glass-card p-3 border border-slate-800/60 hover:border-slate-700/80 transition-all duration-200">
    <div className="flex items-start justify-between mb-2">
      <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold truncate">
        {label}
      </span>
      {Icon && <Icon className="h-3.5 w-3.5 text-slate-500" />}
    </div>
    <p className={`font-bold text-sm ${color} truncate`}>
      {value}
    </p>
    {subtitle && (
      <p className="text-xs text-slate-600 mt-1 truncate">
        {subtitle}
      </p>
    )}
  </div>
);

const DayRangeBar = ({ dayHigh, dayLow, currentPrice }) => {
  if (!dayHigh || !dayLow || !currentPrice) return null;
  
  const range = dayHigh - dayLow;
  const position = range > 0 ? ((currentPrice - dayLow) / range) * 100 : 50;
  
  return (
    <div className="mt-4 p-3 glass-card border border-slate-800/60">
      <div className="flex justify-between text-xs text-slate-500 mb-2">
        <span>Low ₹{fmt(dayLow)}</span>
        <span className="font-semibold text-slate-300">₹{fmt(currentPrice)}</span>
        <span>High ₹{fmt(dayHigh)}</span>
      </div>
      <div className="relative h-2 bg-slate-800 rounded-full overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-red-500/30 via-yellow-500/30 to-emerald-500/30 rounded-full" />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3 w-1 bg-white rounded-sm shadow-lg transition-all duration-500"
          style={{ left: `calc(${Math.min(Math.max(position, 2), 98)}% - 2px)` }}
        />
      </div>
    </div>
  );
};

export default function LivePrice({ ticker }) {
  const [quote, setQuote] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const timerRef = useRef(null);
  const abortControllerRef = useRef(null);

  const fetchLive = async () => {
    if (!ticker) return;
    
    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    abortControllerRef.current = new AbortController();
    setLoading(true);
    
    try {
      const res = await fetch(
        `${API_BASE_URL}/api/live?ticker=${encodeURIComponent(ticker)}`,
        { 
          signal: abortControllerRef.current.signal,
          headers: {
            'Content-Type': 'application/json',
          }
        }
      );
      
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      
      const json = await res.json();
      setQuote(json);
      setLastUpdated(new Date());
      setError(false);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(true);
        console.error('Failed to fetch live price:', err);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLive();
    
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(fetchLive, POLL_INTERVAL_MS);
    
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, [ticker]);

  const up = quote?.changePct >= 0;

  return (
    <div className="glass-card border border-slate-800/60 shadow-2xl hover:shadow-indigo-900/20 transition-all duration-300">
      {/* Header */}
      <div className="flex items-center justify-between p-4 pb-2">
        <StatusIndicator error={error} loading={loading} />
        
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <span className="text-xs text-slate-500 font-mono hidden sm:flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {lastUpdated.toLocaleTimeString('en-IN', { 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit' 
              })}
            </span>
          )}
          <button
            onClick={fetchLive}
            disabled={loading}
            className="p-2 rounded-lg hover:bg-slate-800/60 text-slate-500 hover:text-slate-300 transition-all disabled:opacity-50 focus-ring"
            title="Refresh quote"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="px-4 pb-4">
        {quote ? (
          <>
            {/* Main Price Display */}
            <div className="flex items-end gap-4 mb-6">
              <div>
                <p className="text-xs text-slate-500 font-mono mb-1 font-semibold">
                  {quote.ticker}
                </p>
                <p className="text-4xl font-black text-white tracking-tight">
                  ₹{fmt(quote.price)}
                </p>
              </div>
              <div className="pb-1">
                <PriceChangeIndicator 
                  change={quote.change} 
                  changePct={quote.changePct} 
                />
                <p className={`text-sm font-semibold ${up ? 'text-emerald-500' : 'text-red-500'}`}>
                  {quote.change >= 0 ? '+' : ''}₹{fmt(quote.change)} today
                </p>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
              <MetricCard
                label="Prev Close"
                value={`₹${fmt(quote.prevClose)}`}
                subtitle="yesterday"
                icon={Activity}
              />
              <MetricCard
                label="Day High"
                value={`₹${fmt(quote.dayHigh)}`}
                color="text-emerald-400"
                icon={TrendingUp}
              />
              <MetricCard
                label="Day Low"
                value={`₹${fmt(quote.dayLow)}`}
                color="text-red-400"
                icon={TrendingDown}
              />
              <MetricCard
                label="Volume"
                value={formatVolume(quote.volume)}
                subtitle="today"
                icon={BarChart3}
              />
            </div>

            {/* Day Range Visualization */}
            <DayRangeBar 
              dayHigh={quote.dayHigh}
              dayLow={quote.dayLow}
              currentPrice={quote.price}
            />

            {/* Footer */}
            <div className="flex justify-between items-center text-xs text-slate-600 mt-4 pt-3 border-t border-slate-800/40">
              <span>Data via Yahoo Finance</span>
              <span>Auto-refresh: 30s</span>
            </div>
          </>
        ) : error ? (
          <div className="flex items-center justify-center gap-3 text-slate-500 py-8">
            <WifiOff className="h-5 w-5" />
            <div className="text-center">
              <p className="font-semibold">Connection Failed</p>
              <p className="text-xs">Backend may be offline</p>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center gap-3 text-slate-500 py-8 animate-pulse">
            <RefreshCw className="h-5 w-5 animate-spin" />
            <div className="text-center">
              <p className="font-semibold">Loading Quote</p>
              <p className="text-xs">Fetching latest data...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
