'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Star, 
  X, 
  TrendingUp, 
  TrendingDown, 
  Plus, 
  Trash2, 
  Search, 
  ChevronRight,
  RefreshCw,
  Loader2,
  Sparkles
} from 'lucide-react';
import { smartSearch, fetchSmartTickers } from '../utils/smartSearch';

const STORAGE_KEY = 'stockiq_pro_watchlist';
const API_BASE = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

const DEFAULT_WATCHLIST = [
  { symbol: 'NIFTYBEES.NS', name: 'Nippon India Nifty 50 BeES', sector: 'ETF' },
  { symbol: 'GOLDBEES.NS', name: 'Nippon India Gold BeES', sector: 'ETF' },
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries Limited', sector: 'Energy' },
  { symbol: 'TCS.NS', name: 'Tata Consultancy Services', sector: 'IT' },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Limited', sector: 'Banking' }
];

export default function WatchlistDrawer({ isOpen, onClose, onSelectTicker, currentTicker }) {
  const [watchlist, setWatchlist] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [allTickers, setAllTickers] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [quotes, setQuotes] = useState({});
  const [loadingQuotes, setLoadingQuotes] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const searchInputRef = useRef(null);

  // ── 1. Load watchlist from localStorage on mount ────────────────────────
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setWatchlist(parsed);
          return;
        }
      }
    } catch (e) {
      console.warn('Failed to parse watchlist from storage:', e);
    }
    setWatchlist(DEFAULT_WATCHLIST);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_WATCHLIST));
    } catch (_) {}
  }, []);

  // ── 2. Listen to global watchlist updates from other components ────────
  useEffect(() => {
    const handleStorage = () => {
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) setWatchlist(JSON.parse(saved));
      } catch (_) {}
    };
    window.addEventListener('stockiq-watchlist-changed', handleStorage);
    return () => window.removeEventListener('stockiq-watchlist-changed', handleStorage);
  }, []);

  // ── 3. Load full 7,954 master database for instant local smart search ───
  useEffect(() => {
    if (allTickers.length === 0 && isOpen) {
      fetch('/tickers.json')
        .then((r) => r.json())
        .then((data) => setAllTickers(data || []))
        .catch(() => {});
    }
  }, [isOpen, allTickers.length]);

  // ── 4. Focus input when opened & listen for Escape key ───────────────────
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 150);
      const handleKeyDown = (e) => {
        if (e.key === 'Escape') onClose();
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    } else {
      setSearchQuery('');
      setSearchResults([]);
    }
  }, [isOpen, onClose]);

  // ── 5. Batch-fetch live price quotes for all watchlist items ───────────
  const fetchWatchlistQuotes = useCallback(async () => {
    if (watchlist.length === 0) return;
    setLoadingQuotes(true);
    try {
      const symbolsParam = watchlist.map((w) => w.symbol).join(',');
      const res = await fetch(`${API_BASE}/api/batch-quotes?tickers=${encodeURIComponent(symbolsParam)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.quotes) {
          setQuotes(data.quotes);
          setLastRefreshed(new Date());
        }
      }
    } catch (e) {
      console.warn('Failed to fetch batch quotes for watchlist:', e);
    } finally {
      setLoadingQuotes(false);
    }
  }, [watchlist]);

  useEffect(() => {
    if (isOpen && watchlist.length > 0) {
      fetchWatchlistQuotes();
      // Auto-poll quotes every 30s while drawer is open
      const timer = setInterval(fetchWatchlistQuotes, 30000);
      return () => clearInterval(timer);
    }
  }, [isOpen, watchlist, fetchWatchlistQuotes]);

  // ── 6. Smart filter search results (Handles aliases, typos, BSE codes) ───
  useEffect(() => {
    const q = searchQuery.trim();
    if (!q) {
      setSearchResults([]);
      return;
    }
    const inWatchlist = new Set(watchlist.map((w) => w.symbol));

    // Instant local smart search
    const localMatches = smartSearch(allTickers, q, { limit: 10 })
      .filter((t) => !inWatchlist.has(t.symbol))
      .slice(0, 6);
    setSearchResults(localMatches);

    // Complement with backend smart search
    let isCurrent = true;
    fetchSmartTickers(q, allTickers, 10).then((apiMatches) => {
      if (isCurrent && apiMatches && apiMatches.length > 0) {
        const filtered = apiMatches.filter((t) => !inWatchlist.has(t.symbol)).slice(0, 6);
        if (filtered.length > 0) setSearchResults(filtered);
      }
    });

    return () => { isCurrent = false; };
  }, [searchQuery, allTickers, watchlist]);

  // ── 7. Watchlist actions ───────────────────────────────────────────────
  const saveWatchlist = useCallback((newList) => {
    setWatchlist(newList);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newList));
      window.dispatchEvent(new CustomEvent('stockiq-watchlist-changed'));
    } catch (_) {}
  }, []);

  const addTicker = (tickerObj) => {
    if (watchlist.some((w) => w.symbol === tickerObj.symbol)) return;
    const item = {
      symbol: tickerObj.symbol,
      name: tickerObj.name,
      sector: tickerObj.sector || 'Others'
    };
    const updated = [item, ...watchlist];
    saveWatchlist(updated);
    setSearchQuery('');
    setSearchResults([]);
  };

  const removeTicker = (symbol, e) => {
    e.stopPropagation();
    const updated = watchlist.filter((w) => w.symbol !== symbol);
    saveWatchlist(updated);
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-[120] flex justify-end bg-black/75 backdrop-blur-md transition-opacity duration-200"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-md bg-[#0d0e12] border-l border-white/10 h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-250 select-none"
        onClick={(e) => e.stopPropagation()}
      >
        {/* ── Drawer Header ─────────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.08] bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/25 flex items-center justify-center shadow-inner">
              <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-white tracking-wide">Watchlist</h2>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 font-bold border border-amber-500/30">
                  {watchlist.length}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Live prices &amp; intraday change</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={fetchWatchlistQuotes}
              disabled={loadingQuotes}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition-colors disabled:opacity-50"
              title="Refresh live prices"
            >
              <RefreshCw className={`w-4 h-4 ${loadingQuotes ? 'animate-spin text-blue-400' : ''}`} />
            </button>
            <button 
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition-colors"
              title="Close drawer (Esc)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* ── Quick Add Search Bar ──────────────────────────────────────── */}
        <div className="p-4 border-b border-white/[0.06] bg-[#0b0c0f]">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search 7,950+ stocks or ETFs to add..."
              className="w-full bg-[#14151b] border border-white/10 rounded-xl pl-10 pr-3 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/70 focus:ring-1 focus:ring-blue-500/40 transition-all"
            />
          </div>

          {/* Search Dropdown Results */}
          {searchResults.length > 0 && (
            <div className="mt-2 bg-[#16171f] border border-white/10 rounded-xl overflow-hidden shadow-2xl max-h-56 overflow-y-auto divide-y divide-white/[0.04]">
              {searchResults.map((t) => (
                <button
                  key={t.symbol}
                  onClick={() => addTicker(t)}
                  className="w-full text-left px-3.5 py-2.5 flex items-center justify-between hover:bg-white/[0.06] transition-colors group"
                >
                  <div className="min-w-0 pr-2">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-bold text-white group-hover:text-blue-400 transition-colors">
                        {t.symbol.replace('.NS', '').replace('.BO', '')}
                      </span>
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-white/[0.06] text-slate-400 font-medium">
                        {t.sector || 'Equity'}
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-400 truncate mt-0.5">{t.name}</p>
                  </div>
                  <div className="flex items-center gap-1 text-[11px] font-semibold text-emerald-400 shrink-0 bg-emerald-500/10 border border-emerald-500/20 px-2 py-1 rounded-md">
                    <Plus className="w-3 h-3" /> Add
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Watchlist Rows with Live Price & Change % ─────────────────── */}
        <div className="flex-1 overflow-y-auto divide-y divide-white/[0.04]">
          {watchlist.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-center px-6">
              <Star className="w-10 h-10 text-slate-700 mb-3" />
              <p className="text-sm text-slate-300 font-semibold">Your watchlist is empty</p>
              <p className="text-xs text-slate-500 mt-1 max-w-[240px]">
                Search above to pin any of our 7,954 NSE, BSE, ETF, or global instruments.
              </p>
            </div>
          ) : (
            watchlist.map((item) => {
              const isSelected = currentTicker === item.symbol;
              const isETF = item.sector === 'ETF';
              const isIndex = item.symbol.startsWith('^') || item.sector === 'Indices';
              const isUS = item.symbol && !item.symbol.endsWith('.NS') && !item.symbol.endsWith('.BO');
              const quote = quotes[item.symbol];
              const currSym = quote?.currency_symbol || (isUS ? '$' : '₹');
              const loc = isUS ? 'en-US' : 'en-IN';

              const price = quote?.price;
              const changePct = quote?.changePct;
              const isPositive = changePct > 0;
              const isNegative = changePct < 0;

              return (
                <div
                  key={item.symbol}
                  onClick={() => {
                    onSelectTicker(item.symbol);
                    onClose();
                  }}
                  className={`group flex items-center justify-between px-4 py-3.5 cursor-pointer transition-all hover:bg-white/[0.04] ${
                    isSelected ? 'bg-blue-500/10 border-l-4 border-blue-500' : 'border-l-4 border-transparent'
                  }`}
                >
                  {/* Left: Ticker identity */}
                  <div className="min-w-0 flex-1 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-white group-hover:text-blue-400 transition-colors">
                        {item.symbol.replace('.NS', '').replace('.BO', '')}
                      </span>
                      {isETF ? (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 font-semibold border border-amber-500/30">
                          ETF
                        </span>
                      ) : isIndex ? (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-sky-500/15 text-sky-300 font-semibold border border-sky-500/30">
                          INDEX
                        </span>
                      ) : (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.05] text-slate-400 font-medium">
                          {item.sector || 'Equity'}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 truncate mt-0.5 font-normal">{item.name}</p>
                  </div>

                  {/* Right: Real-Time Price & Day Change % */}
                  <div className="flex items-center gap-2.5 shrink-0">
                    <div className="text-right">
                      {price !== undefined && price !== null ? (
                        <>
                          <div className="text-xs font-bold text-white font-mono">
                            {currSym}{typeof price === 'number' ? price.toLocaleString(loc, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : price}
                          </div>
                          {changePct !== undefined && changePct !== null && (
                            <div className={`inline-flex items-center gap-0.5 text-[10px] font-bold px-1.5 py-0.5 rounded mt-0.5 ${
                              isPositive ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/25' :
                              isNegative ? 'bg-rose-500/15 text-rose-400 border border-rose-500/25' :
                              'bg-slate-500/15 text-slate-400 border border-slate-500/25'
                            }`}>
                              {isPositive && <TrendingUp className="w-2.5 h-2.5" />}
                              {isNegative && <TrendingDown className="w-2.5 h-2.5" />}
                              <span>{isPositive ? '+' : ''}{changePct.toFixed(2)}%</span>
                            </div>
                          )}
                        </>
                      ) : loadingQuotes ? (
                        <div className="space-y-1 text-right">
                          <div className="w-14 h-3.5 bg-white/10 rounded animate-pulse" />
                          <div className="w-10 h-3 bg-white/5 rounded animate-pulse ml-auto" />
                        </div>
                      ) : (
                        <span className="text-[11px] text-slate-600 font-mono">--</span>
                      )}
                    </div>

                    {/* Delete action */}
                    <button
                      onClick={(e) => removeTicker(item.symbol, e)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 text-slate-500 hover:text-red-400 rounded-lg hover:bg-red-500/10 transition-all"
                      title="Remove from watchlist"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                    <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-slate-300 group-hover:translate-x-0.5 transition-all" />
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* ── Drawer Footer ─────────────────────────────────────────────── */}
        <div className="p-3.5 border-t border-white/[0.08] bg-black/40 flex items-center justify-between text-[11px] text-slate-500">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>100% Live Quotes</span>
          </div>
          <span className="text-slate-600 font-mono">
            {lastRefreshed ? `Updated ${lastRefreshed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}` : 'Auto-synced'}
          </span>
        </div>
      </div>
    </div>
  );
}
