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
  ExternalLink,
  Coins,
  Building2,
  RefreshCw
} from 'lucide-react';
import { smartSearch, fetchSmartTickers } from '../utils/smartSearch';

const STORAGE_KEY = 'stockiq_pro_watchlist';

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
  const searchInputRef = useRef(null);

  // Load watchlist from localStorage on mount
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

  // Listen to global watchlist updates from other components
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

  // Load static tickers database for quick adding
  useEffect(() => {
    if (allTickers.length === 0 && isOpen) {
      fetch('/tickers.json')
        .then((r) => r.json())
        .then((data) => setAllTickers(data || []))
        .catch(() => {});
    }
  }, [isOpen, allTickers.length]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 150);
    } else {
      setSearchQuery('');
      setSearchResults([]);
    }
  }, [isOpen]);

  // Smart filter search results
  useEffect(() => {
    const q = searchQuery.trim();
    if (!q) {
      setSearchResults([]);
      return;
    }
    const inWatchlist = new Set(watchlist.map((w) => w.symbol));

    // 1. Instant local smart search preview (handles aliases, typos, BSE codes)
    const localMatches = smartSearch(allTickers, q, { limit: 10 })
      .filter((t) => !inWatchlist.has(t.symbol))
      .slice(0, 6);
    setSearchResults(localMatches);

    // 2. Fetch from smart backend endpoint to ensure freshest relevance
    let isCurrent = true;
    fetchSmartTickers(q, allTickers, 10).then((apiMatches) => {
      if (isCurrent && apiMatches && apiMatches.length > 0) {
        const filtered = apiMatches.filter((t) => !inWatchlist.has(t.symbol)).slice(0, 6);
        if (filtered.length > 0) setSearchResults(filtered);
      }
    });

    return () => { isCurrent = false; };
  }, [searchQuery, allTickers, watchlist]);

  // Persist watchlist changes
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
      className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm transition-opacity duration-200"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-md bg-[#0a0a0c] border-l border-white/10 h-full flex flex-col shadow-2xl animate-in slide-in-from-right duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.08] bg-white/[0.02]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white tracking-wide">Watchlist</h2>
              <p className="text-[11px] text-slate-400">{watchlist.length} saved instruments</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition-colors"
            title="Close drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Quick Add Search */}
        <div className="p-4 border-b border-white/[0.06] bg-black/40">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Add stock or ETF (e.g. NIFTYBEES, RELIANCE)…"
              className="w-full bg-[#111114] border border-white/10 rounded-lg pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
            />
          </div>

          {/* Search Dropdown Results */}
          {searchResults.length > 0 && (
            <div className="mt-2 bg-[#141418] border border-white/10 rounded-lg overflow-hidden shadow-xl max-h-48 overflow-y-auto">
              {searchResults.map((t) => (
                <button
                  key={t.symbol}
                  onClick={() => addTicker(t)}
                  className="w-full text-left px-3 py-2 flex items-center justify-between hover:bg-white/[0.06] transition-colors border-b border-white/[0.03] last:border-0"
                >
                  <div className="min-w-0 pr-2">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-bold text-white">{t.symbol}</span>
                      {t.sector === 'ETF' && (
                        <span className="text-[9px] px-1.5 py-0.2 bg-amber-500/20 text-amber-300 font-semibold rounded">ETF</span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-400 truncate">{t.name}</p>
                  </div>
                  <div className="flex items-center gap-1 text-[11px] font-semibold text-emerald-400 shrink-0">
                    <Plus className="w-3.5 h-3.5" /> Add
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Watchlist Items */}
        <div className="flex-1 overflow-y-auto divide-y divide-white/[0.04]">
          {watchlist.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-48 text-center px-6">
              <Star className="w-8 h-8 text-slate-700 mb-2" />
              <p className="text-xs text-slate-400 font-medium">Your watchlist is empty</p>
              <p className="text-[11px] text-slate-600 mt-1">Search above to pin any of 7,900+ stocks or ETFs</p>
            </div>
          ) : (
            watchlist.map((item) => {
              const isSelected = currentTicker === item.symbol;
              const isETF = item.sector === 'ETF';
              return (
                <div
                  key={item.symbol}
                  onClick={() => {
                    onSelectTicker(item.symbol);
                    onClose();
                  }}
                  className={`group flex items-center justify-between px-4 py-3 cursor-pointer transition-all hover:bg-white/[0.04] ${
                    isSelected ? 'bg-blue-500/10 border-l-2 border-blue-500' : ''
                  }`}
                >
                  <div className="min-w-0 flex-1 pr-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-white group-hover:text-blue-400 transition-colors">
                        {item.symbol}
                      </span>
                      {isETF ? (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 font-semibold border border-amber-500/30">
                          ETF
                        </span>
                      ) : (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.05] text-slate-400 font-medium">
                          {item.sector || 'Equity'}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 truncate mt-0.5 font-normal">{item.name}</p>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <button
                      onClick={(e) => removeTicker(item.symbol, e)}
                      className="opacity-0 group-hover:opacity-100 p-1.5 text-slate-500 hover:text-red-400 rounded hover:bg-red-500/10 transition-all"
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

        {/* Footer */}
        <div className="p-3.5 border-t border-white/[0.08] bg-white/[0.01] flex items-center justify-between text-[11px] text-slate-500">
          <span>Synced locally across sessions</span>
          <span className="font-mono text-slate-600">StockIQ Universe</span>
        </div>
      </div>
    </div>
  );
}
