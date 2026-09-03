'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Search, 
  X, 
  Clock, 
  TrendingUp, 
  Building2, 
  ChevronRight, 
  Command, 
  Layers, 
  Zap, 
  Coins, 
  Globe2, 
  BarChart3,
  Sparkles,
  Trash2
} from 'lucide-react';
import { API_BASE_URL } from '../config';

const CATEGORIES = [
  { id: 'all', label: 'All', icon: Layers },
  { id: 'equity', label: 'Equities', icon: Building2 },
  { id: 'etf', label: 'ETFs & Funds', icon: Coins },
  { id: 'index', label: 'Indices', icon: BarChart3 },
  { id: 'global', label: 'Global', icon: Globe2 },
];

const TRENDING_PICKS = [
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries Limited', sector: 'Energy', type: 'Equity' },
  { symbol: 'TCS.NS', name: 'Tata Consultancy Services', sector: 'IT', type: 'Equity' },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Limited', sector: 'Banking', type: 'Equity' },
  { symbol: '^NSEI', name: 'NIFTY 50 (Benchmark Index)', sector: 'Indices', type: 'Index' },
  { symbol: 'NIFTYBEES.NS', name: 'Nippon India Nifty 50 ETF', sector: 'ETF', type: 'ETF' },
  { symbol: 'GOLDBEES.NS', name: 'Nippon India Gold ETF', sector: 'ETF', type: 'ETF' },
  { symbol: 'MON100.NS', name: 'Motilal Oswal Nasdaq 100 ETF', sector: 'ETF', type: 'ETF' },
  { symbol: 'SWIGGY.NS', name: 'Swiggy Limited', sector: 'Consumer Tech', type: 'Equity' },
  { symbol: 'WAAREEENER.NS', name: 'Waaree Energies Limited', sector: 'Renewables', type: 'Equity' },
  { symbol: 'ETERNAL.NS', name: 'Eternal Limited (Zomato)', sector: 'Consumer Tech', type: 'Equity' },
];

const SECTOR_STYLES = {
  'Banking': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  'IT': 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  'Energy': 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  'Auto': 'bg-red-500/10 text-red-400 border-red-500/20',
  'FMCG': 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  'Pharma': 'bg-teal-500/10 text-teal-400 border-teal-500/20',
  'ETF': 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
  'Indices': 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  'Global': 'bg-purple-500/10 text-purple-400 border-purple-500/20',
};

export default function StockSearchModal({ isOpen, onClose, onSelect, currentTicker }) {
  const [query, setQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeIdx, setActiveIdx] = useState(0);
  const [recentSearches, setRecentSearches] = useState([]);
  const [localUniverse, setLocalUniverse] = useState([]);

  const inputRef = useRef(null);
  const debounceRef = useRef(null);
  const listRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Load local universe cache eagerly once
  useEffect(() => {
    fetch('/tickers.json')
      .then(res => res.ok ? res.json() : [])
      .then(data => {
        if (Array.isArray(data)) setLocalUniverse(data);
      })
      .catch(() => {});
  }, []);

  // Load recent searches from localStorage
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('stockiq_recent_searches') || '[]');
      setRecentSearches(stored);
    } catch (_) {}
  }, [isOpen]);

  // Global Keyboard Shortcuts (⌘K, Ctrl+K, /)
  useEffect(() => {
    const handleGlobalKey = (e) => {
      // Don't trigger if already in another input/textarea
      const tag = document.activeElement?.tagName;
      const isInputFocused = tag === 'INPUT' || tag === 'TEXTAREA';

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (isOpen) onClose();
        else window.dispatchEvent(new CustomEvent('open-stock-search'));
      } else if (e.key === '/' && !isInputFocused && !isOpen) {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent('open-stock-search'));
      }
    };

    window.addEventListener('keydown', handleGlobalKey);
    return () => window.removeEventListener('keydown', handleGlobalKey);
  }, [isOpen, onClose]);

  // Reset & focus on open
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedCategory('all');
      setResults([]);
      setActiveIdx(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Filter items by category
  const filterByCategory = useCallback((items, cat) => {
    if (!cat || cat === 'all') return items;
    return items.filter(item => {
      const type = (item.type || '').toLowerCase();
      const sector = (item.sector || '').toLowerCase();
      const sym = item.symbol || '';

      if (cat === 'etf') {
        return type.includes('etf') || sector.includes('etf') || sym.includes('BEES');
      }
      if (cat === 'index') {
        return sym.startsWith('^') || type.includes('index') || sector.includes('indices');
      }
      if (cat === 'global') {
        return type.includes('global') || !sym.includes('.');
      }
      if (cat === 'equity') {
        return !sym.startsWith('^') && !type.includes('etf') && !type.includes('index') && !sector.includes('etf');
      }
      return true;
    });
  }, []);

  // Client-side instant fallback search
  const searchLocalUniverse = useCallback((q) => {
    if (!localUniverse.length || !q) return [];
    const qLower = q.toLowerCase().trim();
    const qClean = qLower.replace('.ns', '').replace('.bo', '').replace('^', '');

    const matches = [];
    for (const t of localUniverse) {
      const sym = (t.symbol || '').toLowerCase();
      const name = (t.name || '').toLowerCase();
      const bse = t.bse_code || '';

      if (sym.includes(qClean) || name.includes(qLower) || bse === qLower) {
        matches.push(t);
        if (matches.length >= 30) break;
      }
    }
    return matches;
  }, [localUniverse]);

  // Fetch with debounced API & instant local preview
  const performSearch = useCallback((rawQuery, cat) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (abortControllerRef.current) abortControllerRef.current.abort();

    let cleanQuery = rawQuery.trim();
    let effectiveCat = cat;

    // Smart slash command detection (e.g. /etf gold, /bank hdfc)
    if (cleanQuery.startsWith('/etf')) {
      effectiveCat = 'etf';
      cleanQuery = cleanQuery.replace('/etf', '').trim();
      setSelectedCategory('etf');
    } else if (cleanQuery.startsWith('/index')) {
      effectiveCat = 'index';
      cleanQuery = cleanQuery.replace('/index', '').trim();
      setSelectedCategory('index');
    }

    if (!cleanQuery) {
      setResults([]);
      setLoading(false);
      setActiveIdx(0);
      return;
    }

    // Instant local preview
    const instantLocal = filterByCategory(searchLocalUniverse(cleanQuery), effectiveCat);
    if (instantLocal.length > 0) {
      setResults(instantLocal);
    }

    setLoading(true);
    abortControllerRef.current = new AbortController();

    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/tickers?q=${encodeURIComponent(cleanQuery)}&limit=40`,
          { signal: abortControllerRef.current.signal }
        );
        if (res.ok) {
          const data = await res.json();
          const apiList = data.tickers || [];
          const filtered = filterByCategory(apiList, effectiveCat);
          setResults(filtered.length > 0 ? filtered : instantLocal);
          setActiveIdx(0);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setResults(instantLocal);
        }
      } finally {
        setLoading(false);
      }
    }, 150);
  }, [API_BASE_URL, filterByCategory, searchLocalUniverse]);

  const handleQueryChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    performSearch(val, selectedCategory);
  };

  const handleCategoryChange = (catId) => {
    setSelectedCategory(catId);
    if (query) {
      performSearch(query, catId);
    }
  };

  const saveRecent = (item) => {
    try {
      const prev = JSON.parse(localStorage.getItem('stockiq_recent_searches') || '[]');
      const filtered = prev.filter(r => r.symbol !== item.symbol);
      const updated = [item, ...filtered].slice(0, 8);
      localStorage.setItem('stockiq_recent_searches', JSON.stringify(updated));
      setRecentSearches(updated);
    } catch (_) {}
  };

  const removeRecent = (e, sym) => {
    e.stopPropagation();
    try {
      const updated = recentSearches.filter(r => r.symbol !== sym);
      localStorage.setItem('stockiq_recent_searches', JSON.stringify(updated));
      setRecentSearches(updated);
    } catch (_) {}
  };

  const clearAllRecent = (e) => {
    e.stopPropagation();
    localStorage.removeItem('stockiq_recent_searches');
    setRecentSearches([]);
  };

  const handleSelect = (item) => {
    saveRecent(item);
    if (onSelect) onSelect(item.symbol);
    onClose();
  };

  const displayedList = query ? results : (recentSearches.length > 0 ? recentSearches : TRENDING_PICKS);

  // Keyboard Navigation: ArrowUp, ArrowDown, Enter, Escape
  const handleKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx(prev => Math.min(prev + 1, displayedList.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (displayedList[activeIdx]) {
        handleSelect(displayedList[activeIdx]);
      } else if (query.trim()) {
        const raw = query.trim().toUpperCase();
        handleSelect({ symbol: raw.includes('.') || raw.startsWith('^') ? raw : `${raw}.NS`, name: raw });
      }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      const currIdx = CATEGORIES.findIndex(c => c.id === selectedCategory);
      const nextIdx = (currIdx + 1) % CATEGORIES.length;
      handleCategoryChange(CATEGORIES[nextIdx].id);
    }
  };

  // Scroll active item into view
  useEffect(() => {
    if (listRef.current && listRef.current.children[activeIdx]) {
      listRef.current.children[activeIdx].scrollIntoView({ block: 'nearest' });
    }
  }, [activeIdx]);

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-150"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-2xl bg-[#0c0c10] border border-white/[0.12] rounded-2xl shadow-2xl overflow-hidden text-slate-200 flex flex-col max-h-[82vh]"
        onClick={e => e.stopPropagation()}
      >
        {/* Search Bar Header */}
        <div className="flex items-center px-4 py-3.5 border-b border-white/[0.08] bg-white/[0.02]">
          <Search className={`w-5 h-5 mr-3 transition-colors ${loading ? 'text-blue-400 animate-pulse' : 'text-slate-400'}`} />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleQueryChange}
            onKeyDown={handleKeyDown}
            placeholder="Search 7,950+ stocks, ETFs, indices, or BSE code (e.g. Reliance, Nifty, 500325)…"
            className="flex-1 bg-transparent text-sm text-white placeholder-slate-500 focus:outline-none tracking-wide"
          />
          {query && (
            <button 
              onClick={() => { setQuery(''); setResults([]); inputRef.current?.focus(); }}
              className="p-1 text-slate-400 hover:text-white rounded-md hover:bg-white/[0.06] mr-2"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          <div className="flex items-center gap-1.5 text-[10px] text-slate-500 bg-white/[0.04] border border-white/[0.06] px-2 py-1 rounded-md">
            <Command className="w-3 h-3" />
            <span>K</span>
          </div>
        </div>

        {/* Category Filters Pill Strip */}
        <div className="flex items-center gap-1.5 px-4 py-2 border-b border-white/[0.06] bg-black/30 overflow-x-auto no-scrollbar">
          {CATEGORIES.map(cat => {
            const Icon = cat.icon;
            const isSelected = selectedCategory === cat.id;
            return (
              <button
                key={cat.id}
                onClick={() => handleCategoryChange(cat.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold whitespace-nowrap transition-all cursor-pointer ${
                  isSelected 
                    ? 'bg-blue-600 text-white shadow-sm' 
                    : 'bg-white/[0.03] text-slate-400 hover:text-slate-200 hover:bg-white/[0.06]'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{cat.label}</span>
              </button>
            );
          })}
        </div>

        {/* Section Label */}
        <div className="px-4 py-2 flex items-center justify-between text-[11px] font-semibold tracking-wider text-slate-400 uppercase bg-white/[0.01]">
          <span>{query ? `Results (${displayedList.length})` : (recentSearches.length > 0 ? 'Recent Searches' : 'Trending Benchmarks')}</span>
          {!query && recentSearches.length > 0 && (
            <button 
              onClick={clearAllRecent}
              className="text-slate-500 hover:text-red-400 transition-colors flex items-center gap-1 text-[10px]"
            >
              <Trash2 className="w-3 h-3" /> Clear history
            </button>
          )}
        </div>

        {/* Results List */}
        <div ref={listRef} className="flex-1 overflow-y-auto divide-y divide-white/[0.04] p-1.5">
          {displayedList.length === 0 ? (
            <div className="py-16 text-center text-slate-400">
              <Zap className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-300">No instruments found</p>
              <p className="text-xs text-slate-500 mt-1">Try searching by company name, symbol, or 6-digit BSE scrip code.</p>
            </div>
          ) : (
            displayedList.map((item, idx) => {
              const isSelected = idx === activeIdx;
              const isCurrent = item.symbol === currentTicker;
              const sector = item.sector || (item.symbol?.startsWith('^') ? 'Indices' : 'Other');
              const badgeStyle = SECTOR_STYLES[sector] || 'bg-slate-800 text-slate-400 border-slate-700';

              return (
                <div
                  key={item.symbol || idx}
                  onClick={() => handleSelect(item)}
                  onMouseEnter={() => setActiveIdx(idx)}
                  className={`flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer transition-all ${
                    isSelected 
                      ? 'bg-blue-600/15 border border-blue-500/30' 
                      : 'hover:bg-white/[0.03] border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-black text-[11px] shrink-0 ${
                      item.symbol.startsWith('^') 
                        ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/20' 
                        : item.symbol.includes('BEES') || sector === 'ETF'
                          ? 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/20'
                          : item.symbol.endsWith('.BO')
                            ? 'bg-orange-500/15 text-orange-400 border border-orange-500/20'
                            : 'bg-blue-500/15 text-blue-400 border border-blue-500/20'
                    }`}>
                      {item.symbol.startsWith('^') ? 'IDX' : item.symbol.includes('BEES') ? 'ETF' : item.symbol.endsWith('.BO') ? 'BSE' : 'NSE'}
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-sm text-white tracking-wide truncate">
                          {item.symbol.replace('.NS', '').replace('.BO', '')}
                        </span>
                        {item.bse_code && (
                          <span className="text-[10px] text-slate-500 font-mono">
                            #{item.bse_code}
                          </span>
                        )}
                        {isCurrent && (
                          <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-emerald-500/20 text-emerald-400">
                            ACTIVE
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 truncate mt-0.5">
                        {item.name || item.symbol}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 shrink-0 ml-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${badgeStyle}`}>
                      {sector}
                    </span>
                    {!query && recentSearches.length > 0 && (
                      <button
                        onClick={(e) => removeRecent(e, item.symbol)}
                        className="p-1 text-slate-600 hover:text-slate-300 transition-colors"
                        title="Remove from history"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                    <ChevronRight className="w-4 h-4 text-slate-600" />
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Modal Footer Key Hints */}
        <div className="px-4 py-2.5 border-t border-white/[0.06] bg-black/40 flex items-center justify-between text-[11px] text-slate-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.08] text-slate-400 font-mono text-[10px]">↑↓</kbd> Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.08] text-slate-400 font-mono text-[10px]">↵</kbd> Select
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.08] text-slate-400 font-mono text-[10px]">Tab</kbd> Switch Tab
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 rounded bg-white/[0.08] text-slate-400 font-mono text-[10px]">Esc</kbd> Close
            </span>
          </div>
          <span className="hidden sm:inline text-slate-500 text-[10px]">
            StockIQ Smart Search v2.0
          </span>
        </div>
      </div>
    </div>
  );
}
