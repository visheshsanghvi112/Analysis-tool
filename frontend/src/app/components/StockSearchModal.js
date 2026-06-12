'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, X, RefreshCw, TrendingUp, Building2, ChevronRight } from 'lucide-react';

// Popular/quick-pick stocks shown when modal opens before any search
const POPULAR_STOCKS = [
  { symbol: 'RELIANCE.NS', name: 'Reliance Industries Ltd', sector: 'Energy' },
  { symbol: 'TCS.NS',      name: 'Tata Consultancy Services', sector: 'IT' },
  { symbol: 'HDFCBANK.NS', name: 'HDFC Bank Ltd', sector: 'Banking' },
  { symbol: 'INFY.NS',     name: 'Infosys Ltd', sector: 'IT' },
  { symbol: 'ICICIBANK.NS',name: 'ICICI Bank Ltd', sector: 'Banking' },
  { symbol: 'SBIN.NS',     name: 'State Bank of India', sector: 'Banking' },
  { symbol: 'WIPRO.NS',    name: 'Wipro Ltd', sector: 'IT' },
  { symbol: 'HCLTECH.NS',  name: 'HCL Technologies Ltd', sector: 'IT' },
  { symbol: 'BAJFINANCE.NS', name: 'Bajaj Finance Ltd', sector: 'NBFC' },
  { symbol: 'ASIANPAINT.NS', name: 'Asian Paints Ltd', sector: 'FMCG' },
  { symbol: 'TITAN.NS',    name: 'Titan Company Ltd', sector: 'Consumer' },
  { symbol: 'MARUTI.NS',   name: 'Maruti Suzuki India Ltd', sector: 'Auto' },
];

const SECTOR_COLORS = {
  'IT': 'bg-blue-500/15 text-blue-400 border-blue-500/20',
  'Banking': 'bg-emerald-500/15 text-emerald-400 border-emerald-500/20',
  'Energy': 'bg-orange-500/15 text-orange-400 border-orange-500/20',
  'NBFC': 'bg-purple-500/15 text-purple-400 border-purple-500/20',
  'FMCG': 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20',
  'Auto': 'bg-red-500/15 text-red-400 border-red-500/20',
  'Consumer': 'bg-pink-500/15 text-pink-400 border-pink-500/20',
};

function getSectorStyle(sector) {
  return SECTOR_COLORS[sector] || 'bg-slate-700/40 text-slate-400 border-slate-600/20';
}

export default function StockSearchModal({ isOpen, onClose, onSelect, currentTicker }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [recentSearches, setRecentSearches] = useState([]);
  const inputRef = useRef(null);
  const debounceRef = useRef(null);
  const listRef = useRef(null);
  const abortControllerRef = useRef(null);

  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

  // Load recent searches from localStorage
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('recentStocks') || '[]');
      setRecentSearches(stored);
    } catch { /* ignore */ }
  }, []);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setResults([]);
      setActiveIdx(-1);
      setTimeout(() => inputRef.current?.focus(), 60);
    }
  }, [isOpen]);

  // Close on Escape & cleanup
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    if (isOpen) document.addEventListener('keydown', handler);
    return () => {
      document.removeEventListener('keydown', handler);
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
    };
  }, [isOpen, onClose]);

  // Fetch results with debounce
  const fetchResults = useCallback((q) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (abortControllerRef.current) abortControllerRef.current.abort();

    if (!q || q.length < 1) {
      setResults([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    abortControllerRef.current = new AbortController();

    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(
          `${API_BASE_URL}/api/tickers?q=${encodeURIComponent(q)}`,
          { signal: abortControllerRef.current.signal }
        );
        const json = await res.json();
        if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
          setResults(json.tickers || []);
          setActiveIdx(-1);
        }
      } catch (err) {
        if (err.name !== 'AbortError') {
          setResults([]);
        }
      } finally {
        if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
          setLoading(false);
        }
      }
    }, 200);
  }, [API_BASE_URL]);

  const handleQueryChange = (e) => {
    const val = e.target.value;
    setQuery(val);
    fetchResults(val);
  };

  const saveRecent = (item) => {
    try {
      const prev = JSON.parse(localStorage.getItem('recentStocks') || '[]');
      const filtered = prev.filter((r) => r.symbol !== item.symbol);
      const updated = [item, ...filtered].slice(0, 6);
      localStorage.setItem('recentStocks', JSON.stringify(updated));
      setRecentSearches(updated);
    } catch { /* ignore */ }
  };

  const handleSelect = (item) => {
    saveRecent(item);
    onSelect(item);
    onClose();
  };

  const handleKeyDown = (e) => {
    const list = query ? results : POPULAR_STOCKS;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx((prev) => Math.min(prev + 1, list.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx((prev) => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter') {
      if (activeIdx >= 0 && activeIdx < list.length) {
        handleSelect(list[activeIdx]);
      } else if (query.trim()) {
        // Direct symbol entry
        const sym = query.trim().toUpperCase();
        const direct = { symbol: sym.includes('.') ? sym : sym + '.NS', name: sym };
        handleSelect(direct);
      }
    }
  };

  // Scroll active item into view
  useEffect(() => {
    if (activeIdx >= 0 && listRef.current) {
      const el = listRef.current.children[activeIdx];
      el?.scrollIntoView({ block: 'nearest' });
    }
  }, [activeIdx]);

  if (!isOpen) return null;

  const showPopular = !query;
  const displayList = query ? results : [];

  return (
    /* Backdrop - Mobile Responsive */
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[5vh] sm:pt-[8vh] px-3 sm:px-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      {/* Blurred dark overlay */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

      {/* Modal card - Mobile Responsive */}
      <div className="relative w-full max-w-xl bg-[#0d1424] border border-slate-700 rounded-xl sm:rounded-2xl shadow-2xl overflow-hidden">

        {/* Search Input - Mobile Responsive */}
        <div className="flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-3 sm:py-3.5 border-b border-slate-800">
          <Search className="h-4 sm:h-5 w-4 sm:w-5 text-indigo-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleQueryChange}
            onKeyDown={handleKeyDown}
            placeholder="Search by company name or symbol…"
            className="flex-1 bg-transparent text-white text-sm placeholder-slate-500 focus:outline-none font-medium"
            autoComplete="off"
            spellCheck={false}
          />
          {loading && <RefreshCw className="h-3.5 sm:h-4 w-3.5 sm:w-4 text-slate-500 animate-spin shrink-0" />}
          <button
            onClick={onClose}
            className="ml-1 p-1 rounded-lg active:bg-slate-800 text-slate-500 active:text-slate-300 transition"
          >
            <X className="h-3.5 sm:h-4 w-3.5 sm:w-4" />
          </button>
        </div>

        {/* Body - Mobile Responsive */}
        <div className="max-h-[50vh] sm:max-h-[60vh] overflow-y-auto" ref={listRef}>

          {/* Search results */}
          {query && (
            <>
              {results.length > 0 ? (
                <div>
                  <p className="px-3 sm:px-4 pt-3 pb-1 text-[9px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                    Search Results
                  </p>
                  {results.map((item, idx) => (
                    <StockRow
                      key={item.symbol}
                      item={item}
                      isActive={idx === activeIdx}
                      isCurrent={item.symbol === currentTicker}
                      query={query}
                      onSelect={() => handleSelect(item)}
                    />
                  ))}
                </div>
              ) : !loading && (
                <div className="flex flex-col items-center justify-center py-8 sm:py-12 gap-2 text-slate-500">
                  <Search className="h-6 sm:h-8 w-6 sm:w-8 opacity-30" />
                  <p className="text-xs sm:text-sm text-center px-4">No stocks found for &quot;{query}&quot;</p>
                  <p className="text-[10px] sm:text-xs opacity-60 text-center px-4">Try a company name like &quot;Infosys&quot; or symbol like &quot;INFY&quot;</p>
                </div>
              )}
            </>
          )}

          {/* Default: Recent + Popular */}
          {showPopular && (
            <>
              {/* Recent searches */}
              {recentSearches.length > 0 && (
                <div>
                  <p className="px-3 sm:px-4 pt-3 pb-1 text-[9px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                    Recent
                  </p>
                  {recentSearches.map((item) => (
                    <StockRow
                      key={item.symbol}
                      item={item}
                      isCurrent={item.symbol === currentTicker}
                      onSelect={() => handleSelect(item)}
                    />
                  ))}
                  <div className="border-t border-slate-800/70 my-1" />
                </div>
              )}

              {/* Popular picks */}
              <div>
                <p className="px-3 sm:px-4 pt-3 pb-1 text-[9px] sm:text-[10px] font-bold text-slate-500 uppercase tracking-widest flex items-center gap-1.5">
                  <TrendingUp className="h-2.5 sm:h-3 w-2.5 sm:w-3" /> Popular on NSE
                </p>
                {POPULAR_STOCKS.map((item, idx) => (
                  <StockRow
                    key={item.symbol}
                    item={item}
                    isActive={idx === activeIdx}
                    isCurrent={item.symbol === currentTicker}
                    onSelect={() => handleSelect(item)}
                  />
                ))}
              </div>
            </>
          )}
        </div>

        {/* Footer hint - Mobile Responsive */}
        <div className="border-t border-slate-800 px-3 sm:px-4 py-2 sm:py-2.5 flex items-center gap-2 sm:gap-4 text-[9px] sm:text-[10px] text-slate-600 overflow-x-auto">
          <span className="shrink-0"><kbd className="font-mono bg-slate-800 px-1 rounded text-[8px] sm:text-[9px]">↑↓</kbd> navigate</span>
          <span className="shrink-0"><kbd className="font-mono bg-slate-800 px-1 rounded text-[8px] sm:text-[9px]">↵</kbd> select</span>
          <span className="shrink-0 hidden sm:inline"><kbd className="font-mono bg-slate-800 px-1 rounded">Esc</kbd> close</span>
          <span className="ml-auto shrink-0">NSE &amp; BSE</span>
        </div>
      </div>
    </div>
  );
}

// Highlights matching text in the name/symbol
function Highlight({ text, query }) {
  if (!query || !text) return <span>{text}</span>;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return <span>{text}</span>;
  return (
    <span>
      {text.slice(0, idx)}
      <span className="text-white font-bold">{text.slice(idx, idx + query.length)}</span>
      {text.slice(idx + query.length)}
    </span>
  );
}

function StockRow({ item, isActive, isCurrent, query, onSelect }) {
  // Derive a 2-letter monogram from the symbol
  const mono = item.symbol.replace('.NS', '').replace('.BO', '').slice(0, 2);
  const exchange = item.symbol.endsWith('.BO') ? 'BSE' : 'NSE';

  return (
    <div
      onClick={onSelect}
      className={`flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2 sm:py-2.5 cursor-pointer transition group ${
        isActive
          ? 'bg-indigo-600/20'
          : 'active:bg-slate-800/60'
      } ${isCurrent ? 'opacity-60' : ''}`}
    >
      {/* Avatar - Mobile Responsive */}
      <div className="h-7 sm:h-8 w-7 sm:w-8 rounded-lg bg-gradient-to-br from-indigo-600/40 to-slate-700 flex items-center justify-center shrink-0 text-[10px] sm:text-[11px] font-black text-indigo-200 border border-indigo-500/20">
        {mono}
      </div>

      {/* Info - Mobile Responsive */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1 sm:gap-2 flex-wrap">
          <span className="text-xs sm:text-sm font-bold text-white font-mono tracking-wide">
            <Highlight text={item.symbol.replace('.NS', '').replace('.BO', '')} query={query} />
          </span>
          <span className={`text-[8px] sm:text-[9px] font-bold px-1 sm:px-1.5 py-0.5 rounded border ${exchange === 'NSE' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'}`}>
            {exchange}
          </span>
          {item.sector && (
            <span className={`text-[8px] sm:text-[9px] font-semibold px-1 sm:px-1.5 py-0.5 rounded border hidden sm:inline ${getSectorStyle(item.sector)}`}>
              {item.sector}
            </span>
          )}
          {isCurrent && (
            <span className="text-[8px] sm:text-[9px] text-indigo-400 font-semibold">Currently viewing</span>
          )}
        </div>
        <p className="text-[11px] sm:text-xs text-slate-400 truncate mt-0.5">
          <Highlight text={item.name} query={query} />
        </p>
      </div>

      <ChevronRight className="h-3.5 sm:h-4 w-3.5 sm:w-4 text-slate-700 group-hover:text-slate-400 transition shrink-0" />
    </div>
  );
}
