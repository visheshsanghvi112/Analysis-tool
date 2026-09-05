'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Search,
  TrendingUp,
  Brain,
  Shield,
  Menu,
  X,
  Zap,
  ChevronRight,
  Loader2,
  Briefcase,
  Star,
  Activity,
} from 'lucide-react';
import WatchlistDrawer from './WatchlistDrawer';
import StockSearchModal from './StockSearchModal';
import SideNavDrawer from './SideNavDrawer';

const Header = ({ onTickerSelect, currentTicker }) => {
  const [searchQuery, setSearchQuery]   = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching]   = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [watchlistOpen, setWatchlistOpen] = useState(false);
  const [watchlistCount, setWatchlistCount] = useState(0);
  const [spotlightOpen, setSpotlightOpen] = useState(false);
  const [sideNavOpen, setSideNavOpen]     = useState(false);

  // Sync watchlist count from localStorage
  useEffect(() => {
    const updateCount = () => {
      try {
        const saved = localStorage.getItem('stockiq_pro_watchlist');
        if (saved) {
          const list = JSON.parse(saved);
          setWatchlistCount(Array.isArray(list) ? list.length : 0);
        }
      } catch (_) {}
    };
    updateCount();
    window.addEventListener('stockiq-watchlist-changed', updateCount);
    return () => window.removeEventListener('stockiq-watchlist-changed', updateCount);
  }, []);

  const timeoutRef   = useRef(null);
  const abortRef     = useRef(null);
  const wrapperRef   = useRef(null);
  const inputRef     = useRef(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

  /* Close dropdown on outside click */
  useEffect(() => {
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Handle global search focus events & hotkeys
  useEffect(() => {
    const handleTriggerFocus = () => {
      setSpotlightOpen(true);
    };

    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setSpotlightOpen(prev => !prev);
      }
    };

    window.addEventListener('trigger-search-focus', handleTriggerFocus);
    window.addEventListener('open-stock-search', handleTriggerFocus);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('trigger-search-focus', handleTriggerFocus);
      window.removeEventListener('open-stock-search', handleTriggerFocus);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const fetchTickers = useCallback(async (query) => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setIsSearching(true);
    try {
      const res  = await fetch(
        `${API_BASE}/api/tickers?q=${encodeURIComponent(query)}`,
        { signal: abortRef.current.signal }
      );
      const data = await res.json();
      if (!abortRef.current?.signal.aborted) {
        setSearchResults(data.tickers || []);
        setDropdownOpen(true);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setSearchResults([]);
      }
    } finally {
      if (!abortRef.current?.signal.aborted) setIsSearching(false);
    }
  }, [API_BASE]);

  const handleSearch = (e) => {
    const q = e.target.value;
    setSearchQuery(q);

    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    if (q.length < 2) {
      setSearchResults([]);
      setDropdownOpen(false);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    timeoutRef.current = setTimeout(() => fetchTickers(q), 280);
  };

  const selectTicker = (ticker) => {
    if (onTickerSelect) {
      onTickerSelect(ticker.symbol);
    } else {
      window.location.assign(`/?ticker=${encodeURIComponent(ticker.symbol)}`);
    }
    setSearchQuery('');
    setSearchResults([]);
    setDropdownOpen(false);
    setMobileMenuOpen(false);
  };

  const popularStocks = [
    { symbol: 'HDFCBANK.NS',  name: 'HDFC Bank' },
    { symbol: 'RELIANCE.NS',  name: 'Reliance Industries' },
    { symbol: 'NIFTYBEES.NS',  name: 'NIFTY 50 ETF' },
    { symbol: 'GOLDBEES.NS',   name: 'Gold ETF' },
    { symbol: 'TCS.NS',       name: 'TCS' },
    { symbol: 'INFY.NS',      name: 'Infosys' },
    { symbol: 'MON100.NS',    name: 'Nasdaq 100 ETF' },
    { symbol: '^NSEI',        name: 'NIFTY 50' },
  ];

  return (
    <>
      {/* ── Top nav bar ─────────────────────────────────────────────── */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 100,
          background: 'rgba(0,0,0,0.92)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          borderBottom: '1px solid #222',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div style={{ display: 'flex', alignItems: 'center', height: '60px', gap: '20px' }}>

            {/* ── Left: Logo — click navigates home ─────────────────── */}
            <Link href="/" onClick={() => { window.dispatchEvent(new CustomEvent('reset-selected-ticker')); }} style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0, textDecoration: 'none' }}>
              <div style={{
                width: '32px', height: '32px',
                background: '#fff',
                borderRadius: '6px',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'opacity 0.15s',
              }}
                onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
                onMouseLeave={e => e.currentTarget.style.opacity = '1'}
              >
                <TrendingUp style={{ width: '18px', height: '18px', color: '#000' }} />
              </div>
              <span
                className="hidden sm:block"
                style={{ fontWeight: 700, fontSize: '15px', color: '#fff', letterSpacing: '-0.02em' }}
              >
                StockIQ Pro
              </span>
            </Link>

            {/* ── Divider ─────────────────────────────────────────────── */}
            <div className="hidden sm:block" style={{ width: '1px', height: '20px', background: '#333' }} />

            {/* ── Search (desktop) ────────────────────────────────────── */}
            <div
              ref={wrapperRef}
              className="hidden md:block"
              style={{ flex: 1, maxWidth: '420px', position: 'relative' }}
            >
              <div style={{ position: 'relative' }}>
                {/* Icon left */}
                <div style={{
                  position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)',
                  pointerEvents: 'none', display: 'flex', alignItems: 'center',
                }}>
                  {isSearching
                    ? <Loader2 style={{ width: '15px', height: '15px', color: '#666', animation: 'spin 0.8s linear infinite' }} />
                    : <Search style={{ width: '15px', height: '15px', color: '#666' }} />
                  }
                </div>

                <input
                  ref={inputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setSpotlightOpen(true);
                  }}
                  onClick={() => setSpotlightOpen(true)}
                  onFocus={() => setSpotlightOpen(true)}
                  placeholder="Search 7,950+ stocks, ETFs, indices (NSE/BSE)…"
                  className="v-input"
                  style={{
                    width: '100%',
                    paddingLeft: '38px',
                    paddingRight: '12px',
                    paddingTop: '8px',
                    paddingBottom: '8px',
                    fontSize: '13px',
                    cursor: 'pointer',
                  }}
                />

                {/* Keyboard hint */}
                {!searchQuery && (
                  <div style={{
                    position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)',
                    display: 'flex', gap: '3px',
                  }}>
                    <kbd style={{
                      fontSize: '10px', padding: '2px 5px', background: '#1a1a1a',
                      border: '1px solid #333', borderRadius: '4px', color: '#666',
                      fontFamily: 'inherit',
                    }}>⌘K</kbd>
                  </div>
                )}
              </div>

              {/* ── Dropdown ──────────────────────────────────────────── */}
              {dropdownOpen && (
                <div className="v-dropdown fade-up">
                  {isSearching ? (
                    <div style={{ padding: '24px', textAlign: 'center', color: '#666', fontSize: '13px' }}>
                      <Loader2 style={{ width: '20px', height: '20px', margin: '0 auto 8px', animation: 'spin 0.8s linear infinite' }} />
                      Searching…
                    </div>
                  ) : searchResults.length === 0 ? (
                    <div style={{ padding: '20px', textAlign: 'center', color: '#666', fontSize: '13px' }}>
                      No results found
                    </div>
                  ) : (
                    <div>
                      <div style={{ padding: '8px 12px 6px', borderBottom: '1px solid #1a1a1a' }}>
                        <span style={{ fontSize: '11px', color: '#555', fontWeight: 500, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                          Results — {searchResults.length} instruments
                        </span>
                      </div>
                      {searchResults.map((ticker, i) => (
                        <button
                          key={ticker.symbol}
                          onClick={() => selectTicker(ticker)}
                          style={{
                            width: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '10px 14px',
                            background: 'transparent',
                            border: 'none',
                            borderBottom: i < searchResults.length - 1 ? '1px solid #1a1a1a' : 'none',
                            cursor: 'pointer',
                            color: '#ededed',
                            textAlign: 'left',
                            transition: 'background 0.12s ease',
                          }}
                          onMouseEnter={e => e.currentTarget.style.background = '#1a1a1a'}
                          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                        >
                          <div>
                            <p style={{ fontSize: '13px', fontWeight: 600, color: '#fff', marginBottom: '2px' }}>
                              {ticker.symbol.replace('.NS', '').replace('.BO', '')}
                            </p>
                            <p style={{ fontSize: '12px', color: '#777', maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {ticker.name}
                            </p>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
                            {ticker.sector && (
                              <span style={{
                                fontSize: '11px', padding: '2px 7px',
                                background: ticker.sector === 'ETF' ? 'rgba(234,179,8,0.12)' : ticker.sector === 'Indices' ? 'rgba(56,189,248,0.12)' : ticker.sector === 'Global' ? 'rgba(168,85,247,0.12)' : 'rgba(0,112,243,0.1)',
                                border: ticker.sector === 'ETF' ? '1px solid rgba(234,179,8,0.3)' : ticker.sector === 'Indices' ? '1px solid rgba(56,189,248,0.3)' : ticker.sector === 'Global' ? '1px solid rgba(168,85,247,0.3)' : '1px solid rgba(0,112,243,0.2)',
                                borderRadius: '4px',
                                color: ticker.sector === 'ETF' ? '#facc15' : ticker.sector === 'Indices' ? '#38bdf8' : ticker.sector === 'Global' ? '#c084fc' : '#4fa3ff',
                                whiteSpace: 'nowrap',
                              }}>
                                {ticker.sector}
                              </span>
                            )}
                            <ChevronRight style={{ width: '14px', height: '14px', color: '#444' }} />
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Right: Intraday Desk + Watchlist + Live + Menu Button ─────────────── */}
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px' }}>
              {/* Intraday Desk Link */}
              <Link
                href="/intraday"
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '6px 11px', background: 'rgba(16, 185, 129, 0.08)',
                  border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: '7px',
                  color: '#34d399', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                  transition: 'all 0.15s', textDecoration: 'none',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'rgba(16, 185, 129, 0.16)'; e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.45)'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'rgba(16, 185, 129, 0.08)'; e.currentTarget.style.borderColor = 'rgba(16, 185, 129, 0.25)'; }}
                title="Open Intraday Quantitative Trading Desk"
              >
                <Activity style={{ width: '13px', height: '13px', color: '#34d399' }} />
                <span className="hidden md:inline">Intraday Desk</span>
              </Link>

              {/* Watchlist button */}
              <button
                onClick={() => setWatchlistOpen(true)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '6px 12px', background: 'rgba(234, 179, 8, 0.08)',
                  border: '1px solid rgba(234, 179, 8, 0.25)', borderRadius: '7px',
                  color: '#facc15', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'rgba(234, 179, 8, 0.15)'; e.currentTarget.style.borderColor = 'rgba(234, 179, 8, 0.4)'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'rgba(234, 179, 8, 0.08)'; e.currentTarget.style.borderColor = 'rgba(234, 179, 8, 0.25)'; }}
                title="Open Watchlist"
              >
                <Star style={{ width: '13px', height: '13px', fill: '#facc15', color: '#facc15' }} />
                <span className="hidden sm:inline">Watchlist</span>
                {watchlistCount > 0 && (
                  <span style={{
                    fontSize: '10px', background: '#eab308', color: '#000',
                    padding: '0 6px', borderRadius: '999px', fontWeight: 800, lineHeight: '16px'
                  }}>{watchlistCount}</span>
                )}
              </button>

              {/* Live status badge */}
              <span className="hidden sm:inline-flex v-badge v-badge-green">
                <span className="live-dot" style={{ marginRight: '2px' }} />
                Live
              </span>

              <div className="hidden sm:block" style={{ width: '1px', height: '18px', background: '#222' }} />

              {/* ── Menu Bar Icon on the RIGHT SIDE ───────────────────── */}
              <button
                onClick={() => setSideNavOpen(true)}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  padding: '6px 12px', background: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.14)', borderRadius: '7px',
                  color: '#fff', fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.12)'; e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.28)'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)'; e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.14)'; }}
                title="Open Navigation Menu"
                aria-label="Open Navigation Menu"
              >
                <Menu style={{ width: '16px', height: '16px', color: '#fff' }} />
                <span className="hidden sm:inline">Menu</span>
              </button>
            </div>
          </div>

          {/* ── Analysing ticker strip ─────────────────────────────────── */}
          {currentTicker && (
            <div style={{
              borderTop: '1px solid #1a1a1a',
              padding: '8px 0',
              display: 'flex', alignItems: 'center', gap: '8px',
            }}>
              <span style={{ fontSize: '12px', color: '#666' }}>Analysing</span>
              <span style={{
                fontSize: '12px', fontWeight: 600, color: '#ededed',
                background: '#1a1a1a', border: '1px solid #333',
                borderRadius: '5px', padding: '2px 10px',
                letterSpacing: '0.04em',
              }}>
                {currentTicker.replace('.NS', '').replace('.BO', '')}
              </span>
              <span className="v-badge v-badge-green">
                <span className="live-dot" />
                Live data
              </span>
            </div>
          )}
        </div>
      </header>

      {/* ── Watchlist Drawer ────────────────────────────────────────── */}
      <WatchlistDrawer
        isOpen={watchlistOpen}
        onClose={() => setWatchlistOpen(false)}
        onSelectTicker={(sym) => {
          if (onTickerSelect) onTickerSelect(sym);
        }}
        currentTicker={currentTicker}
      />

      {/* ── Side Navigation Drawer (Left) ─────────────────────────── */}
      <SideNavDrawer
        isOpen={sideNavOpen}
        onClose={() => setSideNavOpen(false)}
        onOpenSearch={() => setSpotlightOpen(true)}
      />

      {/* ── Spotlight Search Command Palette (⌘K) ─────────────────── */}
      <StockSearchModal
        isOpen={spotlightOpen}
        onClose={() => setSpotlightOpen(false)}
        onSelect={(sym) => {
          if (onTickerSelect) onTickerSelect(sym);
        }}
        currentTicker={currentTicker}
      />

      {/* Global CSS for spin animation */}
      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  );
};

export default Header;