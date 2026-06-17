'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
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
} from 'lucide-react';

const Header = ({ onTickerSelect, currentTicker }) => {
  const [searchQuery, setSearchQuery]   = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching]   = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const timeoutRef   = useRef(null);
  const abortRef     = useRef(null);
  const wrapperRef   = useRef(null);
  const inputRef     = useRef(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

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
    onTickerSelect(ticker.symbol);
    setSearchQuery('');
    setSearchResults([]);
    setDropdownOpen(false);
    setMobileMenuOpen(false);
  };

  const popularStocks = [
    { symbol: 'HDFCBANK.NS',  name: 'HDFC Bank' },
    { symbol: 'RELIANCE.NS',  name: 'Reliance Industries' },
    { symbol: 'TCS.NS',       name: 'TCS' },
    { symbol: 'INFY.NS',      name: 'Infosys' },
    { symbol: 'WIPRO.NS',     name: 'Wipro' },
    { symbol: 'ICICIBANK.NS', name: 'ICICI Bank' },
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
          <div style={{ display: 'flex', alignItems: 'center', height: '60px', gap: '24px' }}>

            {/* ── Logo — click navigates home ───────────────────────── */}
            <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0, textDecoration: 'none' }}>
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
                  onChange={handleSearch}
                  onFocus={() => searchResults.length > 0 && setDropdownOpen(true)}
                  placeholder="Search NSE / BSE stocks…"
                  className="v-input"
                  style={{
                    width: '100%',
                    paddingLeft: '38px',
                    paddingRight: '12px',
                    paddingTop: '8px',
                    paddingBottom: '8px',
                    fontSize: '13px',
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
                          Results — {searchResults.length} stocks
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
                                background: 'rgba(0,112,243,0.1)', border: '1px solid rgba(0,112,243,0.2)',
                                borderRadius: '4px', color: '#4fa3ff', whiteSpace: 'nowrap',
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

            {/* ── Right nav + badges ──────────────────────────────────── */}
            <div className="hidden md:flex" style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '16px' }}>
              <Link
                href="/browse"
                style={{
                  fontSize: '13px', fontWeight: 500, color: '#888',
                  textDecoration: 'none', transition: 'color 0.15s',
                  whiteSpace: 'nowrap',
                }}
                onMouseEnter={e => e.currentTarget.style.color = '#fff'}
                onMouseLeave={e => e.currentTarget.style.color = '#888'}
              >
                Browse Stocks
              </Link>
              <Link
                href="/portfolio"
                style={{
                  fontSize: '13px', fontWeight: 500, color: '#888',
                  textDecoration: 'none', transition: 'color 0.15s',
                  whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: '5px',
                }}
                onMouseEnter={e => e.currentTarget.style.color = '#fff'}
                onMouseLeave={e => e.currentTarget.style.color = '#888'}
              >
                <Briefcase style={{ width: '13px', height: '13px' }} />
                Portfolio
              </Link>
              <div style={{ width: '1px', height: '16px', background: '#2a2a2a' }} />
              <span className="v-badge v-badge-green">
                <span className="live-dot" style={{ marginRight: '2px' }} />
                Live
              </span>
              <span className="v-badge v-badge-blue">
                <Brain style={{ width: '11px', height: '11px' }} />
                AI
              </span>
            </div>

            {/* ── Mobile: hamburger ───────────────────────────────────── */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden"
              style={{
                marginLeft: 'auto', padding: '6px', background: 'transparent',
                border: '1px solid #333', borderRadius: '6px', color: '#aaa', cursor: 'pointer',
              }}
            >
              {mobileMenuOpen ? <X style={{ width: '18px', height: '18px' }} /> : <Menu style={{ width: '18px', height: '18px' }} />}
            </button>
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

        {/* ── Mobile menu ─────────────────────────────────────────────── */}
        {mobileMenuOpen && (
          <div style={{
            borderTop: '1px solid #222',
            background: '#0a0a0a',
            padding: '16px',
          }}>
            {/* Mobile search */}
            <div style={{ position: 'relative', marginBottom: '16px' }}>
              <Search style={{
                position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)',
                width: '15px', height: '15px', color: '#666', pointerEvents: 'none',
              }} />
              <input
                type="text"
                value={searchQuery}
                onChange={handleSearch}
                placeholder="Search stocks…"
                className="v-input"
                style={{ width: '100%', paddingLeft: '38px', paddingRight: '12px', paddingTop: '10px', paddingBottom: '10px' }}
              />
            </div>

            {/* Mobile search results */}
            {searchResults.length > 0 && (
              <div style={{
                background: '#111', border: '1px solid #222', borderRadius: '8px',
                overflow: 'hidden', marginBottom: '16px',
              }}>
                {searchResults.slice(0, 8).map((ticker, i) => (
                  <button
                    key={ticker.symbol}
                    onClick={() => selectTicker(ticker)}
                    style={{
                      width: '100%', display: 'flex', justifyContent: 'space-between',
                      alignItems: 'center', padding: '12px 14px', background: 'transparent',
                      border: 'none', borderBottom: i < Math.min(searchResults.length, 8) - 1 ? '1px solid #1a1a1a' : 'none',
                      cursor: 'pointer', color: '#ededed', textAlign: 'left',
                    }}
                  >
                    <div>
                      <p style={{ fontSize: '13px', fontWeight: 600 }}>
                        {ticker.symbol.replace('.NS', '').replace('.BO', '')}
                      </p>
                      <p style={{ fontSize: '12px', color: '#777', marginTop: '2px' }}>{ticker.name}</p>
                    </div>
                    {ticker.sector && (
                      <span style={{
                        fontSize: '11px', padding: '2px 7px', background: 'rgba(0,112,243,0.1)',
                        border: '1px solid rgba(0,112,243,0.2)', borderRadius: '4px', color: '#4fa3ff',
                      }}>
                        {ticker.sector}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}

            {/* Browse link + popular chips */}
            <Link
              href="/browse"
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: '100%', padding: '12px',
                background: '#111', border: '1px solid #222', borderRadius: '8px',
                color: '#fff', fontSize: '13px', fontWeight: 500,
                textDecoration: 'none', marginBottom: '16px',
              }}
              onClick={() => setMobileMenuOpen(false)}
            >
              Browse all sectors →
            </Link>
            <Link
              href="/portfolio"
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                width: '100%', padding: '12px',
                background: '#0d0d18', border: '1px solid #2a2a40', borderRadius: '8px',
                color: '#818cf8', fontSize: '13px', fontWeight: 500,
                textDecoration: 'none', marginBottom: '16px',
              }}
              onClick={() => setMobileMenuOpen(false)}
            >
              <Briefcase style={{ width: '14px', height: '14px' }} />
              Portfolio Tracker
            </Link>
            <p style={{ fontSize: '11px', color: '#555', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Quick access
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {popularStocks.map(s => (
                <button
                  key={s.symbol}
                  onClick={() => selectTicker(s)}
                  style={{
                    fontSize: '13px', fontWeight: 500, padding: '8px 14px',
                    background: '#111', border: '1px solid #222', borderRadius: '8px',
                    color: '#fff', cursor: 'pointer',
                  }}
                >
                  {s.symbol.replace('.NS', '')}
                </button>
              ))}
            </div>
          </div>
        )}
      </header>

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