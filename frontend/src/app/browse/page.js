'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeft, Search, X, TrendingUp, TrendingDown,
  ChevronRight, Sparkles, Filter, RefreshCw, Flame, Award, ShieldAlert
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

const SECTOR_META = {
  'Banking':       { emoji: '🏦', color: '#00e699' }, // vibrant green
  'IT':            { emoji: '💻', color: '#3b82f6' }, // bright blue
  'Energy':        { emoji: '⚡', color: '#f59e0b' }, // amber
  'Pharma':        { emoji: '💊', color: '#a78bfa' }, // purple
  'Auto':          { emoji: '🚗', color: '#ff4d4d' }, // red
  'FMCG':          { emoji: '🛒', color: '#10b981' }, // green
  'Finance':       { emoji: '📈', color: '#06b6d4' }, // cyan
  'Infra':         { emoji: '🏗️', color: '#f97316' }, // orange
  'Metals':        { emoji: '🏭', color: '#94a3b8' }, // steel
  'Telecom':       { emoji: '📡', color: '#22d3ee' }, // sky blue
  'Consumer Tech': { emoji: '📱', color: '#f472b6' }, // pink
  'Others':        { emoji: '📊', color: '#888888' }, // gray
  'All':           { emoji: '🌍', color: '#ffffff' }, // white
};

const TRENDING_SEARCHES = ['HDFC', 'Reliance', 'TCS', 'Infosys', 'Zomato', 'Tata Motors'];

function getMeta(sector) {
  return SECTOR_META[sector] || { emoji: '📊', color: '#888888' };
}

// Escape regex characters
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Smart text highlighter component
function HighlightedText({ text, query }) {
  if (!query.trim()) return <span>{text}</span>;
  
  const words = query.split(/\s+/).filter(Boolean);
  const regexStr = words.map(w => escapeRegExp(w)).join('|');
  if (!regexStr) return <span>{text}</span>;

  const regex = new RegExp(`(${regexStr})`, 'gi');
  const parts = text.split(regex);

  return (
    <span>
      {parts.map((part, i) => 
        regex.test(part) ? (
          <mark key={i} style={{ background: '#3b82f640', color: '#ffffff', borderRadius: '2px', padding: '0 2px', fontWeight: 700 }}>
            {part}
          </mark>
        ) : (
          part
        )
      )}
    </span>
  );
}

function StockCard({ stock, color, query, onSelect }) {
  const sym = stock.symbol.replace('.NS', '').replace('.BO', '');
  const [hov, setHov] = useState(false);

  return (
    <button
      onClick={() => onSelect(stock.symbol)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        flexDirection: 'column',
        padding: '16px',
        width: '100%',
        minHeight: '110px',
        background: hov ? '#1c1c1c' : '#121212',
        border: `1px solid ${hov ? '#ffffff' : '#282828'}`,
        borderRadius: '8px',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.12s ease',
        position: 'relative',
        overflow: 'hidden',
        WebkitTapHighlightColor: 'transparent',
      }}
    >
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: hov ? color : 'transparent', transition: 'background 0.12s' }} />
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', width: '100%', marginBottom: '10px' }}>
        <div style={{
          width: '36px',
          height: '36px',
          borderRadius: '6px',
          background: color + '15',
          border: `1px solid ${color}35`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '11px',
          fontWeight: 800,
          color: color,
        }}>
          {sym.slice(0, 2)}
        </div>
        {stock.sector && (
          <span style={{
            fontSize: '9px',
            color: color,
            background: color + '10',
            border: `1px solid ${color}20`,
            padding: '2px 6px',
            borderRadius: '4px',
            fontWeight: 600
          }}>
            {stock.sector}
          </span>
        )}
      </div>

      <p style={{
        fontSize: '13px',
        fontWeight: 700,
        color: '#ffffff',
        letterSpacing: '0.01em',
        marginBottom: '4px'
      }}>
        <HighlightedText text={sym} query={query} />
      </p>

      <p style={{
        fontSize: '11px',
        color: '#aaaaaa',
        overflow: 'hidden',
        display: '-webkit-box',
        WebkitLineClamp: 2,
        WebkitBoxOrient: 'vertical',
        lineHeight: 1.4
      }}>
        <HighlightedText text={stock.name} query={query} />
      </p>

      <div style={{
        position: 'absolute',
        bottom: '12px',
        right: '12px',
        color: hov ? '#ffffff' : '#555555',
        transition: 'color 0.12s'
      }}>
        <ChevronRight style={{ width: '14px', height: '14px' }} />
      </div>
    </button>
  );
}

function ScreenerStockCard({ stock, type, onSelect }) {
  const sym = stock.ticker.replace('.NS', '').replace('.BO', '');
  const changePct = stock.changePct || 0.0;
  const isPos = changePct >= 0;
  const color = isPos ? '#10b981' : '#ef4444';
  const [hov, setHov] = useState(false);

  const formatVol = (vol) => {
    if (!vol) return 'N/A';
    if (vol >= 10000000) return (vol / 10000000).toFixed(2) + ' Cr';
    if (vol >= 100000) return (vol / 100000).toFixed(2) + ' L';
    return vol.toLocaleString('en-IN');
  };

  return (
    <button
      onClick={() => onSelect(stock.ticker)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '14px 16px',
        width: '100%',
        background: hov ? '#16161a' : '#0c0c0e',
        border: `1px solid ${hov ? 'rgba(255,255,255,0.12)' : '#1f1f23'}`,
        borderRadius: '10px',
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.12s ease',
        WebkitTapHighlightColor: 'transparent',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, minWidth: 0 }}>
        <div style={{
          width: '38px',
          height: '38px',
          borderRadius: '8px',
          background: isPos ? 'rgba(16,185,129,0.06)' : 'rgba(239,68,68,0.06)',
          border: `1px solid ${isPos ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '11px',
          fontWeight: 800,
          color: color,
          flexShrink: 0
        }}>
          {sym.slice(0, 2)}
        </div>
        <div style={{ minWidth: 0 }}>
          <span style={{ fontSize: '13px', fontWeight: 700, color: '#ffffff', display: 'block', lineHeight: 1.2 }}>{sym}</span>
          <span style={{ fontSize: '10px', color: '#888888', display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '160px', lineHeight: 1.2, marginTop: '2px' }}>
            {stock.longName || sym}
          </span>
        </div>
      </div>

      <div style={{ textAlign: 'right', flexShrink: 0 }}>
        <p style={{ fontSize: '13px', fontWeight: 700, color: '#ffffff', marginBottom: '2px' }}>
          ₹{stock.price?.toLocaleString('en-IN') || 'N/A'}
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', justifyContent: 'flex-end' }}>
          {type === 'high52w' && (
            <span style={{ fontSize: '9px', color: '#888888' }}>
              High: ₹{stock.fiftyTwoWeekHigh?.toLocaleString('en-IN')} ({stock.pct_from_52w_high}%)
            </span>
          )}
          {type === 'low52w' && (
            <span style={{ fontSize: '9px', color: '#888888' }}>
              Low: ₹{stock.fiftyTwoWeekLow?.toLocaleString('en-IN')} (+{stock.pct_from_52w_low}%)
            </span>
          )}
          {type === 'volume' && (
            <span style={{ fontSize: '9px', color: '#888888' }}>
              Vol: {formatVol(stock.volume)}
            </span>
          )}
          <span style={{
            fontSize: '10px',
            fontWeight: 800,
            color: color,
            background: isPos ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
            padding: '2px 6px',
            borderRadius: '4px',
          }}>
            {isPos ? '+' : ''}{changePct.toFixed(2)}%
          </span>
        </div>
      </div>
    </button>
  );
}

export default function BrowsePage() {
  const router = useRouter();
  
  const [allTickers, setAllTickers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeSector, setActiveSector] = useState('All');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 48;

  // Screener states
  const [activeTab, setActiveTab] = useState('gainers'); // 'gainers', 'losers', 'volume', 'high52w', 'low52w'
  const [screenerData, setScreenerData] = useState(null);
  const [screenerLoading, setScreenerLoading] = useState(false);
  const [screenerError, setScreenerError] = useState(null);

  // Search query
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const searchContainerRef = useRef(null);

  // Close search suggestions on click outside
  useEffect(() => {
    function handleClickOutside(event) {
      if (searchContainerRef.current && !searchContainerRef.current.contains(event.target)) {
        setIsFocused(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch full static tickers database
  useEffect(() => {
    fetch('/tickers.json')
      .then((res) => res.json())
      .then((data) => {
        setAllTickers(data || []);
      })
      .catch((err) => {
        console.error("Failed to load tickers database:", err);
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  // Fetch live market screener hotlists
  const fetchScreenerData = useCallback(async () => {
    setScreenerLoading(true);
    setScreenerError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/market-screener`);
      if (!res.ok) throw new Error('Failed to load market dashboard data');
      const data = await res.json();
      setScreenerData(data);
    } catch (err) {
      setScreenerError(err.message);
    } finally {
      setScreenerLoading(false);
    }
  }, []);

  // Fetch screener data on mount or tab change
  useEffect(() => {
    if (!screenerData && !screenerLoading) {
      const t = setTimeout(() => {
        fetchScreenerData();
      }, 0);
      return () => clearTimeout(t);
    }
  }, [screenerData, screenerLoading, fetchScreenerData]);

  const handleSelect = useCallback((symbol) => {
    router.push(`/?ticker=${encodeURIComponent(symbol)}`);
  }, [router]);

  // Group tickers by sector dynamically
  const sectorGroups = useMemo(() => {
    const groups = {};
    allTickers.forEach((t) => {
      const sec = t.sector || 'Others';
      if (!groups[sec]) groups[sec] = [];
      groups[sec].push(t);
    });
    return groups;
  }, [allTickers]);

  // Compute sector list with counts
  const sectorList = useMemo(() => {
    const list = Object.keys(sectorGroups).map((sec) => ({
      name: sec,
      count: sectorGroups[sec].length,
    }));
    
    list.sort((a, b) => b.count - a.count);
    const othersIdx = list.findIndex(x => x.name === 'Others');
    if (othersIdx > -1) {
      const [others] = list.splice(othersIdx, 1);
      list.push(others);
    }
    
    return [
      { name: 'All', count: allTickers.length },
      ...list
    ];
  }, [sectorGroups, allTickers]);

  // Smart Fuzzy Matcher
  const filteredTickers = useMemo(() => {
    let list = allTickers;
    
    if (activeSector !== 'All') {
      list = sectorGroups[activeSector] || [];
    }

    if (query.trim()) {
      const words = query.toLowerCase().split(/\s+/).filter(Boolean);
      list = list.filter((t) => {
        const sym = t.symbol.toLowerCase();
        const name = t.name.toLowerCase();
        return words.every(word => sym.includes(word) || name.includes(word));
      });
    }

    return list;
  }, [allTickers, activeSector, sectorGroups, query]);

  // Suggest sector jump if query matches a sector name
  const suggestedSector = useMemo(() => {
    if (!query.trim()) return null;
    const q = query.toLowerCase().trim();
    
    for (const sec of sectorList) {
      if (sec.name === 'All' || sec.name === activeSector) continue;
      if (sec.name.toLowerCase().includes(q) || q.includes(sec.name.toLowerCase())) {
        return sec.name;
      }
    }
    return null;
  }, [query, sectorList, activeSector]);

  const pagedStocks = useMemo(() => {
    return filteredTickers.slice(0, page * PAGE_SIZE);
  }, [filteredTickers, page]);

  const hasMore = pagedStocks.length < filteredTickers.length;

  const activeMeta = getMeta(activeSector);

  // Tab Details
  const tabTitle = useMemo(() => {
    switch (activeTab) {
      case 'gainers': return 'Top gainers today';
      case 'losers': return 'Top losers today';
      case 'volume': return 'Volume shockers today';
      case 'high52w': return '52-Week High shockers';
      case 'low52w': return '52-Week Low shockers';
      default: return 'Top gainers today';
    }
  }, [activeTab]);

  return (
    <div style={{
      minHeight: '100vh',
      background: '#000000',
      color: '#ffffff',
      fontFamily: 'var(--font-poppins), var(--font-inter), sans-serif'
    }}>
      <style>{`
        * { box-sizing: border-box; }
        .stg {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 10px;
        }
        @media(min-width: 480px) {
          .stg { grid-template-columns: repeat(3, 1fr); }
        }
        @media(min-width: 768px) {
          .stg { grid-template-columns: repeat(4, 1fr); gap: 12px; }
        }
        @media(min-width: 1024px) {
          .stg { grid-template-columns: repeat(6, 1fr); }
        }
        @media(min-width: 1280px) {
          .stg { grid-template-columns: repeat(8, 1fr); }
        }

        .screener-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: 12px;
        }
        @media(min-width: 640px) {
          .screener-grid { grid-template-columns: repeat(2, 1fr); }
        }
        @media(min-width: 1024px) {
          .screener-grid { grid-template-columns: repeat(3, 1fr); }
        }
        @media(min-width: 1280px) {
          .screener-grid { grid-template-columns: repeat(4, 1fr); }
        }

        .sg {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }
        @media(min-width: 480px) {
          .sg { grid-template-columns: repeat(4, 1fr); }
        }
        @media(min-width: 768px) {
          .sg { grid-template-columns: repeat(6, 1fr); }
        }
        @media(min-width: 1024px) {
          .sg { grid-template-columns: repeat(12, 1fr); }
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .fi {
          animation: fadeIn 0.2s ease forwards;
        }
      `}</style>

      {/* Header */}
      <header style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: 'rgba(0,0,0,0.96)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid #282828'
      }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 16px', display: 'flex', alignItems: 'center', height: '56px', gap: '12px' }}>
          <Link href="/" style={{ display: 'flex', alignItems: 'center', gap: '4px', color: '#888888', textDecoration: 'none', fontSize: '13px', flexShrink: 0 }}>
            <ArrowLeft style={{ width: '18px', height: '18px', color: '#ffffff' }} />
          </Link>
          <div style={{ width: '1px', height: '18px', background: '#282828', flexShrink: 0 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            <div style={{ width: '28px', height: '28px', background: '#ffffff', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <TrendingUp style={{ width: '14px', height: '14px', color: '#000000' }} />
            </div>
            <span style={{ fontWeight: 700, fontSize: '14px', color: '#ffffff', letterSpacing: '-0.02em' }}>Browse Stocks</span>
          </div>

          {/* Smart Search Bar */}
          <div ref={searchContainerRef} style={{ flex: 1, position: 'relative', maxWidth: '480px', marginLeft: 'auto' }}>
            <div style={{ position: 'relative' }}>
              <Search style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', width: '14px', height: '14px', color: '#aaaaaa' }} />
              <input
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setPage(1);
                }}
                onFocus={() => setIsFocused(true)}
                onClick={() => setIsFocused(true)}
                placeholder="Smart Search (e.g. HDFC Bank)..."
                style={{
                  width: '100%',
                  background: '#121212',
                  border: '1px solid #3a3a3a',
                  borderRadius: '8px',
                  padding: '8px 36px',
                  fontSize: '13px',
                  color: '#ffffff',
                  outline: 'none'
                }}
              />
              {query && (
                <button onClick={() => { setQuery(''); setPage(1); }} style={{ position: 'absolute', right: '10px', top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: '#ffffff' }}>
                  <X style={{ width: '14px', height: '14px' }} />
                </button>
              )}
            </div>

            {/* Smart Search Dropdown (Zero State / Trending Searches) */}
            {isFocused && !query && (
              <div style={{
                position: 'absolute',
                top: 'calc(100% + 8px)',
                left: 0,
                right: 0,
                background: '#121212',
                border: '1px solid #3a3a3a',
                borderRadius: '8px',
                padding: '12px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.8)',
                zIndex: 100,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: '#888888', fontSize: '11px', fontWeight: 600 }}>
                  <Sparkles style={{ width: '12px', height: '12px', color: '#f59e0b' }} />
                  <span>TRENDING SEARCHES</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {TRENDING_SEARCHES.map((item) => (
                    <button
                      key={item}
                      onClick={() => {
                        setQuery(item);
                        setIsFocused(false);
                      }}
                      style={{
                        background: '#1c1c1c',
                        border: '1px solid #282828',
                        borderRadius: '6px',
                        padding: '4px 10px',
                        fontSize: '11px',
                        color: '#ffffff',
                        cursor: 'pointer',
                        transition: 'background 0.12s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = '#282828'}
                      onMouseLeave={(e) => e.currentTarget.style.background = '#1c1c1c'}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main style={{ maxWidth: '1400px', margin: '0 auto', padding: '24px 16px 80px' }}>
        
        {/* ================= SECTION 1: LIVE MARKET SCREENER ================= */}
        <section style={{ marginBottom: '48px', paddingBottom: '32px', borderBottom: '1px solid #1f1f23' }}>
          <div style={{ marginBottom: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, tracking: '0.05em', color: '#3b82f6', background: 'rgba(59,130,246,0.1)', padding: '2px 8px', borderRadius: '4px' }}>LIVE</span>
              <h2 style={{ fontSize: 'clamp(20px, 4vw, 26px)', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em' }}>
                {tabTitle}
              </h2>
            </div>
            <p style={{ fontSize: '12px', color: '#aaaaaa' }}>
              Real-time calculations for Nifty 50 component stocks. Select a filter to view top movers.
            </p>
          </div>

          {/* Dashboard Navigation Filter Tabs (Matches User Image Layout) */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
            borderBottom: '1px solid #1f1f23',
            paddingBottom: '16px',
            marginBottom: '24px',
            flexWrap: 'wrap'
          }}>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
              {[
                { id: 'gainers', label: 'Top gainers' },
                { id: 'losers', label: 'Top losers' },
                { id: 'volume', label: 'Volume shockers' },
                { id: 'volume-top', label: 'Top by volume' },
                { id: 'high52w', label: '52W high' },
                { id: 'low52w', label: '52W low' },
              ].map((t) => {
                const isA = activeTab === t.id || (t.id === 'volume-top' && activeTab === 'volume');
                return (
                  <button
                    key={t.id}
                    onClick={() => {
                      if (t.id === 'volume-top') {
                        setActiveTab('volume');
                      } else {
                        setActiveTab(t.id);
                      }
                    }}
                    style={{
                      padding: '6px 16px',
                      background: isA ? '#f1f5f9' : 'rgba(255,255,255,0.02)',
                      color: isA ? '#0f172a' : '#cbd5e1',
                      border: `1px solid ${isA ? '#f1f5f9' : '#2d3748'}`,
                      borderRadius: '9999px',
                      fontSize: '12px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.12s ease',
                    }}
                  >
                    {t.label}
                  </button>
                );
              })}
            </div>

            {/* Index Selector Dropdown */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 16px',
              background: 'transparent',
              border: '1px solid #2d3748',
              borderRadius: '9999px',
              color: '#cbd5e1',
              fontSize: '12px',
              fontWeight: 600,
              cursor: 'pointer'
            }}>
              <span>Nifty Total Market</span>
              <span style={{ fontSize: '8px', opacity: 0.6 }}>▼</span>
            </div>
          </div>

          {/* Screener Display Content */}
          {screenerLoading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '40px 0', gap: '12px', color: '#888888' }}>
              <RefreshCw style={{ width: '20px', height: '20px', animation: 'spin 1s linear infinite', color: '#ffffff' }} />
              <p style={{ fontSize: '12px' }}>Calculating market statistics...</p>
            </div>
          ) : screenerError ? (
            <div style={{
              background: 'rgba(239,68,68,0.04)',
              border: '1px solid rgba(239,68,68,0.12)',
              padding: '16px',
              borderRadius: '10px',
              textAlign: 'center',
              maxWidth: '400px',
              margin: '10px auto'
            }}>
              <ShieldAlert style={{ width: '28px', height: '28px', color: '#ef4444', margin: '0 auto 8px' }} />
              <p style={{ fontSize: '13px', fontWeight: 700, color: '#ffffff', marginBottom: '2px' }}>Market Data Unavailable</p>
              <p style={{ fontSize: '11px', color: '#aaaaaa', marginBottom: '12px' }}>{screenerError}</p>
              <button
                onClick={fetchScreenerData}
                style={{
                  padding: '6px 14px',
                  background: '#ffffff',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#000000',
                  fontSize: '11px',
                  fontWeight: 700,
                  cursor: 'pointer'
                }}
              >
                Retry
              </button>
            </div>
          ) : screenerData ? (
            <div className="screener-grid fi" key={activeTab}>
              {(screenerData[activeTab] || []).map((stock) => (
                <ScreenerStockCard
                  key={stock.ticker}
                  stock={stock}
                  type={activeTab}
                  onSelect={handleSelect}
                />
              ))}
            </div>
          ) : null}
        </section>

        {/* ================= SECTION 2: DISCOVER CATEGORIES (SECTORS) ================= */}
        <section>
          <div style={{ marginBottom: '20px' }}>
            <h2 style={{ fontSize: 'clamp(20px, 4vw, 26px)', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.02em', marginBottom: '4px' }}>
              Discover Categories
            </h2>
            <p style={{ fontSize: '12px', color: '#aaaaaa' }}>
              Browse all {allTickers.length.toLocaleString()} NSE listed stocks categorized dynamically by business sectors.
            </p>
          </div>

          {/* Sectors Grid */}
          <div className="sg" style={{ marginBottom: '28px' }}>
            {sectorList.map((s) => {
              const isA = s.name === activeSector;
              const meta = getMeta(s.name);
              return (
                <button
                  key={s.name}
                  onClick={() => { setActiveSector(s.name); setPage(1); }}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '12px 6px',
                    background: isA ? meta.color + '20' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${isA ? '#ffffff' : '#282828'}`,
                    borderRadius: '8px',
                    cursor: 'pointer',
                    transition: 'all 0.12s',
                    WebkitTapHighlightColor: 'transparent'
                  }}
                >
                  <span style={{ fontSize: '20px' }}>{meta.emoji}</span>
                  <span style={{
                    fontSize: '10px',
                    fontWeight: isA ? 700 : 400,
                    color: isA ? '#ffffff' : '#aaaaaa',
                    textAlign: 'center',
                    lineHeight: 1.2
                  }}>{s.name}</span>
                  <span style={{ fontSize: '10px', color: '#666666' }}>{s.count}</span>
                </button>
              );
            })}
          </div>

          {/* Smart Sector Jump Suggestion Banner */}
          {suggestedSector && (
            <div style={{
              background: '#1d2433',
              border: '1px solid #3b82f650',
              borderRadius: '8px',
              padding: '12px 16px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              animation: 'fadeIn 0.2s ease forwards'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Filter style={{ width: '16px', height: '16px', color: '#3b82f6' }} />
                <span style={{ fontSize: '12px', color: '#ffffff' }}>
                  Looking for <strong>{suggestedSector}</strong> stocks?
                </span>
              </div>
              <button
                onClick={() => { setActiveSector(suggestedSector); setQuery(''); }}
                style={{
                  background: '#3b82f6',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#ffffff',
                  fontSize: '11px',
                  fontWeight: 700,
                  padding: '6px 12px',
                  cursor: 'pointer'
                }}
              >
                Switch to Sector
              </button>
            </div>
          )}

          {/* Active Sector Header */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '20px',
            paddingBottom: '16px',
            borderBottom: '1px solid #282828'
          }}>
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '8px',
              background: activeMeta.color + '20',
              border: `1px solid ${activeMeta.color}40`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px'
            }}>
              {activeMeta.emoji}
            </div>
            <div>
              <h2 style={{ fontSize: '17px', fontWeight: 700, color: '#ffffff' }}>{activeSector}</h2>
              <p style={{ fontSize: '12px', color: '#aaaaaa' }}>
                {filteredTickers.length} {filteredTickers.length === 1 ? 'stock' : 'stocks'} matches
              </p>
            </div>
          </div>

          {/* Stocks Grid */}
          {filteredTickers.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '60px 16px' }}>
              <p style={{ fontSize: '32px', marginBottom: '8px' }}>🔍</p>
              <p style={{ color: '#ffffff', fontWeight: 600, marginBottom: '4px' }}>No matches found</p>
              <p style={{ color: '#888888', fontSize: '13px' }}>Try resetting filters or searching another keyword.</p>
            </div>
          ) : (
            <>
              <div className="stg fi" key={`${activeSector}-${query}`}>
                {pagedStocks.map((stock) => (
                  <StockCard
                    key={stock.symbol}
                    stock={stock}
                    color={getMeta(stock.sector).color}
                    query={query}
                    onSelect={handleSelect}
                  />
                ))}
              </div>

              {/* Load More Pagination */}
              {hasMore && (
                <div style={{ textAlign: 'center', marginTop: '30px' }}>
                  <button
                    onClick={() => setPage(p => p + 1)}
                    style={{
                      padding: '12px 32px',
                      background: '#121212',
                      border: '1px solid #3a3a3a',
                      borderRadius: '8px',
                      color: '#ffffff',
                      fontSize: '13px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s'
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#ffffff'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#3a3a3a'; }}
                  >
                    Load More ({filteredTickers.length - pagedStocks.length} remaining)
                  </button>
                </div>
              )}
            </>
          )}
        </section>
      </main>

      <footer style={{ borderTop: '1px solid #1c1c1c', padding: '24px 16px', textAlign: 'center', color: '#666666', fontSize: '12px' }}>
        <p>
          © StockIQ Pro · Data via NSE India · For educational purposes only.{' '}
          <Link href="/terms" style={{ color: '#888888', textDecoration: 'underline', marginLeft: '6px' }}>
            Terms &amp; Conditions
          </Link>
        </p>
      </footer>
    </div>
  );
}
