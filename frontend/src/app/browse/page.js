'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Search, X, TrendingUp, ChevronRight, Sparkles, Filter, RefreshCw } from 'lucide-react';

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

export default function BrowsePage() {
  const router = useRouter();
  
  const [allTickers, setAllTickers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeSector, setActiveSector] = useState('All');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 48;

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
    
    // Check if query is similar to any sector names (excluding the currently active one)
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

          {/* Smart Search Bar Container */}
          <div ref={searchContainerRef} style={{ flex: 1, position: 'relative', maxWidth: '480px', marginLeft: 'auto' }}>
            <div style={{ position: 'relative' }}>
              <Search style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', width: '14px', height: '14px', color: '#aaaaaa' }} />
              <input
                type="text"
                value={query}
                onChange={(e) => { setQuery(e.target.value); setPage(1); }}
                onFocus={() => setIsFocused(true)}
                onClick={() => setIsFocused(true)}
                onMouseDown={() => setIsFocused(true)}
                placeholder="Smart Search (e.g. HDFC Bank)..."
                style={{
                  width: '100%',
                  background: '#121212',
                  border: '1px solid #3a3a3a',
                  borderRadius: '8px',
                  padding: '8px 36px 8px 36px',
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
                      onClick={() => { setQuery(item); setIsFocused(false); }}
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
        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '80px 0', gap: '12px', color: '#888888' }}>
            <RefreshCw style={{ width: '24px', height: '24px', animation: 'spin 1s linear infinite', color: '#ffffff' }} />
            <p style={{ fontSize: '13px' }}>Loading NSE stocks list...</p>
          </div>
        ) : (
          <>
            {/* Title Block */}
            <div style={{ marginBottom: '24px' }}>
              <h1 style={{ fontSize: 'clamp(22px, 5vw, 32px)', fontWeight: 700, color: '#ffffff', letterSpacing: '-0.03em', marginBottom: '6px' }}>
                Discover Categories
              </h1>
              <p style={{ fontSize: '13px', color: '#aaaaaa' }}>
                Browse all {allTickers.length.toLocaleString()} NSE listed stocks categorized dynamically.
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
                      background: isA ? meta.color + '20' : '#121212',
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
                    cursor: 'pointer',
                    transition: 'opacity 0.1s'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.opacity = '0.9'}
                  onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
                >
                  Switch to Sector
                </button>
              </div>
            )}

            {/* Active Sector Display */}
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
          </>
        )}
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
