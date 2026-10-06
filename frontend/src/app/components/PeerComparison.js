'use client';
import { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Minus, Trophy, Search, X, ChevronRight, Zap, Shield, Activity, BarChart2, AlertCircle, RefreshCw, Download } from 'lucide-react';
import InfoBadge from './InfoBadge';

const API = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

const SIGNAL_COLOR = {
  'STRONG BUY': '#00e699',
  'BUY': '#22c55e',
  'HOLD': '#f59e0b',
  'SELL': '#ef4444',
  'STRONG SELL': '#dc2626',
};

function fmt(val, suffix = '') {
  if (val === null || val === undefined) return '–';
  return `${val > 0 ? '+' : ''}${val}${suffix}`;
}

function WinnerBadge({ isWinner, isTie }) {
  if (isTie) return <span style={{ fontSize: '9px', background: '#ffffff10', color: '#aaa', borderRadius: '4px', padding: '1px 5px', fontWeight: 700 }}>TIE</span>;
  if (isWinner) return <span style={{ fontSize: '9px', background: '#00e69920', color: '#00e699', borderRadius: '4px', padding: '1px 5px', fontWeight: 700 }}>✓ WIN</span>;
  return null;
}

function MetricRow({ label, valA, valB, winner, tickerA, tickerB, unit = '', higherIsBetter = true }) {
  const wA = winner === tickerA ? 'win' : winner === tickerB ? 'lose' : winner === 'tie' ? 'tie' : null;
  const wB = winner === tickerB ? 'win' : winner === tickerA ? 'lose' : winner === 'tie' ? 'tie' : null;

  const colorA = wA === 'win' ? '#00e699' : wA === 'lose' ? '#ff4d4d' : '#e2e8f0';
  const colorB = wB === 'win' ? '#00e699' : wB === 'lose' ? '#ff4d4d' : '#e2e8f0';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '8px', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid #1c1c1c' }}>
      <div style={{ textAlign: 'right' }}>
        <span style={{ fontSize: '13px', fontWeight: 700, color: colorA }}>{valA !== null && valA !== undefined ? `${valA}${unit}` : '–'}</span>
        {wA === 'win' && <WinnerBadge isWinner />}
        {wA === 'tie' && <WinnerBadge isTie />}
      </div>
      <div style={{ textAlign: 'center' }}>
        <span style={{ fontSize: '10px', color: '#666', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</span>
      </div>
      <div>
        <span style={{ fontSize: '13px', fontWeight: 700, color: colorB }}>{valB !== null && valB !== undefined ? `${valB}${unit}` : '–'}</span>
        {wB === 'win' && <WinnerBadge isWinner />}
        {wB === 'tie' && <WinnerBadge isTie />}
      </div>
    </div>
  );
}

export default function PeerComparison({ ticker, initialPeer = null }) {
  const [peers, setPeers] = useState([]);
  const [sector, setSector] = useState('');
  const [selectedPeer, setSelectedPeer] = useState(null);
  const [customInput, setCustomInput] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [peersLoading, setPeersLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('1v1'); // '1v1' or 'basket'
  const [basketData, setBasketData] = useState(null);
  const [basketLoading, setBasketLoading] = useState(false);
  const searchRef = useRef(null);

  // Trigger comparison when initialPeer is passed from Sector Intelligence
  useEffect(() => {
    if (initialPeer && initialPeer !== ticker) {
      loadComparison(initialPeer);
    }
  }, [initialPeer]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutside = (e) => {
      if (searchRef.current && !searchRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, []);

  // Debounced smart search for custom peer input
  useEffect(() => {
    const q = customInput.trim();
    if (q.length < 1) {
      setSuggestions([]);
      setDropdownOpen(false);
      return;
    }

    setSearching(true);
    const timeout = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/api/tickers?q=${encodeURIComponent(q)}&limit=6`);
        if (res.ok) {
          const data = await res.json();
          setSuggestions(data.tickers || []);
          setDropdownOpen(true);
        }
      } catch (_) {
        setSuggestions([]);
      } finally {
        setSearching(false);
      }
    }, 150);

    return () => clearTimeout(timeout);
  }, [customInput]);

  // Load suggested peers on ticker change
  useEffect(() => {
    if (!ticker) return;
    setPeers([]);
    setSector('');
    setSelectedPeer(null);
    setComparison(null);
    setBasketData(null);
    setError(null);
    setPeersLoading(true);

    fetch(`${API}/api/peers?ticker=${encodeURIComponent(ticker)}`)
      .then(r => r.json())
      .then(d => {
        const pList = d.peers || [];
        setPeers(pList);
        setSector(d.sector || '');
        if (!initialPeer && pList.length > 0) {
          loadComparison(pList[0]);
        }
      })
      .catch(() => setPeers([]))
      .finally(() => setPeersLoading(false));
  }, [ticker]);

  const loadBasket = (pList = null) => {
    if (!ticker) return;
    const target = pList || peers.slice(0, 4);
    if (target.length === 0) return;
    setBasketLoading(true);
    fetch(`${API}/api/multi-compare?ticker=${encodeURIComponent(ticker)}&peers=${encodeURIComponent(target.join(','))}`)
      .then(r => { if (!r.ok) throw new Error('Failed'); return r.json(); })
      .then(d => setBasketData(d))
      .catch(() => { })
      .finally(() => setBasketLoading(false));
  };

  const loadComparison = (peer) => {
    if (!peer || peer === ticker) return;
    setSelectedPeer(peer);
    setComparison(null);
    setError(null);
    setLoading(true);
    setDropdownOpen(false);

    fetch(`${API}/api/peer-compare?ticker=${encodeURIComponent(ticker)}&peer=${encodeURIComponent(peer)}`)
      .then(r => { if (!r.ok) throw new Error('Failed'); return r.json(); })
      .then(d => setComparison(d))
      .catch(() => setError('Could not fetch comparison data. Please try another peer.'))
      .finally(() => setLoading(false));
  };

  const handleCustom = async (e) => {
    e.preventDefault();
    const q = customInput.trim();
    if (!q) return;

    let targetTicker = null;
    // 1. If suggestions already loaded, pick top match
    if (suggestions.length > 0) {
      targetTicker = suggestions[0].symbol;
    } else {
      // 2. Query smart search API to resolve canonical ticker (handles aliases, typos, BSE codes)
      try {
        const res = await fetch(`${API}/api/tickers?q=${encodeURIComponent(q)}&limit=1`);
        if (res.ok) {
          const data = await res.json();
          if (data.tickers && data.tickers.length > 0) {
            targetTicker = data.tickers[0].symbol;
          }
        }
      } catch (_) { }
    }

    if (!targetTicker) {
      // Fallback direct format
      const isTickerUS = ticker && !ticker.endsWith('.NS') && !ticker.endsWith('.BO') && !ticker.startsWith('^');
      const sym = q.toUpperCase();
      targetTicker = sym.endsWith('.NS') || sym.endsWith('.BO') || sym.startsWith('^') || isTickerUS ? sym : sym + '.NS';
    }

    setCustomInput('');
    setDropdownOpen(false);

    if (viewMode === 'basket') {
      const currentList = basketData?.metrics?.map(m => m.ticker) || [ticker, ...peers.slice(0, 3)];
      if (!currentList.includes(targetTicker)) {
        const updated = [...currentList.filter(t => t !== ticker), targetTicker].slice(0, 5);
        loadBasket(updated);
      }
    } else {
      loadComparison(targetTicker);
    }
  };

  const symA = ticker?.replace('.NS', '').replace('.BO', '');
  const symB = selectedPeer?.replace('.NS', '').replace('.BO', '');

  const exportComparisonCSV = () => {
    if (!comparison) return;
    const headers = ['Metric', symA, symB, 'Winner'];
    const rows = [
      ['Current Price', comparison.metrics_a.current_price ?? '', comparison.metrics_b.current_price ?? '', ''],
      ['1-Month Return (%)', comparison.metrics_a.ret_1m ?? '', comparison.metrics_b.ret_1m ?? '', comparison.winners.ret_1m ?? ''],
      ['3-Month Return (%)', comparison.metrics_a.ret_3m ?? '', comparison.metrics_b.ret_3m ?? '', comparison.winners.ret_3m ?? ''],
      ['6-Month Return (%)', comparison.metrics_a.ret_6m ?? '', comparison.metrics_b.ret_6m ?? '', comparison.winners.ret_6m ?? ''],
      ['1-Year Return (%)', comparison.metrics_a.ret_1y ?? '', comparison.metrics_b.ret_1y ?? '', comparison.winners.ret_1y ?? ''],
      ['Sharpe Ratio', comparison.metrics_a.sharpe ?? '', comparison.metrics_b.sharpe ?? '', comparison.winners.sharpe ?? ''],
      ['Sortino Ratio', comparison.metrics_a.sortino ?? '', comparison.metrics_b.sortino ?? '', comparison.winners.sortino ?? ''],
      ['Calmar Ratio', comparison.metrics_a.calmar ?? '', comparison.metrics_b.calmar ?? '', comparison.winners.calmar ?? ''],
      ['VaR 95% (1D) (%)', comparison.metrics_a.var_95 ?? '', comparison.metrics_b.var_95 ?? '', comparison.winners.var_95 ?? ''],
      ['CVaR 95% (ES) (%)', comparison.metrics_a.cvar_95 ?? '', comparison.metrics_b.cvar_95 ?? '', comparison.winners.cvar_95 ?? ''],
      ['Max Drawdown (%)', comparison.metrics_a.max_drawdown ?? '', comparison.metrics_b.max_drawdown ?? '', comparison.winners.max_drawdown ?? ''],
      ['Dist from 52W High (%)', comparison.metrics_a.pct_from_high ?? '', comparison.metrics_b.pct_from_high ?? '', comparison.winners.pct_from_high ?? ''],
      ['Annual Volatility (%)', comparison.metrics_a.annual_vol ?? '', comparison.metrics_b.annual_vol ?? '', comparison.winners.annual_vol ?? ''],
      ['RSI (14)', comparison.metrics_a.rsi ?? '', comparison.metrics_b.rsi ?? '', ''],
      ['ML Return (%)', comparison.metrics_a.ml_return ?? '', comparison.metrics_b.ml_return ?? '', comparison.winners.ml_return ?? ''],
      ['ML Signal', comparison.metrics_a.ml_signal ?? '', comparison.metrics_b.ml_signal ?? '', ''],
    ];
    const csvContent = [headers.join(','), ...rows.map(r => r.map(c => `"${c}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `peer_comparison_${symA}_vs_${symB}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const exportBasketCSV = () => {
    if (!basketData || !basketData.metrics) return;
    const headers = ['Ticker', 'Name', 'Price', '1M Return (%)', '3M Return (%)', '1Y Return (%)', 'Sharpe', 'Sortino', 'Annual Vol (%)', 'Max Drawdown (%)', 'P/E', 'P/B', 'ROE (%)', 'Profit Margin (%)'];
    const rows = basketData.metrics.map(m => [
      m.ticker,
      m.company_name || '',
      m.current_price ?? '',
      m.ret_1m ?? '',
      m.ret_3m ?? '',
      m.ret_1y ?? '',
      m.sharpe ?? '',
      m.sortino ?? '',
      m.annual_vol ?? '',
      m.max_drawdown ?? '',
      m.pe_ratio ?? '',
      m.pb_ratio ?? '',
      m.roe ?? '',
      m.profit_margin ?? '',
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.map(c => `"${c}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `peer_basket_${symA}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div style={{ background: '#0a0a0a', border: '1px solid #1c1c1c', borderRadius: '16px', padding: '20px', color: '#fff', fontFamily: 'var(--font-poppins), sans-serif' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '32px', height: '32px', background: '#3b82f615', border: '1px solid #3b82f630', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <BarChart2 style={{ width: '16px', height: '16px', color: '#3b82f6' }} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: 700, color: '#fff', margin: 0 }}>Peer-to-Peer Comparison</h3>
              <InfoBadge infoKey="peer_valuation" />
            </div>
            {sector && <p style={{ fontSize: '11px', color: '#666', margin: 0 }}>{sector} Sector</p>}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{ display: 'flex', background: '#141414', padding: '3px', borderRadius: '8px', border: '1px solid #222' }}>
            <button
              onClick={() => setViewMode('1v1')}
              style={{
                background: viewMode === '1v1' ? '#2563eb' : 'transparent',
                color: viewMode === '1v1' ? '#fff' : '#888',
                border: 'none', borderRadius: '5px', padding: '4px 10px',
                fontSize: '10px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s'
              }}
            >
              1-on-1 Head-to-Head
            </button>
            <button
              onClick={() => { setViewMode('basket'); if (!basketData) loadBasket(); }}
              style={{
                background: viewMode === 'basket' ? '#2563eb' : 'transparent',
                color: viewMode === 'basket' ? '#fff' : '#888',
                border: 'none', borderRadius: '5px', padding: '4px 10px',
                fontSize: '10px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s'
              }}
            >
              Multi-Peer Basket
            </button>
          </div>

          {comparison && viewMode === '1v1' && (
            <button
              onClick={exportComparisonCSV}
              style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                padding: '6px 12px', background: 'rgba(59, 130, 246, 0.1)',
                border: '1px solid rgba(59, 130, 246, 0.3)', borderRadius: '6px',
                color: '#60a5fa', fontSize: '11px', fontWeight: 600, cursor: 'pointer',
                transition: 'all 0.15s'
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(59, 130, 246, 0.2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'rgba(59, 130, 246, 0.1)'}
              title="Download CSV report of comparison metrics"
            >
              <Download style={{ width: '12px', height: '12px' }} />
              <span>Export CSV</span>
            </button>
          )}
        </div>
      </div>

      {/* Suggested Peer Chips */}
      {peersLoading ? (
        <div style={{ display: 'flex', gap: '6px', marginBottom: '16px' }}>
          {[1, 2, 3, 4].map(i => <div key={i} style={{ width: '80px', height: '28px', background: '#1c1c1c', borderRadius: '6px', animation: 'pulse 1.5s ease-in-out infinite' }} />)}
        </div>
      ) : peers.length > 0 ? (
        <div style={{ marginBottom: '14px' }}>
          <p style={{ fontSize: '10px', color: '#555', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>Suggested Peers</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {peers.map(p => {
              const sym = p.replace('.NS', '').replace('.BO', '');
              const isActive = p === selectedPeer;
              return (
                <button
                  key={p}
                  onClick={() => loadComparison(p)}
                  style={{
                    padding: '5px 12px',
                    background: isActive ? '#3b82f6' : '#141414',
                    border: `1px solid ${isActive ? '#3b82f6' : '#2a2a2a'}`,
                    borderRadius: '6px',
                    color: isActive ? '#fff' : '#aaa',
                    fontSize: '11px',
                    fontWeight: 700,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    letterSpacing: '0.03em',
                  }}
                  onMouseEnter={e => { if (!isActive) { e.currentTarget.style.borderColor = '#3b82f6'; e.currentTarget.style.color = '#fff'; } }}
                  onMouseLeave={e => { if (!isActive) { e.currentTarget.style.borderColor = '#2a2a2a'; e.currentTarget.style.color = '#aaa'; } }}
                >
                  {sym}
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        <p style={{ fontSize: '12px', color: '#444', marginBottom: '14px' }}>No suggested peers in database for this ticker. Enter one below.</p>
      )}

      {/* Custom Peer Input with Smart Autocomplete */}
      <form onSubmit={handleCustom} style={{ display: 'flex', gap: '8px', marginBottom: '20px', position: 'relative' }}>
        <div ref={searchRef} style={{ flex: 1, position: 'relative' }}>
          <Search style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', width: '13px', height: '13px', color: searching ? '#3b82f6' : '#555' }} />
          <input
            value={customInput}
            onChange={e => setCustomInput(e.target.value)}
            onFocus={() => { if (suggestions.length > 0) setDropdownOpen(true); }}
            placeholder="Smart Search: any stock, ETF, alias, or BSE code (e.g. SBI, Tata Motors, 500325)..."
            style={{ width: '100%', background: '#111', border: '1px solid #2a2a2a', borderRadius: '8px', padding: '8px 10px 8px 30px', fontSize: '12px', color: '#fff', outline: 'none' }}
          />

          {/* Smart Suggestions Dropdown */}
          {dropdownOpen && suggestions.length > 0 && (
            <div style={{
              position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
              background: '#0e0e12', border: '1px solid #282835', borderRadius: '10px',
              zIndex: 50, boxShadow: '0 10px 30px rgba(0,0,0,0.8)', overflow: 'hidden'
            }}>
              {suggestions.map((item) => (
                <div
                  key={item.symbol}
                  onClick={() => {
                    setCustomInput('');
                    setDropdownOpen(false);
                    loadComparison(item.symbol);
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid #1a1a22',
                    transition: 'background 0.15s'
                  }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(59, 130, 246, 0.15)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#fff' }}>
                      {item.symbol.replace('.NS', '').replace('.BO', '')}
                    </span>
                    <span style={{ fontSize: '11px', color: '#888', maxWidth: '220px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.name}
                    </span>
                  </div>
                  {item.sector && (
                    <span style={{ fontSize: '10px', padding: '1px 6px', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', color: '#aaa' }}>
                      {item.sector}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
        <button
          type="submit"
          style={{ background: '#3b82f6', border: 'none', borderRadius: '8px', color: '#fff', fontSize: '12px', fontWeight: 700, padding: '8px 16px', cursor: 'pointer', whiteSpace: 'nowrap' }}
        >
          Compare
        </button>
      </form>

      {/* Loading */}
      {loading && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', padding: '32px', color: '#555' }}>
          <RefreshCw style={{ width: '16px', height: '16px', animation: 'spin 1s linear infinite' }} />
          <span style={{ fontSize: '12px' }}>Fetching comparison data...</span>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', background: '#ff4d4d10', border: '1px solid #ff4d4d30', borderRadius: '8px', color: '#ff4d4d', fontSize: '12px' }}>
          <AlertCircle style={{ width: '14px', height: '14px', flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}

      {/* 1-on-1 Comparison Table */}
      {viewMode === '1v1' && comparison && !loading && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>
          <style>{`
            @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
            @keyframes spin { to { transform: rotate(360deg); } }
            @keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 0.8; } }
          `}</style>

          {/* Column Headers */}
          {(() => {
            const isUSA = comparison?.ticker_a && !comparison.ticker_a.endsWith('.NS') && !comparison.ticker_a.endsWith('.BO');
            const isUSB = comparison?.ticker_b && !comparison.ticker_b.endsWith('.NS') && !comparison.ticker_b.endsWith('.BO');
            const currSymA = isUSA ? '$' : '₹';
            const currSymB = isUSB ? '$' : '₹';
            return (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '8px', marginBottom: '8px' }}>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ background: '#1c2a3a', border: '1px solid #3b82f630', borderRadius: '8px', padding: '8px 12px' }}>
                    <p style={{ fontSize: '14px', fontWeight: 800, color: '#3b82f6', margin: 0 }}>{symA}</p>
                    <p style={{ fontSize: '10px', color: '#555', margin: 0 }}>{currSymA}{comparison.metrics_a.current_price?.toLocaleString()}</p>
                  </div>
                </div>
                <div style={{ textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ fontSize: '10px', color: '#444', fontWeight: 700 }}>VS</span>
                </div>
                <div>
                  <div style={{ background: '#1c2a1c', border: '1px solid #22c55e30', borderRadius: '8px', padding: '8px 12px' }}>
                    <p style={{ fontSize: '14px', fontWeight: 800, color: '#22c55e', margin: 0 }}>{symB}</p>
                    <p style={{ fontSize: '10px', color: '#555', margin: 0 }}>{currSymB}{comparison.metrics_b.current_price?.toLocaleString()}</p>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Metrics */}
          <div style={{ background: '#111', borderRadius: '10px', padding: '0 14px' }}>
            <MetricRow label="1-Month Return" valA={comparison.metrics_a.ret_1m} valB={comparison.metrics_b.ret_1m} winner={comparison.winners.ret_1m} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="3-Month Return" valA={comparison.metrics_a.ret_3m} valB={comparison.metrics_b.ret_3m} winner={comparison.winners.ret_3m} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="6-Month Return" valA={comparison.metrics_a.ret_6m} valB={comparison.metrics_b.ret_6m} winner={comparison.winners.ret_6m} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="1-Year Return" valA={comparison.metrics_a.ret_1y} valB={comparison.metrics_b.ret_1y} winner={comparison.winners.ret_1y} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="Sharpe Ratio" valA={comparison.metrics_a.sharpe} valB={comparison.metrics_b.sharpe} winner={comparison.winners.sharpe} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} />
            <MetricRow label="Sortino Ratio" valA={comparison.metrics_a.sortino} valB={comparison.metrics_b.sortino} winner={comparison.winners.sortino} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} />
            <MetricRow label="Calmar Ratio" valA={comparison.metrics_a.calmar} valB={comparison.metrics_b.calmar} winner={comparison.winners.calmar} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} />
            <MetricRow label="VaR 95% (1D)" valA={comparison.metrics_a.var_95} valB={comparison.metrics_b.var_95} winner={comparison.winners.var_95} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" higherIsBetter={true} />
            <MetricRow label="CVaR 95% (ES)" valA={comparison.metrics_a.cvar_95} valB={comparison.metrics_b.cvar_95} winner={comparison.winners.cvar_95} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" higherIsBetter={true} />
            <MetricRow label="Max Drawdown (1Y)" valA={comparison.metrics_a.max_drawdown} valB={comparison.metrics_b.max_drawdown} winner={comparison.winners.max_drawdown} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" higherIsBetter={true} />
            <MetricRow label="Dist from 52W High" valA={comparison.metrics_a.pct_from_high} valB={comparison.metrics_b.pct_from_high} winner={comparison.winners.pct_from_high} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" higherIsBetter={true} />
            <MetricRow label="Annual Volatility" valA={comparison.metrics_a.annual_vol} valB={comparison.metrics_b.annual_vol} winner={comparison.winners.annual_vol} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" higherIsBetter={false} />
            <MetricRow label="RSI (14)" valA={comparison.metrics_a.rsi} valB={comparison.metrics_b.rsi} winner={null} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} />
            {(comparison.metrics_a.ml_return !== null || comparison.metrics_b.ml_return !== null) && (
              <MetricRow label="ML Predicted Return" valA={comparison.metrics_a.ml_return} valB={comparison.metrics_b.ml_return} winner={comparison.winners.ml_return} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            )}
            {(comparison.metrics_a.ml_signal || comparison.metrics_b.ml_signal) && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '8px', alignItems: 'center', padding: '10px 0' }}>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '11px', fontWeight: 800, color: SIGNAL_COLOR[comparison.metrics_a.ml_signal] || '#aaa' }}>
                    {comparison.metrics_a.ml_signal || '–'}
                  </span>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <span style={{ fontSize: '10px', color: '#666', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>ML Signal</span>
                </div>
                <div>
                  <span style={{ fontSize: '11px', fontWeight: 800, color: SIGNAL_COLOR[comparison.metrics_b.ml_signal] || '#aaa' }}>
                    {comparison.metrics_b.ml_signal || '–'}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Win Count Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '14px' }}>
            {[
              { t: comparison.ticker_a, sym: symA, color: '#3b82f6' },
              { t: comparison.ticker_b, sym: symB, color: '#22c55e' },
            ].map(({ t, sym, color }) => {
              const wins = Object.values(comparison.winners).filter(w => w === t).length;
              return (
                <div key={t} style={{ background: `${color}08`, border: `1px solid ${color}20`, borderRadius: '8px', padding: '10px 14px', textAlign: 'center' }}>
                  <p style={{ fontSize: '20px', fontWeight: 800, color, margin: 0 }}>{wins}</p>
                  <p style={{ fontSize: '10px', color: '#666', margin: 0 }}><strong style={{ color }}>{sym}</strong> wins</p>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Multi-Peer Basket View */}
      {viewMode === 'basket' && (
        <div>
          {basketLoading && (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', padding: '36px', color: '#666' }}>
              <RefreshCw style={{ width: '16px', height: '16px', animation: 'spin 1s linear infinite' }} />
              <span style={{ fontSize: '12px' }}>Analyzing multi-peer basket metrics...</span>
            </div>
          )}

          {!basketLoading && !basketData && (
            <div style={{ textAlign: 'center', padding: '32px 16px', color: '#666' }}>
              <p style={{ fontSize: '13px', marginBottom: '12px', color: '#aaa' }}>Load peer basket to compare multiple companies simultaneously.</p>
              <button
                onClick={() => loadBasket()}
                style={{ padding: '8px 18px', background: '#2563eb', color: '#fff', border: 'none', borderRadius: '8px', fontSize: '12px', fontWeight: 700, cursor: 'pointer' }}
              >
                Analyze Basket ({peers.slice(0, 4).length} Peers)
              </button>
            </div>
          )}

          {!basketLoading && basketData && (
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              {/* Category Leaders Pill Row */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px', marginBottom: '16px' }}>
                {[
                  { label: 'Top 1Y Return', key: 'ret_1y', unit: '%', color: '#22c55e' },
                  { label: 'Best Sharpe', key: 'sharpe', unit: '', color: '#3b82f6' },
                  { label: 'Lowest Volatility', key: 'annual_vol', unit: '%', color: '#a855f7' },
                  { label: 'Deep Value (P/E)', key: 'pe_ratio', unit: 'x', color: '#f59e0b' },
                  { label: 'Highest ROE', key: 'roe', unit: '%', color: '#06b6d4' }
                ].map(({ label, key, unit, color }) => {
                  const leadTicker = basketData.leaders?.[key];
                  const leadMetric = basketData.metrics?.find(m => m.ticker === leadTicker);
                  const leadVal = leadMetric?.[key];
                  return (
                    <div key={key} style={{ background: '#111', border: `1px solid ${color}30`, borderRadius: '10px', padding: '10px 12px' }}>
                      <p style={{ fontSize: '9px', color: '#777', textTransform: 'uppercase', letterSpacing: '0.05em', margin: '0 0 4px', fontWeight: 700 }}>{label}</p>
                      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '13px', fontWeight: 800, color }}>{leadTicker?.replace('.NS','').replace('.BO','') || '–'}</span>
                        <span style={{ fontSize: '11px', color: '#ccc', fontWeight: 600 }}>{leadVal !== null && leadVal !== undefined ? `${leadVal}${unit}` : '–'}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Basket Table */}
              <div style={{ overflowX: 'auto', borderRadius: '10px', border: '1px solid #1c1c1c' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px', textAlign: 'left', minWidth: '700px' }}>
                  <thead>
                    <tr style={{ background: '#121212', borderBottom: '1px solid #222', color: '#888' }}>
                      <th style={{ padding: '10px 12px', fontWeight: 700 }}>Company</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>Price</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>1M Return</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>3M Return</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>1Y Return</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>Sharpe</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>Vol (1Y)</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>P/E</th>
                      <th style={{ padding: '10px 10px', fontWeight: 700, textAlign: 'right' }}>ROE</th>
                      <th style={{ padding: '10px 12px', fontWeight: 700, textAlign: 'center' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {basketData.metrics?.map((m) => {
                      const isMain = m.ticker === basketData.queried_ticker;
                      const sym = m.ticker.replace('.NS', '').replace('.BO', '');
                      const isUS = !m.ticker.endsWith('.NS') && !m.ticker.endsWith('.BO');
                      const curr = isUS ? '$' : '₹';

                      const isRet1MLeader = basketData.leaders?.ret_1m === m.ticker;
                      const isRet3MLeader = basketData.leaders?.ret_3m === m.ticker;
                      const isRet1YLeader = basketData.leaders?.ret_1y === m.ticker;
                      const isSharpeLeader = basketData.leaders?.sharpe === m.ticker;
                      const isVolLeader = basketData.leaders?.annual_vol === m.ticker;
                      const isPeLeader = basketData.leaders?.pe_ratio === m.ticker;
                      const isRoeLeader = basketData.leaders?.roe === m.ticker;

                      return (
                        <tr
                          key={m.ticker}
                          style={{
                            background: isMain ? 'rgba(59, 130, 246, 0.08)' : 'transparent',
                            borderBottom: '1px solid #1a1a1a',
                            borderLeft: isMain ? '3px solid #3b82f6' : '3px solid transparent',
                            transition: 'background 0.15s'
                          }}
                        >
                          <td style={{ padding: '10px 12px' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <span style={{ fontWeight: 800, color: isMain ? '#60a5fa' : '#fff' }}>{sym}</span>
                              {isMain && (
                                <span style={{ fontSize: '9px', background: '#3b82f630', color: '#60a5fa', padding: '1px 5px', borderRadius: '4px', fontWeight: 700 }}>
                                  BASE
                                </span>
                              )}
                            </div>
                            <span style={{ fontSize: '10px', color: '#666', display: 'block', maxWidth: '140px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {m.company_name || sym}
                            </span>
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', fontWeight: 600, color: '#ddd' }}>
                            {m.current_price ? `${curr}${m.current_price.toLocaleString()}` : '–'}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', fontWeight: 700, color: m.ret_1m > 0 ? '#22c55e' : m.ret_1m < 0 ? '#ef4444' : '#888' }}>
                            {fmt(m.ret_1m, '%')}
                            {isRet1MLeader && <span title="Leader" style={{ marginLeft: '4px', fontSize: '9px' }}>👑</span>}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', fontWeight: 700, color: m.ret_3m > 0 ? '#22c55e' : m.ret_3m < 0 ? '#ef4444' : '#888' }}>
                            {fmt(m.ret_3m, '%')}
                            {isRet3MLeader && <span title="Leader" style={{ marginLeft: '4px', fontSize: '9px' }}>👑</span>}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', fontWeight: 700, color: m.ret_1y > 0 ? '#22c55e' : m.ret_1y < 0 ? '#ef4444' : '#888' }}>
                            {fmt(m.ret_1y, '%')}
                            {isRet1YLeader && <span title="Leader" style={{ marginLeft: '4px', fontSize: '9px' }}>👑</span>}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', fontWeight: 700, color: m.sharpe > 1.5 ? '#22c55e' : m.sharpe > 1 ? '#60a5fa' : '#aaa' }}>
                            {m.sharpe ?? '–'}
                            {isSharpeLeader && <span title="Best Sharpe" style={{ marginLeft: '4px', fontSize: '9px' }}>👑</span>}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', color: isVolLeader ? '#22c55e' : '#aaa' }}>
                            {m.annual_vol ? `${m.annual_vol}%` : '–'}
                            {isVolLeader && <span title="Lowest Volatility" style={{ marginLeft: '4px', fontSize: '9px' }}>🛡️</span>}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', color: isPeLeader ? '#22c55e' : '#aaa' }}>
                            {m.pe_ratio ? `${m.pe_ratio}x` : '–'}
                            {isPeLeader && <span title="Deep Value P/E" style={{ marginLeft: '4px', fontSize: '9px' }}>💎</span>}
                          </td>
                          <td style={{ padding: '10px 10px', textAlign: 'right', color: isRoeLeader ? '#22c55e' : '#aaa' }}>
                            {m.roe ? `${m.roe}%` : '–'}
                            {isRoeLeader && <span title="Top ROE" style={{ marginLeft: '4px', fontSize: '9px' }}>⭐</span>}
                          </td>
                          <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                            {!isMain && (
                              <button
                                onClick={() => {
                                  setSelectedPeer(m.ticker);
                                  setViewMode('1v1');
                                  loadComparison(m.ticker);
                                }}
                                style={{
                                  background: '#1f2937',
                                  border: '1px solid #374151',
                                  borderRadius: '5px',
                                  color: '#60a5fa',
                                  fontSize: '10px',
                                  fontWeight: 700,
                                  padding: '3px 8px',
                                  cursor: 'pointer',
                                  transition: 'all 0.15s'
                                }}
                                onMouseEnter={e => e.currentTarget.style.background = '#2563eb30'}
                                onMouseLeave={e => e.currentTarget.style.background = '#1f2937'}
                              >
                                1v1 Deep Dive
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px', color: '#666' }}>
                <span>💡 Click any peer chip above to add/remove it from this multi-stock comparison basket (up to 5 stocks).</span>
                <span>👑 = Category Leader</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {viewMode === '1v1' && !comparison && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '24px', color: '#444' }}>
          <BarChart2 style={{ width: '28px', height: '28px', margin: '0 auto 8px', opacity: 0.3 }} />
          <p style={{ fontSize: '12px' }}>Select a peer above or enter a ticker to start comparing</p>
        </div>
      )}
    </div>
  );
}
