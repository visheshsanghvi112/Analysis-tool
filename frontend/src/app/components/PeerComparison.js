'use client';
import { useState, useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Minus, Trophy, Search, X, ChevronRight, Zap, Shield, Activity, BarChart2, AlertCircle, RefreshCw } from 'lucide-react';
import InfoBadge from './InfoBadge';

const API = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

const SIGNAL_COLOR = {
  'STRONG BUY': '#00e699',
  'BUY':        '#22c55e',
  'HOLD':       '#f59e0b',
  'SELL':       '#ef4444',
  'STRONG SELL':'#dc2626',
};

function fmt(val, suffix = '') {
  if (val === null || val === undefined) return '–';
  return `${val > 0 ? '+' : ''}${val}${suffix}`;
}

function WinnerBadge({ isWinner, isTie }) {
  if (isTie)     return <span style={{ fontSize: '9px', background: '#ffffff10', color: '#aaa', borderRadius: '4px', padding: '1px 5px', fontWeight: 700 }}>TIE</span>;
  if (isWinner)  return <span style={{ fontSize: '9px', background: '#00e69920', color: '#00e699', borderRadius: '4px', padding: '1px 5px', fontWeight: 700 }}>✓ WIN</span>;
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

export default function PeerComparison({ ticker }) {
  const [peers, setPeers]           = useState([]);
  const [sector, setSector]         = useState('');
  const [selectedPeer, setSelectedPeer] = useState(null);
  const [customInput, setCustomInput]   = useState('');
  const [suggestions, setSuggestions]   = useState([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [searching, setSearching]       = useState(false);
  const [comparison, setComparison]     = useState(null);
  const [loading, setLoading]           = useState(false);
  const [peersLoading, setPeersLoading] = useState(true);
  const [error, setError]               = useState(null);
  const searchRef                       = useRef(null);

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
    setError(null);
    setPeersLoading(true);

    fetch(`${API}/api/peers?ticker=${encodeURIComponent(ticker)}`)
      .then(r => r.json())
      .then(d => {
        setPeers(d.peers || []);
        setSector(d.sector || '');
      })
      .catch(() => setPeers([]))
      .finally(() => setPeersLoading(false));
  }, [ticker]);

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

    // 1. If suggestions already loaded, pick top match
    if (suggestions.length > 0) {
      const top = suggestions[0].symbol;
      setCustomInput('');
      setDropdownOpen(false);
      loadComparison(top);
      return;
    }

    // 2. Query smart search API to resolve canonical ticker (handles aliases, typos, BSE codes)
    try {
      const res = await fetch(`${API}/api/tickers?q=${encodeURIComponent(q)}&limit=1`);
      if (res.ok) {
        const data = await res.json();
        if (data.tickers && data.tickers.length > 0) {
          setCustomInput('');
          setDropdownOpen(false);
          loadComparison(data.tickers[0].symbol);
          return;
        }
      }
    } catch (_) {}

    // Fallback direct format
    const sym = q.toUpperCase();
    const full = sym.endsWith('.NS') || sym.endsWith('.BO') || sym.startsWith('^') ? sym : sym + '.NS';
    setCustomInput('');
    setDropdownOpen(false);
    loadComparison(full);
  };

  const symA = ticker?.replace('.NS','').replace('.BO','');
  const symB = selectedPeer?.replace('.NS','').replace('.BO','');

  return (
    <div style={{ background: '#0a0a0a', border: '1px solid #1c1c1c', borderRadius: '16px', padding: '20px', color: '#fff', fontFamily: 'var(--font-poppins), sans-serif' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
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

      {/* Suggested Peer Chips */}
      {peersLoading ? (
        <div style={{ display: 'flex', gap: '6px', marginBottom: '16px' }}>
          {[1,2,3,4].map(i => <div key={i} style={{ width: '80px', height: '28px', background: '#1c1c1c', borderRadius: '6px', animation: 'pulse 1.5s ease-in-out infinite' }} />)}
        </div>
      ) : peers.length > 0 ? (
        <div style={{ marginBottom: '14px' }}>
          <p style={{ fontSize: '10px', color: '#555', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>Suggested Peers</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            {peers.map(p => {
              const sym = p.replace('.NS','').replace('.BO','');
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
                  onMouseEnter={e => { if (!isActive) { e.currentTarget.style.borderColor = '#3b82f6'; e.currentTarget.style.color = '#fff'; }}}
                  onMouseLeave={e => { if (!isActive) { e.currentTarget.style.borderColor = '#2a2a2a'; e.currentTarget.style.color = '#aaa'; }}}
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

      {/* Comparison Table */}
      {comparison && !loading && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>
          <style>{`
            @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
            @keyframes spin { to { transform: rotate(360deg); } }
            @keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 0.8; } }
          `}</style>

          {/* Column Headers */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '8px', marginBottom: '8px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ background: '#1c2a3a', border: '1px solid #3b82f630', borderRadius: '8px', padding: '8px 12px' }}>
                <p style={{ fontSize: '14px', fontWeight: 800, color: '#3b82f6', margin: 0 }}>{symA}</p>
                <p style={{ fontSize: '10px', color: '#555', margin: 0 }}>₹{comparison.metrics_a.current_price?.toLocaleString()}</p>
              </div>
            </div>
            <div style={{ textAlign: 'center', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ fontSize: '10px', color: '#444', fontWeight: 700 }}>VS</span>
            </div>
            <div>
              <div style={{ background: '#1c2a1c', border: '1px solid #22c55e30', borderRadius: '8px', padding: '8px 12px' }}>
                <p style={{ fontSize: '14px', fontWeight: 800, color: '#22c55e', margin: 0 }}>{symB}</p>
                <p style={{ fontSize: '10px', color: '#555', margin: 0 }}>₹{comparison.metrics_b.current_price?.toLocaleString()}</p>
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div style={{ background: '#111', borderRadius: '10px', padding: '0 14px' }}>
            <MetricRow label="1-Month Return"   valA={comparison.metrics_a.ret_1m}     valB={comparison.metrics_b.ret_1m}     winner={comparison.winners.ret_1m}     tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="3-Month Return"   valA={comparison.metrics_a.ret_3m}     valB={comparison.metrics_b.ret_3m}     winner={comparison.winners.ret_3m}     tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="6-Month Return"   valA={comparison.metrics_a.ret_6m}     valB={comparison.metrics_b.ret_6m}     winner={comparison.winners.ret_6m}     tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="1-Year Return"    valA={comparison.metrics_a.ret_1y}     valB={comparison.metrics_b.ret_1y}     winner={comparison.winners.ret_1y}     tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" />
            <MetricRow label="Sharpe Ratio"     valA={comparison.metrics_a.sharpe}     valB={comparison.metrics_b.sharpe}     winner={comparison.winners.sharpe}     tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} />
            <MetricRow label="Annual Volatility" valA={comparison.metrics_a.annual_vol} valB={comparison.metrics_b.annual_vol} winner={comparison.winners.annual_vol} tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} unit="%" higherIsBetter={false} />
            <MetricRow label="RSI (14)"          valA={comparison.metrics_a.rsi}        valB={comparison.metrics_b.rsi}        winner={null}                          tickerA={comparison.ticker_a} tickerB={comparison.ticker_b} />
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

      {/* Empty state */}
      {!comparison && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '24px', color: '#444' }}>
          <BarChart2 style={{ width: '28px', height: '28px', margin: '0 auto 8px', opacity: 0.3 }} />
          <p style={{ fontSize: '12px' }}>Select a peer above or enter a ticker to start comparing</p>
        </div>
      )}
    </div>
  );
}
