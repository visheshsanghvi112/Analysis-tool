'use client';
import { useState, useEffect, useMemo } from 'react';
import { 
  Trophy, Flame, Shield, Brain, TrendingUp, TrendingDown, 
  RefreshCw, AlertCircle, Zap, Target, Crown, Download, 
  ArrowRight, BarChart2, Layers, CheckCircle2, ChevronDown, Filter 
} from 'lucide-react';

const API = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

const MEDAL = ['🥇', '🥈', '🥉'];

const SCORE_COLOR = (score) => {
  if (score >= 75) return '#00e699';
  if (score >= 50) return '#f59e0b';
  return '#ef4444';
};

const TIER_CONFIG = {
  'LEADER':            { color: '#00e699', bg: '#00e69918', border: '#00e69940', label: 'LEADER' },
  'OUTPERFORMER':      { color: '#60a5fa', bg: '#3b82f618', border: '#3b82f640', label: 'OUTPERFORMER' },
  'MARKET PERFORMER':  { color: '#f59e0b', bg: '#f59e0b18', border: '#f59e0b40', label: 'MARKET PERFORMER' },
  'LAGGARD':           { color: '#ef4444', bg: '#ef444418', border: '#ef444440', label: 'LAGGARD' },
};

const SIGNAL_COLOR = {
  'STRONG BUY': '#00e699',
  'BUY':        '#22c55e',
  'HOLD':       '#f59e0b',
  'SELL':       '#ef4444',
  'STRONG SELL':'#dc2626',
};

function ScoreBar({ score }) {
  const safeScore = Math.min(Math.max(score || 0, 0), 100);
  return (
    <div style={{ width: '100%', background: '#1c1c1c', borderRadius: '4px', height: '5px', overflow: 'hidden' }}>
      <div style={{
        width: `${safeScore}%`,
        height: '100%',
        background: `linear-gradient(90deg, ${SCORE_COLOR(safeScore)}, ${SCORE_COLOR(safeScore)}aa)`,
        borderRadius: '4px',
        transition: 'width 0.6s ease',
      }} />
    </div>
  );
}

function InsightCard({ icon: Icon, title, ticker, color, description }) {
  const sym = ticker?.replace('.NS','').replace('.BO','');
  return (
    <div style={{ background: `${color}0a`, border: `1px solid ${color}22`, borderRadius: '10px', padding: '12px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <Icon style={{ width: '14px', height: '14px', color }} />
          <span style={{ fontSize: '10px', color: '#777', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title}</span>
        </div>
        <p style={{ fontSize: '16px', fontWeight: 800, color, margin: 0, letterSpacing: '-0.02em' }}>{sym || '–'}</p>
      </div>
      {description && <p style={{ fontSize: '10px', color: '#666', margin: '4px 0 0' }}>{description}</p>}
    </div>
  );
}

function BenchmarkCard({ label, stockVal, sectorVal, unit = '%', isHigherBetter = true, isVol = false }) {
  const sVal = typeof stockVal === 'number' ? stockVal : null;
  const secVal = typeof sectorVal === 'number' ? sectorVal : null;
  
  let diff = null;
  let isOutperforming = false;
  
  if (sVal !== null && secVal !== null) {
    diff = sVal - secVal;
    if (isVol) {
      // Lower volatility is better
      isOutperforming = diff <= 0;
    } else {
      isOutperforming = diff >= 0;
    }
  }

  const pillColor = isOutperforming ? '#00e699' : '#ff4d4d';
  const pillBg = isOutperforming ? '#00e69915' : '#ff4d4d15';

  return (
    <div style={{ background: '#111', border: '1px solid #1f1f1f', borderRadius: '10px', padding: '12px 14px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span style={{ fontSize: '11px', color: '#888', fontWeight: 600 }}>{label}</span>
        {diff !== null && (
          <span style={{ 
            fontSize: '10px', fontWeight: 700, color: pillColor, background: pillBg, 
            padding: '2px 6px', borderRadius: '4px' 
          }}>
            {diff > 0 ? '+' : ''}{diff.toFixed(1)}{unit} {isOutperforming ? 'α' : 'lag'}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <div>
          <span style={{ fontSize: '10px', color: '#555', display: 'block' }}>Stock</span>
          <span style={{ fontSize: '16px', fontWeight: 800, color: '#fff' }}>
            {sVal !== null ? `${sVal > 0 && !isVol ? '+' : ''}${sVal}${unit}` : '–'}
          </span>
        </div>

        <div style={{ textAlign: 'right' }}>
          <span style={{ fontSize: '10px', color: '#555', display: 'block' }}>Sector Avg</span>
          <span style={{ fontSize: '14px', fontWeight: 700, color: '#888' }}>
            {secVal !== null ? `${secVal > 0 && !isVol ? '+' : ''}${secVal}${unit}` : '–'}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function SectorIntelligence({ ticker, onSelectPeer }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [loaded, setLoaded]   = useState(false);
  const [sortBy, setSortBy]   = useState('score'); // 'score', 'ret_3m', 'ret_1y', 'sharpe', 'vol'
  const [filterTier, setFilterTier] = useState('ALL'); // 'ALL', 'TOP5', 'OUTPERFORMERS'

  const load = () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);

    fetch(`${API}/api/sector-rank?ticker=${encodeURIComponent(ticker)}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then(d => { setData(d); setLoaded(true); })
      .catch(e => setError(`Could not load sector rankings. (${e.message})`))
      .finally(() => setLoading(false));
  };

  // Automatically load on mount or ticker change
  useEffect(() => {
    load();
  }, [ticker]);

  const insights        = data?.insights;
  const sectorAverages  = data?.sector_averages || {};
  const rawRanked       = data?.ranked || [];
  const queriedData     = rawRanked.find(m => m.ticker === ticker);

  // Sorting and Filtering
  const displayedPeers = useMemo(() => {
    let list = [...rawRanked];

    // Filter
    if (filterTier === 'TOP5') {
      list = list.slice(0, 5);
    } else if (filterTier === 'OUTPERFORMERS') {
      list = list.filter(m => m.tier === 'LEADER' || m.tier === 'OUTPERFORMER');
    }

    // Sort
    list.sort((a, b) => {
      if (sortBy === 'ret_3m') {
        return (b.ret_3m ?? -999) - (a.ret_3m ?? -999);
      }
      if (sortBy === 'ret_1y') {
        return (b.ret_1y ?? -999) - (a.ret_1y ?? -999);
      }
      if (sortBy === 'sharpe') {
        return (b.sharpe ?? -999) - (a.sharpe ?? -999);
      }
      if (sortBy === 'vol') {
        return (a.annual_vol ?? 999) - (b.annual_vol ?? 999);
      }
      return (b.score ?? 0) - (a.score ?? 0);
    });

    return list;
  }, [rawRanked, sortBy, filterTier]);

  const exportSectorCSV = () => {
    if (!rawRanked || rawRanked.length === 0) return;
    const headers = [
      'Rank', 'Ticker', 'Tier', 'Composite Score (100)', 
      '3M Return (%)', '3M Alpha (%)', '1Y Return (%)', '1Y Alpha (%)', 
      'Sharpe Ratio', 'Annual Volatility (%)', 'RSI 14', 'Signal'
    ];
    const rows = rawRanked.map(m => [
      m.rank,
      m.ticker.replace('.NS', '').replace('.BO', ''),
      m.tier || '',
      m.score,
      m.ret_3m ?? '',
      m.alpha_3m ?? '',
      m.ret_1y ?? '',
      m.alpha_1y ?? '',
      m.sharpe ?? '',
      m.annual_vol ?? '',
      m.rsi ?? '',
      m.ml_signal || ''
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.map(c => `"${c}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sector_rankings_${data?.sector?.toLowerCase().replace(/[^a-z0-9]/g, '_') || 'sector'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const queriedSym = ticker?.replace('.NS','').replace('.BO','');
  const tierConfig = TIER_CONFIG[queriedData?.tier] || TIER_CONFIG['MARKET PERFORMER'];

  return (
    <div style={{ background: '#0a0a0a', border: '1px solid #1c1c1c', borderRadius: '16px', padding: '20px', color: '#fff', fontFamily: 'var(--font-poppins), sans-serif' }}>
      
      {/* ── Top Header ──────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '36px', height: '36px', background: '#f59e0b15', border: '1px solid #f59e0b35', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Trophy style={{ width: '18px', height: '18px', color: '#f59e0b' }} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 800, color: '#fff', margin: 0 }}>Sector Intelligence</h3>
              {data && (
                <span style={{ fontSize: '10px', background: '#f59e0b20', color: '#f59e0b', borderRadius: '5px', padding: '2px 7px', fontWeight: 700 }}>
                  {data.sector}
                </span>
              )}
            </div>
            {data && <p style={{ fontSize: '11px', color: '#666', margin: '2px 0 0' }}>{rawRanked.length} industry bellwethers benchmarked in real-time</p>}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {loaded && rawRanked.length > 0 && (
            <button
              onClick={exportSectorCSV}
              style={{
                display: 'flex', alignItems: 'center', gap: '5px',
                background: '#141414', border: '1px solid #2a2a2a',
                borderRadius: '8px', padding: '7px 12px', color: '#60a5fa',
                fontSize: '11px', fontWeight: 600, cursor: 'pointer', transition: 'all 0.15s'
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = '#3b82f6'}
              onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2a2a'}
              title="Download full sector rankings CSV"
            >
              <Download style={{ width: '12px', height: '12px' }} />
              <span>Export CSV</span>
            </button>
          )}

          <button
            onClick={load}
            disabled={loading}
            style={{
              display: 'flex', alignItems: 'center', gap: '6px',
              background: '#141414',
              border: '1px solid #2a2a2a',
              borderRadius: '8px', padding: '7px 13px', color: '#aaa',
              fontSize: '11px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s',
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = '#444'}
            onMouseLeave={e => e.currentTarget.style.borderColor = '#2a2a2a'}
          >
            <RefreshCw style={{ width: '12px', height: '12px', animation: loading ? 'spin 1s linear infinite' : 'none' }} />
            <span>{loading ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
      `}</style>

      {/* Error state */}
      {error && !loading && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px', background: '#ff4d4d12', border: '1px solid #ff4d4d35', borderRadius: '10px', color: '#ff4d4d', fontSize: '12px', marginBottom: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertCircle style={{ width: '15px', height: '15px', flexShrink: 0 }} />
            <span>{error}</span>
          </div>
          <button 
            onClick={load} 
            style={{ background: '#ff4d4d', color: '#fff', border: 'none', borderRadius: '6px', padding: '4px 10px', fontSize: '11px', fontWeight: 700, cursor: 'pointer' }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading state skeleton */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ height: '70px', background: '#141414', borderRadius: '10px', animation: 'pulse 1.5s ease-in-out infinite' }} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
            {[1,2,3,4].map(i => (
              <div key={i} style={{ height: '70px', background: '#121212', borderRadius: '8px', animation: 'pulse 1.5s ease-in-out infinite' }} />
            ))}
          </div>
          {[1,2,3,4].map(i => (
            <div key={i} style={{ height: '54px', background: '#111', borderRadius: '8px', opacity: 1 - i * 0.15 }} />
          ))}
        </div>
      )}

      {/* Loaded view */}
      {data && !loading && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>

          {/* ── 1. Queried Stock Banner ─────────────────────────────────── */}
          {queriedData && (
            <div style={{
              background: `linear-gradient(135deg, #131b2e 0%, #0d1220 100%)`,
              border: '1px solid #3b82f640',
              borderRadius: '12px',
              padding: '16px 20px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '12px',
            }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{ fontSize: '11px', color: '#8899aa', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    {queriedSym} Active Standing
                  </span>
                  <span style={{ 
                    fontSize: '9px', fontWeight: 800, padding: '2px 7px', borderRadius: '4px',
                    color: tierConfig.color, background: tierConfig.bg, border: `1px solid ${tierConfig.border}`
                  }}>
                    {tierConfig.label}
                  </span>
                  {queriedData.ml_signal && (
                    <span style={{ 
                      fontSize: '9px', fontWeight: 800, padding: '2px 7px', borderRadius: '4px',
                      color: SIGNAL_COLOR[queriedData.ml_signal] || '#fff', 
                      background: `${SIGNAL_COLOR[queriedData.ml_signal] || '#fff'}18` 
                    }}>
                      {queriedData.ml_signal}
                    </span>
                  )}
                </div>
                <p style={{ fontSize: '26px', fontWeight: 900, color: '#3b82f6', margin: 0, lineHeight: 1 }}>
                  #{insights?.queried_rank} <span style={{ fontSize: '13px', color: '#778899', fontWeight: 500 }}>of {insights?.total_peers} in {data.sector}</span>
                </p>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '11px', color: '#778899', display: 'block' }}>3M Sector Alpha</span>
                  <span style={{ fontSize: '16px', fontWeight: 800, color: (queriedData.alpha_3m ?? 0) >= 0 ? '#00e699' : '#ff4d4d' }}>
                    {(queriedData.alpha_3m ?? 0) > 0 ? '+' : ''}{queriedData.alpha_3m ?? 0}%
                  </span>
                </div>
                <div style={{ width: '1px', height: '32px', background: '#1e293b' }} />
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '11px', color: '#778899', display: 'block' }}>Composite Score</span>
                  <p style={{ fontSize: '26px', fontWeight: 900, color: SCORE_COLOR(queriedData.score), margin: 0, lineHeight: 1 }}>
                    {queriedData.score}<span style={{ fontSize: '13px', color: '#556677', fontWeight: 500 }}>/100</span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* ── 2. Stock vs Sector Benchmark Comparison ───────────────── */}
          <div style={{ marginBottom: '16px' }}>
            <p style={{ fontSize: '10px', color: '#666', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>
              {queriedSym} vs {data.sector} Benchmarks
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
              <BenchmarkCard 
                label="3-Month Momentum" 
                stockVal={queriedData?.ret_3m} 
                sectorVal={sectorAverages.avg_ret_3m} 
                unit="%"
              />
              <BenchmarkCard 
                label="1-Year Return" 
                stockVal={queriedData?.ret_1y} 
                sectorVal={sectorAverages.avg_ret_1y} 
                unit="%"
              />
              <BenchmarkCard 
                label="Risk-Adjusted (Sharpe)" 
                stockVal={queriedData?.sharpe} 
                sectorVal={sectorAverages.avg_sharpe} 
                unit=""
              />
              <BenchmarkCard 
                label="Annual Volatility" 
                stockVal={queriedData?.annual_vol} 
                sectorVal={sectorAverages.avg_vol} 
                unit="%"
                isVol={true}
              />
            </div>
          </div>

          {/* ── 3. Category Champions ─────────────────────────────────── */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '8px', marginBottom: '20px' }}>
            <InsightCard icon={Flame}   title="Top Momentum"   ticker={insights?.best_momentum}  color="#f97316" description="Highest 3M return" />
            <InsightCard icon={Shield}  title="Best Risk-Adj"  ticker={insights?.best_risk_adj}  color="#22c55e" description="Highest Sharpe ratio" />
            <InsightCard icon={Crown}   title="Sector Leader"   ticker={insights?.best_ml_signal} color="#a78bfa" description="Highest composite score" />
            <InsightCard icon={Zap}     title="Lowest Vol"      ticker={insights?.lowest_vol}     color="#38bdf8" description="Most stable price action" />
          </div>

          {/* ── 4. Filter & Sort Toolbar ──────────────────────────────── */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ fontSize: '11px', color: '#666', fontWeight: 600, marginRight: '4px' }}>Filter:</span>
              {[
                { id: 'ALL', label: `All (${rawRanked.length})` },
                { id: 'TOP5', label: 'Top 5' },
                { id: 'OUTPERFORMERS', label: 'Outperformers' },
              ].map(f => (
                <button
                  key={f.id}
                  onClick={() => setFilterTier(f.id)}
                  style={{
                    background: filterTier === f.id ? '#1e293b' : '#111',
                    border: `1px solid ${filterTier === f.id ? '#3b82f660' : '#1f1f1f'}`,
                    color: filterTier === f.id ? '#60a5fa' : '#777',
                    padding: '4px 10px', borderRadius: '6px', fontSize: '11px', fontWeight: 600,
                    cursor: 'pointer', transition: 'all 0.15s'
                  }}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '11px', color: '#666', fontWeight: 600 }}>Sort by:</span>
              <select
                value={sortBy}
                onChange={e => setSortBy(e.target.value)}
                style={{
                  background: '#141414', border: '1px solid #2a2a2a', color: '#ddd',
                  borderRadius: '6px', padding: '4px 8px', fontSize: '11px', fontWeight: 600,
                  cursor: 'pointer', outline: 'none'
                }}
              >
                <option value="score">Composite Score</option>
                <option value="ret_3m">3-Month Return</option>
                <option value="ret_1y">1-Year Return</option>
                <option value="sharpe">Sharpe Ratio</option>
                <option value="vol">Volatility (Lowest)</option>
              </select>
            </div>
          </div>

          {/* ── 5. Full Sector Leaderboard Table ──────────────────────── */}
          <div style={{ border: '1px solid #1a1a1a', borderRadius: '10px', overflow: 'hidden', background: '#0e0e0e' }}>
            {/* Table Header */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: '40px 1.8fr 1fr 1fr 1fr 1fr 80px', 
              gap: '8px', padding: '10px 14px', background: '#141414', 
              fontSize: '10px', color: '#777', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' 
            }}>
              <span>Rank</span>
              <span>Asset</span>
              <span style={{ textAlign: 'right' }}>Score</span>
              <span style={{ textAlign: 'right' }}>3M Return</span>
              <span style={{ textAlign: 'right' }}>1Y Return</span>
              <span style={{ textAlign: 'right' }}>Sharpe</span>
              <span style={{ textAlign: 'center' }}>Action</span>
            </div>

            {/* Table Rows */}
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              {displayedPeers.map((m, i) => {
                const sym = m.ticker.replace('.NS','').replace('.BO','');
                const isQueried = m.ticker === ticker;
                const ret3m = m.ret_3m;
                const ret1y = m.ret_1y;
                const tier = TIER_CONFIG[m.tier] || TIER_CONFIG['MARKET PERFORMER'];

                return (
                  <div
                    key={m.ticker}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '40px 1.8fr 1fr 1fr 1fr 1fr 80px',
                      gap: '8px',
                      alignItems: 'center',
                      padding: '11px 14px',
                      background: isQueried ? '#141e33' : i % 2 === 0 ? '#0c0c0c' : '#0e0e0e',
                      borderBottom: '1px solid #181818',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => {
                      if (!isQueried) e.currentTarget.style.background = '#161616';
                    }}
                    onMouseLeave={e => {
                      if (!isQueried) e.currentTarget.style.background = i % 2 === 0 ? '#0c0c0c' : '#0e0e0e';
                    }}
                  >
                    {/* Rank */}
                    <span style={{ fontSize: '13px', textAlign: 'center', fontWeight: 700, color: '#888' }}>
                      {m.rank <= 3 ? MEDAL[m.rank - 1] : `#${m.rank}`}
                    </span>

                    {/* Asset Symbol & Badges */}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                        <span style={{ fontSize: '13px', fontWeight: 800, color: isQueried ? '#60a5fa' : '#fff' }}>
                          {sym}
                        </span>
                        {isQueried && (
                          <span style={{ fontSize: '8px', background: '#3b82f625', color: '#60a5fa', border: '1px solid #3b82f640', borderRadius: '3px', padding: '1px 4px', fontWeight: 800 }}>
                            YOU
                          </span>
                        )}
                        <span style={{ fontSize: '8px', background: tier.bg, color: tier.color, border: `1px solid ${tier.border}`, borderRadius: '3px', padding: '1px 4px', fontWeight: 700 }}>
                          {m.tier}
                        </span>
                      </div>
                      <div style={{ marginTop: '4px', maxWidth: '120px' }}>
                        <ScoreBar score={m.score} />
                      </div>
                    </div>

                    {/* Composite Score */}
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '14px', fontWeight: 800, color: SCORE_COLOR(m.score) }}>
                        {m.score}
                      </span>
                    </div>

                    {/* 3M Return & Alpha */}
                    <div style={{ textAlign: 'right' }}>
                      {ret3m !== null && ret3m !== undefined ? (
                        <div>
                          <span style={{ fontSize: '12px', fontWeight: 700, color: ret3m >= 0 ? '#00e699' : '#ff4d4d' }}>
                            {ret3m > 0 ? '+' : ''}{ret3m}%
                          </span>
                          {m.alpha_3m !== null && m.alpha_3m !== undefined && (
                            <span style={{ display: 'block', fontSize: '9px', color: m.alpha_3m >= 0 ? '#00e699aa' : '#ff4d4daa' }}>
                              {m.alpha_3m > 0 ? '+' : ''}{m.alpha_3m}% α
                            </span>
                          )}
                        </div>
                      ) : (
                        <span style={{ color: '#555', fontSize: '12px' }}>–</span>
                      )}
                    </div>

                    {/* 1Y Return */}
                    <div style={{ textAlign: 'right' }}>
                      {ret1y !== null && ret1y !== undefined ? (
                        <span style={{ fontSize: '12px', fontWeight: 700, color: ret1y >= 0 ? '#00e699' : '#ff4d4d' }}>
                          {ret1y > 0 ? '+' : ''}{ret1y}%
                        </span>
                      ) : (
                        <span style={{ color: '#555', fontSize: '12px' }}>–</span>
                      )}
                    </div>

                    {/* Sharpe Ratio */}
                    <div style={{ textAlign: 'right' }}>
                      <span style={{ fontSize: '12px', fontWeight: 600, color: (m.sharpe ?? 0) >= 1 ? '#00e699' : '#ccc' }}>
                        {m.sharpe !== null && m.sharpe !== undefined ? m.sharpe : '–'}
                      </span>
                    </div>

                    {/* Action Button */}
                    <div style={{ textAlign: 'center' }}>
                      {isQueried ? (
                        <span style={{ fontSize: '10px', color: '#555', fontWeight: 600 }}>Active</span>
                      ) : (
                        <button
                          onClick={() => {
                            if (onSelectPeer) {
                              onSelectPeer(m.ticker);
                            } else {
                              window.location.href = `/?ticker=${encodeURIComponent(m.ticker)}`;
                            }
                          }}
                          style={{
                            background: '#1a2234',
                            border: '1px solid #3b82f640',
                            borderRadius: '6px',
                            padding: '4px 8px',
                            color: '#60a5fa',
                            fontSize: '10px',
                            fontWeight: 700,
                            cursor: 'pointer',
                            transition: 'all 0.15s',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '3px'
                          }}
                          onMouseEnter={e => {
                            e.currentTarget.style.background = '#2563eb';
                            e.currentTarget.style.color = '#fff';
                          }}
                          onMouseLeave={e => {
                            e.currentTarget.style.background = '#1a2234';
                            e.currentTarget.style.color = '#60a5fa';
                          }}
                          title={`Compare ${queriedSym} vs ${sym} head-to-head`}
                        >
                          <BarChart2 style={{ width: '10px', height: '10px' }} />
                          <span>Compare</span>
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── 6. Quantitative Methodology Note ──────────────────────── */}
          <div style={{ marginTop: '14px', padding: '10px 14px', background: '#0e0e0e', border: '1px solid #1a1a1a', borderRadius: '8px', fontSize: '10px', color: '#555', lineHeight: 1.6 }}>
            <span style={{ color: '#888', fontWeight: 700 }}>Institutional Benchmark Methodology:</span> Composite Sector Score is calculated using percentile rankings across 5 quantitative factors: Sharpe Ratio (30%), 3-Month Momentum (25%), Volatility Stability (20%), RSI Normalization (15%), and 1-Year Compound Performance (10%). Sector alphas reflect relative outperformance against sector medians.
          </div>
        </div>
      )}
    </div>
  );
}
