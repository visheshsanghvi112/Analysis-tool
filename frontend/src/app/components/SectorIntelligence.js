'use client';
import { useState, useEffect } from 'react';
import { Trophy, Flame, Shield, Brain, TrendingUp, TrendingDown, RefreshCw, AlertCircle, Zap, Target, Crown } from 'lucide-react';

const API = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

const MEDAL = ['🥇', '🥈', '🥉'];

const SCORE_COLOR = (score) => {
  if (score >= 75) return '#00e699';
  if (score >= 50) return '#f59e0b';
  return '#ef4444';
};

const SCORE_BG = (score) => {
  if (score >= 75) return '#00e69910';
  if (score >= 50) return '#f59e0b10';
  return '#ef444410';
};

const SIGNAL_COLOR = {
  'STRONG BUY': '#00e699',
  'BUY':        '#22c55e',
  'HOLD':       '#f59e0b',
  'SELL':       '#ef4444',
  'STRONG SELL':'#dc2626',
};

function ScoreBar({ score }) {
  return (
    <div style={{ width: '100%', background: '#1c1c1c', borderRadius: '4px', height: '6px', overflow: 'hidden' }}>
      <div style={{
        width: `${score}%`,
        height: '100%',
        background: `linear-gradient(90deg, ${SCORE_COLOR(score)}, ${SCORE_COLOR(score)}99)`,
        borderRadius: '4px',
        transition: 'width 0.6s ease',
      }} />
    </div>
  );
}

function InsightCard({ icon: Icon, title, ticker, color, description }) {
  const sym = ticker?.replace('.NS','').replace('.BO','');
  return (
    <div style={{ background: `${color}08`, border: `1px solid ${color}20`, borderRadius: '10px', padding: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
        <Icon style={{ width: '14px', height: '14px', color }} />
        <span style={{ fontSize: '10px', color: '#666', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title}</span>
      </div>
      <p style={{ fontSize: '15px', fontWeight: 800, color, margin: 0 }}>{sym || '–'}</p>
      {description && <p style={{ fontSize: '10px', color: '#555', margin: '3px 0 0' }}>{description}</p>}
    </div>
  );
}

export default function SectorIntelligence({ ticker }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [loaded, setLoaded]   = useState(false);

  const load = () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);

    fetch(`${API}/api/sector-rank?ticker=${encodeURIComponent(ticker)}`)
      .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
      .then(d => { setData(d); setLoaded(true); })
      .catch(e => setError(`Could not load sector rankings. (${e.message})`))
      .finally(() => setLoading(false));
  };

  // Auto-load on ticker change
  useEffect(() => {
    setData(null);
    setLoaded(false);
    setError(null);
  }, [ticker]);

  const insights = data?.insights;
  const ranked   = data?.ranked || [];
  const queriedData = ranked.find(m => m.ticker === ticker);

  return (
    <div style={{ background: '#0a0a0a', border: '1px solid #1c1c1c', borderRadius: '16px', padding: '20px', color: '#fff', fontFamily: 'var(--font-poppins), sans-serif' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '32px', height: '32px', background: '#f59e0b15', border: '1px solid #f59e0b30', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Trophy style={{ width: '16px', height: '16px', color: '#f59e0b' }} />
          </div>
          <div>
            <h3 style={{ fontSize: '14px', fontWeight: 700, color: '#fff', margin: 0 }}>Sector Intelligence</h3>
            {data && <p style={{ fontSize: '11px', color: '#666', margin: 0 }}>{data.sector} · {ranked.length} peers ranked</p>}
          </div>
        </div>
        <button
          onClick={load}
          disabled={loading}
          style={{
            display: 'flex', alignItems: 'center', gap: '5px',
            background: loaded ? '#141414' : '#f59e0b',
            border: `1px solid ${loaded ? '#2a2a2a' : '#f59e0b'}`,
            borderRadius: '8px', padding: '7px 14px', color: loaded ? '#aaa' : '#000',
            fontSize: '11px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s',
          }}
        >
          <RefreshCw style={{ width: '12px', height: '12px', animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          {loading ? 'Ranking...' : loaded ? 'Refresh' : 'Run Sector Analysis'}
        </button>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
      `}</style>

      {/* Error */}
      {error && !loading && (
        <div style={{ display: 'flex', gap: '8px', padding: '12px', background: '#ff4d4d10', border: '1px solid #ff4d4d30', borderRadius: '8px', color: '#ff4d4d', fontSize: '12px', marginBottom: '12px' }}>
          <AlertCircle style={{ width: '14px', height: '14px', flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {[1,2,3,4,5].map(i => (
            <div key={i} style={{ height: '52px', background: '#141414', borderRadius: '8px', opacity: 1 - i * 0.12, animation: 'pulse 1.5s ease-in-out infinite' }} />
          ))}
        </div>
      )}

      {/* Not loaded yet */}
      {!loaded && !loading && !error && (
        <div style={{ textAlign: 'center', padding: '32px', color: '#444' }}>
          <Trophy style={{ width: '32px', height: '32px', margin: '0 auto 10px', opacity: 0.2 }} />
          <p style={{ fontSize: '13px', marginBottom: '4px', color: '#666' }}>Sector benchmarking not loaded</p>
          <p style={{ fontSize: '11px', color: '#444' }}>Click "Run Sector Analysis" to rank all {ticker?.replace('.NS','').replace('.BO','')} sector peers</p>
        </div>
      )}

      {/* Results */}
      {data && !loading && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>

          {/* Queried Stock Position Banner */}
          {queriedData && (
            <div style={{
              background: `linear-gradient(135deg, #1a1a2e, #16213e)`,
              border: '1px solid #3b82f630',
              borderRadius: '10px',
              padding: '14px 16px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '8px',
            }}>
              <div>
                <p style={{ fontSize: '11px', color: '#555', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', margin: '0 0 3px' }}>
                  {ticker?.replace('.NS','').replace('.BO','')} Sector Rank
                </p>
                <p style={{ fontSize: '24px', fontWeight: 900, color: '#3b82f6', margin: 0, lineHeight: 1 }}>
                  #{insights.queried_rank} <span style={{ fontSize: '14px', color: '#666' }}>of {insights.total_peers}</span>
                </p>
              </div>
              <div style={{ textAlign: 'right' }}>
                <p style={{ fontSize: '11px', color: '#555', margin: '0 0 3px' }}>Composite Score</p>
                <p style={{ fontSize: '24px', fontWeight: 900, color: SCORE_COLOR(queriedData.score), margin: 0 }}>
                  {queriedData.score}<span style={{ fontSize: '12px', color: '#555' }}>/100</span>
                </p>
              </div>
            </div>
          )}

          {/* Insight Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', marginBottom: '16px' }}>
            <InsightCard icon={Flame}   title="Best Momentum"   ticker={insights.best_momentum}  color="#f97316" description="Highest 3-month return" />
            <InsightCard icon={Shield}  title="Best Risk-Adj."  ticker={insights.best_risk_adj}  color="#22c55e" description="Highest Sharpe ratio" />
            <InsightCard icon={Brain}   title="Best ML Signal"  ticker={insights.best_ml_signal} color="#a78bfa" description="Best ML predicted return" />
            <InsightCard icon={Zap}     title="Lowest Volatility" ticker={insights.lowest_vol}   color="#3b82f6" description="Most stable performer" />
          </div>

          {/* Full Leaderboard */}
          <div>
            <p style={{ fontSize: '10px', color: '#555', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>
              Full Sector Leaderboard
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {ranked.map((m, i) => {
                const sym = m.ticker.replace('.NS','').replace('.BO','');
                const isQueried = m.ticker === ticker;
                const ret3m = m.ret_3m;
                return (
                  <div
                    key={m.ticker}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '24px 1fr auto',
                      gap: '10px',
                      alignItems: 'center',
                      padding: '10px 12px',
                      background: isQueried ? '#1a233a' : '#111',
                      border: `1px solid ${isQueried ? '#3b82f640' : '#1c1c1c'}`,
                      borderRadius: '8px',
                      transition: 'border-color 0.15s',
                    }}
                  >
                    <span style={{ fontSize: '14px', textAlign: 'center' }}>
                      {i < 3 ? MEDAL[i] : <span style={{ fontSize: '11px', color: '#555', fontWeight: 700 }}>#{m.rank}</span>}
                    </span>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                        <span style={{ fontSize: '13px', fontWeight: 800, color: isQueried ? '#3b82f6' : '#fff' }}>{sym}</span>
                        {isQueried && <span style={{ fontSize: '9px', background: '#3b82f620', color: '#3b82f6', borderRadius: '4px', padding: '1px 5px', fontWeight: 700 }}>YOU</span>}
                        {m.ml_signal && (
                          <span style={{ fontSize: '9px', background: `${SIGNAL_COLOR[m.ml_signal]}15`, color: SIGNAL_COLOR[m.ml_signal], borderRadius: '4px', padding: '1px 5px', fontWeight: 700 }}>
                            {m.ml_signal}
                          </span>
                        )}
                      </div>
                      <ScoreBar score={m.score} />
                    </div>
                    <div style={{ textAlign: 'right', minWidth: '80px' }}>
                      <p style={{ fontSize: '15px', fontWeight: 800, color: SCORE_COLOR(m.score), margin: 0 }}>{m.score}</p>
                      {ret3m !== null && (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '2px' }}>
                          {ret3m >= 0 ? <TrendingUp style={{ width: '10px', height: '10px', color: '#00e699' }} /> : <TrendingDown style={{ width: '10px', height: '10px', color: '#ff4d4d' }} />}
                          <span style={{ fontSize: '10px', color: ret3m >= 0 ? '#00e699' : '#ff4d4d', fontWeight: 600 }}>
                            {ret3m > 0 ? '+' : ''}{ret3m}%
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Methodology Note */}
          <p style={{ fontSize: '10px', color: '#333', marginTop: '14px', lineHeight: 1.6 }}>
            Composite Score = Sharpe (30%) + 3M Return Rank (25%) + Low Volatility (20%) + RSI Health (15%) + 1Y Return (10%). Rankings update on refresh.
          </p>
        </div>
      )}
    </div>
  );
}
