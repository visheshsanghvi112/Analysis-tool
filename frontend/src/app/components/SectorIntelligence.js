'use client';
import { useState, useEffect, useMemo, useRef } from 'react';
import { 
  Trophy, Flame, Shield, Brain, TrendingUp, TrendingDown, 
  RefreshCw, AlertCircle, Zap, Target, Crown, Download, 
  ArrowRight, BarChart2, Layers, CheckCircle2, ChevronDown, 
  Filter, LineChart, Scale, Crosshair, Compass 
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

const VALUATION_CONFIG = {
  'DEEP VALUE':      { color: '#00e699', bg: '#00e69918', border: '#00e69940', label: 'DEEP VALUE' },
  'FAIR VALUE':      { color: '#60a5fa', bg: '#3b82f618', border: '#3b82f640', label: 'FAIR VALUE' },
  'GROWTH PREMIUM':  { color: '#f59e0b', bg: '#f59e0b18', border: '#f59e0b40', label: 'GROWTH PREMIUM' },
  'HIGH PREMIUM':    { color: '#ef4444', bg: '#ef444418', border: '#ef444440', label: 'HIGH PREMIUM' },
  'N/A':             { color: '#888888', bg: '#22222218', border: '#44444440', label: 'N/A' },
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

// ── Multi-Peer Performance SVG Line Chart ───────────────────────────
function ComparativeChart({ ticker, sector }) {
  const [period, setPeriod] = useState('6mo');
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hiddenSeries, setHiddenSeries] = useState(new Set());
  const [hoverIndex, setHoverIndex] = useState(null);
  const svgRef = useRef(null);

  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    fetch(`${API}/api/sector-chart?ticker=${encodeURIComponent(ticker)}&period=${period}`)
      .then(r => r.json())
      .then(d => { setChartData(d); })
      .catch(() => setChartData(null))
      .finally(() => setLoading(false));
  }, [ticker, period]);

  const toggleSeries = (sym) => {
    setHiddenSeries(prev => {
      const next = new Set(prev);
      if (next.has(sym)) next.delete(sym);
      else next.add(sym);
      return next;
    });
  };

  const dates = chartData?.dates || [];
  const series = (chartData?.series || []).filter(s => !hiddenSeries.has(s.ticker));
  const benchmark = !hiddenSeries.has('BENCHMARK') ? chartData?.benchmark : null;

  // Compute Bounds
  let minVal = 0;
  let maxVal = 0;
  [...series, ...(benchmark ? [benchmark] : [])].forEach(s => {
    s.data?.forEach(v => {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    });
  });

  const padding = (maxVal - minVal) * 0.1 || 5;
  const yMin = Math.floor(minVal - padding);
  const yMax = Math.ceil(maxVal + padding);
  const yRange = yMax - yMin || 1;

  const width = 800;
  const height = 260;
  const padLeft = 45;
  const padRight = 20;
  const padTop = 15;
  const padBottom = 25;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const getX = (idx) => padLeft + (idx / Math.max(dates.length - 1, 1)) * plotW;
  const getY = (val) => padTop + plotH - ((val - yMin) / yRange) * plotH;
  const zeroY = getY(0);

  const handleMouseMove = (e) => {
    if (!svgRef.current || dates.length === 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const normX = (mouseX / rect.width) * width;
    const idx = Math.round(((normX - padLeft) / plotW) * (dates.length - 1));
    if (idx >= 0 && idx < dates.length) {
      setHoverIndex(idx);
    }
  };

  return (
    <div style={{ background: '#0e0e0e', border: '1px solid #1a1a1a', borderRadius: '12px', padding: '16px', marginBottom: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
            <LineChart style={{ width: '16px', height: '16px', color: '#3b82f6' }} />
            <span style={{ fontSize: '13px', fontWeight: 800, color: '#fff' }}>Comparative Performance Rebased (%)</span>
          </div>
          <span style={{ fontSize: '11px', color: '#666' }}>All assets normalized to 0.0% at baseline for relative trajectory</span>
        </div>

        {/* Period Selector */}
        <div style={{ display: 'flex', gap: '4px', background: '#141414', padding: '3px', borderRadius: '8px', border: '1px solid #222' }}>
          {['1mo', '3mo', '6mo', '1y', 'ytd'].map(p => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              style={{
                background: period === p ? '#2563eb' : 'transparent',
                color: period === p ? '#fff' : '#888',
                border: 'none', borderRadius: '5px', padding: '4px 9px',
                fontSize: '10px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s'
              }}
            >
              {p.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div style={{ height: `${height}px`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontSize: '12px' }}>
          <RefreshCw style={{ width: '16px', height: '16px', animation: 'spin 1s linear infinite', marginRight: '8px' }} />
          Calculating normalized peer histories...
        </div>
      )}

      {!loading && chartData && dates.length > 0 && (
        <>
          <div style={{ position: 'relative', width: '100%', overflow: 'hidden' }}>
            <svg
              ref={svgRef}
              viewBox={`0 0 ${width} ${height}`}
              style={{ width: '100%', height: 'auto', display: 'block', cursor: 'crosshair' }}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverIndex(null)}
            >
              {/* Zero reference line */}
              <line x1={padLeft} y1={zeroY} x2={width - padRight} y2={zeroY} stroke="#333" strokeDasharray="4 4" strokeWidth="1" />
              <text x={padLeft - 6} y={zeroY + 3} textAnchor="end" fill="#666" fontSize="9" fontWeight="600">0%</text>

              {/* Top & Bottom Y-axis ticks */}
              <text x={padLeft - 6} y={getY(yMax) + 8} textAnchor="end" fill="#555" fontSize="9">{yMax > 0 ? `+${yMax}%` : `${yMax}%`}</text>
              <text x={padLeft - 6} y={getY(yMin)} textAnchor="end" fill="#555" fontSize="9">{yMin}%</text>

              {/* Sector Benchmark Line */}
              {benchmark && benchmark.data && (
                <path
                  d={benchmark.data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(v)}`).join(' ')}
                  fill="none"
                  stroke={benchmark.color}
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                  opacity="0.85"
                />
              )}

              {/* Peer Series Lines */}
              {series.map(s => {
                if (!s.data || s.data.length === 0) return null;
                const pathStr = s.data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(v)}`).join(' ');
                return (
                  <path
                    key={s.ticker}
                    d={pathStr}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={s.is_queried ? "2.6" : "1.8"}
                    opacity={s.is_queried ? "1.0" : "0.75"}
                  />
                );
              })}

              {/* Hover Cursor Vertical Line */}
              {hoverIndex !== null && (
                <line
                  x1={getX(hoverIndex)} y1={padTop}
                  x2={getX(hoverIndex)} y2={height - padBottom}
                  stroke="#ffffff40" strokeWidth="1" strokeDasharray="2 2"
                />
              )}

              {/* Start & End Dates on X-axis */}
              <text x={padLeft} y={height - 6} fill="#555" fontSize="9">{dates[0]}</text>
              <text x={width - padRight} y={height - 6} textAnchor="end" fill="#555" fontSize="9">{dates[dates.length - 1]}</text>
            </svg>

            {/* Hover Tooltip Box */}
            {hoverIndex !== null && (
              <div style={{
                position: 'absolute', top: '10px', right: '10px',
                background: '#141414f0', backdropFilter: 'blur(8px)',
                border: '1px solid #2a2a2a', borderRadius: '8px', padding: '8px 12px',
                fontSize: '10px', color: '#fff', pointerEvents: 'none', zIndex: 10
              }}>
                <span style={{ color: '#888', fontWeight: 700, display: 'block', marginBottom: '4px' }}>
                  {dates[hoverIndex]}
                </span>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                  {[...series, ...(benchmark ? [benchmark] : [])].map(s => {
                    const val = s.data[hoverIndex];
                    return (
                      <div key={s.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
                        <span style={{ color: s.color, fontWeight: 700 }}>{s.name}:</span>
                        <span style={{ fontWeight: 800, color: val >= 0 ? '#00e699' : '#ff4d4d' }}>
                          {val > 0 ? '+' : ''}{val}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Interactive Legend Pill Toggles */}
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px', marginTop: '12px' }}>
            {chartData.series.map(s => {
              const isHidden = hiddenSeries.has(s.ticker);
              return (
                <button
                  key={s.ticker}
                  onClick={() => toggleSeries(s.ticker)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '5px',
                    background: isHidden ? '#121212' : `${s.color}15`,
                    border: `1px solid ${isHidden ? '#222' : `${s.color}60`}`,
                    borderRadius: '6px', padding: '3px 8px', fontSize: '10px', fontWeight: 700,
                    color: isHidden ? '#555' : s.color, cursor: 'pointer', transition: 'all 0.15s'
                  }}
                >
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: isHidden ? '#444' : s.color }} />
                  <span>{s.name}</span>
                  <span style={{ color: s.final_return >= 0 ? '#00e699' : '#ff4d4d', marginLeft: '3px' }}>
                    {s.final_return > 0 ? '+' : ''}{s.final_return}%
                  </span>
                </button>
              );
            })}

            {chartData.benchmark && (
              <button
                onClick={() => toggleSeries('BENCHMARK')}
                style={{
                  display: 'flex', alignItems: 'center', gap: '5px',
                  background: hiddenSeries.has('BENCHMARK') ? '#121212' : '#f59e0b15',
                  border: `1px solid ${hiddenSeries.has('BENCHMARK') ? '#222' : '#f59e0b60'}`,
                  borderRadius: '6px', padding: '3px 8px', fontSize: '10px', fontWeight: 700,
                  color: hiddenSeries.has('BENCHMARK') ? '#555' : '#f59e0b', cursor: 'pointer', transition: 'all 0.15s'
                }}
              >
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: hiddenSeries.has('BENCHMARK') ? '#444' : '#f59e0b' }} />
                <span>Sector Avg</span>
                <span style={{ color: chartData.benchmark.final_return >= 0 ? '#00e699' : '#ff4d4d', marginLeft: '3px' }}>
                  {chartData.benchmark.final_return > 0 ? '+' : ''}{chartData.benchmark.final_return}%
                </span>
              </button>
            )}
          </div>

          {/* Statistical Pair Spread Mean-Reversion Box */}
          {chartData.pair_spread && (
            <div style={{
              marginTop: '12px', padding: '10px 12px', background: '#121622',
              border: '1px solid #1e2d4a', borderRadius: '8px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              flexWrap: 'wrap', gap: '8px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
                <Scale style={{ width: '13px', height: '13px', color: '#60a5fa' }} />
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#93c5fd' }}>
                  Statistical Pair Spread (vs {chartData.pair_spread.primary_peer.replace('.NS','')}):
                </span>
                <span style={{ fontSize: '11px', color: '#cbd5e1' }}>
                  {chartData.pair_spread.insight}
                </span>
              </div>
              <span style={{
                fontSize: '9px', fontWeight: 800, padding: '2px 6px', borderRadius: '4px',
                background: chartData.pair_spread.z_score <= -1.75 ? '#00e69920' : chartData.pair_spread.z_score >= 1.75 ? '#ff4d4d20' : '#ffffff10',
                color: chartData.pair_spread.z_score <= -1.75 ? '#00e699' : chartData.pair_spread.z_score >= 1.75 ? '#ff4d4d' : '#94a3b8'
              }}>
                {chartData.pair_spread.status}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ── Risk-Return 2D Quadrant Matrix Scatter View ───────────────────────
function RiskReturnQuadrant({ ranked, meta, ticker, onSelectPeer }) {
  const [hoverStock, setHoverStock] = useState(null);

  const medVol = meta?.median_vol || 20.0;
  const medRet = meta?.median_return || 0.0;

  // Coordinate Mapping for SVG
  const width = 750;
  const height = 300;
  const padL = 40;
  const padR = 40;
  const padT = 30;
  const padB = 30;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const volMin = Math.max((meta?.vol_min || 10.0) * 0.8, 5.0);
  const volMax = (meta?.vol_max || 45.0) * 1.15;
  const retMin = Math.min((meta?.ret_min || -20.0) * 1.2, -15.0);
  const retMax = Math.max((meta?.ret_max || 50.0) * 1.2, 30.0);

  const getX = (vol) => padL + ((vol - volMin) / (volMax - volMin)) * plotW;
  const getY = (ret) => padT + plotH - ((ret - retMin) / (retMax - retMin)) * plotH;

  const crossX = getX(medVol);
  const crossY = getY(medRet);

  return (
    <div style={{ background: '#0e0e0e', border: '1px solid #1a1a1a', borderRadius: '12px', padding: '16px', marginBottom: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
            <Crosshair style={{ width: '15px', height: '15px', color: '#10b981' }} />
            <span style={{ fontSize: '13px', fontWeight: 800, color: '#fff' }}>2D Risk-Return "Alpha Quadrant" Matrix</span>
          </div>
          <span style={{ fontSize: '11px', color: '#666' }}>Crosshairs indicate sector median return & volatility. Hover dots for details.</span>
        </div>
      </div>

      <div style={{ position: 'relative', width: '100%', overflow: 'hidden' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Quadrant Tint Backgrounds */}
          {/* Top-Left: Alpha Compounder */}
          <rect x={padL} y={padT} width={Math.max(crossX - padL, 0)} height={Math.max(crossY - padT, 0)} fill="#00e69906" />
          <text x={padL + 10} y={padT + 18} fill="#00e699aa" fontSize="10" fontWeight="800">🏆 ALPHA COMPOUNDERS (Low Vol, High Return)</text>

          {/* Top-Right: High-Beta Momentum */}
          <rect x={crossX} y={padT} width={Math.max(width - padR - crossX, 0)} height={Math.max(crossY - padT, 0)} fill="#3b82f606" />
          <text x={width - padR - 10} y={padT + 18} textAnchor="end" fill="#60a5faaa" fontSize="10" fontWeight="800">🚀 HIGH-BETA MOMENTUM (High Vol, High Return)</text>

          {/* Bottom-Left: Defensive Consolidators */}
          <rect x={padL} y={crossY} width={Math.max(crossX - padL, 0)} height={Math.max(height - padB - crossY, 0)} fill="#f59e0b06" />
          <text x={padL + 10} y={height - padB - 10} fill="#f59e0baa" fontSize="10" fontWeight="800">🛡️ DEFENSIVE SAFE (Low Vol, Low Return)</text>

          {/* Bottom-Right: Underperforming Traps */}
          <rect x={crossX} y={crossY} width={Math.max(width - padR - crossX, 0)} height={Math.max(height - padB - crossY, 0)} fill="#ef444406" />
          <text x={width - padR - 10} y={height - padB - 10} textAnchor="end" fill="#ef4444aa" fontSize="10" fontWeight="800">⚠️ UNDERPERFORMING TRAPS (High Vol, Low Return)</text>

          {/* Crosshair Medians */}
          <line x1={crossX} y1={padT} x2={crossX} y2={height - padB} stroke="#3b82f660" strokeWidth="1.5" strokeDasharray="3 3" />
          <line x1={padL} y1={crossY} x2={width - padR} y2={crossY} stroke="#3b82f660" strokeWidth="1.5" strokeDasharray="3 3" />
          <text x={crossX + 4} y={padT + 12} fill="#3b82f6" fontSize="9" fontWeight="700">Median Vol: {medVol}%</text>
          <text x={padL + 6} y={crossY - 5} fill="#3b82f6" fontSize="9" fontWeight="700">Median Ret: {medRet}%</text>

          {/* Stock Points */}
          {ranked.map(s => {
            const vol = s.annual_vol || 20.0;
            const ret = s.ret_1y ?? s.ret_3m ?? 0.0;
            const cx = getX(vol);
            const cy = getY(ret);
            const isQ = s.ticker === ticker;
            const sym = s.ticker.replace('.NS', '').replace('.BO', '');

            return (
              <g 
                key={s.ticker} 
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoverStock(s)}
                onClick={() => !isQ && onSelectPeer && onSelectPeer(s.ticker)}
              >
                {/* Active stock pulsing ring */}
                {isQ && (
                  <circle cx={cx} cy={cy} r="14" fill="none" stroke="#3b82f6" strokeWidth="2" opacity="0.6" strokeDasharray="2 2" />
                )}
                <circle 
                  cx={cx} cy={cy} 
                  r={isQ ? "8" : "5"} 
                  fill={isQ ? "#3b82f6" : "#ffffff"} 
                  stroke={isQ ? "#ffffff" : "#1a1a1a"} 
                  strokeWidth="2" 
                />
                <text 
                  x={cx} y={cy - 9} 
                  textAnchor="middle" 
                  fill={isQ ? "#60a5fa" : "#bbb"} 
                  fontSize="9" 
                  fontWeight={isQ ? "900" : "600"}
                >
                  {sym}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Hover Detail Modal */}
        {hoverStock && (
          <div style={{
            position: 'absolute', bottom: '10px', right: '10px',
            background: '#141414f0', backdropFilter: 'blur(8px)',
            border: '1px solid #2a2a2a', borderRadius: '8px', padding: '10px 14px',
            fontSize: '11px', color: '#fff', pointerEvents: 'none', zIndex: 10
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <span style={{ fontWeight: 800, fontSize: '13px', color: '#60a5fa' }}>
                {hoverStock.ticker.replace('.NS','')}
              </span>
              <span style={{ fontSize: '9px', background: '#3b82f620', color: '#60a5fa', padding: '1px 5px', borderRadius: '4px', fontWeight: 700 }}>
                {hoverStock.quadrant_label}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '6px' }}>
              <div><span style={{ color: '#666' }}>1Y Return: </span><span style={{ fontWeight: 700, color: (hoverStock.ret_1y ?? 0) >= 0 ? '#00e699' : '#ff4d4d' }}>{hoverStock.ret_1y}%</span></div>
              <div><span style={{ color: '#666' }}>Volatility: </span><span style={{ fontWeight: 700, color: '#fff' }}>{hoverStock.annual_vol}%</span></div>
              <div><span style={{ color: '#666' }}>Sharpe: </span><span style={{ fontWeight: 700, color: '#fff' }}>{hoverStock.sharpe}</span></div>
              <div><span style={{ color: '#666' }}>Score: </span><span style={{ fontWeight: 700, color: SCORE_COLOR(hoverStock.score) }}>{hoverStock.score}</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Valuation & Fundamental Multiples Comparison View ────────────────
function ValuationMatrix({ ranked, averages, ticker, onSelectPeer }) {
  const queriedStock = ranked.find(m => m.ticker === ticker);
  const medPE = averages?.median_pe;
  const medPB = averages?.median_pb;
  const medEV = averages?.median_ev_ebitda;
  const medROE = averages?.median_roe;

  return (
    <div>
      {/* 4 Multiples Summary Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '10px', marginBottom: '16px' }}>
        <BenchmarkCard 
          label="P/E Multiple (TTM)" 
          stockVal={queriedStock?.pe_ratio} 
          sectorVal={medPE} 
          unit="x"
          isVol={true} 
        />
        <BenchmarkCard 
          label="Price to Book (P/B)" 
          stockVal={queriedStock?.pb_ratio} 
          sectorVal={medPB} 
          unit="x"
          isVol={true} 
        />
        <BenchmarkCard 
          label="EV / EBITDA" 
          stockVal={queriedStock?.ev_ebitda} 
          sectorVal={medEV} 
          unit="x"
          isVol={true} 
        />
        <BenchmarkCard 
          label="Return on Equity (ROE)" 
          stockVal={queriedStock?.roe} 
          sectorVal={medROE} 
          unit="%"
        />
      </div>

      {/* Full Valuation Multiples Table */}
      <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#0e0e0e]">
        <div style={{ minWidth: '680px' }}>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '1.8fr 1fr 1fr 1fr 1fr 1.2fr 80px', 
            gap: '8px', padding: '10px 14px', background: '#141414', 
            fontSize: '10px', color: '#777', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' 
          }}>
            <span>Company / Asset</span>
            <span style={{ textAlign: 'right' }}>P/E Ratio</span>
            <span style={{ textAlign: 'right' }}>P/B Ratio</span>
            <span style={{ textAlign: 'right' }}>EV/EBITDA</span>
            <span style={{ textAlign: 'right' }}>ROE (%)</span>
            <span style={{ textAlign: 'center' }}>Valuation Standing</span>
            <span style={{ textAlign: 'center' }}>Action</span>
          </div>

        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {ranked.map((m, i) => {
            const sym = m.ticker.replace('.NS','').replace('.BO','');
            const isQueried = m.ticker === ticker;
            const vConfig = VALUATION_CONFIG[m.valuation_verdict] || VALUATION_CONFIG['FAIR VALUE'];

            return (
              <div
                key={m.ticker}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1.8fr 1fr 1fr 1fr 1fr 1.2fr 80px',
                  gap: '8px',
                  alignItems: 'center',
                  padding: '11px 14px',
                  background: isQueried ? '#141e33' : i % 2 === 0 ? '#0c0c0c' : '#0e0e0e',
                  borderBottom: '1px solid #181818',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ fontSize: '13px', fontWeight: 800, color: isQueried ? '#60a5fa' : '#fff' }}>
                      {sym}
                    </span>
                    {isQueried && (
                      <span style={{ fontSize: '8px', background: '#3b82f625', color: '#60a5fa', border: '1px solid #3b82f640', borderRadius: '3px', padding: '1px 4px', fontWeight: 800 }}>
                        YOU
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: '10px', color: '#666', display: 'block', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {m.company_name || sym}
                  </span>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: '#fff' }}>
                    {m.pe_ratio ? `${m.pe_ratio}x` : '–'}
                  </span>
                  {m.pe_vs_sector !== null && (
                    <span style={{ display: 'block', fontSize: '9px', color: m.pe_vs_sector <= 0 ? '#00e699aa' : '#f59e0baa' }}>
                      {m.pe_vs_sector > 0 ? '+' : ''}{m.pe_vs_sector}% vs med
                    </span>
                  )}
                </div>

                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: '#ddd' }}>
                    {m.pb_ratio ? `${m.pb_ratio}x` : '–'}
                  </span>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: '#ddd' }}>
                    {m.ev_ebitda ? `${m.ev_ebitda}x` : '–'}
                  </span>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '13px', fontWeight: 700, color: (m.roe ?? 0) >= 15 ? '#00e699' : '#ddd' }}>
                    {m.roe ? `${m.roe}%` : '–'}
                  </span>
                </div>

                <div style={{ textAlign: 'center' }}>
                  <span style={{
                    fontSize: '9px', fontWeight: 800, padding: '2px 6px', borderRadius: '4px',
                    color: vConfig.color, background: vConfig.bg, border: `1px solid ${vConfig.border}`
                  }}>
                    {vConfig.label}
                  </span>
                </div>

                <div style={{ textAlign: 'center' }}>
                  {isQueried ? (
                    <span style={{ fontSize: '10px', color: '#555', fontWeight: 600 }}>Active</span>
                  ) : (
                    <button
                      onClick={() => onSelectPeer && onSelectPeer(m.ticker)}
                      style={{
                        background: '#1a2234', border: '1px solid #3b82f640',
                        borderRadius: '6px', padding: '4px 8px', color: '#60a5fa',
                        fontSize: '10px', fontWeight: 700, cursor: 'pointer'
                      }}
                    >
                      Compare
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
    </div>
  );
}

// ── Main SectorIntelligence Component ──────────────────────────────
export default function SectorIntelligence({ ticker, onSelectPeer }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [loaded, setLoaded]   = useState(false);
  const [activeView, setActiveView] = useState('overview'); // 'overview', 'chart', 'quadrant', 'valuation'
  const [sortBy, setSortBy]   = useState('score');
  const [filterTier, setFilterTier] = useState('ALL');

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

  useEffect(() => {
    load();
  }, [ticker]);

  const insights       = data?.insights;
  const sectorAverages = data?.sector_averages || {};
  const quadrantMeta   = data?.quadrant_meta || {};
  const rawRanked      = data?.ranked || [];
  const queriedData    = rawRanked.find(m => m.ticker === ticker);

  const displayedPeers = useMemo(() => {
    let list = [...rawRanked];
    if (filterTier === 'TOP5') list = list.slice(0, 5);
    else if (filterTier === 'OUTPERFORMERS') list = list.filter(m => m.tier === 'LEADER' || m.tier === 'OUTPERFORMER');

    list.sort((a, b) => {
      if (sortBy === 'ret_3m') return (b.ret_3m ?? -999) - (a.ret_3m ?? -999);
      if (sortBy === 'ret_1y') return (b.ret_1y ?? -999) - (a.ret_1y ?? -999);
      if (sortBy === 'sharpe') return (b.sharpe ?? -999) - (a.sharpe ?? -999);
      if (sortBy === 'vol') return (a.annual_vol ?? 999) - (b.annual_vol ?? 999);
      if (sortBy === 'pe') return (a.pe_ratio ?? 999) - (b.pe_ratio ?? 999);
      return (b.score ?? 0) - (a.score ?? 0);
    });
    return list;
  }, [rawRanked, sortBy, filterTier]);

  const exportSectorCSV = () => {
    if (!rawRanked || rawRanked.length === 0) return;
    const headers = [
      'Rank', 'Ticker', 'Company', 'Tier', 'Score', 
      '3M Ret (%)', '3M Alpha (%)', '1Y Ret (%)', 'Sharpe', 'Annual Vol (%)', 
      'P/E', 'P/B', 'EV/EBITDA', 'ROE (%)', 'Valuation Verdict'
    ];
    const rows = rawRanked.map(m => [
      m.rank, m.ticker.replace('.NS', ''), m.company_name || '', m.tier || '', m.score,
      m.ret_3m ?? '', m.alpha_3m ?? '', m.ret_1y ?? '', m.sharpe ?? '', m.annual_vol ?? '',
      m.pe_ratio ?? '', m.pb_ratio ?? '', m.ev_ebitda ?? '', m.roe ?? '', m.valuation_verdict || ''
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.map(c => `"${c}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sector_intelligence_${data?.sector?.toLowerCase().replace(/[^a-z0-9]/g, '_') || 'sector'}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const queriedSym = ticker?.replace('.NS','').replace('.BO','');
  const tierConfig = TIER_CONFIG[queriedData?.tier] || TIER_CONFIG['MARKET PERFORMER'];

  return (
    <div className="glass-card p-4 sm:p-6 text-white" style={{ fontFamily: 'var(--font-poppins), sans-serif' }}>
      
      {/* ── Top Header ──────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '36px', height: '36px', background: 'rgba(245, 158, 11, 0.12)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Trophy style={{ width: '18px', height: '18px', color: '#f59e0b' }} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '15px', fontWeight: 800, color: '#fff', margin: 0 }}>Sector Intelligence Terminal</h3>
              {data && (
                <span style={{ fontSize: '10px', background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: '6px', padding: '2px 8px', fontWeight: 700 }}>
                  {data.sector}
                </span>
              )}
            </div>
            {data && <p style={{ fontSize: '11px', color: '#94a3b8', margin: '2px 0 0' }}>{rawRanked.length} competitors benchmarked across Momentum, Volatility &amp; Multiples</p>}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {loaded && rawRanked.length > 0 && (
            <button
              onClick={exportSectorCSV}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] hover:bg-blue-600/20 text-blue-300 border border-white/[0.08] hover:border-blue-500/30 text-xs font-semibold cursor-pointer transition-all shadow-sm"
              title="Download full sector terminal CSV"
            >
              <Download style={{ width: '12px', height: '12px' }} />
              <span>Export CSV</span>
            </button>
          )}

          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] text-slate-300 hover:text-white border border-white/[0.08] text-xs font-semibold cursor-pointer transition-all disabled:opacity-40"
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
          <button onClick={load} style={{ background: '#ff4d4d', color: '#fff', border: 'none', borderRadius: '6px', padding: '4px 10px', fontSize: '11px', fontWeight: 700, cursor: 'pointer' }}>
            Retry
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ height: '70px', background: '#141414', borderRadius: '10px', animation: 'pulse 1.5s ease-in-out infinite' }} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px' }}>
            {[1,2,3,4].map(i => (
              <div key={i} style={{ height: '70px', background: '#121212', borderRadius: '8px', animation: 'pulse 1.5s ease-in-out infinite' }} />
            ))}
          </div>
          <div style={{ height: '220px', background: '#111', borderRadius: '10px' }} />
        </div>
      )}

      {/* Loaded view */}
      {data && !loading && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>

          {/* ── Standing Banner ────────────────────────────────────────── */}
          {queriedData && (
            <div style={{
              background: `linear-gradient(135deg, #131b2e 0%, #0d1220 100%)`,
              border: '1px solid #3b82f640', borderRadius: '12px',
              padding: '16px 20px', marginBottom: '16px',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              flexWrap: 'wrap', gap: '12px',
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
                  {queriedData.quadrant_label && (
                    <span style={{ fontSize: '9px', fontWeight: 800, padding: '2px 7px', borderRadius: '4px', background: '#3b82f620', color: '#60a5fa', border: '1px solid #3b82f640' }}>
                      {queriedData.quadrant_label}
                    </span>
                  )}
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

              <div className="flex items-center gap-3 sm:gap-5 flex-wrap sm:flex-nowrap mt-3 sm:mt-0 justify-between w-full sm:w-auto pt-2 sm:pt-0 border-t sm:border-t-0 border-white/[0.06]">
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '11px', color: '#778899', display: 'block' }}>3M Alpha vs Sector</span>
                  <span style={{ fontSize: '16px', fontWeight: 800, color: (queriedData.alpha_3m ?? 0) >= 0 ? '#00e699' : '#ff4d4d' }}>
                    {(queriedData.alpha_3m ?? 0) > 0 ? '+' : ''}{queriedData.alpha_3m ?? 0}%
                  </span>
                </div>
                <div className="hidden sm:block" style={{ width: '1px', height: '32px', background: '#1e293b' }} />
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '11px', color: '#778899', display: 'block' }}>P/E vs Sector Med</span>
                  <span style={{ fontSize: '16px', fontWeight: 800, color: (queriedData.pe_vs_sector ?? 0) <= 0 ? '#00e699' : '#f59e0b' }}>
                    {queriedData.pe_vs_sector !== null ? `${queriedData.pe_vs_sector > 0 ? '+' : ''}${queriedData.pe_vs_sector}%` : '–'}
                  </span>
                </div>
                <div className="hidden sm:block" style={{ width: '1px', height: '32px', background: '#1e293b' }} />
                <div style={{ textAlign: 'right' }}>
                  <span style={{ fontSize: '11px', color: '#778899', display: 'block' }}>Composite Score</span>
                  <p style={{ fontSize: '26px', fontWeight: 900, color: SCORE_COLOR(queriedData.score), margin: 0, lineHeight: 1 }}>
                    {queriedData.score}<span style={{ fontSize: '13px', color: '#556677', fontWeight: 500 }}>/100</span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* ── View Navigation Tabs ───────────────────────────────────── */}
          <div className="flex gap-1.5 mb-4 bg-[#111] p-1 rounded-xl border border-[#1f1f1f] overflow-x-auto no-scrollbar">
            {[
              { id: 'overview',   label: 'Leaderboard & Scorecards', icon: BarChart2 },
              { id: 'chart',      label: 'Performance Overlay (%)',   icon: LineChart },
              { id: 'quadrant',   label: 'Risk-Return Quadrant',     icon: Crosshair },
              { id: 'valuation',  label: 'Valuation & Multiples',    icon: Scale },
            ].map(tab => {
              const Icon = tab.icon;
              const isA = activeView === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveView(tab.id)}
                  className="flex-1 min-w-[130px] sm:min-w-0"
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                    padding: '8px 12px', whiteSpace: 'nowrap',
                    background: isA ? '#1e293b' : 'transparent',
                    border: `1px solid ${isA ? '#3b82f650' : 'transparent'}`,
                    borderRadius: '7px',
                    color: isA ? '#60a5fa' : '#888',
                    fontSize: '11px', fontWeight: 700, cursor: 'pointer', transition: 'all 0.15s'
                  }}
                >
                  <Icon style={{ width: '13px', height: '13px' }} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* ── View 1: Leaderboard & Scorecards ──────────────────────── */}
          {activeView === 'overview' && (
            <div>
              {/* Benchmark Summary Cards */}
              <div style={{ marginBottom: '16px' }}>
                <p style={{ fontSize: '10px', color: '#666', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>
                  {queriedSym} vs {data.sector} Medians
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
                  <BenchmarkCard label="3-Month Momentum" stockVal={queriedData?.ret_3m} sectorVal={sectorAverages.avg_ret_3m} unit="%" />
                  <BenchmarkCard label="1-Year Return" stockVal={queriedData?.ret_1y} sectorVal={sectorAverages.avg_ret_1y} unit="%" />
                  <BenchmarkCard label="Risk-Adjusted (Sharpe)" stockVal={queriedData?.sharpe} sectorVal={sectorAverages.avg_sharpe} unit="" />
                  <BenchmarkCard label="Annual Volatility" stockVal={queriedData?.annual_vol} sectorVal={sectorAverages.avg_vol} unit="%" isVol={true} />
                </div>
              </div>

              {/* Category Champions */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '8px', marginBottom: '18px' }}>
                <InsightCard icon={Flame}  title="Top Momentum"  ticker={insights?.best_momentum} color="#f97316" description="Highest 3M return" />
                <InsightCard icon={Shield} title="Best Risk-Adj" ticker={insights?.best_risk_adj} color="#22c55e" description="Highest Sharpe ratio" />
                <InsightCard icon={Crown}  title="Sector Leader"  ticker={insights?.best_ml_signal} color="#a78bfa" description="Highest composite score" />
                <InsightCard icon={Zap}    title="Lowest Vol"     ticker={insights?.lowest_vol}    color="#38bdf8" description="Most stable price action" />
              </div>

              {/* Filter & Sort Bar */}
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
                        cursor: 'pointer'
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
                    <option value="pe">P/E Ratio (Lowest)</option>
                  </select>
                </div>
              </div>

              {/* Table */}
              <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#0e0e0e]">
                <div style={{ minWidth: '660px' }}>
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
                        >
                          <span style={{ fontSize: '13px', textAlign: 'center', fontWeight: 700, color: '#888' }}>
                            {m.rank <= 3 ? MEDAL[m.rank - 1] : `#${m.rank}`}
                          </span>

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

                          <div style={{ textAlign: 'right' }}>
                            <span style={{ fontSize: '14px', fontWeight: 800, color: SCORE_COLOR(m.score) }}>
                              {m.score}
                            </span>
                          </div>

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

                          <div style={{ textAlign: 'right' }}>
                            {ret1y !== null && ret1y !== undefined ? (
                              <span style={{ fontSize: '12px', fontWeight: 700, color: ret1y >= 0 ? '#00e699' : '#ff4d4d' }}>
                                {ret1y > 0 ? '+' : ''}{ret1y}%
                              </span>
                            ) : (
                              <span style={{ color: '#555', fontSize: '12px' }}>–</span>
                            )}
                          </div>

                          <div style={{ textAlign: 'right' }}>
                            <span style={{ fontSize: '12px', fontWeight: 600, color: (m.sharpe ?? 0) >= 1 ? '#00e699' : '#ccc' }}>
                              {m.sharpe !== null && m.sharpe !== undefined ? m.sharpe : '–'}
                            </span>
                          </div>

                          <div style={{ textAlign: 'center' }}>
                            {isQueried ? (
                              <span style={{ fontSize: '10px', color: '#555', fontWeight: 600 }}>Active</span>
                            ) : (
                              <button
                                onClick={() => onSelectPeer && onSelectPeer(m.ticker)}
                                style={{
                                  background: '#1a2234', border: '1px solid #3b82f640',
                                  borderRadius: '6px', padding: '4px 8px', color: '#60a5fa',
                                  fontSize: '10px', fontWeight: 700, cursor: 'pointer',
                                  display: 'inline-flex', alignItems: 'center', gap: '3px'
                                }}
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
              </div>
            </div>
          )}

          {/* ── View 2: Multi-Peer Performance Chart ──────────────────── */}
          {activeView === 'chart' && (
            <ComparativeChart ticker={ticker} sector={data.sector} />
          )}

          {/* ── View 3: Risk-Return Quadrant Matrix ───────────────────── */}
          {activeView === 'quadrant' && (
            <RiskReturnQuadrant 
              ranked={rawRanked} 
              meta={quadrantMeta} 
              ticker={ticker} 
              onSelectPeer={onSelectPeer} 
            />
          )}

          {/* ── View 4: Valuation & Multiples ─────────────────────────── */}
          {activeView === 'valuation' && (
            <ValuationMatrix 
              ranked={rawRanked} 
              averages={sectorAverages} 
              ticker={ticker} 
              onSelectPeer={onSelectPeer} 
            />
          )}

          {/* ── Bottom Quantitative Methodology ──────────────────────── */}
          <div style={{ marginTop: '16px', padding: '10px 14px', background: '#0e0e0e', border: '1px solid #1a1a1a', borderRadius: '8px', fontSize: '10px', color: '#555', lineHeight: 1.6 }}>
            <span style={{ color: '#888', fontWeight: 700 }}>Quantitative Institutional Rules:</span> Rankings synthesize Sharpe ratio (30%), 3M momentum (25%), volatility stability (20%), RSI health (15%), and 1Y return (10%). Sector alphas reflect relative outperformance against sector medians. Spread Z-scores evaluate statistical mean-reversion over rolling 60-day price ratios.
          </div>
        </div>
      )}
    </div>
  );
}
