'use client';

import { useState, useEffect, useMemo } from 'react';
import { 
  Newspaper, 
  RefreshCw, 
  ExternalLink, 
  AlertTriangle, 
  Zap, 
  TrendingUp, 
  Clock, 
  BarChart3, 
  ThumbsUp, 
  ThumbsDown, 
  Minus, 
  Sparkles, 
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  Calendar,
  Filter,
  ArrowRight,
  Flame,
  Info
} from 'lucide-react';
import InfoBadge from './InfoBadge';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://stock-analysis-backend-seven.vercel.app');

export default function AdvancedNews({ ticker, companyName }) {
  const [newsData, setNewsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sentimentFilter, setSentimentFilter] = useState('ALL'); // 'ALL' | 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'
  const [recencyFilter, setRecencyFilter] = useState('ALL');     // 'ALL' | '24h' | '3d' | '7d' | '15d'

  const fetchAdvancedNews = async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    
    try {
      const url = `${API_BASE_URL}/api/advanced-news?ticker=${ticker}${companyName ? `&company_name=${encodeURIComponent(companyName)}` : ''}`;
      const res = await fetch(url);
      const json = await res.json();
      
      if (!res.ok) throw new Error(json.detail || 'Failed to fetch news');
      
      setNewsData(json.news_intelligence);
    } catch (err) {
      setError(err.message);
      setNewsData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdvancedNews();
  }, [ticker, companyName]);

  const getSentimentColor = (sentiment) => {
    if (sentiment > 0.1) return 'text-emerald-400';
    if (sentiment < -0.1) return 'text-rose-400';
    return 'text-slate-400';
  };

  const getSentimentIcon = (sentiment) => {
    if (sentiment > 0.1) return <ThumbsUp className="h-3 w-3" />;
    if (sentiment < -0.1) return <ThumbsDown className="h-3 w-3" />;
    return <Minus className="h-3 w-3" />;
  };

  const getImpactColor = (impact) => {
    if (impact > 70) return 'text-red-400 bg-red-500/10 border-red-500/20';
    if (impact > 50) return 'text-orange-400 bg-orange-500/10 border-orange-500/20';
    if (impact > 30) return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/20';
    return 'text-slate-400 bg-slate-500/10 border-slate-500/20';
  };

  const formatTimeAgo = (dateString, ageHours) => {
    if (ageHours !== undefined && ageHours !== null) {
      if (ageHours < 1) return 'Just now';
      if (ageHours < 24) return `${Math.round(ageHours)}h ago`;
      const days = Math.round(ageHours / 24);
      return `${days}d ago`;
    }
    const now = new Date();
    const date = new Date(dateString);
    const diffInHours = Math.floor((now - date) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays}d ago`;
  };

  const articles = newsData?.articles || [];
  const decision = newsData?.decision;
  const recencyDist = newsData?.recency_distribution || {};

  const positiveCount = useMemo(() => articles.filter(a => a.sentiment > 0.1).length, [articles]);
  const negativeCount = useMemo(() => articles.filter(a => a.sentiment < -0.1).length, [articles]);
  const neutralCount = useMemo(() => articles.filter(a => a.sentiment >= -0.1 && a.sentiment <= 0.1).length, [articles]);

  const filteredArticles = useMemo(() => {
    return articles.filter(a => {
      // Sentiment filter
      if (sentimentFilter === 'POSITIVE' && a.sentiment <= 0.1) return false;
      if (sentimentFilter === 'NEGATIVE' && a.sentiment >= -0.1) return false;
      if (sentimentFilter === 'NEUTRAL' && (a.sentiment > 0.1 || a.sentiment < -0.1)) return false;

      // Recency bucket filter
      if (recencyFilter !== 'ALL') {
        const bucket = a.recency_bucket || '15d';
        if (recencyFilter === '24h' && bucket !== '24h') return false;
        if (recencyFilter === '3d' && bucket !== '24h' && bucket !== '3d') return false;
        if (recencyFilter === '7d' && bucket === '15d') return false;
      }
      return true;
    });
  }, [articles, sentimentFilter, recencyFilter]);

  // Action Signal Theme styling
  const getActionTheme = (signal) => {
    switch (signal) {
      case 'STRONG BUY':
        return {
          bg: 'from-emerald-500/20 via-teal-500/15 to-emerald-950/40 border-emerald-500/50',
          badge: 'bg-emerald-500/30 text-emerald-200 border-emerald-400/60 shadow-[0_0_15px_rgba(16,185,129,0.3)]',
          text: 'text-emerald-300',
          meter: 'bg-gradient-to-r from-emerald-500 to-teal-400'
        };
      case 'BUY':
        return {
          bg: 'from-teal-500/20 via-emerald-500/10 to-slate-900 border-teal-500/40',
          badge: 'bg-teal-500/25 text-teal-200 border-teal-400/50',
          text: 'text-teal-300',
          meter: 'bg-gradient-to-r from-teal-500 to-emerald-400'
        };
      case 'HOLD / WATCHLIST':
        return {
          bg: 'from-amber-500/15 via-yellow-500/10 to-slate-900 border-amber-500/35',
          badge: 'bg-amber-500/25 text-amber-200 border-amber-400/50',
          text: 'text-amber-300',
          meter: 'bg-gradient-to-r from-yellow-500 to-amber-400'
        };
      case 'REDUCE / CAUTION':
        return {
          bg: 'from-orange-500/20 via-rose-500/10 to-slate-900 border-orange-500/40',
          badge: 'bg-orange-500/25 text-orange-200 border-orange-400/50',
          text: 'text-orange-300',
          meter: 'bg-gradient-to-r from-orange-500 to-rose-400'
        };
      case 'STRONG AVOID / DO NOT BUY':
      default:
        return {
          bg: 'from-rose-500/25 via-red-950/40 to-slate-900 border-rose-500/50 shadow-[0_0_20px_rgba(244,63,94,0.15)]',
          badge: 'bg-rose-500/30 text-rose-200 border-rose-400/60 animate-pulse',
          text: 'text-rose-300',
          meter: 'bg-gradient-to-r from-red-600 to-rose-500'
        };
    }
  };

  const actionTheme = getActionTheme(decision?.action_signal || 'HOLD / WATCHLIST');

  return (
    <div className="glass-card p-4 sm:p-6 relative overflow-hidden">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 sm:mb-5">
        <div className="flex items-center gap-2.5">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-blue-500/25 to-cyan-500/20 flex items-center justify-center border border-blue-500/30 shadow-sm">
            <Newspaper className="h-5 w-5 text-blue-400" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h3 className="text-sm sm:text-base font-bold text-white tracking-tight">News Intelligence & Decision Engine</h3>
              <InfoBadge infoKey="news_intelligence" />
            </div>
            <div className="flex items-center gap-2 text-[10px] sm:text-xs text-slate-400">
              <span className="flex items-center gap-1 text-cyan-400">
                <Clock className="h-3 w-3" /> Strict 15-Day Limit
              </span>
              <span>•</span>
              <span className="text-slate-400">Recency Decay Weighted</span>
            </div>
          </div>
        </div>
        
        <button
          onClick={fetchAdvancedNews}
          disabled={loading}
          className="p-2 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] active:bg-slate-800 text-slate-400 hover:text-white transition disabled:opacity-40 cursor-pointer border border-white/[0.06]"
          title="Refresh Live News"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin text-blue-400' : ''}`} />
        </button>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-12 text-slate-400">
          <div className="relative">
            <div className="w-12 h-12 rounded-full border-2 border-blue-500/20 border-t-blue-400 animate-spin" />
            <Newspaper className="h-5 w-5 absolute inset-0 m-auto text-blue-400 animate-pulse" />
          </div>
          <span className="text-xs font-medium text-slate-300 mt-3">Scraping live 15-day financial catalysts...</span>
          <span className="text-[10px] text-slate-500 mt-1">Filtering out stale news & applying time-decay weighting</span>
        </div>
      ) : error ? (
        <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 text-sm">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : newsData ? (
        <>
          {/* ======================================================== */}
          {/* 1. DECISION COMMAND CENTER: "BUY OR NOT" ACTION DIRECTIVE */}
          {/* ======================================================== */}
          {decision && (
            <div className={`mb-5 p-4 rounded-2xl border bg-gradient-to-br ${actionTheme.bg} transition-all duration-300 shadow-lg`}>
              
              {/* Top Banner: Action Badge & Decision Score */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3.5 pb-3.5 border-b border-white/[0.08]">
                <div>
                  <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                    <span className="text-[10px] font-black uppercase tracking-wider text-slate-300 opacity-90 flex items-center gap-1">
                      <Flame className="h-3 w-3 text-amber-400" /> Decisive Action Directive
                    </span>
                    <span className={`text-xs font-black px-3 py-1 rounded-full border uppercase tracking-wider ${actionTheme.badge}`}>
                      {decision.action_signal}
                    </span>
                  </div>
                  <p className="text-xs text-white/90 font-medium">
                    {decision.trade_directive}
                  </p>
                </div>

                {/* Score & Conviction Pill */}
                <div className="flex items-center gap-3 shrink-0">
                  <div className="text-right">
                    <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Decision Score</div>
                    <div className="text-lg font-black font-mono text-white flex items-baseline justify-end gap-1">
                      <span>{decision.decision_score}</span>
                      <span className="text-[10px] text-slate-400 font-normal">/ 100</span>
                    </div>
                  </div>
                  <div className="h-9 w-px bg-white/[0.1]" />
                  <div className="text-left">
                    <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Conviction</div>
                    <div className="text-xs font-bold font-mono text-emerald-400">
                      {decision.conviction_pct || decision.conviction_score}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Decision Score Bar */}
              <div className="mb-3.5">
                <div className="flex justify-between items-center text-[10px] text-slate-400 mb-1">
                  <span>0 (Strong Avoid)</span>
                  <span className="font-semibold text-slate-300 font-mono">Current: {decision.decision_score} pts</span>
                  <span>100 (Strong Buy)</span>
                </div>
                <div className="w-full h-2 rounded-full bg-slate-900/80 p-0.5 border border-white/[0.08] overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-700 ${actionTheme.meter}`}
                    style={{ width: `${Math.max(4, Math.min(100, decision.decision_score))}%` }}
                  />
                </div>
              </div>

              {/* Risk Gate VETO Status Box */}
              {decision.is_buy_vetoed ? (
                <div className="p-3 mb-3 rounded-xl bg-rose-500/20 border border-rose-500/40 flex items-start gap-2.5 text-rose-200">
                  <ShieldAlert className="h-5 w-5 text-rose-400 shrink-0 mt-0.5 animate-pulse" />
                  <div>
                    <span className="text-xs font-bold text-white block">HARD RISK GATE VETO ACTIVATED</span>
                    <span className="text-[11px] leading-relaxed block text-rose-300 mt-0.5">
                      {decision.veto_reason || 'Severe asymmetric risk catalyst detected in 15-day news flow. Fundamental veto overrides technical buy signals.'}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="px-3 py-2 mb-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-between gap-2 text-emerald-300 text-xs">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0" />
                    <span className="font-semibold">Risk Gate Status: Passed</span>
                  </div>
                  <span className="text-[10px] text-slate-400">0 fraud, SEBI probe, or circuit lock flags</span>
                </div>
              )}

              {/* Plain-English Executive Rationale */}
              {decision.executive_rationale && (
                <div className="p-3 rounded-xl bg-black/30 border border-white/[0.07] text-xs leading-relaxed text-slate-200 mb-3">
                  <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-cyan-400 mb-1">
                    <Sparkles className="h-3 w-3 text-cyan-400" />
                    <span>Executive Trading Rationale</span>
                  </div>
                  {decision.executive_rationale}
                </div>
              )}

              {/* Concrete Decision Drivers */}
              {decision.decision_drivers && decision.decision_drivers.length > 0 && (
                <div className="space-y-1">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
                    Key Decision Drivers (15-Day Window)
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                    {decision.decision_drivers.map((driver, dIdx) => (
                      <div key={dIdx} className="text-[11px] text-slate-300 bg-white/[0.03] border border-white/[0.05] rounded-lg px-2.5 py-1.5 flex items-center gap-2">
                        <CheckCircle2 className="h-3 w-3 text-cyan-400 shrink-0" />
                        <span className="truncate">{driver}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ======================================================== */}
          {/* 2. RECENT NEWS VELOCITY & 15-DAY FILTER CONTROL BAR       */}
          {/* ======================================================== */}
          <div className="p-3 mb-5 rounded-xl bg-slate-900/60 border border-white/[0.06]">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-2.5">
              <div className="flex items-center gap-2">
                <Calendar className="h-4 w-4 text-indigo-400" />
                <span className="text-xs font-bold text-white">15-Day Recency Velocity</span>
                <span className="text-[10px] font-mono text-slate-400">
                  ({recencyDist.total_recent || articles.length} parsed / {recencyDist.stale_discarded || 0} discarded &gt;15d)
                </span>
              </div>

              {/* Recency Time Filter Tabs */}
              <div className="flex items-center gap-1 bg-black/40 p-1 rounded-lg border border-white/[0.05] text-[10px] overflow-x-auto max-w-full">
                {[
                  { id: 'ALL', label: `15d Horizon (${articles.length})` },
                  { id: '24h', label: `<24h Breaking (${recencyDist.within_24h || 0})` },
                  { id: '3d', label: `1-3 Days (${(recencyDist.within_24h || 0) + (recencyDist.within_3d || 0)})` },
                  { id: '7d', label: `4-7 Days (${recencyDist.within_7d || 0})` },
                ].map(rf => (
                  <button
                    key={rf.id}
                    onClick={() => setRecencyFilter(rf.id)}
                    className={`px-2 py-0.5 rounded font-semibold transition cursor-pointer whitespace-nowrap ${
                      recencyFilter === rf.id 
                        ? 'bg-indigo-500/25 text-indigo-300 border border-indigo-500/40' 
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {rf.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Recency Breakdown Micro Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-white/[0.04] text-center">
              <div className="p-1.5 rounded-lg bg-white/[0.02]">
                <div className="text-[9px] text-slate-400">Within 24h</div>
                <div className="text-xs font-bold text-cyan-400 font-mono">{recencyDist.within_24h || 0} stories</div>
                <div className="text-[8px] text-slate-500">Weight: 1.00x</div>
              </div>
              <div className="p-1.5 rounded-lg bg-white/[0.02]">
                <div className="text-[9px] text-slate-400">1 - 3 Days</div>
                <div className="text-xs font-bold text-blue-400 font-mono">{recencyDist.within_3d || 0} stories</div>
                <div className="text-[8px] text-slate-500">Weight: 0.85x</div>
              </div>
              <div className="p-1.5 rounded-lg bg-white/[0.02]">
                <div className="text-[9px] text-slate-400">4 - 7 Days</div>
                <div className="text-xs font-bold text-slate-300 font-mono">{recencyDist.within_7d || 0} stories</div>
                <div className="text-[8px] text-slate-500">Weight: 0.55x</div>
              </div>
              <div className="p-1.5 rounded-lg bg-white/[0.02]">
                <div className="text-[9px] text-slate-400">8 - 15 Days</div>
                <div className="text-xs font-bold text-slate-400 font-mono">{recencyDist.within_15d || 0} stories</div>
                <div className="text-[8px] text-slate-500">Weight: 0.25x</div>
              </div>
            </div>
          </div>

          {/* ======================================================== */}
          {/* 3. SENTIMENT METRICS OVERVIEW                            */}
          {/* ======================================================== */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
            <div className="text-center p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="flex items-center justify-center gap-1 mb-1">
                {getSentimentIcon(newsData.sentiment.overall_sentiment)}
                <span className={`font-bold text-sm font-mono ${getSentimentColor(newsData.sentiment.overall_sentiment)}`}>
                  {newsData.sentiment.overall_sentiment > 0 ? '+' : ''}{newsData.sentiment.overall_sentiment}
                </span>
              </div>
              <p className="text-[10px] text-slate-400">Weighted Sentiment</p>
              <p className="text-[9px] text-slate-500 mt-0.5">{newsData.sentiment.sentiment_label}</p>
            </div>
            
            <div className="text-center p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="flex items-center justify-center gap-1 mb-1">
                <BarChart3 className="h-3 w-3 text-indigo-400" />
                <span className="font-bold text-sm text-indigo-400 font-mono">
                  {newsData.sentiment.market_impact_score}
                </span>
              </div>
              <p className="text-[10px] text-slate-400">Market Impact</p>
              <p className="text-[9px] text-slate-500 mt-0.5">0-100 scale</p>
            </div>
            
            <div className="text-center p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="flex items-center justify-center gap-1 mb-1">
                <TrendingUp className="h-3 w-3 text-emerald-400" />
                <span className="font-bold text-sm text-emerald-400 font-mono">
                  {positiveCount}
                </span>
              </div>
              <p className="text-[10px] text-slate-400">Bullish Stories</p>
              <p className="text-[9px] text-slate-500 mt-0.5">within 15 days</p>
            </div>
            
            <div className="text-center p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
              <div className="flex items-center justify-center gap-1 mb-1">
                <TrendingDown className="h-3 w-3 text-rose-400" />
                <span className="font-bold text-sm text-rose-400 font-mono">
                  {negativeCount}
                </span>
              </div>
              <p className="text-[10px] text-slate-400">Bearish Stories</p>
              <p className="text-[9px] text-slate-500 mt-0.5">within 15 days</p>
            </div>
          </div>

          {/* ======================================================== */}
          {/* 4. EXTRACTED CORPORATE CATALYSTS                         */}
          {/* ======================================================== */}
          {newsData.catalysts && newsData.catalysts.length > 0 && (
            <div className="mb-5">
              <div className="flex items-center gap-2 mb-2.5">
                <Sparkles className="h-4 w-4 text-emerald-400" />
                <h4 className="font-bold text-xs uppercase tracking-wider text-emerald-400">Identified Corporate Catalysts</h4>
                <InfoBadge
                  title="Corporate Catalysts"
                  what="Automated extraction of high-impact events: contract orders, earnings revisions, capacity expansion, or litigation."
                  why="Fundamental catalysts trigger immediate institutional re-rating and volume spikes."
                  interpretation="Bullish catalysts trigger accumulation; management or regulatory flags warrant defensive positioning."
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {newsData.catalysts.map((cat, cidx) => (
                  <div key={cidx} className={`p-2.5 rounded-xl border flex items-start justify-between gap-2 ${
                    cat.polarity > 0 ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300' :
                    cat.polarity < 0 ? 'bg-rose-500/10 border-rose-500/20 text-rose-300' :
                    'bg-slate-500/10 border-slate-500/20 text-slate-300'
                  }`}>
                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-wider opacity-75">{cat.type}</div>
                      <div className="text-xs font-semibold mt-0.5">{cat.highlight}</div>
                    </div>
                    <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-black/40 shrink-0 font-mono">
                      {cat.impact}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ======================================================== */}
          {/* 5. BREAKING / HIGH-IMPACT STORIES                        */}
          {/* ======================================================== */}
          {newsData.breaking_news && newsData.breaking_news.length > 0 && (
            <div className="mb-5">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="h-4 w-4 text-amber-400" />
                <h4 className="font-bold text-xs uppercase tracking-wider text-amber-400">High-Impact / Breaking Stories</h4>
              </div>
              
              <div className="space-y-2">
                {newsData.breaking_news.slice(0, 3).map((item, idx) => (
                  <div key={idx} className={`p-3 rounded-xl border ${getImpactColor(item.impact_score)}`}>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-xs leading-tight mb-1">{item.title}</p>
                        <div className="flex items-center gap-2 text-[9px] text-slate-400">
                          <Clock className="h-2.5 w-2.5" />
                          <span>{formatTimeAgo(item.published)}</span>
                          <span>•</span>
                          <span className="font-semibold">{item.urgency} Urgency</span>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="font-bold text-xs font-mono">{item.impact_score}</div>
                        <div className="text-[9px] text-slate-500">impact</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ======================================================== */}
          {/* 6. 15-DAY FILTERED ARTICLES FEED                         */}
          {/* ======================================================== */}
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2">
                <h4 className="font-bold text-xs uppercase tracking-wider text-slate-300">Filtered 15-Day Feed</h4>
                <span className="text-[10px] text-slate-400 font-mono">({filteredArticles.length} shown)</span>
              </div>

              {/* Sentiment Filter Pills */}
              <div className="flex items-center gap-1 p-0.5 rounded-lg bg-white/[0.02] border border-white/[0.05] text-[10px]">
                {[
                  { id: 'ALL', label: `All (${articles.length})` },
                  { id: 'POSITIVE', label: `Bullish (${positiveCount})`, activeColor: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30' },
                  { id: 'NEGATIVE', label: `Bearish (${negativeCount})`, activeColor: 'text-rose-400 bg-rose-500/15 border-rose-500/30' },
                  { id: 'NEUTRAL', label: `Neutral (${neutralCount})`, activeColor: 'text-slate-300 bg-slate-500/15 border-slate-500/30' },
                ].map(f => (
                  <button
                    key={f.id}
                    onClick={() => setSentimentFilter(f.id)}
                    className={`px-2 py-0.5 rounded border transition cursor-pointer font-medium ${
                      sentimentFilter === f.id
                        ? f.activeColor || 'bg-white/[0.1] text-white font-bold border-white/20'
                        : 'border-transparent text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>
            
            {filteredArticles.length > 0 ? (
              <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                {filteredArticles.slice(0, 10).map((article, idx) => (
                  <a
                    key={idx}
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.06] hover:border-white/[0.12] transition group"
                  >
                    <div className="flex items-start gap-2.5">
                      <div className={`mt-1 h-2 w-2 rounded-full shrink-0 ${
                        article.sentiment > 0.1 ? 'bg-emerald-400' : article.sentiment < -0.1 ? 'bg-rose-400' : 'bg-slate-500'
                      }`} />
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1 flex-wrap">
                          <p className="font-semibold text-xs text-slate-200 group-hover:text-white leading-tight">
                            {article.title}
                          </p>
                        </div>
                        <p className="text-[10px] text-slate-400 leading-relaxed mb-2 line-clamp-2">
                          {article.summary}
                        </p>
                        {article.catalysts && article.catalysts.length > 0 && (
                          <div className="flex flex-wrap gap-1 mb-2">
                            {article.catalysts.map((c, i) => (
                              <span key={i} className="text-[8px] font-semibold px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                                {c}
                              </span>
                            ))}
                          </div>
                        )}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-[9px] text-slate-500">
                            <span className="font-medium text-slate-400">{article.source}</span>
                            <span>•</span>
                            <span className="flex items-center gap-1">
                              <Clock className="h-2.5 w-2.5" />
                              {formatTimeAgo(article.published, article.age_hours)}
                            </span>
                            {article.recency_bucket && (
                              <span className={`px-1.5 py-0.2 rounded font-mono font-bold text-[8px] ${
                                article.recency_bucket === '24h' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' :
                                article.recency_bucket === '3d' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' :
                                'bg-slate-800 text-slate-400'
                              }`}>
                                {article.recency_bucket === '24h' ? '<24h (1.0x)' : `${article.recency_bucket} (${article.recency_weight || 1}x)`}
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className={`text-[10px] font-bold font-mono ${getSentimentColor(article.sentiment)}`}>
                              {article.sentiment > 0 ? '+' : ''}{article.sentiment}
                            </span>
                            <ExternalLink className="h-3 w-3 text-slate-500 group-hover:text-slate-300 transition" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </a>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-slate-500">
                <Newspaper className="h-8 w-8 mx-auto mb-2 opacity-30" />
                <p className="text-xs">No articles matching active filters within 15 days</p>
              </div>
            )}
          </div>

          {/* Bottom Update Bar */}
          <div className="mt-4 pt-3 border-t border-white/[0.06] flex justify-between items-center text-[9px] text-slate-500">
            <span className="flex items-center gap-1">
              <ShieldCheck className="h-3 w-3 text-emerald-500" />
              15-day strict limit verified • Scrapling live catalyst reader
            </span>
            <span>
              Updated {new Date(newsData.last_updated).toLocaleTimeString('en-IN', { 
                hour: '2-digit', minute: '2-digit' 
              })}
            </span>
          </div>
        </>
      ) : (
        <div className="text-center py-8 text-slate-500">
          <Newspaper className="h-8 w-8 mx-auto mb-2 opacity-30" />
          <p className="text-sm">Click refresh to analyze 15-day news</p>
        </div>
      )}
    </div>
  );
}