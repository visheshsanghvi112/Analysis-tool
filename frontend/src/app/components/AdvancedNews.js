'use client';

import { useState, useEffect } from 'react';
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
  Minus
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || (
  process.env.NODE_ENV === 'production'
    ? 'https://your-backend.vercel.app'
    : 'http://127.0.0.1:8000'
);

export default function AdvancedNews({ ticker, companyName }) {
  const [newsData, setNewsData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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

  const formatTimeAgo = (dateString) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffInHours = Math.floor((now - date) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'Just now';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays}d ago`;
  };

  return (
    <div className="bg-[#0d1424] rounded-xl sm:rounded-2xl border border-slate-800 p-4 sm:p-6 shadow-xl">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 sm:mb-5">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500/20 to-cyan-500/20 flex items-center justify-center border border-blue-500/20">
            <Newspaper className="h-4 w-4 text-blue-400" />
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white">AI News Intelligence</h3>
            <p className="text-[10px] sm:text-xs text-slate-400">Advanced Sentiment & Impact Analysis</p>
          </div>
        </div>
        
        <button
          onClick={fetchAdvancedNews}
          disabled={loading}
          className="p-2 rounded-lg active:bg-slate-800 text-slate-500 active:text-slate-300 transition disabled:opacity-40 cursor-pointer"
          title="Refresh News"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8 text-slate-500">
          <div className="flex items-center gap-2">
            <Newspaper className="h-5 w-5 animate-pulse" />
            <span className="text-sm">Analyzing latest news...</span>
          </div>
        </div>
      ) : error ? (
        <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : newsData ? (
        <>
          {/* Sentiment Overview */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
            <div className="text-center p-3 bg-[#111827]/50 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-center gap-1 mb-1">
                {getSentimentIcon(newsData.sentiment.overall_sentiment)}
                <span className={`font-bold text-sm ${getSentimentColor(newsData.sentiment.overall_sentiment)}`}>
                  {newsData.sentiment.overall_sentiment > 0 ? '+' : ''}{newsData.sentiment.overall_sentiment}
                </span>
              </div>
              <p className="text-[10px] text-slate-500">Overall Sentiment</p>
              <p className="text-[9px] text-slate-600 mt-0.5">{newsData.sentiment.sentiment_label}</p>
            </div>
            
            <div className="text-center p-3 bg-[#111827]/50 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-center gap-1 mb-1">
                <BarChart3 className="h-3 w-3 text-indigo-400" />
                <span className="font-bold text-sm text-indigo-400">
                  {newsData.sentiment.market_impact_score}
                </span>
              </div>
              <p className="text-[10px] text-slate-500">Impact Score</p>
              <p className="text-[9px] text-slate-600 mt-0.5">0-100 scale</p>
            </div>
            
            <div className="text-center p-3 bg-[#111827]/50 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-center gap-1 mb-1">
                <TrendingUp className="h-3 w-3 text-emerald-400" />
                <span className="font-bold text-sm text-emerald-400">
                  {newsData.sentiment.positive_count}
                </span>
              </div>
              <p className="text-[10px] text-slate-500">Positive</p>
            </div>
            
            <div className="text-center p-3 bg-[#111827]/50 rounded-lg border border-slate-800/80">
              <div className="flex items-center justify-center gap-1 mb-1">
                <span className="font-bold text-sm text-rose-400">
                  {newsData.sentiment.negative_count}
                </span>
              </div>
              <p className="text-[10px] text-slate-500">Negative</p>
            </div>
          </div>

          {/* Breaking News */}
          {newsData.breaking_news && newsData.breaking_news.length > 0 && (
            <div className="mb-5">
              <div className="flex items-center gap-2 mb-3">
                <Zap className="h-4 w-4 text-red-400" />
                <h4 className="font-bold text-sm text-red-400">Breaking News Impact</h4>
              </div>
              
              <div className="space-y-2">
                {newsData.breaking_news.slice(0, 3).map((item, idx) => (
                  <div key={idx} className={`p-3 rounded-lg border ${getImpactColor(item.impact_score)}`}>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-xs leading-tight mb-1">{item.title}</p>
                        <div className="flex items-center gap-2 text-[9px] text-slate-500">
                          <Clock className="h-2.5 w-2.5" />
                          <span>{formatTimeAgo(item.published)}</span>
                          <span className="font-semibold">{item.urgency}</span>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="font-bold text-xs">{item.impact_score}</div>
                        <div className="text-[9px] text-slate-500">impact</div>
                      </div>
                    </div>
                    
                    {item.reasons && item.reasons.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {item.reasons.slice(0, 2).map((reason, ridx) => (
                          <span key={ridx} className="text-[8px] px-1.5 py-0.5 rounded bg-slate-800/50 text-slate-400">
                            {reason.split(':')[0]}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Articles */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-bold text-sm text-slate-300">Recent Articles</h4>
              <span className="text-[10px] text-slate-500">{newsData.articles?.length || 0} found</span>
            </div>
            
            {newsData.articles && newsData.articles.length > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {newsData.articles.slice(0, 6).map((article, idx) => (
                  <a
                    key={idx}
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block p-3 rounded-lg bg-[#111827]/50 border border-slate-800/80 active:bg-slate-800/50 transition group"
                  >
                    <div className="flex items-start gap-2">
                      <div className={`mt-1 h-2 w-2 rounded-full shrink-0 ${getSentimentColor(article.sentiment) === 'text-emerald-400' ? 'bg-emerald-400' : getSentimentColor(article.sentiment) === 'text-rose-400' ? 'bg-rose-400' : 'bg-slate-500'}`} />
                      
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-xs text-slate-200 group-hover:text-white leading-tight mb-1">
                          {article.title}
                        </p>
                        <p className="text-[10px] text-slate-500 leading-relaxed mb-2 line-clamp-2">
                          {article.summary}
                        </p>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-[9px] text-slate-600">
                            <span>{article.source}</span>
                            <span>•</span>
                            <span>{formatTimeAgo(article.published)}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <span className={`text-[9px] font-semibold ${getSentimentColor(article.sentiment)}`}>
                              {article.sentiment > 0 ? '+' : ''}{article.sentiment}
                            </span>
                            <ExternalLink className="h-3 w-3 text-slate-600 group-hover:text-slate-400" />
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
                <p className="text-sm">No relevant news found</p>
              </div>
            )}
          </div>

          {/* Summary & Update Time */}
          <div className="mt-4 pt-3 border-t border-slate-800/70">
            <p className="text-[11px] text-slate-500 leading-relaxed mb-2">
              {newsData.summary}
            </p>
            <div className="flex justify-between items-center text-[9px] text-slate-600">
              <span>AI-powered sentiment analysis</span>
              <span>Updated {new Date(newsData.last_updated).toLocaleTimeString('en-IN', { 
                hour: '2-digit', minute: '2-digit' 
              })}</span>
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-8 text-slate-500">
          <Newspaper className="h-8 w-8 mx-auto mb-2 opacity-30" />
          <p className="text-sm">Click refresh to analyze news</p>
        </div>
      )}
    </div>
  );
}