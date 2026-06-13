'use client';

import { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, Target, Zap, AlertCircle, RefreshCw } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

export default function MLPrediction({ ticker }) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState('2y');
  const [predictionCache, setPredictionCache] = useState({});

  // Dynamic default dates for custom range
  const [startDate, setStartDate] = useState(() => {
    const today = new Date();
    const twoYearsAgo = new Date(today.getFullYear() - 2, today.getMonth(), today.getDate());
    return twoYearsAgo.toISOString().split('T')[0];
  });
  const [endDate, setEndDate] = useState(() => {
    return new Date().toISOString().split('T')[0];
  });

  const fetchPrediction = async (period = selectedPeriod, start = startDate, end = endDate, isBackground = false) => {
    if (!ticker) return;
    if (!isBackground) {
      setLoading(true);
      setError(null);
    }
    
    try {
      let url = `${API_BASE_URL}/api/ml-predict?ticker=${ticker}`;
      if (period === 'custom') {
         url += `&period=custom&start_date=${start}&end_date=${end}`;
      } else {
         url += `&period=${period}`;
      }
      const res = await fetch(url);
      const json = await res.json();
      
      if (!res.ok) throw new Error(json.detail || 'Failed to get prediction');
      
      // Update cache key
      const cacheKey = period === 'custom' 
        ? `${ticker}_custom_${start}_${end}` 
        : `${ticker}_${period}`;
        
      setPredictionCache(prev => ({
        ...prev,
        [cacheKey]: json.prediction
      }));
      
      const isCurrentActive = period === selectedPeriod && 
        (period !== 'custom' || (start === startDate && end === endDate));

      if (!isBackground || isCurrentActive) {
        setPrediction(json.prediction);
      }
    } catch (err) {
      if (!isBackground) {
        setError(err.message);
        setPrediction(null);
      }
    } finally {
      if (!isBackground) {
        setLoading(false);
      }
    }
  };

  // 1. Ticker changes: reset cache
  useEffect(() => {
    if (!ticker) return;
    setPredictionCache({});
  }, [ticker]);

  // 2. Period changes: check cache or trigger fetch
  useEffect(() => {
    if (!ticker) return;
    
    const isCustom = selectedPeriod === 'custom';
    const cacheKey = isCustom 
      ? `${ticker}_custom_${startDate}_${endDate}` 
      : `${ticker}_${selectedPeriod}`;
      
    if (predictionCache[cacheKey]) {
      setPrediction(predictionCache[cacheKey]);
      setError(null);
    } else {
      fetchPrediction(selectedPeriod, startDate, endDate, false);
    }
  }, [ticker, selectedPeriod]);

  const handlePeriodChange = (p) => {
    setSelectedPeriod(p);
  };

  const handleRefresh = async () => {
    setPredictionCache({});
    await fetchPrediction(selectedPeriod, startDate, endDate, false);
  };

  const getSignalStyle = (signal) => {
    switch (signal) {
      case 'STRONG BUY':
        return { bg: 'bg-emerald-500/20', border: 'border-emerald-500/40', text: 'text-emerald-400', dot: 'bg-emerald-400' };
      case 'BUY':
        return { bg: 'bg-emerald-500/15', border: 'border-emerald-500/30', text: 'text-emerald-400', dot: 'bg-emerald-400' };
      case 'STRONG SELL':
        return { bg: 'bg-rose-500/20', border: 'border-rose-500/40', text: 'text-rose-400', dot: 'bg-rose-400' };
      case 'SELL':
        return { bg: 'bg-rose-500/15', border: 'border-rose-500/30', text: 'text-rose-400', dot: 'bg-rose-400' };
      default:
        return { bg: 'bg-amber-500/15', border: 'border-amber-500/30', text: 'text-amber-400', dot: 'bg-amber-400' };
    }
  };

  const formatPeriodLabel = (p) => {
    if (p === '1y') return '1 Year';
    if (p === '2y') return '2 Years';
    if (p === '5y') return '5 Years';
    if (p === 'max') return 'Max History';
    if (p === 'custom') return `Custom Range (${startDate} to ${endDate})`;
    return p;
  };

  const getStabilityStyle = (stability) => {
    switch (stability) {
      case 'STABLE CONSENSUS':
        return 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5';
      case 'MODERATE VOLATILITY':
        return 'text-amber-400 border-amber-500/20 bg-amber-500/5';
      case 'HIGH UNCERTAINTY':
        return 'text-rose-400 border-rose-500/20 bg-rose-500/5';
      default:
        return 'text-slate-400 border-white/[0.06] bg-white/[0.02]';
    }
  };

  return (
    <div className="glass-card p-4 sm:p-6">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 sm:mb-5">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-purple-500/20 to-indigo-500/20 flex items-center justify-center border border-purple-500/20">
            <Brain className="h-4 w-4 text-purple-400" />
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white">AI Price Prediction</h3>
            <p className="text-[10px] sm:text-xs text-slate-400">Machine Learning Forecast</p>
          </div>
        </div>
        
        <button
          onClick={handleRefresh}
          disabled={loading}
          className="p-2 rounded-lg active:bg-slate-800 text-slate-500 active:text-slate-300 transition disabled:opacity-40 cursor-pointer"
          title="Refresh Prediction"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Training Period Selector */}
      <div className="mb-5">
        <p className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Training Data Horizon</p>
        <div className="flex p-0.5 bg-white/[0.03] rounded-lg border border-white/[0.06] gap-1">
          {['1y', '2y', '5y', 'max', 'custom'].map((p) => (
            <button
              key={p}
              onClick={() => handlePeriodChange(p)}
              disabled={loading}
              className={`flex-1 py-1.5 text-[11px] font-semibold rounded-md transition ${
                selectedPeriod === p
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              } disabled:opacity-50`}
            >
              {p === 'max' ? 'Max' : p === 'custom' ? 'Custom' : p.toUpperCase()}
            </button>
          ))}
        </div>

        {selectedPeriod === 'custom' && (
          <div className="mt-3 flex flex-col gap-3 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] animate-fadeIn">
            <div className="flex gap-2">
              <div className="flex-1">
                <label className="text-[9px] text-slate-500 font-bold block mb-1">START DATE</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full bg-white/[0.05] border border-white/[0.08] text-white rounded px-2 py-1 text-xs focus:outline-none focus:border-purple-500"
                />
              </div>
              <div className="flex-1">
                <label className="text-[9px] text-slate-500 font-bold block mb-1">END DATE</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full bg-white/[0.05] border border-white/[0.08] text-white rounded px-2 py-1 text-xs focus:outline-none focus:border-purple-500"
                />
              </div>
            </div>
            <button
              onClick={() => fetchPrediction('custom', startDate, endDate)}
              disabled={loading}
              className="w-full bg-purple-600 hover:bg-purple-700 active:bg-purple-800 text-white font-bold py-1.5 rounded text-xs transition cursor-pointer disabled:opacity-50"
            >
              {loading ? 'Retraining...' : 'Train Model on Selected Dates'}
            </button>
          </div>
        )}
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-12 text-slate-500 gap-3">
          <Brain className="h-6 w-6 animate-pulse text-purple-400" />
          <span className="text-xs text-slate-400">Retraining model with {formatPeriodLabel(selectedPeriod)} data...</span>
        </div>
      ) : error ? (
        <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm">
          <AlertCircle className="h-5 w-5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : prediction ? (
        <>
          {/* Main Prediction */}
          <div className={`rounded-xl sm:rounded-2xl p-4 sm:p-5 border mb-4 ${getSignalStyle(prediction.signal).bg} ${getSignalStyle(prediction.signal).border}`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className={`h-2 w-2 rounded-full ${getSignalStyle(prediction.signal).dot} animate-pulse`} />
                <span className={`font-bold text-sm sm:text-base ${getSignalStyle(prediction.signal).text}`}>
                  {prediction.signal}
                </span>
              </div>
              <div className="text-right">
                <p className="text-[10px] sm:text-xs text-slate-400">5-Day Target</p>
                <p className="font-bold text-lg sm:text-xl text-white">Rs.{prediction.predicted_price?.toLocaleString()}</p>
              </div>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 sm:gap-4">
              <div className="text-center">
                <div className="flex items-center justify-center gap-1 mb-1">
                  {prediction.predicted_return >= 0 ? (
                    <TrendingUp className="h-3 sm:h-4 w-3 sm:w-4 text-emerald-400" />
                  ) : (
                    <TrendingDown className="h-3 sm:h-4 w-3 sm:w-4 text-rose-400" />
                  )}
                  <span className={`font-bold text-sm sm:text-base ${prediction.predicted_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {prediction.predicted_return >= 0 ? '+' : ''}{prediction.predicted_return}%
                  </span>
                </div>
                <p className="text-[10px] sm:text-xs text-slate-500">Expected Return</p>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center gap-1 mb-1">
                  <Target className="h-3 sm:h-4 w-3 sm:w-4 text-indigo-400" />
                  <span className="font-bold text-sm sm:text-base text-indigo-400">
                    {prediction.confidence}%
                  </span>
                </div>
                <p className="text-[10px] sm:text-xs text-slate-500">Confidence</p>
              </div>
              
              <div className="text-center col-span-2 sm:col-span-1">
                <div className="flex items-center justify-center gap-1 mb-1">
                  <Zap className="h-3 sm:h-4 w-3 sm:w-4 text-yellow-400" />
                  <span className="font-bold text-sm sm:text-base text-yellow-400">
                    {prediction.signal_strength}/100
                  </span>
                </div>
                <p className="text-[10px] sm:text-xs text-slate-500">Signal Strength</p>
              </div>
            </div>
          </div>

          {/* Prediction Details */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 mb-4 animate-fadeIn">
            <div className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-bold mb-0.5">Current Price</p>
              <p className="font-bold text-xs sm:text-sm text-slate-200">Rs.{prediction.current_price?.toLocaleString()}</p>
            </div>
            
            <div className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-bold mb-0.5">Horizon</p>
              <p className="font-bold text-xs sm:text-sm text-slate-200">{prediction.prediction_horizon_days} Days</p>
            </div>

            <div className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-bold mb-0.5">HMM Regime</p>
              <p className={`font-bold text-[9px] uppercase tracking-wider ${
                prediction.regime === 'LOW_VOLATILITY' ? 'text-emerald-400' :
                prediction.regime === 'MEDIUM_VOLATILITY' ? 'text-indigo-400' :
                prediction.regime === 'HIGH_VOLATILITY' ? 'text-rose-400' : 'text-slate-400'
              }`}>
                {prediction.regime?.replace('_', ' ') || 'SIDEWAYS'}
              </p>
            </div>

            <div className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-bold mb-0.5">GARCH Volatility</p>
              <p className="font-bold text-xs sm:text-sm text-slate-200">
                {prediction.garch_volatility ? `${prediction.garch_volatility}%` : 'N/A'}
              </p>
            </div>

            <div className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-bold mb-0.5">Consensus</p>
              <p className={`font-bold text-[9px] uppercase tracking-wider ${
                prediction.stability === 'STABLE CONSENSUS' ? 'text-emerald-400' :
                prediction.stability === 'MODERATE VOLATILITY' ? 'text-yellow-500' : 'text-rose-400'
              }`}>
                {prediction.stability?.replace(' CONSENSUS', '')?.replace(' VOLATILITY', '') || 'STABLE'}
              </p>
            </div>

            <div className="rounded-lg p-2.5 bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[9px] text-slate-500 uppercase tracking-wider font-bold mb-0.5">Risk/Reward</p>
              <p className={`font-bold text-xs sm:text-sm ${
                prediction.risk_reward_ratio >= 2.0 ? 'text-emerald-400' :
                prediction.risk_reward_ratio >= 1.0 ? 'text-yellow-500' : 'text-rose-400'
              }`}>
                {prediction.risk_reward_ratio}x
              </p>
            </div>
          </div>

          {/* Walk-Forward Validation Metrics */}
          <div className="mb-4 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
            <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Walk-Forward Backtest (Out-of-Sample)</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <div className="text-center p-2 rounded bg-white/[0.01] border border-white/[0.04]">
                <span className="text-xs font-bold text-slate-200">{prediction.direction_accuracy}%</span>
                <p className="text-[9px] text-slate-500 mt-0.5">Hit Rate (Dir)</p>
              </div>
              <div className="text-center p-2 rounded bg-white/[0.01] border border-white/[0.04]">
                <span className="text-xs font-bold text-slate-200">{prediction.profit_factor}x</span>
                <p className="text-[9px] text-slate-500 mt-0.5">Profit Factor</p>
              </div>
              <div className="text-center p-2 rounded bg-white/[0.01] border border-white/[0.04]">
                <span className="text-xs font-bold text-emerald-400">+{prediction.avg_win_pct}%</span>
                <p className="text-[9px] text-slate-500 mt-0.5">Avg Win</p>
              </div>
              <div className="text-center p-2 rounded bg-white/[0.01] border border-white/[0.04]">
                <span className="text-xs font-bold text-rose-400">{prediction.avg_loss_pct}%</span>
                <p className="text-[9px] text-slate-500 mt-0.5">Avg Loss</p>
              </div>
            </div>
            {prediction.ensemble_vs_xgb_delta !== undefined && (
              <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between px-1">
                <span>Ensemble Advantage vs Standalone XGBoost:</span>
                <span className={`font-semibold ${prediction.ensemble_vs_xgb_delta >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {prediction.ensemble_vs_xgb_delta >= 0 ? '+' : ''}{prediction.ensemble_vs_xgb_delta}% Direction Accuracy
                </span>
              </div>
            )}
          </div>

          {/* Sentiment Fusion & Risk/Reward */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
            {/* Sentiment Fusion */}
            <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-xs">
              <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Sentiment Fusion Nudges</p>
              <div className="space-y-1.5">
                <div className="flex justify-between">
                  <span className="text-slate-500">Raw ML Model Return:</span>
                  <span className="font-semibold text-slate-300">{prediction.ml_raw_return >= 0 ? '+' : ''}{prediction.ml_raw_return}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">News Sentiment Score:</span>
                  <span className={`font-semibold ${
                    prediction.news_sentiment_used > 0.1 ? 'text-emerald-400' :
                    prediction.news_sentiment_used < -0.1 ? 'text-rose-400' : 'text-slate-400'
                  }`}>
                    {prediction.news_sentiment_used > 0 ? '+' : ''}{prediction.news_sentiment_used}
                  </span>
                </div>
              </div>
            </div>

            {/* Risk Reward */}
            <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-xs">
              <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Volatility & Risk Profile</p>
              <div className="space-y-1.5">
                <div className="flex justify-between">
                  <span className="text-slate-500">GARCH Volatility (5d):</span>
                  <span className="font-semibold text-slate-200">
                    {prediction.garch_volatility ? `${prediction.garch_volatility}% (Ann.)` : 'Calculating...'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Markov Regime State:</span>
                  <span className={`font-semibold ${
                    prediction.regime === 'LOW_VOLATILITY' ? 'text-emerald-400' :
                    prediction.regime === 'MEDIUM_VOLATILITY' ? 'text-indigo-400' :
                    prediction.regime === 'HIGH_VOLATILITY' ? 'text-rose-400' : 'text-slate-300'
                  }`}>
                    {prediction.regime?.replace('_', ' ') || 'SIDEWAYS'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Risk/Reward (ATR-based):</span>
                  <span className={`font-semibold ${prediction.risk_reward_ratio >= 1.5 ? 'text-emerald-400' : 'text-slate-300'}`}>
                    {prediction.risk_reward_ratio}x
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Interpretability (SHAP / Feature Importance) */}
          {prediction.top_features && Object.keys(prediction.top_features).length > 0 && (
            <div className="mb-4 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06]">
              <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">
                Feature Influence ({prediction.shap_powered ? 'SHAP Contribution' : 'Tree Importance'})
              </p>
              <div className="space-y-2">
                {Object.entries(prediction.top_features).map(([feat, val]) => {
                  // Normalize value for display bar
                  const pct = Math.min(100, Math.max(1, val * 100));
                  return (
                    <div key={feat} className="text-[11px]">
                      <div className="flex justify-between text-slate-300 mb-0.5">
                        <span className="font-mono">{feat}</span>
                        <span className="text-slate-500">{val.toFixed(4)}</span>
                      </div>
                      <div className="w-full bg-white/[0.04] h-1.5 rounded-full overflow-hidden">
                        <div 
                          className="bg-purple-500 h-full rounded-full" 
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Multi-Model Stacking Breakdown */}
          {prediction.models && (
            <div className="mt-4 p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] animate-fadeIn">
              <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-2">Base Models Stacked Targets</p>
              <div className="space-y-2">
                {Object.entries(prediction.models).map(([name, val]) => (
                  <div key={name} className="flex items-center justify-between text-xs border-b border-white/[0.04] pb-2 last:border-0 last:pb-0">
                    <span className="text-slate-400 font-medium capitalize flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-purple-500" />
                      {name.replace(/_/g, ' ')}
                    </span>
                    <div className="text-right">
                      <span className="font-bold text-white block">Rs.{val.predicted_price.toLocaleString()}</span>
                      <span className={`text-[10px] font-semibold ${val.predicted_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {val.predicted_return >= 0 ? '+' : ''}{val.predicted_return}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div className="mt-4 p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
            <div className="flex items-start gap-2">
              <AlertCircle className="h-4 w-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <p className="text-[11px] sm:text-xs text-amber-300 font-semibold mb-1">Investment Disclaimer</p>
                <p className="text-[10px] sm:text-[11px] text-amber-200/80 leading-relaxed">
                  ML predictions are educational tools based on historical patterns. Past performance doesn't guarantee future results. 
                  Always conduct thorough research and consider consulting financial advisors before making investment decisions.
                </p>
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="mt-3 text-center">
            <p className="text-[9px] sm:text-[10px] text-slate-500">
              Meta-stacked Ensemble ({prediction.models_used?.join(', ').toUpperCase() || 'RF/GB/MLP'}) · Trained on {formatPeriodLabel(selectedPeriod)} · {new Date(prediction.timestamp).toLocaleString('en-IN', { 
                day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' 
              })}
            </p>
          </div>
        </>
      ) : (
        <div className="text-center py-8 text-slate-500">
          <Brain className="h-8 w-8 mx-auto mb-2 opacity-30" />
          <p className="text-sm">Click refresh to generate prediction</p>
        </div>
      )}
    </div>
  );
}