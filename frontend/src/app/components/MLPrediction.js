'use client';

import { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, Target, Zap, AlertCircle, RefreshCw } from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

export default function MLPrediction({ ticker }) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPrediction = async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${API_BASE_URL}/api/ml-predict?ticker=${ticker}`);
      const json = await res.json();
      
      if (!res.ok) throw new Error(json.detail || 'Failed to get prediction');
      
      setPrediction(json.prediction);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPrediction();
  }, [ticker]);

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

  return (
    <div className="bg-[#0d1424] rounded-xl sm:rounded-2xl border border-slate-800 p-4 sm:p-6 shadow-xl">
      
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
          onClick={fetchPrediction}
          disabled={loading}
          className="p-2 rounded-lg active:bg-slate-800 text-slate-500 active:text-slate-300 transition disabled:opacity-40 cursor-pointer"
          title="Refresh Prediction"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8 text-slate-500">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 animate-pulse" />
            <span className="text-sm">Training model & generating prediction...</span>
          </div>
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
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-[#111827]/50 rounded-lg p-3 border border-slate-800/80">
              <p className="text-[10px] sm:text-xs text-slate-500 mb-1">Current Price</p>
              <p className="font-bold text-sm sm:text-base text-slate-200">Rs.{prediction.current_price?.toLocaleString()}</p>
            </div>
            
            <div className="bg-[#111827]/50 rounded-lg p-3 border border-slate-800/80">
              <p className="text-[10px] sm:text-xs text-slate-500 mb-1">Horizon</p>
              <p className="font-bold text-sm sm:text-base text-slate-200">{prediction.prediction_horizon_days} Days</p>
            </div>
          </div>

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
            <p className="text-[9px] sm:text-[10px] text-slate-600">
              Random Forest Model · {new Date(prediction.timestamp).toLocaleString('en-IN', { 
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