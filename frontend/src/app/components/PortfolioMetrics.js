'use client';

import { useState, useEffect } from 'react';
import { 
  Shield, 
  RefreshCw, 
  AlertTriangle, 
  TrendingUp, 
  TrendingDown,
  BarChart3,
  Calculator,
  Target,
  Activity
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

export default function PortfolioMetrics({ ticker }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPortfolioMetrics = async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${API_BASE_URL}/api/portfolio-metrics?ticker=${ticker}`);
      const json = await res.json();
      
      if (!res.ok) throw new Error(json.detail || 'Failed to fetch portfolio metrics');
      
      setMetrics(json);
    } catch (err) {
      setError(err.message);
      setMetrics(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolioMetrics();
  }, [ticker]);

  const getRiskColor = (value, thresholds = { good: 1, moderate: 2 }) => {
    const absValue = Math.abs(value);
    if (absValue <= thresholds.good) return 'text-emerald-400';
    if (absValue <= thresholds.moderate) return 'text-yellow-400';
    return 'text-rose-400';
  };

  const getPerformanceColor = (value) => {
    if (value > 1) return 'text-emerald-400';
    if (value > 0.5) return 'text-yellow-400';
    return 'text-rose-400';
  };

  return (
    <div className="glass-card p-4 sm:p-6">
      
      {/* Header */}
      <div className="flex items-center justify-between mb-4 sm:mb-5">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center border border-indigo-500/20">
            <Shield className="h-4 w-4 text-indigo-400" />
          </div>
          <div>
            <h3 className="text-sm sm:text-base font-bold text-white">Portfolio Analytics</h3>
            <p className="text-[10px] sm:text-xs text-slate-400">Advanced Risk & Performance Metrics</p>
          </div>
        </div>
        
        <button
          onClick={fetchPortfolioMetrics}
          disabled={loading}
          className="p-2 rounded-lg active:bg-slate-800 text-slate-500 active:text-slate-300 transition disabled:opacity-40 cursor-pointer"
          title="Refresh Metrics"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8 text-slate-500">
          <div className="flex items-center gap-2">
            <Calculator className="h-5 w-5 animate-pulse" />
            <span className="text-sm">Calculating portfolio metrics...</span>
          </div>
        </div>
      ) : error ? (
        <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-lg text-rose-400 text-sm">
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <span>{error}</span>
        </div>
      ) : metrics ? (
        <div className="space-y-5">
          
          {/* Risk Metrics */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="h-4 w-4 text-orange-400" />
              <h4 className="font-bold text-sm text-orange-400">Risk Assessment</h4>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3">
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className={`font-bold text-sm mb-1 ${getRiskColor(metrics.risk_metrics.var_95_daily, {good: 2, moderate: 4})}`}>
                  {metrics.risk_metrics.var_95_daily}%
                </div>
                <p className="text-[10px] text-slate-500">VaR 95% (1D)</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Daily risk</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className={`font-bold text-sm mb-1 ${getRiskColor(metrics.risk_metrics.var_99_daily, {good: 3, moderate: 6})}`}>
                  {metrics.risk_metrics.var_99_daily}%
                </div>
                <p className="text-[10px] text-slate-500">VaR 99% (1D)</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Extreme risk</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className={`font-bold text-sm mb-1 ${getRiskColor(metrics.risk_metrics.max_drawdown, {good: 10, moderate: 25})}`}>
                  {metrics.risk_metrics.max_drawdown}%
                </div>
                <p className="text-[10px] text-slate-500">Max Drawdown</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Peak-to-trough</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className="font-bold text-sm mb-1 text-slate-200">
                  {metrics.risk_metrics.annual_volatility}%
                </div>
                <p className="text-[10px] text-slate-500">Annual Vol</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Price volatility</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className="font-bold text-sm mb-1 text-slate-200">
                  {metrics.risk_metrics.skewness}
                </div>
                <p className="text-[10px] text-slate-500">Skewness</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Return asymmetry</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className={`font-bold text-sm mb-1 ${getPerformanceColor(metrics.risk_metrics.sharpe_ratio)}`}>
                  {metrics.risk_metrics.sharpe_ratio}
                </div>
                <p className="text-[10px] text-slate-500">Sharpe Ratio</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Risk-adj return</p>
              </div>
            </div>
          </div>

          {/* Market Metrics */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="h-4 w-4 text-blue-400" />
              <h4 className="font-bold text-sm text-blue-400">Market Relationship</h4>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className={`font-bold text-sm mb-1 ${metrics.market_metrics.beta !== null ? (Math.abs(metrics.market_metrics.beta - 1) < 0.3 ? 'text-emerald-400' : Math.abs(metrics.market_metrics.beta - 1) < 0.7 ? 'text-yellow-400' : 'text-rose-400') : 'text-slate-400'}`}>
                  {metrics.market_metrics.beta || 'N/A'}
                </div>
                <p className="text-[10px] text-slate-500">Beta</p>
                <p className="text-[9px] text-slate-600 mt-0.5">vs Nifty</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center">
                <div className={`font-bold text-sm mb-1 ${metrics.market_metrics.correlation_with_nifty !== null ? (Math.abs(metrics.market_metrics.correlation_with_nifty) > 0.7 ? 'text-yellow-400' : 'text-emerald-400') : 'text-slate-400'}`}>
                  {metrics.market_metrics.correlation_with_nifty || 'N/A'}
                </div>
                <p className="text-[10px] text-slate-500">Correlation</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Market sync</p>
              </div>
              
              <div className="p-3 rounded-lg bg-white/[0.03] border border-white/[0.06] text-center col-span-2">
                <div className={`font-bold text-sm mb-1 ${metrics.market_metrics.information_ratio !== null ? getPerformanceColor(Math.abs(metrics.market_metrics.information_ratio)) : 'text-slate-400'}`}>
                  {metrics.market_metrics.information_ratio || 'N/A'}
                </div>
                <p className="text-[10px] text-slate-500">Information Ratio</p>
                <p className="text-[9px] text-slate-600 mt-0.5">Active management skill</p>
              </div>
            </div>
          </div>

          {/* Options Pricing */}
          {metrics.options_pricing && !metrics.options_pricing.error && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Target className="h-4 w-4 text-purple-400" />
                <h4 className="font-bold text-sm text-purple-400">Options Analysis</h4>
                <span className="text-[10px] text-slate-500 px-2 py-0.5 rounded bg-slate-800">Black-Scholes</span>
              </div>
              
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
                <div className="p-3 bg-purple-500/5 rounded-lg border border-purple-500/20 text-center">
                  <div className="font-bold text-sm mb-1 text-purple-300">
                    Rs.{metrics.options_pricing.call_price}
                  </div>
                  <p className="text-[10px] text-slate-500">Call Price</p>
                  <p className="text-[9px] text-slate-600 mt-0.5">ATM, 30D</p>
                </div>
                
                <div className="p-3 bg-purple-500/5 rounded-lg border border-purple-500/20 text-center">
                  <div className="font-bold text-sm mb-1 text-purple-300">
                    Rs.{metrics.options_pricing.put_price}
                  </div>
                  <p className="text-[10px] text-slate-500">Put Price</p>
                  <p className="text-[9px] text-slate-600 mt-0.5">ATM, 30D</p>
                </div>
                
                <div className="p-3 bg-purple-500/5 rounded-lg border border-purple-500/20 text-center">
                  <div className="font-bold text-sm mb-1 text-purple-300">
                    {metrics.options_pricing.delta}
                  </div>
                  <p className="text-[10px] text-slate-500">Delta</p>
                  <p className="text-[9px] text-slate-600 mt-0.5">Price sensitivity</p>
                </div>
                
                <div className="p-3 bg-purple-500/5 rounded-lg border border-purple-500/20 text-center">
                  <div className="font-bold text-sm mb-1 text-purple-300">
                    {metrics.options_pricing.vega}
                  </div>
                  <p className="text-[10px] text-slate-500">Vega</p>
                  <p className="text-[9px] text-slate-600 mt-0.5">Vol sensitivity</p>
                </div>
              </div>
            </div>
          )}

          {/* Risk Distribution */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Activity className="h-4 w-4 text-cyan-400" />
              <h4 className="font-bold text-sm text-cyan-400">Risk Distribution</h4>
            </div>
            
            <div className="p-4 rounded-lg bg-white/[0.03] border border-white/[0.06]">
              <div className="flex justify-between text-xs mb-2">
                <span className="text-slate-400">Expected Shortfall (Tail Risk)</span>
                <span className="text-slate-300">95% | 99%</span>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-3">
                <div className="text-center">
                  <div className={`text-lg font-bold mb-1 ${getRiskColor(metrics.risk_metrics.expected_shortfall_95, {good: 3, moderate: 6})}`}>
                    {metrics.risk_metrics.expected_shortfall_95}%
                  </div>
                  <p className="text-[10px] text-slate-500">ES 95%</p>
                </div>
                
                <div className="text-center">
                  <div className={`text-lg font-bold mb-1 ${getRiskColor(metrics.risk_metrics.expected_shortfall_99, {good: 5, moderate: 10})}`}>
                    {metrics.risk_metrics.expected_shortfall_99}%
                  </div>
                  <p className="text-[10px] text-slate-500">ES 99%</p>
                </div>
              </div>
              
              <div className="text-center">
                <p className="text-[11px] text-slate-500 leading-relaxed">
                  Average loss when VaR threshold is breached. Higher values indicate fat-tail risk.
                </p>
              </div>
            </div>
          </div>

          {/* Update timestamp */}
          <div className="text-center pt-2 border-t border-slate-800/70">
            <p className="text-[9px] text-slate-600">
              Portfolio analytics · Updated {metrics.last_updated}
            </p>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-slate-500">
          <Shield className="h-8 w-8 mx-auto mb-2 opacity-30" />
          <p className="text-sm">Click refresh to calculate metrics</p>
        </div>
      )}
    </div>
  );
}