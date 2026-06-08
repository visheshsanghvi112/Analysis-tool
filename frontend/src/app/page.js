'use client';

import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Search, 
  AlertTriangle, 
  Layers, 
  DollarSign, 
  ShieldAlert, 
  Newspaper, 
  RefreshCw,
  GitCompare,
  Percent,
  CheckCircle2
} from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamic import for StockChart to avoid server-side rendering mismatch issues with Recharts
const StockChart = dynamic(() => import('./components/StockChart'), { ssr: false });

export default function Home() {
  const [ticker, setTicker] = useState('HDFCBANK.NS');
  const [searchQuery, setSearchQuery] = useState('HDFCBANK.NS');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  // Peer Comparison states
  const [peersInput, setPeersInput] = useState('SBIN.NS, ICICIBANK.NS');
  const [peersData, setPeersData] = useState([]);
  const [peersLoading, setPeersLoading] = useState(false);
  const [peersError, setPeersError] = useState(null);

  // Fetch full analysis for a ticker
  const fetchAnalysis = async (symbolToFetch = searchQuery) => {
    setLoading(true);
    setError(null);
    try {
      const formattedSymbol = symbolToFetch.trim().toUpperCase();
      let url = `http://127.0.0.1:8000/api/analyze?ticker=${formattedSymbol}`;
      if (startDate) url += `&start_date=${startDate}`;
      if (endDate) url += `&end_date=${endDate}`;

      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.detail || 'Failed to fetch analysis data.');
      }
      setData(json);
      setTicker(formattedSymbol);
    } catch (err) {
      setError(err.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  // Fetch Peer comparison data
  const handleCompare = async () => {
    if (!peersInput.trim()) return;
    setPeersLoading(true);
    setPeersError(null);
    try {
      // Always include current ticker in comparison
      const formattedPeers = `${ticker},${peersInput}`
        .split(',')
        .map(t => t.trim().toUpperCase())
        .filter((v, i, a) => a.indexOf(v) === i) // Unique values
        .join(',');

      const res = await fetch(`http://127.0.0.1:8000/api/compare?tickers=${formattedPeers}`);
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.detail || 'Failed to fetch comparison.');
      }
      setPeersData(json.comparison || []);
    } catch (err) {
      setPeersError(err.message);
    } finally {
      setPeersLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalysis();
  }, []);

  // Format Helper
  const fmt = (val, prefix = '', suffix = '') => {
    if (val === null || val === undefined) return 'N/A';
    return `${prefix}${val.toLocaleString()}${suffix}`;
  };

  const getSignalColor = (signal) => {
    if (signal === 'BUY') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    if (signal === 'SELL') return 'bg-rose-500/20 text-rose-400 border-rose-500/30';
    return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
  };

  const getSentimentColor = (label) => {
    if (label === 'Positive') return 'text-emerald-400';
    if (label === 'Negative') return 'text-rose-400';
    return 'text-slate-400';
  };

  return (
    <main className="min-h-screen bg-[#080c14] text-slate-100 font-sans selection:bg-indigo-500/30">
      {/* Top Navigation */}
      <header className="border-b border-slate-800 bg-[#0d1424]/90 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-emerald-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <span className="font-bold text-white text-lg tracking-wider">VS</span>
            </div>
            <div>
              <h1 className="font-extrabold text-base tracking-tight text-white">Stock Analysis Tool</h1>
              <p className="text-[10px] text-slate-400 font-mono">by Vishesh Sanghvi</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs px-2 py-1 rounded bg-indigo-500/10 text-indigo-300 font-medium border border-indigo-500/20">SEBI Interview Edition</span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Search controls & Inputs */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 bg-[#0d1424] p-5 rounded-2xl border border-slate-800 shadow-xl">
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Ticker Symbol</label>
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="e.g. HDFCBANK.NS"
                className="w-full bg-[#182235] border border-slate-700 rounded-xl px-4 py-2.5 text-sm font-medium focus:outline-none focus:border-indigo-500 transition pl-10"
              />
              <Search className="absolute left-3.5 top-3.5 h-4 w-4 text-slate-500" />
            </div>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full bg-[#182235] border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition text-slate-300"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full bg-[#182235] border border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:border-indigo-500 transition text-slate-300"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={() => fetchAnalysis(searchQuery)}
              disabled={loading}
              className="w-full bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white py-2.5 rounded-xl font-semibold text-sm transition shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            >
              {loading ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  Analyze Stock
                </>
              )}
            </button>
          </div>
        </div>

        {/* Errors view */}
        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-center gap-3 text-rose-400 text-sm">
            <AlertTriangle className="h-5 w-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Dashboard Grid Content */}
        {data && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Left Column: Summary & Metrics */}
            <div className="space-y-6">
              
              {/* Core summary card */}
              <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-6 relative overflow-hidden shadow-xl">
                <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-b from-indigo-500/5 to-transparent rounded-full blur-2xl"></div>
                
                <div className="flex justify-between items-start">
                  <div>
                    <h2 className="text-2xl font-black text-white">{data.ticker}</h2>
                    <p className="text-2xl font-bold tracking-tight mt-1 text-slate-200">
                      {fmt(data.summary.close, 'Rs. ')}
                    </p>
                  </div>
                  <div className={`px-4 py-1.5 rounded-lg border font-black tracking-wider text-sm ${getSignalColor(data.summary.signal)}`}>
                    {data.summary.signal}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 border-t border-slate-800/80 pt-4 text-sm">
                  <div>
                    <span className="text-slate-400 text-xs">Support</span>
                    <p className="font-bold text-slate-200 mt-0.5">{fmt(data.summary.support, 'Rs. ')}</p>
                  </div>
                  <div>
                    <span className="text-slate-400 text-xs">Resistance</span>
                    <p className="font-bold text-slate-200 mt-0.5">{fmt(data.summary.resistance, 'Rs. ')}</p>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 border-t border-slate-800/80 pt-4 text-xs">
                  <div>
                    <span className="text-slate-400 block mb-0.5">Fib 23.6%</span>
                    <span className="font-semibold">{fmt(data.summary.fib236, 'Rs. ')}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block mb-0.5">Fib 38.2%</span>
                    <span className="font-semibold">{fmt(data.summary.fib382, 'Rs. ')}</span>
                  </div>
                  <div>
                    <span className="text-slate-400 block mb-0.5">Fib 61.8%</span>
                    <span className="font-semibold">{fmt(data.summary.fib618, 'Rs. ')}</span>
                  </div>
                </div>
              </div>

              {/* Technical Indicator Status Table */}
              <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-4 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <Layers className="h-4 w-4 text-indigo-400" /> Technical Scorecard
                </h3>
                <div className="space-y-3.5 text-sm">
                  <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                    <span className="text-slate-400">RSI (14)</span>
                    <span className={`font-bold ${data.summary.rsi > 70 ? 'text-rose-400' : data.summary.rsi < 30 ? 'text-emerald-400' : 'text-slate-200'}`}>
                      {data.summary.rsi}
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                    <span className="text-slate-400">MACD Line</span>
                    <span className="font-semibold">{data.summary.macd}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                    <span className="text-slate-400">Signal Line</span>
                    <span className="font-semibold">{data.summary.macdSignal}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                    <span className="text-slate-400">ADX (Trend Strength)</span>
                    <span className={`font-bold ${data.summary.adx > 25 ? 'text-indigo-400' : 'text-slate-400'}`}>
                      {data.summary.adx}
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-slate-400">VWAP</span>
                    <span className="font-semibold">{fmt(data.summary.vwap, 'Rs. ')}</span>
                  </div>
                </div>
              </div>

              {/* Relative Strength vs Nifty */}
              {data.relativeStrength && Object.keys(data.relativeStrength).length > 0 && (
                <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-4 shadow-xl">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <GitCompare className="h-4 w-4 text-indigo-400" /> Relative Strength (vs Nifty 50)
                  </h3>
                  <div className="space-y-3.5 text-sm">
                    <div className="flex justify-between items-center py-1.5 border-b border-slate-800/50">
                      <span className="text-slate-400">Stock Return</span>
                      <span className={`font-bold ${data.relativeStrength.stockReturn >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {data.relativeStrength.stockReturn}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center py-1.5 border-b border-slate-800/50">
                      <span className="text-slate-400">Nifty 50 Return</span>
                      <span className="font-semibold">{data.relativeStrength.niftyReturn}%</span>
                    </div>
                    <div className="flex justify-between items-center py-1.5">
                      <span className="text-slate-400">Outperformance</span>
                      <span className={`font-extrabold ${data.relativeStrength.outperformance >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {data.relativeStrength.outperformance > 0 ? '+' : ''}{data.relativeStrength.outperformance}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* News & Sentiment */}
              <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-4 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <Newspaper className="h-4 w-4 text-indigo-400" /> Sentiment Analysis
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-sm">Sentiment Score</span>
                    <span className={`font-bold ${getSentimentColor(data.sentiment.label)}`}>
                      {data.sentiment.score} ({data.sentiment.label})
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${data.sentiment.score > 0.05 ? 'bg-emerald-500' : data.sentiment.score < -0.05 ? 'bg-rose-500' : 'bg-slate-500'}`} 
                      style={{ width: `${Math.min(Math.max((data.sentiment.score + 1) * 50, 0), 100)}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-slate-500 italic">
                    Calculated live by downloading real Google News RSS headlines and parsing polarities via TextBlob.
                  </p>
                </div>
              </div>

            </div>

            {/* Middle & Right Column: Interactive Charts, Fundamentals, Peer Matrix */}
            <div className="lg:col-span-2 space-y-6">
              
              {/* Interactive Candlestick Charts Component */}
              <StockChart data={data.chartData} />

              {/* Fundamentals Grid */}
              <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-6 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <DollarSign className="h-4 w-4 text-indigo-400" /> Fundamental Indicators
                </h3>
                
                {data.fundamentals && Object.keys(data.fundamentals).length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-6 text-sm">
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">P/E Ratio</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.peRatio)}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">PEG Ratio</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.pegRatio)}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">Market Cap</span>
                      <p className="font-bold text-white text-base mt-1">
                        {data.fundamentals.marketCap ? `Rs. ${(data.fundamentals.marketCap / 1e9).toFixed(2)}B` : 'N/A'}
                      </p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">Return on Equity (ROE)</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.roe, '', '%')}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">Debt / Equity</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.debtToEquity)}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">Beta</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.beta)}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">Revenue Growth (YoY)</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.revenueGrowth, '', '%')}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">Dividend Yield</span>
                      <p className="font-bold text-white text-base mt-1">{fmt(data.fundamentals.dividendYield, '', '%')}</p>
                    </div>
                    <div className="bg-[#111827]/40 rounded-xl p-3 border border-slate-800/80">
                      <span className="text-slate-400 text-xs">52W Position</span>
                      <p className="text-xs font-semibold text-indigo-300 mt-1">{fmt(data.fundamentals.fiftyTwoWeekHigh ? 'Position Active' : null)}</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">
                        High: {fmt(data.fundamentals.fiftyTwoWeekHigh)} | Low: {fmt(data.fundamentals.fiftyTwoWeekLow)}
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="text-slate-500 text-xs italic">Fundamental data currently unavailable for this ticker.</p>
                )}
              </div>

              {/* Risk metrics card */}
              <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-6 shadow-xl">
                <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 text-indigo-400" /> Risk Management Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="bg-[#111827]/40 border border-slate-800 p-4 rounded-xl text-center">
                    <span className="text-xs text-slate-400 block mb-1">Ann. Volatility</span>
                    <span className="font-extrabold text-white text-lg">{fmt(data.risk.annualizedVolatility, '', '%')}</span>
                  </div>
                  <div className="bg-[#111827]/40 border border-slate-800 p-4 rounded-xl text-center">
                    <span className="text-xs text-slate-400 block mb-1">Max Drawdown</span>
                    <span className="font-extrabold text-rose-400 text-lg">{fmt(data.risk.maxDrawdown, '', '%')}</span>
                  </div>
                  <div className="bg-[#111827]/40 border border-slate-800 p-4 rounded-xl text-center">
                    <span className="text-xs text-slate-400 block mb-1">VaR (95%, 1D)</span>
                    <span className="font-extrabold text-rose-400 text-lg">{fmt(data.risk.var95_1D, '', '%')}</span>
                  </div>
                  <div className="bg-[#111827]/40 border border-slate-800 p-4 rounded-xl text-center">
                    <span className="text-xs text-slate-400 block mb-1">Sharpe Ratio</span>
                    <span className="font-extrabold text-white text-lg">{fmt(data.risk.sharpeRatio)}</span>
                  </div>
                </div>
              </div>

              {/* Peer Comparison Matrix Tool */}
              <div className="bg-[#0d1424] rounded-2xl border border-slate-800 p-6 space-y-6 shadow-xl">
                <div className="flex justify-between items-center flex-wrap gap-4">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                    <GitCompare className="h-4 w-4 text-indigo-400" /> Peer Comparison Matrix
                  </h3>
                  <div className="flex items-center gap-2 w-full md:w-auto">
                    <input
                      type="text"
                      value={peersInput}
                      onChange={(e) => setPeersInput(e.target.value)}
                      placeholder="e.g. SBIN.NS, ICICIBANK.NS"
                      className="bg-[#182235] border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold focus:outline-none focus:border-indigo-500 w-full md:w-56"
                    />
                    <button
                      onClick={handleCompare}
                      disabled={peersLoading}
                      className="bg-indigo-600 hover:bg-indigo-500 text-white px-3.5 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1 cursor-pointer disabled:opacity-50"
                    >
                      {peersLoading ? 'Comparing...' : 'Compare'}
                    </button>
                  </div>
                </div>

                {peersError && (
                  <p className="text-xs text-rose-400">{peersError}</p>
                )}

                {peersData.length > 0 ? (
                  <div className="overflow-x-auto border border-slate-800 rounded-xl">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-900 border-b border-slate-800 text-slate-400 uppercase tracking-wider">
                          <th className="p-3">Ticker</th>
                          <th className="p-3 text-right">Price</th>
                          <th className="p-3 text-right">P/E</th>
                          <th className="p-3 text-right">PEG</th>
                          <th className="p-3 text-right">ROE</th>
                          <th className="p-3 text-right">D/E</th>
                          <th className="p-3 text-right">Rev Growth</th>
                          <th className="p-3 text-right">Beta</th>
                        </tr>
                      </thead>
                      <tbody>
                        {peersData.map((peer, idx) => (
                          <tr 
                            key={peer.ticker} 
                            className={`border-b border-slate-800/60 hover:bg-slate-800/30 transition ${peer.ticker === ticker ? 'bg-indigo-500/5 font-semibold text-indigo-300' : ''}`}
                          >
                            <td className="p-3">{peer.ticker}</td>
                            <td className="p-3 text-right">{fmt(peer.currentPrice, 'Rs. ')}</td>
                            <td className="p-3 text-right">{fmt(peer.peRatio)}</td>
                            <td className="p-3 text-right">{fmt(peer.pegRatio)}</td>
                            <td className="p-3 text-right">{fmt(peer.roe, '', '%')}</td>
                            <td className="p-3 text-right">{fmt(peer.debtToEquity)}</td>
                            <td className="p-3 text-right">{fmt(peer.revenueGrowth, '', '%')}</td>
                            <td className="p-3 text-right">{fmt(peer.beta)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 italic">Enter peer stock tickers and click Compare to evaluate metrics side-by-side.</p>
                )}
              </div>

            </div>

          </div>
        )}
      </div>
    </main>
  );
}
