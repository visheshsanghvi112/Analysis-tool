'use client';

import { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  HelpCircle, 
  CheckCircle, 
  XCircle, 
  Activity, 
  Percent, 
  DollarSign, 
  ShieldAlert,
  Info,
  Scale,
  Award
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://stock-analysis-backend-seven.vercel.app';

export default function LongTermAnalysis({ ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Interactive DCF State variables
  const [dcfFlow, setDcfFlow] = useState(0); // raw starting cash flow in INR
  const [dcfGrowth, setDcfGrowth] = useState(8); // in % (e.g. 8 for 8%)
  const [dcfDiscount, setDcfDiscount] = useState(11); // WACC in %
  const [dcfTerminal, setDcfTerminal] = useState(4.5); // terminal growth in %
  const [flowType, setFlowType] = useState('Free Cash Flow');

  // Load valuation data on ticker change
  useEffect(() => {
    const fetchValuation = async () => {
      if (!ticker) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE_URL}/api/valuation?ticker=${encodeURIComponent(ticker)}`);
        const json = await res.json();
        
        if (!res.ok) throw new Error(json.detail || 'Failed to fetch valuation data');
        
        setData(json);
        // Initialize DCF inputs from backend defaults
        if (json.dcf_defaults) {
          setDcfFlow(json.dcf_defaults.starting_flow);
          setDcfGrowth(Math.round(json.dcf_defaults.growth_rate * 1000) / 10);
          setDcfDiscount(Math.round(json.dcf_defaults.discount_rate * 1000) / 10);
          setDcfTerminal(Math.round(json.dcf_defaults.terminal_growth * 1000) / 10);
          setFlowType(json.dcf_defaults.flow_type);
        }
      } catch (err) {
        setError(err.message);
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchValuation();
  }, [ticker]);

  // Recalculate DCF on the fly when inputs change
  const calculateDCF = () => {
    if (!data || !data.shares_outstanding) return null;

    const g = dcfGrowth / 100;
    const d = dcfDiscount / 100;
    const tg = dcfTerminal / 100;
    const shares = data.shares_outstanding;
    const cash = data.total_cash || 0;
    const debt = data.total_debt || 0;

    let pvFcfSum = 0;
    let currentFlow = dcfFlow;
    const projections = [];

    for (let year = 1; year <= 5; year++) {
      currentFlow = currentFlow * (1 + g);
      const pv = currentFlow / Math.pow(1 + d, year);
      pvFcfSum += pv;
      projections.push({
        year,
        flow: currentFlow,
        pv
      });
    }

    const terminalFlow = currentFlow * (1 + tg);
    const effectiveDiscount = d > tg ? d - tg : 0.01;
    const terminalValue = terminalFlow / effectiveDiscount;
    const pvTerminalValue = terminalValue / Math.pow(1 + d, 5);

    const enterpriseValue = pvFcfSum + pvTerminalValue;
    const equityValue = enterpriseValue + cash - debt;
    const intrinsicValue = equityValue / shares;

    return {
      intrinsicValue: Math.max(0, intrinsicValue),
      enterpriseValue,
      equityValue,
      pvFcfSum,
      pvTerminalValue,
      projections
    };
  };

  const dcfResults = calculateDCF();

  // Helper to format currency values cleanly in Indian format or standard Billions/Millions
  const formatVal = (val, suffix = '') => {
    if (val === null || val === undefined) return 'N/A';
    
    const absVal = Math.abs(val);
    if (absVal >= 1e12) {
      return `₹${(val / 1e12).toFixed(2)}T${suffix}`;
    }
    if (absVal >= 1e9) {
      return `₹${(val / 1e9).toFixed(2)}B${suffix}`;
    }
    if (absVal >= 1e7) {
      return `₹${(val / 1e7).toFixed(2)}Cr${suffix}`;
    }
    if (absVal >= 1e5) {
      return `₹${(val / 1e5).toFixed(2)}L${suffix}`;
    }
    return `₹${val.toFixed(2)}${suffix}`;
  };

  if (loading) {
    return (
      <div className="glass-card p-6 flex flex-col items-center justify-center min-h-[300px]">
        <div className="h-10 w-10 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin mb-4"></div>
        <p className="text-slate-400 text-sm font-medium animate-pulse">Decomposing ROE &amp; computing DCF models...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card p-6 border border-rose-500/20 bg-rose-500/5 text-rose-400 text-sm flex items-center gap-3">
        <ShieldAlert className="h-6 w-6 shrink-0" />
        <div>
          <h4 className="font-bold text-base mb-0.5">Fundamentals Loading Error</h4>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const currentPrice = data.current_price || 0;
  const marginOfSafety = dcfResults 
    ? ((dcfResults.intrinsicValue - currentPrice) / dcfResults.intrinsicValue) * 100
    : 0;

  // Compute helper health colors
  const getHealthScoreColor = (score) => {
    if (score >= 8) return 'text-emerald-400 border-emerald-400/20 bg-emerald-400/[0.03]';
    if (score >= 5) return 'text-yellow-400 border-yellow-400/20 bg-yellow-400/[0.03]';
    return 'text-rose-400 border-rose-500/20 bg-rose-500/[0.03]';
  };

  return (
    <div className="space-y-6">
      
      {/* SECTION HEADER */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20 rounded-2xl">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-indigo-500/20 flex items-center justify-center border border-indigo-500/30 shadow-inner">
            <Scale className="h-5 w-5 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-base sm:text-lg font-bold text-white leading-tight">Long-Term Investment &amp; Valuation Hub</h2>
            <p className="text-xs text-slate-400">Intrinsic Value Calculations, Profitability Drivers, and Financial Strength</p>
          </div>
        </div>
        
        {/* Overall Health Score Badge */}
        <div className={`px-4 py-2 rounded-xl border flex items-center gap-2 ${getHealthScoreColor(data.health_score)}`}>
          <Award className="h-5 w-5 shrink-0" />
          <div>
            <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Health Score</div>
            <div className="text-sm sm:text-base font-extrabold">{data.health_score} / 10</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* INTERACTIVE DCF CALCULATOR */}
        <div className="glass-card p-5 sm:p-6 space-y-6 flex flex-col justify-between">
          <div className="space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-indigo-400" />
                <h3 className="text-sm sm:text-base font-bold text-white">Interactive DCF Intrinsic Value</h3>
              </div>
              <span className="text-[10px] bg-indigo-500/10 text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-500/20 font-bold">CAPM / WACC Model</span>
            </div>

            {/* DCF METRIC PANEL */}
            {dcfResults && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 p-4 bg-white/[0.02] border border-white/[0.04] rounded-xl text-center">
                
                {/* Intrinsic Value */}
                <div className="space-y-1">
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Intrinsic Value</div>
                  <div className="text-lg sm:text-xl font-extrabold text-indigo-400">
                    ₹{dcfResults.intrinsicValue.toFixed(1)}
                  </div>
                  <p className="text-[9px] text-slate-600">Fair value per share</p>
                </div>
                
                {/* Current Price */}
                <div className="space-y-1 border-t sm:border-t-0 sm:border-x border-slate-800/80 pt-2 sm:pt-0">
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Current Price</div>
                  <div className="text-lg sm:text-xl font-extrabold text-slate-200">
                    ₹{currentPrice.toFixed(1)}
                  </div>
                  <p className="text-[9px] text-slate-600">Market quote</p>
                </div>
                
                {/* Margin of Safety */}
                <div className="space-y-1 border-t sm:border-t-0 pt-2 sm:pt-0">
                  <div className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Margin of Safety</div>
                  <div className={`text-lg sm:text-xl font-extrabold ${marginOfSafety >= 15 ? 'text-emerald-400' : marginOfSafety >= 0 ? 'text-yellow-400' : 'text-rose-400'}`}>
                    {marginOfSafety >= 0 ? '+' : ''}{marginOfSafety.toFixed(1)}%
                  </div>
                  <p className="text-[9px] text-slate-600">
                    {marginOfSafety >= 15 ? 'Undervalued (Safe)' : marginOfSafety >= 0 ? 'Fair Value' : 'Overvalued'}
                  </p>
                </div>
              </div>
            )}

            {/* SLIDERS & CONTROLS */}
            <div className="space-y-4 pt-1">
              
              {/* Starting Cash Flow */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-semibold flex items-center gap-1.5">
                    Starting Cash Flow Proxy 
                    <span className="text-[10px] text-indigo-300 font-medium bg-indigo-500/10 px-1.5 py-0.2 rounded">({flowType})</span>
                  </span>
                  <span className="text-slate-200 font-bold">{formatVal(dcfFlow)}</span>
                </div>
                <div className="flex gap-2">
                  <select 
                    value={flowType}
                    onChange={(e) => {
                      const type = e.target.value;
                      setFlowType(type);
                      if (type === 'Free Cash Flow') {
                        setDcfFlow(data.free_cashflow);
                      } else if (type === 'Net Income') {
                        setDcfFlow(data.net_income || data.market_cap * 0.05); // Fallback estimate
                      } else if (type === 'Operating Cash Flow') {
                        setDcfFlow(data.operating_cashflow);
                      }
                    }}
                    className="bg-slate-900 border border-slate-800 text-[11px] rounded-lg px-2 text-slate-300 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="Free Cash Flow">Free Cash Flow</option>
                    <option value="Net Income">Net Income</option>
                    <option value="Operating Cash Flow">Operating Cash Flow</option>
                  </select>
                  <input 
                    type="range"
                    min={Math.max(1000000, dcfFlow * 0.1)}
                    max={Math.max(1000000000000, dcfFlow * 3)}
                    step={Math.max(100000, dcfFlow * 0.05)}
                    value={dcfFlow}
                    onChange={(e) => setDcfFlow(parseFloat(e.target.value))}
                    className="flex-1 accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                  />
                </div>
              </div>

              {/* Growth Rate */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400 font-semibold">Growth Rate (Years 1-5)</span>
                  <span className="text-indigo-400 font-bold">{dcfGrowth.toFixed(1)}%</span>
                </div>
                <input 
                  type="range"
                  min="0"
                  max="30"
                  step="0.5"
                  value={dcfGrowth}
                  onChange={(e) => setDcfGrowth(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              {/* Discount Rate (WACC) */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400 font-semibold">Discount Rate (WACC)</span>
                  <span className="text-indigo-400 font-bold">{dcfDiscount.toFixed(1)}%</span>
                </div>
                <input 
                  type="range"
                  min="5"
                  max="20"
                  step="0.5"
                  value={dcfDiscount}
                  onChange={(e) => setDcfDiscount(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>

              {/* Terminal Growth Rate */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400 font-semibold">Terminal Growth Rate</span>
                  <span className="text-indigo-400 font-bold">{dcfTerminal.toFixed(1)}%</span>
                </div>
                <input 
                  type="range"
                  min="1"
                  max="8"
                  step="0.1"
                  value={dcfTerminal}
                  onChange={(e) => setDcfTerminal(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Quick math breakdown disclosure */}
          <div className="mt-4 p-3 bg-white/[0.01] border border-white/[0.03] rounded-lg text-[10px] text-slate-500 leading-relaxed space-y-1">
            <span className="font-bold text-slate-400 block mb-0.5">DCF Valuation Method Note:</span>
            Enterprise Value = PV of cash flows for years 1-5 + PV of terminal value. 
            Equity Value = Enterprise Value + Cash ({formatVal(data.total_cash)}) - Debt ({formatVal(data.total_debt)}).
            Intrinsic Value = Equity Value / Shares Outstanding ({data.shares_outstanding ? (data.shares_outstanding/1e9).toFixed(2) + 'B' : 'N/A'}).
          </div>
        </div>

        {/* FINANCIAL HEALTH CHECKLIST */}
        <div className="glass-card p-5 sm:p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-emerald-400" />
              <h3 className="text-sm sm:text-base font-bold text-white">Long-Term Safety Health Check</h3>
            </div>
            <span className="text-[10px] bg-emerald-500/10 text-emerald-300 px-2 py-0.5 rounded-full border border-emerald-500/20 font-bold">10-Criteria strength profile</span>
          </div>

          {/* Grid layout for checklist items */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {data.health_checklist?.map((item, idx) => (
              <div 
                key={idx} 
                className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.04] transition duration-150"
              >
                <div className="flex items-center gap-2 min-w-0">
                  {item.passed ? (
                    <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0" />
                  ) : (
                    <XCircle className="h-4 w-4 text-rose-500 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="text-[11px] font-bold text-slate-300 truncate">{item.metric}</p>
                    <p className="text-[9px] text-slate-500">Condition: {item.condition}</p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <span className={`text-[11px] font-extrabold ${item.passed ? 'text-emerald-400' : 'text-slate-400'}`}>
                    {item.value}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Graham Number Panel */}
          {data.graham_number && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.04] rounded-xl flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-7 w-7 rounded bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
                  <Info className="h-4 w-4 text-amber-400" />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-amber-400">Graham Valuation Number</h4>
                  <p className="text-[9px] text-slate-500">Max purchase threshold (√(22.5 × EPS × Book Value))</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-bold text-amber-400">₹{data.graham_number}</div>
                <p className="text-[9px] text-slate-500">
                  {currentPrice <= data.graham_number 
                    ? '✓ Price <= Graham (Discount)' 
                    : `Premium over Graham (+${((currentPrice - data.graham_number) / data.graham_number * 100).toFixed(0)}%)`}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* DUPONT ANALYSIS DECOMPOSITION */}
      {data.dupont && (
        <div className="glass-card p-5 sm:p-6 space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-400" />
              <h3 className="text-sm sm:text-base font-bold text-white">DuPont ROE Profitability Analysis</h3>
            </div>
            <span className="text-[10px] bg-purple-500/10 text-purple-300 px-2 py-0.5 rounded-full border border-purple-500/20 font-bold">DuPont Model</span>
          </div>

          <p className="text-xs text-slate-400 leading-relaxed">
            The DuPont equation decomposes <strong>Return on Equity (ROE)</strong> to show how a business drives shareholder returns.
            It reveals whether profitability is driven by high margins (efficiency), high asset utilization (operational speed), or leverage (financial risk).
          </p>

          <div className="flex flex-col md:flex-row items-stretch md:items-center gap-3">
            
            {/* Net Profit Margin */}
            <div className="flex-1 p-4 bg-purple-500/5 border border-purple-500/20 rounded-xl text-center space-y-1">
              <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Net Profit Margin</div>
              <div className="text-xl font-extrabold text-purple-300">
                {data.dupont.net_profit_margin ? `${data.dupont.net_profit_margin}%` : 'N/A'}
              </div>
              <p className="text-[9px] text-slate-500">Operating efficiency (Net Income / Revenue)</p>
            </div>

            {/* Multiply Sign */}
            <div className="hidden md:block text-center font-extrabold text-slate-500 text-2xl shrink-0">×</div>
            <div className="md:hidden text-center font-bold text-slate-600 text-sm">× Asset Turnover ×</div>

            {/* Asset Turnover */}
            <div className="flex-1 p-4 bg-indigo-500/5 border border-indigo-500/20 rounded-xl text-center space-y-1">
              <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Asset Turnover</div>
              <div className="text-xl font-extrabold text-indigo-300">
                {data.dupont.asset_turnover ? `${data.dupont.asset_turnover}x` : 'N/A'}
              </div>
              <p className="text-[9px] text-slate-500">Asset efficiency (Revenue / Total Assets)</p>
            </div>

            {/* Multiply Sign */}
            <div className="hidden md:block text-center font-extrabold text-slate-500 text-2xl shrink-0">×</div>

            {/* Equity Multiplier */}
            <div className="flex-1 p-4 bg-sky-500/5 border border-sky-500/20 rounded-xl text-center space-y-1">
              <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Equity Multiplier</div>
              <div className="text-xl font-extrabold text-sky-300">
                {data.dupont.equity_multiplier ? `${data.dupont.equity_multiplier}x` : 'N/A'}
              </div>
              <p className="text-[9px] text-slate-500">Financial leverage (Total Assets / Equity)</p>
            </div>

            {/* Equals Sign + ROE */}
            <div className="hidden md:block text-center font-extrabold text-slate-500 text-2xl shrink-0">=</div>

            {/* DuPont ROE Result */}
            <div className="flex-1 p-4 bg-gradient-to-b from-indigo-500/10 to-purple-500/10 border border-indigo-500/30 rounded-xl text-center space-y-1">
              <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">DuPont ROE</div>
              <div className="text-xl font-extrabold text-indigo-400">
                {data.dupont.calculated_roe ? `${data.dupont.calculated_roe}%` : 'N/A'}
              </div>
              <p className="text-[9px] text-slate-500">Return on Shareholders&apos; Equity</p>
            </div>

          </div>


          {/* Analyst Insight */}
          <div className="p-3.5 bg-white/[0.02] border border-white/[0.05] rounded-xl text-[11px] text-slate-400 leading-relaxed">
            <span className="font-bold text-slate-300">Analyst Insight: </span>
            {data.dupont.net_profit_margin > 15
              ? `Strong net margins (${data.dupont.net_profit_margin}%) indicate the company has significant pricing power and cost discipline — the primary driver of ROE.`
              : data.dupont.equity_multiplier > 2
              ? `ROE is leveraged by debt (equity multiplier: ${data.dupont.equity_multiplier}x). The business relies on financial leverage to amplify returns — higher risk, but manageable if cash flows are stable.`
              : `Moderate margins (${data.dupont.net_profit_margin}%) balanced by asset efficiency (${data.dupont.asset_turnover}x turnover). Watch for margin expansion as the primary catalyst for ROE improvement.`}
          </div>


        </div>
      )}

    </div>
  );
}
