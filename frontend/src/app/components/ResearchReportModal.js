'use client';

import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Printer, 
  X, 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  ShieldAlert, 
  Activity, 
  CheckCircle2, 
  Clock,
  ExternalLink,
  Target
} from 'lucide-react';
import { API_BASE_URL } from '../config';

export default function ResearchReportModal({ isOpen, onClose, ticker }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen || !ticker) return;

    let isMounted = true;
    setLoading(true);

    // Fetch live quote and fundamentals for report
    Promise.all([
      fetch(`${API_BASE_URL}/api/live?ticker=${encodeURIComponent(ticker)}`).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`${API_BASE_URL}/api/valuation?ticker=${encodeURIComponent(ticker)}`).then(r => r.ok ? r.json() : null).catch(() => null),
      fetch(`${API_BASE_URL}/api/fundamentals?ticker=${encodeURIComponent(ticker)}`).then(r => r.ok ? r.json() : null).catch(() => null)
    ]).then(([quote, valuation, fundamentals]) => {
      if (!isMounted) return;
      const isUS = ticker && !ticker.endsWith('.NS') && !ticker.endsWith('.BO');
      const loc = isUS ? 'en-US' : 'en-IN';
      setData({
        quote,
        valuation,
        fundamentals,
        generatedAt: new Date().toLocaleString(loc, {
          dateStyle: 'medium',
          timeStyle: 'short'
        })
      });
      setLoading(false);
    });

    return () => { isMounted = false; };
  }, [isOpen, ticker]);

  if (!isOpen) return null;

  const handlePrint = () => {
    window.print();
  };

  const isUS = ticker && !ticker.endsWith('.NS') && !ticker.endsWith('.BO');
  const q = data?.quote || {};
  const v = data?.valuation || {};
  const f = data?.fundamentals || {};
  const currSym = q.currency_symbol || (isUS ? '$' : '₹');
  const loc = isUS ? 'en-US' : 'en-IN';

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto"
      onClick={onClose}
    >
      <div 
        className="w-full max-w-4xl bg-[#0e0e12] border border-white/10 rounded-2xl shadow-2xl overflow-hidden text-slate-200 my-8 animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Top Bar (Hidden on print) */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08] bg-white/[0.02] print:hidden">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white tracking-wide">Executive Equity Research Memo</h2>
              <p className="text-[11px] text-slate-400">Institutional Snapshot · {ticker}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow transition-all cursor-pointer"
            >
              <Printer className="w-3.5 h-3.5" /> Print / Save PDF
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Printable Report Body */}
        <div className="p-8 bg-[#0a0a0d] print:bg-white print:text-black print:p-6" id="printable-research-report">
          {/* Print Header */}
          <div className="flex items-start justify-between border-b border-white/10 pb-6 mb-6 print:border-black">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-black tracking-tight text-white print:text-black">
                  {q.longName || q.shortName || ticker}
                </h1>
                <span className="px-2.5 py-0.5 rounded text-xs font-bold bg-blue-500/15 text-blue-400 border border-blue-500/30 print:border-black print:text-black">
                  {ticker}
                </span>
              </div>
              <p className="text-xs text-slate-400 print:text-slate-600 mt-1">
                Exchange: {ticker.endsWith('.NS') ? 'National Stock Exchange (NSE)' : ticker.endsWith('.BO') ? 'Bombay Stock Exchange (BSE)' : 'Global'} · Currency: INR
              </p>
            </div>
            <div className="text-right">
              <span className="text-[10px] uppercase font-bold tracking-wider text-blue-400 print:text-black block">StockIQ Pro Intelligence</span>
              <p className="text-xs text-slate-400 print:text-slate-600 mt-0.5">{data?.generatedAt || 'Live'}</p>
            </div>
          </div>

          {loading ? (
            <div className="py-20 text-center text-slate-400">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
              <p className="text-xs">Generating analytical research report…</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Executive Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] print:border-slate-300">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">Current Price</span>
                  <span className="text-xl font-bold text-white print:text-black">
                    {currSym}{q.price ? q.price.toLocaleString(loc) : '—'}
                  </span>
                  <span className={`text-xs block mt-1 font-semibold ${(q.change || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {(q.change || 0) >= 0 ? '+' : ''}{q.change ? q.change.toFixed(2) : 0} ({q.changePct ? q.changePct.toFixed(2) : 0}%)
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] print:border-slate-300">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">DCF Fair Value</span>
                  <span className="text-xl font-bold text-white print:text-black">
                    {v.fair_value ? `${currSym}${v.fair_value.toLocaleString(loc)}` : '—'}
                  </span>
                  <span className={`text-xs block mt-1 font-semibold ${(v.upside_downside || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {(v.upside_downside || 0) >= 0 ? '+' : ''}{v.upside_downside ? v.upside_downside.toFixed(1) : '—'}% Fair Value Gap
                  </span>
                </div>

                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] print:border-slate-300">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">52W Range</span>
                  <span className="text-xs font-bold text-white print:text-black block mt-1">
                    {currSym}{q.fiftyTwoWeekLow ? q.fiftyTwoWeekLow.toLocaleString(loc) : '—'} - {currSym}{q.fiftyTwoWeekHigh ? q.fiftyTwoWeekHigh.toLocaleString(loc) : '—'}
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-1">Annual Volatility Corridor</span>
                </div>

                <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] print:border-slate-300">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold block mb-1">Valuation Status</span>
                  <span className={`text-sm font-bold block mt-1 ${
                    (v.upside_downside || 0) > 10 ? 'text-emerald-400' : (v.upside_downside || 0) < -10 ? 'text-amber-400' : 'text-blue-400'
                  }`}>
                    {(v.upside_downside || 0) > 15 ? 'UNDERVALUED' : (v.upside_downside || 0) < -15 ? 'PREMIUM PRICED' : 'FAIR VALUE'}
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-1">Discounted Cash Flow</span>
                </div>
              </div>

              {/* Two-Column Deep Dive */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Financial Health & Fundamentals */}
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] print:border-slate-300">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 print:text-black mb-3 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-blue-400" /> Fundamental Health Ratios
                  </h3>
                  <div className="space-y-2.5 text-xs">
                    <div className="flex justify-between py-1 border-b border-white/[0.04] print:border-slate-200">
                      <span className="text-slate-400 print:text-slate-600">Trailing P/E Ratio</span>
                      <span className="font-bold text-white print:text-black">{f.trailingPE ? f.trailingPE.toFixed(2) : '—'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-white/[0.04] print:border-slate-200">
                      <span className="text-slate-400 print:text-slate-600">Price to Book (P/B)</span>
                      <span className="font-bold text-white print:text-black">{f.priceToBook ? f.priceToBook.toFixed(2) : '—'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-white/[0.04] print:border-slate-200">
                      <span className="text-slate-400 print:text-slate-600">Return on Equity (RoE)</span>
                      <span className="font-bold text-white print:text-black">{f.returnOnEquity ? `${(f.returnOnEquity * 100).toFixed(2)}%` : '—'}</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-white/[0.04] print:border-slate-200">
                      <span className="text-slate-400 print:text-slate-600">Profit Margin</span>
                      <span className="font-bold text-white print:text-black">{f.profitMargins ? `${(f.profitMargins * 100).toFixed(2)}%` : '—'}</span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-slate-400 print:text-slate-600">Debt to Equity</span>
                      <span className="font-bold text-white print:text-black">{f.debtToEquity ? (f.debtToEquity / 100).toFixed(2) : '—'}</span>
                    </div>
                  </div>
                </div>

                {/* Analytical Synthesis & Model View */}
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] print:border-slate-300">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 print:text-black mb-3 flex items-center gap-1.5">
                    <Brain className="w-3.5 h-3.5 text-purple-400" /> Quantitative Intelligence Summary
                  </h3>
                  <div className="space-y-3 text-xs leading-relaxed text-slate-300 print:text-slate-700">
                    <p>
                      <strong>Valuation Horizon:</strong> The DCF model projects intrinsic worth based on free cash flows discounted at a normalized Cost of Equity. The gap indicates potential medium-to-long term margin of safety.
                    </p>
                    <p>
                      <strong>Volatility Profiling:</strong> Operating within a 52-week corridor of {currSym}{q.fiftyTwoWeekLow?.toFixed(0) || '—'} to {currSym}{q.fiftyTwoWeekHigh?.toFixed(0) || '—'}. Positioned at {
                        q.price && q.fiftyTwoWeekHigh
                          ? `${(((q.fiftyTwoWeekHigh - q.price) / q.fiftyTwoWeekHigh) * 100).toFixed(1)}% below annual peak.`
                          : 'balanced distribution.'
                      }
                    </p>
                  </div>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="p-3.5 rounded-lg bg-white/[0.02] border border-white/[0.04] text-[10px] text-slate-500 print:text-slate-600 leading-normal">
                <p>
                  <strong>Disclaimer:</strong> This automated institutional equity research report is compiled for informational and quantitative analytical purposes by StockIQ Pro. It does not constitute financial advice, solicitation, or a recommendation to buy or sell securities. Past returns and machine learning forecasts do not guarantee future outcomes.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
