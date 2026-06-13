'use client';

import Link from 'next/link';
import { ArrowLeft, ShieldAlert, Scale, FileText, Globe, AlertCircle } from 'lucide-react';

export default function TermsAndConditions() {
  return (
    <div className="min-height-screen bg-black text-white selection:bg-purple-500 selection:text-white">
      {/* Header / Nav */}
      <header className="border-b border-[#111] bg-black/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <Link 
            href="/"
            className="flex items-center gap-2 text-slate-400 hover:text-white transition text-sm font-semibold"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
          <span className="text-xs text-slate-500 font-mono">
            Created: June 10, 2026 by{' '}
            <a href="https://visheshsanghvi.qzz.io/" target="_blank" rel="noopener noreferrer" className="underline text-slate-400 hover:text-white transition">
              Vishesh Sanghvi
            </a>
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-12">
        <div className="flex items-center gap-3 mb-6">
          <div className="h-10 w-10 bg-purple-500/10 border border-purple-500/20 rounded-xl flex items-center justify-center">
            <Scale className="h-5 w-5 text-purple-400" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">Terms of Service</h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-1">Legal Disclaimers, AI Model Transparency & Limitations of Liability</p>
          </div>
        </div>

        {/* Warning Alert */}
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 sm:p-5 mb-10 flex gap-3 items-start">
          <ShieldAlert className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm text-amber-200/90 leading-relaxed">
            <strong className="text-white block mb-1">CRITICAL NOTICE: READ BEFORE PROCEEDING</strong>
            StockIQ Pro is a pure simulation, data analysis, and educational tool. The machine learning outputs, stock predictions, and other analytical modules do not represent certified financial, investment, legal, or tax advice. All data and predictions are provided "as-is" without warranty of any kind.
          </div>
        </div>

        {/* Sections */}
        <div className="space-y-10 text-slate-300 leading-relaxed text-sm sm:text-base">
          {/* Section 1 */}
          <section className="border-b border-[#111] pb-8">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-3">
              <span className="text-purple-500 font-mono">01.</span> No Financial Advice Disclaimer
            </h2>
            <p className="text-slate-400 text-xs sm:text-sm mb-4">
              None of the content, machine learning predictions, signal indices, technical patterns, or risk analytics displayed on StockIQ Pro constitutes a recommendation to buy, sell, or hold any security or financial instrument. 
            </p>
            <p className="text-slate-400 text-xs sm:text-sm">
              We are not registered financial advisors, brokers, or wealth managers. All decisions made based on information presented on this platform are the sole responsibility of the user. You are strongly advised to perform independent research and/or consult with a licensed, certified financial advisor before committing funds to live markets.
            </p>
          </section>

          {/* Section 2 */}
          <section className="border-b border-[#111] pb-8">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-3">
              <span className="text-purple-500 font-mono">02.</span> Machine Learning Predictions & Consensus Accuracy
            </h2>
            <p className="text-slate-400 text-xs sm:text-sm mb-4">
              Our AI Price Predictions are generated using a mathematical Random Forest Regressor model trained on historical data. These predictions represent statistical extrapolations of technical price patterns (e.g. moving averages, RSI, MACD, historical return distributions) over user-defined training horizons.
            </p>
            <div className="bg-[#0a0a0a] border border-[#242424] rounded-lg p-4 mb-4 text-xs text-slate-500">
              <div className="font-semibold text-slate-300 mb-2">Model Disagreement & Uncertainty Bounds</div>
              The "Model Consensus" and "Confidence Score" are calculated by measuring the variance (disagreement) across all 100 individual decision trees inside our Random Forest ensemble:
              <ul className="list-disc pl-5 mt-2 space-y-1">
                <li><span className="text-emerald-400 font-semibold">Stable Consensus:</span> Low tree variance, indicating standard technical indicators are highly aligned.</li>
                <li><span className="text-amber-400 font-semibold">Moderate Volatility:</span> Medium tree variance, signaling conflicting indicators.</li>
                <li><span className="text-rose-400 font-semibold">High Uncertainty:</span> High tree variance, signaling erratic price actions or severe macro outliers.</li>
              </ul>
            </div>
            <p className="text-slate-400 text-xs sm:text-sm">
              Historical patterns are not guarantees of future performance. Market conditions are subject to sudden macroeconomic shocks, regulatory updates, earnings adjustments, and liquidity events that no machine learning model can predict.
            </p>
          </section>

          {/* Section 3 */}
          <section className="border-b border-[#111] pb-8">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-3">
              <span className="text-purple-500 font-mono">03.</span> Limitation of Liability
            </h2>
            <p className="text-slate-400 text-xs sm:text-sm mb-3 font-semibold text-slate-200">
              TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL STOCKIQ PRO, ITS DEVELOPERS, AFFILIATES, OR DATA PROVIDERS BE LIABLE FOR ANY DIRECT, INDIRECT, PUNITIVE, INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES WHATSOEVER.
            </p>
            <p className="text-slate-400 text-xs sm:text-sm">
              This includes, but is not limited to, damages for loss of profits, trading losses, capital losses, data loss, or business interruption, arising out of or in any way connected with the use or performance of the site, the delay or inability to use the site, or any decision made in reliance on information obtained through this service.
            </p>
          </section>

          {/* Section 4 */}
          <section className="border-b border-[#111] pb-8">
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-3">
              <span className="text-purple-500 font-mono">04.</span> Data Sources & Latency
            </h2>
            <p className="text-slate-400 text-xs sm:text-sm">
              All live prices and historical tickers are fetched using rest client proxies connecting to third-party public API endpoints. This data is subject to standard latency delays (typically ~15 minutes). We make no warranties regarding the uptime, correctness, accuracy, or completeness of the fetched pricing streams.
            </p>
          </section>

          {/* Section 5 */}
          <section>
            <h2 className="text-lg font-bold text-white flex items-center gap-2 mb-3">
              <span className="text-purple-500 font-mono">05.</span> Indemnification
            </h2>
            <p className="text-slate-400 text-xs sm:text-sm">
              By using this site, you agree to indemnify, defend, and hold harmless StockIQ Pro, its core contributors, and developers from and against any and all claims, liabilities, losses, costs, or damages (including reasonable legal fees) arising from your use of the platform or violation of these terms.
            </p>
          </section>
        </div>

        {/* Contact Info Footer */}
        <div className="mt-16 pt-8 border-t border-[#111] text-center text-xs text-slate-600 flex flex-col items-center gap-2">
          <Globe className="h-4 w-4 text-slate-500" />
          <span>
            Created on June 10, 2026 by{' '}
            <a href="https://visheshsanghvi.qzz.io/" target="_blank" rel="noopener noreferrer" className="underline text-slate-400 hover:text-white transition">
              Vishesh Sanghvi
            </a>
          </span>
          <span>© 2026 StockIQ Pro · Professional Stock Analysis Platform · All rights reserved. Use at your own risk.</span>
        </div>
      </main>
    </div>
  );
}
