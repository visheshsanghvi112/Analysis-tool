# StockIQ Pro — Frontend Web Application 📈

> **High-Performance Institutional Quantitative Workstation Client**  
> Built with **Next.js 16 (Turbopack)**, **React 19**, **Recharts 3**, and **Tailwind CSS**.

---

## 🌟 Overview

The StockIQ Pro frontend is a dark-themed, glassmorphic financial terminal designed to deliver institutional-grade analytical tools to retail investors and quantitative researchers. It connects dynamically to the FastAPI backend service to provide real-time pricing, multi-model ML forecasts, deep full-article news intelligence, portfolio optimization, and intraday technical charting across 7,954 NSE, BSE, ETF, and global instruments.

---

## 🗺️ Application Routes

The application features a modern multi-page Next.js App Router architecture:

| Route | Name | Description |
| :--- | :--- | :--- |
| **`/`** | **Analysis Workstation** | Master terminal with asset-adaptive architecture: single-stock fundamental analysis (DCF, DuPont, Graham, Altman Z, Beneish M) or ETF long-term intelligence (AUM, Expense Ratio, Tracking Error vs Benchmark, 6-Point Health Checklist, SIP Suitability Scoring, Top 10 Holdings & Sector Breakdown, 3-Year Rolling Returns Distribution), alongside 6-model ML price forecasts with Street Consensus comparisons, GARCH volatility, Monte Carlo simulations, options Greeks, news intelligence, and executive PDF memos. |
| **`/intraday`** | **Intraday Terminal** | High-frequency technical analysis workspace with live candlestick charting, Supertrend indicators, EMA Ribbons (9, 21, 50, 200), Multi-timeframe VWAP with volatility bands ($\pm 1\sigma, \pm 2\sigma$), RSI, MACD, ATR, and one-click CSV export. |
| **`/portfolio`** | **Portfolio & Capital Advisor** | Modern Portfolio Theory (Markowitz Efficient Frontier), Tangency & GMV allocation, Value-at-Risk (VaR 95/99%), dynamic rebalancing, and the **Smart Capital Advisor** for automated loss recovery and averaging down. |
| **`/browse`** | **Market Universe Browser** | Search, filter, and explore the complete 7,954 instrument database across NSE equities, BSE scrips, ETFs, sectoral indices, and global ADRs. |
| **`/features`** | **Architecture & Capabilities** | Interactive technical showcase demonstrating the underlying mathematical models, Scrapling stealth web scraping, ensemble ML algorithms, and system performance benchmarks. |
| **`/terms`** | **Terms & Disclaimers** | Complete platform terms of service, open-source licensing, and SEBI regulatory risk disclaimers. |

---

## 🧩 Component Directory (`src/app/components/`)

| Component | Role & Functionality |
| :--- | :--- |
| **`Header.js`** | Global navigation header with quick links, live market status indicator, and Spotlight search trigger. |
| **`StockSearchModal.js`** | Universal Spotlight search palette (<kbd>⌘K</kbd> / <kbd>/</kbd>) with categorized tabs (All, Equities, ETFs, Indices, Global), BSE 6-digit code resolution, and fuzzy typo tolerance. |
| **`StockChart.js`** | Primary interactive candlestick and line chart with volume profiles, moving averages, and technical indicators. |
| **`IntradayTerminal.js`** | Dedicated intraday trading terminal featuring real-time intervals (1m, 2m, 5m, 15m, 30m, 60m), Supertrend signals, multi-band VWAP, and CSV indicator exports. |
| **`MLPrediction.js`** | Asset-aware 6-model ML ensemble (Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, Bayesian Ridge) with SHAP waterfall attribution. Adapts dynamically: 5-day horizon and news fusion for stocks; 30-day horizon, 38 long-term features (Golden Cross, quarterly momentum, annual drawdown), ACCUMULATE/AVOID signals, and zeroed company news for ETFs. Features a Street Analyst Consensus vs 6-Model AI Ensemble Target comparison for equities with divergence spread analysis. |
| **`ETFLongTermPanel.js`** | Dedicated institutional ETF analysis panel replacing DCF/Graham for ETF tickers. Computes 1Y/3Y/5Y CAGR vs benchmark, annualised tracking error, Sharpe ratio (3yr), max drawdown, 6-Point ETF Health Checklist (AUM, expense ratio, tracking error, CAGR alpha, Sharpe, NAV discount), SIP Suitability Score (0–10), yearly returns sparklines, Top 10 Underlying Holdings & Sector Weightings (with live Yahoo Finance API and curated Indian ETF database fallback), and a 3-Year Rolling Returns Distribution card (median CAGR, min/max range, probability of profit). |
| **`MonteCarloSimulation.js`** | Geometric Brownian Motion (GBM) stochastic price path simulation across 30 to 365-day horizons with quantile corridors ($P_{2.5}$ to $P_{97.5}$) and CSV export. |
| **`AdvancedNews.js`** | Deep news reader powered by Scrapling, featuring live sentiment analysis, catalyst tags (Order Wins, Earnings Beat, Regulatory, Solvency), and sentiment filtering. |
| **`FundamentalsAnalysis.js`** | Institutional valuation suite: 10-step DCF, Graham Formula, Peter Lynch Fair Value, 3/5-Stage DuPont decomposition, Altman Z-Score, and Beneish M-Score. |
| **`LongTermAnalysis.js`** | Historical revenue/profit CAGR, balance sheet health, debt-to-equity trends, and return on equity trajectories for equities. |
| **`SmartCapitalAdvisor.js`** | Capital allocation advisor that analyzes underwater holdings, computes priority recovery weights, calculates shares to purchase, and exports actionable plans to CSV. |
| **`PortfolioTracker.js`** | Multi-asset portfolio tracker with live P&L tracking, sector diversification charts, and Markowitz portfolio rebalancing. |
| **`PortfolioMetrics.js`** | Quantitative risk metrics: Sharpe Ratio, Sortino Ratio, Beta, Treynor Ratio, Jensen's Alpha, and Maximum Historical Drawdown. |
| **`PeerComparison.js`** | Side-by-side relative valuation matrix comparing target stocks against sector peers across P/E, EV/EBITDA, ROE, ROCE, and Dividend Yield. |
| **`SectorIntelligence.js`** | Sectoral rotation analysis, top gainer/loser screener, and market-wide breadth indicators. |
| **`SIPCalculator.js`** | Interactive Systematic Investment Plan (SIP) wealth planner with step-up rate adjustments, inflation indexing, and compound interest breakdown. |
| **`WatchlistDrawer.js`** | Slide-out persistent watchlist drawer with real-time price polling, 24h change %, and quick asset selection. |
| **`ResearchReportModal.js`** | Executive research memo generator producing print-ready A4 investment reports in PDF and Markdown formats. |
| **`ErrorBoundary.js`** | Global React error boundary ensuring seamless error catching and graceful UI fallbacks. |
| **`InfoBadge.js`** | Contextual tooltip and badge component explaining complex financial and quantitative definitions. |

---

## ⚙️ Configuration & API Integration (`src/app/config.js`)

The frontend uses an intelligent API client resolver:
*   **Environment Variable Override**: If `NEXT_PUBLIC_API_URL` is set, it takes first priority.
*   **Localhost / LAN Auto-Detection**: When browsing on `localhost`, `127.0.0.1`, `192.168.x.x`, or `.local`, it automatically routes requests to `http://localhost:8000`.
*   **Production Cloud Fallback**: Automatically connects to the deployed backend microservice in production.

---

## 🛠️ Development & Build Commands

### Prerequisites
*   **Node.js 18+** (Node.js 20+ recommended)
*   **npm** or **pnpm**

### Install Dependencies
```bash
cd frontend
npm install
```

### Start Development Server (Turbopack)
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the application with hot module reloading.

### Production Build
```bash
npm run build
```
Compiles and optimizes all routes, running TypeScript and static page generation via Turbopack.

### Start Production Server
```bash
npm run start
```

### Run ESLint
```bash
npm run lint
```
Performs static code analysis and enforces React 19 / Next.js best practices.

---

## 🎨 Design System & Styling Architecture

*   **Asset-Adaptive Layout Engine**: The dashboard dynamically detects asset type (`EQUITY` vs `ETF`). For single stocks, it renders deep forensic accounting panels (DCF, DuPont, Graham, Altman Z). For ETFs, it seamlessly switches to the `ETFLongTermPanel` (AUM, expense ratio, tracking error vs benchmark, 6-point health checklist, SIP score, top 10 holdings, sector breakdown, and 3-year rolling returns distribution), preventing irrelevant corporate valuation metrics from polluting ETF analysis.
*   **Glassmorphic UI System**: Engineered using Tailwind CSS 3.4 with layered backdrop blur (`backdrop-blur-xl`), sub-pixel borders (`border-white/[0.06]`), and semi-transparent dark backgrounds (`bg-white/[0.02]`, `bg-white/[0.03]`).
*   **Curated Financial Color Palette**:
    *   **Backgrounds**: Deep obsidian canvas (`#050811`, `#0B0F19`, `#0F172A`).
    *   **Bullish / Growth**: Emerald gradient accents (`#10B981`, `#34D399`) with glow shadows (`rgba(52,211,153,0.35)`).
    *   **Bearish / Risk**: Rose gradient accents (`#F43F5E`, `#FB7185`) with glow shadows (`rgba(251,113,133,0.35)`).
    *   **Neutral / Caution**: Amber and violet highlights (`#F59E0B`, `#8B5CF6`) for consensus stability and SHAP factor attribution.
*   **Institutional Data Density**: High-density typography utilizing tabular numerals (`font-mono`, `tabular-nums`) to ensure zero-jitter price updates and precise alignment across high-resolution ultra-wide financial displays.
*   **Dynamic Micro-Interactions**: Smooth CSS transitions for SHAP factor impact bars, interactive candlestick tooltips, animated SIP progress rings, and printable research report themes.
