# StockIQ Pro — Frontend Quantitative Workstation 📈

> **High-Performance Institutional Financial Terminal Client**  
> Built with **Next.js 16 (Turbopack)**, **React 19**, **Recharts 3**, and **Tailwind CSS 3.4**.

<div align="center">

[![Next.js](https://img.shields.io/badge/Next.js_16_(Turbopack)-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS_3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Recharts](https://img.shields.io/badge/Recharts_3-22C55E?style=for-the-badge&logo=d3.js&logoColor=white)](https://recharts.org/)
[![FastAPI Backend](https://img.shields.io/badge/FastAPI_Client-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Universe Scale](https://img.shields.io/badge/Universe-7%2C954_Instruments-8B5CF6?style=for-the-badge)](https://stockiq-pro.vercel.app/browse)
[![Deployment](https://img.shields.io/badge/Deployed_on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/)

<br/>

**[🌐 Live Workstation](https://stockiq-pro.vercel.app)** · **[📖 Master Architecture Blueprint](../docs/architecture/ARCHITECTURE.md)** · **[⚡ Quick Start Guide](../docs/guides/quick_start.md)** · **[🚀 Deployment Guide](../docs/guides/deployment.md)**

</div>

---

## 🌟 Executive Overview

The **StockIQ Pro** frontend is an institutional-grade, dark-themed, glassmorphic financial terminal client designed to bring Wall Street and Dalal Street quantitative desk capabilities directly to individual investors and quantitative researchers.

It interfaces dynamically with the **FastAPI backend microservice**, delivering real-time pricing, multi-model machine learning forecasts, deep full-article news intelligence (powered by Scrapling), Modern Portfolio Theory optimization, high-frequency intraday technical charting, and long-term ETF analytics across **7,954 instruments** (NSE, BSE, ETFs, sectoral indices, and global ADRs).

```
                      STOCKIQ PRO FRONTEND ARCHITECTURE
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                           Next.js 16 Client Tier                            │
 │                                                                             │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │        Header & Spotlight Palette (⌘K) — 7,954 Master Universe        │  │
 │  └───────────────────────────────────┬───────────────────────────────────┘  │
 │                                      ▼                                      │
 │  ┌───────────────────────────────────────────────────────────────────────┐  │
 │  │                  Asset-Adaptive Layout Dispatcher                     │  │
 │  │          (Auto-detects assetType: 'EQUITY' vs 'ETF' vs 'INDEX')        │  │
 │  └───────────────────┬───────────────────────────────────┬───────────────┘  │
 │                      ▼                                   ▼                  │
 │      ┌───────────────────────────────┐   ┌───────────────────────────────┐  │
 │      │       EQUITY WORKSTATION      │   │       ETF LONG-TERM SUITE     │  │
 │      │  • 10-Step DCF Intrinsic Val  │   │  • AUM & Expense Ratio        │  │
 │      │  • Graham & Peter Lynch Fair  │   │  • Tracking Error vs Bench    │  │
 │      │  • 3/5-Stage DuPont Analysis  │   │  • 1Y / 3Y / 5Y CAGR & Alpha  │  │
 │      │  • Altman Z & Beneish M-Score │   │  • Top 10 Holdings & Weights  │  │
 │      │  • Piotroski F-Score (0-9)    │   │  • Sector Breakdown Gradients │  │
 │      │  • 5-Day ML Return & News     │   │  • 3-Year Rolling Returns     │  │
 │      │  • Street Consensus vs AI     │   │  • 6-Point Health Checklist   │  │
 │      │  • Black-Scholes Greeks       │   │  • SIP Suitability (0-10)     │  │
 │      └───────────────────────────────┘   └───────────────────────────────┘  │
 │                                      │                                      │
 │  ┌───────────────────────────────────┴───────────────────────────────────┐  │
 │  │                     Shared Quantitative Infrastructure                │  │
 │  │  • 6-Model ML Ensemble (RF, ET, GB, XGBoost, LightGBM, Bayesian Ridge) │  │
 │  │  • SHAP Waterfall Game-Theory Feature Attribution                     │  │
 │  │  • GARCH(1,1) Econometric Volatility & 3-State Markov Regimes (HMM)   │  │
 │  │  • Monte Carlo Stochastic Simulation (1,000 Geometric Brownian Paths) │  │
 │  │  • Scrapling Deep News Reader with Sentiment & Financial Catalysts    │  │
 │  │  • Executive Research Memo Generator (Print-Ready A4 PDF & Markdown)   │  │
 │  └───────────────────────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Application Route Architecture

StockIQ Pro uses the Next.js App Router (`src/app/`) to deliver 6 dedicated, modular workspaces:

| Route | Workspace Name | Core Capabilities & User Experience |
| :--- | :--- | :--- |
| **`/`** | **Analysis Workstation** | Master terminal with asset-adaptive architecture. Automatically toggles between deep corporate forensic valuation (DCF, DuPont, Graham, Altman Z, Beneish M) for equities, and the institutional ETF Long-Term Suite (AUM, expense ratio, tracking error, top 10 holdings, sector breakdown, 3Y rolling returns, 6-point checklist, SIP score) for ETFs. Features 6-model ML forecasts, Street Consensus comparisons, GARCH volatility, Monte Carlo simulations, options Greeks, and executive PDF memos. |
| **`/intraday`** | **Intraday Terminal** | High-frequency technical analysis workspace with real-time multi-timeframe candles (1m, 2m, 5m, 15m, 30m, 60m), Supertrend directional signals ($\pm 1$), Multi-timeframe VWAP with volatility bands ($\pm 1\sigma, \pm 2\sigma$), EMA Ribbons (9, 21, 50, 200), RSI-14, MACD (12, 26, 9), ATR-14, and one-click CSV time-series export. |
| **`/portfolio`** | **Portfolio & Capital Advisor** | Modern Portfolio Theory (Markowitz Efficient Frontier), Tangency Portfolio (Maximum Sharpe), Global Minimum Variance (GMV) allocation, Value-at-Risk (VaR 95%, 99%), Conditional VaR (Expected Shortfall), dynamic portfolio rebalancing, the **Smart Capital Advisor** for automated loss recovery and strategic averaging down, and the **SIP & Wealth Planner**. |
| **`/browse`** | **Market Universe Browser** | Search, filter, and explore the complete 7,954 instrument database across NSE equities, BSE scrips, ETFs, sectoral indices, and global ADRs with market cap, sector tags, and live price feeds. |
| **`/features`** | **Architecture & Capabilities** | Interactive technical showcase demonstrating the underlying mathematical models, Scrapling stealth web scraping, ensemble ML algorithms, and sub-millisecond system performance benchmarks. |
| **`/terms`** | **Terms & Disclaimers** | Complete platform terms of service, open-source licensing, and SEBI regulatory risk disclaimers. |

---

## ⚡ The Dual-Mode Asset-Adaptive Engine

A fundamental flaw of conventional financial tools is evaluating ETFs as if they were corporate businesses. Treating an ETF like a single company results in nonsensical DCF models, broken earnings manipulation scores, and meaningless corporate news feeds.

StockIQ Pro solves this with an **Asset-Adaptive Layout Engine**:

```
                       INCOMING ASSET DISPATCH LOGIC
                                     │
                        [ User Selects Ticker ]
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    Asset Type Detection     │
                      │   (EQUITY vs ETF vs INDEX)  │
                      └──────────────┬──────────────┘
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
   [ assetType === 'EQUITY' ]                         [ assetType === 'ETF' ]
            │                                                 │
  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐
  │   Corporate Forensics Engine    │               │     ETF Long-Term Analytics     │
  ├─────────────────────────────────┤               ├─────────────────────────────────┤
  │ • 10-Step DCF Intrinsic Value   │               │ • AUM Liquidity ($500M / ₹500Cr)│
  │ • Graham Formula Benchmark      │               │ • Expense Ratio (TER %)         │
  │ • Peter Lynch Fair Value        │               │ • Annualised Tracking Error     │
  │ • 3 & 5-Stage DuPont Breakdown  │               │ • 1Y / 3Y / 5Y CAGR vs Benchmark│
  │ • Altman Z-Score (Bankruptcy)   │               │ • 3-Year Sharpe Ratio           │
  │ • Beneish M-Score (Accounting)  │               │ • Top 10 Underlying Holdings    │
  │ • Piotroski F-Score (Health)    │               │ • Sector Weightings Breakdown   │
  │ • 5-Day ML Horizon + News NLP   │               │ • 3-Year Rolling Returns & Win% │
  │ • Street Consensus vs AI Target │               │ • 6-Point Health Checklist      │
  │ • Black-Scholes Options Greeks  │               │ • SIP Suitability Score (0-10)  │
  └─────────────────────────────────┘               └─────────────────────────────────┘
```

### Direct Comparison Matrix:

| Analytical Dimension | Equity Mode (`assetType === 'EQUITY'`) | ETF Mode (`assetType === 'ETF'`) |
| :--- | :--- | :--- |
| **Primary Valuation Panel** | `FundamentalsAnalysis.js` (DCF, Graham, Lynch, DuPont) | `ETFLongTermPanel.js` (AUM, TER, Tracking Error, CAGR) |
| **Forensic Accounting** | Altman Z-Score & Beneish M-Score manipulation tests | **Hidden** (Not applicable to index baskets) |
| **Portfolio Transparency** | Peer comparison matrix vs sector competitors | **Top 10 Holdings & Sector Breakdown** cards |
| **Long-Term Return Engine** | Historical financial CAGR (Revenue, Net Profit, ROE) | **3-Year Rolling Returns Distribution** (480+ periods) |
| **Investment Suitability** | Piotroski F-Score & Debt-to-Equity trajectory | **0–10 SIP Suitability Score** & 6-Point Checklist |
| **ML Forecast Horizon** | 5 Trading Days (`target_return_5d`) | **30 Trading Days** (`target_return_30d`) |
| **ML Feature Tensor** | 40+ short-term technicals, order flow, momentum | **38 long-term features** (Golden Cross, 63d momentum) |
| **ML Signals** | `STRONG BUY`, `BUY`, `HOLD`, `SELL`, `STRONG SELL` | `ACCUMULATE`, `HOLD`, `AVOID` |
| **News Sentiment Integration**| Scrapling full-article catalyst & sentiment fusion | **Neutral / Zeroed** (single-company news is noise) |
| **Institutional Consensus** | **Street Analyst Consensus vs AI Target** comparison | **Hidden** (sell-side coverage applies to equities) |

---

## 🧩 Component Directory (`src/app/components/`)

The component architecture is divided into specialized, cohesive functional suites:

### 1. 🔍 Navigation & Search Suite
*   **`Header.js`**: Sticky global navigation bar featuring real-time market status indicators (NSE/BSE open/closed status), active route indicator, responsive mobile menu drawer, and the Spotlight search trigger.
*   **`StockSearchModal.js`**: Universal Spotlight command palette (<kbd>⌘K</kbd> / <kbd>/</kbd>) backed by our 5-layer search engine:
    *   *Categorized Tabs*: `All`, `Equities`, `ETFs`, `Indices`, `Global`.
    *   *BSE Code Resolution*: Resolves 6-digit scrips (e.g. `500325` $\rightarrow$ `RELIANCE.BO`).
    *   *Corporate Aliases*: Resolves `"sbi"`, `"tata motors"`, `"gold etf"`, `"zomato"`.
    *   *Levenshtein Typo Tolerance*: Fuzzy string distance matching ($\ge 0.78$).
    *   *Keyboard Navigation*: Full arrow-key selection, Escape to dismiss, Enter to select.
*   **`WatchlistDrawer.js`**: Persistent slide-out monitoring drawer with live price polling, daily change %, quick symbol switching, and local storage persistence.

### 2. 🧠 Quantitative ML & Forecasting Suite
*   **`MLPrediction.js`**: 6-model stacked ensemble meta-blender (Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, Bayesian Ridge):
    *   *SHAP Waterfall Attribution*: Visualizes top positive and negative value-driving features based on Shapley game-theory values.
    *   *Econometric Volatility & Regimes*: Surfaces GARCH(1,1) conditional volatility forecasts and 3-state Hidden Markov Model (HMM) regime alerts.
    *   *Street Analyst Consensus vs AI Ensemble*: Side-by-side comparison of sell-side consensus mean target vs 6-model AI ensemble target, including analyst count, recommendation rating, and quantitative divergence spread.
    *   *Asset-Adaptive Signals*: Automatically outputs `ACCUMULATE/AVOID` over 30 days for ETFs, and `BUY/SELL` over 5 days for equities.
*   **`MonteCarloSimulation.js`**: Stochastic price trajectory engine generating 1,000 Geometric Brownian Motion (GBM) paths:
    *   *Horizons*: 30, 60, 90, 180, and 365 trading days.
    *   *Quantile Corridors*: Visualizes $P_{2.5}, P_{25}, P_{50}, P_{75}, P_{97.5}$ confidence envelopes.
    *   *Export*: Full percentile distribution downloadable via CSV.

### 3. 📊 ETF Long-Term Intelligence Suite
*   **`ETFLongTermPanel.js`**: Dedicated institutional ETF evaluation panel:
    *   *Underlying Portfolio Decomposition*: Interactive toggle between **Top 10 Holdings** (with asset weight bars) and **Sector Breakdown** (with gradient bars), backed by live Yahoo Finance data and a curated Indian ETF database (`_ETF_CURATED_HOLDINGS`).
    *   *3-Year Rolling Returns Distribution*: Computes annualized 3-year rolling returns across 5 years of daily data (480+ periods). Displays Median CAGR, Minimum 3Y CAGR, Maximum 3Y CAGR, Current 3Y CAGR, and a **Probability of Profit Pill** (e.g. `100.0% Profitable`).
    *   *6-Point Institutional Health Checklist*: Evaluates AUM liquidity ($> ₹500\,\text{Cr}$), Expense Ratio ($\le 0.50\%$), Tracking Error ($\le 0.50\%$), CAGR Alpha ($\ge 0\%$), 3Y Sharpe ($\ge 0.5$), and NAV Premium/Discount ($|\Delta| \le 1.0\%$).
    *   *SIP Suitability Score (0 to 10)*: Algorithmic rating evaluating recurring investment safety.
    *   *Performance Comparison*: 1Y, 3Y, and 5Y CAGR vs primary benchmark with alpha calculation.
    *   *Historical Annual Returns*: Yearly performance breakdown sparklines.

### 4. ⚖️ Forensic Accounting & Valuation Suite
*   **`FundamentalsAnalysis.js`**: Institutional forensic accounting and intrinsic valuation:
    *   *Discounted Cash Flow (DCF)*: 10-step multi-stage projection with customizable WACC and terminal growth rates.
    *   *Classic Benchmarks*: Benjamin Graham Fair Value Formula ($\sqrt{22.5 \times EPS \times BVPS}$) and Peter Lynch Fair Value.
    *   *DuPont Decomposition*: 3-stage and 5-stage return on equity breakdown (tax burden, interest burden, operating margin, asset turnover, financial leverage).
    *   *Risk Forensics*: Altman Z-Score (bankruptcy distress) and Beneish M-Score (earnings manipulation detection).
    *   *Piotroski F-Score*: 9-point fundamental health index.
*   **`LongTermAnalysis.js`**: Multi-year CAGR trajectories across revenue, operating profit, and net income, alongside balance sheet leverage and debt-to-equity trends.
*   **`PeerComparison.js`**: Relative valuation screener benchmarking the target asset against its direct sector peers across P/E, EV/EBITDA, ROE, ROCE, and Dividend Yield.

### 5. 📈 High-Frequency Technicals & Charting Suite
*   **`StockChart.js`**: Primary interactive candlestick and line chart with volume histograms, 20/50/200 EMA overlays, Bollinger Bands, and responsive tooltips.
*   **`IntradayTerminal.js`**: Dedicated high-frequency trading workspace:
    *   *Real-Time Candle Feeds*: 1m, 2m, 5m, 15m, 30m, and 60m resolution.
    *   *Supertrend Indicator*: Trend-following volatility filter ($ATR \times 3.0$) with directional signals (+1 Bullish / -1 Bearish).
    *   *Multi-Timeframe VWAP*: Volume-Weighted Average Price with $\pm 1\sigma$ and $\pm 2\sigma$ standard deviation bands.
    *   *EMA Ribbons*: 9, 21, 50, and 200 exponential moving average ribbon overlays.
    *   *Oscillator Sub-Charts*: RSI-14, MACD (12, 26, 9) histogram, and ATR-14.
    *   *One-Click CSV Export*: Downloads raw OHLCV bars and indicator columns.

### 6. 💼 Portfolio Construction & Capital Allocation Suite
*   **`PortfolioTracker.js`**: Multi-asset portfolio tracker with live P&L tracking, portfolio weight breakdown, and Markowitz portfolio rebalancing.
*   **`PortfolioMetrics.js`**: Quantitative risk metrics: Sharpe Ratio, Sortino Ratio, Beta, Treynor Ratio, Jensen's Alpha, and Maximum Historical Drawdown.
*   **`SmartCapitalAdvisor.js`**: Automated capital allocation advisor for underwater holdings:
    *   Calculates dynamic recovery priority scores based on technical oversold conditions, margin of safety, and market cap.
    *   Generates actionable purchase allocations: allocated ₹ amount, target shares, and new break-even price.
    *   One-click CSV allocation plan download.
*   **`SIPCalculator.js`**: Interactive Systematic Investment Plan wealth planner with annual step-up multipliers, inflation indexing, and compound wealth milestone projections.

### 7. 📰 News Intelligence & Research Reporting Suite
*   **`AdvancedNews.js`**: Deep news intelligence powered by Scrapling:
    *   Bypasses Cloudflare/Akamai bot detection to extract full article bodies.
    *   Categorizes corporate catalysts: `Order Wins`, `Earnings Beat`, `Deleveraging`, `Regulatory Scrutiny`.
    *   Sentiment scoring ($-1.0$ to $+1.0$) and Market Impact Score ($0$ to $100$).
    *   Ad-stripped, clean full-article reader modal.
*   **`SectorIntelligence.js`**: Sectoral rotation matrix, market breadth indicators, and top gainer/loser screener.
*   **`ResearchReportModal.js`**: Institutional investment memo generator producing print-ready A4 reports in both PDF and Markdown formats with thesis, valuation models, risk disclosures, and quantitative charts.

### 8. 🛡️ UI Infrastructure & Contextual Badging
*   **`ErrorBoundary.js`**: Global React error boundary ensuring seamless error catching and graceful UI fallbacks.
*   **`InfoBadge.js`**: Contextual tooltip and badge explaining complex quantitative and financial definitions on hover/click.

---

## 🎨 Institutional Design System & Glassmorphic Tokens

The frontend design system is engineered with **Tailwind CSS 3.4** to deliver an immersive, zero-distraction financial terminal:

```
                               CURATED COLOR PALETTE
 ┌──────────────────────┬──────────────────────┬──────────────────────┐
 │   Deep Obsidian      │   Emerald Bullish    │     Rose Bearish     │
 │   Canvas (#050811)   │   Growth  (#10B981)  │     Risk   (#F43F5E) │
 ├──────────────────────┼──────────────────────┼──────────────────────┤
 │   Slate Surface      │   Amber Caution      │   Violet Consensus   │
 │   Card   (#0F172A)   │   Notice  (#F59E0B)  │   Stability (#8B5CF6)│
 └──────────────────────┴──────────────────────┴──────────────────────┘
```

*   **Obsidian Canvas**: `#050811` (canvas), `#0B0F19` (elevated background), `#0F172A` (surface cards).
*   **Glassmorphism Tokens**:
    *   Backdrop blur: `backdrop-blur-xl`
    *   Sub-pixel borders: `border-white/[0.06]`, `border-white/[0.10]`
    *   Semi-transparent fills: `bg-white/[0.02]`, `bg-white/[0.03]`, `bg-white/[0.05]`
*   **Financial Color Encoding**:
    *   **Bullish / Growth**: Emerald `#10B981` / `#34D399` with glow shadows (`rgba(52,211,153,0.35)`).
    *   **Bearish / Risk**: Rose `#F43F5E` / `#FB7185` with glow shadows (`rgba(251,113,133,0.35)`).
    *   **Caution / Warning**: Amber `#F59E0B` for elevated volatility or neutral signals.
    *   **Consensus / Attribution**: Violet `#8B5CF6` for SHAP feature attribution and Street consensus alignment.
*   **Zero-Jitter Typography**: Tabular numerals (`font-mono`, `tabular-nums`) across all prices, weights, returns, and Greek metrics to eliminate layout shift during live polling.
*   **Micro-Interactions**: Smooth CSS transitions for SHAP factor impact bars, interactive candlestick tooltips, animated SIP progress rings, and printable research report themes.

---

## ⚙️ Configuration & Network Architecture (`src/app/config.js`)

The frontend uses an auto-resolving network client designed for multi-environment resilience:

```javascript
// Dynamic API URL Resolution Logic:
1. Process Environment: process.env.NEXT_PUBLIC_API_URL (Explicit override)
2. Localhost Auto-Detect: If window.location is 'localhost', '127.0.0.1', '192.168.x.x', or '.local'
   → Routes automatically to http://localhost:8000
3. Production Fallback: https://stock-analysis-backend-seven.vercel.app
```

### Network Resilience Features:
*   **Request Cancellation**: Uses `AbortController` signals to cancel inflight requests when users rapidly switch tickers, preventing race conditions.
*   **Graceful Degradation**: Endpoints return structured error payloads without crashing the UI, handled gracefully by `ErrorBoundary.js`.
*   **Security Middleware (`src/middleware.js`)**: Enforces strict Content Security Policy (CSP), `X-Frame-Options: DENY`, and `X-Content-Type-Options: nosniff`.

---

## 📁 Source Directory Structure

```
frontend/
├── public/                      # Static assets, icons, and logos
├── src/
│   ├── app/
│   │   ├── components/          # 21 modular functional components
│   │   │   ├── AdvancedNews.js           # Scrapling deep news reader
│   │   │   ├── ErrorBoundary.js          # Global React error boundary
│   │   │   ├── ETFLongTermPanel.js       # ETF analytics, holdings & rolling returns
│   │   │   ├── FundamentalsAnalysis.js   # DCF, Graham, DuPont, Altman Z, Beneish M
│   │   │   ├── Header.js                 # Global navigation & live market status
│   │   │   ├── InfoBadge.js              # Quantitative definition tooltips
│   │   │   ├── IntradayTerminal.js       # High-frequency candles, Supertrend, VWAP
│   │   │   ├── LongTermAnalysis.js       # Multi-year corporate growth & balance sheet
│   │   │   ├── MLPrediction.js           # 6-model ML ensemble & Street consensus
│   │   │   ├── MonteCarloSimulation.js   # 1,000 GBM stochastic price paths
│   │   │   ├── PeerComparison.js         # Relative valuation matrix vs peers
│   │   │   ├── PortfolioMetrics.js       # Sharpe, Sortino, Treynor, Alpha, Beta
│   │   │   ├── PortfolioTracker.js       # Multi-asset tracker & rebalancing
│   │   │   ├── ResearchReportModal.js    # Print-ready A4 PDF & Markdown memos
│   │   │   ├── SectorIntelligence.js     # Sector rotation & market breadth
│   │   │   ├── SIPCalculator.js          # Wealth planner & step-up SIP curves
│   │   │   ├── SmartCapitalAdvisor.js    # Automated averaging down for recoveries
│   │   │   ├── StockChart.js             # Interactive candlestick chart with EMAs
│   │   │   ├── StockSearchModal.js       # Spotlight ⌘K search across 7,954 assets
│   │   │   └── WatchlistDrawer.js        # Persistent multi-stock watchlist drawer
│   │   ├── browse/
│   │   │   └── page.js                   # Market universe browser (7,954 assets)
│   │   ├── features/
│   │   │   └── page.js                   # System architecture & capabilities showcase
│   │   ├── intraday/
│   │   │   └── page.js                   # Dedicated intraday trading terminal route
│   │   ├── portfolio/
│   │   │   └── page.js                   # Modern Portfolio Theory & Capital Advisor
│   │   ├── terms/
│   │   │   └── page.js                   # Platform terms & SEBI risk disclosures
│   │   ├── config.js                     # Multi-environment API client configuration
│   │   ├── globals.css                   # Global styles & glassmorphic tokens
│   │   ├── layout.js                     # Root layout with font optimization
│   │   ├── page.js                       # Master quantitative workstation route (/)
│   ├── middleware.js                     # Security headers & CSP enforcement
├── package.json                          # Dependencies & scripts
├── tailwind.config.js                    # Tailwind styling configuration
└── vercel.json                           # Production deployment configuration
```

---

## ⌨️ Keyboard Shortcuts & Power User Cheatsheet

| Shortcut | Action | Description |
| :--- | :--- | :--- |
| <kbd>⌘K</kbd> / <kbd>Ctrl+K</kbd> | **Open Spotlight Search** | Universal asset palette across 7,954 instruments |
| <kbd>/</kbd> | **Quick Search** | Activates the search palette from anywhere on the page |
| <kbd>Esc</kbd> | **Dismiss Modal / Drawer** | Closes Spotlight, Watchlist, News Reader, or Memo Modal |
| <kbd>↑</kbd> / <kbd>↓</kbd> | **Navigate Results** | Moves selection cursor through search results |
| <kbd>Enter</kbd> | **Select Asset** | Loads the highlighted stock or ETF into the workstation |
| <kbd>⌘P</kbd> / <kbd>Ctrl+P</kbd> | **Print Research Memo** | Generates an institutional print-ready A4 PDF report |

---

## 🛠️ Development & Build Commands

### Prerequisites
*   **Node.js 18+** (Node.js 20+ recommended)
*   **npm** (or **pnpm**)

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server (Turbopack)
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) to view the workstation with instant Turbopack hot module reloading.

### 3. Run Static Code Analysis & Linter
```bash
npm run lint
```
Enforces React 19 / Next.js best practices and validates JSX syntax (currently **0 errors**).

### 4. Build for Production
```bash
npm run build
```
Compiles and optimizes all routes, executing TypeScript checks and static page generation via Turbopack.

### 5. Start Production Server Locally
```bash
npm run start
```
Spins up the optimized production server at [http://localhost:3000](http://localhost:3000).

---

## 🚀 Production Deployment (Vercel)

The frontend is fully configured for zero-configuration deployment on **Vercel**:

1. Push your changes to GitHub:
   ```bash
   git push origin main
   ```
2. Import the repository in [Vercel](https://vercel.com):
   *   **Framework Preset**: `Next.js`
   *   **Root Directory**: `frontend`
   *   **Build Command**: `npm run build`
   *   **Output Directory**: `.next`
3. Configure Environment Variables:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend-api.vercel.app
   NEXT_PUBLIC_APP_ENV=production
   ```
4. Click **Deploy**.

---

<div align="center">

**StockIQ Pro — Engineered for Quantitative Precision.**

Made with ❤️ by Vishesh Sanghvi

[Live Workstation](https://stockiq-pro.vercel.app) · [Backend API](https://stock-analysis-backend-seven.vercel.app) · [GitHub Repository](https://github.com/visheshsanghvi112/Analysis-tool)

</div>
