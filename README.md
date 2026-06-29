<div align="center">

![StockIQ Pro Banner](./stockiq_pro_banner.png)

# StockIQ Pro 📈

### _Institutional-grade stock intelligence for the individual Indian investor_

**The same analytical firepower used by hedge funds and quant desks —  
made free, open-source, and built for NSE & BSE.**

<br/>

[![Next.js](https://img.shields.io/badge/Next.js_16-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python_3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com/)

<br/>

**[🌐 Live Demo](https://stockiq-pro.vercel.app)** · **[📖 API Docs](https://stock-analysis-backend-seven.vercel.app/docs)** · **[🐛 Report Bug](https://github.com/visheshsanghvi112/Analysis-tool/issues)**

</div>

---

## The Problem: You're Investing Blind

You open Zerodha or Groww. You see a price chart and a buy button.  
But **what is that price actually telling you?**

- Is the stock overbought or building momentum?
- Is the recent rally backed by volume or just noise?
- What does the market *feel* about this stock right now — fear or greed?
- Is the stock fairly valued, or are you overpaying for it?
- What is this company's intrinsic worth based on its cash flows?
- Is management efficiently converting revenue into shareholder returns?
- How much could you lose on a bad day — statistically?

**Most retail investors don't have answers to these questions.** Not because they're not smart — but because the tools that answer them cost ₹2,00,000+/year (Bloomberg Terminal), require a CFA to interpret, or simply don't exist for Indian markets.

**StockIQ Pro changes that.**

---

## What StockIQ Pro Does For You As An Investor

Imagine you're considering buying HDFC Bank. Here's what StockIQ Pro gives you in under 60 seconds:

```
✅ Live price with intraday range — where is it trading right now?
✅ Full technical picture — RSI, MACD, Bollinger Bands, ADX on one dashboard
✅ 5-day AI price prediction — where is it likely headed?
✅ Market sentiment — what is the news saying, and does it support the trade?
✅ Risk profile — how much can I lose? What's my risk-adjusted return?
✅ Strategy backtest — if I had followed this signal in the past, would I have made money?
✅ Peer comparison — is HDFC Bank actually better than ICICI Bank right now?
✅ DCF intrinsic value — what is this stock actually worth? Am I getting a discount?
✅ DuPont ROE breakdown — is the company profitable because it's efficient, or just leveraged?
✅ Financial health score — does this pass the 10-criteria long-term investor checklist?
✅ Portfolio recovery advisor — which of my losing stocks should I average down on?
```

This is not a simple screener. This is a **decision-support system** — built for investors who want to go beyond price and make informed, evidence-based decisions.

---

## How It's Different

| Feature | StockIQ Pro | Zerodha / Groww | Moneycontrol | Bloomberg |
|---|---|---|---|---|
| ML Price Predictions | ✅ 5-model ensemble | ❌ | ❌ | ✅ (₹2L/yr) |
| Explainable AI (SHAP) | ✅ See *why* the model predicted | ❌ | ❌ | ❌ |
| Signal Backtesting | ✅ RSI+MACD strategy, equity curve | ❌ | ❌ | ✅ (₹2L/yr) |
| Market Regime (HMM) | ✅ 3-state Hidden Markov Model | ❌ | ❌ | Limited |
| GARCH Volatility Forecast | ✅ 5-day forward vol | ❌ | ❌ | ✅ (₹2L/yr) |
| News Sentiment AI | ✅ Per-article impact scoring | ❌ | Manual tags | ✅ |
| Options Greeks | ✅ Full Black-Scholes + Greeks | Limited | Limited | ✅ |
| VaR / Expected Shortfall | ✅ 95% & 99% | ❌ | ❌ | ✅ |
| **DCF Valuation Model** | ✅ **Interactive sliders, real-time** | ❌ | Basic | ✅ |
| **DuPont ROE Decomposition** | ✅ **3-factor breakdown** | ❌ | ❌ | ✅ |
| **Financial Health Score** | ✅ **10-criteria checklist** | ❌ | ❌ | ✅ |
| **Graham Number** | ✅ **Benjamin Graham formula** | ❌ | ❌ | Limited |
| **Portfolio Recovery Advisor** | ✅ **RSI + sentiment + avg-down calc** | ❌ | ❌ | ❌ |
| Sector Peer Ranking | ✅ Composite score vs peers | Basic | Basic | ✅ |
| NSE Coverage | ✅ 1,900+ stocks | ✅ | ✅ | ✅ |
| **Cost** | **Free & Open Source** | Free (basic) | Free (ads) | ₹2,00,000/yr |

---

## Feature Deep-Dive

### 💰 Long-Term Investment & Valuation Hub _(New)_

> _"Price is what you pay. Value is what you get." — Warren Buffett_

The most important question in investing isn't "where is the price going?" — it's **"what is this business worth?"** StockIQ Pro now answers this with four interconnected models:

#### 1. Interactive DCF Intrinsic Value Calculator

A fully dynamic **Discounted Cash Flow model** that lets you explore valuation scenarios in real time:

- **Starting Cash Flow**: Switch between Free Cash Flow, Net Income, or Operating Cash Flow as your baseline
- **Growth Rate** slider (0–30%): Adjust the projected 5-year growth assumption
- **WACC** slider (5–20%): Set your weighted average cost of capital (pre-filled via CAPM: Risk-free rate + Beta × Equity Risk Premium)
- **Terminal Growth Rate** slider (1–8%): Set the perpetuity growth rate after year 5

The model computes:
```
Enterprise Value = Σ (FCF × (1+g)^t / (1+d)^t) for t=1..5 + Terminal Value / (1+d)^5
Equity Value    = Enterprise Value + Cash − Debt
Intrinsic Value = Equity Value / Shares Outstanding
Margin of Safety = (Intrinsic Value − Market Price) / Intrinsic Value
```

The **Margin of Safety** badge turns green (undervalued), yellow (fair), or red (overvalued) in real time as you move sliders — giving you immediate visual feedback on your assumptions.

#### 2. DuPont ROE Decomposition

Breaks **Return on Equity (ROE)** into its three fundamental drivers using the DuPont Identity:

```
ROE = Net Profit Margin × Asset Turnover × Equity Multiplier
    = (Net Income / Revenue) × (Revenue / Assets) × (Assets / Equity)
```

| Component | What it reveals |
|---|---|
| **Net Profit Margin** | How much of each rupee of revenue becomes profit (pricing power & cost control) |
| **Asset Turnover** | How efficiently the company uses its assets to generate revenue |
| **Equity Multiplier** | How much of the balance sheet is funded by debt (financial leverage) |

This decomposition tells you *why* a company has a high or low ROE — whether it's because they're operationally excellent, asset-light, or simply highly leveraged.

#### 3. Graham Defensive Valuation Number

Based on Benjamin Graham's formula from *The Intelligent Investor*:

```
Graham Number = √(22.5 × EPS × Book Value Per Share)
```

This represents the **maximum price a defensive investor should pay** for a stock. If the current market price exceeds the Graham Number, the stock is trading at a premium over its fundamental defensive value.

#### 4. 10-Criteria Long-Term Financial Health Score

A transparent, checklist-based scoring system (0–10) that evaluates:

| # | Criterion | Threshold |
|---|---|---|
| 1 | Return on Equity (ROE) | ≥ 12% |
| 2 | Return on Assets (ROA) | ≥ 5% |
| 3 | Net Profit Margin | ≥ 8% |
| 4 | Debt to Equity Ratio | ≤ 1.0x |
| 5 | Current Ratio | ≥ 1.2x |
| 6 | Free Cash Flow | > 0 |
| 7 | Price to Earnings (P/E) | < 30x |
| 8 | Promoter / Insider Holding | ≥ 40% |
| 9 | Revenue Growth (YoY) | ≥ 8% |
| 10 | Earnings Growth (YoY) | ≥ 5% |

A score of **8–10** = Strong long-term candidate. **5–7** = Watchlist. **0–4** = Caution.

---

### 🧠 AI Price Prediction — 6-Model Stacked Ensemble

> _"Single models fail in finance. Markets are noisy, non-linear, and regime-dependent. The solution is ensemble stacking."_

StockIQ Pro trains **5 diverse base learners** simultaneously on 2 years of daily data, then combines them using a **Ridge meta-learner** that learns the optimal weight for each model's output:

| Model | What it captures |
|---|---|
| **Random Forest** | Non-linear price patterns via bagging |
| **Gradient Boosting** | Sequential error correction |
| **XGBoost** | Regularized tree boosting (L1/L2) |
| **LightGBM** | Leaf-wise growth for deep feature interactions |
| **Extra Trees** | High-variance reduction via extreme randomization |
| **Ridge Meta-Learner** | Combines all 5, prevents multicollinearity |

**40+ engineered features** feed into the ensemble:
- **Momentum**: RSI(14), Williams %R, Stochastic %K/%D
- **Trend**: MACD, EMA Cross (9/21), MA Ratios (5/10/20/50/100d)
- **Volatility**: ATR Ratio, Bollinger Band Position & Width, 20d/60d σ
- **Volume**: Volume Ratio, OBV Ratio
- **Statistical**: Rolling Skewness & Kurtosis (20d), 52-week high/low proximity
- **Calendar**: Day-of-week effect, month effect
- **Lagged returns**: 1d, 2d, 3d, 5d, 10d lags

**News sentiment fusion** — the final prediction is an 80/20 blend of ML signal and live news sentiment, so breaking news nudges the model the same way it moves real markets.

---

### 🔍 SHAP Explainability — _Why_ Did the Model Predict That?

Most AI tools are black boxes. StockIQ Pro is not.

Every prediction comes with a **SHAP (SHapley Additive exPlanations) waterfall chart** showing exactly which features pushed the model toward a BUY signal and which pushed it toward SELL:

```
▲ RSI (14)              +0.0312   ████████████████████ → Bullish push
▲ Bollinger Band Pos    +0.0187   ████████████         → Bullish push  
▼ MACD Histogram        -0.0241   ███████████████████  → Bearish push
▲ Volume Ratio          +0.0094   ██████               → Bullish push
▼ 20d Volatility        -0.0156   ████████████         → Bearish push
```

**Net Bullish → BUY signal.** Now you understand *why*, not just *what*.

---

### 🏛️ Market Regime Detection (Hidden Markov Model)

Markets don't behave the same every day. There are distinct **structural regimes**:
- 📈 **Low Volatility** — stable bull market, trending up
- ➡️ **Medium Volatility** — sideways/neutral, consolidating
- 📉 **High Volatility** — panic, selling pressure, high risk

StockIQ Pro fits a **3-state Gaussian Hidden Markov Model** directly to a stock's log-return series. The HMM learns transition probabilities between states and classifies which regime the stock is currently in — **without any hardcoded rules**.

**Why this matters for you:** In a High Volatility regime, the prediction thresholds widen automatically (from 1.8% to 3.0% expected return required to trigger a signal). This prevents false buy signals in choppy markets.

---

### 📊 GARCH(1,1) Volatility Forecasting

Volatility clusters — calm periods are followed by calm periods, and turbulent periods by turbulent ones. **GARCH(1,1)** models this conditional variance:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

StockIQ Pro fits GARCH to each stock's return series and forecasts **expected annualized volatility over the next 5 days**. This forward-looking vol estimate is displayed alongside the prediction — so you know not just the direction, but how rough the ride might be.

---

### 🧪 Signal Backtesting — Did This Strategy Actually Work?

> _"Past performance doesn't guarantee future results — but understanding the past is the only rational basis for future decisions."_

The backtesting engine runs a **RSI(14) + MACD Crossover + ATR Stop-Loss** strategy on historical data and shows:

**Entry signal:** RSI crosses above 35 from oversold + MACD line above signal line  
**Exit signal:** RSI reaches 65 (overbought) OR MACD turns bearish OR price drops below 2×ATR stop

**What you get:**
- 📈 **Equity curve** — your ₹1,00,000 vs buy-and-hold vs Nifty 50, plotted over time
- 📊 **Alpha** — did the strategy beat simply holding the stock?
- 🏆 **Sharpe Ratio** — risk-adjusted return
- 📉 **Max Drawdown** — worst-case loss from peak
- 🎯 **Calmar Ratio** — return per unit of drawdown risk
- 📋 **Full trade log** — every buy/sell date, entry price, exit price, P&L

---

### 📰 AI News Intelligence

Every news article about a stock is scored in real time:

| Metric | What it measures |
|---|---|
| **Sentiment Score** | -1.0 (very bearish) to +1.0 (very bullish) |
| **Impact Score** | 0–100 based on keyword relevance and recency |
| **Breaking News Flag** | High-impact articles less than 6 hours old |

Aggregate stats give you: overall market mood, positive vs negative article count, and a composite market impact score. This sentiment score is then **fused directly into the ML prediction** — a strongly negative news day will nudge the model's signal accordingly.

---

### 💼 Portfolio Risk Analytics

For every stock, StockIQ Pro computes institutional-grade risk metrics from 1 year of daily return history:

| Metric | What it tells you |
|---|---|
| **VaR 95% / 99%** | "On 95% of days, I won't lose more than X%" |
| **Expected Shortfall** | Average loss when things *do* go bad |
| **Max Drawdown** | Worst peak-to-trough loss over the trailing year |
| **Sharpe Ratio** | How much return per unit of risk (6.5% India risk-free) |
| **Beta vs Nifty 50** | How much does this stock amplify the market's moves? |
| **Information Ratio** | Skill of the stock vs the benchmark |
| **Skewness / Kurtosis** | Are returns normally distributed, or are there fat tails? |

---

### ⚙️ Black-Scholes Options Pricing

For any NSE stock, StockIQ Pro computes at-the-money (ATM) European options for a 30-day horizon:

- **Call & Put prices** in ₹
- **Delta** — directional exposure (how much the option moves per ₹1 stock move)
- **Gamma** — rate of delta change
- **Vega** — sensitivity to volatility
- **Theta** — time decay per day
- **Implied Volatility** — the market's forward-looking volatility estimate

---

### 🏆 Peer Comparison & Sector Ranking

**Head-to-head:** Compare any two stocks on 8 metrics — 1M/3M/6M/1Y returns, Sharpe ratio, volatility, RSI, ML signal. Winner is highlighted per metric.

**Sector leaderboard:** Rank all stocks in a sector using a **composite score**:
- Sharpe Ratio weight: 30%
- 3M Return rank: 25%
- Low Volatility rank: 20%
- RSI health (40–65 ideal): 15%
- 1Y Return rank: 10%

---

### 🩺 Portfolio Recovery Advisor _(New)_

For investors with losing positions, StockIQ Pro provides an intelligent recovery analysis for each holding:

- **RSI Signal** — Is the stock oversold (potential reversal) or overbought?
- **News Sentiment** — Is the macro/news environment supportive of recovery?
- **Averaging-Down Calculator** — Exact cost of doubling your position, new average price, and how much the stock needs to recover to break even
- **Recommendation engine**: `AVERAGE_DOWN` / `HOLD & MONITOR` / `CUT LOSS` / `BOOK PROFIT`

**Smart Capital Allocator** distributes spare capital across your losing positions by composite score (RSI + sentiment + momentum + volatility fit).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐   │
│   │              Next.js 16 Frontend (React 19)             │   │
│   │                                                         │   │
│   │  LivePrice → StockChart → MLPrediction (SHAP chart)    │   │
│   │  Backtesting → PortfolioMetrics → AdvancedNews         │   │
│   │  LongTermAnalysis (DCF · DuPont · Health Score)        │   │
│   │  PeerComparison → SectorIntelligence                   │   │
│   │  PortfolioTracker → RecoveryAdvisor → CapitalAdvisor   │   │
│   └──────────────────────────┬─────────────────────────────┘   │
└─────────────────────────────-│──────────────────────────────────┘
                               │ REST API (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Python)                     │
│                                                                 │
│   /api/live              → yf_client.get_quote()               │
│   /api/analyze           → engine.analyze_ticker()             │
│   /api/ml-predict        → 6-model stacked ensemble + SHAP     │
│   /api/backtest          → RSI+MACD strategy simulation        │
│   /api/portfolio-metrics → VaR, Sharpe, Beta, Black-Scholes    │
│   /api/advanced-news     → sentiment scoring + impact analysis  │
│   /api/peer-compare      → head-to-head metrics                │
│   /api/sector-rank       → composite peer ranking              │
│   /api/valuation         → DCF, DuPont, Graham, Health Score   │
│   /api/portfolio-insight → Recovery Advisor per holding        │
│   /api/capital-allocate  → Smart capital distribution engine   │
│   /api/tickers           → 1,900+ NSE stocks (in-memory cache) │
│                                                                 │
│   Data Layer (yf_client.py):                                    │
│   ├── Yahoo Finance v8 /chart API (OHLCV, live quotes)         │
│   ├── Yahoo Finance v10 quoteSummary (crumb + cookie auth)     │
│   ├── NSE India CSV (EQUITY_L.csv — ticker universe)           │
│   └── RSS feeds + News APIs (multi-source aggregation)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Node.js 18+ and Python 3.9+

### 1. Clone
```bash
git clone https://github.com/visheshsanghvi112/Analysis-tool.git
cd Analysis-tool
```

### 2. Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
python main.py
# → API at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

### 3. Frontend
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
# → App at http://localhost:3000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/tickers?q=hdfc` | Search 1,900+ NSE stocks |
| `GET` | `/api/live?ticker=HDFCBANK.NS` | Live price quote |
| `GET` | `/api/analyze?ticker=HDFCBANK.NS` | Full technical analysis |
| `GET` | `/api/ml-predict?ticker=HDFCBANK.NS` | 5-day prediction + SHAP |
| `GET` | `/api/backtest?ticker=HDFCBANK.NS&period=2y` | RSI+MACD backtest |
| `GET` | `/api/portfolio-metrics?ticker=HDFCBANK.NS` | VaR, Sharpe, Greeks |
| `GET` | `/api/advanced-news?ticker=HDFCBANK.NS` | News + AI sentiment |
| `GET` | `/api/compare?tickers=TCS.NS,INFY.NS` | Peer comparison |
| `GET` | `/api/sector-rank?ticker=HDFCBANK.NS` | Sector leaderboard |
| `GET` | `/api/valuation?ticker=HDFCBANK.NS` | **DCF · DuPont · Graham · Health Score** |
| `POST` | `/api/portfolio-insight` | Recovery Advisor for portfolio holdings |
| `POST` | `/api/capital-allocate` | Smart capital allocation plan |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, Tailwind CSS 3, Recharts, Lucide Icons |
| **Backend** | FastAPI, Python 3.9+, uvicorn (ASGI) |
| **ML / Quantitative** | scikit-learn, XGBoost, LightGBM, SHAP, hmmlearn, arch (GARCH) |
| **NLP / Sentiment** | TextBlob, feedparser, multi-source RSS aggregation |
| **Data Sources** | Yahoo Finance REST API (v8 chart + v10 quoteSummary with crumb auth), NSE India EQUITY_L.csv |
| **Deployment** | Vercel (serverless, edge CDN, global) |
| **Security** | slowapi rate limiting, input regex validation, CORS whitelist |

---

## Security

| Protection | Implementation |
|---|---|
| Rate Limiting | `slowapi` — 30 req/min per IP on data endpoints |
| Input Validation | Regex ticker validation `[A-Z0-9&.-]{1,15}` |
| CORS | Origin whitelist via environment variable |
| AbortController | Client-side stale request cancellation |
| Error Handling | No stack traces or sensitive data in API responses |
| Yahoo Finance Auth | Dynamic crumb + cookie seeding for v10 quoteSummary endpoint |

---

## Roadmap

- [x] ~~**DCF Valuation model** — intrinsic value vs market price~~ ✅ _Shipped_
- [x] ~~**DuPont Analysis** — ROE decomposition into 3 drivers~~ ✅ _Shipped_
- [x] ~~**Financial Health Score** — 10-criteria long-term checklist~~ ✅ _Shipped_
- [x] ~~**Portfolio Recovery Advisor** — RSI + sentiment + avg-down calculator~~ ✅ _Shipped_
- [x] ~~**Smart Capital Allocator** — distribute spare capital by recovery score~~ ✅ _Shipped_
- [ ] **WebSocket real-time streaming** — true live prices without polling
- [ ] **Sector heatmap** — Finviz-style treemap for all NSE stocks
- [ ] **Altman Z-Score** — financial health / bankruptcy risk gauge
- [ ] **Earnings surprise predictor** — beat/miss classifier
- [ ] **User accounts + watchlist** — auth via Clerk or Supabase
- [ ] **Mobile app** — React Native

---

## Deployment

Both services deploy independently to Vercel.

```bash
# Backend
cd backend && vercel --prod

# Frontend (set NEXT_PUBLIC_API_URL in Vercel dashboard first)
cd frontend && vercel --prod
```

**Backend `.env` variables:**
```env
ENVIRONMENT=production
ALLOWED_ORIGINS=https://your-frontend.vercel.app
RATE_LIMIT_PER_MINUTE=30
```

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/sector-heatmap`
3. Commit: `git commit -m 'feat: add sector heatmap component'`
4. Push and open a Pull Request

**Code standards:**
- JavaScript: ESLint (`eslint-config-next`)
- Python: `black` formatter, `ruff` linter
- Commits: [Conventional Commits](https://www.conventionalcommits.org/)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by **[Vishesh Sanghvi](https://github.com/visheshsanghvi112)**

[![GitHub](https://img.shields.io/badge/GitHub-visheshsanghvi112-181717?style=flat-square&logo=github)](https://github.com/visheshsanghvi112)
[![Email](https://img.shields.io/badge/Email-visheshsanghvi112@gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:visheshsanghvi112@gmail.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-visheshsanghvi.qzz.io-6366f1?style=flat-square&logo=vercel&logoColor=white)](https://visheshsanghvi.qzz.io)

⭐ **If StockIQ Pro helps your investment decisions, please star the repo** — it helps others discover it.

_"The goal of this project is simple: every individual investor deserves the same analytical tools as a hedge fund. Free, transparent, and built for Indian markets."_

</div>