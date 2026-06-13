<div align="center">

<br/>

# StockIQ Pro

### Professional-grade stock intelligence for the Indian market — powered by Machine Learning and real-time data.

<br/>

[![Next.js](https://img.shields.io/badge/Next.js_16-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python_3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-black?style=flat-square&logo=vercel)](https://vercel.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

<br/>

**[🌐 Live Demo](https://your-app.vercel.app)** · **[📖 API Docs](https://your-backend.vercel.app/docs)** · **[🐛 Report Bug](https://github.com/visheshsanghvi112/Analysis-tool/issues)** · **[✨ Request Feature](https://github.com/visheshsanghvi112/Analysis-tool/issues)**

</div>

---

## What is StockIQ Pro?

StockIQ Pro is a **full-stack stock analysis platform** built for the Indian equity market (NSE & BSE). It brings institutional-quality financial analytics — typically locked behind Bloomberg Terminal ($24,000/year) or expensive quant platforms — and makes them **completely free and open-source**.

A user can search any of the **1,900+ NSE-listed stocks**, and instantly get:

- A **live price quote** with day range visualization (auto-refreshes every 30 seconds)
- **Full technical analysis** — RSI, MACD, Bollinger Bands, ADX, Moving Averages — plotted on interactive charts
- **5-day AI price predictions** from a trained Random Forest model with confidence scores
- **Professional portfolio risk metrics** — VaR (95% & 99%), Sharpe Ratio, Maximum Drawdown, Beta vs Nifty
- **Black-Scholes options pricing** with Greeks (Delta, Gamma, Vega, Theta) for any stock
- **AI-powered news intelligence** — multi-source aggregation with real-time sentiment scoring and impact analysis

---

## Features In Depth

### 📊 Live Market Data
Real-time NSE/BSE price quotes fetched via Yahoo Finance's undocumented-but-stable chart API (`v8/finance/chart`). Includes current price, day high/low, previous close, volume, and market cap. Data is ~15 minutes delayed for NSE stocks (Yahoo Finance standard). A visual range bar shows where the current price sits within the day's trading range.

### 🧠 Machine Learning Predictions (6-Model Stacked Ensemble)
To generate highly reliable price projections, StockIQ Pro utilizes a multi-stage **Stacked Ensemble Regressor** trained on 2 years of daily historical data. Naive machine learning models often fail in finance due to high noise-to-signal ratios. Our ensemble architecture overcomes this by combining diverse model families:
1.  **Base Learners (Diverse Feature Spaces):**
    *   **Random Forest Regressor:** Handles bagging and reduces variance.
    *   **Gradient Boosting Regressor:** Sequentially fits trees to minimize residuals.
    *   **XGBoost:** Implements regularized (L1/L2) tree boosting for maximum tabular speed and accuracy.
    *   **LightGBM:** Uses leaf-wise growth (GOSS) to efficiently capture deep feature interactions.
    *   **Extra Trees Regressor:** Extremely randomized trees that add high variance reduction to shield the ensemble from price anomalies.
2.  **Meta-Learner (Ridge Stacking):**
    *   Base learners' out-of-sample predictions are combined by a **Ridge Regressor (L2 Regularization)**. The Ridge meta-learner restricts the regression weights to prevent multicollinearity and overfitting, providing a stable final 5-day return prediction.
3.  **Features Engineered (40+ Indicators):**
    *   **Momentum:** RSI (14), Williams %R, Stochastic %K/%D.
    *   **Trend:** MACD, MACD Signal, MACD Histogram, EMA Cross (9/21), MA Ratios (5, 10, 20, 50, 100).
    *   **Volatility:** ATR Ratio, Bollinger Band Position & Width, 20d & 60d standard deviation.
    *   **Volume:** Volume Ratio, OBV Ratio.
    *   **Moments & Calendars:** Rolling skewness/kurtosis (20d), 52-week proximity, calendar effects (day of week, month), and multi-day lagged returns.

### 📐 Out-of-Sample Walk-Forward Backtesting
Standard cross-validation (like K-Fold) leaks future data into the past, producing artificially high backtest metrics. We implement an **Expanding Walk-Forward Backtest** (5 folds) to simulate real-world trading performance:
*   **Directional Hit Rate:** Measures the percentage of times the model correctly predicted the sign (direction) of the 5-day return.
*   **Profit Factor:** The ratio of gross profits to gross losses:
$$\text{Profit Factor} = \frac{\sum \text{Wins}}{\sum |\text{Losses}|}$$
*   **Ensemble Advantage:** The absolute accuracy improvement of the stacked ensemble over a standalone baseline XGBoost model, demonstrating the statistical edge of stacking.

### 🏛️ Advanced Volatility & Regime Modeling (HMM + GARCH)
Markets are non-stationary and fluctuate between structural periods. StockIQ Pro models these periods using advanced econometrics:
1.  **Gaussian Hidden Markov Model (HMM):**
    *   Instead of arbitrary moving average crossovers, a 3-state HMM is fitted directly to the stock's log returns. The model identifies hidden latent states: **LOW_VOLATILITY** (stable bull), **MEDIUM_VOLATILITY** (neutral/orderly), and **HIGH_VOLATILITY** (bearish/panic).
    *   The model solves the transition probability matrix to classify the current day's active regime.
2.  **GARCH(1,1) Volatility Forecasting:**
    *   Models the conditional variance ($\sigma_t^2$) of returns using the standard GARCH(1,1) specification to capture volatility clustering:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
    *   The forecasted variance over the 5-day horizon is annualized to yield the expected forward-looking volatility.
3.  **Regime-Aware Decision Engine:**
    *   During a **HIGH_VOLATILITY** regime, the model automatically widens the prediction signal thresholds (widening to $3.0\%$ expected return) to prevent false trading signals caused by noise. In stable regimes, the threshold narrows to $1.8\%$.



### 📈 Technical Analysis Charts
Interactive charts rendered via Recharts, built on a `ComposedChart` with:
- **Price chart**: Candlestick-style close price with MA20, MA50, MA100 overlays and Bollinger Bands
- **Volume histogram**: Color-coded green/red vs rolling average, with a Volume MA20 line
- **RSI (14)**: With overbought (70) and oversold (30) reference lines
- **MACD + Histogram**: Signal line crossover visualization
- **ADX**: Trend strength gauge with 15 and 25 threshold references

### 💼 Portfolio & Risk Analytics
Computes institutional-grade metrics from 1 year of daily return history:

| Metric | Description |
|---|---|
| **VaR 95% / 99%** | Maximum expected daily loss at 95% / 99% confidence |
| **Expected Shortfall** | Average loss when VaR threshold is breached (tail risk) |
| **Max Drawdown** | Peak-to-trough loss over the trailing year |
| **Sharpe Ratio** | Risk-adjusted return (6.5% India risk-free rate) |
| **Beta vs Nifty 50** | Systematic risk relative to the broad market |
| **Correlation** | Price co-movement with the Nifty 50 index |
| **Information Ratio** | Active management skill vs the benchmark |
| **Skewness / Kurtosis** | Return distribution shape (fat-tail analysis) |
| **Annual Volatility** | Annualized standard deviation of daily returns |

### ⚙️ Black-Scholes Options Pricing
For any given stock, the platform computes at-the-money (ATM) European options pricing for a 30-day horizon, including:
- **Call price** and **Put price** in ₹
- **Delta** (directional exposure)
- **Gamma** (rate of delta change)
- **Vega** (sensitivity to volatility)
- **Theta** (time decay per day)
- **Implied Volatility** (derived from the trailing year's annualized vol)

### 📰 AI News Intelligence
Aggregates news from multiple RSS and API sources. Each article is scored by:
- **Sentiment** (-1.0 to +1.0) using TextBlob NLP
- **Impact score** (0–100) based on keyword relevance and recency
- **Breaking news detection** — flags high-impact, recent articles

Summary stats include: overall sentiment, positive/negative count, and a market impact score.

### 🔍 Smart Stock Search
Searches across the full NSE equity list (~1,900 stocks) loaded on startup. Results prioritize **symbol matches** over name matches, with sector badges (IT, Banking, FMCG, etc.) and keyboard navigation support.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐   │
│   │              Next.js 16 Frontend (React 19)             │   │
│   │                                                         │   │
│   │  Header (Search)  →  StockChart  →  LivePrice           │   │
│   │  MLPrediction     →  AdvancedNews →  PortfolioMetrics   │   │
│   └──────────────────────────┬─────────────────────────────┘   │
└─────────────────────────────-│──────────────────────────────────┘
                               │ REST API (JSON)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Python)                     │
│                                                                 │
│   /api/live          → yf_client.get_quote()                   │
│   /api/analyze       → engine.analyze_ticker()                 │
│   /api/ml-predict    → ml_models.get_ml_prediction()           │
│   /api/portfolio-metrics → VaR, Sharpe, Beta, Black-Scholes    │
│   /api/advanced-news → news_intelligence.get_advanced_news()   │
│   /api/tickers       → NSE EQUITY_L.csv (in-memory cache)      │
│                                                                 │
│   External Calls:                                               │
│   ├── Yahoo Finance v8/v10 API (OHLCV, quote, fundamentals)    │
│   ├── NSE India CSV (ticker list, cached at startup)           │
│   └── RSS feeds + News APIs (news aggregation)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Analysis-tool/
├── frontend/                          # Next.js 16 application
│   ├── src/app/
│   │   ├── components/
│   │   │   ├── Header.js              # Search bar (debounced, AbortController)
│   │   │   ├── LivePrice.js           # Real-time quote with auto-refresh
│   │   │   ├── StockChart.js          # Multi-panel interactive charts
│   │   │   ├── MLPrediction.js        # AI forecast card
│   │   │   ├── AdvancedNews.js        # News sentiment feed
│   │   │   ├── PortfolioMetrics.js    # Risk analytics dashboard
│   │   │   ├── StockSearchModal.js    # Full-screen search modal
│   │   │   └── ErrorBoundary.js       # React error boundary
│   │   ├── globals.css                # Design system (professional dark theme)
│   │   ├── layout.js                  # Root layout + metadata
│   │   └── page.js                    # Main dashboard
│   ├── next.config.mjs
│   ├── tailwind.config.js
│   └── package.json
│
├── backend/                           # FastAPI application
│   ├── main.py                        # Routes + startup ticker preload
│   ├── engine.py                      # Technical analysis engine
│   ├── ml_models.py                   # Random Forest training & inference
│   ├── news_intelligence.py           # News aggregation + NLP sentiment
│   ├── yf_client.py                   # Yahoo Finance direct REST client
│   ├── requirements.txt
│   └── vercel.json
│
└── README.md
```

---

## Quick Start

### Prerequisites
- **Node.js** 18+
- **Python** 3.9+

### 1. Clone
```bash
git clone https://github.com/visheshsanghvi112/Analysis-tool.git
cd Analysis-tool
```

### 2. Backend
```bash
cd backend
pip install -r requirements.txt

# Optional: set allowed frontend origin
cp .env.example .env

# Start the API server
python main.py
```
→ API running at `http://localhost:8000`  
→ Interactive docs at `http://localhost:8000/docs`

### 3. Frontend
```bash
cd frontend
npm install

# Point at your local backend
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

npm run dev
```
→ App running at `http://localhost:3000`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/tickers?q=hdfc` | Search NSE stocks |
| `GET` | `/api/live?ticker=HDFCBANK.NS` | Live price quote |
| `GET` | `/api/analyze?ticker=HDFCBANK.NS` | Full technical analysis + chart data |
| `GET` | `/api/ml-predict?ticker=HDFCBANK.NS` | 5-day ML price prediction |
| `GET` | `/api/portfolio-metrics?ticker=HDFCBANK.NS` | VaR, Sharpe, options, Greeks |
| `GET` | `/api/advanced-news?ticker=HDFCBANK.NS` | News + AI sentiment |
| `GET` | `/api/compare?tickers=TCS.NS,INFY.NS` | Peer comparison |

<details>
<summary><b>Example: ML Prediction Response</b></summary>

```json
{
  "ticker": "HDFCBANK.NS",
  "prediction": {
    "predicted_price": 1785.40,
    "current_price": 1763.20,
    "predicted_return": 1.26,
    "signal": "BUY",
    "signal_strength": 67,
    "confidence": 72.4,
    "prediction_horizon_days": 5,
    "timestamp": "2026-06-13T08:00:00"
  },
  "disclaimer": "Predictions are for educational purposes only. Not financial advice."
}
```

</details>

<details>
<summary><b>Example: Portfolio Metrics Response</b></summary>

```json
{
  "ticker": "HDFCBANK.NS",
  "risk_metrics": {
    "var_95_daily": -1.83,
    "var_99_daily": -2.91,
    "expected_shortfall_95": -2.34,
    "max_drawdown": -14.72,
    "sharpe_ratio": 0.847,
    "annual_volatility": 21.6
  },
  "market_metrics": {
    "beta": 1.12,
    "correlation_with_nifty": 0.74,
    "information_ratio": 0.312
  },
  "options_pricing": {
    "call_price": 42.80,
    "put_price": 35.10,
    "delta": 0.523,
    "gamma": 0.0041,
    "theta": -1.24,
    "vega": 3.87,
    "implied_volatility": 21.6
  }
}
```

</details>

---

## Deployment (Vercel)

Both the frontend and backend deploy independently to Vercel.

**Backend:**
```bash
cd backend
vercel --prod
```

**Frontend** — set your backend URL first:
```bash
# In Vercel dashboard, add environment variable:
# NEXT_PUBLIC_API_URL = https://your-backend.vercel.app

cd frontend
vercel --prod
```

**Backend environment variables (`.env`):**
```env
ENVIRONMENT=production
ALLOWED_ORIGINS=https://your-frontend.vercel.app
RATE_LIMIT_PER_MINUTE=30
```

---

## Security

| Protection | Implementation |
|---|---|
| Rate Limiting | `slowapi` — 30 req/min per IP on all data endpoints |
| Input Validation | Regex-based ticker validation (`[A-Z0-9&.-]{1,15}`) |
| CORS | Origin whitelist via environment variable |
| AbortController | Client-side: stale search requests are cancelled |
| Error Handling | No internal stack traces or sensitive data in error responses |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, Tailwind CSS 3, Recharts, Lucide Icons |
| **Backend** | FastAPI, Python 3.9+, uvicorn |
| **ML / Analytics** | scikit-learn (RandomForest), NumPy, Pandas, SciPy |
| **NLP** | TextBlob (sentiment analysis) |
| **Data Sources** | Yahoo Finance REST API (v8/v10), NSE India CSV, RSS feeds |
| **Deployment** | Vercel (serverless, edge CDN) |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit with a clear message: `git commit -m 'feat: add peer comparison chart'`
4. Push and open a Pull Request

**Code standards:**
- JavaScript: follow ESLint config (`eslint-config-next`)
- Python: format with `black`, lint with `ruff`
- Commit messages: follow [Conventional Commits](https://www.conventionalcommits.org/)

---

## Roadmap

- [ ] WebSocket-based true real-time price streaming
- [ ] Multi-stock portfolio tracker with P&L
- [ ] Backtesting engine for ML signals
- [ ] Sector heatmap and screener
- [ ] User accounts + watchlist (auth via Clerk or Supabase)
- [ ] Mobile app (React Native)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by **[Vishesh Sanghvi](https://github.com/visheshsanghvi112)**

[![GitHub](https://img.shields.io/badge/GitHub-visheshsanghvi112-181717?style=flat-square&logo=github)](https://github.com/visheshsanghvi112)
[![Email](https://img.shields.io/badge/Email-visheshsanghvi112@gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:visheshsanghvi112@gmail.com)

⭐ If this project is useful to you, please star it — it helps others discover it.

</div>