# 📈 Stock Analysis & Investment Dashboard

**Built by [Vishesh Sanghvi](https://github.com/visheshsanghvi112)**

A premium, full-stack stock analysis dashboard designed for technical, fundamental, risk, and peer analysis of Indian markets (NSE/BSE). The application features a modular **FastAPI** Python backend engine and an interactive **Next.js** (React) web dashboard frontend utilizing styled Tailwind CSS and native **Recharts** visualizations.

---

## 🛠️ Tech Stack & Architecture

```
                                  +----------------------------+
                                  |    Next.js Web Frontend    |
                                  | (React, Tailwind, Recharts) |
                                  +-------------+--------------+
                                                |
                                                | HTTP JSON API
                                                v
                                  +----------------------------+
                                  |    FastAPI Python Server   |
                                  |    (Uvicorn, FastAPI)      |
                                  +-------------+--------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
          +----------------------------+                  +----------------------------+
          |     Yahoo Finance API      |                  |    Google News RSS Feed    |
          |  (OHLCV & Fundamentals)    |                  |  (Live Scraping & NLP)     |
          +----------------------------+                  +----------------------------+
```

* **Frontend**: Next.js (App Router, Client Components, Dynamic Imports), Recharts (interactive SVG charting overlays), Tailwind CSS (sleek dark mode design).
* **Backend**: FastAPI, Uvicorn, Pandas, NumPy, yfinance, TextBlob (NLP sentiment analyzer), Feedparser.

---

## ✨ Features & Metrics Engine

| Feature | Description |
|---|---|
| **Live Price & Technicals** | Real-time OHLCV candles, 20/50/100 MAs, Bollinger Bands, and VWAP (daily reset). |
| **Actionable Signals** | Multi-indicator signal generator (BUY/SELL/HOLD) scoring RSI, MACD crossovers, VWAP, and ADX. |
| **Advanced Fundamentals** | ROE (Return on Equity), PEG Ratio, Market Cap, Debt/Equity ratio, Free Cash Flow (FCF), YoY Revenue Growth, Beta, and 52W range relative positioning. |
| **Risk Management** | Annualized Volatility, Max Drawdown, Sharpe Ratio (using risk-free rate benchmarks), and 95% 1-day Value at Risk (VaR). |
| **Index Relative Strength** | Direct comparative performance benchmarking vs **Nifty 50** Index over selected dates. |
| **Peer Comparison Matrix** | Side-by-side comparative grid showing PE, PEG, ROE, Debt/Equity, Revenue Growth, and Beta for peers. |
| **Google News Sentiment** | Real-time headline scraping via Google News RSS parser with polarities calculated via TextBlob. |

---

## 🚀 Getting Started & Local Setup

### 1. Run the FastAPI Backend Server
```bash
cd backend
pip install -r requirements.txt
python main.py
```
* **API base URL**: `http://127.0.0.1:8000`
* **Auto-generated API Docs**: `http://127.0.0.1:8000/docs`

### 2. Run the Next.js Frontend
```bash
cd frontend
npm install
npm run dev
```
* **Dashboard URL**: `http://localhost:3000`

---

## 📄 License & Disclaimer
This tool is built for **educational and research purposes only** (SEBI Job Interview portfolio showcase). It is not financial advice. MIT License © Vishesh Sanghvi.
