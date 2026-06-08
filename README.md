# 📈 Stock Analysis Tool

**Built by [Vishesh Sanghvi](https://github.com/visheshsanghvi112)**

A comprehensive Python-based stock analysis tool for Indian markets (NSE/BSE) that fetches live data, computes 10+ technical indicators, analyses real-time news sentiment, generates Buy/Sell/Hold signals, and visualises everything with interactive Plotly charts.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Live Data** | Fetches OHLCV data from Yahoo Finance (NSE `.NS` / BSE `.BO`) |
| **Technical Indicators** | RSI, MACD, Bollinger Bands, ATR, VWAP (daily reset), OBV, ADX, Moving Averages |
| **Buy / Sell / Hold Signal** | Combines RSI + MACD + VWAP + ADX to generate an actionable signal |
| **Real News Sentiment** | Fetches live headlines from Google News RSS, scored with TextBlob NLP |
| **Fundamentals** | PE Ratio, Market Cap, 52-Week High/Low, Dividend Yield via yfinance |
| **Support & Resistance** | Auto-calculated from historical price data |
| **Fibonacci Retracement** | 23.6%, 38.2%, 61.8% levels |
| **Interactive Charts** | Candlestick + MAs + Bollinger Bands + RSI + MACD using Plotly (dark theme) |
| **Correlation Heatmap** | Seaborn heatmap of all indicator correlations |
| **Multi-ticker** | Analyse multiple stocks in one run |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install yfinance numpy pandas matplotlib seaborn plotly textblob feedparser
```

### 2. Run

```bash
python test.py
```

### 3. Example Input

```
Enter tickers: HDFCBANK.NS, TATAMOTORS.NS, RELIANCE.NS
Start date: 2024-01-01
End date:   2024-06-30
```

---

## 📊 Indicators Explained

| Indicator | What it tells you |
|---|---|
| **RSI** | Momentum — <30 oversold (buy zone), >70 overbought (sell zone) |
| **MACD** | Trend direction — MACD crossing above signal line = bullish |
| **Bollinger Bands** | Volatility — price near upper band = overbought, lower = oversold |
| **VWAP** | Fair intraday price weighted by volume (resets daily) |
| **ATR** | Average daily price range — higher = more volatile |
| **OBV** | Volume confirming price — rising OBV with rising price = strong trend |
| **ADX** | Trend strength — >25 means a strong trend exists |

---

## 📡 Data Sources

- **Price Data** — [Yahoo Finance](https://finance.yahoo.com/) via `yfinance` (free, no API key)
- **News Sentiment** — [Google News RSS](https://news.google.com/rss) (free, no API key)
- **Fundamentals** — Yahoo Finance via `yfinance`

---

## ⚠️ Disclaimer

This tool is built for **educational and research purposes only**. It is not financial advice. Always do your own research before making investment decisions.

---

## 📄 License

MIT License © Vishesh Sanghvi
