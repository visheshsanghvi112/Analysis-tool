# ============================================================
# Stock Analysis Tool — by Vishesh Sanghvi
# ============================================================

import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import feedparser
import warnings
warnings.filterwarnings('ignore')


# ── 1. Support & Resistance ──────────────────────────────────
def calculate_support_resistance(data):
    resistance = data['Close'].max()
    support = data['Close'].min()
    return support, resistance


# ── 2. Fibonacci Retracement ─────────────────────────────────
def calculate_fibonacci_levels(data):
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    diff = max_price - min_price
    return (
        max_price - 0.236 * diff,
        max_price - 0.382 * diff,
        max_price - 0.618 * diff,
    )


# ── 3. Correlation Heatmap ───────────────────────────────────
def correlation_heatmap(data):
    correlation = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=.5)
    plt.title("Correlation Heatmap of Indicators")
    plt.tight_layout()
    plt.show()


# ── 4. REAL News Sentiment (Google News RSS — no API key) ────
def fetch_news_sentiment(ticker):
    """
    Fetches live news headlines from Google News RSS for the ticker
    and scores them with TextBlob NLP. No API key required.
    """
    try:
        # Strip .NS / .BO suffix for cleaner news search
        search_term = ticker.replace('.NS', '').replace('.BO', '')
        url = (f"https://news.google.com/rss/search?"
               f"q={search_term}+stock+India&hl=en-IN&gl=IN&ceid=IN:en")
        feed = feedparser.parse(url)
        entries = feed.entries[:8]   # top 8 headlines

        if not entries:
            print(f"  [!] No news found for {ticker}, defaulting to Neutral.")
            return 0.0, 0.0

        sentiment_score = 0.0
        subjectivity_score = 0.0
        for entry in entries:
            text = entry.get('title', '') + ' ' + entry.get('summary', '')
            analysis = TextBlob(text)
            sentiment_score += analysis.sentiment.polarity
            subjectivity_score += analysis.sentiment.subjectivity

        avg_sentiment = sentiment_score / len(entries)
        avg_subjectivity = subjectivity_score / len(entries)
        return avg_sentiment, avg_subjectivity

    except Exception as e:
        print(f"  [!] Sentiment fetch failed: {e}. Defaulting to Neutral.")
        return 0.0, 0.0


# ── 5. Buy / Sell / Hold Signal ──────────────────────────────
def generate_signal(rsi, macd, signal_line, close, vwap, adx):
    """
    Combines RSI + MACD + VWAP + ADX to produce a trading signal.
    - BUY  : RSI oversold + MACD crossing up + price near/below VWAP
    - SELL : RSI overbought + MACD crossing down + price above VWAP
    - HOLD : Everything else
    """
    score = 0

    # RSI signal
    if rsi < 35:
        score += 2      # oversold → bullish
    elif rsi > 65:
        score -= 2      # overbought → bearish

    # MACD signal
    if macd > signal_line:
        score += 1
    else:
        score -= 1

    # Price vs VWAP
    if close < vwap:
        score += 1      # price below fair value → potential buy
    else:
        score -= 1

    # ADX (trend strength) — only act if trend is strong
    if adx < 20:
        score = 0       # weak trend, no clear signal → HOLD

    if score >= 2:
        return "🟢 BUY"
    elif score <= -2:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"


# ── 6. Fundamentals ─────────────────────────────────────────
def fetch_fundamentals(ticker):
    """Fetches PE ratio, Market Cap, 52w High/Low from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        pe     = info.get('trailingPE', 'N/A')
        mktcap = info.get('marketCap', None)
        high52 = info.get('fiftyTwoWeekHigh', 'N/A')
        low52  = info.get('fiftyTwoWeekLow', 'N/A')
        div    = info.get('dividendYield', None)

        mktcap_str = (f"₹{mktcap/1e9:.2f}B" if mktcap else "N/A")
        div_str    = (f"{div*100:.2f}%" if div else "N/A")

        return {
            'PE Ratio'   : round(pe, 2) if isinstance(pe, float) else pe,
            'Market Cap' : mktcap_str,
            '52W High'   : high52,
            '52W Low'    : low52,
            'Div Yield'  : div_str,
        }
    except Exception:
        return {}


# ── 7. VWAP — daily reset (correct implementation) ──────────
def calculate_daily_vwap(data):
    """
    VWAP resets each trading day. Groups by date and computes
    cumulative (price × volume) / cumulative volume per day.
    """
    data = data.copy()
    data['Date'] = data.index.date
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['TPV'] = data['TP'] * data['Volume']

    data['CumTPV'] = data.groupby('Date')['TPV'].cumsum()
    data['CumVol']  = data.groupby('Date')['Volume'].cumsum()
    data['VWAP']    = data['CumTPV'] / data['CumVol']
    data.drop(columns=['Date', 'TP', 'TPV', 'CumTPV', 'CumVol'], inplace=True)
    return data


# ── 8. Main Analysis ─────────────────────────────────────────
def detailed_stock_analysis(tickers, start_date=None, end_date=None):
    if not start_date or not end_date:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=90)

    for ticker in tickers:
        print(f"\n{'='*55}")
        print(f"  Analysing: {ticker}")
        print(f"{'='*55}")

        try:
            # ── Fetch data ───────────────────────────────────
            stock_data = yf.download(
                ticker, start=start_date, end=end_date, interval="1d", progress=False
            )
            if stock_data.empty:
                print(f"  [!] No data found for {ticker}. Skipping.")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            # ── Moving Averages ──────────────────────────────
            stock_data['20 Day MA']  = stock_data['Close'].rolling(20).mean()
            stock_data['50 Day MA']  = stock_data['Close'].rolling(50).mean()
            stock_data['100 Day MA'] = stock_data['Close'].rolling(100).mean()

            # ── Bollinger Bands ──────────────────────────────
            stock_data['20 Day STD'] = stock_data['Close'].rolling(20).std()
            stock_data['Upper Band'] = stock_data['20 Day MA'] + 2 * stock_data['20 Day STD']
            stock_data['Lower Band'] = stock_data['20 Day MA'] - 2 * stock_data['20 Day STD']

            # ── RSI ──────────────────────────────────────────
            delta    = stock_data['Close'].diff(1)
            gain     = delta.where(delta > 0, 0)
            loss     = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs       = avg_gain / avg_loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))

            # ── MACD ─────────────────────────────────────────
            stock_data['12 EMA']     = stock_data['Close'].ewm(span=12, adjust=False).mean()
            stock_data['26 EMA']     = stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD']       = stock_data['12 EMA'] - stock_data['26 EMA']
            stock_data['Signal Line']= stock_data['MACD'].ewm(span=9, adjust=False).mean()

            # ── ATR ──────────────────────────────────────────
            hl  = stock_data['High'] - stock_data['Low']
            hc  = np.abs(stock_data['High'] - stock_data['Close'].shift())
            lc  = np.abs(stock_data['Low']  - stock_data['Close'].shift())
            tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            stock_data['ATR'] = tr.rolling(14).mean()

            # ── VWAP (daily reset — correct) ─────────────────
            stock_data = calculate_daily_vwap(stock_data)

            # ── OBV ──────────────────────────────────────────
            stock_data['OBV'] = (
                np.sign(stock_data['Close'].diff()) * stock_data['Volume']
            ).fillna(0).cumsum()

            # ── ADX ──────────────────────────────────────────
            def calculate_adx(df, n=14):
                df['+DM'] = np.where(
                    (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                    df['High'] - df['High'].shift(1), 0)
                df['-DM'] = np.where(
                    (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                    df['Low'].shift(1) - df['Low'], 0)
                df['+DI'] = 100 * (df['+DM'].rolling(n).sum() / df['ATR'])
                df['-DI'] = 100 * (df['-DM'].rolling(n).sum() / df['ATR'])
                df['DX']  = (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
                df['ADX'] = df['DX'].rolling(n).mean()
                return df

            stock_data = calculate_adx(stock_data)

            # ── Latest values for signal ─────────────────────
            latest       = stock_data.dropna().iloc[-1]
            latest_rsi   = float(latest['RSI'])
            latest_macd  = float(latest['MACD'])
            latest_sig   = float(latest['Signal Line'])
            latest_close = float(latest['Close'])
            latest_vwap  = float(latest['VWAP'])
            latest_adx   = float(latest['ADX'])

            # ── Signal ───────────────────────────────────────
            signal = generate_signal(
                latest_rsi, latest_macd, latest_sig,
                latest_close, latest_vwap, latest_adx
            )

            # ── Support / Resistance / Fibonacci ─────────────
            support, resistance = calculate_support_resistance(stock_data)
            fib1, fib2, fib3   = calculate_fibonacci_levels(stock_data)

            # ── Fundamentals ─────────────────────────────────
            fundamentals = fetch_fundamentals(ticker)

            # ── Sentiment ────────────────────────────────────
            avg_sentiment, avg_subjectivity = fetch_news_sentiment(ticker)
            sentiment_label = (
                "Positive 📈" if avg_sentiment > 0.05
                else "Negative 📉" if avg_sentiment < -0.05
                else "Neutral ➡️"
            )

            # ── Print Summary ─────────────────────────────────
            print(f"\n  📌 Latest Close  : ₹{latest_close:.2f}")
            print(f"  📊 Signal        : {signal}")
            print(f"\n  — Technical —")
            print(f"  RSI              : {latest_rsi:.2f}  (>70 overbought, <30 oversold)")
            print(f"  MACD             : {latest_macd:.4f}  |  Signal: {latest_sig:.4f}")
            print(f"  ADX              : {latest_adx:.2f}  (>25 = strong trend)")
            print(f"  VWAP             : ₹{latest_vwap:.2f}")
            print(f"  Support          : ₹{float(support):.2f}")
            print(f"  Resistance       : ₹{float(resistance):.2f}")
            print(f"  Fibonacci 23.6%  : ₹{float(fib1):.2f}")
            print(f"  Fibonacci 38.2%  : ₹{float(fib2):.2f}")
            print(f"  Fibonacci 61.8%  : ₹{float(fib3):.2f}")

            if fundamentals:
                print(f"\n  — Fundamentals —")
                for k, v in fundamentals.items():
                    print(f"  {k:<16} : {v}")

            print(f"\n  — Sentiment (Live News) —")
            print(f"  Score            : {avg_sentiment:.3f}  →  {sentiment_label}")
            print(f"  Subjectivity     : {avg_subjectivity:.3f}")

            # ── Plotly Chart ─────────────────────────────────
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(
                    f"{ticker} — Price & Indicators",
                    "RSI  (Overbought >70 | Oversold <30)",
                    "MACD"
                )
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'], high=stock_data['High'],
                low=stock_data['Low'],  close=stock_data['Close'],
                name='Price'), row=1, col=1)

            # MAs
            for col, color in [('20 Day MA','blue'), ('50 Day MA','orange'), ('100 Day MA','green')]:
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data[col],
                    mode='lines', name=col,
                    line=dict(color=color, width=1)), row=1, col=1)

            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=stock_data.index, y=stock_data['Upper Band'],
                mode='lines', name='Upper Band',
                line=dict(color='red', dash='dash', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=stock_data.index, y=stock_data['Lower Band'],
                mode='lines', name='Lower Band',
                line=dict(color='red', dash='dash', width=1),
                fill='tonexty', fillcolor='rgba(255,0,0,0.05)'), row=1, col=1)

            # VWAP
            fig.add_trace(go.Scatter(
                x=stock_data.index, y=stock_data['VWAP'],
                mode='lines', name='VWAP',
                line=dict(color='magenta', width=1.5, dash='dot')), row=1, col=1)

            # RSI
            fig.add_trace(go.Scatter(
                x=stock_data.index, y=stock_data['RSI'],
                mode='lines', name='RSI',
                line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash='dash', line_color='red',   row=2, col=1)
            fig.add_hline(y=30, line_dash='dash', line_color='green',  row=2, col=1)

            # MACD
            fig.add_trace(go.Scatter(
                x=stock_data.index, y=stock_data['MACD'],
                mode='lines', name='MACD',
                line=dict(color='teal')), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=stock_data.index, y=stock_data['Signal Line'],
                mode='lines', name='Signal Line',
                line=dict(color='orange')), row=3, col=1)

            fig.update_layout(
                title=dict(
                    text=f"{ticker} — Stock Analysis  |  Signal: {signal}  |  Sentiment: {sentiment_label}",
                    font=dict(size=16)
                ),
                template='plotly_dark',
                xaxis_rangeslider_visible=False,
                height=800,
            )
            fig.show()

            # ── Correlation Heatmap ───────────────────────────
            heatmap_cols = [
                'Close', '20 Day MA', '50 Day MA',
                'Upper Band', 'Lower Band', 'RSI',
                'MACD', 'ATR', 'VWAP', 'OBV', 'ADX'
            ]
            correlation_heatmap(stock_data[heatmap_cols].dropna())

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")


# ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔷 Stock Analysis Tool — by Vishesh Sanghvi\n")

    tickers_input = input("Enter tickers (e.g. HDFCBANK.NS, TATAMOTORS.NS): ")
    tickers = [t.strip() for t in tickers_input.split(',')]

    start_input = input("Start date (YYYY-MM-DD) or Enter for last 3 months: ").strip()
    end_input   = input("End date   (YYYY-MM-DD) or Enter for today         : ").strip()

    start_date = start_input or None
    end_date   = end_input   or None

    detailed_stock_analysis(tickers, start_date, end_date)
