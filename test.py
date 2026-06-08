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


# ── Helper: safely extract a scalar from a pandas Series/value ──
def _scalar(val):
    """Ensures any pandas Series or numpy scalar becomes a plain Python float."""
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val)


# ── Helper: flatten yfinance MultiIndex columns safely ──────────
def _flatten_columns(df):
    """
    yfinance >=0.2.x returns MultiIndex columns like ('Close','HDFCBANK.NS').
    This strips the ticker level and deduplicates if needed.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    # Drop any duplicate columns (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ── 1. Support & Resistance ──────────────────────────────────────
def calculate_support_resistance(data):
    resistance = _scalar(data['Close'].max())
    support    = _scalar(data['Close'].min())
    return support, resistance


# ── 2. Fibonacci Retracement ─────────────────────────────────────
def calculate_fibonacci_levels(data):
    max_price = _scalar(data['Close'].max())
    min_price = _scalar(data['Close'].min())
    diff = max_price - min_price
    return (
        max_price - 0.236 * diff,
        max_price - 0.382 * diff,
        max_price - 0.618 * diff,
    )


# ── 3. Correlation Heatmap ───────────────────────────────────
def correlation_heatmap(data):
    data = data.select_dtypes(include=[np.number])
    if data.empty or data.shape[1] < 2:
        print("  [!] Not enough data for correlation heatmap.")
        return
    correlation = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=.5)
    plt.title("Correlation Heatmap of Indicators")
    plt.tight_layout()
    plt.show()


# ── 3b. Plotly Price Chart ───────────────────────────────────
def show_price_chart(sd, ticker, signal, sent_label, latest_close):
    """Interactive candlestick chart with MAs, Bollinger Bands, VWAP, RSI, MACD."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.05,
        subplot_titles=(
            f"{ticker} — Price, MAs, Bollinger Bands & VWAP",
            "RSI  (Overbought >70 | Oversold <30)",
            "MACD & Signal Line"
        )
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=sd.index,
        open=sd['Open'], high=sd['High'],
        low=sd['Low'],   close=sd['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # Moving Averages
    for ma, color in [('20 Day MA','#2196F3'), ('50 Day MA','#FF9800'), ('100 Day MA','#4CAF50')]:
        if ma in sd.columns:
            fig.add_trace(go.Scatter(
                x=sd.index, y=sd[ma], mode='lines', name=ma,
                line=dict(color=color, width=1)
            ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['Upper Band'], mode='lines',
        name='Upper Band', line=dict(color='#ff6b6b', dash='dash', width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['Lower Band'], mode='lines',
        name='Lower Band', line=dict(color='#ff6b6b', dash='dash', width=1),
        fill='tonexty', fillcolor='rgba(255,107,107,0.06)'
    ), row=1, col=1)

    # VWAP
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['VWAP'], mode='lines',
        name='VWAP', line=dict(color='#E040FB', width=1.5, dash='dot')
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['RSI'], mode='lines',
        name='RSI', line=dict(color='#9C27B0', width=1.5)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[sd.index[0], sd.index[-1]], y=[70, 70],
        mode='lines', line=dict(color='red', dash='dash', width=1),
        showlegend=False, name='Overbought'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[sd.index[0], sd.index[-1]], y=[30, 30],
        mode='lines', line=dict(color='green', dash='dash', width=1),
        showlegend=False, name='Oversold'
    ), row=2, col=1)

    # MACD histogram
    macd_diff   = (sd['MACD'] - sd['Signal Line']).fillna(0)
    macd_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in macd_diff]
    fig.add_trace(go.Bar(
        x=sd.index, y=macd_diff,
        name='MACD Histogram', marker_color=macd_colors, opacity=0.5
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['MACD'], mode='lines',
        name='MACD', line=dict(color='#00BCD4', width=1.5)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=sd.index, y=sd['Signal Line'], mode='lines',
        name='Signal Line', line=dict(color='#FF9800', width=1.5)
    ), row=3, col=1)

    fig.update_layout(
        title=dict(
            text=(f"<b>{ticker}</b>  |  Signal: {signal}  "
                  f"|  Sentiment: {sent_label}  |  Close: ₹{latest_close:.2f}"),
            font=dict(size=15)
        ),
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        height=850,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
    )
    fig.show()


# ── 3c. Interactive Menu after each stock ────────────────────
def show_menu(sd, ticker, signal, sent_label, latest_close, start_date, end_date):
    """After printing the summary, let the user choose what to open."""
    hmap_cols = ['Close','20 Day MA','50 Day MA','Upper Band',
                 'Lower Band','RSI','MACD','ATR','VWAP','OBV','ADX']
    while True:
        print()
        print("  What do you want to do next?")
        print("  [1] Show price chart (candlestick + RSI + MACD)")
        print("  [2] Show correlation heatmap")
        print("  [3] Show both")
        print("  [4] Peer comparison (compare with other stocks)")
        print("  [5] Next stock / Exit")
        choice = input("  Enter choice (1/2/3/4/5): ").strip()

        if choice == '1':
            show_price_chart(sd, ticker, signal, sent_label, latest_close)
        elif choice == '2':
            hmap_data = sd[[c for c in hmap_cols if c in sd.columns]].dropna()
            correlation_heatmap(hmap_data)
        elif choice == '3':
            show_price_chart(sd, ticker, signal, sent_label, latest_close)
            hmap_data = sd[[c for c in hmap_cols if c in sd.columns]].dropna()
            correlation_heatmap(hmap_data)
        elif choice == '4':
            peers_raw = input("  Enter peer tickers (e.g. KOTAKBANK.NS, ICICIBANK.NS): ").strip()
            peers = [p.strip() for p in peers_raw.split(',') if p.strip()]
            if peers:
                print(f"\n  --- Peer Comparison (includes {ticker}) ---")
                peer_comparison([ticker] + peers)
            else:
                print("  No peers entered.")
        elif choice == '5':
            break
        else:
            print("  Invalid choice. Enter 1, 2, 3, 4 or 5.")


# ── 4. Real News Sentiment (Google News RSS — no API key) ────────
def fetch_news_sentiment(ticker):
    """
    Fetches live news from Google News RSS and scores with TextBlob.
    Falls back to Neutral (0.0) gracefully if anything fails.
    """
    try:
        search_term = ticker.replace('.NS', '').replace('.BO', '')
        url = (
            f"https://news.google.com/rss/search?"
            f"q={search_term}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed    = feedparser.parse(url)
        entries = feed.entries[:8]

        if not entries:
            print(f"  [!] No news found for {ticker}, using Neutral.")
            return 0.0, 0.0

        sentiment_score    = 0.0
        subjectivity_score = 0.0
        for entry in entries:
            text     = entry.get('title', '') + ' ' + entry.get('summary', '')
            analysis = TextBlob(text)
            sentiment_score    += analysis.sentiment.polarity
            subjectivity_score += analysis.sentiment.subjectivity

        return sentiment_score / len(entries), subjectivity_score / len(entries)

    except Exception as e:
        print(f"  [!] Sentiment error: {e}. Using Neutral.")
        return 0.0, 0.0


# ── 5. Buy / Sell / Hold Signal ──────────────────────────────────
def generate_signal(rsi, macd, signal_line, close, vwap, adx):
    """
    Scores RSI + MACD crossover + price vs VWAP + trend strength (ADX).
    Returns a clear BUY / SELL / HOLD string.
    """
    score = 0

    if rsi < 35:
        score += 2   # oversold
    elif rsi > 65:
        score -= 2   # overbought

    if macd > signal_line:
        score += 1   # bullish crossover
    else:
        score -= 1

    if close < vwap:
        score += 1   # below fair value
    else:
        score -= 1

    if adx < 20:
        score = 0    # weak trend → no trade

    if score >= 2:
        return "🟢 BUY"
    elif score <= -2:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"


# ── 6. Full Fundamentals ─────────────────────────────────────────
def fetch_fundamentals(ticker):
    """
    Fetches deep fundamental data:
    PE, PEG, Market Cap, ROE, ROCE approx, D/E, FCF, Revenue Growth,
    52W High/Low, Dividend Yield, Beta.
    """
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        def _fmt(val, prefix='', suffix='', decimals=2, scale=1):
            if val is None or val == 'N/A':
                return 'N/A'
            try:
                return f"{prefix}{val/scale:.{decimals}f}{suffix}"
            except Exception:
                return str(val)

        pe       = info.get('trailingPE')
        peg      = info.get('trailingPegRatio')
        mktcap   = info.get('marketCap')
        roe      = info.get('returnOnEquity')        # decimal, e.g. 0.18 = 18%
        de_ratio = info.get('debtToEquity')          # already a ratio
        fcf      = info.get('freeCashflow')
        rev_g    = info.get('revenueGrowth')          # YoY decimal
        div      = info.get('dividendYield')
        beta     = info.get('beta')
        high52   = info.get('fiftyTwoWeekHigh')
        low52    = info.get('fiftyTwoWeekLow')
        cur_pr   = info.get('currentPrice') or info.get('regularMarketPrice')

        # 52W position: where in the 52w range is current price?
        if high52 and low52 and cur_pr and high52 != low52:
            pos52 = f"{((cur_pr - low52)/(high52 - low52))*100:.1f}% from 52W low"
        else:
            pos52 = 'N/A'

        # Revenue growth YoY from financials if info doesn't have it
        if rev_g is None:
            try:
                fin = t.financials
                if fin is not None and not fin.empty and 'Total Revenue' in fin.index:
                    rev = fin.loc['Total Revenue'].dropna()
                    if len(rev) >= 2:
                        rev_g = (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1])
            except Exception:
                pass

        return {
            'PE Ratio'      : _fmt(pe, decimals=2) if pe else 'N/A',
            'PEG Ratio'     : _fmt(peg, decimals=2) if peg else 'N/A',
            'Market Cap'    : _fmt(mktcap, prefix='Rs.', suffix='B', scale=1e9) if mktcap else 'N/A',
            'ROE'           : _fmt(roe, suffix='%', scale=0.01) if roe else 'N/A',
            'Debt/Equity'   : _fmt(de_ratio, decimals=2) if de_ratio else 'N/A',
            'Free Cash Flow': _fmt(fcf, prefix='Rs.', suffix='Cr', scale=1e7) if fcf else 'N/A',
            'Revenue Growth': _fmt(rev_g, suffix='%', scale=0.01) if rev_g else 'N/A',
            'Dividend Yield': _fmt(div, suffix='%', scale=0.01) if div else 'N/A',
            'Beta'          : _fmt(beta, decimals=2) if beta else 'N/A',
            '52W High'      : _fmt(high52, prefix='Rs.', decimals=2) if high52 else 'N/A',
            '52W Low'       : _fmt(low52, prefix='Rs.', decimals=2) if low52 else 'N/A',
            '52W Position'  : pos52,
        }
    except Exception as e:
        return {}


# ── 6b. Risk Metrics ─────────────────────────────────────────────
def calculate_risk_metrics(close_series):
    """
    Max Drawdown, Annualized Volatility, Value at Risk (95%), Sharpe Ratio.
    """
    returns = close_series.pct_change().dropna()

    # Annualized Volatility
    ann_vol = returns.std() * np.sqrt(252) * 100

    # Max Drawdown
    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = drawdown.min() * 100

    # Value at Risk (95% confidence, daily)
    var_95 = np.percentile(returns, 5) * 100

    # Sharpe Ratio (assume risk-free = 6.5% annualized for India)
    rf_daily   = 0.065 / 252
    excess_ret = returns - rf_daily
    sharpe     = (excess_ret.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    return {
        'Ann. Volatility' : f"{ann_vol:.2f}%",
        'Max Drawdown'    : f"{max_dd:.2f}%",
        'VaR (95%, 1D)'   : f"{var_95:.2f}%",
        'Sharpe Ratio'    : f"{sharpe:.2f}",
    }


# ── 6c. Relative Strength vs Nifty ───────────────────────────────
def calculate_relative_strength(ticker, start_date, end_date):
    """
    Compares stock return vs Nifty 50 return over the same period.
    RS > 1 means stock outperformed Nifty.
    """
    try:
        nifty = yf.download('^NSEI', start=start_date, end=end_date,
                            interval='1d', progress=False, auto_adjust=True)
        stock = yf.download(ticker, start=start_date, end=end_date,
                            interval='1d', progress=False, auto_adjust=True)

        nifty = _flatten_columns(nifty)
        stock = _flatten_columns(stock)

        if nifty.empty or stock.empty:
            return None

        nifty_ret = (_scalar(nifty['Close'].iloc[-1]) / _scalar(nifty['Close'].iloc[0]) - 1) * 100
        stock_ret = (_scalar(stock['Close'].iloc[-1]) / _scalar(stock['Close'].iloc[0]) - 1) * 100
        rs        = stock_ret - nifty_ret

        return {
            'Stock Return'   : f"{stock_ret:+.2f}%",
            'Nifty Return'   : f"{nifty_ret:+.2f}%",
            'Outperformance' : f"{rs:+.2f}%  {'(Outperforming)' if rs > 0 else '(Underperforming)'}",
        }
    except Exception:
        return None


# ── 6d. Peer Comparison ───────────────────────────────────────────
def peer_comparison(peers):
    """
    Prints a side-by-side table of PE, ROE, D/E, Revenue Growth for a list of tickers.
    """
    rows = []
    print(f"\n  {'Ticker':<18} {'PE':>8} {'PEG':>8} {'ROE':>8} {'D/E':>8} {'Rev Grw':>9} {'Beta':>7}")
    print("  " + "-" * 62)
    for p in peers:
        try:
            info  = yf.Ticker(p).info
            pe    = info.get('trailingPE')
            peg   = info.get('trailingPegRatio')
            roe   = info.get('returnOnEquity')
            de    = info.get('debtToEquity')
            revg  = info.get('revenueGrowth')
            beta  = info.get('beta')

            pe_s  = f"{pe:.1f}"    if isinstance(pe, float)   else 'N/A'
            peg_s = f"{peg:.2f}"   if isinstance(peg, float)  else 'N/A'
            roe_s = f"{roe*100:.1f}%" if isinstance(roe, float) else 'N/A'
            de_s  = f"{de:.2f}"    if isinstance(de, float)   else 'N/A'
            rg_s  = f"{revg*100:.1f}%" if isinstance(revg, float) else 'N/A'
            bt_s  = f"{beta:.2f}"  if isinstance(beta, float) else 'N/A'

            print(f"  {p:<18} {pe_s:>8} {peg_s:>8} {roe_s:>8} {de_s:>8} {rg_s:>9} {bt_s:>7}")
        except Exception:
            print(f"  {p:<18} {'ERROR':>8}")
    print()


# ── 7. VWAP — correct daily-reset implementation ─────────────────
def calculate_daily_vwap(data):
    """
    True VWAP resets every trading day.
    Groups by calendar date, then computes cumulative TP*Vol / cumVol.
    """
    data        = data.copy()
    data['_dt'] = data.index.date
    data['_tp'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['_tpv']= data['_tp'] * data['Volume']

    data['_ctpv'] = data.groupby('_dt')['_tpv'].cumsum()
    data['_cvol']  = data.groupby('_dt')['Volume'].cumsum()
    data['VWAP']   = data['_ctpv'] / data['_cvol']

    data.drop(columns=['_dt', '_tp', '_tpv', '_ctpv', '_cvol'], inplace=True)
    return data


# ── 8. ATR — named Series to avoid pd.concat column clash ────────
def calculate_atr(df):
    """True Range uses named Series so pd.concat doesn't create duplicate column names."""
    hl = (df['High'] - df['Low']).rename('hl')
    hc = np.abs(df['High'] - df['Close'].shift()).rename('hc')
    lc = np.abs(df['Low']  - df['Close'].shift()).rename('lc')
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14).mean()


# ── 9. ADX ───────────────────────────────────────────────────────
def calculate_adx(df, n=14):
    df = df.copy()
    df['+DM'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        (df['High'] - df['High'].shift(1)).clip(lower=0), 0)
    df['-DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        (df['Low'].shift(1) - df['Low']).clip(lower=0), 0)
    df['+DI'] = 100 * (df['+DM'].rolling(n).sum() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(n).sum() / df['ATR'])
    denom     = (df['+DI'] + df['-DI']).replace(0, np.nan)   # avoid div-by-zero
    df['DX']  = (np.abs(df['+DI'] - df['-DI']) / denom) * 100
    df['ADX'] = df['DX'].rolling(n).mean()
    return df


# ── 10. RSI ──────────────────────────────────────────────────────
def calculate_rsi(series, period=14):
    delta    = series.diff(1)
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── 11. Main Analysis ────────────────────────────────────────────
def detailed_stock_analysis(tickers, start_date=None, end_date=None):
    if not start_date or not end_date:
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=365)   # 1 year default for better MA coverage

    for ticker in tickers:
        print(f"\n{'='*55}")
        print(f"  Analysing : {ticker}")
        print(f"{'='*55}")

        try:
            # ── Fetch & clean data ────────────────────────────
            raw = yf.download(ticker, start=start_date, end=end_date,
                              interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                print(f"  [!] No data for {ticker}. Check ticker symbol.")
                continue

            sd = _flatten_columns(raw)

            # Ensure we have required columns
            required = {'Open', 'High', 'Low', 'Close', 'Volume'}
            missing  = required - set(sd.columns)
            if missing:
                print(f"  [!] Missing columns {missing} for {ticker}. Skipping.")
                continue

            # Ensure numeric dtypes
            for col in required:
                sd[col] = pd.to_numeric(sd[col], errors='coerce')
            sd.dropna(subset=['Close', 'Volume'], inplace=True)

            if len(sd) < 30:
                print(f"  [!] Too few rows ({len(sd)}) for {ticker}. Try a wider date range.")
                continue

            # ── Moving Averages ───────────────────────────────
            sd['20 Day MA']  = sd['Close'].rolling(20).mean()
            sd['50 Day MA']  = sd['Close'].rolling(50).mean()
            sd['100 Day MA'] = sd['Close'].rolling(100).mean()

            # ── Bollinger Bands ───────────────────────────────
            sd['20 Day STD'] = sd['Close'].rolling(20).std()
            sd['Upper Band'] = sd['20 Day MA'] + 2 * sd['20 Day STD']
            sd['Lower Band'] = sd['20 Day MA'] - 2 * sd['20 Day STD']

            # ── RSI ───────────────────────────────────────────
            sd['RSI'] = calculate_rsi(sd['Close'])

            # ── MACD ──────────────────────────────────────────
            sd['12 EMA']      = sd['Close'].ewm(span=12, adjust=False).mean()
            sd['26 EMA']      = sd['Close'].ewm(span=26, adjust=False).mean()
            sd['MACD']        = sd['12 EMA'] - sd['26 EMA']
            sd['Signal Line'] = sd['MACD'].ewm(span=9, adjust=False).mean()

            # ── ATR ───────────────────────────────────────────
            sd['ATR'] = calculate_atr(sd)

            # ── VWAP (daily reset) ────────────────────────────
            sd = calculate_daily_vwap(sd)

            # ── OBV ───────────────────────────────────────────
            sd['OBV'] = (np.sign(sd['Close'].diff()) * sd['Volume']).fillna(0).cumsum()

            # ── ADX ───────────────────────────────────────────
            sd = calculate_adx(sd)

            # ── Latest non-NaN row ────────────────────────────
            key_cols = ['RSI', 'MACD', 'Signal Line', 'VWAP', 'ADX']
            valid    = sd.dropna(subset=key_cols)

            if valid.empty:
                print(f"  [!] Indicators still NaN for {ticker}. "
                      f"Try extending the date range (need at least 100 days).")
                continue

            latest       = valid.iloc[-1]
            latest_rsi   = _scalar(latest['RSI'])
            latest_macd  = _scalar(latest['MACD'])
            latest_sig   = _scalar(latest['Signal Line'])
            latest_close = _scalar(latest['Close'])
            latest_vwap  = _scalar(latest['VWAP'])
            latest_adx   = _scalar(latest['ADX'])

            # ── Signal ────────────────────────────────────────
            signal = generate_signal(
                latest_rsi, latest_macd, latest_sig,
                latest_close, latest_vwap, latest_adx
            )

            # ── Support / Resistance / Fibonacci ──────────────
            support, resistance = calculate_support_resistance(sd)
            fib1, fib2, fib3   = calculate_fibonacci_levels(sd)

            # ── Fundamentals ──────────────────────────────────
            fundamentals = fetch_fundamentals(ticker)

            # ── Sentiment ─────────────────────────────────────
            avg_sent, avg_subj = fetch_news_sentiment(ticker)
            if avg_sent > 0.05:
                sent_label = "Positive"
            elif avg_sent < -0.05:
                sent_label = "Negative"
            else:
                sent_label = "Neutral"

            # ── Risk Metrics ──────────────────────────────────
            risk = calculate_risk_metrics(sd['Close'])

            # ── Relative Strength vs Nifty ────────────────────
            rs_data = calculate_relative_strength(ticker, start_date, end_date)

            # ── Console Output ────────────────────────────────
            print(f"\n  Latest Close     : Rs.{latest_close:.2f}")
            print(f"  Signal           : {signal}")

            print(f"\n  --- Technical Indicators ---")
            print(f"  RSI              : {latest_rsi:.2f}   (>70 overbought | <30 oversold)")
            print(f"  MACD             : {latest_macd:.4f}  |  Signal Line: {latest_sig:.4f}")
            print(f"  ADX              : {latest_adx:.2f}   (>25 = strong trend)")
            print(f"  VWAP             : Rs.{latest_vwap:.2f}")
            print(f"  Support          : Rs.{support:.2f}")
            print(f"  Resistance       : Rs.{resistance:.2f}")
            print(f"  Fibonacci 23.6%  : Rs.{fib1:.2f}")
            print(f"  Fibonacci 38.2%  : Rs.{fib2:.2f}")
            print(f"  Fibonacci 61.8%  : Rs.{fib3:.2f}")

            if fundamentals:
                print(f"\n  --- Fundamentals ---")
                for k, v in fundamentals.items():
                    print(f"  {k:<16} : {v}")

            print(f"\n  --- Risk Metrics ---")
            for k, v in risk.items():
                print(f"  {k:<16} : {v}")

            if rs_data:
                print(f"\n  --- vs Nifty 50 (same period) ---")
                for k, v in rs_data.items():
                    print(f"  {k:<16} : {v}")

            print(f"\n  --- News Sentiment ---")
            print(f"  Score            : {avg_sent:.3f}  ->  {sent_label}")
            print(f"  Subjectivity     : {avg_subj:.3f}")

            # ── Hand off to menu ──────────────────────────────
            show_menu(sd, ticker, signal, sent_label, latest_close, start_date, end_date)

        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
            import traceback
            traceback.print_exc()


# ── Entry Point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Stock Analysis Tool -- by Vishesh Sanghvi")
    print("="*55 + "\n")

    tickers_raw = input("Enter tickers (e.g. HDFCBANK.NS, TATAMOTORS.NS): ")
    tickers     = [t.strip() for t in tickers_raw.split(',') if t.strip()]

    if not tickers:
        print("No tickers entered. Exiting.")
        exit()

    start_input = input("Start date (YYYY-MM-DD) or Enter for last 1 year : ").strip()
    end_input   = input("End date   (YYYY-MM-DD) or Enter for today        : ").strip()

    detailed_stock_analysis(
        tickers,
        start_date=start_input or None,
        end_date=end_input or None
    )
