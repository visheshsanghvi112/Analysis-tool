# ============================================================
# Stock Analysis Engine — by Vishesh Sanghvi
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import feedparser
import warnings
warnings.filterwarnings('ignore')


def _scalar(val):
    """Safely extracts a scalar float from pandas Series or numpy types."""
    if isinstance(val, pd.Series):
        if val.empty:
            return 0.0
        val = val.iloc[0]
    try:
        return float(val)
    except Exception:
        return 0.0


def _flatten_columns(df):
    """Strips multiindex levels and keeps the first column level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def calculate_support_resistance(data):
    resistance = _scalar(data['Close'].max())
    support    = _scalar(data['Close'].min())
    return support, resistance


def calculate_fibonacci_levels(data):
    max_price = _scalar(data['Close'].max())
    min_price = _scalar(data['Close'].min())
    diff = max_price - min_price
    return (
        max_price - 0.236 * diff,
        max_price - 0.382 * diff,
        max_price - 0.618 * diff,
    )


def fetch_news_sentiment(ticker):
    """Fetches live news headlines from Google News RSS and scores them with TextBlob."""
    try:
        search_term = ticker.replace('.NS', '').replace('.BO', '')
        url = (
            f"https://news.google.com/rss/search?"
            f"q={search_term}+stock+India&hl=en-IN&gl=IN&ceid=IN:en"
        )
        feed    = feedparser.parse(url)
        entries = feed.entries[:8]

        if not entries:
            return 0.0, 0.0

        sentiment_score    = 0.0
        subjectivity_score = 0.0
        for entry in entries:
            text     = entry.get('title', '') + ' ' + entry.get('summary', '')
            analysis = TextBlob(text)
            sentiment_score    += analysis.sentiment.polarity
            subjectivity_score += analysis.sentiment.subjectivity

        return sentiment_score / len(entries), subjectivity_score / len(entries)
    except Exception:
        return 0.0, 0.0


def generate_signal(rsi, macd, signal_line, close, vwap, adx):
    """Scores RSI, MACD, VWAP, and ADX to produce a BUY, SELL, or HOLD recommendation."""
    score = 0

    if rsi < 35:
        score += 2
    elif rsi > 65:
        score -= 2

    if macd > signal_line:
        score += 1
    else:
        score -= 1

    if close < vwap:
        score += 1
    else:
        score -= 1

    if adx < 20:
        score = 0

    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"


def fetch_fundamentals(ticker):
    """Fetches key valuation, leverage, cash flow and growth metrics from yfinance."""
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        def _raw_val(val):
            return val if (val is not None and val != 'N/A') else None

        pe       = _raw_val(info.get('trailingPE'))
        peg      = _raw_val(info.get('trailingPegRatio'))
        mktcap   = _raw_val(info.get('marketCap'))
        roe      = _raw_val(info.get('returnOnEquity'))
        de_ratio = _raw_val(info.get('debtToEquity'))
        fcf      = _raw_val(info.get('freeCashflow'))
        rev_g    = _raw_val(info.get('revenueGrowth'))
        div      = _raw_val(info.get('dividendYield'))
        beta     = _raw_val(info.get('beta'))
        high52   = _raw_val(info.get('fiftyTwoWeekHigh'))
        low52    = _raw_val(info.get('fiftyTwoWeekLow'))
        cur_pr   = _raw_val(info.get('currentPrice') or info.get('regularMarketPrice'))

        # Fetch revenue growth from financials if info is empty
        if rev_g is None:
            try:
                fin = t.financials
                if fin is not None and not fin.empty and 'Total Revenue' in fin.index:
                    rev = fin.loc['Total Revenue'].dropna()
                    if len(rev) >= 2:
                        rev_g = float((rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1]))
            except Exception:
                pass

        return {
            'peRatio'       : round(pe, 2) if pe else None,
            'pegRatio'      : round(peg, 2) if peg else None,
            'marketCap'     : mktcap,
            'roe'           : round(roe * 100, 2) if roe else None,
            'debtToEquity'  : round(de_ratio, 2) if de_ratio else None,
            'freeCashFlow'  : fcf,
            'revenueGrowth' : round(rev_g * 100, 2) if rev_g else None,
            'dividendYield' : round(div * 100, 2) if div else None,
            'beta'          : round(beta, 2) if beta else None,
            'fiftyTwoWeekHigh': high52,
            'fiftyTwoWeekLow' : low52,
            'currentPrice'  : cur_pr
        }
    except Exception:
        return {}


def calculate_daily_vwap(data):
    """Computes daily resetting Volume Weighted Average Price."""
    data        = data.copy()
    data['_dt'] = data.index.date
    data['_tp'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['_tpv']= data['_tp'] * data['Volume']

    data['_ctpv'] = data.groupby('_dt')['_tpv'].cumsum()
    data['_cvol']  = data.groupby('_dt')['Volume'].cumsum()
    data['VWAP']   = data['_ctpv'] / data['_cvol']

    data.drop(columns=['_dt', '_tp', '_tpv', '_ctpv', '_cvol'], inplace=True)
    return data


def calculate_atr(df):
    hl = (df['High'] - df['Low']).rename('hl')
    hc = np.abs(df['High'] - df['Close'].shift()).rename('hc')
    lc = np.abs(df['Low']  - df['Close'].shift()).rename('lc')
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14).mean()


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
    denom     = (df['+DI'] + df['-DI']).replace(0, np.nan)
    df['DX']  = (np.abs(df['+DI'] - df['-DI']) / denom) * 100
    df['ADX'] = df['DX'].rolling(n).mean()
    return df


def calculate_rsi(series, period=14):
    delta    = series.diff(1)
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_risk_metrics(close_series):
    """Computes drawdown, Sharpe ratio, daily VaR, volatility."""
    returns = close_series.pct_change().dropna()
    if returns.empty:
        return {}

    ann_vol = float(returns.std() * np.sqrt(252) * 100)

    cumulative  = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = float(drawdown.min() * 100)

    var_95 = float(np.percentile(returns, 5) * 100)

    rf_daily   = 0.065 / 252
    excess_ret = returns - rf_daily
    sharpe     = float((excess_ret.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0

    return {
        'annualizedVolatility': round(ann_vol, 2),
        'maxDrawdown': round(max_dd, 2),
        'var95_1D': round(var_95, 2),
        'sharpeRatio': round(sharpe, 2)
    }


def calculate_relative_strength(ticker, start_date, end_date):
    """Compares the cumulative return of the stock vs Nifty 50 Index."""
    try:
        nifty = yf.download('^NSEI', start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)
        stock = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=True)

        nifty = _flatten_columns(nifty)
        stock = _flatten_columns(stock)

        if nifty.empty or stock.empty:
            return {}

        n_ret = (_scalar(nifty['Close'].iloc[-1]) / _scalar(nifty['Close'].iloc[0]) - 1) * 100
        s_ret = (_scalar(stock['Close'].iloc[-1]) / _scalar(stock['Close'].iloc[0]) - 1) * 100
        outperformance = s_ret - n_ret

        return {
            'stockReturn': round(s_ret, 2),
            'niftyReturn': round(n_ret, 2),
            'outperformance': round(outperformance, 2)
        }
    except Exception:
        return {}


def analyze_ticker(ticker, start_date=None, end_date=None):
    """Runs full analytics suite on a ticker and returns a structured dictionary."""
    if not start_date or not end_date:
        end_dt   = datetime.today()
        start_dt = end_dt - timedelta(days=365)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date   = end_dt.strftime('%Y-%m-%d')

    raw = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        return {"error": f"No data found for ticker {ticker}."}

    sd = _flatten_columns(raw)

    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    missing  = required - set(sd.columns)
    if missing:
        return {"error": f"Missing required price fields: {missing}"}

    for col in required:
        sd[col] = pd.to_numeric(sd[col], errors='coerce')
    sd.dropna(subset=['Close', 'Volume'], inplace=True)

    if len(sd) < 30:
        return {"error": "Insufficient historical data (less than 30 valid days)."}

    # Calculations
    sd['20 Day MA']  = sd['Close'].rolling(20).mean()
    sd['50 Day MA']  = sd['Close'].rolling(50).mean()
    sd['100 Day MA'] = sd['Close'].rolling(100).mean()
    sd['20 Day STD'] = sd['Close'].rolling(20).std()
    sd['Upper Band'] = sd['20 Day MA'] + 2 * sd['20 Day STD']
    sd['Lower Band'] = sd['20 Day MA'] - 2 * sd['20 Day STD']

    sd['RSI'] = calculate_rsi(sd['Close'])
    sd['12 EMA']      = sd['Close'].ewm(span=12, adjust=False).mean()
    sd['26 EMA']      = sd['Close'].ewm(span=26, adjust=False).mean()
    sd['MACD']        = sd['12 EMA'] - sd['26 EMA']
    sd['Signal Line'] = sd['MACD'].ewm(span=9, adjust=False).mean()

    sd['ATR'] = calculate_atr(sd)
    sd        = calculate_daily_vwap(sd)
    sd['OBV'] = (np.sign(sd['Close'].diff()) * sd['Volume']).fillna(0).cumsum()
    sd        = calculate_adx(sd)

    valid = sd.dropna(subset=['RSI', 'MACD', 'Signal Line', 'VWAP', 'ADX'])
    if valid.empty:
        return {"error": "Indicators could not be computed (try a larger date range)."}

    latest       = valid.iloc[-1]
    latest_rsi   = _scalar(latest['RSI'])
    latest_macd  = _scalar(latest['MACD'])
    latest_sig   = _scalar(latest['Signal Line'])
    latest_close = _scalar(latest['Close'])
    latest_vwap  = _scalar(latest['VWAP'])
    latest_adx   = _scalar(latest['ADX'])

    signal = generate_signal(latest_rsi, latest_macd, latest_sig, latest_close, latest_vwap, latest_adx)
    support, resistance = calculate_support_resistance(sd)
    fib1, fib2, fib3   = calculate_fibonacci_levels(sd)

    fundamentals = fetch_fundamentals(ticker)
    risk         = calculate_risk_metrics(sd['Close'])
    rs_data      = calculate_relative_strength(ticker, start_date, end_date)
    avg_sent, avg_subj = fetch_news_sentiment(ticker)

    # Convert timeseries to JSON-friendly format for Plotly/Recharts
    timeseries_data = []
    # Drop rows that are entirely NaN in the main columns to save payload size
    chart_df = sd.copy().dropna(subset=['Close'])
    for idx, row in chart_df.iterrows():
        timeseries_data.append({
            'date': idx.strftime('%Y-%m-%d'),
            'open': round(_scalar(row['Open']), 2),
            'high': round(_scalar(row['High']), 2),
            'low': round(_scalar(row['Low']), 2),
            'close': round(_scalar(row['Close']), 2),
            'volume': int(_scalar(row['Volume'])),
            'ma20': round(_scalar(row['20 Day MA']), 2) if not pd.isna(row['20 Day MA']) else None,
            'ma50': round(_scalar(row['50 Day MA']), 2) if not pd.isna(row['50 Day MA']) else None,
            'ma100': round(_scalar(row['100 Day MA']), 2) if not pd.isna(row['100 Day MA']) else None,
            'upperBand': round(_scalar(row['Upper Band']), 2) if not pd.isna(row['Upper Band']) else None,
            'lowerBand': round(_scalar(row['Lower Band']), 2) if not pd.isna(row['Lower Band']) else None,
            'vwap': round(_scalar(row['VWAP']), 2) if not pd.isna(row['VWAP']) else None,
            'rsi': round(_scalar(row['RSI']), 2) if not pd.isna(row['RSI']) else None,
            'macd': round(_scalar(row['MACD']), 2) if not pd.isna(row['MACD']) else None,
            'macdSignal': round(_scalar(row['Signal Line']), 2) if not pd.isna(row['Signal Line']) else None,
            'adx': round(_scalar(row['ADX']), 2) if not pd.isna(row['ADX']) else None,
        })

    return {
        'ticker': ticker,
        'summary': {
            'close': round(latest_close, 2),
            'signal': signal,
            'support': round(support, 2),
            'resistance': round(resistance, 2),
            'fib236': round(fib1, 2),
            'fib382': round(fib2, 2),
            'fib618': round(fib3, 2),
            'rsi': round(latest_rsi, 2),
            'macd': round(latest_macd, 4),
            'macdSignal': round(latest_sig, 4),
            'adx': round(latest_adx, 2),
            'vwap': round(latest_vwap, 2)
        },
        'fundamentals': fundamentals,
        'risk': risk,
        'relativeStrength': rs_data,
        'sentiment': {
            'score': round(avg_sent, 3),
            'subjectivity': round(avg_subj, 3),
            'label': 'Positive' if avg_sent > 0.05 else 'Negative' if avg_sent < -0.05 else 'Neutral'
        },
        'chartData': timeseries_data
    }
