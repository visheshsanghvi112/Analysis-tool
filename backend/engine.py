# ============================================================
# Stock Analysis Engine — by Vishesh Sanghvi
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import feedparser
import warnings
warnings.filterwarnings('ignore')

from yf_client import get_history, get_quote, get_info


def _scalar(val):
    if isinstance(val, pd.Series):
        if val.empty:
            return 0.0
        val = val.iloc[0]
    try:
        return float(val)
    except Exception:
        return 0.0


def _flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def calculate_support_resistance(data, window=10):
    if len(data) < 2 * window + 1:
        cur = _scalar(data['Close'].iloc[-1]) if not data.empty else 0.0
        return round(cur * 0.95, 2), round(cur * 1.05, 2)

    highs = data['High']
    lows  = data['Low']
    curr_price = _scalar(data['Close'].iloc[-1])
    
    swing_highs, swing_lows = [], []

    for i in range(window, len(data) - window):
        if highs.iloc[i] == highs.iloc[i - window: i + window + 1].max():
            swing_highs.append(_scalar(highs.iloc[i]))
        if lows.iloc[i] == lows.iloc[i - window: i + window + 1].min():
            swing_lows.append(_scalar(lows.iloc[i]))

    # Valid candidate supports: swing lows strictly below current market price
    valid_supports = [s for s in swing_lows if s < curr_price * 0.999]
    if valid_supports:
        recent_support = valid_supports[-1]
    else:
        recent_low = _scalar(lows.tail(50).min())
        recent_support = recent_low if recent_low < curr_price else curr_price * 0.95

    # Valid candidate resistances: swing highs strictly above current market price
    valid_resistances = [r for r in swing_highs if r > curr_price * 1.001]
    if valid_resistances:
        recent_resistance = valid_resistances[-1]
    else:
        recent_high = _scalar(highs.tail(50).max())
        recent_resistance = recent_high if recent_high > curr_price else curr_price * 1.05

    # Guarantee support strictly < curr_price strictly < resistance
    if recent_support >= curr_price:
        recent_support = curr_price * 0.96
    if recent_resistance <= curr_price:
        recent_resistance = curr_price * 1.04

    return round(recent_support, 2), round(recent_resistance, 2)


def calculate_fibonacci_levels(data):
    lookback  = data.tail(126)
    max_price = _scalar(lookback['High'].max())
    min_price = _scalar(lookback['Low'].min())
    diff = max_price - min_price
    if diff == 0:
        return max_price, max_price, max_price
    return (
        max_price - 0.236 * diff,
        max_price - 0.382 * diff,
        max_price - 0.618 * diff,
    )


def fetch_news_sentiment(ticker):
    try:
        search_term = ticker.replace('.NS', '').replace('.BO', '')
        is_us = not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^')
        query = f"{search_term}+stock" if is_us else f"{search_term}+stock+India"
        hl_gl = "hl=en-US&gl=US&ceid=US:en" if is_us else "hl=en-IN&gl=IN&ceid=IN:en"
        url = f"https://news.google.com/rss/search?q={query}&{hl_gl}"
        feed    = feedparser.parse(url)
        entries = feed.entries[:10]
        if not entries:
            return 0.0, 0.0, []

        sentiment_score = subjectivity_score = 0.0
        headlines = []
        for entry in entries:
            text     = entry.get('title', '') + ' ' + entry.get('summary', '')
            analysis = TextBlob(text)
            sentiment_score    += analysis.sentiment.polarity
            subjectivity_score += analysis.sentiment.subjectivity
            headlines.append({
                'title':     entry.get('title', ''),
                'link':      entry.get('link', ''),
                'published': entry.get('published', ''),
                'polarity':  round(analysis.sentiment.polarity, 3),
            })

        n = len(entries)
        return sentiment_score / n, subjectivity_score / n, headlines
    except Exception:
        return 0.0, 0.0, []


def generate_signal(rsi, macd, signal_line, close, upper_band, lower_band,
                    adx, prev_macd=None, prev_signal=None, ma50=None, ma200=None):
    reasons = []
    score   = 0

    if rsi < 35:
        score += 2; reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 65:
        score -= 2; reasons.append(f"RSI overbought ({rsi:.1f})")
    else:
        reasons.append(f"RSI neutral ({rsi:.1f})")

    if prev_macd is not None and prev_signal is not None:
        was_above = prev_macd > prev_signal
        was_below = prev_macd < prev_signal
        is_above  = macd > signal_line
        is_below  = macd < signal_line

        if was_below and is_above:
            score += 2; reasons.append("MACD bullish crossover")
        elif was_above and is_below:
            score -= 2; reasons.append("MACD bearish crossover")
        elif is_above:
            score += 1; reasons.append("MACD above signal (bullish)")
        elif is_below:
            score -= 1; reasons.append("MACD below signal (bearish)")
        else:
            reasons.append("MACD converged on signal line")
    else:
        if macd > signal_line:
            score += 1; reasons.append("MACD above signal (bullish)")
        elif macd < signal_line:
            score -= 1; reasons.append("MACD below signal (bearish)")
        else:
            reasons.append("MACD converged on signal line")

    band_range = upper_band - lower_band
    if band_range > 0:
        pct_b = (close - lower_band) / band_range
        if pct_b < 0.2:
            score += 1; reasons.append(f"%B near lower band ({pct_b:.2f})")
        elif pct_b > 0.8:
            score -= 1; reasons.append(f"%B near upper band ({pct_b:.2f})")

    if adx < 15:
        score = max(-1, min(1, score))
        reasons.append(f"ADX weak ({adx:.1f}) — ranging market")
    elif adx > 25:
        score += 1 if score > 0 else (-1 if score < 0 else 0)
        reasons.append(f"ADX strong ({adx:.1f}) — trend confirmed")
    else:
        reasons.append(f"ADX moderate ({adx:.1f})")

    if ma50 is not None and ma200 is not None and not np.isnan(ma50) and not np.isnan(ma200):
        if ma50 > ma200:
            score += 1; reasons.append(f"Golden Cross structure (50 DMA > 200 DMA)")
        else:
            score -= 1; reasons.append(f"Death Cross structure (50 DMA < 200 DMA)")

    signal = "BUY" if score >= 3 else "SELL" if score <= -3 else "HOLD"
    return signal, score, reasons


def fetch_fundamentals(ticker):
    try:
        info = get_info(ticker)
        if not info:
            return {}

        def _v(key):
            val = info.get(key)
            return val if val not in (None, 'N/A') else None

        pe         = _v('trailingPE') or _v('forwardPE')
        fwd_pe     = _v('forwardPE')
        peg        = _v('trailingPegRatio') or _v('pegRatio')
        pb         = _v('priceToBook')
        ps         = _v('priceToSalesTrailing12Months')
        mktcap     = _v('marketCap')
        roe        = _v('returnOnEquity')
        roa        = _v('returnOnAssets')
        de_ratio   = _v('debtToEquity')
        current_r  = _v('currentRatio')
        quick_r    = _v('quickRatio')
        fcf        = _v('freeCashflow')
        rev_g      = _v('revenueGrowth')
        earn_g     = _v('earningsGrowth')
        div        = _v('dividendYield') or _v('trailingAnnualDividendYield')
        beta       = _v('beta')
        high52     = _v('fiftyTwoWeekHigh')
        low52      = _v('fiftyTwoWeekLow')
        cur_pr     = _v('regularMarketPrice') or _v('currentPrice')
        prev_close = _v('regularMarketPreviousClose') or _v('previousClose')
        open_pr    = _v('regularMarketOpen')
        day_high   = _v('regularMarketDayHigh')
        day_low    = _v('regularMarketDayLow')
        volume     = _v('regularMarketVolume')
        avg_volume = _v('averageVolume') or _v('averageDailyVolume10Day')
        eps        = _v('trailingEps')
        fwd_eps    = _v('forwardEps')
        book_val   = _v('bookValue')
        op_margin  = _v('operatingMargins')
        prof_mar   = _v('profitMargins')
        gross_mar  = _v('grossMargins')
        sector     = info.get('sector') or info.get('sectorDisp')
        industry   = info.get('industry') or info.get('industryDisp')
        name       = info.get('longName') or info.get('shortName', ticker)
        short_pct  = _v('shortPercentOfFloat')
        shares_out = _v('sharesOutstanding')
        float_sh   = _v('floatShares')

        price_change = price_change_pct = None
        if cur_pr and prev_close and prev_close != 0:
            price_change     = round(cur_pr - prev_close, 2)
            price_change_pct = round((cur_pr - prev_close) / prev_close * 100, 2)

        pos_52w = None
        if high52 and low52 and high52 != low52 and cur_pr:
            pos_52w = round((cur_pr - low52) / (high52 - low52) * 100, 1)

        vol_ratio = None
        if volume and avg_volume and avg_volume > 0:
            vol_ratio = round(volume / avg_volume, 2)

        return {
            'name': name, 'sector': sector, 'industry': industry,
            'peRatio':       round(pe, 2)       if pe       else None,
            'forwardPE':     round(fwd_pe, 2)   if fwd_pe   else None,
            'pegRatio':      round(peg, 2)       if peg      else None,
            'pbRatio':       round(pb, 2)        if pb       else None,
            'psRatio':       round(ps, 2)        if ps       else None,
            'marketCap':     mktcap,
            'eps':           round(eps, 2)       if eps      else None,
            'forwardEps':    round(fwd_eps, 2)   if fwd_eps  else None,
            'bookValue':     round(book_val, 2)  if book_val else None,
            'roe':           round(roe * 100, 2) if roe      else None,
            'roa':           round(roa * 100, 2) if roa      else None,
            'operatingMargin': round(op_margin * 100, 2) if op_margin else None,
            'profitMargin':  round(prof_mar * 100, 2)    if prof_mar  else None,
            'grossMargin':   round(gross_mar * 100, 2)   if gross_mar else None,
            'revenueGrowth': round(rev_g * 100, 2)       if rev_g     else None,
            'earningsGrowth':round(earn_g * 100, 2)      if earn_g    else None,
            'debtToEquity':  round(de_ratio, 2)          if de_ratio  else None,
            'currentRatio':  round(current_r, 2)         if current_r else None,
            'quickRatio':    round(quick_r, 2)           if quick_r   else None,
            'freeCashFlow':  fcf,
            'dividendYield': round(div * 100, 2)         if div       else None,
            'beta':          round(beta, 2)               if beta      else None,
            'shortPercent':  round(short_pct * 100, 2)   if short_pct else None,
            'fiftyTwoWeekHigh': high52,
            'fiftyTwoWeekLow':  low52,
            'fiftyTwoWeekPos':  pos_52w,
            'currentPrice':  cur_pr,
            'previousClose': prev_close,
            'openPrice':     open_pr,
            'dayHigh':       day_high,
            'dayLow':        day_low,
            'priceChange':   price_change,
            'priceChangePct':price_change_pct,
            'volume':        int(volume)     if volume     else None,
            'avgVolume':     int(avg_volume) if avg_volume else None,
            'volumeRatio':   vol_ratio,
            'sharesOutstanding': shares_out,
            'floatShares':   float_sh,
        }
    except Exception:
        return {}


def calculate_atr(df, n=14):
    hl = (df['High'] - df['Low']).rename('hl')
    hc = np.abs(df['High'] - df['Close'].shift()).rename('hc')
    lc = np.abs(df['Low']  - df['Close'].shift()).rename('lc')
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    # Welles Wilder exponential smoothing (RMA)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def calculate_adx(df, n=14):
    df = df.copy()
    up_move = df['High'] - df['High'].shift(1)
    down_move = df['Low'].shift(1) - df['Low']

    df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Welles Wilder smoothed +DM, -DM, and TR
    smooth_pdm = df['+DM'].ewm(alpha=1.0 / n, adjust=False).mean()
    smooth_mdm = df['-DM'].ewm(alpha=1.0 / n, adjust=False).mean()
    smooth_tr  = df['ATR'].replace(0, np.nan)

    df['+DI'] = (100.0 * (smooth_pdm / smooth_tr)).clip(lower=0.0, upper=100.0)
    df['-DI'] = (100.0 * (smooth_mdm / smooth_tr)).clip(lower=0.0, upper=100.0)

    di_sum    = (df['+DI'] + df['-DI']).replace(0, np.nan)
    df['DX']  = ((np.abs(df['+DI'] - df['-DI']) / di_sum) * 100.0).clip(lower=0.0, upper=100.0)
    df['ADX'] = df['DX'].ewm(alpha=1.0 / n, adjust=False).mean().clip(lower=0.0, upper=100.0)
    return df


def calculate_rsi(series, period=14):
    delta    = series.diff(1)
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_risk_metrics(close_series):
    returns = close_series.pct_change().dropna()
    if returns.empty:
        return {}

    ann_vol    = float(returns.std() * np.sqrt(252) * 100)
    cumulative = (1 + returns).cumprod()
    rolling_max= cumulative.cummax()
    drawdown   = (cumulative - rolling_max) / rolling_max
    max_dd     = float(drawdown.min() * 100)
    var_95     = float(np.percentile(returns, 5) * 100)
    var_99     = float(np.percentile(returns, 1) * 100)
    
    # Expected Shortfall (CVaR 95%): average loss beyond 95% VaR
    tail_95    = returns[returns <= np.percentile(returns, 5)]
    cvar_95    = float(tail_95.mean() * 100) if len(tail_95) > 0 else var_95

    rf_daily   = 0.065 / 252
    excess_ret = returns - rf_daily
    sharpe     = float((excess_ret.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0
    downside   = excess_ret[excess_ret < 0]
    sortino    = float((excess_ret.mean() / downside.std()) * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0.0

    # Calmar Ratio: Compound Annual Growth Rate (CAGR) / |Max Drawdown|
    n_days = max(len(returns), 1)
    if cumulative.iloc[-1] > 0:
        cagr = float(((cumulative.iloc[-1]) ** (252.0 / n_days) - 1.0) * 100.0)
    else:
        cagr = -100.0
    abs_dd     = abs(max_dd)
    calmar     = round(cagr / abs_dd, 2) if abs_dd > 0.5 else None

    skew       = round(float(returns.skew()), 2) if len(returns) > 2 else 0.0
    kurt       = round(float(returns.kurtosis()), 2) if len(returns) > 3 else 0.0

    return {
        'annualizedVolatility': round(ann_vol, 2),
        'maxDrawdown':          round(max_dd, 2),
        'var95_1D':             round(var_95, 2),
        'var99_1D':             round(var_99, 2),
        'cvar95_1D':            round(cvar_95, 2),
        'sharpeRatio':          round(sharpe, 2),
        'sortinoRatio':         round(sortino, 2),
        'calmarRatio':          calmar,
        'skewness':             skew,
        'kurtosis':             kurt,
    }


def calculate_relative_strength(ticker, start_date=None, end_date=None, days=None):
    try:
        is_us = not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^')
        bench_ticker = '^GSPC' if is_us else '^NSEI'
        bench_name   = 'S&P 500' if is_us else 'NIFTY 50'

        period = '1y'
        if days:
            if days <= 35:
                period = '1mo'
            elif days <= 100:
                period = '3mo'
            elif days <= 200:
                period = '6mo'
            elif days <= 370:
                period = '1y'
            elif days <= 750:
                period = '2y'
            else:
                period = '5y'

        bench = get_history(bench_ticker, period=period)
        stock = get_history(ticker, period=period)
        if bench is None or stock is None or bench.empty or stock.empty:
            return {}

        # Synchronize dates by intersecting timezone-neutral dates
        s_close = stock['Close'].dropna()
        b_close = bench['Close'].dropna()

        s_idx = s_close.index.tz_localize(None) if hasattr(s_close.index, 'tz') and s_close.index.tz else s_close.index
        b_idx = b_close.index.tz_localize(None) if hasattr(b_close.index, 'tz') and b_close.index.tz else b_close.index

        s_series = pd.Series(s_close.values, index=s_idx)
        b_series = pd.Series(b_close.values, index=b_idx)

        common_dates = s_series.index.intersection(b_series.index)
        if len(common_dates) >= 5:
            s_start = _scalar(s_series.loc[common_dates[0]])
            s_end   = _scalar(s_series.loc[common_dates[-1]])
            b_start = _scalar(b_series.loc[common_dates[0]])
            b_end   = _scalar(b_series.loc[common_dates[-1]])
        else:
            s_start = _scalar(s_series.iloc[0])
            s_end   = _scalar(s_series.iloc[-1])
            b_start = _scalar(b_series.iloc[0])
            b_end   = _scalar(b_series.iloc[-1])

        if b_start <= 0 or s_start <= 0:
            return {}

        b_ret = (b_end / b_start - 1) * 100
        s_ret = (s_end / s_start - 1) * 100
        return {
            'stockReturn':     round(s_ret, 2),
            'benchmarkReturn': round(b_ret, 2),
            'niftyReturn':     round(b_ret, 2),
            'benchmarkName':   bench_name,
            'outperformance':  round(s_ret - b_ret, 2),
        }
    except Exception:
        return {}


def analyze_ticker(ticker, start_date=None, end_date=None):
    MIN_CALENDAR_DAYS = 400

    if not end_date:
        end_dt   = datetime.today()
        end_date = end_dt.strftime('%Y-%m-%d')
    else:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    if not start_date:
        start_dt   = end_dt - timedelta(days=365)
        start_date = start_dt.strftime('%Y-%m-%d')
        user_start = None
    else:
        user_start = start_date
        start_dt   = datetime.strptime(start_date, '%Y-%m-%d')

    # Fetch 5y to have enough history for 5-year charts and technical indicators
    raw = get_history(ticker, period='5y', interval='1d')
    if raw.empty:
        return {"error": f"No data found for ticker {ticker}."}

    sd = raw.copy()
    if isinstance(sd.columns, pd.MultiIndex):
        sd.columns = [col[0] for col in sd.columns]
    sd = sd.loc[:, ~sd.columns.duplicated()]

    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    missing  = required - set(sd.columns)
    if missing:
        return {"error": f"Missing required price fields: {missing}"}

    for col in required:
        sd[col] = pd.to_numeric(sd[col], errors='coerce')
    sd.dropna(subset=['Close', 'Volume'], inplace=True)

    if len(sd) < 20:
        return {"error": f"Insufficient data for {ticker} — only {len(sd)} trading days found."}

    sd['20 Day MA']  = sd['Close'].rolling(20).mean()
    sd['50 Day MA']  = sd['Close'].rolling(50).mean()
    sd['100 Day MA'] = sd['Close'].rolling(100).mean()
    sd['200 Day MA'] = sd['Close'].rolling(200).mean()
    sd['20 Day STD'] = sd['Close'].rolling(20).std()
    sd['Upper Band'] = sd['20 Day MA'] + 2 * sd['20 Day STD']
    sd['Lower Band'] = sd['20 Day MA'] - 2 * sd['20 Day STD']
    sd['RSI']        = calculate_rsi(sd['Close'])
    sd['12 EMA']     = sd['Close'].ewm(span=12, adjust=False).mean()
    sd['26 EMA']     = sd['Close'].ewm(span=26, adjust=False).mean()
    sd['MACD']       = sd['12 EMA'] - sd['26 EMA']
    sd['Signal Line']= sd['MACD'].ewm(span=9, adjust=False).mean()
    sd['MACD Hist']  = sd['MACD'] - sd['Signal Line']
    sd['ATR']        = calculate_atr(sd)
    sd               = calculate_adx(sd)
    sd['OBV']        = (np.sign(sd['Close'].diff()) * sd['Volume']).fillna(0).cumsum()
    sd['Volume MA20']= sd['Volume'].rolling(20).mean()

    valid = sd.dropna(subset=['RSI', 'MACD', 'Signal Line', 'ATR', 'ADX'])
    if valid.empty:
        return {"error": "Indicators could not be computed (try a larger date range)."}

    latest = valid.iloc[-1]
    prev   = valid.iloc[-2] if len(valid) > 1 else None

    latest_rsi   = _scalar(latest['RSI'])
    latest_macd  = _scalar(latest['MACD'])
    latest_sig   = _scalar(latest['Signal Line'])
    latest_close = _scalar(latest['Close'])
    latest_adx   = _scalar(latest['ADX'])
    latest_atr   = _scalar(latest['ATR'])
    latest_upper = _scalar(latest['Upper Band']) if not pd.isna(latest['Upper Band']) else latest_close * 1.02
    latest_lower = _scalar(latest['Lower Band']) if not pd.isna(latest['Lower Band']) else latest_close * 0.98

    prev_macd = _scalar(prev['MACD'])       if prev is not None else None
    prev_sig  = _scalar(prev['Signal Line']) if prev is not None else None

    latest_ma50  = _scalar(latest['50 Day MA']) if '50 Day MA' in latest and not pd.isna(latest['50 Day MA']) else None
    latest_ma200 = _scalar(latest['200 Day MA']) if '200 Day MA' in latest and not pd.isna(latest['200 Day MA']) else None

    signal, signal_score, signal_reasons = generate_signal(
        latest_rsi, latest_macd, latest_sig,
        latest_close, latest_upper, latest_lower,
        latest_adx, prev_macd, prev_sig,
        ma50=latest_ma50, ma200=latest_ma200
    )

    support, resistance = calculate_support_resistance(sd)
    fib1, fib2, fib3    = calculate_fibonacci_levels(sd)
    fundamentals        = fetch_fundamentals(ticker)
    risk                = calculate_risk_metrics(sd['Close'])
    rs_data             = calculate_relative_strength(ticker, start_date, end_date)

    avg_sent, avg_subj, headlines = fetch_news_sentiment(ticker)

    # Trim chart data to the user's requested range
    chart_df = sd.copy().dropna(subset=['Close'])
    if user_start:
        chart_df = chart_df[chart_df.index >= pd.Timestamp(user_start, tz='Asia/Kolkata')]

    timeseries_data = []
    for idx, row in chart_df.iterrows():
        timeseries_data.append({
            'date':       idx.strftime('%Y-%m-%d'),
            'open':       round(_scalar(row['Open']),  2),
            'high':       round(_scalar(row['High']),  2),
            'low':        round(_scalar(row['Low']),   2),
            'close':      round(_scalar(row['Close']), 2),
            'volume':     int(_scalar(row['Volume'])),
            'volumeMa20': round(_scalar(row['Volume MA20']), 0) if not pd.isna(row['Volume MA20']) else None,
            'ma20':       round(_scalar(row['20 Day MA']),   2) if not pd.isna(row['20 Day MA'])   else None,
            'ma50':       round(_scalar(row['50 Day MA']),   2) if not pd.isna(row['50 Day MA'])   else None,
            'ma100':      round(_scalar(row['100 Day MA']),  2) if not pd.isna(row['100 Day MA'])  else None,
            'ma200':      round(_scalar(row['200 Day MA']),  2) if not pd.isna(row['200 Day MA'])  else None,
            'upperBand':  round(_scalar(row['Upper Band']),  2) if not pd.isna(row['Upper Band'])  else None,
            'lowerBand':  round(_scalar(row['Lower Band']),  2) if not pd.isna(row['Lower Band'])  else None,
            'rsi':        round(_scalar(row['RSI']),         2) if not pd.isna(row['RSI'])         else None,
            'macd':       round(_scalar(row['MACD']),        2) if not pd.isna(row['MACD'])        else None,
            'macdSignal': round(_scalar(row['Signal Line']), 2) if not pd.isna(row['Signal Line']) else None,
            'macdHist':   round(_scalar(row['MACD Hist']),   2) if not pd.isna(row['MACD Hist'])   else None,
            'adx':        round(_scalar(row['ADX']),         2) if not pd.isna(row['ADX'])         else None,
            'atr':        round(_scalar(row['ATR']),         2) if not pd.isna(row['ATR'])         else None,
            'obv':        int(_scalar(row['OBV'])),
        })

    is_ticker_us = not ticker.endswith('.NS') and not ticker.endswith('.BO') and not ticker.startswith('^')
    curr_sym     = '$' if is_ticker_us else '₹'

    return {
        'ticker':          ticker,
        'currency_symbol': curr_sym,
        'summary': {
            'close':           round(latest_close, 2),
            'signal':          signal,
            'signalScore':     signal_score,
            'signalReasons':   signal_reasons,
            'support':         support,
            'resistance':      resistance,
            'fib236':          round(fib1, 2),
            'fib382':          round(fib2, 2),
            'fib618':          round(fib3, 2),
            'rsi':             round(latest_rsi,  2),
            'macd':            round(latest_macd, 4),
            'macdSignal':      round(latest_sig,  4),
            'adx':             round(latest_adx,  2),
            'atr':             round(latest_atr,  2),
            'ma50':            round(latest_ma50, 2) if latest_ma50 is not None else None,
            'ma200':           round(latest_ma200, 2) if latest_ma200 is not None else None,
            'currency_symbol': curr_sym,
        },
        'fundamentals':     fundamentals,
        'risk':             risk,
        'relativeStrength': rs_data,
        'sentiment': {
            'score':       round(avg_sent,  3),
            'subjectivity':round(avg_subj,  3),
            'label':       'Positive' if avg_sent > 0.05 else 'Negative' if avg_sent < -0.05 else 'Neutral',
            'headlines':   headlines,
        },
        'chartData': timeseries_data,
    }
