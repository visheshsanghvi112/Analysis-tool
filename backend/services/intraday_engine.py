"""
StockIQ Pro — Canonical Intraday Engine Service.

Reusable, single-source-of-truth calculations for intraday market microstructure:
- Session-anchored VWAP and Multi-Sigma Volatility Bands
- Supertrend (period=10, multiplier=3.0)
- Opening Range Breakout (ORB)
- Welles Wilder ATR & RSI
- Candle synchronization with live quotes
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from yf_client import get_history, get_quote


def _safe_float(val: Any, default: float = 0.0, decimals: int = 2) -> float:
    """Safely converts numpy/pandas values to python float with rounding."""
    if val is None or pd.isna(val) or np.isinf(val):
        return default
    try:
        return round(float(val), decimals)
    except Exception:
        return default


def sync_candle_with_quote(df: pd.DataFrame, live_quote_price: Optional[float]) -> pd.DataFrame:
    """
    Synchronizes the latest candle with the live quote price before
    computing indicators so that VWAP, Supertrend, and moving averages
    incorporate the real-time quote.
    """
    if df.empty or live_quote_price is None or live_quote_price <= 0:
        return df

    df_synced = df.copy()
    last_idx = df_synced.index[-1]
    df_synced.loc[last_idx, "Close"] = live_quote_price
    if live_quote_price > df_synced.loc[last_idx, "High"]:
        df_synced.loc[last_idx, "High"] = live_quote_price
    if live_quote_price < df_synced.loc[last_idx, "Low"]:
        df_synced.loc[last_idx, "Low"] = live_quote_price
    return df_synced


def calculate_session_vwap_and_bands(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Computes session-anchored Volume-Weighted Average Price (VWAP)
    and Standard Deviation Volatility Bands (±1σ, ±2σ, ±3σ).
    Resets at each new calendar day/trading session boundary.
    """
    if df.empty:
        empty_arr = np.array([])
        return {
            "vwap": empty_arr,
            "upper_1": empty_arr, "lower_1": empty_arr,
            "upper_2": empty_arr, "lower_2": empty_arr,
            "upper_3": empty_arr, "lower_3": empty_arr,
        }

    tp = ((df["High"] + df["Low"] + df["Close"]) / 3.0).values
    vol = df["Volume"].fillna(0).values

    # Identify session boundaries by calendar date
    dates = pd.to_datetime(df.index).date
    n = len(df)
    vwap = np.zeros(n)
    vwap_std = np.zeros(n)

    unique_dates, split_indices = np.unique(dates, return_index=True)
    session_starts = list(split_indices) + [n]

    for s_idx in range(len(session_starts) - 1):
        start = session_starts[s_idx]
        end = session_starts[s_idx + 1]

        s_tp = tp[start:end]
        s_vol = vol[start:end]

        cum_vol = np.cumsum(s_vol)
        cum_vp = np.cumsum(s_tp * s_vol)

        s_vwap = np.where(cum_vol > 0, cum_vp / np.maximum(cum_vol, 1e-9), s_tp)

        # Canonical Volume-Weighted Variance: Var(X) = E_w[X^2] - (E_w[X])^2
        cum_vp2 = np.cumsum(s_vol * (s_tp ** 2))
        s_mean_sq = np.where(cum_vol > 0, cum_vp2 / np.maximum(cum_vol, 1e-9), s_tp ** 2)
        s_var = np.maximum(0.0, s_mean_sq - (s_vwap ** 2))
        s_std = np.sqrt(s_var)

        vwap[start:end] = s_vwap
        vwap_std[start:end] = s_std

    return {
        "vwap": vwap,
        "upper_1": vwap + vwap_std,
        "lower_1": vwap - vwap_std,
        "upper_2": vwap + 2 * vwap_std,
        "lower_2": vwap - 2 * vwap_std,
        "upper_3": vwap + 3 * vwap_std,
        "lower_3": vwap - 3 * vwap_std,
    }


def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, np.ndarray]:
    """
    Computes institutional Supertrend indicator with Welles Wilder ATR trailing stop series.
    Direction: 1 for BULLISH, -1 for BEARISH.
    """
    n = len(df)
    if n == 0:
        return {"supertrend": np.array([]), "direction": np.array([]), "atr": np.array([])}

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    # Canonical Welles Wilder ATR smoothing
    atr = calculate_atr(df, period=period)

    hl2 = (h + l) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)

    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    supertrend[0] = final_lb[0]

    for i in range(1, n):
        if basic_ub[i] < final_ub[i - 1] or c[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or c[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        if direction[i - 1] == 1:
            if c[i] < final_lb[i]:
                direction[i] = -1
                supertrend[i] = final_ub[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lb[i]
        else:
            if c[i] > final_ub[i]:
                direction[i] = 1
                supertrend[i] = final_lb[i]
            else:
                direction[i] = -1
                supertrend[i] = final_ub[i]

    return {"supertrend": supertrend, "direction": direction, "atr": atr}


def calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """
    Computes Welles Wilder Average True Range (ATR) across candlestick bars.
    Uses exact Welles Wilder specification:
    - First ATR value at index (period - 1) is the SMA of the first 'period' True Ranges.
    - Subsequent ATR values use Wilder's exponential smoothing recurrence:
      ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period
    """
    n = len(df)
    if n < 2:
        return np.zeros(n)
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    tr1 = h - l
    tr2 = np.abs(h - np.roll(c, 1))
    tr3 = np.abs(l - np.roll(c, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = np.zeros(n)
    if n < period:
        atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values
        return np.nan_to_num(atr, nan=0.0)

    # 1. First ATR is SMA of first 'period' TR values
    atr[period - 1] = float(np.mean(tr[:period]))
    # 2. Subsequent ATRs follow Wilder's smoothing recurrence
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    # For bars before 'period', fill with expanding mean for continuity
    for i in range(period - 1):
        atr[i] = float(np.mean(tr[:i + 1]))
    return np.nan_to_num(atr, nan=0.0)


def calculate_rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    """Computes Welles Wilder Relative Strength Index (RSI)."""
    if len(series) < 2:
        return np.full(len(series), 50.0)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi.values, nan=50.0)


def calculate_orb(df: pd.DataFrame, interval: str = "5m") -> Dict[str, Any]:
    """
    Computes Opening Range Breakout (ORB) boundaries anchored to the latest trading session.
    Returns None for status if insufficient session data is available to establish ORB.
    """
    if df.empty:
        return {"high_15m": None, "low_15m": None, "status": None, "high_30m": None, "low_30m": None}

    # Extract the latest session's data
    dates = pd.to_datetime(df.index).date
    latest_date = dates[-1]
    today_df = df[dates == latest_date]
    if today_df.empty:
        today_df = df

    candle_minutes = 5
    if "1m" in interval:
        candle_minutes = 1
    elif "3m" in interval:
        candle_minutes = 3
    elif "5m" in interval:
        candle_minutes = 5
    elif "15m" in interval:
        candle_minutes = 15
    elif "30m" in interval:
        candle_minutes = 30
    elif "1h" in interval:
        candle_minutes = 60

    count_15m = max(1, int(15 / candle_minutes))
    count_30m = max(1, int(30 / candle_minutes))

    if len(today_df) < count_15m:
        return {
            "high_15m": None,
            "low_15m": None,
            "high_30m": None,
            "low_30m": None,
            "status": None,  # Insufficient data, do NOT default to INSIDE_RANGE
            "pct_from_15m_high": None,
            "pct_from_15m_low": None,
        }

    orb_15m_slice = today_df.iloc[:count_15m]
    orb_30m_slice = today_df.iloc[:min(len(today_df), count_30m)]

    high_15m = float(orb_15m_slice["High"].max())
    low_15m = float(orb_15m_slice["Low"].min())
    high_30m = float(orb_30m_slice["High"].max())
    low_30m = float(orb_30m_slice["Low"].min())

    curr_close = float(today_df["Close"].iloc[-1])

    if curr_close > high_15m:
        status = "BULLISH_BREAKOUT"
    elif curr_close < low_15m:
        status = "BEARISH_BREAKDOWN"
    else:
        status = "INSIDE_RANGE"

    return {
        "high_15m": _safe_float(high_15m),
        "low_15m": _safe_float(low_15m),
        "high_30m": _safe_float(high_30m),
        "low_30m": _safe_float(low_30m),
        "status": status,
        "pct_from_15m_high": _safe_float(((curr_close - high_15m) / high_15m) * 100) if high_15m > 0 else 0.0,
        "pct_from_15m_low": _safe_float(((curr_close - low_15m) / low_15m) * 100) if low_15m > 0 else 0.0,
    }


def calculate_intraday_snapshot(
    ticker: str,
    interval: str = "5m",
    df: Optional[pd.DataFrame] = None,
    live_quote: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assembles a complete canonical intraday snapshot for a ticker using 5m high-resolution bars.
    """
    clean_ticker = ticker.strip().upper()

    if df is None or df.empty:
        # Fetch 5 days of 5m candles to ensure full session context & 200-period EMA
        df = get_history(clean_ticker, period="5d", interval=interval)

    if df is None or df.empty or len(df) < 5:
        return {"available": False, "reason": "Insufficient intraday candle data"}

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if live_quote is None:
        live_quote = get_quote(clean_ticker) or {}

    quote_price = _safe_float(live_quote.get("price"))
    if quote_price <= 0:
        quote_price = _safe_float(df["Close"].iloc[-1])

    # Synchronize candle with live quote
    df_synced = sync_candle_with_quote(df, quote_price)
    c_series = df_synced["Close"]
    volume = df_synced["Volume"]

    # Canonical Session VWAP
    vwap_dict = calculate_session_vwap_and_bands(df_synced)
    curr_vwap = _safe_float(vwap_dict["vwap"][-1]) if len(vwap_dict["vwap"]) > 0 else None

    # Canonical Supertrend (10, 3)
    st_dict = calculate_supertrend(df_synced, period=10, multiplier=3.0)
    st_dir_num = int(st_dict["direction"][-1]) if len(st_dict["direction"]) > 0 else 1
    supertrend_direction = "BULLISH" if st_dir_num == 1 else "BEARISH"
    curr_st = _safe_float(st_dict["supertrend"][-1]) if len(st_dict["supertrend"]) > 0 else None

    # EMAs
    ema9 = _safe_float(c_series.ewm(span=9, adjust=False).mean().iloc[-1])
    ema20 = _safe_float(c_series.ewm(span=20, adjust=False).mean().iloc[-1])
    ema21 = _safe_float(c_series.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = _safe_float(c_series.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = _safe_float(
        c_series.ewm(span=200, adjust=False).mean().iloc[-1]
        if len(c_series) >= 150
        else c_series.ewm(span=len(c_series), adjust=False).mean().iloc[-1]
    )

    # RSI (14)
    rsi_vals = calculate_rsi(c_series, period=14)
    curr_rsi = _safe_float(rsi_vals[-1]) if len(rsi_vals) > 0 else 50.0

    # ATR (14)
    atr_vals = calculate_atr(df_synced, period=14)
    curr_atr = _safe_float(atr_vals[-1]) if len(atr_vals) > 0 else None
    atr_pct = _safe_float((curr_atr / quote_price * 100.0) if (curr_atr and quote_price and quote_price > 0) else None)

    # RVOL (Volume vs 20-period Volume MA)
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    curr_rvol = _safe_float(volume.iloc[-1] / vol_ma20 if (vol_ma20 and vol_ma20 > 0) else 1.0)

    # Canonical ORB
    orb_dict = calculate_orb(df_synced, interval=interval)

    # Delta absorption check
    delta_absorption = False
    if curr_rvol and curr_rvol > 1.5 and curr_rsi and curr_rsi <= 30:
        delta_absorption = True

    # Timestamps & Age
    last_bar_dt = df_synced.index[-1]
    now_utc = pd.Timestamp.now(tz="UTC")
    bar_dt_utc = last_bar_dt if last_bar_dt.tzinfo else last_bar_dt.tz_localize("UTC")
    bar_age_sec = (now_utc - bar_dt_utc).total_seconds() if hasattr(now_utc, "__sub__") else 0.0

    return {
        "available": True,
        "interval": interval,
        "price": quote_price,
        "session_vwap": curr_vwap,
        "supertrend": curr_st,
        "supertrend_direction": supertrend_direction,
        "orb_status": orb_dict.get("status"),
        "orb_high_15m": orb_dict.get("high_15m"),
        "orb_low_15m": orb_dict.get("low_15m"),
        "ema9": ema9,
        "ema20": ema20,
        "ema21": ema21,
        "ema50": ema50,
        "ema200": ema200,
        "rsi14": curr_rsi,
        "atr": curr_atr,
        "atr_pct": atr_pct,
        "rvol": curr_rvol,
        "delta_absorption": delta_absorption,
        "bar_as_of": str(last_bar_dt),
        "as_of": str(last_bar_dt),
        "bar_age_seconds": max(0.0, bar_age_sec),
    }
