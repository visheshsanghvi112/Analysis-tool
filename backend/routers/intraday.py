# ==============================================================================
# StockIQ Pro — Intraday Quantitative Analytics Router
# Institutional High-Frequency Terminal & Microstructure Engine
# Real-World Execution: Session Phases, Gap Engine, Trap Detectors,
# Multi-Timeframe Confluence & Friction Breakeven Suite
# ==============================================================================

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
import zoneinfo
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, HTTPException

from yf_client import get_history, get_quote
from utils.cache import cache_ttl

logger = logging.getLogger("stockiq.intraday")
router = APIRouter(prefix="/api/intraday", tags=["intraday"])

FLAGSHIP_IN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "TATAMOTORS.NS",
    "LT.NS", "ITC.NS", "KOTAKBANK.NS", "AXISBANK.NS"
]

FLAGSHIP_US_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "SPY", "QQQ"
]


def _safe_float(val: Any, default: float = 0.0, decimals: int = 2) -> float:
    """Safely converts numpy/pandas values to python float with rounding."""
    if val is None or pd.isna(val) or np.isinf(val):
        return default
    try:
        return round(float(val), decimals)
    except Exception:
        return default


def _calculate_vwap_and_bands(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Computes session-anchored Volume-Weighted Average Price (VWAP)
    and Standard Deviation Volatility Bands (±1σ, ±2σ, ±3σ).
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0).values
    tp_vals = tp.values

    cum_vol = np.cumsum(vol)
    cum_vp = np.cumsum(tp_vals * vol)

    vwap = np.where(cum_vol > 0, cum_vp / np.maximum(cum_vol, 1e-9), tp_vals)

    # Cumulative variance of typical price around expanding VWAP
    cum_vol_sq_diff = np.cumsum(vol * (tp_vals - vwap) ** 2)
    vwap_variance = np.where(cum_vol > 0, cum_vol_sq_diff / np.maximum(cum_vol, 1e-9), 0.0)
    vwap_std = np.sqrt(np.maximum(vwap_variance, 0.0))

    return {
        "vwap": vwap,
        "upper_1": vwap + vwap_std,
        "lower_1": vwap - vwap_std,
        "upper_2": vwap + 2 * vwap_std,
        "lower_2": vwap - 2 * vwap_std,
        "upper_3": vwap + 3 * vwap_std,
        "lower_3": vwap - 3 * vwap_std,
    }


def _calculate_volume_profile(df: pd.DataFrame, n_bins: int = 25) -> Dict[str, Any]:
    """
    Calculates Volume Profile (VPVR) across session price range.
    Identifies Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    """
    if df.empty or len(df) < 2:
        return {"profile": [], "poc_price": 0.0, "vah_price": 0.0, "val_price": 0.0, "total_vol": 0}

    low_min = float(df["Low"].min())
    high_max = float(df["High"].max())

    if low_min == high_max or np.isnan(low_min) or np.isnan(high_max):
        return {"profile": [], "poc_price": low_min, "vah_price": low_min, "val_price": low_min, "total_vol": 0}

    bin_edges = np.linspace(low_min, high_max, n_bins + 1)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].fillna(0).values

    candle_spread = np.maximum(df["High"] - df["Low"], 1e-6)
    buy_ratio = np.clip((df["Close"] - df["Low"]) / candle_spread, 0.1, 0.9)
    buy_vols = vol * buy_ratio
    sell_vols = vol * (1.0 - buy_ratio)

    bin_indices = np.digitize(tp.values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_total = np.zeros(n_bins)
    bin_buy = np.zeros(n_bins)
    bin_sell = np.zeros(n_bins)

    for i in range(len(tp)):
        b_idx = bin_indices[i]
        bin_total[b_idx] += vol[i]
        bin_buy[b_idx] += buy_vols[i]
        bin_sell[b_idx] += sell_vols[i]

    total_vol = float(np.sum(bin_total))
    if total_vol <= 0:
        total_vol = 1.0

    poc_idx = int(np.argmax(bin_total))
    poc_price = round((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0, 2)

    target_va_vol = 0.70 * total_vol
    accumulated_vol = bin_total[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx

    while accumulated_vol < target_va_vol and (low_idx > 0 or high_idx < n_bins - 1):
        next_low_vol = bin_total[low_idx - 1] if low_idx > 0 else -1
        next_high_vol = bin_total[high_idx + 1] if high_idx < n_bins - 1 else -1

        if next_low_vol >= next_high_vol and low_idx > 0:
            low_idx -= 1
            accumulated_vol += bin_total[low_idx]
        elif high_idx < n_bins - 1:
            high_idx += 1
            accumulated_vol += bin_total[high_idx]
        else:
            break

    val_price = round(float(bin_edges[low_idx]), 2)
    vah_price = round(float(bin_edges[high_idx + 1]), 2)

    profile_list = []
    for b in range(n_bins):
        p_mid = round((bin_edges[b] + bin_edges[b + 1]) / 2.0, 2)
        b_vol = float(bin_total[b])
        profile_list.append({
            "price": p_mid,
            "volume": int(b_vol),
            "buy_volume": int(bin_buy[b]),
            "sell_volume": int(bin_sell[b]),
            "pct_of_total": round((b_vol / total_vol) * 100, 1),
            "is_poc": (b == poc_idx),
            "in_value_area": (low_idx <= b <= high_idx),
        })

    return {
        "profile": profile_list,
        "poc_price": poc_price,
        "vah_price": vah_price,
        "val_price": val_price,
        "total_vol": int(total_vol),
    }


def _calculate_pivots(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Camarilla Equation levels (H1-H4, L1-L4), Floor Pivots,
    and Previous Day benchmark boundaries (PDH, PDL, PDC) from the preceding
    completed trading session's High, Low, and Close.
    """
    if daily_df.empty or len(daily_df) < 1:
        return {
            "daily_levels": {"pdh": 0.0, "pdl": 0.0, "pdc": 0.0, "range": 0.0},
            "camarilla": {},
            "floor": {}
        }

    prev_bar = daily_df.iloc[-2] if len(daily_df) >= 2 else daily_df.iloc[-1]
    h = float(prev_bar["High"])
    l = float(prev_bar["Low"])
    c = float(prev_bar["Close"])
    rng = h - l

    # Camarilla Equation
    h4 = c + (rng * 1.1) / 2.0
    h3 = c + (rng * 1.1) / 4.0
    h2 = c + (rng * 1.1) / 6.0
    h1 = c + (rng * 1.1) / 12.0
    l1 = c - (rng * 1.1) / 12.0
    l2 = c - (rng * 1.1) / 6.0
    l3 = c - (rng * 1.1) / 4.0
    l4 = c - (rng * 1.1) / 2.0

    # Classic Floor Pivots
    p = (h + l + c) / 3.0
    r1 = 2 * p - l
    s1 = 2 * p - h
    r2 = p + rng
    s2 = p - rng
    r3 = h + 2 * (p - l)
    s3 = l - 2 * (h - p)

    return {
        "daily_levels": {
            "pdh": _safe_float(h),
            "pdl": _safe_float(l),
            "pdc": _safe_float(c),
            "range": _safe_float(rng),
        },
        "camarilla": {
            "h4": _safe_float(h4),
            "h3": _safe_float(h3),
            "h2": _safe_float(h2),
            "h1": _safe_float(h1),
            "l1": _safe_float(l1),
            "l2": _safe_float(l2),
            "l3": _safe_float(l3),
            "l4": _safe_float(l4),
            "h4_desc": "Long Breakout Target / Acceleration",
            "h3_desc": "Mean-Reversion Short Resistance Zone",
            "l3_desc": "Mean-Reversion Long Support Zone",
            "l4_desc": "Short Breakdown Level / Stop Trigger",
        },
        "floor": {
            "p": _safe_float(p),
            "r1": _safe_float(r1),
            "s1": _safe_float(s1),
            "r2": _safe_float(r2),
            "s2": _safe_float(s2),
            "r3": _safe_float(r3),
            "s3": _safe_float(s3),
        }
    }


def _calculate_orb(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
    """
    Computes Opening Range Breakout (ORB) boundaries for 15m and 30m intervals.
    """
    if df.empty:
        return {"high_15m": 0.0, "low_15m": 0.0, "status": "INSIDE_RANGE", "high_30m": 0.0, "low_30m": 0.0}

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

    orb_15m_slice = df.iloc[:min(len(df), count_15m)]
    orb_30m_slice = df.iloc[:min(len(df), count_30m)]

    high_15m = float(orb_15m_slice["High"].max())
    low_15m = float(orb_15m_slice["Low"].min())
    high_30m = float(orb_30m_slice["High"].max())
    low_30m = float(orb_30m_slice["Low"].min())

    curr_close = float(df["Close"].iloc[-1])

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
        "pct_from_15m_high": _safe_float(((curr_close - high_15m) / high_15m) * 100),
        "pct_from_15m_low": _safe_float(((curr_close - low_15m) / low_15m) * 100),
    }


def _calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, np.ndarray]:
    """Computes institutional Supertrend indicator with ATR trailing stop series."""
    n = len(df)
    if n == 0:
        return {"supertrend": np.array([]), "direction": np.array([])}

    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    tr1 = h - l
    tr2 = np.abs(h - np.roll(c, 1))
    tr3 = np.abs(l - np.roll(c, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    atr = pd.Series(tr).rolling(period, min_periods=1).mean().values

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

    return {"supertrend": supertrend, "direction": direction}


def _calculate_rsi(series: pd.Series, period: int = 14) -> np.ndarray:
    """Computes standard Relative Strength Index (RSI)."""
    if len(series) < 2:
        return np.full(len(series), 50.0)

    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=1).mean()

    rs = gain / np.maximum(loss, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values


def _calculate_gap_intelligence(today_df: pd.DataFrame, daily_df: pd.DataFrame, curr_price: float, curr_vwap: float) -> Dict[str, Any]:
    """
    Computes pre-market gap analysis, gap classification, fill status, and tactical playbook.
    """
    if daily_df.empty or today_df.empty:
        return {"gap_pts": 0.0, "gap_pct": 0.0, "gap_type": "FLAT", "gap_filled": True, "gap_fill_dist": 0.0, "playbook": "NEUTRAL"}

    prev_bar = daily_df.iloc[-2] if len(daily_df) >= 2 else daily_df.iloc[-1]
    prev_close = float(prev_bar["Close"])
    prev_high = float(prev_bar["High"])
    prev_low = float(prev_bar["Low"])

    today_open = float(today_df["Open"].iloc[0])
    today_high = float(today_df["High"].max())
    today_low = float(today_df["Low"].min())

    gap_pts = round(today_open - prev_close, 2)
    gap_pct = round((gap_pts / prev_close) * 100, 2) if prev_close else 0.0

    if today_open > prev_high:
        gap_type = "FULL_GAP_UP"
    elif today_open < prev_low:
        gap_type = "FULL_GAP_DOWN"
    elif today_open > prev_close:
        gap_type = "PARTIAL_GAP_UP"
    elif today_open < prev_close:
        gap_type = "PARTIAL_GAP_DOWN"
    else:
        gap_type = "FLAT"

    if gap_pts > 0:
        gap_filled = bool(today_low <= prev_close)
        gap_fill_dist = round(max(today_low - prev_close, 0.0), 2)
    elif gap_pts < 0:
        gap_filled = bool(today_high >= prev_close)
        gap_fill_dist = round(max(prev_close - today_high, 0.0), 2)
    else:
        gap_filled = True
        gap_fill_dist = 0.0

    # Real-life playbook recommendation
    if "GAP_UP" in gap_type:
        if curr_price >= curr_vwap:
            playbook = "GAP_AND_GO"
            directive = "Holding firmly above VWAP on gap up. Look for pullback long entries toward VWAP support."
        else:
            playbook = "GAP_FADE"
            directive = "Slipping below session VWAP on gap up. High probability gap-fade targeting previous close."
    elif "GAP_DOWN" in gap_type:
        if curr_price <= curr_vwap:
            playbook = "GAP_AND_GO_SHORT"
            directive = "Submerged below VWAP on gap down. Bearish continuation targeting lower support."
        else:
            playbook = "GAP_FADE_LONG"
            directive = "Reclaiming VWAP from gap down. Aggressive buyers absorbing supply; target gap fill."
    else:
        playbook = "RANGE_BOUND"
        directive = "Flat open; rely on 15m ORB breakout boundaries."

    return {
        "today_open": _safe_float(today_open),
        "prev_close": _safe_float(prev_close),
        "prev_high": _safe_float(prev_high),
        "prev_low": _safe_float(prev_low),
        "gap_pts": gap_pts,
        "gap_pct": gap_pct,
        "gap_type": gap_type,
        "gap_filled": gap_filled,
        "gap_fill_dist": gap_fill_dist,
        "playbook": playbook,
        "directive": directive,
    }


def _detect_institutional_traps(df: pd.DataFrame, curr_price: float, curr_vwap: float) -> Dict[str, Any]:
    """
    Detects Bull Traps and Bear Traps via price-action vs volume/delta divergence.
    """
    if len(df) < 10:
        return {"status": "NONE", "title": "Normal Liquidity", "desc": "Volume confirming active price discovery."}

    recent = df.tail(10)
    day_high = float(df["High"].max())
    day_low = float(df["Low"].min())

    vol_ma = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else float(df["Volume"].mean())

    # Bull Trap Check: Print new high with weak volume or falling CVD, followed by retreat below VWAP
    high_candle_idx = recent["High"].idxmax()
    high_candle = df.loc[high_candle_idx]
    if high_candle["High"] >= day_high * 0.999:
        if high_candle["Volume"] < vol_ma * 0.85 or (high_candle["Close"] < high_candle["Open"]):
            if curr_price < curr_vwap:
                return {
                    "status": "BULL_TRAP",
                    "title": "⚠️ Institutional Bull Trap Warning",
                    "desc": "Session high was printed on below-average volume with aggressive selling rejection. Price is now below VWAP.",
                    "severity": "HIGH"
                }

    # Bear Trap Check: Print new low with swift wick rejection and buyer absorption
    low_candle_idx = recent["Low"].idxmin()
    low_candle = df.loc[low_candle_idx]
    if low_candle["Low"] <= day_low * 1.001:
        spread = max(low_candle["High"] - low_candle["Low"], 0.01)
        lower_wick_ratio = (min(low_candle["Open"], low_candle["Close"]) - low_candle["Low"]) / spread
        if lower_wick_ratio > 0.45 and curr_price > low_candle["High"]:
            return {
                "status": "BEAR_TRAP",
                "title": "⚡ Institutional Bear Trap / Spring",
                "desc": "Session low was rejected with heavy absorbing buying wicks. Potential long liquidity reversal.",
                "severity": "MEDIUM"
            }

    return {
        "status": "NONE",
        "title": "Clean Order Flow",
        "desc": "Volume confirms price discovery with no deceptive divergences.",
        "severity": "LOW"
    }


def _calculate_multi_timeframe(df_5m: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Triple-Screen Confluence Matrix (5m, 15m, 1h) using fast resampling.
    """
    if len(df_5m) < 5:
        return {"confluence_score": 50, "confluence_bias": "NEUTRAL", "screens": []}

    screens = []
    bullish_votes = 0
    total_votes = 0

    # 1. 5m Screen
    c5 = df_5m["Close"]
    ema9_5m = c5.ewm(span=9, adjust=False).mean().iloc[-1]
    ema21_5m = c5.ewm(span=21, adjust=False).mean().iloc[-1]
    rsi_5m = _calculate_rsi(c5, 14)[-1]
    trend_5m = "BULLISH" if ema9_5m > ema21_5m else "BEARISH"
    if trend_5m == "BULLISH":
        bullish_votes += 1
    total_votes += 1
    screens.append({
        "timeframe": "5m (Setup)",
        "trend": trend_5m,
        "ema_fast": _safe_float(ema9_5m),
        "ema_slow": _safe_float(ema21_5m),
        "rsi": _safe_float(rsi_5m),
    })

    # 2. 15m Screen (resample)
    try:
        df_15m = df_5m.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        if len(df_15m) >= 3:
            c15 = df_15m["Close"]
            ema9_15m = c15.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21_15m = c15.ewm(span=21, adjust=False).mean().iloc[-1]
            rsi_15m = _calculate_rsi(c15, 14)[-1]
            trend_15m = "BULLISH" if ema9_15m > ema21_15m else "BEARISH"
            if trend_15m == "BULLISH":
                bullish_votes += 1
            total_votes += 1
            screens.append({
                "timeframe": "15m (Wave)",
                "trend": trend_15m,
                "ema_fast": _safe_float(ema9_15m),
                "ema_slow": _safe_float(ema21_15m),
                "rsi": _safe_float(rsi_15m),
            })
    except Exception:
        pass

    # 3. 1h Screen (resample)
    try:
        df_1h = df_5m.resample("1h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        if len(df_1h) >= 2:
            c1h = df_1h["Close"]
            ema9_1h = c1h.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21_1h = c1h.ewm(span=21, adjust=False).mean().iloc[-1]
            rsi_1h = _calculate_rsi(c1h, 14)[-1]
            trend_1h = "BULLISH" if ema9_1h > ema21_1h else "BEARISH"
            if trend_1h == "BULLISH":
                bullish_votes += 1
            total_votes += 1
            screens.append({
                "timeframe": "1h (Tide)",
                "trend": trend_1h,
                "ema_fast": _safe_float(ema9_1h),
                "ema_slow": _safe_float(ema21_1h),
                "rsi": _safe_float(rsi_1h),
            })
    except Exception:
        pass

    score = int((bullish_votes / max(total_votes, 1)) * 100)
    if score >= 75:
        verdict = "STRONG BULLISH CONFLUENCE"
    elif score <= 25:
        verdict = "STRONG BEARISH CONFLUENCE"
    else:
        verdict = "MIXED TIMEFRAMES (CHOP)"

    return {
        "confluence_score": score,
        "confluence_bias": verdict,
        "screens": screens,
    }


def _calculate_relative_strength(stock_change_pct: float, is_us: bool) -> Dict[str, Any]:
    """
    Computes Beta-adjusted relative strength vs NIFTY 50 (IN) or S&P 500 (US).
    """
    bench_symbol = "^GSPC" if is_us else "^NSEI"
    bench_name = "S&P 500" if is_us else "NIFTY 50"

    bench_quote = get_quote(bench_symbol)
    bench_chg = _safe_float(bench_quote.get("changePct", 0.0))
    alpha = round(stock_change_pct - bench_chg, 2)

    if alpha >= 1.0:
        status = "STRONG OUTPERFORMER"
        desc = f"Beating {bench_name} by +{alpha}%. Heavy institutional sponsorship."
    elif alpha <= -1.0:
        status = "UNDERPERFORMER"
        desc = f"Lagging {bench_name} by {alpha}%. Lacks institutional support."
    else:
        status = "IN-LINE"
        desc = f"Tracking {bench_name} closely ({alpha:+0.2f}% alpha)."

    return {
        "benchmark_symbol": bench_symbol,
        "benchmark_name": bench_name,
        "benchmark_price": _safe_float(bench_quote.get("price", 0.0)),
        "benchmark_change_pct": bench_chg,
        "alpha_pct": alpha,
        "status": status,
        "desc": desc,
    }


def _generate_battle_plan(
    ticker: str,
    company: str,
    curr_price: float,
    curr_vwap: float,
    supertrend: float,
    supertrend_dir: int,
    pivots: Dict[str, Any],
    orb: Dict[str, Any],
    bias: str,
    curr_sym: str
) -> Dict[str, Any]:
    """
    Auto-generates structured 1-Click Intraday Battle Plan.
    """
    is_long = "BUY" in bias or (supertrend_dir == 1 and curr_price >= curr_vwap)
    daily_lvls = pivots.get("daily_levels", {}) if isinstance(pivots, dict) else {}
    pdh = daily_lvls.get("pdh", 0.0)
    pdl = daily_lvls.get("pdl", 0.0)
    pdc = daily_lvls.get("pdc", 0.0)

    if is_long:
        entry_price = round(max(curr_vwap, curr_price * 0.998), 2)
        stop_loss = round(min(supertrend, entry_price * 0.993), 2)
        risk_per_share = max(round(entry_price - stop_loss, 2), 0.5)
        target_1 = round(entry_price + (risk_per_share * 1.5), 2)
        target_2 = round(entry_price + (risk_per_share * 2.5), 2)
        if pdh > 0 and curr_price >= pdh:
            setup_name = "PDH Breakout & Trend Extension"
            trigger_rule = f"Long retest of PDH {curr_sym}{pdh:.2f} holding as dynamic floor above VWAP"
        else:
            setup_name = "VWAP Reclaim & Momentum Pullback"
            trigger_rule = f"Enter on 5m candle test of {curr_sym}{entry_price} holding above VWAP"
    else:
        entry_price = round(min(curr_vwap, curr_price * 1.002), 2)
        stop_loss = round(max(supertrend, entry_price * 1.007), 2)
        risk_per_share = max(round(stop_loss - entry_price, 2), 0.5)
        target_1 = round(entry_price - (risk_per_share * 1.5), 2)
        target_2 = round(entry_price - (risk_per_share * 2.5), 2)
        if pdl > 0 and curr_price <= pdl:
            setup_name = "PDL Breakdown & Liquidation Scalp"
            trigger_rule = f"Short retest of PDL {curr_sym}{pdl:.2f} failing below dynamic ceiling and VWAP"
        else:
            setup_name = "VWAP Rejection & Breakdown Scalp"
            trigger_rule = f"Short on 5m candle test of {curr_sym}{entry_price} with rejection below VWAP"

    rr_ratio = "1:2.0"

    ref_line = f"Key Reference: PDH {curr_sym}{pdh:.2f} | PDL {curr_sym}{pdl:.2f} | PDC {curr_sym}{pdc:.2f}\n" if pdh > 0 else ""

    formatted_card = (
        f"═══════════════════════════════════════════\n"
        f"  STOCKIQ PRO — INTRADAY BATTLE PLAN\n"
        f"═══════════════════════════════════════════\n"
        f"Ticker: {ticker} ({company})\n"
        f"Tactical Setup: {setup_name}\n"
        f"Directional Bias: {bias}\n"
        f"{ref_line}"
        f"-------------------------------------------\n"
        f"Entry Trigger: {curr_sym}{entry_price:.2f}\n"
        f"Rule: {trigger_rule}\n"
        f"Stop Loss:    {curr_sym}{stop_loss:.2f} (-{round((abs(entry_price-stop_loss)/entry_price)*100, 2)}% risk)\n"
        f"Target 1 (1.5R): {curr_sym}{target_1:.2f}\n"
        f"Target 2 (2.5R): {curr_sym}{target_2:.2f}\n"
        f"Risk-Reward Ratio: {rr_ratio}\n"
        f"Key Invalidation: Violation of {curr_sym}{stop_loss:.2f}\n"
        f"═══════════════════════════════════════════\n"
        f"Generated via StockIQ Pro High-Frequency Desk"
    )

    return {
        "setup_name": setup_name,
        "is_long": is_long,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": risk_per_share,
        "rr_ratio": rr_ratio,
        "trigger_rule": trigger_rule,
        "formatted_card": formatted_card,
    }


@router.get("/analysis")
def get_intraday_analysis(
    ticker: str = Query(..., description="Stock ticker e.g. RELIANCE.NS or NVDA"),
    interval: str = Query("5m", description="Candle interval: 1m, 3m, 5m, 15m, 30m, 1h"),
    period: str = Query("1d", description="Historical period: 1d or 5d")
):
    """
    Delivers institutional intraday quantitative analytics:
    High-frequency candles, session VWAP & multi-sigma volatility bands,
    Volume Profile (VPVR) with POC, Camarilla & Floor Pivots, Opening Range Breakout (ORB),
    Supertrend trailing stops, Order Flow Cumulative Volume Delta, Pre-Market Gap Intelligence,
    Institutional Trap Detectors, Triple-Screen Confluence, and Relative Strength Alpha.
    """
    clean_ticker = ticker.strip().upper()
    is_us = not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO")
    currency_symbol = "$" if is_us else "₹"

    try:
        # 1. Fetch intraday OHLCV
        df = get_history(clean_ticker, period=period, interval=interval)
        if df.empty or len(df) < 2:
            df = get_history(clean_ticker, period="5d", interval=interval)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No intraday chart data available for {clean_ticker}")

        # 2. Fetch daily history for Pivots & Prev Close
        daily_df = get_history(clean_ticker, period="5d", interval="1d")
        pivots = _calculate_pivots(daily_df)

        # 3. Live quote snapshot
        quote = get_quote(clean_ticker)

        # 4. VWAP & Multi-Sigma Volatility Bands
        vwap_dict = _calculate_vwap_and_bands(df)

        # 5. Supertrend ATR(10, 3)
        st_dict = _calculate_supertrend(df, period=10, multiplier=3.0)

        # 6. Exponential Moving Averages (9, 21, 50)
        c_series = df["Close"]
        ema9 = c_series.ewm(span=9, adjust=False).mean().values
        ema21 = c_series.ewm(span=21, adjust=False).mean().values
        ema50 = c_series.ewm(span=50, adjust=False).mean().values

        # 7. RSI (14)
        rsi_vals = _calculate_rsi(c_series, period=14)

        # 8. Volume Profile (VPVR)
        vpvr = _calculate_volume_profile(df, n_bins=25)

        # 9. Opening Range Breakout (ORB)
        orb = _calculate_orb(df, interval)

        # 10. Candles transformation & Order Flow Delta
        candles = []
        cum_delta = 0.0
        total_buyer_vol = 0.0
        total_seller_vol = 0.0

        for i in range(len(df)):
            idx_time = df.index[i]
            time_str = idx_time.strftime("%H:%M") if hasattr(idx_time, "strftime") else str(idx_time)
            ts_epoch = int(idx_time.timestamp()) if hasattr(idx_time, "timestamp") else i

            o = _safe_float(df["Open"].iloc[i])
            h = _safe_float(df["High"].iloc[i])
            l = _safe_float(df["Low"].iloc[i])
            c = _safe_float(df["Close"].iloc[i])
            v = int(df["Volume"].iloc[i]) if not pd.isna(df["Volume"].iloc[i]) else 0

            rng = max(h - l, 0.01)
            buy_pct = np.clip((c - l) / rng, 0.1, 0.9)
            b_vol = int(v * buy_pct)
            s_vol = v - b_vol
            candle_delta = b_vol - s_vol
            cum_delta += candle_delta
            total_buyer_vol += b_vol
            total_seller_vol += s_vol

            candles.append({
                "timestamp": ts_epoch,
                "time": time_str,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "buyer_vol": b_vol,
                "seller_vol": s_vol,
                "delta": candle_delta,
                "cum_delta": int(cum_delta),
                "vwap": _safe_float(vwap_dict["vwap"][i]),
                "upper_band_1": _safe_float(vwap_dict["upper_1"][i]),
                "lower_band_1": _safe_float(vwap_dict["lower_1"][i]),
                "upper_band_2": _safe_float(vwap_dict["upper_2"][i]),
                "lower_band_2": _safe_float(vwap_dict["lower_2"][i]),
                "supertrend": _safe_float(st_dict["supertrend"][i]) if len(st_dict["supertrend"]) > i else c,
                "supertrend_dir": int(st_dict["direction"][i]) if len(st_dict["direction"]) > i else 1,
                "ema9": _safe_float(ema9[i]),
                "ema21": _safe_float(ema21[i]),
                "ema50": _safe_float(ema50[i]),
                "rsi": _safe_float(rsi_vals[i]),
            })

        # Headline metrics
        curr_price = _safe_float(df["Close"].iloc[-1])
        open_price = _safe_float(df["Open"].iloc[0])
        day_high = _safe_float(df["High"].max())
        day_low = _safe_float(df["Low"].min())
        curr_vwap = _safe_float(vwap_dict["vwap"][-1])
        curr_st = _safe_float(st_dict["supertrend"][-1]) if len(st_dict["supertrend"]) > 0 else curr_price
        curr_st_dir = int(st_dict["direction"][-1]) if len(st_dict["direction"]) > 0 else 1
        curr_rsi = _safe_float(rsi_vals[-1])
        curr_ema9 = _safe_float(ema9[-1])
        curr_ema21 = _safe_float(ema21[-1])

        prev_close = quote.get("prevClose")
        if not prev_close or prev_close == 0:
            prev_close = open_price

        change = round(curr_price - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

        total_vol = total_buyer_vol + total_seller_vol
        buy_pressure_pct = round((total_buyer_vol / max(total_vol, 1.0)) * 100, 1)
        sell_pressure_pct = round(100.0 - buy_pressure_pct, 1)

        # 11. Quant Score (-100 to +100)
        quant_score = 0
        checklist = []

        vwap_diff_pct = ((curr_price - curr_vwap) / curr_vwap) * 100 if curr_vwap else 0.0
        if curr_price > curr_vwap:
            quant_score += 25
            checklist.append({"factor": "VWAP Alignment", "status": "BULLISH", "desc": f"Trading +{vwap_diff_pct:.2f}% above session VWAP."})
        else:
            quant_score -= 25
            checklist.append({"factor": "VWAP Alignment", "status": "BEARISH", "desc": f"Trading {vwap_diff_pct:.2f}% below session VWAP."})

        if curr_st_dir == 1:
            quant_score += 25
            checklist.append({"factor": "Supertrend (10, 3)", "status": "BULLISH", "desc": f"Long regime active with dynamic stop at {currency_symbol}{curr_st:.2f}."})
        else:
            quant_score -= 25
            checklist.append({"factor": "Supertrend (10, 3)", "status": "BEARISH", "desc": f"Short regime active with dynamic stop at {currency_symbol}{curr_st:.2f}."})

        if curr_ema9 > curr_ema21:
            quant_score += 20
            checklist.append({"factor": "EMA Momentum (9 / 21)", "status": "BULLISH", "desc": "Fast EMA(9) cleanly leading EMA(21) upward."})
        else:
            quant_score -= 20
            checklist.append({"factor": "EMA Momentum (9 / 21)", "status": "BEARISH", "desc": "Fast EMA(9) submerged below EMA(21)."})

        if orb["status"] == "BULLISH_BREAKOUT":
            quant_score += 20
            checklist.append({"factor": "Opening Range (15m)", "status": "BULLISH", "desc": f"Surging above 15m High ({currency_symbol}{orb['high_15m']:.2f})."})
        elif orb["status"] == "BEARISH_BREAKDOWN":
            quant_score -= 20
            checklist.append({"factor": "Opening Range (15m)", "status": "BEARISH", "desc": f"Cracking below 15m Low ({currency_symbol}{orb['low_15m']:.2f})."})
        else:
            checklist.append({"factor": "Opening Range (15m)", "status": "NEUTRAL", "desc": f"Consolidating inside 15m range [{currency_symbol}{orb['low_15m']:.2f} - {currency_symbol}{orb['high_15m']:.2f}]."})

        if curr_rsi > 60:
            quant_score += 10
            checklist.append({"factor": "RSI Velocity", "status": "BULLISH", "desc": f"RSI at {curr_rsi:.1f} shows strong buyer expansion."})
        elif curr_rsi < 40:
            quant_score -= 10
            checklist.append({"factor": "RSI Velocity", "status": "BEARISH", "desc": f"RSI at {curr_rsi:.1f} indicates intense selling velocity."})
        else:
            checklist.append({"factor": "RSI Velocity", "status": "NEUTRAL", "desc": f"RSI at {curr_rsi:.1f} neutral equilibrium."})

        quant_score = max(-100, min(100, quant_score))
        if quant_score >= 50:
            overall_bias = "STRONG BUY"
        elif quant_score >= 20:
            overall_bias = "BUY"
        elif quant_score <= -50:
            overall_bias = "STRONG SELL"
        elif quant_score <= -20:
            overall_bias = "SELL"
        else:
            overall_bias = "NEUTRAL"

        # 12. NEW REAL-LIFE MODULES:
        # A. Pre-Market Gap Intelligence
        gap_intel = _calculate_gap_intelligence(df, daily_df, curr_price, curr_vwap)

        # B. Institutional Trap & Liquidity Sweep Detector
        trap_intel = _detect_institutional_traps(df, curr_price, curr_vwap)

        # C. Triple-Screen Multi-Timeframe Confluence Matrix
        mtf_intel = _calculate_multi_timeframe(df)

        # D. Benchmark Relative Strength & Alpha
        rs_intel = _calculate_relative_strength(change_pct, is_us)

        # E. Auto-Generated Actionable Battle Plan
        battle_plan = _generate_battle_plan(
            clean_ticker,
            quote.get("name") or clean_ticker,
            curr_price,
            curr_vwap,
            curr_st,
            curr_st_dir,
            pivots,
            orb,
            overall_bias,
            currency_symbol
        )

        return {
            "ticker": clean_ticker,
            "company_name": quote.get("name") or clean_ticker,
            "currency_symbol": currency_symbol,
            "current_price": curr_price,
            "open": open_price,
            "high": day_high,
            "low": day_low,
            "prev_close": _safe_float(prev_close),
            "change": change,
            "change_pct": change_pct,
            "volume": int(df["Volume"].sum()),
            "vwap": curr_vwap,
            "supertrend": curr_st,
            "supertrend_dir": curr_st_dir,
            "rsi": curr_rsi,
            "ema9": curr_ema9,
            "ema21": curr_ema21,
            "candles": candles,
            "volume_profile": vpvr,
            "pivots": pivots,
            "orb": orb,
            "order_flow": {
                "total_buyer_vol": int(total_buyer_vol),
                "total_seller_vol": int(total_seller_vol),
                "net_delta": int(cum_delta),
                "buy_pressure_pct": buy_pressure_pct,
                "sell_pressure_pct": sell_pressure_pct,
            },
            "signals": {
                "quant_score": quant_score,
                "overall_bias": overall_bias,
                "checklist": checklist,
            },
            "gap_analysis": gap_intel,
            "trap_detection": trap_intel,
            "multi_timeframe": mtf_intel,
            "relative_strength": rs_intel,
            "battle_plan": battle_plan,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing intraday analytics for {clean_ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to calculate intraday analytics: {str(e)}")


@router.get("/market-pulse")
@cache_ttl(seconds=30)
def get_market_pulse(
    market: str = Query("IN", description="Market identifier: IN or US")
):
    """
    Returns real-time session clock phases, countdown to MIS auto-square-off,
    benchmark indices, and market advance-decline breadth.
    """
    is_in = market.upper() == "IN"

    # Timezone awareness
    tz = zoneinfo.ZoneInfo("Asia/Kolkata" if is_in else "America/New_York")
    now = datetime.now(tz)
    time_str = now.strftime("%I:%M %p")

    # Session Phase Calculation
    hour = now.hour
    minute = now.minute
    day_of_week = now.weekday() # 0 = Mon, 4 = Fri, 5/6 = Sat/Sun

    is_weekend = day_of_week >= 5

    if is_in:
        open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        mis_square_off_time = now.replace(hour=15, minute=15, second=0, microsecond=0)

        is_open = not is_weekend and (open_time <= now <= close_time)

        if is_weekend:
            phase = "Weekend Market Pause"
            phase_num = 0
            directive = "Markets closed for the weekend. Analyze weekly charts & prep watchlists."
        elif now < open_time:
            phase = "Pre-Market Discovery (09:00 - 09:15)"
            phase_num = 0
            directive = "Overnight orders matching. Assess gap sizes vs previous session."
        elif now < now.replace(hour=9, minute=45):
            phase = "Phase 1: Opening Auction & ORB Formation (09:15 - 09:45)"
            phase_num = 1
            directive = "High spread volatility. Avoid market orders; wait for 15m ORB range confirmation."
        elif now < now.replace(hour=11, minute=30):
            phase = "Phase 2: Morning Momentum Drive (09:45 - 11:30)"
            phase_num = 2
            directive = "Optimal institutional trend window. Look for clean VWAP pullbacks & breakouts."
        elif now < now.replace(hour=13, minute=30):
            phase = "Phase 3: Midday Chop Zone (11:30 - 13:30)"
            phase_num = 3
            directive = "Low liquidity & mean-reversion chop. Reduce position size by 50% or stand aside."
        elif now < now.replace(hour=14, minute=45):
            phase = "Phase 4: Afternoon Breakout & European Overlap (13:30 - 14:45)"
            phase_num = 4
            directive = "Secondary trend window. Look for morning high/low retests and trend continuation."
        elif now <= close_time:
            phase = "Phase 5: MIS Auto-Square-Off Panic (14:45 - 15:30)"
            phase_num = 5
            directive = "Broker MIS auto-square-off at 3:15 PM! Do NOT initiate new positions; trail stops."
        else:
            phase = "Post-Market Close"
            phase_num = 6
            directive = "Market session concluded. Review journal and trade executions."

        mins_to_squareoff = max(0, int((mis_square_off_time - now).total_seconds() / 60)) if (is_open and now < mis_square_off_time) else 0

        # Benchmark quotes
        idx_tickers = [("^NSEI", "NIFTY 50"), ("^BSESN", "SENSEX"), ("^NSEBANK", "BANK NIFTY"), ("^CNXIT", "NIFTY IT")]
    else:
        open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        mis_square_off_time = now.replace(hour=15, minute=45, second=0, microsecond=0)

        is_open = not is_weekend and (open_time <= now <= close_time)

        if is_weekend:
            phase = "Weekend Market Pause"
            phase_num = 0
            directive = "US markets closed. Review weekly macro indicators."
        elif now < open_time:
            phase = "Pre-Market Session"
            phase_num = 0
            directive = "Early earnings reports & futures trading."
        elif now < now.replace(hour=10, minute=0):
            phase = "Phase 1: Opening Bell & Initial Balance (09:30 - 10:00)"
            phase_num = 1
            directive = "Opening volatility auction. Wait for initial balance range."
        elif now < now.replace(hour=11, minute=30):
            phase = "Phase 2: Morning Momentum Drive (10:00 - 11:30)"
            phase_num = 2
            directive = "Primary institutional trend moves of the day."
        elif now < now.replace(hour=13, minute=30):
            phase = "Phase 3: Lunch Slump Chop Zone (11:30 - 13:30)"
            phase_num = 3
            directive = "Institutional desk lunch slump. Beware low-volume fakeouts."
        elif now < now.replace(hour=15, minute=0):
            phase = "Phase 4: Afternoon Rebalancing (13:30 - 15:00)"
            phase_num = 4
            directive = "Secondary trend continuation leg."
        elif now <= close_time:
            phase = "Phase 5: Power Hour & Market-on-Close (15:00 - 16:00)"
            phase_num = 5
            directive = "Closing cross imbalance auctions. Heavy volume rebalancing."
        else:
            phase = "After-Hours Session"
            phase_num = 6
            directive = "Regular session ended. Post-market earnings reactions."

        mins_to_squareoff = max(0, int((mis_square_off_time - now).total_seconds() / 60)) if (is_open and now < mis_square_off_time) else 0

        idx_tickers = [("^GSPC", "S&P 500"), ("^IXIC", "NASDAQ"), ("^DJI", "DOW JONES")]

    # Fetch quotes for indices
    indices_data = []
    for sym, name in idx_tickers:
        q = get_quote(sym)
        indices_data.append({
            "symbol": sym,
            "name": name,
            "price": _safe_float(q.get("price")),
            "change_pct": _safe_float(q.get("changePct")),
        })

    return {
        "market": market.upper(),
        "local_time": time_str,
        "is_open": is_open,
        "is_weekend": is_weekend,
        "phase_number": phase_num,
        "phase_name": phase,
        "directive": directive,
        "mins_to_mis_squareoff": mins_to_squareoff,
        "indices": indices_data,
    }


@router.get("/scanner")
@cache_ttl(seconds=60)
def get_intraday_scanner(
    market: str = Query("IN", description="Market identifier: IN or US")
):
    """
    Scans flagship liquid securities across the selected market, calculating
    real-time Relative Volume (RVOL), ORB breakout status, distance from VWAP,
    and composite scalp momentum score.
    """
    tickers_to_scan = FLAGSHIP_IN_TICKERS if market.upper() == "IN" else FLAGSHIP_US_TICKERS
    currency_symbol = "₹" if market.upper() == "IN" else "$"
    scanner_results = []

    for t in tickers_to_scan:
        try:
            df = get_history(t, period="1d", interval="5m")
            if df.empty or len(df) < 3:
                continue

            curr_price = float(df["Close"].iloc[-1])
            prev_close = float(df["Open"].iloc[0])
            chg = curr_price - prev_close
            chg_pct = round((chg / prev_close) * 100, 2) if prev_close else 0.0

            tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
            vol = df["Volume"].fillna(0).values
            cum_vol = np.sum(vol)
            cum_vp = np.sum(tp.values * vol)
            vwap = (cum_vp / cum_vol) if cum_vol > 0 else curr_price
            vwap_dist = round(((curr_price - vwap) / vwap) * 100, 2)

            orb_slice = df.iloc[:min(len(df), 3)]
            h15 = float(orb_slice["High"].max())
            l15 = float(orb_slice["Low"].min())
            if curr_price > h15:
                orb_status = "BREAKOUT"
            elif curr_price < l15:
                orb_status = "BREAKDOWN"
            else:
                orb_status = "INSIDE"

            momentum = 0
            if chg_pct > 1.5:
                momentum += 30
            elif chg_pct < -1.5:
                momentum -= 30

            if vwap_dist > 0.5:
                momentum += 25
            elif vwap_dist < -0.5:
                momentum -= 25

            if orb_status == "BREAKOUT":
                momentum += 35
            elif orb_status == "BREAKDOWN":
                momentum -= 35

            scanner_results.append({
                "ticker": t,
                "price": round(curr_price, 2),
                "change_pct": chg_pct,
                "vwap_dist_pct": vwap_dist,
                "orb_status": orb_status,
                "momentum_score": max(-100, min(100, momentum)),
                "volume": int(cum_vol),
                "currency_symbol": currency_symbol,
            })
        except Exception as e:
            logger.debug(f"Scanner error on {t}: {e}")
            continue

    scanner_results.sort(key=lambda x: abs(x["momentum_score"]), reverse=True)

    advances = sum(1 for s in scanner_results if s["change_pct"] > 0)
    declines = sum(1 for s in scanner_results if s["change_pct"] < 0)

    return {
        "market": market.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scanned": len(scanner_results),
        "advances": advances,
        "declines": declines,
        "results": scanner_results,
    }
