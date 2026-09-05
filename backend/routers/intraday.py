# ==============================================================================
# StockIQ Pro — Intraday Quantitative Analytics Router
# Institutional High-Frequency Terminal & Microstructure Engine
# ==============================================================================

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
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

    # Approximate buying vs selling volume allocation per candle
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

    # Value Area: 70% of total volume expanding outward from POC
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
    Computes Camarilla Equation levels (H1-H4, L1-L4) and Floor Pivots
    from the preceding completed trading session's High, Low, and Close.
    """
    if daily_df.empty or len(daily_df) < 1:
        return {"camarilla": {}, "floor": {}}

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

    # Number of candles in 15m and 30m based on current candle interval
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
    """
    Computes institutional Supertrend indicator with ATR trailing stop series.
    """
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
    direction = np.ones(n, dtype=int)  # 1 = bullish, -1 = bearish

    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    supertrend[0] = final_lb[0]

    for i in range(1, n):
        # Final Upper Band
        if basic_ub[i] < final_ub[i - 1] or c[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        # Final Lower Band
        if basic_lb[i] > final_lb[i - 1] or c[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        # Trend direction evaluation
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


@router.get("/analysis")
def get_intraday_analysis(
    ticker: str = Query(..., description="Stock ticker e.g. RELIANCE.NS or NVDA"),
    interval: str = Query("5m", description="Candle interval: 1m, 3m, 5m, 15m, 30m, 1h"),
    period: str = Query("1d", description="Historical period: 1d or 5d")
):
    """
    Delivers full-spectrum institutional intraday quantitative analytics:
    High-frequency candles, session VWAP & multi-sigma volatility bands,
    Volume Profile (VPVR) with POC, Camarilla & Floor Pivots, Opening Range Breakout (ORB),
    Supertrend trailing stops, and Order Flow Cumulative Volume Delta.
    """
    clean_ticker = ticker.strip().upper()
    is_us = not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO")
    currency_symbol = "$" if is_us else "₹"

    try:
        # 1. Fetch intraday OHLCV
        df = get_history(clean_ticker, period=period, interval=interval)
        if df.empty or len(df) < 2:
            # Fallback if 1d is empty (e.g. before pre-market open)
            df = get_history(clean_ticker, period="5d", interval=interval)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"No intraday chart data available for {clean_ticker}")

        # 2. Fetch daily history for Camarilla / Floor Pivots & Prev Close
        daily_df = get_history(clean_ticker, period="5d", interval="1d")
        pivots = _calculate_pivots(daily_df)

        # 3. Live quote snapshot for company name & headline stats
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
            # Format time string based on timezone
            time_str = idx_time.strftime("%H:%M") if hasattr(idx_time, "strftime") else str(idx_time)
            ts_epoch = int(idx_time.timestamp()) if hasattr(idx_time, "timestamp") else i

            o = _safe_float(df["Open"].iloc[i])
            h = _safe_float(df["High"].iloc[i])
            l = _safe_float(df["Low"].iloc[i])
            c = _safe_float(df["Close"].iloc[i])
            v = int(df["Volume"].iloc[i]) if not pd.isna(df["Volume"].iloc[i]) else 0

            # Estimate buyer vs seller volume per candle
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

        # Benchmark reference for day change
        prev_close = quote.get("prevClose")
        if not prev_close or prev_close == 0:
            prev_close = open_price

        change = round(curr_price - prev_close, 2)
        change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

        # Order flow pressure ratio
        total_vol = total_buyer_vol + total_seller_vol
        buy_pressure_pct = round((total_buyer_vol / max(total_vol, 1.0)) * 100, 1)
        sell_pressure_pct = round(100.0 - buy_pressure_pct, 1)

        # 11. Multi-factor Quant Intraday Bias Scoring (-100 to +100)
        quant_score = 0
        checklist = []

        # VWAP Rule
        vwap_diff_pct = ((curr_price - curr_vwap) / curr_vwap) * 100 if curr_vwap else 0.0
        if curr_price > curr_vwap:
            quant_score += 25
            checklist.append({"factor": "VWAP Alignment", "status": "BULLISH", "desc": f"Trading +{vwap_diff_pct:.2f}% above session VWAP."})
        else:
            quant_score -= 25
            checklist.append({"factor": "VWAP Alignment", "status": "BEARISH", "desc": f"Trading {vwap_diff_pct:.2f}% below session VWAP."})

        # Supertrend Rule
        if curr_st_dir == 1:
            quant_score += 25
            checklist.append({"factor": "Supertrend (10, 3)", "status": "BULLISH", "desc": f"Long regime active with dynamic stop at {currency_symbol}{curr_st:.2f}."})
        else:
            quant_score -= 25
            checklist.append({"factor": "Supertrend (10, 3)", "status": "BEARISH", "desc": f"Short regime active with dynamic stop at {currency_symbol}{curr_st:.2f}."})

        # EMA Ribbon (9 vs 21)
        if curr_ema9 > curr_ema21:
            quant_score += 20
            checklist.append({"factor": "EMA Momentum (9 / 21)", "status": "BULLISH", "desc": "Fast EMA(9) cleanly leading EMA(21) upward."})
        else:
            quant_score -= 20
            checklist.append({"factor": "EMA Momentum (9 / 21)", "status": "BEARISH", "desc": "Fast EMA(9) submerged below EMA(21)."})

        # ORB 15m Rule
        if orb["status"] == "BULLISH_BREAKOUT":
            quant_score += 20
            checklist.append({"factor": "Opening Range (15m)", "status": "BULLISH", "desc": f"Surging above 15m High ({currency_symbol}{orb['high_15m']:.2f})."})
        elif orb["status"] == "BEARISH_BREAKDOWN":
            quant_score -= 20
            checklist.append({"factor": "Opening Range (15m)", "status": "BEARISH", "desc": f"Cracking below 15m Low ({currency_symbol}{orb['low_15m']:.2f})."})
        else:
            checklist.append({"factor": "Opening Range (15m)", "status": "NEUTRAL", "desc": f"Consolidating inside 15m range [{currency_symbol}{orb['low_15m']:.2f} - {currency_symbol}{orb['high_15m']:.2f}]."})

        # RSI Momentum
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
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing intraday analytics for {clean_ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to calculate intraday analytics: {str(e)}")


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

            # VWAP calculation
            tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
            vol = df["Volume"].fillna(0).values
            cum_vol = np.sum(vol)
            cum_vp = np.sum(tp.values * vol)
            vwap = (cum_vp / cum_vol) if cum_vol > 0 else curr_price
            vwap_dist = round(((curr_price - vwap) / vwap) * 100, 2)

            # ORB 15m
            orb_slice = df.iloc[:min(len(df), 3)]
            h15 = float(orb_slice["High"].max())
            l15 = float(orb_slice["Low"].min())
            if curr_price > h15:
                orb_status = "BREAKOUT"
            elif curr_price < l15:
                orb_status = "BREAKDOWN"
            else:
                orb_status = "INSIDE"

            # Scalp Momentum Score
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

    # Sort primarily by absolute momentum score
    scanner_results.sort(key=lambda x: abs(x["momentum_score"]), reverse=True)

    return {
        "market": market.upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scanned": len(scanner_results),
        "results": scanner_results,
    }
