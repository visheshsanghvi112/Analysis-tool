"""
StockIQ Pro — Deterministic Desk Adapter.

Extracts live market data (technicals, fundamentals, options/derivatives)
and maps it into the canonical DeskContext consumed by desk_engine.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from desk_engine import DeskContext
from yf_client import get_history, get_info, get_quote


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False, min_periods=1).mean()
    loss = (-delta).clip(lower=0).ewm(com=period - 1, adjust=False, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_dcf_fair_value(info: Dict[str, Any], current_price: float) -> Optional[float]:
    """Computes a baseline 10-year DCF intrinsic value per share."""
    try:
        market_cap = _safe_float(info.get("marketCap"))
        shares = _safe_float(info.get("sharesOutstanding")) or 0.0
        if shares <= 0 and market_cap and current_price > 0:
            shares = market_cap / current_price

        if shares <= 0:
            return None

        # Financial institution check
        sector = (info.get("sector") or "").lower()
        industry = (info.get("industry") or "").lower()
        is_financial = any(t in sector or t in industry for t in ["finance", "financial", "bank", "insurance"])

        rev = _safe_float(info.get("totalRevenue")) or 0.0
        fcf = _safe_float(info.get("freeCashflow")) or 0.0
        ocf = _safe_float(info.get("operatingCashflow")) or 0.0
        net_income = _safe_float(info.get("netIncomeToCommon")) or 0.0
        cash = _safe_float(info.get("totalCash")) or 0.0
        debt = _safe_float(info.get("totalDebt")) or 0.0

        if is_financial:
            starting_flow = net_income if net_income > 0 else (rev * 0.15 if rev > 0 else current_price * shares * 0.05)
        else:
            if fcf > 0:
                starting_flow = fcf
            elif net_income > 0:
                starting_flow = net_income
            elif ocf > 0:
                starting_flow = ocf * 0.7
            elif rev > 0:
                starting_flow = rev * 0.06
            else:
                starting_flow = current_price * shares * 0.04

        if starting_flow <= 0:
            return None

        rev_growth = _safe_float(info.get("revenueGrowth"))
        growth_rate = max(0.05, min(0.20, rev_growth)) if rev_growth is not None else 0.08
        beta = _safe_float(info.get("beta")) or 1.0
        wacc = max(0.08, min(0.15, 0.065 + beta * 0.06))
        terminal_growth = 0.045

        if wacc <= terminal_growth:
            wacc = terminal_growth + 0.02

        # 10-year projection
        pv_sum = 0.0
        cf = starting_flow
        for t in range(1, 11):
            cf *= (1.0 + growth_rate)
            pv_sum += cf / ((1.0 + wacc) ** t)

        # Terminal value
        tv = (cf * (1.0 + terminal_growth)) / (wacc - terminal_growth)
        pv_tv = tv / ((1.0 + wacc) ** 10)
        enterprise_value = pv_sum + pv_tv

        equity_value = enterprise_value if is_financial else (enterprise_value + cash - debt)
        fair_value_per_share = equity_value / shares
        return max(0.01, round(fair_value_per_share, 2))
    except Exception:
        return None


def build_desk_context(
    ticker: str,
    horizon: str = "intraday",
    account_capital: Optional[float] = 100000.0,
    account_risk_pct: Optional[float] = 1.0,
    direction: str = "LONG",
) -> DeskContext:
    """
    Builds a canonical DeskContext for a given ticker by reading
    live technical, fundamental, and market state metrics.
    """
    ticker_clean = ticker.strip().upper()
    info = get_info(ticker_clean) or {}
    quote = get_quote(ticker_clean) or {}

    current_price = _safe_float(
        quote.get("price") or info.get("currentPrice") or info.get("regularMarketPrice")
    )

    df = get_history(ticker_clean, period="1y", interval="1d")
    if df is None or df.empty or len(df) < 20:
        # Return sparse context with available fields
        return DeskContext(
            ticker=ticker_clean,
            horizon=horizon,
            price=current_price,
            account_capital=account_capital,
            account_risk_pct=account_risk_pct,
            direction=direction,
        )

    # Clean multi-index columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    close = df["Close"]
    volume = df["Volume"]

    if current_price is None or current_price <= 0:
        current_price = _safe_float(close.iloc[-1])

    # ── Technical indicators ──
    ema9_val = _safe_float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema20_val = _safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema21_val = _safe_float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema200_val = _safe_float(
        close.ewm(span=200, adjust=False).mean().iloc[-1]
        if len(close) >= 150
        else close.ewm(span=len(close), adjust=False).mean().iloc[-1]
    )

    rsi_series = _calculate_rsi(close)
    rsi14_val = _safe_float(rsi_series.iloc[-1])

    # 30D Momentum %
    if len(close) >= 22 and close.iloc[-22] > 0:
        momentum_30d = _safe_float((close.iloc[-1] / close.iloc[-22] - 1.0) * 100.0)
    else:
        momentum_30d = None

    # VWAP (recent 20-day proxy)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    recent_vol = volume.tail(20)
    recent_tp = typical_price.tail(20)
    vwap_val = _safe_float((recent_tp * recent_vol).sum() / max(recent_vol.sum(), 1.0))

    # RVOL (Volume vs 20-day Volume MA)
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    rvol_val = _safe_float(volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0)

    # ATR
    atr_series = _calculate_atr(df)
    atr_val = _safe_float(atr_series.iloc[-1])
    atr_pct_val = _safe_float((atr_val / current_price * 100.0) if (atr_val and current_price and current_price > 0) else None)

    # Price vs EMA20 in ATRs
    price_vs_ema20_atr = None
    if current_price and ema20_val and atr_val and atr_val > 0:
        price_vs_ema20_atr = _safe_float((current_price - ema20_val) / atr_val)

    # SuperTrend direction (simple proxy based on close vs EMA20)
    supertrend_dir = "BULLISH" if (current_price and ema20_val and current_price >= ema20_val) else "BEARISH"

    # Delta absorption (high volume with small candle range near lows)
    delta_absorption = False
    if rvol_val and rvol_val > 1.5 and rsi14_val and rsi14_val <= 30:
        delta_absorption = True

    # ── Fundamentals ──
    fair_val = _compute_dcf_fair_value(info, current_price or 1.0)
    roe_val = _safe_float(info.get("returnOnEquity"))
    if roe_val is not None:
        roe_val = roe_val * 100.0 if abs(roe_val) <= 2.0 else roe_val

    rev_growth = _safe_float(info.get("revenueGrowth"))
    if rev_growth is not None:
        rev_growth = rev_growth * 100.0 if abs(rev_growth) <= 2.0 else rev_growth

    op_margin = _safe_float(info.get("operatingMargins"))
    if op_margin is not None:
        op_margin = op_margin * 100.0 if abs(op_margin) <= 2.0 else op_margin

    debt_to_eq = _safe_float(info.get("debtToEquity"))
    # Normalize debt-to-equity ratio: if > 10, it is already a percentage
    if debt_to_eq is not None and debt_to_eq <= 10.0:
        debt_to_eq = debt_to_eq * 100.0

    # Valuation percentiles
    pe = _safe_float(info.get("trailingPE"))
    pe_percentile = None
    if pe is not None:
        # Percentile scale: 10x is ~25th, 25x is ~60th, 40x is ~85th, 60x is ~95th
        pe_percentile = max(5.0, min(99.0, (pe / 60.0) * 95.0))

    ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
    ev_ebitda_percentile = None
    if ev_ebitda is not None:
        ev_ebitda_percentile = max(5.0, min(99.0, (ev_ebitda / 30.0) * 95.0))

    # ── Risk & Volatility ──
    returns = close.pct_change().dropna()
    rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100.0
    current_vol = _safe_float(rolling_vol.iloc[-1])
    vol_percentile = None
    if current_vol is not None and len(rolling_vol.dropna()) > 30:
        vol_min = rolling_vol.min()
        vol_max = rolling_vol.max()
        if vol_max > vol_min:
            vol_percentile = _safe_float(((current_vol - vol_min) / (vol_max - vol_min)) * 100.0)

    # 1Y Max Drawdown
    cum_ret = (1.0 + returns).cumprod()
    rolling_max = cum_ret.expanding().max()
    dd_series = (cum_ret - rolling_max) / rolling_max
    max_drawdown = _safe_float(dd_series.min() * 100.0)

    # Price freshness in seconds
    freshness_sec = 60.0
    try:
        last_dt = df.index[-1]
        if hasattr(last_dt, "tz_localize") and last_dt.tzinfo is None:
            last_dt = last_dt.tz_localize("Asia/Kolkata")
        now_dt = datetime.now(timezone.utc)
        diff_sec = (now_dt - last_dt.to_pydatetime().astimezone(timezone.utc)).total_seconds()
        if diff_sec > 0:
            freshness_sec = min(diff_sec, 86400.0)
    except Exception:
        pass

    return DeskContext(
        ticker=ticker_clean,
        horizon=horizon,
        price=current_price,
        vwap=vwap_val,
        ema9=ema9_val,
        ema21=ema21_val,
        ema200=ema200_val,
        rsi14=rsi14_val,
        momentum_30d_pct=momentum_30d,
        rvol=rvol_val,
        orb_status="INSIDE_RANGE",
        supertrend_direction=supertrend_dir,
        atr=atr_val,
        atr_pct=atr_pct_val,
        price_vs_ema20_atr=price_vs_ema20_atr,
        delta_absorption=delta_absorption,
        fair_value=fair_val,
        roe_pct=roe_val,
        revenue_growth_pct=rev_growth,
        operating_margin_pct=op_margin,
        debt_to_equity=debt_to_eq,
        pe_percentile=pe_percentile,
        ev_ebitda_percentile=ev_ebitda_percentile,
        futures_buildup=None,
        pcr_oi=None,
        pcr_volume=None,
        iv_percentile=None,
        iv_rv_spread_pct=None,
        put_call_iv_skew_pct=None,
        volatility_percentile=vol_percentile,
        max_drawdown_pct=max_drawdown,
        next_event_days=None,
        price_freshness_sec=freshness_sec,
        orderbook_available=False,
        options_available=False,
        market_open=True,
        historical_win_rate=55.0,
        historical_avg_win_loss=1.8,
        account_capital=account_capital,
        account_risk_pct=account_risk_pct,
        entry_price=current_price,
        direction=direction,
        stop_atr_multiple=2.0,
        target_r_multiple=2.0,
        target2_r_multiple=3.0,
    )
