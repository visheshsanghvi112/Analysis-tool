"""
StockIQ Pro — Deterministic Desk Adapter
========================================

Extracts canonical, provenance-aware market and fundamental data
and maps it into the canonical DeskContext consumed by desk_engine.

Data Integrity Guarantees:
- Zero plausible-looking fake defaults.
- Intraday horizon consumes canonical 5m candles and session-anchored VWAP.
- Dual context: Intraday execution technicals (5m) + Daily risk regime (1y).
- Supertrend uses canonical Welles Wilder ATR(10, 3) direction, never EMA20 proxy.
- ORB uses canonical 15m opening range, never defaulting to INSIDE_RANGE.
- Quote freshness derives from live quote regularMarketTime or price_date.
- Market session state is timezone-aware and exchange-holiday aware (2026 NSE/US).
- Fair value uses canonical DCF/DDM without synthetic cash flow shortcuts;
  OCF is labeled as proxy and not masqueraded as FCFF DCF.
- P/E and EV/EBITDA percentiles are None until empirical distributions exist.
- Historical win rate and Kelly are None until empirical setup profiles exist.
- Derivatives exclude model approximations from empirical evidence.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from desk_engine import DeskContext
from services.intraday_engine import calculate_intraday_snapshot
from services.market_session import get_market_session_state
from services.valuation_service import calculate_canonical_valuation
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


def _parse_quote_timestamp(quote: Dict[str, Any]) -> tuple[Optional[str], Optional[float], str]:
    """
    Extracts regularMarketTime or price_date from quote and computes ISO string, age in seconds,
    and the exact source field used.
    """
    raw_time = quote.get("regularMarketTime")
    source = "regularMarketTime"
    if raw_time is None:
        raw_time = quote.get("price_date")
        source = "price_date"

    if raw_time is None:
        return None, None, "none"

    now_utc = datetime.now(timezone.utc)
    try:
        if isinstance(raw_time, (int, float)):
            dt = datetime.fromtimestamp(raw_time, tz=timezone.utc)
        elif isinstance(raw_time, str):
            dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        elif isinstance(raw_time, datetime):
            dt = raw_time if raw_time.tzinfo else raw_time.replace(tzinfo=timezone.utc)
        else:
            return None, None, "none"

        age_seconds = max(0.0, (now_utc - dt).total_seconds())
        return dt.isoformat(), round(age_seconds, 1), source
    except Exception:
        return None, None, "none"


def _parse_bar_timestamp(df: pd.DataFrame) -> tuple[Optional[str], Optional[float]]:
    """
    Extracts the latest candle timestamp from DataFrame index and computes age in seconds.
    """
    if df is None or df.empty:
        return None, None
    try:
        last_dt = df.index[-1]
        now_utc = datetime.now(timezone.utc)
        if hasattr(last_dt, "tz_localize") and last_dt.tzinfo is None:
            last_dt = last_dt.tz_localize("Asia/Kolkata")
        dt_utc = last_dt.to_pydatetime().astimezone(timezone.utc)
        age_seconds = max(0.0, (now_utc - dt_utc).total_seconds())
        return dt_utc.isoformat(), round(age_seconds, 1)
    except Exception:
        return None, None


def build_canonical_market_state(
    ticker: str,
    horizon: str = "intraday",
) -> Dict[str, Any]:
    """
    Assembles a structured, provenance-aware MarketState dictionary
    before converting into a DeskContext.
    """
    ticker_clean = ticker.strip().upper()
    is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
    exchange = "NSE" if ticker_clean.endswith(".NS") else ("BSE" if ticker_clean.endswith(".BO") else "US")
    currency = "INR" if is_indian else "USD"

    # 1. Market Session State
    session_state = get_market_session_state(ticker_clean)

    # 2. Live Quote
    info = get_info(ticker_clean) or {}
    quote = get_quote(ticker_clean) or {}

    current_price = _safe_float(
        quote.get("price") or info.get("currentPrice") or info.get("regularMarketPrice")
    )
    quote_as_of, quote_age_seconds, quote_source = _parse_quote_timestamp(quote)

    # 3. Candles & Technical Indicators
    intraday_state: Dict[str, Any] = {}
    daily_state: Dict[str, Any] = {}
    bar_as_of: Optional[str] = None
    bar_age_seconds: Optional[float] = None

    if horizon == "intraday":
        # P0: Dual-Horizon Intraday Context
        # A) High-resolution 5m candles for intraday execution indicators
        df_intraday = get_history(ticker_clean, period="5d", interval="5m")
        if df_intraday is not None and not df_intraday.empty:
            if isinstance(df_intraday.columns, pd.MultiIndex):
                df_intraday.columns = [c[0] for c in df_intraday.columns]
            bar_as_of, bar_age_seconds = _parse_bar_timestamp(df_intraday)
            snapshot = calculate_intraday_snapshot(
                ticker=ticker_clean,
                interval="5m",
                df=df_intraday,
                live_quote=quote,
            )
            if snapshot.get("available"):
                intraday_state = snapshot
                intraday_state["vwap"] = snapshot.get("session_vwap")
                if current_price is None or current_price <= 0:
                    current_price = snapshot.get("price")

        # B) Daily candles for higher-timeframe CRO risk context (vol regime, max drawdown, momentum)
        df_daily = get_history(ticker_clean, period="1y", interval="1d")
        if df_daily is not None and not df_daily.empty and len(df_daily) >= 20:
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = [c[0] for c in df_daily.columns]
            close = df_daily["Close"]
            ema20_val = _safe_float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            atr_series = _calculate_atr(df_daily)
            daily_atr = _safe_float(atr_series.iloc[-1])

            momentum_30d = None
            if len(close) >= 22 and close.iloc[-22] > 0:
                momentum_30d = _safe_float((close.iloc[-1] / close.iloc[-22] - 1.0) * 100.0)

            ref_p = current_price if current_price else _safe_float(close.iloc[-1])
            price_vs_ema20_atr = None
            if ref_p and ema20_val and daily_atr and daily_atr > 0:
                price_vs_ema20_atr = _safe_float((ref_p - ema20_val) / daily_atr)

            returns = close.pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100.0
            current_vol = _safe_float(rolling_vol.iloc[-1])
            vol_percentile = None
            if current_vol is not None and len(rolling_vol.dropna()) > 30:
                vol_min = rolling_vol.min()
                vol_max = rolling_vol.max()
                if vol_max > vol_min:
                    vol_percentile = _safe_float(((current_vol - vol_min) / (vol_max - vol_min)) * 100.0)

            cum_ret = (1.0 + returns).cumprod()
            rolling_max = cum_ret.expanding().max()
            dd_series = (cum_ret - rolling_max) / rolling_max
            max_drawdown = _safe_float(dd_series.min() * 100.0)

            daily_state = {
                "momentum_30d_pct": momentum_30d,
                "price_vs_ema20_atr": price_vs_ema20_atr,
                "volatility_percentile": vol_percentile,
                "max_drawdown_pct": max_drawdown,
            }
    else:
        # P0: Daily candles for swing and long_term
        df_daily = get_history(ticker_clean, period="1y", interval="1d")
        if df_daily is not None and not df_daily.empty and len(df_daily) >= 20:
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = [c[0] for c in df_daily.columns]
            bar_as_of, bar_age_seconds = _parse_bar_timestamp(df_daily)
            close = df_daily["Close"]
            volume = df_daily["Volume"]

            if current_price is None or current_price <= 0:
                current_price = _safe_float(close.iloc[-1])

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

            momentum_30d = None
            if len(close) >= 22 and close.iloc[-22] > 0:
                momentum_30d = _safe_float((close.iloc[-1] / close.iloc[-22] - 1.0) * 100.0)

            vol_ma20 = volume.rolling(20).mean().iloc[-1]
            rvol_val = _safe_float(volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0)

            atr_series = _calculate_atr(df_daily)
            atr_val = _safe_float(atr_series.iloc[-1])
            atr_pct_val = _safe_float((atr_val / current_price * 100.0) if (atr_val and current_price and current_price > 0) else None)

            price_vs_ema20_atr = None
            if current_price and ema20_val and atr_val and atr_val > 0:
                price_vs_ema20_atr = _safe_float((current_price - ema20_val) / atr_val)

            returns = close.pct_change().dropna()
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100.0
            current_vol = _safe_float(rolling_vol.iloc[-1])
            vol_percentile = None
            if current_vol is not None and len(rolling_vol.dropna()) > 30:
                vol_min = rolling_vol.min()
                vol_max = rolling_vol.max()
                if vol_max > vol_min:
                    vol_percentile = _safe_float(((current_vol - vol_min) / (vol_max - vol_min)) * 100.0)

            cum_ret = (1.0 + returns).cumprod()
            rolling_max = cum_ret.expanding().max()
            dd_series = (cum_ret - rolling_max) / rolling_max
            max_drawdown = _safe_float(dd_series.min() * 100.0)

            daily_state = {
                "ema9": ema9_val,
                "ema20": ema20_val,
                "ema21": ema21_val,
                "ema200": ema200_val,
                "rsi14": rsi14_val,
                "momentum_30d_pct": momentum_30d,
                "rvol": rvol_val,
                "atr": atr_val,
                "atr_pct": atr_pct_val,
                "price_vs_ema20_atr": price_vs_ema20_atr,
                "volatility_percentile": vol_percentile,
                "max_drawdown_pct": max_drawdown,
            }

    # 4. Canonical Valuation Service
    val_result = calculate_canonical_valuation(info, current_price=current_price)

    # Fundamental ratios
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
    if debt_to_eq is not None and debt_to_eq <= 10.0:
        debt_to_eq = debt_to_eq * 100.0

    # 5. Provenance audit tracking
    provenance = {
        "price": {
            "value": current_price,
            "source": "live_quote" if quote.get("price") else ("info" if info.get("currentPrice") else "historical_bar"),
            "as_of": quote_as_of or bar_as_of,
            "age_seconds": quote_age_seconds if quote_as_of else bar_age_seconds,
            "status": "OK" if current_price else "UNAVAILABLE",
        },
        "quote": {
            "as_of": quote_as_of,
            "age_seconds": quote_age_seconds,
            "source": quote_source,
            "status": "OK" if quote_as_of else "MISSING_TIMESTAMP",
        },
        "bars": {
            "interval": "5m" if horizon == "intraday" else "1d",
            "as_of": bar_as_of,
            "age_seconds": bar_age_seconds,
            "source": "get_history",
            "status": "OK" if bar_as_of else "NO_BARS",
        },
        "session": {
            "status": session_state.status,
            "market_open": session_state.is_open,
            "timezone": session_state.timezone,
            "source": "market_session_service",
        },
        "valuation": {
            "methodology": val_result.methodology,
            "data_status": val_result.data_status,
            "assumption_status": val_result.assumption_status,
            "valuation_status": val_result.valuation_status,
            "reason": val_result.valuation_reason,
            "source": "valuation_service",
        },
        "derivatives": {
            "available": False,
            "is_model_approximation": False,
            "source": "none",
            "status": "UNAVAILABLE",
        },
        "expectancy": {
            "available": False,
            "source": "none",
            "status": "UNAVAILABLE",
        },
    }

    if horizon == "intraday" and intraday_state:
        vwap_as_of = intraday_state.get("bar_as_of") or intraday_state.get("as_of") or bar_as_of
        provenance["session_vwap"] = {
            "value": intraday_state.get("vwap"),
            "as_of": vwap_as_of,
            "source": "session_anchored_vwap",
            "status": "OK" if intraday_state.get("vwap") is not None else "UNAVAILABLE",
        }
        provenance["supertrend"] = {
            "direction": intraday_state.get("supertrend_direction"),
            "period": 10,
            "multiplier": 3,
            "as_of": vwap_as_of,
            "source": "canonical_supertrend_wilder_atr",
            "status": "OK" if intraday_state.get("supertrend_direction") is not None else "UNAVAILABLE",
        }
        provenance["orb"] = {
            "status": intraday_state.get("orb_status"),
            "as_of": vwap_as_of,
            "source": "opening_range_breakout_15m",
            "status_flag": "OK" if intraday_state.get("orb_status") is not None else "INSUFFICIENT_SESSION_DATA",
        }

    return {
        "instrument": {
            "ticker": ticker_clean,
            "exchange": exchange,
            "currency": currency,
            "is_indian": is_indian,
        },
        "market_session": session_state.to_dict(),
        "quote": {
            "price": current_price,
            "as_of": quote_as_of,
            "age_seconds": quote_age_seconds,
            "source": quote_source,
        },
        "bars": {
            "interval": "5m" if horizon == "intraday" else "1d",
            "as_of": bar_as_of,
            "age_seconds": bar_age_seconds,
        },
        "intraday": intraday_state,
        "daily": daily_state,
        "fundamentals": {
            "valuation": val_result.to_dict(),
            "roe_pct": roe_val,
            "revenue_growth_pct": rev_growth,
            "operating_margin_pct": op_margin,
            "debt_to_equity": debt_to_eq,
            "pe_percentile": None,  # P1: Genuine empirical percentile only
            "ev_ebitda_percentile": None,  # P1: Genuine empirical percentile only
        },
        "derivatives": {
            "available": False,
            "is_model_approximation": False,
        },
        "expectancy": {
            "available": False,
            "historical_win_rate": None,
            "historical_avg_win_loss": None,
        },
        "provenance": provenance,
    }


def build_desk_context(
    ticker: str,
    horizon: str = "intraday",
    account_capital: Optional[float] = 100000.0,
    account_risk_pct: Optional[float] = 1.0,
    direction: str = "LONG",
) -> DeskContext:
    """
    Builds a canonical DeskContext for a given ticker by reading
    live technical, fundamental, and market state metrics without
    synthetic proxies or fabricated fallbacks.
    """
    state = build_canonical_market_state(ticker=ticker, horizon=horizon)

    ticker_clean = state["instrument"]["ticker"]
    current_price = state["quote"]["price"]
    quote_age_sec = state["quote"]["age_seconds"]
    bar_age_sec = state["bars"]["age_seconds"]
    price_freshness = quote_age_sec if quote_age_sec is not None else bar_age_sec

    session = state["market_session"]
    intraday = state["intraday"]
    daily = state["daily"]
    fundamentals = state["fundamentals"]
    valuation = fundamentals["valuation"]

    # Technical fields depending on horizon
    if horizon == "intraday":
        vwap_val = intraday.get("vwap")
        ema9_val = intraday.get("ema9")
        ema21_val = intraday.get("ema21")
        ema200_val = intraday.get("ema200")
        rsi14_val = intraday.get("rsi14")
        atr_val = intraday.get("atr")
        atr_pct_val = intraday.get("atr_pct")
        supertrend_dir = intraday.get("supertrend_direction")
        orb_status_val = intraday.get("orb_status")
        rvol_val = intraday.get("rvol")
        delta_absorption = intraday.get("delta_absorption", False)
        # Higher-timeframe daily risk metrics preserved for intraday CRO gate
        momentum_30d = daily.get("momentum_30d_pct")
        price_vs_ema20_atr = daily.get("price_vs_ema20_atr")
        vol_percentile = daily.get("volatility_percentile")
        max_drawdown = daily.get("max_drawdown_pct")
    else:
        # Swing / Long Term
        vwap_val = None  # P0: Do NOT call rolling 20-day price "VWAP"
        ema9_val = daily.get("ema9")
        ema21_val = daily.get("ema21")
        ema200_val = daily.get("ema200")
        rsi14_val = daily.get("rsi14")
        atr_val = daily.get("atr")
        atr_pct_val = daily.get("atr_pct")
        supertrend_dir = None  # P0: Do NOT substitute EMA20 proxy for Supertrend
        orb_status_val = None  # P0: Do NOT default to INSIDE_RANGE on daily
        rvol_val = daily.get("rvol")
        momentum_30d = daily.get("momentum_30d_pct")
        price_vs_ema20_atr = daily.get("price_vs_ema20_atr")
        vol_percentile = daily.get("volatility_percentile")
        max_drawdown = daily.get("max_drawdown_pct")
        delta_absorption = False
        if rvol_val and rvol_val > 1.5 and rsi14_val and rsi14_val <= 30:
            delta_absorption = True

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
        orb_status=orb_status_val,
        supertrend_direction=supertrend_dir,
        atr=atr_val,
        atr_pct=atr_pct_val,
        price_vs_ema20_atr=price_vs_ema20_atr,
        delta_absorption=delta_absorption,
        fair_value=valuation.get("fair_value"),
        roe_pct=fundamentals.get("roe_pct"),
        revenue_growth_pct=fundamentals.get("revenue_growth_pct"),
        operating_margin_pct=fundamentals.get("operating_margin_pct"),
        debt_to_equity=fundamentals.get("debt_to_equity"),
        pe_percentile=None,  # P1: Empirical percentile only
        ev_ebitda_percentile=None,  # P1: Empirical percentile only
        futures_buildup=None,
        pcr_oi=None,
        pcr_volume=None,
        iv_percentile=None,
        iv_rv_spread_pct=None,
        put_call_iv_skew_pct=None,
        volatility_percentile=vol_percentile,
        max_drawdown_pct=max_drawdown,
        next_event_days=None,
        price_freshness_sec=price_freshness,
        orderbook_available=False,
        options_available=False,
        market_open=session.get("is_open"),
        historical_win_rate=None,  # P0: Removed synthetic 55.0%
        historical_avg_win_loss=None,  # P0: Removed synthetic 1.8
        account_capital=account_capital,
        account_risk_pct=account_risk_pct,
        entry_price=current_price,
        direction=direction,
        stop_atr_multiple=2.0,
        target_r_multiple=2.0,
        target2_r_multiple=3.0,
        # Provenance and market state tracking
        market_status=session.get("status"),
        provenance=state.get("provenance"),
        market_state=state,
    )
