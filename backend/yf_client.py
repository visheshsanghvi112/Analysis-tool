# ============================================================
# Yahoo Finance direct REST client — bypasses the yfinance
# Python library which gets blocked on Vercel serverless IPs.
# Uses Yahoo's undocumented but stable v8/v10 JSON endpoints.
# ============================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
})

_CHART_URL  = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_QUOTE_URL  = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"


def _get(url, params=None, retries=2):
    """GET with simple retry logic."""
    for attempt in range(retries + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt < retries:
                time.sleep(1.5)
                continue
        except Exception:
            if attempt < retries:
                time.sleep(1)
    return None


def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Returns OHLCV DataFrame for the given ticker.
    period:   1d 5d 1mo 3mo 6mo 1y 2y 5y
    interval: 1m 5m 15m 1h 1d 1wk 1mo
    """
    data = _get(
        _CHART_URL.format(ticker=ticker),
        params={"interval": interval, "range": period, "includeAdjustedClose": "true"},
    )
    if not data:
        return pd.DataFrame()

    try:
        result = data["chart"]["result"]
        if not result:
            return pd.DataFrame()
        r = result[0]
        timestamps = r.get("timestamp", [])
        quote = r["indicators"]["quote"][0]
        adjclose_list = r["indicators"].get("adjclose", [{}])
        adjclose = adjclose_list[0].get("adjclose", []) if adjclose_list else []

        df = pd.DataFrame({
            "Open":   quote.get("open",   []),
            "High":   quote.get("high",   []),
            "Low":    quote.get("low",    []),
            "Close":  quote.get("close",  []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata"))

        if adjclose:
            df["Adj Close"] = adjclose

        df.index.name = "Date"
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def get_quote(ticker: str) -> dict:
    """
    Returns a live price snapshot dict using the chart meta endpoint.
    Fields: price, prevClose, dayHigh, dayLow, volume, marketCap,
            change, changePct, timestamp
    """
    data = _get(
        _CHART_URL.format(ticker=ticker),
        params={"interval": "1d", "range": "2d"},
    )
    if not data:
        return {}

    try:
        result = data["chart"]["result"]
        if not result:
            return {}
        meta = result[0]["meta"]

        price      = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        day_high   = meta.get("regularMarketDayHigh")
        day_low    = meta.get("regularMarketDayLow")
        volume     = meta.get("regularMarketVolume")
        mkt_cap    = meta.get("marketCap")

        change = change_pct = None
        if price and prev_close and prev_close != 0:
            change     = round(price - prev_close, 2)
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        return {
            "price":      round(price, 2) if price else None,
            "prevClose":  round(prev_close, 2) if prev_close else None,
            "dayHigh":    round(day_high, 2) if day_high else None,
            "dayLow":     round(day_low, 2) if day_low else None,
            "volume":     int(volume) if volume else None,
            "marketCap":  mkt_cap,
            "change":     change,
            "changePct":  change_pct,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return {}


def get_info(ticker: str) -> dict:
    """
    Returns fundamental info via quoteSummary (price + defaultKeyStatistics
    + financialData modules).
    """
    data = _get(
        _QUOTE_URL.format(ticker=ticker),
        params={"modules": "price,defaultKeyStatistics,financialData,summaryDetail"},
    )
    if not data:
        return {}

    try:
        res = data.get("quoteSummary", {}).get("result", [])
        if not res:
            return {}

        out = {}
        for module in res[0].values():
            if isinstance(module, dict):
                for k, v in module.items():
                    if isinstance(v, dict) and "raw" in v:
                        out[k] = v["raw"]
                    elif not isinstance(v, dict):
                        out[k] = v
        return out
    except Exception:
        return {}
