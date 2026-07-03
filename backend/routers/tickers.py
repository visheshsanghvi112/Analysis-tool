import re
import concurrent.futures
from fastapi import APIRouter, Query, HTTPException, Request
from utils.limiter import limiter
from utils.constants import NIFTY_50_TICKERS
from services.ticker_manager import ensure_ticker_list
from yf_client import get_quote, get_fundamentals_data

router = APIRouter(prefix="/api", tags=["tickers"])

@router.get("/tickers")
def search_tickers(
    q: str = Query("", description="Query string to search tickers"),
    sector: str = Query("", description="Filter by sector name"),
    limit: int = Query(30, description="Max results", le=2000)
):
    """
    Search NSE tickers by symbol or company name.
    Optionally filter by sector. Symbol matches ranked higher than name matches.
    """
    ticker_list = ensure_ticker_list()

    # Sector-only filter — return all stocks in that sector
    if sector and not q:
        sector_lower = sector.lower()
        results = [t for t in ticker_list if t.get("sector", "").lower() == sector_lower]
        return {"tickers": results[:limit], "total": len(results)}

    if not q:
        return {"tickers": ticker_list[:limit], "total": len(ticker_list)}

    q_lower = q.lower()
    symbol_matches = []
    name_matches = []

    for t in ticker_list:
        # Optional sector pre-filter
        if sector and t.get("sector", "").lower() != sector.lower():
            continue
        sym_lower = t["symbol"].lower().replace(".ns", "")
        name_lower = t["name"].lower()
        if q_lower in sym_lower:
            symbol_matches.append(t)
        elif q_lower in name_lower:
            name_matches.append(t)
        if len(symbol_matches) + len(name_matches) >= 120:
            break

    combined = symbol_matches[:limit//2] + name_matches[:limit//2]
    return {"tickers": combined[:limit], "total": len(combined)}


@router.get("/sectors")
def get_sectors():
    """
    Returns all stocks from TICKER_LIST grouped by sector.
    Stocks without a known sector are placed in 'Others'.
    Also returns a list of all unique sectors with counts.
    """
    ticker_list = ensure_ticker_list()

    grouped = {}
    for t in ticker_list:
        sec = t.get("sector") or "Others"
        if sec not in grouped:
            grouped[sec] = []
        grouped[sec].append({"symbol": t["symbol"], "name": t["name"]})

    # Sort each sector alphabetically by symbol
    for sec in grouped:
        grouped[sec].sort(key=lambda x: x["symbol"])

    # Build sector summary list sorted by count desc
    summary = [
        {"sector": sec, "count": len(stocks)}
        for sec, stocks in grouped.items()
    ]
    summary.sort(key=lambda x: -x["count"])

    return {
        "sectors": summary,
        "grouped": grouped,
        "total_stocks": len(ticker_list)
    }


@router.get("/market-screener")
@limiter.limit("20/minute")
def get_market_screener(request: Request):
    """
    Returns real-time lists of Top Gainers, Top Losers, Volume Shockers,
    52-Week Highs, and 52-Week Lows for Nifty 50 stocks.
    """
    def fetch_quote_safe(ticker):
        try:
            q = get_quote(ticker)
            if q and q.get("price") is not None:
                q["ticker"] = ticker
                return q
        except Exception:
            pass
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        raw_results = list(executor.map(fetch_quote_safe, NIFTY_50_TICKERS))
    
    quotes = [r for r in raw_results if r is not None]

    if not quotes:
        return {
            "gainers": [],
            "losers": [],
            "volume": [],
            "high52w": [],
            "low52w": []
        }

    # 1. Gainers & Losers (sorted by daily changePct)
    valid_change = [q for q in quotes if q.get("changePct") is not None]
    sorted_change = sorted(valid_change, key=lambda x: x["changePct"], reverse=True)
    gainers = sorted_change
    losers = list(reversed(sorted_change))

    # 2. Volume Shockers (highest volume)
    valid_vol = [q for q in quotes if q.get("volume") is not None]
    top_volume = sorted(valid_vol, key=lambda x: x["volume"], reverse=True)

    # 3. Proximity to 52-Week High / Low
    high_52w = []
    low_52w = []

    for q in quotes:
        price = q.get("price")
        
        # 52w High Proximity
        high = q.get("fiftyTwoWeekHigh")
        if price and high:
            pct = ((high - price) / high) * 100
            q_copy = dict(q)
            q_copy["pct_from_52w_high"] = round(pct, 2)
            high_52w.append(q_copy)
    high_52w = sorted(high_52w, key=lambda x: x["pct_from_52w_high"])

    for q in quotes:
        price = q.get("price")
        
        # 52w Low Proximity
        low = q.get("fiftyTwoWeekLow")
        if price and low:
            pct = ((price - low) / low) * 100
            q_copy = dict(q)
            q_copy["pct_from_52w_low"] = round(pct, 2)
            low_52w.append(q_copy)
    low_52w = sorted(low_52w, key=lambda x: x["pct_from_52w_low"])

    return {
        "gainers": gainers[:20],
        "losers": losers[:20],
        "volume": top_volume[:20],
        "high52w": high_52w[:20],
        "low52w": low_52w[:20]
    }


@router.get("/live")
@limiter.limit("30/minute")
def get_live_price(
    request: Request,
    ticker: str = Query(..., description="Stock ticker, e.g. HDFCBANK.NS", max_length=20)
):
    """
    Returns the most recent price snapshot from yfinance fast_info.
    Yahoo Finance data is ~15 minutes delayed for NSE stocks.
    """
    try:
        ticker_clean = ticker.strip().upper()
        
        if not re.match(r'^[A-Z0-9&.-]{1,15}(\.NS|\.BO)?$', ticker_clean):
            raise HTTPException(status_code=400, detail="Invalid ticker format")

        q = get_quote(ticker_clean)
        if not q or q.get("price") is None:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker_clean}")

        return {"ticker": ticker_clean, **q}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/fundamentals")
async def fundamentals(request: Request, ticker: str = Query(..., description="NSE/BSE ticker, e.g. HDFCBANK.NS")):
    """
    Returns deep fundamental data for long-term investors.
    """
    ticker_clean = ticker.strip().upper()
    data = get_fundamentals_data(ticker_clean)
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch fundamental data for {ticker_clean}. The ticker may be invalid or data unavailable."
        )
    return data
