import re
import concurrent.futures
from fastapi import APIRouter, Query, HTTPException, Request
from utils.limiter import limiter
from utils.constants import NIFTY_50_TICKERS
from utils.cache import cache_ttl
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

    q_lower = q.lower().strip()
    q_clean = q_lower.replace(".ns", "").replace(".bo", "").replace("^", "")

    exact_matches = []
    prefix_sym_matches = []
    substr_sym_matches = []
    prefix_name_matches = []
    substr_name_matches = []

    seen_in_search = set()

    for t in ticker_list:
        if sector and t.get("sector", "").lower() != sector.lower():
            continue

        sym = t.get("symbol", "")
        name = t.get("name", "")
        bse_code = t.get("bse_code", "")

        sym_clean = sym.lower().replace(".ns", "").replace(".bo", "").replace("^", "")
        name_lower = name.lower()

        # Match by BSE code
        if bse_code and q_clean == bse_code:
            exact_matches.append(t)
            seen_in_search.add(sym)
            continue

        if sym_clean == q_clean:
            exact_matches.append(t)
            seen_in_search.add(sym)
        elif sym_clean.startswith(q_clean):
            prefix_sym_matches.append(t)
            seen_in_search.add(sym)
        elif q_clean in sym_clean:
            substr_sym_matches.append(t)
            seen_in_search.add(sym)
        elif name_lower.startswith(q_lower):
            prefix_name_matches.append(t)
            seen_in_search.add(sym)
        elif q_lower in name_lower:
            substr_name_matches.append(t)
            seen_in_search.add(sym)

        if len(seen_in_search) >= 200:
            break

    # Combine with strict relevance priority
    ordered = (
        exact_matches +
        prefix_sym_matches +
        substr_sym_matches +
        prefix_name_matches +
        substr_name_matches
    )
    
    # Ensure distinct symbols while preserving order
    deduped = []
    seen = set()
    for item in ordered:
        s = item["symbol"]
        if s not in seen:
            seen.add(s)
            deduped.append(item)

    return {"tickers": deduped[:limit], "total": len(deduped)}



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


SCREENER_POOL = NIFTY_50_TICKERS + [
    'NIFTYBEES.NS', 'BANKBEES.NS', 'GOLDBEES.NS', 'SILVERBEES.NS', 'ITBEES.NS', 'MON100.NS', 'CPSEETF.NS'
]

@cache_ttl(seconds=60)
def _compute_market_screener():
    """
    Computes real-time market screener data across Nifty 50 and key ETFs.
    Cached for 60 seconds to ensure sub-millisecond responses and eliminate rate limits.
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        raw_results = list(executor.map(fetch_quote_safe, SCREENER_POOL))
    
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


@router.get("/market-screener")
@limiter.limit("60/minute")
def get_market_screener(request: Request):
    """
    Returns real-time lists of Top Gainers, Top Losers, Volume Shockers,
    52-Week Highs, and 52-Week Lows for Nifty 50 and popular ETFs.
    """
    return _compute_market_screener()



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
        
        if not re.match(r'^\^?[A-Z0-9&.=-]{1,20}(\.NS|\.BO)?$', ticker_clean):
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
