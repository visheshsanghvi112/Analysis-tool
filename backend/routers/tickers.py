import re
import concurrent.futures
from difflib import SequenceMatcher
from fastapi import APIRouter, Query, HTTPException, Request
from utils.limiter import limiter
from utils.constants import NIFTY_50_TICKERS
from utils.cache import cache_ttl
from services.ticker_manager import ensure_ticker_list
from yf_client import get_quote, get_fundamentals_data

router = APIRouter(prefix="/api", tags=["tickers"])

# Curated financial aliases mapping colloquial/common terms directly to canonical tickers
SEARCH_ALIASES = {
    # Banking & NBFC
    'sbi': 'SBIN.NS', 'state bank': 'SBIN.NS', 'state bank of india': 'SBIN.NS',
    'hdfc': 'HDFCBANK.NS', 'hdfc bank': 'HDFCBANK.NS', 'hdfcbk': 'HDFCBANK.NS',
    'icici': 'ICICIBANK.NS', 'icici bank': 'ICICIBANK.NS',
    'kotak': 'KOTAKBANK.NS', 'kotak bank': 'KOTAKBANK.NS',
    'axis': 'AXISBANK.NS', 'axis bank': 'AXISBANK.NS',
    'indusind': 'INDUSINDBK.NS', 'pnb': 'PNB.NS', 'bob': 'BANKBARODA.NS',
    'bajaj finance': 'BAJFINANCE.NS', 'bajfinance': 'BAJFINANCE.NS',
    'bajaj finserv': 'BAJAJFINSV.NS', 'bajfinsv': 'BAJAJFINSV.NS',
    # Conglomerates & Auto
    'reliance': 'RELIANCE.NS', 'ril': 'RELIANCE.NS', 'jio': 'RELIANCE.NS',
    'l&t': 'LT.NS', 'larsen': 'LT.NS', 'larsen & toubro': 'LT.NS', 'larsen and toubro': 'LT.NS',
    'm&m': 'M&M.NS', 'mahindra': 'M&M.NS', 'mahindra & mahindra': 'M&M.NS',
    'tata motors': 'TMCV.NS', 'tatamotors': 'TMCV.NS', 'tata motor': 'TMCV.NS',
    'tata power': 'TATAPOWER.NS', 'tata steel': 'TATASTEEL.NS', 'tata tech': 'TATATECH.NS',
    'maruti': 'MARUTI.NS', 'maruti suzuki': 'MARUTI.NS',
    'bajaj auto': 'BAJAJ-AUTO.NS', 'hero': 'HEROMOTOCO.NS', 'hero motocorp': 'HEROMOTOCO.NS',
    'eicher': 'EICHERMOT.NS', 'royal enfield': 'EICHERMOT.NS',
    # IT & Tech
    'tcs': 'TCS.NS', 'tata consultancy': 'TCS.NS',
    'infy': 'INFY.NS', 'infosys': 'INFY.NS',
    'wipro': 'WIPRO.NS', 'hcl': 'HCLTECH.NS', 'hcl tech': 'HCLTECH.NS',
    'tech mahindra': 'TECHM.NS', 'techm': 'TECHM.NS', 'ltim': 'LTIM.NS', 'mindtree': 'LTIM.NS',
    # FMCG & Consumer
    'hul': 'HINDUNILVR.NS', 'unilever': 'HINDUNILVR.NS', 'hindustan unilever': 'HINDUNILVR.NS',
    'itc': 'ITC.NS', 'nestle': 'NESTLEIND.NS', 'britannia': 'BRITANNIA.NS',
    'asian paints': 'ASIANPAINT.NS', 'asian paint': 'ASIANPAINT.NS',
    'titan': 'TITAN.NS', 'pidilite': 'PIDILITIND.NS', 'fevicol': 'PIDILITIND.NS',
    'dabur': 'DABUR.NS', 'marico': 'MARICO.NS',
    # Pharma
    'sun pharma': 'SUNPHARMA.NS', 'dr reddy': 'DRREDDY.NS', 'cipla': 'CIPLA.NS',
    'divis': 'DIVISLAB.NS', 'apollo hospital': 'APOLLOHOSP.NS', 'zydus': 'ZYDUSLIFE.NS', 'cadila': 'ZYDUSLIFE.NS',
    # Energy, Metals & PSU
    'ongc': 'ONGC.NS', 'ntpc': 'NTPC.NS', 'powergrid': 'POWERGRID.NS', 'power grid': 'POWERGRID.NS',
    'coal india': 'COALINDIA.NS', 'bpcl': 'BPCL.NS', 'ioc': 'IOC.NS', 'indian oil': 'IOC.NS',
    'gail': 'GAIL.NS', 'sail': 'SAIL.NS', 'bhel': 'BHEL.NS', 'bel': 'BEL.NS', 'hal': 'HAL.NS',
    'irctc': 'IRCTC.NS', 'irfc': 'IRFC.NS', 'pfc': 'PFC.NS', 'rec': 'RECLTD.NS',
    'jsw steel': 'JSWSTEEL.NS', 'hindalco': 'HINDALCO.NS', 'vedanta': 'VEDL.NS',
    'adani ent': 'ADANIENT.NS', 'adani ports': 'ADANIPORTS.NS', 'adani power': 'ADANIPOWER.NS',
    # New-Age Tech & IPOs
    'zomato': 'ETERNAL.NS', 'eternal': 'ETERNAL.NS',
    'paytm': 'PAYTM.NS', 'swiggy': 'SWIGGY.NS', 'ola': 'OLAELEC.NS', 'ola electric': 'OLAELEC.NS',
    'nykaa': 'NYKAA.NS', 'policybazaar': 'POLICYBZR.NS', 'delhivery': 'DELHIVERY.NS',
    'hyundai': 'HYUNDAI.NS', 'waaree': 'WAAREEENER.NS', 'waaree energies': 'WAAREEENER.NS',
    'premier energies': 'PREMIERENE.NS', 'afcons': 'AFCONS.NS', 'ntpc green': 'NTPCGREEN.NS',
    # Core Benchmark Indices
    'nifty': '^NSEI', 'nifty 50': '^NSEI', 'nifty50': '^NSEI',
    'sensex': '^BSESN', 'bse sensex': '^BSESN',
    'bank nifty': '^NSEBANK', 'banknifty': '^NSEBANK', 'nifty bank': '^NSEBANK',
    'nifty it': '^CNXIT', 'it index': '^CNXIT',
    'nifty auto': '^CNXAUTO', 'nifty pharma': '^CNXPHARMA', 'nifty metal': '^CNXMETAL',
    'nifty 500': '^CRSLDX',
    # ETFs & Commodities
    'gold etf': 'GOLDBEES.NS', 'gold bees': 'GOLDBEES.NS', 'goldbees': 'GOLDBEES.NS', 'gold': 'GOLDBEES.NS',
    'silver etf': 'SILVERBEES.NS', 'silver bees': 'SILVERBEES.NS', 'silverbees': 'SILVERBEES.NS', 'silver': 'SILVERBEES.NS',
    'nifty etf': 'NIFTYBEES.NS', 'nifty bees': 'NIFTYBEES.NS', 'niftybees': 'NIFTYBEES.NS',
    'bank etf': 'BANKBEES.NS', 'bank bees': 'BANKBEES.NS', 'bankbees': 'BANKBEES.NS',
    'it etf': 'ITBEES.NS', 'it bees': 'ITBEES.NS', 'itbees': 'ITBEES.NS',
    'junior bees': 'JUNIORBEES.NS', 'juniorbees': 'JUNIORBEES.NS', 'nifty next 50 etf': 'JUNIORBEES.NS',
    'nasdaq': 'MON100.NS', 'nasdaq etf': 'MON100.NS', 'nasdaq 100': 'MON100.NS', 'mon100': 'MON100.NS',
    'fang': 'MAFANG.NS', 'faang': 'MAFANG.NS', 'fang etf': 'MAFANG.NS', 'mafang': 'MAFANG.NS',
    'cpse': 'CPSEETF.NS', 'psu etf': 'CPSEETF.NS', 'cpse etf': 'CPSEETF.NS',
    'liquid etf': 'LIQUIDBEES.NS', 'liquid bees': 'LIQUIDBEES.NS',
    # Global Mega-Caps
    'apple': 'AAPL', 'nvidia': 'NVDA', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
    'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
    'sp500': 'SPY', 's&p 500': 'SPY', 'spy': 'SPY', 'qqq': 'QQQ'
}

# Prominence scores for high-liquidity / bluechip assets
BLUECHIP_PROMINENCE = {
    'RELIANCE.NS': 100, 'TCS.NS': 99, 'HDFCBANK.NS': 98, 'INFY.NS': 97, 'ICICIBANK.NS': 96,
    'SBIN.NS': 95, 'BHARTIARTL.NS': 94, 'ITC.NS': 93, 'LT.NS': 92, 'HINDUNILVR.NS': 91,
    'TMCV.NS': 90, 'TATASTEEL.NS': 89, 'TATAPOWER.NS': 88, 'BAJFINANCE.NS': 87,
    'M&M.NS': 86, 'MARUTI.NS': 85, 'SUNPHARMA.NS': 84, 'KOTAKBANK.NS': 83, 'AXISBANK.NS': 82,
    'NTPC.NS': 81, 'POWERGRID.NS': 80, 'ONGC.NS': 79, 'COALINDIA.NS': 78, 'TITAN.NS': 77,
    'NIFTYBEES.NS': 95, 'BANKBEES.NS': 94, 'GOLDBEES.NS': 93, 'SILVERBEES.NS': 92, 'ITBEES.NS': 91,
    'MON100.NS': 90, 'CPSEETF.NS': 89, 'MAFANG.NS': 88, '^NSEI': 100, '^BSESN': 100, '^NSEBANK': 99,
    'SWIGGY.NS': 85, 'ETERNAL.NS': 88, 'WAAREEENER.NS': 86, 'HYUNDAI.NS': 85, 'OLAELEC.NS': 84
}

@router.get("/tickers")
def search_tickers(
    q: str = Query("", description="Query string to search tickers"),
    sector: str = Query("", description="Filter by sector name"),
    limit: int = Query(30, description="Max results", le=2000)
):
    """
    Intelligent search across all 7,954 instruments with:
    - Exact symbol match & BSE scrip code lookup
    - Curated financial aliases & acronyms
    - Multi-token name & symbol matching
    - Bluechip prominence weighting (Nifty 50 prioritized over penny stocks)
    - Fuzzy typo tolerance for misspellings
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
    q_clean = q_lower.replace(".ns", "").replace(".bo", "").replace("^", "").replace("-", "").replace("&", "")
    q_tokens = [w for w in re.split(r'[\s\-_&]+', q_lower) if len(w) > 0]

    scored = []

    # 1. Alias lookup
    alias_target = SEARCH_ALIASES.get(q_lower) or SEARCH_ALIASES.get(q_clean)
    if alias_target:
        found = next((t for t in ticker_list if t['symbol'] == alias_target), None)
        if found:
            scored.append((1000, found))

    for t in ticker_list:
        if sector and t.get("sector", "").lower() != sector.lower():
            continue

        sym = t.get("symbol", "")
        name = t.get("name", "")
        bse_code = t.get("bse_code", "")

        sym_clean = sym.lower().replace(".ns", "").replace(".bo", "").replace("^", "").replace("-", "").replace("&", "")
        name_lower = name.lower()

        # Match by BSE code
        if bse_code and q_lower == bse_code:
            scored.append((950, t))
            continue

        prominence = BLUECHIP_PROMINENCE.get(sym, 0)
        score = 0

        # Exact symbol match
        if sym_clean == q_clean:
            score = 850 + prominence
        # Exact start of symbol
        elif sym_clean.startswith(q_clean):
            score = 650 + prominence - len(sym_clean)
        # Substring in symbol
        elif q_clean in sym_clean:
            score = 450 + prominence - len(sym_clean)
        # Multi-token match across name/symbol
        elif q_tokens and all(tok in name_lower or tok in sym_clean for tok in q_tokens):
            score = 350 + prominence
        # Name starts with query
        elif name_lower.startswith(q_lower):
            score = 280 + prominence
        # Name contains full query substring
        elif q_lower in name_lower:
            score = 200 + prominence
        # Fuzzy match for typos (if query is at least 4 chars)
        elif len(q_clean) >= 4:
            sym_ratio = SequenceMatcher(None, q_clean, sym_clean).ratio()
            if sym_ratio >= 0.78:
                score = int(180 * sym_ratio) + prominence
            else:
                first_name_word = name_lower.split()[0] if name_lower else ''
                word_ratio = SequenceMatcher(None, q_clean, first_name_word).ratio()
                if word_ratio >= 0.80:
                    score = int(150 * word_ratio) + prominence

        # Bonus for NSE primary listing vs BSE dual-listing
        if score > 0 and sym.endswith('.NS'):
            score += 15

        if score > 0:
            scored.append((score, t))

    # Sort strictly by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate symbols while preserving rank order
    deduped = []
    seen = set()
    for _, item in scored:
        s = item["symbol"]
        if s not in seen:
            seen.add(s)
            deduped.append(item)
            if len(deduped) >= limit:
                break

    return {"tickers": deduped, "total": len(deduped)}



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
