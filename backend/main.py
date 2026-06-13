import requests
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import os, re
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from engine import analyze_ticker
from ml_models import get_ml_prediction, retrain_model
from news_intelligence import get_advanced_news_analysis
from yf_client import get_quote, get_history, get_info

app = FastAPI(
    title="Stock Analysis Tool API",
    description="Backend API powering the Stock Analysis Dashboard by Vishesh Sanghvi",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Preload the ticker list on startup to avoid delay on first search request."""
    import threading
    threading.Thread(target=_ensure_ticker_list, daemon=True).start()

# Global ticker list — lazily populated on first /api/tickers request
TICKER_LIST = []
_ticker_list_loaded = False


def _ensure_ticker_list():
    """Loads the NSE ticker list on first call. Safe to call multiple times."""
    global TICKER_LIST, _ticker_list_loaded
    if _ticker_list_loaded:
        return
    _ticker_list_loaded = True  # set early to avoid duplicate fetches on concurrent cold starts
    try:
        url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        if res.ok:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                raw_symbol = str(row['SYMBOL']).strip()
                symbol = raw_symbol + ".NS"
                name = str(row['NAME OF COMPANY']).strip()
                sector = SECTOR_MAP.get(raw_symbol, None)
                entry = {"symbol": symbol, "name": name}
                if sector:
                    entry["sector"] = sector
                TICKER_LIST.append(entry)
            print(f"Loaded {len(TICKER_LIST)} NSE tickers successfully.")
        else:
            print("Failed to download NSE tickers, using empty list")
    except Exception as e:
        print(f"Error loading ticker list: {e}")
        _ticker_list_loaded = False  # allow retry on next request

# Security: Restrict CORS origins for production
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow all Vercel preview/production URLs plus localhost
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# In development, always allow localhost
if os.getenv("ENVIRONMENT") == "development" or not os.getenv("ALLOWED_ORIGINS"):
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-yf-client",  # Updated to verify new deployment
        "environment": os.getenv("ENVIRONMENT", "development"),
        "using_yf_client": True
    }

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "name": "StockIQ Pro API",
        "description": "Professional Stock Analysis Platform API",
        "version": "1.0.0",
        "author": "Vishesh Sanghvi",
        "docs": "/docs",
        "health": "/health"
    }

# NSE sector/industry CSV (bhavcopy for sector mapping - best effort)
SECTOR_MAP = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking", "INFY": "IT",
    "ICICIBANK": "Banking", "HINDUNILVR": "FMCG", "ITC": "FMCG", "SBIN": "Banking",
    "BAJFINANCE": "NBFC", "BHARTIARTL": "Telecom", "KOTAKBANK": "Banking",
    "LT": "Infrastructure", "AXISBANK": "Banking", "ASIANPAINT": "FMCG",
    "MARUTI": "Auto", "TITAN": "Consumer", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "ONGC": "Energy", "NTPC": "Power", "POWERGRID": "Power",
    "COALINDIA": "Mining", "JSWSTEEL": "Metals", "TATASTEEL": "Metals",
    "HINDALCO": "Metals", "ADANIENT": "Diversified", "ADANIPORTS": "Logistics",
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "BAJAJFINSV": "NBFC",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "TATAMOTORS": "Auto", "M&M": "Auto", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "DABUR": "FMCG", "GODREJCP": "FMCG", "PIDILITIND": "Chemicals",
    "BERGEPAINT": "FMCG", "INDUSINDBK": "Banking", "FEDERALBNK": "Banking",
    "BANDHANBNK": "Banking", "IDFCFIRSTB": "Banking", "PNB": "Banking",
    "BANKBARODA": "Banking", "CANBK": "Banking", "UNIONBANK": "Banking",
    "HDFCLIFE": "Insurance", "SBILIFE": "Insurance", "ICICIPRULI": "Insurance",
    "MUTHOOTFIN": "NBFC", "CHOLAFIN": "NBFC", "RECLTD": "NBFC", "PFC": "NBFC",
    "ZOMATO": "Consumer Tech", "PAYTM": "Fintech", "NYKAA": "Consumer Tech",
    "POLICYBZR": "Fintech", "DELHIVERY": "Logistics",
}


@app.get("/api/tickers")
def search_tickers(
    q: str = Query("", description="Query string to search tickers"),
    sector: str = Query("", description="Filter by sector name"),
    limit: int = Query(30, description="Max results", le=2000)
):
    """
    Search NSE tickers by symbol or company name.
    Optionally filter by sector. Symbol matches ranked higher than name matches.
    """
    _ensure_ticker_list()

    # Sector-only filter — return all stocks in that sector
    if sector and not q:
        sector_lower = sector.lower()
        results = [t for t in TICKER_LIST if t.get("sector", "").lower() == sector_lower]
        return {"tickers": results[:limit], "total": len(results)}

    if not q:
        return {"tickers": TICKER_LIST[:limit], "total": len(TICKER_LIST)}

    q_lower = q.lower()
    symbol_matches = []
    name_matches = []

    for t in TICKER_LIST:
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


@app.get("/api/sectors")
def get_sectors():
    """
    Returns all stocks from TICKER_LIST grouped by sector.
    Stocks without a known sector are placed in 'Others'.
    Also returns a list of all unique sectors with counts.
    """
    _ensure_ticker_list()

    grouped: dict = {}
    for t in TICKER_LIST:
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
        "total_stocks": len(TICKER_LIST)
    }

@app.get("/api/live")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute
def get_live_price(
    request: Request,
    ticker: str = Query(..., description="Stock ticker, e.g. HDFCBANK.NS", max_length=20)
):
    """
    Returns the most recent price snapshot from yfinance fast_info.
    Yahoo Finance data is ~15 minutes delayed for NSE stocks — suitable for
    a near-live display. Poll this every 30 seconds from the frontend.
    """
    try:
        # Input sanitization
        ticker_clean = ticker.strip().upper()
        
        # Validate ticker format (basic security check)
        import re
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


@app.get("/api/ml-predict")
def get_ml_prediction_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    period: str = Query("2y", description="Training data time period, e.g., 1y, 2y, 5y"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Returns ML-powered price prediction with confidence intervals, diverse stacked ensemble,
    walk-forward stats, and news sentiment fusion.
    """
    try:
        ticker_clean = ticker.strip().upper()
        
        # Resolve sentiment score for fusion (default to neutral 0.0 if fetch fails or no news)
        news_sentiment = 0.0
        try:
            news_res = get_advanced_news_analysis(ticker_clean)
            if news_res and "sentiment" in news_res and "overall_sentiment" in news_res["sentiment"]:
                news_sentiment = float(news_res["sentiment"]["overall_sentiment"])
        except Exception:
            pass

        prediction, error = get_ml_prediction(
            ticker_clean, 
            period=period, 
            start_date=start_date, 
            end_date=end_date, 
            news_sentiment=news_sentiment
        )
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "ticker": ticker_clean,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
            "prediction": prediction,
            "disclaimer": "Predictions are for educational purposes only. Not financial advice."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.post("/api/retrain-model")
def retrain_ml_model(
    ticker: str = Query(..., description="Stock ticker to retrain model for"),
    period: str = Query("2y", description="Training data time period, e.g., 1y, 2y, 5y"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Force retrain the ML model with latest data for improved accuracy.
    """
    try:
        ticker_clean = ticker.strip().upper()
        success, result = retrain_model(ticker_clean, period=period, start_date=start_date, end_date=end_date)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        return {
            "ticker": ticker_clean,
            "period": period,
            "status": "success",
            "metrics": result,
            "message": "Model retrained successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

@app.get("/api/advanced-news")
def get_advanced_news_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    company_name: Optional[str] = Query(None, description="Company name for better news matching")
):
    """
    Returns advanced news intelligence with AI sentiment analysis, 
    breaking news detection, and market impact scoring.
    """
    try:
        ticker_clean = ticker.strip().upper()
        news_analysis = get_advanced_news_analysis(ticker_clean, company_name)
        
        return {
            "ticker": ticker_clean,
            "news_intelligence": news_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")

@app.get("/api/portfolio-metrics")
def get_portfolio_metrics(ticker: str = Query(..., description="Stock ticker symbol")):
    """
    Returns advanced portfolio-grade metrics including options pricing,
    VaR calculations, and correlation analysis.
    """
    try:
        ticker_clean = ticker.strip().upper()

        hist = get_history(ticker_clean, period='1y')
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for ticker")

        returns    = hist['Close'].pct_change().dropna()
        
        # Value at Risk (VaR) calculations
        var_95 = returns.quantile(0.05) * 100  # 95% VaR (daily)
        var_99 = returns.quantile(0.01) * 100  # 99% VaR (daily)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
        es_99 = returns[returns <= returns.quantile(0.01)].mean() * 100
        
        # Volatility metrics
        daily_vol = returns.std() * 100
        annual_vol = daily_vol * (252 ** 0.5)
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Maximum Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Beta calculation (vs NIFTY 50)
        try:
            nifty_raw = get_history('^NSEI', period='1y')
            if nifty_raw.empty:
                raise ValueError("Empty nifty data")
            nifty         = nifty_raw['Close']
            nifty_returns = nifty.pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(nifty_returns.index)
            if len(common_dates) > 50:
                stock_aligned = returns[common_dates]
                nifty_aligned = nifty_returns[common_dates]
                
                covariance = stock_aligned.cov(nifty_aligned)
                nifty_variance = nifty_aligned.var()
                beta = covariance / nifty_variance if nifty_variance != 0 else None
                correlation = stock_aligned.corr(nifty_aligned)
            else:
                beta = None
                correlation = None
        except:
            beta = None
            correlation = None
        
        # Sharpe Ratio (assuming 6.5% risk-free rate)
        risk_free_daily = 0.065 / 252
        excess_returns = returns - risk_free_daily
        sharpe_ratio = excess_returns.mean() / returns.std() * (252 ** 0.5) if returns.std() != 0 else 0
        
        # Information Ratio (vs NIFTY)
        if beta is not None and correlation is not None:
            tracking_error = (stock_aligned - nifty_aligned).std() * (252 ** 0.5)
            active_return = (stock_aligned.mean() - nifty_aligned.mean()) * 252
            information_ratio = active_return / tracking_error if tracking_error != 0 else 0
        else:
            information_ratio = None
            tracking_error = None
        
        # Simple Black-Scholes option pricing (at-the-money call, 30 days)
        try:
            from scipy.stats import norm
            import math
            
            S = hist['Close'].iloc[-1]  # Current stock price
            K = S  # Strike price (at-the-money)
            T = 30/365  # Time to expiration (30 days)
            r = 0.065  # Risk-free rate
            sigma = annual_vol / 100  # Volatility
            
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            call_price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            put_price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
            # Greeks
            delta_call = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            theta_call = -(S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(d2)) / 365
            vega = S * norm.pdf(d1) * math.sqrt(T) / 100
            
            options_data = {
                "call_price": round(call_price, 2),
                "put_price": round(put_price, 2),
                "delta": round(delta_call, 3),
                "gamma": round(gamma, 4),
                "theta": round(theta_call, 3),
                "vega": round(vega, 3),
                "implied_volatility": round(sigma * 100, 2),
                "moneyness": "ATM"
            }
        except Exception as e:
            options_data = {"error": f"Options pricing failed: {str(e)}"}
        
        return {
            "ticker": ticker_clean,
            "risk_metrics": {
                "var_95_daily": round(var_95, 2),
                "var_99_daily": round(var_99, 2),
                "expected_shortfall_95": round(es_95, 2),
                "expected_shortfall_99": round(es_99, 2),
                "daily_volatility": round(daily_vol, 2),
                "annual_volatility": round(annual_vol, 2),
                "max_drawdown": round(max_drawdown, 2),
                "skewness": round(skewness, 3),
                "kurtosis": round(kurtosis, 3),
                "sharpe_ratio": round(sharpe_ratio, 3)
            },
            "market_metrics": {
                "beta": round(beta, 3) if beta is not None else None,
                "correlation_with_nifty": round(correlation, 3) if correlation is not None else None,
                "information_ratio": round(information_ratio, 3) if information_ratio is not None else None,
                "tracking_error": round(tracking_error, 2) if tracking_error is not None else None
            },
            "options_pricing": options_data,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio metrics calculation failed: {str(e)}")

@app.get("/api/analyze")
def get_analysis(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Returns full technical indicators, sentiment, risk, fundamentals, and charting time-series.
    """
    ticker_clean = ticker.strip().upper()
    res = analyze_ticker(ticker_clean, start_date, end_date)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.get("/api/compare")
def compare_peers(
    tickers: str = Query(..., description="Comma-separated list of stock tickers to compare")
):
    """
    Returns comparative basic fundamental data for a group of stocks.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Please provide a valid list of tickers.")

    comparison_results = []
    for ticker in ticker_list:
        try:
            info  = get_info(ticker)
            pe   = info.get('trailingPE')
            peg  = info.get('trailingPegRatio')
            roe  = info.get('returnOnEquity')
            de   = info.get('debtToEquity')
            revg = info.get('revenueGrowth')
            beta = info.get('beta')
            price = info.get('currentPrice') or info.get('regularMarketPrice')

            comparison_results.append({
                'ticker': ticker,
                'currentPrice': price,
                'peRatio': round(pe, 2) if isinstance(pe, (int, float)) else None,
                'pegRatio': round(peg, 2) if isinstance(peg, (int, float)) else None,
                'roe': round(roe * 100, 2) if isinstance(roe, (int, float)) else None,
                'debtToEquity': round(de, 2) if isinstance(de, (int, float)) else None,
                'revenueGrowth': round(revg * 100, 2) if isinstance(revg, (int, float)) else None,
                'beta': round(beta, 2) if isinstance(beta, (int, float)) else None,
            })
        except Exception:
            comparison_results.append({
                'ticker': ticker,
                'error': "Failed to fetch peer data"
            })
    return {"comparison": comparison_results}


# ─────────────────────────────────────────────────────────────────────────────
# Peer Data Helpers
# ─────────────────────────────────────────────────────────────────────────────
from peer_data import get_peers, get_all_sector_members, SECTOR_PEERS


def _compute_quick_metrics(ticker: str) -> dict | None:
    """
    Compute lightweight per-stock metrics without full ML training.
    Uses 1Y price history to derive returns, volatility, RSI, and Sharpe.
    """
    try:
        df = get_history(ticker, period='1y')
        if df is None or df.empty or len(df) < 30:
            return None

        close = df['Close']
        returns = close.pct_change().dropna()

        current_price = float(close.iloc[-1])

        # Returns over multiple horizons
        def safe_ret(n):
            if len(close) > n:
                return round((close.iloc[-1] / close.iloc[-n] - 1) * 100, 2)
            return None

        ret_1m  = safe_ret(22)
        ret_3m  = safe_ret(66)
        ret_6m  = safe_ret(132)
        ret_1y  = safe_ret(len(close) - 1) if len(close) > 5 else None

        # Annualized Volatility
        annual_vol = round(float(returns.std() * np.sqrt(252) * 100), 2)

        # Sharpe (India risk-free ~6.5%)
        rf_daily = 0.065 / 252
        excess = returns - rf_daily
        sharpe = round(float(excess.mean() / excess.std() * np.sqrt(252)), 3) if excess.std() > 0 else 0.0

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        rsi   = round(float(rsi_series.iloc[-1]), 1)


        # 52-week high/low proximity
        high52 = float(close.max())
        low52  = float(close.min())
        pct_from_high = round((current_price - high52) / high52 * 100, 2)

        # ML signal not available in quick metrics (use /api/ml-predict for full ML)
        ml_signal = None
        ml_return = None
        garch_vol = None

        return {

            'ticker':        ticker,
            'current_price': round(current_price, 2),
            'ret_1m':        ret_1m,
            'ret_3m':        ret_3m,
            'ret_6m':        ret_6m,
            'ret_1y':        ret_1y,
            'annual_vol':    annual_vol,
            'sharpe':        sharpe,
            'rsi':           rsi,
            'pct_from_high': pct_from_high,
            'ml_signal':     ml_signal,
            'ml_return':     ml_return,
            'garch_vol':     garch_vol,
        }
    except Exception as e:
        print(f"[PEER] quick_metrics failed for {ticker}: {e}")
        return None


def _sector_composite_score(metrics: dict, all_metrics: list[dict]) -> float:
    """
    Compute a 0–100 composite score for one stock relative to its sector peers.
    Weights: Sharpe (30%), 3M return rank (25%), Low Vol (20%), RSI health (15%), 1Y return (10%)
    """
    def percentile_rank(val, values):
        valid = [v for v in values if v is not None]
        if not valid or val is None:
            return 50.0
        below = sum(1 for v in valid if v < val)
        return round(below / len(valid) * 100, 1)

    sharpes  = [m.get('sharpe')   for m in all_metrics]
    ret3ms   = [m.get('ret_3m')   for m in all_metrics]
    ret1ys   = [m.get('ret_1y')   for m in all_metrics]
    vols     = [m.get('annual_vol') for m in all_metrics]

    sharpe_rank  = percentile_rank(metrics.get('sharpe'),      sharpes)
    ret3m_rank   = percentile_rank(metrics.get('ret_3m'),      ret3ms)
    ret1y_rank   = percentile_rank(metrics.get('ret_1y'),      ret1ys)
    # Lower vol = better → invert
    vol_rank     = 100 - percentile_rank(metrics.get('annual_vol'), vols)

    # RSI health: 40–65 is ideal (momentum without being overbought)
    rsi = metrics.get('rsi') or 50
    if 45 <= rsi <= 65:
        rsi_score = 100
    elif 35 <= rsi < 45 or 65 < rsi <= 75:
        rsi_score = 65
    else:
        rsi_score = 25

    score = (
        sharpe_rank  * 0.30 +
        ret3m_rank   * 0.25 +
        vol_rank     * 0.20 +
        rsi_score    * 0.15 +
        ret1y_rank   * 0.10
    )
    return round(score, 1)


# ─────────────────────────────────────────────────────────────────────────────
# /api/peers  — Returns sector name + suggested peers for a ticker
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/peers")
def get_peers_endpoint(ticker: str = Query(...)):
    try:
        ticker = ticker.upper().strip()
        result = get_peers(ticker)
        return {
            "ticker":  ticker,
            "sector":  result["sector"],
            "peers":   result["peers"],
            "found":   result["found"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /api/peer-compare  — Side-by-side metrics for two stocks
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/peer-compare")
def peer_compare_endpoint(
    ticker: str = Query(...),
    peer:   str = Query(...),
):
    try:
        ticker = ticker.upper().strip()
        peer   = peer.upper().strip()

        m_a = _compute_quick_metrics(ticker)
        m_b = _compute_quick_metrics(peer)

        if not m_a:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        if not m_b:
            raise HTTPException(status_code=404, detail=f"No data for {peer}")

        # Determine winner on each metric
        def winner(key, higher_is_better=True):
            a, b = m_a.get(key), m_b.get(key)
            if a is None or b is None:
                return None
            if higher_is_better:
                return ticker if a > b else peer if b > a else "tie"
            else:
                return ticker if a < b else peer if b < a else "tie"

        winners = {
            'current_price': None,           # N/A — not comparable
            'ret_1m':        winner('ret_1m'),
            'ret_3m':        winner('ret_3m'),
            'ret_6m':        winner('ret_6m'),
            'ret_1y':        winner('ret_1y'),
            'sharpe':        winner('sharpe'),
            'annual_vol':    winner('annual_vol', higher_is_better=False),
            'rsi':           None,           # context-dependent
            'ml_return':     winner('ml_return'),
        }

        peer_info_a = get_peers(ticker)
        peer_info_b = get_peers(peer)

        return {
            "ticker_a":  ticker,
            "ticker_b":  peer,
            "sector_a":  peer_info_a["sector"],
            "sector_b":  peer_info_b["sector"],
            "metrics_a": m_a,
            "metrics_b": m_b,
            "winners":   winners,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# /api/sector-rank  — Ranks all sector peers with composite scores + insights
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/sector-rank")
def sector_rank_endpoint(ticker: str = Query(...)):
    try:
        ticker = ticker.upper().strip()
        peer_info = get_peers(ticker)
        sector    = peer_info["sector"]
        peers     = peer_info["peers"]

        all_tickers = [ticker] + peers

        # Compute metrics for all tickers in parallel-ish
        all_metrics = []
        for t in all_tickers:
            m = _compute_quick_metrics(t)
            if m:
                all_metrics.append(m)

        if not all_metrics:
            raise HTTPException(status_code=503, detail="Could not fetch sector data")

        # Score all
        for m in all_metrics:
            m['score'] = _sector_composite_score(m, all_metrics)

        # Sort by composite score descending
        ranked = sorted(all_metrics, key=lambda x: x['score'], reverse=True)

        # Assign ranks
        for i, m in enumerate(ranked):
            m['rank'] = i + 1

        # Sector insights
        valid = [m for m in ranked if m.get('ret_3m') is not None]
        best_momentum   = max(valid, key=lambda x: x.get('ret_3m', -999))  if valid else None
        best_sharpe     = max(all_metrics, key=lambda x: x.get('sharpe', -999))
        best_ml         = max([m for m in all_metrics if m.get('ml_return') is not None],
                               key=lambda x: x.get('ml_return', -999), default=None)
        lowest_vol      = min(all_metrics, key=lambda x: x.get('annual_vol', 999))

        # Find rank of the queried ticker
        queried_rank = next((m['rank'] for m in ranked if m['ticker'] == ticker), None)
        total        = len(ranked)

        insights = {
            'sector':           sector,
            'total_peers':      total,
            'queried_rank':     queried_rank,
            'best_momentum':    best_momentum['ticker'] if best_momentum else None,
            'best_risk_adj':    best_sharpe['ticker'],
            'best_ml_signal':   best_ml['ticker'] if best_ml else None,
            'lowest_vol':       lowest_vol['ticker'],
        }

        return {
            "ticker":   ticker,
            "sector":   sector,
            "ranked":   ranked,
            "insights": insights,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

