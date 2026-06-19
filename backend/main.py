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



# ─────────────────────────────────────────────────────────────────────────────
# /api/backtest  — Signal-based backtesting engine
# Strategy: RSI(14) + MACD crossover with regime filter
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/backtest")
def run_backtest(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. HDFCBANK.NS"),
    period: str = Query("2y", description="Lookback period: 1y, 2y, 5y"),
    initial_capital: float = Query(100000, description="Starting capital in INR"),
):
    """
    Simulates a RSI+MACD momentum strategy on historical data.
    Returns per-trade log, equity curve vs Nifty benchmark, and aggregate stats.
    """
    try:
        ticker_clean = ticker.strip().upper()

        df = get_history(ticker_clean, period=period)
        if df is None or df.empty or len(df) < 60:
            raise HTTPException(status_code=400, detail="Insufficient historical data")

        close = df["Close"].copy()
        high  = df["High"].copy()
        low   = df["Low"].copy()

        # ── Technical indicators ──────────────────────────────────────────
        # RSI(14)
        delta    = close.diff()
        gain     = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        loss     = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        rsi      = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

        # MACD(12,26,9)
        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd     = ema12 - ema26
        signal   = macd.ewm(span=9, adjust=False).mean()

        # ATR(14) for stop-loss width
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        # ── Signal generation ─────────────────────────────────────────────
        # BUY  : RSI crosses above 35 from below AND MACD > Signal
        # SELL : RSI crosses below 65 from above OR  MACD < Signal
        positions = pd.Series(0, index=close.index)
        in_trade  = False
        entry_px  = 0.0
        stop_loss = 0.0

        for i in range(1, len(close)):
            prev_rsi = rsi.iloc[i - 1]
            curr_rsi = rsi.iloc[i]
            macd_bull = macd.iloc[i] > signal.iloc[i]
            macd_bear = macd.iloc[i] < signal.iloc[i]

            if not in_trade:
                # Entry: RSI recovering from oversold + MACD bullish
                if prev_rsi < 35 and curr_rsi >= 35 and macd_bull:
                    in_trade = True
                    entry_px  = float(close.iloc[i])
                    stop_loss = entry_px - 2.0 * float(atr.iloc[i])
                    positions.iloc[i] = 1
                else:
                    positions.iloc[i] = 0
            else:
                curr_px = float(close.iloc[i])
                # Exit: RSI overbought, MACD bearish, or stop-loss triggered
                if curr_rsi >= 65 or macd_bear or curr_px < stop_loss:
                    in_trade = False
                    positions.iloc[i] = 0
                else:
                    positions.iloc[i] = 1

        # ── Returns calculation ───────────────────────────────────────────
        daily_returns    = close.pct_change().fillna(0)
        strategy_returns = positions.shift(1).fillna(0) * daily_returns

        equity           = (1 + strategy_returns).cumprod() * initial_capital
        bh_equity        = (1 + daily_returns).cumprod() * initial_capital   # buy-and-hold

        # Nifty benchmark
        try:
            nifty_raw    = get_history("^NSEI", period=period)
            nifty_r      = nifty_raw["Close"].pct_change().fillna(0)
            # align
            common       = strategy_returns.index.intersection(nifty_r.index)
            nifty_equity = (1 + nifty_r.loc[common]).cumprod() * initial_capital
        except Exception:
            nifty_equity = None

        # ── Aggregate stats ───────────────────────────────────────────────
        total_return   = float((equity.iloc[-1] / initial_capital - 1) * 100)
        bh_return      = float((bh_equity.iloc[-1] / initial_capital - 1) * 100)

        ann_factor     = 252
        strat_ann      = float(strategy_returns.mean() * ann_factor * 100)
        strat_vol      = float(strategy_returns.std() * np.sqrt(ann_factor) * 100)
        rf_daily       = 0.065 / 252
        sharpe         = float((strategy_returns - rf_daily).mean() / (strategy_returns.std() + 1e-9) * np.sqrt(ann_factor))

        # Max drawdown
        roll_max       = equity.cummax()
        drawdown       = (equity - roll_max) / roll_max
        max_dd         = float(drawdown.min() * 100)
        calmar         = (strat_ann / abs(max_dd)) if abs(max_dd) > 0 else 0.0

        # Trade-level analysis
        trades         = []
        in_t           = False
        t_entry_date   = None
        t_entry_px     = 0.0

        for i in range(1, len(positions)):
            if not in_t and positions.iloc[i] == 1 and positions.iloc[i - 1] == 0:
                in_t         = True
                t_entry_date = str(close.index[i].date())
                t_entry_px   = float(close.iloc[i])
            elif in_t and positions.iloc[i] == 0 and positions.iloc[i - 1] == 1:
                in_t         = False
                exit_date    = str(close.index[i].date())
                exit_px      = float(close.iloc[i])
                ret          = (exit_px - t_entry_px) / t_entry_px * 100
                trades.append({
                    "entry_date": t_entry_date,
                    "exit_date":  exit_date,
                    "entry_price": round(t_entry_px, 2),
                    "exit_price":  round(exit_px,    2),
                    "return_pct":  round(ret, 2),
                    "result":     "WIN" if ret > 0 else "LOSS",
                })

        wins      = [t for t in trades if t["result"] == "WIN"]
        losses    = [t for t in trades if t["result"] == "LOSS"]
        win_rate  = (len(wins) / len(trades) * 100) if trades else 0
        avg_win   = float(np.mean([t["return_pct"] for t in wins]))   if wins   else 0.0
        avg_loss  = float(np.mean([t["return_pct"] for t in losses])) if losses else 0.0
        profit_factor = (
            abs(sum(t["return_pct"] for t in wins) / sum(t["return_pct"] for t in losses))
            if losses and sum(t["return_pct"] for t in losses) != 0 else float("inf")
        )

        # ── Equity curve series (sampled for payload size) ────────────────
        def _curve(series, label):
            sampled = series.resample("W").last().dropna() if len(series) > 200 else series
            return [
                {"date": str(d.date()), "value": round(float(v), 2), "label": label}
                for d, v in sampled.items()
            ]

        strategy_curve = _curve(equity,    "Strategy")
        bh_curve       = _curve(bh_equity, "Buy & Hold")
        nifty_curve    = _curve(nifty_equity, "Nifty 50") if nifty_equity is not None else []

        return {
            "ticker": ticker_clean,
            "period": period,
            "strategy": "RSI(14) + MACD Crossover + ATR Stop-Loss",
            "stats": {
                "initial_capital":  round(initial_capital, 2),
                "final_value":      round(float(equity.iloc[-1]), 2),
                "total_return_pct": round(total_return, 2),
                "bh_return_pct":    round(bh_return, 2),
                "alpha":            round(total_return - bh_return, 2),
                "annualized_return": round(strat_ann, 2),
                "annualized_vol":   round(strat_vol, 2),
                "sharpe_ratio":     round(sharpe, 3),
                "max_drawdown_pct": round(max_dd, 2),
                "calmar_ratio":     round(calmar, 3),
                "total_trades":     len(trades),
                "win_rate_pct":     round(win_rate, 1),
                "avg_win_pct":      round(avg_win, 2),
                "avg_loss_pct":     round(avg_loss, 2),
                "profit_factor":    round(min(profit_factor, 99.9), 2),
            },
            "equity_curves": {
                "strategy": strategy_curve,
                "buy_and_hold": bh_curve,
                "nifty": nifty_curve,
            },
            "trades": trades[-30:],   # Last 30 trades to cap payload
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")



# ─────────────────────────────────────────────────────────────────────────────
# /api/portfolio-analyze  — Multi-stock portfolio analytics
# ─────────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel
from typing import List, Optional
from capital_allocator import allocate_capital

class Holding(BaseModel):
    ticker: str
    qty: float
    buy_price: float

class PortfolioRequest(BaseModel):
    holdings: List[Holding]

class CapitalAllocatorRequest(BaseModel):
    holdings: List[Holding]
    floating_capital: float          # INR available to invest
    horizon_days: int                # Investment horizon in days
    mode: Optional[str] = "recovery"  # "recovery" or "market_buys"
    max_stock_price: Optional[float] = None
    sector: Optional[str] = None

@app.post("/api/portfolio-analyze")
def analyze_portfolio(req: PortfolioRequest):
    """
    Accepts a list of holdings and returns:
    - Per-holding: live price, P&L, return%, current value
    - Portfolio: total value, total P&L, allocation weights
    - Risk: covariance-matrix VaR (95%/99%), portfolio Sharpe, max drawdown
    - Correlation matrix across holdings
    - 1Y return history for each ticker (for equity curve)
    """
    if not req.holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")
    if len(req.holdings) > 15:
        raise HTTPException(status_code=400, detail="Max 15 holdings supported")

    # ── Step 1: fetch live prices ──────────────────────────────────────────
    holdings_out = []
    total_cost   = 0.0
    total_value  = 0.0

    for h in req.holdings:
        tk = h.ticker.strip().upper()
        try:
            q = get_quote(tk)
            live_px = float(q.get("price") or h.buy_price)
        except Exception:
            live_px = h.buy_price

        cost        = h.qty * h.buy_price
        curr_val    = h.qty * live_px
        pnl         = curr_val - cost
        pnl_pct     = (pnl / cost * 100) if cost > 0 else 0.0

        holdings_out.append({
            "ticker":      tk,
            "qty":         h.qty,
            "buy_price":   round(h.buy_price, 2),
            "live_price":  round(live_px, 2),
            "cost":        round(cost, 2),
            "curr_value":  round(curr_val, 2),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl_pct, 2),
        })
        total_cost  += cost
        total_value += curr_val

    total_pnl     = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    # ── Step 2: allocation weights ────────────────────────────────────────
    for h in holdings_out:
        h["weight_pct"] = round(h["curr_value"] / total_value * 100, 2) if total_value > 0 else 0.0

    # ── Step 3: 1-year returns for risk calculations ──────────────────────
    returns_map  = {}
    history_map  = {}

    for h in holdings_out:
        try:
            df = get_history(h["ticker"], period="1y")
            if df is not None and not df.empty and len(df) > 30:
                r = df["Close"].pct_change().dropna()
                returns_map[h["ticker"]]  = r
                # Weekly-sampled close for equity curve (cap payload)
                wk = df["Close"].resample("W").last().dropna()
                history_map[h["ticker"]] = [
                    {"date": str(d.date()), "price": round(float(v), 2)}
                    for d, v in wk.items()
                ]
        except Exception:
            pass

    # ── Step 4: portfolio-level risk ──────────────────────────────────────
    portfolio_risk = {}
    if len(returns_map) >= 2:
        try:
            tickers_in  = list(returns_map.keys())
            ret_df      = pd.DataFrame(returns_map).dropna()
            weights     = np.array([
                next(h["weight_pct"] for h in holdings_out if h["ticker"] == t) / 100
                for t in tickers_in
            ])

            cov_matrix  = ret_df.cov().values
            port_var    = float(weights @ cov_matrix @ weights)
            port_std    = float(np.sqrt(port_var))

            # VaR (parametric, normal assumption)
            z95, z99    = 1.645, 2.326
            var_95      = round(-z95 * port_std * total_value, 2)
            var_99      = round(-z99 * port_std * total_value, 2)
            ann_vol     = round(port_std * np.sqrt(252) * 100, 2)

            # Portfolio daily return series
            port_ret    = ret_df @ weights
            rf_daily    = 0.065 / 252
            sharpe      = float((port_ret.mean() - rf_daily) / (port_ret.std() + 1e-9) * np.sqrt(252))

            # Max drawdown on weighted portfolio
            cum         = (1 + port_ret).cumprod()
            roll_max    = cum.cummax()
            dd          = (cum - roll_max) / roll_max
            max_dd      = round(float(dd.min() * 100), 2)

            # Correlation matrix
            corr        = ret_df.corr().round(3)
            corr_list   = [
                {"a": a, "b": b, "corr": float(corr.loc[a, b])}
                for a in corr.index for b in corr.columns
            ]

            portfolio_risk = {
                "ann_volatility_pct":  ann_vol,
                "var_95_rupees":       var_95,
                "var_99_rupees":       var_99,
                "sharpe_ratio":        round(sharpe, 3),
                "max_drawdown_pct":    max_dd,
                "correlation_pairs":   corr_list,
            }
        except Exception as e:
            portfolio_risk = {"error": str(e)}
    elif len(returns_map) == 1:
        try:
            tk = list(returns_map.keys())[0]
            r  = returns_map[tk]
            rf = 0.065 / 252
            sharpe = float((r.mean() - rf) / (r.std() + 1e-9) * np.sqrt(252))
            vol    = round(float(r.std() * np.sqrt(252) * 100), 2)
            var95  = round(-1.645 * float(r.std()) * total_value, 2)
            portfolio_risk = {
                "ann_volatility_pct": vol,
                "var_95_rupees":      var95,
                "var_99_rupees":      round(-2.326 * float(r.std()) * total_value, 2),
                "sharpe_ratio":       round(sharpe, 3),
                "max_drawdown_pct":   None,
                "correlation_pairs":  [],
            }
        except Exception:
            portfolio_risk = {}

    return {
        "holdings":       holdings_out,
        "summary": {
            "total_cost":      round(total_cost, 2),
            "total_value":     round(total_value, 2),
            "total_pnl":       round(total_pnl, 2),
            "total_pnl_pct":   round(total_pnl_pct, 2),
            "num_holdings":    len(holdings_out),
        },
        "risk":           portfolio_risk,
        "price_history":  history_map,
        "as_of":          datetime.now().isoformat(),
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ─────────────────────────────────────────────────────────────────────────────
# /api/portfolio-insight — Recovery Advisor
# Per holding: RSI signal + news sentiment + recommendation + avg-down calc
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/portfolio-insight")
def portfolio_insight(req: PortfolioRequest):
    """
    For each holding returns:
    - RSI-based signal (OVERSOLD / NEUTRAL / OVERBOUGHT)
    - News sentiment score
    - Recommendation: AVERAGE_DOWN / HOLD_MONITOR / CUT_LOSS / BOOK_PROFIT
    - Break-even metrics and averaging-down calculator
    Portfolio-level: total capital to average down, priority action list
    """
    insights = []

    for h in req.holdings:
        tk = h.ticker.strip().upper()

        # ── Live price ─────────────────────────────────────────────────────
        try:
            q       = get_quote(tk)
            live_px = float(q.get("price") or h.buy_price)
        except Exception:
            live_px = h.buy_price

        pnl_pct = (live_px - h.buy_price) / h.buy_price * 100
        in_loss = pnl_pct < -0.5   # >0.5% loss to avoid noise

        # ── RSI(14) from 3-month history ───────────────────────────────────
        rsi_val = None
        signal  = "NEUTRAL"
        try:
            df = get_history(tk, period="3mo")
            if df is not None and len(df) > 20:
                delta = df["Close"].diff()
                gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
                loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
                rsi   = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
                rsi_val = round(float(rsi.iloc[-1]), 1)
                if rsi_val < 35:
                    signal = "OVERSOLD"
                elif rsi_val > 65:
                    signal = "OVERBOUGHT"
        except Exception:
            pass

        # ── News sentiment ─────────────────────────────────────────────────
        sentiment = 0.0
        try:
            nd = get_advanced_news_analysis(tk, max_articles=5)
            sentiment = float(nd.get("aggregate", {}).get("avg_sentiment", 0))
        except Exception:
            pass

        # ── Recovery metrics ───────────────────────────────────────────────
        gain_to_breakeven = round(((h.buy_price / live_px) - 1) * 100, 2) if in_loss else 0.0

        # Averaging down: buy same qty again at current price
        avg_down_info = None
        if in_loss:
            add_qty        = h.qty
            new_avg        = (h.qty * h.buy_price + add_qty * live_px) / (h.qty + add_qty)
            add_cost       = add_qty * live_px
            new_gain_to_be = round(((new_avg / live_px) - 1) * 100, 2)
            pct_reduction  = round(((h.buy_price - new_avg) / h.buy_price) * 100, 2)
            avg_down_info  = {
                "add_qty":             add_qty,
                "add_cost":            round(add_cost, 2),
                "new_avg_price":       round(new_avg, 2),
                "new_gain_to_breakeven_pct": new_gain_to_be,
                "avg_cost_reduction_pct":    pct_reduction,
            }

        # ── Recommendation logic ───────────────────────────────────────────
        if in_loss:
            if signal == "OVERSOLD" and sentiment >= 0.0:
                rec        = "AVERAGE_DOWN"
                rec_label  = "Average Down"
                rec_color  = "emerald"
                rec_reason = (
                    f"RSI at {rsi_val} (oversold) with "
                    + ("positive" if sentiment > 0.1 else "neutral")
                    + " news sentiment. Technically the stock is near a potential reversal — buying more lowers your average cost."
                )
            elif signal == "OVERBOUGHT" or sentiment < -0.25:
                rec        = "CUT_LOSS"
                rec_label  = "Cut Loss"
                rec_color  = "rose"
                rec_reason = (
                    ("RSI overbought despite price being below cost — unusual, momentum may not support a bounce. " if signal == "OVERBOUGHT" else "")
                    + ("Negative news sentiment suggests further downside pressure. " if sentiment < -0.25 else "")
                    + "Consider exiting to redeploy capital into stronger opportunities."
                )
            else:
                rec        = "HOLD_MONITOR"
                rec_label  = "Hold & Monitor"
                rec_color  = "amber"
                rec_reason = (
                    f"Mixed signals (RSI: {rsi_val}, sentiment: {round(sentiment,2)}). "
                    + "No clear catalyst for recovery yet. Hold and wait for RSI to drop below 35 or sentiment to improve before adding."
                )
        else:
            if signal == "OVERBOUGHT" and pnl_pct > 15:
                rec        = "BOOK_PROFIT"
                rec_label  = "Book Partial Profit"
                rec_color  = "indigo"
                rec_reason = (
                    f"You're up {round(pnl_pct,1)}% and RSI is overbought at {rsi_val}. "
                    + "Consider booking 30–50% to lock in gains while leaving room for further upside."
                )
            else:
                rec        = "STAY_INVESTED"
                rec_label  = "Stay Invested"
                rec_color  = "emerald"
                rec_reason = (
                    f"Profitable position with RSI at {rsi_val}. "
                    + ("Positive sentiment supports continued holding." if sentiment > 0 else "Monitor sentiment for any negative shifts.")
                )

        insights.append({
            "ticker":               tk,
            "live_price":           round(live_px, 2),
            "buy_price":            h.buy_price,
            "qty":                  h.qty,
            "pnl_pct":              round(pnl_pct, 2),
            "in_loss":              in_loss,
            "rsi":                  rsi_val,
            "signal":               signal,
            "news_sentiment":       round(sentiment, 3),
            "gain_to_breakeven_pct": gain_to_breakeven,
            "avg_down":             avg_down_info,
            "recommendation":       rec,
            "rec_label":            rec_label,
            "rec_color":            rec_color,
            "rec_reason":           rec_reason,
        })

    # ── Portfolio-level summary ────────────────────────────────────────────
    loss_items   = [i for i in insights if i["in_loss"]]
    profit_items = [i for i in insights if not i["in_loss"]]
    sentiments   = [i["news_sentiment"] for i in insights]
    avg_sent     = round(float(np.mean(sentiments)), 3) if sentiments else 0.0
    total_avg_down_capital = round(
        sum(i["avg_down"]["add_cost"] for i in insights if i.get("avg_down")), 2
    )
    priority = sorted(loss_items, key=lambda x: x["pnl_pct"])[:3]

    return {
        "insights": insights,
        "portfolio_summary": {
            "total_holdings":           len(insights),
            "in_loss":                  len(loss_items),
            "in_profit":                len(profit_items),
            "avg_portfolio_sentiment":  avg_sent,
            "sentiment_label":          "Positive" if avg_sent > 0.1 else "Negative" if avg_sent < -0.1 else "Neutral",
            "total_capital_to_avg_down": total_avg_down_capital,
            "priority_actions":         priority,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# /api/capital-allocate — Smart Capital Allocation Engine
# Takes floating money + investment horizon → ranked allocation plan
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/capital-allocate")
def capital_allocate(req: CapitalAllocatorRequest):
    """
    Given the user's floating (spare) capital and investment horizon,
    scores every losing position in the portfolio and returns a
    prioritised, confidence-weighted allocation plan telling the user
    exactly how many shares to buy of each stock and why.

    Factors used:
    - RSI(14) signal (oversold ↑ score)
    - 3-month news sentiment
    - 1-month price momentum
    - Loss depth vs horizon suitability
    - Volatility fit for the given time window
    """
    if req.mode != "market_buys" and not req.holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")
    if req.holdings and len(req.holdings) > 15:
        raise HTTPException(status_code=400, detail="Max 15 holdings supported")
    if req.floating_capital <= 0:
        raise HTTPException(status_code=400, detail="floating_capital must be > 0")
    if req.horizon_days <= 0:
        raise HTTPException(status_code=400, detail="horizon_days must be > 0")

    try:
        holdings_dicts = [
            {"ticker": h.ticker.strip().upper(), "qty": h.qty, "buy_price": h.buy_price}
            for h in req.holdings
        ] if req.holdings else []
        result = allocate_capital(
            holdings=holdings_dicts,
            floating_capital=req.floating_capital,
            horizon_days=req.horizon_days,
            mode=req.mode or "recovery",
            max_stock_price=req.max_stock_price,
            sector=req.sector,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capital allocation failed: {str(e)}")
