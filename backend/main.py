import requests
import io
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import datetime
import yfinance as yf

# Import analysis engine functions
from engine import analyze_ticker
from ml_models import get_ml_prediction, retrain_model
from news_intelligence import get_advanced_news_analysis

app = FastAPI(
    title="Stock Analysis Tool API",
    description="Backend API powering the Stock Analysis Dashboard by Vishesh Sanghvi",
    version="1.0.0"
)

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
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development")
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
def search_tickers(q: str = Query("", description="Query string to search tickers")):
    """
    Search NSE tickers by symbol or company name.
    Returns top 30 matches. Symbol matches are ranked higher than name matches.
    """
    _ensure_ticker_list()

    if not q:
        return {"tickers": TICKER_LIST[:30]}

    q_lower = q.lower()
    symbol_matches = []
    name_matches = []

    for t in TICKER_LIST:
        sym_lower = t["symbol"].lower().replace(".ns", "")
        name_lower = t["name"].lower()
        if q_lower in sym_lower:
            symbol_matches.append(t)
        elif q_lower in name_lower:
            name_matches.append(t)
        if len(symbol_matches) + len(name_matches) >= 60:
            break

    # Symbol matches first, then name matches, total cap at 30
    combined = symbol_matches[:15] + name_matches[:15]
    return {"tickers": combined[:30]}

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
        
        t = yf.Ticker(ticker_clean)
        fi   = t.fast_info

        price      = getattr(fi, 'last_price', None)
        prev_close = getattr(fi, 'previous_close', None)
        day_high   = getattr(fi, 'day_high', None)
        day_low    = getattr(fi, 'day_low', None)
        volume     = getattr(fi, 'last_volume', None)
        mktcap     = getattr(fi, 'market_cap', None)

        change     = None
        change_pct = None
        if price and prev_close and prev_close != 0:
            change     = round(price - prev_close, 2)
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        return {
            "ticker"     : ticker.strip().upper(),
            "price"      : round(price, 2) if price else None,
            "change"     : change,
            "changePct"  : change_pct,
            "dayHigh"    : round(day_high, 2) if day_high else None,
            "dayLow"     : round(day_low, 2) if day_low else None,
            "volume"     : int(volume) if volume else None,
            "marketCap"  : mktcap,
            "prevClose"  : round(prev_close, 2) if prev_close else None,
            "timestamp"  : datetime.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/ml-predict")
def get_ml_prediction_endpoint(ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS")):
    """
    Returns ML-powered price prediction with confidence intervals.
    Uses Random Forest with technical indicators for next 5-day prediction.
    """
    try:
        ticker_clean = ticker.strip().upper()
        prediction, error = get_ml_prediction(ticker_clean)
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "ticker": ticker_clean,
            "prediction": prediction,
            "disclaimer": "Predictions are for educational purposes only. Not financial advice."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.post("/api/retrain-model")
def retrain_ml_model(ticker: str = Query(..., description="Stock ticker to retrain model for")):
    """
    Force retrain the ML model with latest data for improved accuracy.
    """
    try:
        ticker_clean = ticker.strip().upper()
        success, result = retrain_model(ticker_clean)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        return {
            "ticker": ticker_clean,
            "training_results": result,
            "message": "Model retrained successfully"
        }
        
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
        
        # Fetch stock data
        stock = yf.Ticker(ticker_clean)
        hist = stock.history(period="1y")
        info = stock.info
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data found for ticker")
        
        # Calculate returns
        returns = hist['Close'].pct_change().dropna()
        
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
            nifty = yf.download("^NSEI", period="1y", progress=False)['Close']
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
            info = yf.Ticker(ticker).info
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
