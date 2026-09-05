import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from pydantic import BaseModel
from scipy.optimize import minimize
from fastapi import APIRouter, Query, HTTPException, Request

from yf_client import get_quote, get_history
from news_intelligence import get_advanced_news_analysis
from capital_allocator import allocate_capital
from services.ticker_manager import SECTOR_MAP

router = APIRouter(prefix="/api", tags=["portfolio"])

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


def _safe_float(val, default=0.0, ndigits=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, ndigits) if ndigits is not None else f
    except (TypeError, ValueError):
        return default


@router.get("/portfolio-metrics")
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

        returns = hist['Close'].pct_change().dropna()
        
        # Value at Risk (VaR) calculations
        var_95 = _safe_float(returns.quantile(0.05) * 100, default=0.0, ndigits=2)
        var_99 = _safe_float(returns.quantile(0.01) * 100, default=0.0, ndigits=2)
        
        # Expected Shortfall (Conditional VaR)
        tail_95_series = returns[returns <= returns.quantile(0.05)]
        tail_99_series = returns[returns <= returns.quantile(0.01)]
        es_95 = _safe_float(tail_95_series.mean() * 100 if len(tail_95_series) > 0 else var_95, default=var_95, ndigits=2)
        es_99 = _safe_float(tail_99_series.mean() * 100 if len(tail_99_series) > 0 else var_99, default=var_99, ndigits=2)
        
        # Volatility metrics
        daily_vol = _safe_float(returns.std() * 100, default=0.0, ndigits=2)
        annual_vol = _safe_float(daily_vol * (252 ** 0.5), default=0.0, ndigits=2)
        
        # Skewness and Kurtosis
        skewness = _safe_float(returns.skew(), default=0.0, ndigits=3)
        kurtosis = _safe_float(returns.kurtosis(), default=0.0, ndigits=3)
        
        # Maximum Drawdown calculation
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = _safe_float(drawdown.min() * 100, default=0.0, ndigits=2)
        
        # Adaptive benchmark & risk-free rate (India vs US/Global)
        is_indian = ticker_clean.endswith('.NS') or ticker_clean.endswith('.BO')
        bench_symbol = '^NSEI' if is_indian else '^GSPC'
        bench_name   = 'NIFTY 50' if is_indian else 'S&P 500'
        rf_rate      = 0.065 if is_indian else 0.045
        rf_daily     = rf_rate / 252.0

        # Beta calculation (vs dynamic benchmark)
        try:
            bench_raw = get_history(bench_symbol, period='1y')
            if bench_raw is None or bench_raw.empty:
                raise ValueError("Empty benchmark data")
            bench_close   = bench_raw['Close']
            bench_returns = bench_close.pct_change().dropna()
            
            # Timezone-safe date alignment
            stock_idx = returns.index.tz_localize(None) if hasattr(returns.index, 'tz') and returns.index.tz else returns.index
            bench_idx = bench_returns.index.tz_localize(None) if hasattr(bench_returns.index, 'tz') and bench_returns.index.tz else bench_returns.index
            
            stock_series = pd.Series(returns.values, index=stock_idx)
            bench_series = pd.Series(bench_returns.values, index=bench_idx)

            common_dates = stock_series.index.intersection(bench_series.index)
            if len(common_dates) > 50:
                stock_aligned = stock_series.loc[common_dates]
                bench_aligned = bench_series.loc[common_dates]
                
                covariance     = stock_aligned.cov(bench_aligned)
                bench_variance = bench_aligned.var()
                beta = covariance / bench_variance if bench_variance != 0 else None
                correlation = stock_aligned.corr(bench_aligned)
            else:
                beta = None
                correlation = None
        except Exception:
            beta = None
            correlation = None
        
        # Sharpe Ratio (using market-appropriate risk-free rate)
        excess_returns = returns - rf_daily
        ret_std = float(returns.std())
        sharpe_ratio = _safe_float(float(excess_returns.mean() / (ret_std + 1e-9) * (252 ** 0.5)), default=0.0, ndigits=3) if ret_std != 0 else 0.0
        
        # Information Ratio & Tracking Error (vs Benchmark)
        if beta is not None and correlation is not None:
            raw_tracking_error = (stock_aligned - bench_aligned).std() * (252 ** 0.5)
            active_return = (stock_aligned.mean() - bench_aligned.mean()) * 252
            tracking_error = _safe_float(raw_tracking_error, default=None, ndigits=2)
            information_ratio = _safe_float(active_return / (raw_tracking_error + 1e-9), default=None, ndigits=3)
        else:
            information_ratio = None
            tracking_error = None
        
        # Complete Closed-Form Black-Scholes Options Pricing Suite (30D ATM)
        try:
            from scipy.stats import norm
            import math
            
            S = float(hist['Close'].iloc[-1])   # Spot price
            K = S                                # At-the-money strike
            T = 30.0 / 365.0                     # 30-day horizon in years
            r = rf_rate                          # Risk-free rate
            sigma = max(annual_vol / 100.0, 0.05) # Annualized volatility
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            put_price  = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Analytical Options Greeks
            delta_call = norm.cdf(d1)
            delta_put  = norm.cdf(d1) - 1.0
            gamma      = norm.pdf(d1) / (S * sigma * math.sqrt(T))
            # Daily Theta decay
            theta_call = -(S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0
            theta_put  = -(S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(-d2)) / 365.0
            # Vega per 1% move in implied volatility
            vega       = S * norm.pdf(d1) * math.sqrt(T) / 100.0
            # Rho per 1% change in interest rates
            rho_call   = K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
            rho_put    = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0
            
            options_data = {
                "call_price":         _safe_float(call_price, default=0.0, ndigits=2),
                "put_price":          _safe_float(put_price, default=0.0, ndigits=2),
                "delta":              _safe_float(delta_call, default=0.0, ndigits=3),
                "delta_put":          _safe_float(delta_put, default=0.0, ndigits=3),
                "gamma":              _safe_float(gamma, default=0.0, ndigits=4),
                "theta":              _safe_float(theta_call, default=0.0, ndigits=3),
                "theta_put":          _safe_float(theta_put, default=0.0, ndigits=3),
                "vega":               _safe_float(vega, default=0.0, ndigits=3),
                "rho_call":           _safe_float(rho_call, default=0.0, ndigits=4),
                "rho_put":            _safe_float(rho_put, default=0.0, ndigits=4),
                "implied_volatility": _safe_float(sigma * 100, default=0.0, ndigits=2),
                "risk_free_rate_pct": _safe_float(r * 100, default=0.0, ndigits=2),
                "benchmark_used":     bench_name,
                "moneyness":          "ATM",
                "currency_symbol":    "₹" if is_indian else "$"
            }
        except Exception as e:
            options_data = {"error": f"Options pricing failed: {str(e)}"}
        
        return {
            "ticker": ticker_clean,
            "currency_symbol": "₹" if is_indian else "$",
            "benchmark_name": bench_name,
            "risk_metrics": {
                "var_95_daily": var_95,
                "var_99_daily": var_99,
                "expected_shortfall_95": es_95,
                "expected_shortfall_99": es_99,
                "daily_volatility": daily_vol,
                "annual_volatility": annual_vol,
                "max_drawdown": max_drawdown,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "sharpe_ratio": sharpe_ratio
            },
            "market_metrics": {
                "beta": _safe_float(beta, default=None, ndigits=3),
                "correlation": _safe_float(correlation, default=None, ndigits=3),
                "correlation_with_nifty": _safe_float(correlation, default=None, ndigits=3),
                "benchmark_name": bench_name,
                "information_ratio": _safe_float(information_ratio, default=None, ndigits=3),
                "tracking_error": _safe_float(tracking_error, default=None, ndigits=2)
            },
            "options_pricing": options_data,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio metrics calculation failed: {str(e)}")


@router.post("/portfolio-analyze")
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
        company_name = tk
        high_52w = None
        low_52w = None
        price_date = ""
        try:
            q = get_quote(tk)
            live_px = float(q.get("price") or h.buy_price)
            company_name = q.get("longName") or tk
            high_52w = q.get("fiftyTwoWeekHigh")
            low_52w = q.get("fiftyTwoWeekLow")
            price_date = q.get("price_date") or ""
        except Exception:
            live_px = h.buy_price

        cost        = h.qty * h.buy_price
        curr_val    = h.qty * live_px
        pnl         = curr_val - cost
        pnl_pct     = (pnl / cost * 100) if cost > 0 else 0.0

        raw_symbol  = tk.replace(".NS", "").replace(".BO", "")
        sector      = SECTOR_MAP.get(raw_symbol, "Other")
        is_holding_us = not tk.endswith(".NS") and not tk.endswith(".BO") and not tk.startswith("^")
        curr_sym    = "$" if is_holding_us else "₹"

        holdings_out.append({
            "ticker":        tk,
            "company_name":  company_name,
            "qty":           h.qty,
            "buy_price":     round(h.buy_price, 2),
            "live_price":    round(live_px, 2),
            "cost":          round(cost, 2),
            "curr_value":    round(curr_val, 2),
            "pnl":           round(pnl, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "sector":        sector,
            "currency_symbol": curr_sym,
            "high_52w":      high_52w,
            "low_52w":       low_52w,
            "price_date":    price_date
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
        if h["high_52w"] is None:
            h["high_52w"] = h["live_price"]
        if h["low_52w"] is None:
            h["low_52w"] = h["live_price"]
        try:
            df = get_history(h["ticker"], period="1y")
            if df is not None and not df.empty and len(df) > 30:
                if h["high_52w"] == h["live_price"]:
                    h["high_52w"] = round(float(df["High"].max()), 2)
                if h["low_52w"] == h["live_price"]:
                    h["low_52w"]  = round(float(df["Low"].min()), 2)
                r = df["Close"].pct_change().dropna()
                returns_map[h["ticker"]]  = r
                # Weekly-sampled close for equity curve
                wk = df["Close"].resample("W").last().dropna()
                history_map[h["ticker"]] = [
                    {"date": str(d.date()), "price": round(float(v), 2)}
                    for d, v in wk.items()
                ]
        except Exception:
            pass

    # Fetch Nifty 50 benchmark history
    nifty_df = None
    try:
        nifty_df = get_history("^NSEI", period="1y")
        if nifty_df is not None and not nifty_df.empty and len(nifty_df) > 30:
            wk = nifty_df["Close"].resample("W").last().dropna()
            history_map["NIFTY50"] = [
                {"date": str(d.date()), "price": round(float(v), 2)}
                for d, v in wk.items()
            ]
    except Exception as e:
        print(f"[PORTFOLIO] Failed to fetch Nifty benchmark: {e}")

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

            # Beta calculation relative to Nifty 50
            beta = 1.0
            try:
                if nifty_df is not None and not nifty_df.empty:
                    nifty_ret = nifty_df["Close"].pct_change().dropna()
                    combined = pd.DataFrame({"port": port_ret, "nifty": nifty_ret}).dropna()
                    if len(combined) > 10:
                        cov = float(combined.cov().values[0, 1])
                        nifty_var = float(combined["nifty"].var())
                        beta = round(cov / (nifty_var + 1e-9), 3)
            except Exception as e:
                print(f"[PORTFOLIO] Failed to compute Beta: {e}")

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

            is_portfolio_us = all(not (h.ticker or "").endswith(".NS") and not (h.ticker or "").endswith(".BO") for h in req.holdings) if req.holdings else False
            portfolio_curr_sym = "$" if is_portfolio_us else "₹"

            portfolio_risk = {
                "ann_volatility_pct":  _safe_float(ann_vol),
                "var_95":              _safe_float(var_95),
                "var_99":              _safe_float(var_99),
                "var_95_rupees":       _safe_float(var_95),
                "var_99_rupees":       _safe_float(var_99),
                "sharpe_ratio":        _safe_float(sharpe, 3),
                "max_drawdown_pct":    _safe_float(max_dd),
                "correlation_pairs":   corr_list,
                "beta":                _safe_float(beta, 3),
                "currency_symbol":     portfolio_curr_sym,
            }
        except Exception as e:
            portfolio_risk = {"error": str(e)}
    elif len(returns_map) == 1:
        try:
            is_portfolio_us = all(not (h.ticker or "").endswith(".NS") and not (h.ticker or "").endswith(".BO") for h in req.holdings) if req.holdings else False
            portfolio_curr_sym = "$" if is_portfolio_us else "₹"
            tk = list(returns_map.keys())[0]
            r  = returns_map[tk]
            rf = 0.065 / 252
            sharpe = float((r.mean() - rf) / (r.std() + 1e-9) * np.sqrt(252))
            vol    = round(float(r.std() * np.sqrt(252) * 100), 2)
            var95  = round(-1.645 * float(r.std()) * total_value, 2)
            var99  = round(-2.326 * float(r.std()) * total_value, 2)
            portfolio_risk = {
                "ann_volatility_pct": _safe_float(vol),
                "var_95":             _safe_float(var95),
                "var_99":             _safe_float(var99),
                "var_95_rupees":      _safe_float(var95),
                "var_99_rupees":      _safe_float(var99),
                "sharpe_ratio":       _safe_float(sharpe, 3),
                "max_drawdown_pct":   None,
                "correlation_pairs":  [],
                "currency_symbol":    portfolio_curr_sym,
            }
        except Exception:
            portfolio_risk = {}

    is_portfolio_us = all(not (h.ticker or "").endswith(".NS") and not (h.ticker or "").endswith(".BO") for h in req.holdings) if req.holdings else False
    portfolio_curr_sym = "$" if is_portfolio_us else "₹"

    return {
        "holdings":       holdings_out,
        "summary": {
            "total_cost":          round(total_cost, 2),
            "total_invested":      round(total_cost, 2),
            "total_value":         round(total_value, 2),
            "total_current_value": round(total_value, 2),
            "total_pnl":           round(total_pnl, 2),
            "total_pnl_pct":       round(total_pnl_pct, 2),
            "num_holdings":        len(holdings_out),
            "currency_symbol":     portfolio_curr_sym,
        },
        "risk":           portfolio_risk,
        "price_history":  history_map,
        "as_of":          datetime.now().isoformat(),
    }


@router.post("/portfolio-insight")
def portfolio_insight(req: PortfolioRequest):
    """
    For each holding returns:
    - RSI-based signal (OVERSOLD / NEUTRAL / OVERBOUGHT)
    - News sentiment score
    - Recommendation: AVERAGE_DOWN / HOLD_MONITOR / CUT_LOSS / BOOK_PROFIT
    - Break-even metrics and averaging-down calculator
    """
    insights = []

    for h in req.holdings:
        tk = h.ticker.strip().upper()

        # Live price
        try:
            q       = get_quote(tk)
            live_px = float(q.get("price") or h.buy_price)
        except Exception:
            live_px = h.buy_price

        pnl_pct = ((live_px - h.buy_price) / h.buy_price * 100) if h.buy_price > 0 else 0.0
        in_loss = pnl_pct < -0.5

        # RSI(14) from 3-month history
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

        # News sentiment
        sentiment = 0.0
        try:
            nd = get_advanced_news_analysis(tk)
            # Safe access to overall sentiment score
            if nd and "sentiment" in nd and "overall_sentiment" in nd["sentiment"]:
                sentiment = float(nd["sentiment"]["overall_sentiment"])
        except Exception:
            pass

        # Recovery metrics
        gain_to_breakeven = round(((h.buy_price / live_px) - 1) * 100, 2) if (in_loss and live_px > 0) else 0.0

        # Averaging down: buy same qty again at current price
        avg_down_info = None
        if in_loss and live_px > 0:
            add_qty        = h.qty
            new_avg        = (h.qty * h.buy_price + add_qty * live_px) / (h.qty + add_qty) if (h.qty + add_qty) > 0 else live_px
            add_cost       = add_qty * live_px
            new_gain_to_be = round(((new_avg / live_px) - 1) * 100, 2) if live_px > 0 else 0.0
            pct_reduction  = round(((h.buy_price - new_avg) / h.buy_price) * 100, 2) if h.buy_price > 0 else 0.0
            avg_down_info  = {
                "add_qty":             add_qty,
                "add_cost":            round(add_cost, 2),
                "new_avg_price":       round(new_avg, 2),
                "new_gain_to_breakeven_pct": new_gain_to_be,
                "avg_cost_reduction_pct":    pct_reduction,
            }

        # Recommendation logic
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

        is_holding_us = not tk.endswith(".NS") and not tk.endswith(".BO") and not tk.startswith("^")
        curr_sym      = "$" if is_holding_us else "₹"

        insights.append({
            "ticker":               tk,
            "live_price":           round(live_px, 2),
            "buy_price":            h.buy_price,
            "qty":                  h.qty,
            "pnl_pct":              round(pnl_pct, 2),
            "in_loss":              in_loss,
            "rsi":                  rsi_val,
            "signal":               signal,
            "currency_symbol":      curr_sym,
            "news_sentiment":       round(sentiment, 3),
            "gain_to_breakeven_pct": gain_to_breakeven,
            "avg_down":             avg_down_info,
            "recommendation":       rec,
            "rec_label":            rec_label,
            "rec_color":            rec_color,
            "rec_reason":           rec_reason,
        })

    # Portfolio-level summary
    loss_items   = [i for i in insights if i["in_loss"]]
    profit_items = [i for i in insights if not i["in_loss"]]
    sentiments   = [i["news_sentiment"] for i in insights]
    avg_sent     = round(float(np.mean(sentiments)), 3) if sentiments else 0.0
    total_avg_down_capital = round(
        sum(i["avg_down"]["add_cost"] for i in insights if i.get("avg_down")), 2
    )
    priority = sorted(loss_items, key=lambda x: x["pnl_pct"])[:3]

    is_portfolio_us = all(not i["ticker"].endswith(".NS") and not i["ticker"].endswith(".BO") for i in insights) if insights else False
    portfolio_curr_sym = "$" if is_portfolio_us else "₹"

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
            "currency_symbol":          portfolio_curr_sym,
        },
    }


@router.post("/capital-allocate")
def capital_allocate(req: CapitalAllocatorRequest):
    """
    Given the user's floating (spare) capital and investment horizon,
    scores every losing position in the portfolio and returns a
    prioritised, confidence-weighted allocation plan.
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


@router.post("/portfolio-optimize")
def optimize_portfolio(req: PortfolioRequest):
    """
    Accepts a list of holdings and returns:
    - Expected annual returns, annualized volatility, and Sharpe ratio of current portfolio
    - Expected annual returns, annualized volatility, and Sharpe ratio of Max Sharpe Portfolio
    - Expected annual returns, annualized volatility, and Sharpe ratio of Min Volatility Portfolio
    - A sample of simulated portfolios for the scatter plot
    """
    if not req.holdings:
        raise HTTPException(status_code=400, detail="No holdings provided")
    if len(req.holdings) < 2:
        raise HTTPException(status_code=400, detail="At least 2 holdings are required for optimization")
    if len(req.holdings) > 15:
        raise HTTPException(status_code=400, detail="Max 15 holdings supported")

    returns_dict = {}
    current_prices = {}
    
    for h in req.holdings:
        tk = h.ticker.strip().upper()
        try:
            df = get_history(tk, period="1y")
            if df is not None and not df.empty and len(df) > 30:
                r = df["Close"].pct_change().dropna()
                returns_dict[tk] = r
                current_prices[tk] = float(df["Close"].iloc[-1])
            else:
                raise ValueError(f"Insufficient historical data")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch 1Y history for {tk}: {str(e)}")

    tickers = list(returns_dict.keys())
    if len(tickers) < 2:
        raise HTTPException(status_code=400, detail="Insufficient historical price data for optimization")

    ret_df = pd.DataFrame(returns_dict).dropna()
    if len(ret_df) < 30:
        raise HTTPException(status_code=400, detail="Insufficient overlapping price history between stocks")

    mean_daily_returns = ret_df.mean()
    cov_matrix = ret_df.cov()

    ann_returns = mean_daily_returns * 252
    ann_cov = cov_matrix * 252
    rf = 0.065

    total_val = 0.0
    weights_current = []
    
    for tk in tickers:
        h = next(item for item in req.holdings if item.ticker == tk)
        val = h.qty * current_prices[tk]
        total_val += val
        weights_current.append(val)
        
    if total_val == 0:
        weights_current = np.ones(len(tickers)) / len(tickers)
    else:
        weights_current = np.array(weights_current) / total_val

    def portfolio_performance(w):
        ret = float(np.sum(ann_returns * w))
        variance = float(np.dot(w.T, np.dot(ann_cov, w)))
        vol = float(np.sqrt(max(variance, 1e-8)))
        sharpe = (ret - rf) / vol if vol > 1e-6 else 0.0
        return ret, vol, sharpe

    curr_ret, curr_vol, curr_sharpe = portfolio_performance(weights_current)

    num_portfolios = 1000
    sim_results = []
    all_weights = []
    
    for _ in range(num_portfolios):
        w = np.random.random(len(tickers))
        w = w / np.sum(w)
        all_weights.append(w)
        ret, vol, sharpe = portfolio_performance(w)
        sim_results.append({
            "return_pct": _safe_float(ret * 100.0, default=0.0, ndigits=2),
            "volatility_pct": _safe_float(vol * 100.0, default=0.0, ndigits=2),
            "sharpe": _safe_float(sharpe, default=0.0, ndigits=3)
        })

    all_weights = np.array(all_weights)
    sim_sharpes = np.array([p["sharpe"] for p in sim_results])
    max_sharpe_idx = np.argmax(sim_sharpes)
    w_max_sharpe_sim = all_weights[max_sharpe_idx]
    
    sim_vols = np.array([p["volatility_pct"] for p in sim_results])
    min_vol_idx = np.argmin(sim_vols)
    w_min_vol_sim = all_weights[min_vol_idx]

    def neg_sharpe(w):
        ret, vol, sharpe = portfolio_performance(w)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(len(tickers)))
    init_guess = np.ones(len(tickers)) / len(tickers)

    opt_max_sharpe = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    w_max_sharpe = opt_max_sharpe.x if opt_max_sharpe.success else w_max_sharpe_sim

    def portfolio_vol(w):
        return portfolio_performance(w)[1]

    opt_min_vol = minimize(portfolio_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    w_min_vol = opt_min_vol.x if opt_min_vol.success else w_min_vol_sim

    ms_ret, ms_vol, ms_sharpe = portfolio_performance(w_max_sharpe)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(w_min_vol)

    def make_weights_list(w_arr):
        return [
            {
                "ticker": tk,
                "ticker_short": tk.replace('.NS', '').replace('.BO', ''),
                "weight_pct": _safe_float(w_arr[i] * 100.0, default=0.0, ndigits=2)
            }
            for i, tk in enumerate(tickers)
        ]

    sampled_sim = random.sample(sim_results, 250) if len(sim_results) > 250 else sim_results

    return {
        "tickers": tickers,
        "current": {
            "return_pct": _safe_float(curr_ret * 100.0, default=0.0, ndigits=2),
            "volatility_pct": _safe_float(curr_vol * 100.0, default=0.0, ndigits=2),
            "sharpe": _safe_float(curr_sharpe, default=0.0, ndigits=3),
            "weights": make_weights_list(weights_current)
        },
        "max_sharpe": {
            "return_pct": _safe_float(ms_ret * 100.0, default=0.0, ndigits=2),
            "volatility_pct": _safe_float(ms_vol * 100.0, default=0.0, ndigits=2),
            "sharpe": _safe_float(ms_sharpe, default=0.0, ndigits=3),
            "weights": make_weights_list(w_max_sharpe)
        },
        "min_volatility": {
            "return_pct": _safe_float(mv_ret * 100.0, default=0.0, ndigits=2),
            "volatility_pct": _safe_float(mv_vol * 100.0, default=0.0, ndigits=2),
            "sharpe": _safe_float(mv_sharpe, default=0.0, ndigits=3),
            "weights": make_weights_list(w_min_vol)
        },
        "simulated_portfolios": sampled_sim
    }


@router.get("/monte-carlo")
def get_monte_carlo(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. HDFCBANK.NS"),
    horizon_days: int = Query(30, description="Horizon in trading days: 30, 60, 90, 252, 756, 1260"),
    simulations: int = Query(1000, description="Number of simulation paths"),
    simulation_mode: str = Query("gbm", description="Mode: 'gbm' (Geometric Brownian Motion with Student-t fat tails) or 'bootstrap' (Historical Bootstrap)"),
):
    """
    Simulates future price paths using Geometric Brownian Motion (GBM) with Student-t fat-tail shocks
    and Bayesian drift shrinkage, or Non-Parametric Historical Bootstrapping.
    """
    try:
        # Safely unwrap parameters if called directly as Python function without FastAPI DI
        ticker_clean = ticker.strip().upper()
        h_days = int(horizon_days.default if hasattr(horizon_days, 'default') else horizon_days)
        sim_count = int(simulations.default if hasattr(simulations, 'default') else simulations)
        mode_val = str(simulation_mode.default if hasattr(simulation_mode, 'default') else simulation_mode)

        mode_clean = mode_val.strip().lower()
        if mode_clean not in ["gbm", "bootstrap"]:
            mode_clean = "gbm"

        ALLOWED = [30, 60, 90, 252, 756, 1260]
        N = min(ALLOWED, key=lambda x: abs(x - h_days))
        M = max(100, min(sim_count, 5000))

        history_period = "5y" if N >= 252 else "1y"
        df = get_history(ticker_clean, period=history_period)
        if df is None or df.empty or len(df) < 30:
            if history_period == "5y":
                df = get_history(ticker_clean, period="2y")
                if df is None or df.empty or len(df) < 30:
                    df = get_history(ticker_clean, period="1y")
            if df is None or df.empty or len(df) < 30:
                raise HTTPException(status_code=400, detail="Insufficient historical price data")

        close_prices = df["Close"].dropna()
        last_price = float(close_prices.iloc[-1])
        last_date = close_prices.index[-1]

        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        if len(log_returns) < 10:
            raise HTTPException(status_code=400, detail="Insufficient returns data for volatility calculation")

        mu_daily_raw = float(log_returns.mean())
        sigma_daily = float(log_returns.std())

        if sigma_daily <= 0:
            sigma_daily = 0.001

        # Annualized raw metrics
        raw_ann_drift = mu_daily_raw * 252.0
        ann_volatility = sigma_daily * np.sqrt(252.0)

        # ── Bayesian Drift Shrinkage ───────────────────────────────────────────
        # Combats estimation risk: 1Y/5Y sample means have high standard errors.
        # We shrink the sample mean towards a long-term equity equilibrium prior
        # (12% annualized return). Shorter horizons retain more momentum; longer
        # horizons shrink more heavily towards equilibrium to prevent runaway compounding.
        market_prior_annual = 0.12
        prior_daily = np.log(1.0 + market_prior_annual) / 252.0

        # Shrinkage factor alpha in [0.15, 0.75] depending on horizon length
        alpha = float(np.clip(N / 1260.0, 0.15, 0.75))
        mu_daily_shrunk = (1.0 - alpha) * mu_daily_raw + alpha * prior_daily
        ann_drift_shrunk = mu_daily_shrunk * 252.0

        N = int(N)
        M = int(M)
        paths = np.zeros((M, N + 1))
        paths[:, 0] = last_price

        np.random.seed(42)  # For reproducible institutional comparisons

        if mode_clean == "bootstrap":
            # ── Non-Parametric Historical Bootstrap ─────────────────────────────
            # Resamples with replacement from the empirical log-returns distribution,
            # naturally preserving real-world fat tails, skewness, and jump shocks.
            hist_rets_arr = log_returns.values
            for t in range(1, N + 1):
                rand_indices = np.random.randint(0, len(hist_rets_arr), size=M)
                step_returns = hist_rets_arr[rand_indices]
                paths[:, t] = paths[:, t - 1] * np.exp(step_returns)
        else:
            # ── Parametric GBM with Student-t Fat Tails (nu = 5) ───────────────
            # Mathematical Fix: mu_daily_shrunk is ALREADY the mean of log returns.
            # No double subtraction of 0.5 * sigma^2!
            # Student-t with df=5 has variance df/(df-2) = 5/3.
            # Multiplying by sqrt(3/5) standardizes variance to exactly 1.0.
            df_t = 5
            scale_t = np.sqrt((df_t - 2.0) / df_t)
            for t in range(1, N + 1):
                t_shocks = np.random.standard_t(df=df_t, size=M) * scale_t
                paths[:, t] = paths[:, t - 1] * np.exp(mu_daily_shrunk + sigma_daily * t_shocks)

        # Percentiles over time for the fan chart
        percentiles = {}
        for pct in [2.5, 25.0, 50.0, 75.0, 97.5]:
            percentiles[pct] = np.percentile(paths, pct, axis=0)

        # Calendar generation for trading days
        future_dates = []
        curr_date = last_date
        while len(future_dates) < N:
            curr_date += timedelta(days=1)
            if curr_date.weekday() < 5:  # Mon-Fri
                future_dates.append(curr_date)

        if N >= 756:
            step = 21  # monthly
        elif N >= 252:
            step = 5   # weekly
        else:
            step = 1   # daily

        hist_data = []
        hist_len = min(30, len(close_prices))
        hist_subset = close_prices.iloc[-hist_len:]
        for d, p in hist_subset.items():
            hist_data.append({
                "date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                "close": round(float(p), 2),
                "is_simulated": False
            })

        sim_data = []
        sim_data.append({
            "date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date)[:10],
            "p025": round(last_price, 2),
            "p250": round(last_price, 2),
            "p500": round(last_price, 2),
            "p750": round(last_price, 2),
            "p975": round(last_price, 2),
            "is_simulated": True
        })

        for t in range(step, N + 1, step):
            d = future_dates[t - 1]
            sim_data.append({
                "date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                "p025": round(float(percentiles[2.5][t]), 2),
                "p250": round(float(percentiles[25.0][t]), 2),
                "p500": round(float(percentiles[50.0][t]), 2),
                "p750": round(float(percentiles[75.0][t]), 2),
                "p975": round(float(percentiles[97.5][t]), 2),
                "is_simulated": True
            })

        sample_paths = []
        num_paths_to_send = min(10, M)
        for i in range(num_paths_to_send):
            path_points = []
            path_points.append({
                "date": str(last_date.date()) if hasattr(last_date, "date") else str(last_date)[:10],
                "price": round(last_price, 2)
            })
            for t in range(step, N + 1, step):
                d = future_dates[t - 1]
                path_points.append({
                    "date": str(d.date()) if hasattr(d, "date") else str(d)[:10],
                    "price": round(float(paths[i, t]), 2)
                })
            sample_paths.append(path_points)

        horizon_prices = paths[:, -1]
        horizon_returns_pct = ((horizon_prices / last_price) - 1.0) * 100.0
        losses_pct = -horizon_returns_pct  # positive value indicates a loss

        # ── Institutional Tail Risk Metrics (VaR & CVaR) ────────────────────────
        # 95% VaR: 95% of outcomes have loss <= var_95
        var_95 = float(np.percentile(losses_pct, 95.0))
        var_99 = float(np.percentile(losses_pct, 99.0))
        
        tail_95 = losses_pct[losses_pct >= var_95]
        cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

        # Maximum Drawdown across each path
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (paths - running_max) / running_max
        max_dd_per_path = np.min(drawdowns, axis=1) * 100.0  # negative percentage
        avg_max_drawdown = float(np.mean(max_dd_per_path))
        worst_drawdown = float(np.min(max_dd_per_path))

        # Horizon probabilities
        prob_up = float(np.sum(horizon_prices > last_price) / M) * 100
        prob_gain_5 = float(np.sum(horizon_prices > last_price * 1.05) / M) * 100
        prob_gain_10 = float(np.sum(horizon_prices > last_price * 1.10) / M) * 100
        prob_gain_20 = float(np.sum(horizon_prices > last_price * 1.20) / M) * 100
        prob_loss_5 = float(np.sum(horizon_prices < last_price * 0.95) / M) * 100
        prob_loss_10 = float(np.sum(horizon_prices < last_price * 0.90) / M) * 100
        prob_loss_20 = float(np.sum(horizon_prices < last_price * 0.80) / M) * 100

        is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
        curr_sym = "₹" if is_indian else "$"

        stats = {
            "current_price": _safe_float(last_price, default=0.0, ndigits=2),
            "expected_price_horizon": _safe_float(float(np.mean(horizon_prices)), default=0.0, ndigits=2),
            "expected_return_pct": _safe_float(((np.mean(horizon_prices) / last_price) - 1.0) * 100.0, default=0.0, ndigits=2),
            "median_price_horizon": _safe_float(float(np.median(horizon_prices)), default=0.0, ndigits=2),
            "max_simulated_price": _safe_float(float(np.max(horizon_prices)), default=0.0, ndigits=2),
            "min_simulated_price": _safe_float(float(np.min(horizon_prices)), default=0.0, ndigits=2),
            "ann_volatility_pct": _safe_float(ann_volatility * 100.0, default=0.0, ndigits=2),
            "ann_drift_pct": _safe_float(ann_drift_shrunk * 100.0, default=0.0, ndigits=2),
            "raw_drift_pct": _safe_float(raw_ann_drift * 100.0, default=0.0, ndigits=2),
            "shrinkage_alpha": _safe_float(alpha, default=0.0, ndigits=2),
            "simulation_mode": mode_clean,
            "var_95_pct": _safe_float(var_95, default=0.0, ndigits=2),
            "var_99_pct": _safe_float(var_99, default=0.0, ndigits=2),
            "cvar_95_pct": _safe_float(cvar_95, default=0.0, ndigits=2),
            "avg_max_drawdown_pct": _safe_float(avg_max_drawdown, default=0.0, ndigits=2),
            "worst_drawdown_pct": _safe_float(worst_drawdown, default=0.0, ndigits=2),
            "prob_up": _safe_float(prob_up, default=0.0, ndigits=2),
            "prob_gain_5": _safe_float(prob_gain_5, default=0.0, ndigits=2),
            "prob_gain_10": _safe_float(prob_gain_10, default=0.0, ndigits=2),
            "prob_gain_20": _safe_float(prob_gain_20, default=0.0, ndigits=2),
            "prob_loss_5": _safe_float(prob_loss_5, default=0.0, ndigits=2),
            "prob_loss_10": _safe_float(prob_loss_10, default=0.0, ndigits=2),
            "prob_loss_20": _safe_float(prob_loss_20, default=0.0, ndigits=2),
            "currency_symbol": curr_sym
        }

        return {
            "ticker": ticker_clean,
            "currency_symbol": curr_sym,
            "horizon_days": N,
            "simulations": M,
            "historical": hist_data,
            "simulated": sim_data,
            "sample_paths": sample_paths,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {str(e)}")
