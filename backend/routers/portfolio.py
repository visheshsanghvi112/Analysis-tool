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

            portfolio_risk = {
                "ann_volatility_pct":  ann_vol,
                "var_95_rupees":       var_95,
                "var_99_rupees":       var_99,
                "sharpe_ratio":        round(sharpe, 3),
                "max_drawdown_pct":    max_dd,
                "correlation_pairs":   corr_list,
                "beta":                beta,
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

        pnl_pct = (live_px - h.buy_price) / h.buy_price * 100
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

    # Portfolio-level summary
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
        ret = np.sum(ann_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))
        sharpe = (ret - rf) / vol if vol > 0 else 0.0
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
            "return_pct": round(float(ret * 100), 2),
            "volatility_pct": round(float(vol * 100), 2),
            "sharpe": round(float(sharpe), 3)
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
                "weight_pct": round(float(w_arr[i] * 100), 2)
            }
            for i, tk in enumerate(tickers)
        ]

    sampled_sim = random.sample(sim_results, 250) if len(sim_results) > 250 else sim_results

    return {
        "tickers": tickers,
        "current": {
            "return_pct": round(float(curr_ret * 100), 2),
            "volatility_pct": round(float(curr_vol * 100), 2),
            "sharpe": round(float(curr_sharpe), 3),
            "weights": make_weights_list(weights_current)
        },
        "max_sharpe": {
            "return_pct": round(float(ms_ret * 100), 2),
            "volatility_pct": round(float(ms_vol * 100), 2),
            "sharpe": round(float(ms_sharpe), 3),
            "weights": make_weights_list(w_max_sharpe)
        },
        "min_volatility": {
            "return_pct": round(float(mv_ret * 100), 2),
            "volatility_pct": round(float(mv_vol * 100), 2),
            "sharpe": round(float(mv_sharpe), 3),
            "weights": make_weights_list(w_min_vol)
        },
        "simulated_portfolios": sampled_sim
    }


@router.get("/monte-carlo")
def get_monte_carlo(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. HDFCBANK.NS"),
    horizon_days: int = Query(30, description="Horizon in trading days: 30, 60, 90, 252, 756, 1260"),
    simulations: int = Query(1000, description="Number of simulation paths"),
):
    """
    Simulates future price paths using Geometric Brownian Motion (GBM).
    """
    try:
        ticker_clean = ticker.strip().upper()

        ALLOWED = [30, 60, 90, 252, 756, 1260]
        N = min(ALLOWED, key=lambda x: abs(x - int(horizon_days)))
        M = max(100, min(int(simulations), 5000))

        history_period = "5y" if N >= 252 else "1y"
        df = get_history(ticker_clean, period=history_period)
        if df is None or df.empty or len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient historical price data")

        close_prices = df["Close"].dropna()
        last_price = float(close_prices.iloc[-1])
        last_date = close_prices.index[-1]

        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        if len(log_returns) < 10:
            raise HTTPException(status_code=400, detail="Insufficient returns data for volatility calculation")

        mu_daily = float(log_returns.mean())
        sigma_daily = float(log_returns.std())

        mu = mu_daily * 252
        sigma = sigma_daily * np.sqrt(252)

        if sigma <= 0:
            sigma = 0.01

        N = int(N)
        M = int(M)
        dt = 1.0 / 252.0

        paths = np.zeros((M, N + 1))
        paths[:, 0] = last_price

        for t in range(1, N + 1):
            Z = np.random.standard_normal(M)
            paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        percentiles = {}
        for pct in [2.5, 25.0, 50.0, 75.0, 97.5]:
            percentiles[pct] = np.percentile(paths, pct, axis=0)

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
        prob_up = float(np.sum(horizon_prices > last_price) / M) * 100
        prob_gain_5 = float(np.sum(horizon_prices > last_price * 1.05) / M) * 100
        prob_gain_10 = float(np.sum(horizon_prices > last_price * 1.10) / M) * 100
        prob_gain_20 = float(np.sum(horizon_prices > last_price * 1.20) / M) * 100
        prob_loss_5 = float(np.sum(horizon_prices < last_price * 0.95) / M) * 100
        prob_loss_10 = float(np.sum(horizon_prices < last_price * 0.90) / M) * 100
        prob_loss_20 = float(np.sum(horizon_prices < last_price * 0.80) / M) * 100

        stats = {
            "current_price": round(last_price, 2),
            "expected_price_horizon": round(float(np.mean(horizon_prices)), 2),
            "expected_return_pct": round(((np.mean(horizon_prices) / last_price) - 1) * 100, 2),
            "median_price_horizon": round(float(np.median(horizon_prices)), 2),
            "max_simulated_price": round(float(np.max(horizon_prices)), 2),
            "min_simulated_price": round(float(np.min(horizon_prices)), 2),
            "ann_volatility_pct": round(sigma * 100, 2),
            "ann_drift_pct": round(mu * 100, 2),
            "prob_up": round(prob_up, 2),
            "prob_gain_5": round(prob_gain_5, 2),
            "prob_gain_10": round(prob_gain_10, 2),
            "prob_gain_20": round(prob_gain_20, 2),
            "prob_loss_5": round(prob_loss_5, 2),
            "prob_loss_10": round(prob_loss_10, 2),
            "prob_loss_20": round(prob_loss_20, 2)
        }

        return {
            "ticker": ticker_clean,
            "horizon_days": N,
            "simulations": M,
            "historical": hist_data,
            "simulated": sim_data,
            "sample_paths": sample_paths,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {str(e)}")
