import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from yf_client import get_history, get_quote, get_info
from peer_data import get_peers

router = APIRouter(prefix="/api", tags=["analysis"])

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

def _clean_val(v):
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None

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

        def safe_ret(n):
            if len(close) > n and float(close.iloc[-n]) > 0:
                val = (float(close.iloc[-1]) / float(close.iloc[-n]) - 1.0) * 100.0
                return _safe_float(val, default=None, ndigits=2)
            return None

        ret_1m  = safe_ret(22)
        ret_3m  = safe_ret(66)
        ret_6m  = safe_ret(132)
        ret_1y  = safe_ret(len(close) - 1) if len(close) > 5 else None

        annual_vol = _safe_float(float(returns.std() * np.sqrt(252) * 100), default=0.0, ndigits=2)

        # Sharpe (India risk-free ~6.5%)
        rf_daily = 0.065 / 252
        excess = returns - rf_daily
        sharpe = _safe_float(float(excess.mean() / excess.std() * np.sqrt(252)), default=0.0, ndigits=3) if excess.std() > 0 else 0.0

        # RSI (14)
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        rsi   = _safe_float(float(rsi_series.iloc[-1]), default=50.0, ndigits=1)

        high52 = float(close.max())
        pct_from_high = _safe_float((current_price - high52) / high52 * 100, default=0.0, ndigits=2)

        return {
            'ticker':        ticker,
            'current_price': _safe_float(current_price, default=0.0, ndigits=2),
            'ret_1m':        ret_1m,
            'ret_3m':        ret_3m,
            'ret_6m':        ret_6m,
            'ret_1y':        ret_1y,
            'annual_vol':    annual_vol,
            'sharpe':        sharpe,
            'rsi':           rsi,
            'pct_from_high': pct_from_high,
            'ml_signal':     None,
            'ml_return':     None,
            'garch_vol':     None,
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
    vol_rank     = 100 - percentile_rank(metrics.get('annual_vol'), vols)

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


@router.get("/valuation")
def calculate_dcf(
    ticker: str = Query(..., description="NSE ticker, e.g. INFIBEAM.NS"),
    growth_rate: Optional[float] = Query(None, description="Custom growth rate (decimal, e.g. 0.08)"),
    discount_rate: Optional[float] = Query(None, description="Custom discount rate/WACC (decimal, e.g. 0.10)"),
    terminal_growth: Optional[float] = Query(None, description="Custom terminal growth (decimal, e.g. 0.045)"),
    starting_flow: Optional[float] = Query(None, description="Custom starting cash flow (rupees)")
):
    """
    Computes a comprehensive valuation suite:
    - 10-step DCF Intrinsic Value calculator
    - 10-point Financial Health Checklist score
    - DuPont Analysis decomposition
    - Graham Number calculation
    """
    try:
        ticker_clean = ticker.strip().upper()
        info = get_info(ticker_clean)
        
        if not info:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not find company details for {ticker_clean}. Verify NSE/BSE suffix."
            )

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price is None:
            raise HTTPException(status_code=404, detail=f"No price information found for {ticker_clean}")

        is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO") or info.get("currency") == "INR"
        curr_sym = "₹" if is_indian else ("$" if info.get("currency") == "USD" else (info.get("currency") or "$"))

        # Fetch underlying data fields with NaN/Inf sanitization
        market_cap = _clean_val(info.get("marketCap"))
        shares = _clean_val(info.get("sharesOutstanding")) or 0
        eps = _clean_val(info.get("trailingEps"))
        book_value = _clean_val(info.get("bookValue"))
        pb = _clean_val(info.get("priceToBook"))
        pe = _clean_val(info.get("trailingPE"))
        long_name = info.get("longName") or ticker_clean

        peg = _clean_val(info.get("pegRatio"))
        div_yield = _clean_val(info.get("dividendYield"))
        payout = _clean_val(info.get("payoutRatio"))
        insiders = _clean_val(info.get("heldPercentInsiders"))
        institutions = _clean_val(info.get("heldPercentInstitutions"))
        rev_growth = _clean_val(info.get("revenueGrowth"))
        earn_growth = _clean_val(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
        npm = _clean_val(info.get("profitMargins"))
        opm = _clean_val(info.get("operatingMargins"))
        gpm = _clean_val(info.get("grossMargins"))
        roe = _clean_val(info.get("returnOnEquity"))
        roa = _clean_val(info.get("returnOnAssets"))
        de = _clean_val(info.get("debtToEquity"))
        curr_ratio = _clean_val(info.get("currentRatio"))
        quick_ratio = _clean_val(info.get("quickRatio"))
        cash = _clean_val(info.get("totalCash")) or 0.0
        debt = _clean_val(info.get("totalDebt")) or 0.0
        fcf = _clean_val(info.get("freeCashflow")) or 0.0
        ocf = _clean_val(info.get("operatingCashflow")) or 0.0
        rev = _clean_val(info.get("totalRevenue")) or 0.0
        net_income = _clean_val(info.get("netIncomeToCommon"))

        # Solvency / Debt to Equity handling (ensure ratio format)
        de_ratio = None
        if de is not None:
            if de > 2.0:
                de_ratio = de / 100.0
            else:
                de_ratio = de

        # Graham Number calculation
        graham_number = None
        if eps is not None and book_value is not None and eps > 0 and book_value > 0:
            graham_number = _safe_float(math.sqrt(22.5 * eps * book_value), default=None, ndigits=2)

        # DuPont Analysis Decomposition
        dupont = None
        if roe is not None and npm is not None:
            equity = book_value * shares if (book_value and shares) else None
            if not equity and market_cap and pb:
                equity = market_cap / pb
            
            if equity and equity > 0:
                assets = equity + debt
                if rev and assets > 0:
                    asset_turnover = rev / assets
                elif npm and abs(npm) > 1e-6 and assets > 0:
                    asset_turnover = roe / (npm * (assets / equity))
                else:
                    asset_turnover = None

                equity_multiplier = assets / equity if equity > 0 else None
                
                dupont = {
                    "net_profit_margin": _safe_float(npm * 100, default=None, ndigits=2),
                    "asset_turnover": _safe_float(asset_turnover, default=None, ndigits=3),
                    "equity_multiplier": _safe_float(equity_multiplier, default=None, ndigits=2),
                    "calculated_roe": _safe_float(roe * 100, default=None, ndigits=2)
                }

        # Health Checklist items
        health_checklist = []
        score = 0
        
        # 1. ROE Check
        if roe is not None:
            passed = roe >= 0.12
            score += 1 if passed else 0
            health_checklist.append({"metric": "Return on Equity (ROE)", "value": f"{round(roe*100, 2)}%", "condition": ">= 12%", "passed": passed})
        else:
            health_checklist.append({"metric": "Return on Equity (ROE)", "value": "N/A", "condition": ">= 12%", "passed": False})
            
        # 2. ROA Check
        if roa is not None:
            passed = roa >= 0.05
            score += 1 if passed else 0
            health_checklist.append({"metric": "Return on Assets (ROA)", "value": f"{round(roa*100, 2)}%", "condition": ">= 5%", "passed": passed})
        else:
            health_checklist.append({"metric": "Return on Assets (ROA)", "value": "N/A", "condition": ">= 5%", "passed": False})
            
        # 3. NPM Check
        if npm is not None:
            passed = npm >= 0.08
            score += 1 if passed else 0
            health_checklist.append({"metric": "Net Profit Margin", "value": f"{round(npm*100, 2)}%", "condition": ">= 8%", "passed": passed})
        else:
            health_checklist.append({"metric": "Net Profit Margin", "value": "N/A", "condition": ">= 8%", "passed": False})
            
        # 4. Solvency Check (D/E ratio)
        if de_ratio is not None:
            passed = de_ratio <= 1.0
            score += 1 if passed else 0
            health_checklist.append({"metric": "Debt to Equity Ratio", "value": f"{round(de_ratio, 2)}x", "condition": "<= 1.0x", "passed": passed})
        else:
            health_checklist.append({"metric": "Debt to Equity Ratio", "value": "0.0x (No Debt)", "condition": "<= 1.0x", "passed": True})
            score += 1
            
        # 5. Liquidity Check
        if curr_ratio is not None:
            passed = curr_ratio >= 1.2
            score += 1 if passed else 0
            health_checklist.append({"metric": "Current Ratio", "value": f"{round(curr_ratio, 2)}x", "condition": ">= 1.2x", "passed": passed})
        else:
            health_checklist.append({"metric": "Current Ratio", "value": "N/A", "condition": ">= 1.2x", "passed": False})
            
        # 6. Cash Flow Check (FCF)
        passed_fcf = fcf > 0 or ocf > 0
        score += 1 if passed_fcf else 0
        fcf_val_str = f"{curr_sym}{round(fcf/1e9, 2)}B" if fcf else (f"{curr_sym}{round(ocf/1e9, 2)}B (OCF)" if ocf else "Negative/Zero")
        health_checklist.append({"metric": "Free Cash Flow", "value": fcf_val_str, "condition": "> 0", "passed": passed_fcf})
        
        # 7. Valuation (PE ratio check)
        if pe is not None:
            passed = pe < 30
            score += 1 if passed else 0
            health_checklist.append({"metric": "Price to Earnings (P/E)", "value": f"{round(pe, 1)}x", "condition": "< 30x", "passed": passed})
        else:
            health_checklist.append({"metric": "Price to Earnings (P/E)", "value": "N/A", "condition": "< 30x", "passed": False})
            
        # 8. Insider Ownership (Promoters)
        if insiders is not None:
            passed = insiders >= 0.40
            score += 1 if passed else 0
            health_checklist.append({"metric": "Promoter Holding", "value": f"{round(insiders*100, 1)}%", "condition": ">= 40%", "passed": passed})
        else:
            health_checklist.append({"metric": "Promoter Holding", "value": "N/A", "condition": ">= 40%", "passed": False})
            
        # 9. Revenue Growth (YoY)
        if rev_growth is not None:
            passed = rev_growth >= 0.08
            score += 1 if passed else 0
            health_checklist.append({"metric": "Revenue Growth (YoY)", "value": f"{round(rev_growth*100, 1)}%", "condition": ">= 8%", "passed": passed})
        else:
            health_checklist.append({"metric": "Revenue Growth (YoY)", "value": "N/A", "condition": ">= 8%", "passed": False})
            
        # 10. Earnings Growth (YoY)
        if earn_growth is not None:
            passed = earn_growth >= 0.05
            score += 1 if passed else 0
            health_checklist.append({"metric": "Earnings Growth (YoY)", "value": f"{round(earn_growth*100, 1)}%", "condition": ">= 5%", "passed": passed})
        else:
            health_checklist.append({"metric": "Earnings Growth (YoY)", "value": "N/A", "condition": ">= 5%", "passed": False})

        # WACC default calculation using CAPM: Rf + Beta * ERP
        beta_val = info.get("beta") or 1.0
        calculated_wacc = 0.065 + beta_val * 0.06
        calculated_wacc = max(0.08, min(0.15, calculated_wacc))

        calculated_growth = 0.08
        if rev_growth is not None:
            calculated_growth = max(0.05, min(0.20, rev_growth))

        # Default Cash Flow for DCF
        default_dcf_flow = fcf
        flow_type = "Free Cash Flow"
        if default_dcf_flow <= 0:
            if info.get("netIncomeToCommon") and info.get("netIncomeToCommon") > 0:
                default_dcf_flow = info.get("netIncomeToCommon")
                flow_type = "Net Income"
            elif ocf > 0:
                default_dcf_flow = ocf * 0.7
                flow_type = "70% of Operating Cash Flow"
            elif rev > 0:
                default_dcf_flow = rev * 0.06
                flow_type = "6% of Revenue (Normalized Proxy)"
            else:
                default_dcf_flow = (current_price * shares * 0.04) if shares > 0 else 1000000000
                flow_type = "Estimated 4% Equity Yield"

        return {
            "ticker": ticker_clean,
            "company_name": long_name,
            "current_price": current_price,
            "currency": info.get("currency") or ("INR" if is_indian else "USD"),
            "currency_symbol": curr_sym,
            "market_cap": market_cap,
            "shares_outstanding": shares,
            "book_value": book_value,
            "eps": eps,
            "pe_ratio": pe,
            "pb_ratio": pb,
            "peg_ratio": peg,
            "dividend_yield": div_yield,
            "payout_ratio": payout,
            "held_insiders_pct": insiders,
            "held_institutions_pct": institutions,
            "revenue_growth": rev_growth,
            "earnings_growth": earn_growth,
            "profit_margins": npm,
            "operating_margins": opm,
            "gross_margins": gpm,
            "return_on_equity": roe,
            "return_on_assets": roa,
            "debt_to_equity": de_ratio,
            "current_ratio": curr_ratio,
            "quick_ratio": quick_ratio,
            "total_cash": cash,
            "total_debt": debt,
            "free_cashflow": fcf,
            "operating_cashflow": ocf,
            "net_income": net_income,
            "total_revenue": rev,
            "graham_number": graham_number,
            "dupont": dupont,
            "health_score": score,
            "health_checklist": health_checklist,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "business_summary": info.get("longBusinessSummary"),
            "recommendation_key": info.get("recommendationKey"),
            "recommendation_mean": info.get("recommendationMean"),
            "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "target_median_price": info.get("targetMedianPrice"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "dcf_defaults": {
                "starting_flow": round(default_dcf_flow, 2),
                "flow_type": flow_type,
                "growth_rate": round(calculated_growth, 3),
                "discount_rate": round(calculated_wacc, 3),
                "terminal_growth": 0.045
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Valuation calculation failed: {str(e)}")


@router.get("/analyze")
def get_analysis(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)")
):
    """
    Returns full technical indicators, sentiment, risk, fundamentals, and charting time-series.
    """
    from engine import analyze_ticker
    ticker_clean = ticker.strip().upper()
    res = analyze_ticker(ticker_clean, start_date, end_date)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@router.get("/compare")
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
            is_in = ticker.endswith(".NS") or ticker.endswith(".BO") or info.get("currency") == "INR"
            curr_sym = "₹" if is_in else ("$" if info.get("currency") == "USD" else (info.get("currency") or "$"))

            comparison_results.append({
                'ticker': ticker,
                'currency_symbol': curr_sym,
                'currentPrice': _safe_float(price, default=None, ndigits=2),
                'peRatio': _safe_float(pe, default=None, ndigits=2),
                'pegRatio': _safe_float(peg, default=None, ndigits=2),
                'roe': _safe_float(roe * 100.0 if roe is not None else None, default=None, ndigits=2),
                'debtToEquity': _safe_float(de, default=None, ndigits=2),
                'revenueGrowth': _safe_float(revg * 100.0 if revg is not None else None, default=None, ndigits=2),
                'beta': _safe_float(beta, default=None, ndigits=2),
            })
        except Exception:
            comparison_results.append({
                'ticker': ticker,
                'error': "Failed to fetch peer data"
            })
    return {"comparison": comparison_results}


@router.get("/peers")
def get_peers_endpoint(ticker: str = Query(...)):
    try:
        ticker_clean = ticker.upper().strip()
        result = get_peers(ticker_clean)
        return {
            "ticker":  ticker_clean,
            "sector":  result["sector"],
            "peers":   result["peers"],
            "found":   result["found"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/peer-compare")
def peer_compare_endpoint(
    ticker: str = Query(...),
    peer:   str = Query(...),
):
    try:
        ticker_clean = ticker.upper().strip()
        peer_clean   = peer.upper().strip()

        m_a = _compute_quick_metrics(ticker_clean)
        m_b = _compute_quick_metrics(peer_clean)

        if not m_a:
            raise HTTPException(status_code=404, detail=f"No data for {ticker_clean}")
        if not m_b:
            raise HTTPException(status_code=404, detail=f"No data for {peer_clean}")

        def winner(key, higher_is_better=True):
            a, b = m_a.get(key), m_b.get(key)
            if a is None or b is None:
                return None
            if higher_is_better:
                return ticker_clean if a > b else peer_clean if b > a else "tie"
            else:
                return ticker_clean if a < b else peer_clean if b < a else "tie"

        winners = {
            'current_price': None,
            'ret_1m':        winner('ret_1m'),
            'ret_3m':        winner('ret_3m'),
            'ret_6m':        winner('ret_6m'),
            'ret_1y':        winner('ret_1y'),
            'sharpe':        winner('sharpe'),
            'annual_vol':    winner('annual_vol', higher_is_better=False),
            'rsi':           None,
            'ml_return':     winner('ml_return'),
        }

        peer_info_a = get_peers(ticker_clean)
        peer_info_b = get_peers(peer_clean)

        return {
            "ticker_a":  ticker_clean,
            "ticker_b":  peer_clean,
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


@router.get("/sector-rank")
def sector_rank_endpoint(ticker: str = Query(...)):
    try:
        ticker_clean = ticker.upper().strip()
        peer_info = get_peers(ticker_clean)
        sector    = peer_info["sector"]
        peers     = peer_info["peers"]

        all_tickers = [ticker_clean] + peers

        all_metrics = []
        for t in all_tickers:
            m = _compute_quick_metrics(t)
            if m:
                all_metrics.append(m)

        if not all_metrics:
            raise HTTPException(status_code=503, detail="Could not fetch sector data")

        for m in all_metrics:
            m['score'] = _sector_composite_score(m, all_metrics)

        ranked = sorted(all_metrics, key=lambda x: x['score'], reverse=True)

        for i, m in enumerate(ranked):
            m['rank'] = i + 1

        valid = [m for m in ranked if m.get('ret_3m') is not None]
        best_momentum   = max(valid, key=lambda x: x.get('ret_3m', -999))  if valid else None
        best_sharpe     = max(all_metrics, key=lambda x: x.get('sharpe', -999))
        best_ml         = max([m for m in all_metrics if m.get('ml_return') is not None],
                               key=lambda x: x.get('ml_return', -999), default=None)
        lowest_vol      = min(all_metrics, key=lambda x: x.get('annual_vol', 999))

        queried_rank = next((m['rank'] for m in ranked if m['ticker'] == ticker_clean), None)
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
            "ticker":   ticker_clean,
            "sector":   sector,
            "ranked":   ranked,
            "insights": insights,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backtest")
def run_backtest(
    ticker: str = Query(..., description="Stock ticker symbol, e.g. HDFCBANK.NS"),
    period: str = Query("2y", description="Lookback period: 1y, 2y, 5y"),
    initial_capital: float = Query(100000.0, description="Starting capital in currency units"),
):
    """
    Simulates an institutional-grade Dual-Momentum (RSI + MACD Crossover)
    strategy with Dynamic ATR Trailing Stop-Loss, realistic execution friction,
    and adaptive multi-market benchmark comparison (NIFTY 50 / S&P 500).
    """
    try:
        ticker_clean = ticker.strip().upper()
        # Safe cast for initial_capital when called directly or through FastAPI
        cap = float(initial_capital.default if hasattr(initial_capital, 'default') else initial_capital)
        if cap <= 0:
            cap = 100000.0

        df = get_history(ticker_clean, period=period)
        if df is None or df.empty or len(df) < 60:
            raise HTTPException(status_code=400, detail="Insufficient historical data (need >= 60 trading days)")

        close  = df["Close"].copy()
        high   = df["High"].copy()
        low    = df["Low"].copy()
        open_p = df["Open"].copy() if "Open" in df else close.copy()

        # ── 1. Technical Indicators ────────────────────────────────────
        delta  = close.diff()
        gain   = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        loss   = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        rsi    = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

        ema12  = close.ewm(span=12, adjust=False).mean()
        ema26  = close.ewm(span=26, adjust=False).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist   = macd - signal

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        # ── 2. Trade Execution Simulation ──────────────────────────────
        positions = pd.Series(0, index=close.index)
        in_trade  = False
        entry_px  = 0.0
        stop_loss = 0.0
        t_entry_date = None
        trades    = []

        # 15 bps (0.15%) roundtrip transaction friction (STT/taxes + brokerage + slippage)
        FRICTION_RATE = 0.0015

        for i in range(1, len(close)):
            curr_close = float(close.iloc[i])
            curr_low   = float(low.iloc[i])
            curr_high  = float(high.iloc[i])
            curr_open  = float(open_p.iloc[i])
            curr_rsi   = float(rsi.iloc[i])
            prev_rsi   = float(rsi.iloc[i - 1])
            curr_macd  = float(macd.iloc[i])
            prev_macd  = float(macd.iloc[i - 1])
            curr_sig   = float(signal.iloc[i])
            prev_sig   = float(signal.iloc[i - 1])
            curr_hist  = float(hist.iloc[i])
            prev_hist  = float(hist.iloc[i - 1])
            curr_atr   = float(atr.iloc[i])

            # Dual-Momentum Entry Conditions:
            # 1. Trend Resumption: MACD crosses above Signal Line with RSI in healthy momentum (40 <= RSI <= 68)
            # 2. Mean-Reversion Bounce: RSI exits oversold (<35 to >=35) with expanding MACD histogram momentum
            macd_cross_up = (prev_macd <= prev_sig) and (curr_macd > curr_sig) and (40.0 <= curr_rsi <= 68.0)
            rsi_rebound   = (prev_rsi < 35.0) and (curr_rsi >= 35.0) and (curr_hist > prev_hist)

            if not in_trade:
                if macd_cross_up or rsi_rebound:
                    in_trade     = True
                    entry_px     = curr_close
                    # Dynamic 2.0x ATR initial stop
                    stop_loss    = entry_px - 2.0 * curr_atr
                    t_entry_date = str(close.index[i].date())
                    positions.iloc[i] = 1
                else:
                    positions.iloc[i] = 0
            else:
                # Dynamic ATR Trailing Stop Ratchet (only moves upwards to lock in gains)
                stop_loss = max(stop_loss, curr_close - 2.0 * curr_atr)

                # Exit conditions:
                hit_stop        = curr_low <= stop_loss
                macd_cross_down = (prev_macd >= prev_sig) and (curr_macd < curr_sig)
                rsi_overbought  = curr_rsi >= 70.0

                if hit_stop or macd_cross_down or rsi_overbought:
                    in_trade = False
                    positions.iloc[i] = 0
                    
                    # If stop hit intraday, execute at stop_loss (or open if gapped down below stop)
                    if hit_stop:
                        exit_px = min(curr_open, stop_loss) if curr_open < stop_loss else stop_loss
                        exit_reason = "TRAILING_STOP"
                    elif rsi_overbought:
                        exit_px = curr_close
                        exit_reason = "RSI_OVERBOUGHT"
                    else:
                        exit_px = curr_close
                        exit_reason = "MACD_CROSS_DOWN"

                    net_trade_ret = ((exit_px / entry_px) - 1.0 - FRICTION_RATE) * 100.0
                    trades.append({
                        "entry_date":   t_entry_date,
                        "exit_date":    str(close.index[i].date()),
                        "entry_price":  round(entry_px, 2),
                        "exit_price":   round(exit_px, 2),
                        "return_pct":   round(net_trade_ret, 2),
                        "result":       "WIN" if net_trade_ret > 0 else "LOSS",
                        "exit_reason":  exit_reason
                    })
                else:
                    positions.iloc[i] = 1

        # ── 3. Strategy & Benchmark Equity Curves ──────────────────────
        daily_returns    = close.pct_change().fillna(0)
        strategy_returns = (positions.shift(1).fillna(0) * daily_returns).copy()
        
        # Apply trade friction (half-spread on entry, half-spread on exit)
        trade_entries = (positions == 1) & (positions.shift(1) == 0)
        trade_exits   = (positions == 0) & (positions.shift(1) == 1)
        strategy_returns[trade_entries] -= (FRICTION_RATE / 2.0)
        strategy_returns[trade_exits]   -= (FRICTION_RATE / 2.0)

        equity           = (1.0 + strategy_returns).cumprod() * cap
        bh_equity        = (1.0 + daily_returns).cumprod() * cap

        # Adaptive multi-market benchmark selection
        is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
        benchmark_sym  = "^NSEI" if is_indian else "^GSPC"
        benchmark_name = "NIFTY 50" if is_indian else "S&P 500"
        rf_rate        = 0.065 if is_indian else 0.045
        rf_daily       = rf_rate / 252.0

        try:
            bench_raw    = get_history(benchmark_sym, period=period)
            if bench_raw is not None and not bench_raw.empty:
                bench_r      = bench_raw["Close"].pct_change().fillna(0)
                common       = strategy_returns.index.intersection(bench_r.index)
                bench_equity = (1.0 + bench_r.loc[common]).cumprod() * cap
            else:
                bench_equity = None
        except Exception:
            bench_equity = None

        total_return = _safe_float((equity.iloc[-1] / cap - 1.0) * 100.0, default=0.0, ndigits=2)
        bh_return    = _safe_float((bh_equity.iloc[-1] / cap - 1.0) * 100.0, default=0.0, ndigits=2)

        # ── 4. Institutional Quantitative Performance Metrics ─────────
        n_days = max(len(close), 1)
        years = n_days / 252.0
        # True Compound Annual Growth Rate (CAGR)
        cagr = _safe_float(((equity.iloc[-1] / cap) ** (1.0 / max(years, 0.1)) - 1.0) * 100.0 if equity.iloc[-1] > 0 else -100.0, default=0.0, ndigits=2)
        strat_vol = _safe_float(float(strategy_returns.std() * np.sqrt(252.0) * 100.0), default=0.0, ndigits=2)

        excess_ret = strategy_returns - rf_daily
        strat_std = float(strategy_returns.std())
        sharpe = _safe_float(float((excess_ret.mean() / (strat_std + 1e-9)) * np.sqrt(252.0)), default=0.0, ndigits=3)

        # Sortino Ratio (penalizing only downside deviations)
        downside_returns = excess_ret[excess_ret < 0]
        downside_std = float(downside_returns.std() * np.sqrt(252.0)) if len(downside_returns) > 1 else 1e-9
        sortino = _safe_float(float(excess_ret.mean() * 252.0 / (downside_std + 1e-9)), default=0.0, ndigits=3)

        # Maximum Drawdown and Max Drawdown Duration
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        max_dd   = _safe_float(float(drawdown.min() * 100.0), default=0.0, ndigits=2)
        calmar   = _safe_float((cagr / abs(max_dd)) if abs(max_dd) > 0 else 0.0, default=0.0, ndigits=3)

        # Drawdown duration calculation (consecutive trading days below peak)
        underwater_days = 0
        max_dd_duration = 0
        for dd_val in drawdown:
            if dd_val < 0:
                underwater_days += 1
                max_dd_duration = max(max_dd_duration, underwater_days)
            else:
                underwater_days = 0

        # Trade analytics
        wins   = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        win_rate  = _safe_float((len(wins) / len(trades) * 100.0) if trades else 0.0, default=0.0, ndigits=1)
        avg_win   = _safe_float(np.mean([t["return_pct"] for t in wins]) if wins else 0.0, default=0.0, ndigits=2)
        avg_loss  = _safe_float(np.mean([t["return_pct"] for t in losses]) if losses else 0.0, default=0.0, ndigits=2)
        gross_win_sum  = sum(t["return_pct"] for t in wins)
        gross_loss_sum = abs(sum(t["return_pct"] for t in losses))
        raw_profit_factor = (gross_win_sum / gross_loss_sum) if gross_loss_sum > 0 else (99.9 if gross_win_sum > 0 else 0.0)
        profit_factor = _safe_float(min(raw_profit_factor, 99.9), default=0.0, ndigits=2)

        def _curve(series, label):
            if series is None or series.empty:
                return []
            sampled = series.resample("W").last().dropna() if len(series) > 200 else series
            return [
                {"date": str(d.date()) if hasattr(d, "date") else str(d)[:10], "value": _safe_float(v, default=0.0, ndigits=2), "label": label}
                for d, v in sampled.items()
            ]

        strategy_curve = _curve(equity, "Strategy")
        bh_curve       = _curve(bh_equity, "Buy & Hold")
        bench_curve    = _curve(bench_equity, benchmark_name)

        return {
            "ticker": ticker_clean,
            "period": period,
            "strategy": "Dual-Momentum (RSI + MACD) + Dynamic ATR Trailing Stop",
            "stats": {
                "initial_capital":       _safe_float(cap, default=100000.0, ndigits=2),
                "final_value":           _safe_float(float(equity.iloc[-1]), default=cap, ndigits=2),
                "total_return_pct":      total_return,
                "bh_return_pct":         bh_return,
                "alpha":                 _safe_float(total_return - bh_return, default=0.0, ndigits=2),
                "annualized_return":     cagr,
                "annualized_vol":        strat_vol,
                "sharpe_ratio":          sharpe,
                "sortino_ratio":         sortino,
                "max_drawdown_pct":      max_dd,
                "max_drawdown_days":     int(max_dd_duration),
                "calmar_ratio":          calmar,
                "total_trades":          len(trades),
                "win_rate_pct":          win_rate,
                "avg_win_pct":           avg_win,
                "avg_loss_pct":          avg_loss,
                "profit_factor":         profit_factor,
                "benchmark_name":        benchmark_name,
                "currency_symbol":       "₹" if is_indian else "$",
            },
            "equity_curves": {
                "strategy": strategy_curve,
                "buy_and_hold": bh_curve,
                "nifty": bench_curve,
            },
            "trades": trades[-30:],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
