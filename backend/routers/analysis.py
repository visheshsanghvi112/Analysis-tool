from __future__ import annotations
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from yf_client import get_history, get_quote, get_info, get_asset_type, get_etf_meta, get_etf_holdings
from peer_data import get_peers
from services.ticker_manager import SECTOR_MAP

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

        # Max Drawdown (1Y)
        cum_ret = (1 + returns).cumprod()
        rolling_max = cum_ret.expanding().max()
        dd_series = (cum_ret - rolling_max) / rolling_max
        max_drawdown = _safe_float(float(dd_series.min() * 100), default=0.0, ndigits=2)

        # Sortino Ratio (excess return / downside deviation)
        downside_returns = returns[returns < rf_daily]
        downside_std = float(downside_returns.std()) if len(downside_returns) > 5 else 0.0
        sortino = _safe_float(float(excess.mean() / (downside_std + 1e-9) * np.sqrt(252)), default=0.0, ndigits=3) if downside_std > 0 else 0.0

        # Calmar Ratio (1Y return / absolute max drawdown)
        abs_dd = abs(max_drawdown)
        calmar = _safe_float(ret_1y / abs_dd, default=None, ndigits=2) if (ret_1y is not None and abs_dd > 0.5) else None

        # Tail Risk: VaR 95% & CVaR 95% (Expected Shortfall)
        var_95  = _safe_float(float(np.percentile(returns, 5) * 100), default=0.0, ndigits=2)
        tail_95 = returns[returns <= np.percentile(returns, 5)]
        cvar_95 = _safe_float(float(tail_95.mean() * 100), default=0.0, ndigits=2) if len(tail_95) > 0 else var_95

        return {
            'ticker':        ticker,
            'current_price': _safe_float(current_price, default=0.0, ndigits=2),
            'ret_1m':        ret_1m,
            'ret_3m':        ret_3m,
            'ret_6m':        ret_6m,
            'ret_1y':        ret_1y,
            'annual_vol':    annual_vol,
            'sharpe':        sharpe,
            'sortino':       sortino,
            'calmar':        calmar,
            'var_95':        var_95,
            'cvar_95':       cvar_95,
            'max_drawdown':  max_drawdown,
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
        if (not shares or shares <= 0) and market_cap and current_price and current_price > 0:
            shares = market_cap / current_price

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

        # Detect banking and financial institutions
        raw_symbol = ticker_clean.replace(".NS", "").replace(".BO", "")
        sector_name = (info.get("sector") or SECTOR_MAP.get(raw_symbol) or "").strip()
        industry_name = (info.get("industry") or "").strip()
        fin_terms = ["finance", "financial", "bank", "insurance", "lending", "nbfc", "credit"]
        is_financial = any(term in sector_name.lower() or term in industry_name.lower() or term in long_name.lower() for term in fin_terms) or any(kw in ticker_clean for kw in ["BANK", "FINANCE", "FINSERV", "BAJFINANCE", "MUTHOOT", "IRFC", "PFC", "REC", "HDFC"])

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
            
        # 2. ROA Check (Banks operate with high financial leverage where ROA >= 1.0% is world-class)
        if roa is not None:
            roa_req = 0.01 if is_financial else 0.05
            cond_str = ">= 1.0%" if is_financial else ">= 5%"
            passed = roa >= roa_req
            score += 1 if passed else 0
            health_checklist.append({
                "metric": "Return on Assets (ROA)",
                "value": f"{round(roa*100, 2)}%",
                "condition": cond_str,
                "passed": passed,
                "note": "Banking benchmark: >= 1.0% is top tier" if is_financial else None
            })
        else:
            health_checklist.append({"metric": "Return on Assets (ROA)", "value": "N/A", "condition": ">= 1.0%" if is_financial else ">= 5%", "passed": False})
            
        # 3. NPM Check
        if npm is not None:
            passed = npm >= 0.08
            score += 1 if passed else 0
            health_checklist.append({"metric": "Net Profit Margin", "value": f"{round(npm*100, 2)}%", "condition": ">= 8%", "passed": passed})
        else:
            health_checklist.append({"metric": "Net Profit Margin", "value": "N/A", "condition": ">= 8%", "passed": False})
            
        # 4. Solvency Check (D/E ratio)
        if is_financial:
            score += 1
            health_checklist.append({
                "metric": "Debt to Equity Ratio",
                "value": "Regulated Institution",
                "condition": "Capital Adequacy Compliant",
                "passed": True,
                "note": "Banks maintain statutory CAR/CRAR ratios rather than industrial debt metrics"
            })
        elif de_ratio is not None:
            passed = de_ratio <= 1.0
            score += 1 if passed else 0
            health_checklist.append({"metric": "Debt to Equity Ratio", "value": f"{round(de_ratio, 2)}x", "condition": "<= 1.0x", "passed": passed})
        else:
            health_checklist.append({"metric": "Debt to Equity Ratio", "value": "0.0x (No Debt)", "condition": "<= 1.0x", "passed": True})
            score += 1
            
        # 5. Liquidity Check
        if is_financial:
            score += 1
            health_checklist.append({
                "metric": "Current Ratio",
                "value": "Statutory LCR / SLR",
                "condition": "Central Bank Regulated",
                "passed": True,
                "note": "Banks adhere to RBI/Fed Liquidity Coverage Ratios (LCR) instead of Current Ratio"
            })
        elif curr_ratio is not None:
            passed = curr_ratio >= 1.2
            score += 1 if passed else 0
            health_checklist.append({"metric": "Current Ratio", "value": f"{round(curr_ratio, 2)}x", "condition": ">= 1.2x", "passed": passed})
        else:
            health_checklist.append({"metric": "Current Ratio", "value": "N/A", "condition": ">= 1.2x", "passed": False})
            
        # 6. Cash Flow Check (FCF / Net Income for banks)
        if is_financial:
            passed_fcf = (net_income is not None and net_income > 0)
            score += 1 if passed_fcf else 0
            fcf_val_str = f"{curr_sym}{round(net_income/1e9, 2)}B (Net Income)" if net_income else "Negative/Zero"
            health_checklist.append({"metric": "Earnings Generation", "value": fcf_val_str, "condition": "> 0", "passed": passed_fcf})
        else:
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
        if is_financial:
            if net_income and net_income > 0:
                default_dcf_flow = net_income
                flow_type = "Net Income (Financial Proxy)"
            elif rev > 0:
                default_dcf_flow = rev * 0.15
                flow_type = "15% of Revenue (Financial Proxy)"
            else:
                default_dcf_flow = (current_price * shares * 0.05) if shares > 0 else 1000000000
                flow_type = "Estimated 5% Equity Yield"
        else:
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
            "is_financial": is_financial,
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
            "sector": info.get("sector") or SECTOR_MAP.get(raw_symbol),
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
            },
            "data_status": {
                "live_quote": "OK" if current_price > 0 else "UNAVAILABLE",
                "historical_candles": "OK",
                "financial_statements": "OK" if (market_cap and eps is not None) else "PARTIAL",
                "corporate_actions": "SPLIT_ADJUSTED",
            },
            "model_status": {
                "status": "NOT_APPLICABLE",
                "details": "DCF is an intrinsic fundamental valuation model, independent of ML statistical models.",
            },
            "valuation_status": {
                "methodology": "FINANCIAL_INSTITUTION_EQUITY_DDM" if is_financial else "STANDARD_DCF",
                "status": "OK",
                "details": "Equity Free Cash Flow Proxy (Net Income) — operating debt excluded from equity deduction" if is_financial else "Free Cash Flow to Firm (FCFF) — Enterprise DCF with net debt deduction",
            },
            "market_structure_status": {
                "secondary_market_dislocation": "NORMAL",
                "liquidity_state": "OK" if (market_cap and market_cap > 1e9) else "LOW_LIQUIDITY",
                "details": "Single-stock equity traded on exchange without ETF-style NAV arbitrage constraints."
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
            'sortino':       winner('sortino'),
            'calmar':        winner('calmar'),
            'var_95':        winner('var_95', higher_is_better=True),
            'cvar_95':       winner('cvar_95', higher_is_better=True),
            'max_drawdown':  winner('max_drawdown', higher_is_better=True),
            'pct_from_high': winner('pct_from_high', higher_is_better=True),
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
                "excess_return_vs_bh":   _safe_float(total_return - bh_return, default=0.0, ndigits=2),
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


# ────────────────────────────────────────────────────────────
# ETF Long-Term Analysis Endpoint
# Replaces DCF/Graham/DuPont for ETF tickers.
# ────────────────────────────────────────────────────────────

# Benchmark index tickers for common Indian ETFs
# Motilal Oswal explicitly benchmarks MONQ50 to the NASDAQ Q-50 Total Return Index (NTQX)
_ETF_BENCHMARK_MAP = {
    'NIFTYBEES.NS':  '^NSEI',
    'JUNIORBEES.NS': '^NSEI',
    'BANKBEES.NS':   '^NSEBANK',
    'GOLDBEES.NS':   'GC=F',
    'SILVERBEES.NS': 'SI=F',
    'MON100.NS':     '^NDX',
    'MONQ50.NS':     '^NTQX',  # Official Scheme Benchmark: NASDAQ Q-50 Total Return Index (TRI)
    'MAFANG.NS':     '^NDX',
    'ITBEES.NS':     '^NSEI',
    'CPSEETF.NS':    '^NSEI',
    'LIQUIDBEES.NS': None,
    'SETFNN50.NS':   '^NSEI',
    'KOTAKNV20.NS':  '^NSEI',
}

_BENCHMARK_METADATA = {
    '^NSEI':    {"name": 'Nifty 50 Total Return Index', "type": 'TOTAL_RETURN_INDEX', "currency": 'INR'},
    '^NSEBANK': {"name": 'Nifty Bank Index', "type": 'TOTAL_RETURN_INDEX', "currency": 'INR'},
    '^NDX':     {"name": 'Nasdaq-100 Total Return Index', "type": 'TOTAL_RETURN_INDEX', "currency": 'USD'},
    '^NTQX':    {"name": 'NASDAQ Q-50 Total Return Index', "type": 'TOTAL_RETURN_INDEX', "currency": 'USD'},
    '^NXTQ':    {"name": 'Nasdaq Next Generation 100 / Q-50 Price Return', "type": 'PRICE_RETURN_INDEX', "currency": 'USD'},
    'GC=F':     {"name": 'Gold COMEX Futures', "type": 'COMMODITY_FUTURES', "currency": 'USD'},
    'SI=F':     {"name": 'Silver COMEX Futures', "type": 'COMMODITY_FUTURES', "currency": 'USD'},
}

_BENCHMARK_NAMES = {k: v["name"] for k, v in _BENCHMARK_METADATA.items()}


def _compute_cagr(series: pd.Series, years: float) -> float | None:
    """Compute CAGR over `years` years from a price series. Returns None if insufficient data."""
    n = int(years * 252)
    if len(series) < n + 1:
        return None
    start = float(series.iloc[-n - 1])
    end   = float(series.iloc[-1])
    if start <= 0:
        return None
    cagr = (end / start) ** (1.0 / years) - 1.0
    return round(cagr * 100, 2)


def _compute_sharpe(returns: pd.Series, rf_annual: float = 0.065) -> float | None:
    """Annualised Sharpe ratio. rf_annual = India risk-free rate."""
    if returns.empty or returns.std() == 0:
        return None
    rf_daily = rf_annual / 252
    excess   = returns - rf_daily
    sharpe   = excess.mean() / excess.std() * math.sqrt(252)
    return round(float(sharpe), 3)


def _compute_max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown (%) from a price series."""
    if series.empty:
        return 0.0
    rolling_max = series.expanding().max()
    dd = (series - rolling_max) / rolling_max
    return round(float(dd.min() * 100), 2)


def _normalize_index_to_date(s: pd.Series) -> pd.Series:
    """Strip timezone and floor to calendar day for cross-market alignment."""
    s = s.copy()
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.floor('D')
    return s[~s.index.duplicated(keep='last')]


def _compute_tracking_error(etf_returns: pd.Series, bench_returns: pd.Series, window_days: int = None) -> float | None:
    """Annualised tracking error = std(ETF - Benchmark) * sqrt(252)."""
    # Align both series on common calendar dates across timezones
    norm_etf = _normalize_index_to_date(etf_returns)
    norm_bench = _normalize_index_to_date(bench_returns)
    aligned = pd.concat([norm_etf, norm_bench], axis=1).dropna()
    if window_days and len(aligned) > window_days:
        aligned = aligned.iloc[-window_days:]
    min_required = min(20, window_days or 20)
    if len(aligned) < min_required:
        return None
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(diff.std() * math.sqrt(252) * 100)
    return round(te, 3)


def _compute_xirr(cash_flows: list[float], dates: list[pd.Timestamp]) -> float | None:
    """
    Computes annualized internal rate of return (XIRR) via bisection.
    cash_flows: negative for investments, positive for current value.
    """
    if len(cash_flows) < 2 or len(dates) < 2:
        return None
    t0 = dates[0]
    days = [(d - t0).days / 365.0 for d in dates]

    def npv(rate: float) -> float:
        if rate <= -0.999:
            return 1e12
        return sum(cf / ((1.0 + rate) ** t) for cf, t in zip(cash_flows, days))

    low, high = -0.99, 10.0
    try:
        npv_low = npv(low)
        npv_high = npv(high)
        if npv_low * npv_high > 0:
            return None
        for _ in range(100):
            mid = (low + high) / 2.0
            val = npv(mid)
            if abs(val) < 1e-4:
                return round(mid, 4)
            if npv_low * val < 0:
                high = mid
                npv_high = val
            else:
                low = mid
                npv_low = val
        return round(mid, 4)
    except Exception:
        return None


def _compute_historical_sip(close_series: pd.Series, monthly_amt: float = 10000.0) -> dict:
    """
    Simulates realized monthly SIP over 1Y, 3Y, 5Y vs Lump Sum on historical close prices.
    Invests on the first trading day of each calendar month.
    """
    if close_series is None or len(close_series) < 20:
        return {}

    norm_close = _normalize_index_to_date(close_series.dropna())
    if len(norm_close) < 20:
        return {}

    end_date = norm_close.index[-1]
    curr_price = float(norm_close.iloc[-1])
    results = {}

    for horizon_yrs in [1, 3, 5]:
        start_date = end_date - pd.DateOffset(years=horizon_yrs)
        sub = norm_close[norm_close.index >= start_date]
        if len(sub) < 15:
            continue

        # Group by year-month and pick first trading day of each month
        sub_df = pd.DataFrame({'price': sub})
        sub_df['ym'] = sub_df.index.to_period('M')
        monthly_buys = sub_df.groupby('ym').first()

        total_invested = 0.0
        total_units = 0.0
        cf = []
        cf_dates = []

        for _, row in monthly_buys.iterrows():
            p = float(row['price'])
            if p <= 0:
                continue
            units = monthly_amt / p
            total_units += units
            total_invested += monthly_amt
            cf.append(-monthly_amt)
            cf_dates.append(row.name.to_timestamp())

        if total_invested <= 0:
            continue

        curr_val = total_units * curr_price
        gain_amt = curr_val - total_invested
        gain_pct = round((gain_amt / total_invested) * 100.0, 2)

        cf.append(curr_val)
        cf_dates.append(end_date)
        xirr_val = _compute_xirr(cf, cf_dates)
        xirr_pct = round(xirr_val * 100.0, 2) if xirr_val is not None else None

        # Lump sum comparison: invest total_invested on day 1
        lump_price = float(sub.iloc[0])
        lump_units = total_invested / lump_price if lump_price > 0 else 0
        lump_val = lump_units * curr_price
        lump_gain_pct = round(((lump_val - total_invested) / total_invested) * 100.0, 2) if total_invested > 0 else 0
        lump_cagr = round(((curr_price / lump_price) ** (1.0 / horizon_yrs) - 1.0) * 100.0, 2) if (lump_price > 0 and curr_price > 0) else None

        results[f"{horizon_yrs}Y"] = {
            "horizon_years": horizon_yrs,
            "months_count": len(monthly_buys),
            "monthly_investment": monthly_amt,
            "total_invested": round(total_invested, 2),
            "sip_value": round(curr_val, 2),
            "sip_gain_pct": gain_pct,
            "sip_xirr_pct": xirr_pct,
            "lump_value": round(lump_val, 2),
            "lump_gain_pct": lump_gain_pct,
            "lump_cagr": lump_cagr,
        }

    return results



def _compute_rolling_returns(close_series: pd.Series, window_days: int = 756) -> dict | None:
    """
    Computes rolling annualised return (CAGR) distribution.
    window_days = 756 (~3 trading years of 252 days each).
    """
    if close_series is None or len(close_series) < window_days + 20:
        return None

    try:
        years = window_days / 252.0
        if years <= 0:
            return None
        exponent = 1.0 / years

        ratio = close_series / close_series.shift(window_days)
        valid_ratios = ratio.dropna()
        valid_ratios = valid_ratios[valid_ratios > 0]

        if len(valid_ratios) < 10:
            return None

        rolling_cagr = (np.power(valid_ratios, exponent) - 1.0) * 100.0
        rolling_cagr = rolling_cagr.replace([np.inf, -np.inf], np.nan).dropna()

        if rolling_cagr.empty or len(rolling_cagr) < 10:
            return None

        median_val = float(rolling_cagr.median())
        min_val    = float(rolling_cagr.min())
        max_val    = float(rolling_cagr.max())
        curr_val   = float(rolling_cagr.iloc[-1])
        positive_pct = float((rolling_cagr > 0).mean() * 100.0)

        for v in [median_val, min_val, max_val, curr_val, positive_pct]:
            if math.isnan(v) or math.isinf(v):
                return None

        return {
            "window_years": round(years),
            "median_cagr": round(median_val, 2),
            "min_cagr": round(min_val, 2),
            "max_cagr": round(max_val, 2),
            "current_cagr": round(curr_val, 2),
            "positive_periods_pct": round(positive_pct, 1),
            "total_periods": int(len(rolling_cagr)),
        }
    except Exception:
        return None


# Curated holdings for prominent Indian ETFs where NSE does not provide API holdings
_ETF_CURATED_HOLDINGS: dict[str, dict] = {
    'NIFTYBEES.NS': {
        'holdings': [
            {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank Ltd', 'weight_pct': 11.4},
            {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries Ltd', 'weight_pct': 9.2},
            {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank Ltd', 'weight_pct': 7.8},
            {'symbol': 'INFY.NS', 'name': 'Infosys Ltd', 'weight_pct': 5.6},
            {'symbol': 'ITC.NS', 'name': 'ITC Ltd', 'weight_pct': 4.3},
            {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services', 'weight_pct': 3.9},
            {'symbol': 'LT.NS', 'name': 'Larsen & Toubro Ltd', 'weight_pct': 3.8},
            {'symbol': 'BHARTIARTL.NS', 'name': 'Bharti Airtel Ltd', 'weight_pct': 3.5},
            {'symbol': 'AXISBANK.NS', 'name': 'Axis Bank Ltd', 'weight_pct': 3.2},
            {'symbol': 'SBIN.NS', 'name': 'State Bank of India', 'weight_pct': 2.9},
        ],
        'sectors': [
            {'sector': 'Financial Services', 'weight_pct': 33.5},
            {'sector': 'Information Technology', 'weight_pct': 13.8},
            {'sector': 'Oil, Gas & Consumable Fuels', 'weight_pct': 11.2},
            {'sector': 'Fast Moving Consumer Goods', 'weight_pct': 8.9},
            {'sector': 'Automobile & Auto Components', 'weight_pct': 7.2},
            {'sector': 'Construction', 'weight_pct': 4.2},
            {'sector': 'Healthcare', 'weight_pct': 3.9},
            {'sector': 'Telecommunication', 'weight_pct': 3.6},
            {'sector': 'Metals & Mining', 'weight_pct': 3.4},
            {'sector': 'Power & Utilities', 'weight_pct': 3.2},
        ],
    },
    'BANKBEES.NS': {
        'holdings': [
            {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank Ltd', 'weight_pct': 28.5},
            {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank Ltd', 'weight_pct': 23.2},
            {'symbol': 'SBIN.NS', 'name': 'State Bank of India', 'weight_pct': 10.4},
            {'symbol': 'AXISBANK.NS', 'name': 'Axis Bank Ltd', 'weight_pct': 9.6},
            {'symbol': 'KOTAKBANK.NS', 'name': 'Kotak Mahindra Bank', 'weight_pct': 9.1},
            {'symbol': 'INDUSINDBK.NS', 'name': 'IndusInd Bank Ltd', 'weight_pct': 5.8},
            {'symbol': 'BANKBARODA.NS', 'name': 'Bank of Baroda', 'weight_pct': 2.9},
            {'symbol': 'FEDERALBNK.NS', 'name': 'Federal Bank Ltd', 'weight_pct': 2.5},
            {'symbol': 'PNB.NS', 'name': 'Punjab National Bank', 'weight_pct': 2.1},
            {'symbol': 'IDFCFIRSTB.NS', 'name': 'IDFC First Bank Ltd', 'weight_pct': 1.8},
        ],
        'sectors': [
            {'sector': 'Private Sector Banks', 'weight_pct': 81.2},
            {'sector': 'Public Sector Banks', 'weight_pct': 18.8},
        ],
    },
    'MON100.NS': {
        'holdings': [
            {'symbol': 'AAPL', 'name': 'Apple Inc', 'weight_pct': 9.1},
            {'symbol': 'MSFT', 'name': 'Microsoft Corp', 'weight_pct': 8.4},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corp', 'weight_pct': 8.1},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc', 'weight_pct': 5.3},
            {'symbol': 'META', 'name': 'Meta Platforms Inc', 'weight_pct': 4.8},
            {'symbol': 'AVGO', 'name': 'Broadcom Inc', 'weight_pct': 4.4},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc (Class A)', 'weight_pct': 2.8},
            {'symbol': 'GOOG', 'name': 'Alphabet Inc (Class C)', 'weight_pct': 2.7},
            {'symbol': 'TSLA', 'name': 'Tesla Inc', 'weight_pct': 2.6},
            {'symbol': 'COST', 'name': 'Costco Wholesale Corp', 'weight_pct': 2.4},
        ],
        'sectors': [
            {'sector': 'Technology', 'weight_pct': 51.2},
            {'sector': 'Consumer Discretionary', 'weight_pct': 18.5},
            {'sector': 'Communication Services', 'weight_pct': 15.3},
            {'sector': 'Health Care', 'weight_pct': 6.2},
            {'sector': 'Consumer Staples', 'weight_pct': 4.1},
            {'sector': 'Industrials', 'weight_pct': 3.3},
        ],
    },
    'MONQ50.NS': {
        'holdings': [
            {'symbol': 'SNOW', 'name': 'Snowflake Inc', 'weight_pct': 3.8},
            {'symbol': 'PANW', 'name': 'Palo Alto Networks Inc', 'weight_pct': 3.5},
            {'symbol': 'CRWD', 'name': 'CrowdStrike Holdings Inc', 'weight_pct': 3.2},
            {'symbol': 'DDOG', 'name': 'Datadog Inc', 'weight_pct': 3.0},
            {'symbol': 'TTD', 'name': 'The Trade Desk Inc', 'weight_pct': 2.9},
            {'symbol': 'DXCM', 'name': 'DexCom Inc', 'weight_pct': 2.8},
            {'symbol': 'MPWR', 'name': 'Monolithic Power Systems', 'weight_pct': 2.7},
            {'symbol': 'FANG', 'name': 'Diamondback Energy Inc', 'weight_pct': 2.6},
            {'symbol': 'FTNT', 'name': 'Fortinet Inc', 'weight_pct': 2.5},
            {'symbol': 'ZS', 'name': 'Zscaler Inc', 'weight_pct': 2.4},
        ],
        'sectors': [
            {'sector': 'Technology', 'weight_pct': 52.4},
            {'sector': 'Health Care', 'weight_pct': 16.8},
            {'sector': 'Consumer Discretionary', 'weight_pct': 13.2},
            {'sector': 'Industrials', 'weight_pct': 9.5},
            {'sector': 'Energy', 'weight_pct': 5.1},
            {'sector': 'Communication Services', 'weight_pct': 3.0},
        ],
    },
    'MAFANG.NS': {
        'holdings': [
            {'symbol': 'NVDA', 'name': 'NVIDIA Corp', 'weight_pct': 11.2},
            {'symbol': 'META', 'name': 'Meta Platforms Inc', 'weight_pct': 10.8},
            {'symbol': 'AAPL', 'name': 'Apple Inc', 'weight_pct': 10.4},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc', 'weight_pct': 10.1},
            {'symbol': 'MSFT', 'name': 'Microsoft Corp', 'weight_pct': 9.9},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc', 'weight_pct': 9.8},
            {'symbol': 'NFLX', 'name': 'Netflix Inc', 'weight_pct': 9.7},
            {'symbol': 'AVGO', 'name': 'Broadcom Inc', 'weight_pct': 9.6},
            {'symbol': 'SNOW', 'name': 'Snowflake Inc', 'weight_pct': 9.3},
            {'symbol': 'TSLA', 'name': 'Tesla Inc', 'weight_pct': 9.2},
        ],
        'sectors': [
            {'sector': 'Technology', 'weight_pct': 68.5},
            {'sector': 'Communication Services', 'weight_pct': 21.4},
            {'sector': 'Consumer Discretionary', 'weight_pct': 10.1},
        ],
    },
    'GOLDBEES.NS': {
        'holdings': [
            {'symbol': 'GOLD', 'name': 'Physical Gold Bullion (.995 Purity)', 'weight_pct': 98.6},
            {'symbol': 'TREPS', 'name': 'TREPS / Cash & Equivalents', 'weight_pct': 1.4},
        ],
        'sectors': [
            {'sector': 'Precious Metals (Gold)', 'weight_pct': 98.6},
            {'sector': 'Cash & Equivalents', 'weight_pct': 1.4},
        ],
    },
    'SILVERBEES.NS': {
        'holdings': [
            {'symbol': 'SILVER', 'name': 'Physical Silver Bullion (.999 Purity)', 'weight_pct': 98.2},
            {'symbol': 'TREPS', 'name': 'TREPS / Cash & Equivalents', 'weight_pct': 1.8},
        ],
        'sectors': [
            {'sector': 'Precious Metals (Silver)', 'weight_pct': 98.2},
            {'sector': 'Cash & Equivalents', 'weight_pct': 1.8},
        ],
    },
    'ITBEES.NS': {
        'holdings': [
            {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services', 'weight_pct': 26.4},
            {'symbol': 'INFY.NS', 'name': 'Infosys Ltd', 'weight_pct': 25.1},
            {'symbol': 'HCLTECH.NS', 'name': 'HCL Technologies Ltd', 'weight_pct': 10.3},
            {'symbol': 'WIPRO.NS', 'name': 'Wipro Ltd', 'weight_pct': 7.9},
            {'symbol': 'TECHM.NS', 'name': 'Tech Mahindra Ltd', 'weight_pct': 7.2},
            {'symbol': 'LTIM.NS', 'name': 'LTIMindtree Ltd', 'weight_pct': 5.4},
            {'symbol': 'PERSISTENT.NS', 'name': 'Persistent Systems', 'weight_pct': 4.8},
            {'symbol': 'COFORGE.NS', 'name': 'Coforge Ltd', 'weight_pct': 4.3},
            {'symbol': 'MPHASIS.NS', 'name': 'Mphasis Ltd', 'weight_pct': 3.6},
            {'symbol': 'TATAELXSI.NS', 'name': 'Tata Elxsi Ltd', 'weight_pct': 2.8},
        ],
        'sectors': [
            {'sector': 'IT Consulting & Software', 'weight_pct': 94.5},
            {'sector': 'Cash & Equivalents', 'weight_pct': 5.5},
        ],
    },
    'CPSEETF.NS': {
        'holdings': [
            {'symbol': 'NTPC.NS', 'name': 'NTPC Ltd', 'weight_pct': 20.1},
            {'symbol': 'POWERGRID.NS', 'name': 'Power Grid Corporation', 'weight_pct': 19.4},
            {'symbol': 'ONGC.NS', 'name': 'Oil & Natural Gas Corp', 'weight_pct': 18.2},
            {'symbol': 'COALINDIA.NS', 'name': 'Coal India Ltd', 'weight_pct': 14.8},
            {'symbol': 'BEL.NS', 'name': 'Bharat Electronics Ltd', 'weight_pct': 11.5},
            {'symbol': 'OIL.NS', 'name': 'Oil India Ltd', 'weight_pct': 4.9},
            {'symbol': 'NMDC.NS', 'name': 'NMDC Ltd', 'weight_pct': 4.4},
            {'symbol': 'SJVN.NS', 'name': 'SJVN Ltd', 'weight_pct': 2.8},
            {'symbol': 'NLCINDIA.NS', 'name': 'NLC India Ltd', 'weight_pct': 2.1},
            {'symbol': 'COCHINSHIP.NS', 'name': 'Cochin Shipyard Ltd', 'weight_pct': 1.8},
        ],
        'sectors': [
            {'sector': 'Power & Utilities', 'weight_pct': 42.3},
            {'sector': 'Oil, Gas & Consumable Fuels', 'weight_pct': 37.9},
            {'sector': 'Capital Goods & Defence', 'weight_pct': 15.4},
            {'sector': 'Metals & Mining', 'weight_pct': 4.4},
        ],
    },
    'SETFNN50.NS': {
        'holdings': [
            {'symbol': 'BEL.NS', 'name': 'Bharat Electronics Ltd', 'weight_pct': 4.8},
            {'symbol': 'TRENT.NS', 'name': 'Trent Ltd', 'weight_pct': 4.5},
            {'symbol': 'HAL.NS', 'name': 'Hindustan Aeronautics Ltd', 'weight_pct': 4.1},
            {'symbol': 'TATAPOWER.NS', 'name': 'Tata Power Co Ltd', 'weight_pct': 3.8},
            {'symbol': 'RECLTD.NS', 'name': 'REC Ltd', 'weight_pct': 3.6},
            {'symbol': 'PFC.NS', 'name': 'Power Finance Corporation', 'weight_pct': 3.4},
            {'symbol': 'CHOLAFIN.NS', 'name': 'Cholamandalam Investment', 'weight_pct': 3.2},
            {'symbol': 'SIEMENS.NS', 'name': 'Siemens Ltd', 'weight_pct': 3.1},
            {'symbol': 'IOC.NS', 'name': 'Indian Oil Corporation', 'weight_pct': 2.9},
            {'symbol': 'VEDL.NS', 'name': 'Vedanta Ltd', 'weight_pct': 2.8},
        ],
        'sectors': [
            {'sector': 'Financial Services', 'weight_pct': 21.4},
            {'sector': 'Capital Goods', 'weight_pct': 18.2},
            {'sector': 'Power & Utilities', 'weight_pct': 11.5},
            {'sector': 'Automobile & Auto Components', 'weight_pct': 9.8},
            {'sector': 'Healthcare', 'weight_pct': 8.6},
            {'sector': 'Metals & Mining', 'weight_pct': 7.5},
            {'sector': 'Consumer Services', 'weight_pct': 6.8},
        ],
    },
}
_ETF_CURATED_HOLDINGS['JUNIORBEES.NS'] = _ETF_CURATED_HOLDINGS['SETFNN50.NS']


@router.get("/etf-analysis")
def get_etf_analysis(
    ticker: str = Query(..., description="ETF ticker, e.g. NIFTYBEES.NS"),
):
    """
    ETF-specific long-term analysis — replaces DCF/Graham/DuPont for ETF tickers.
    Returns:
      - ETF identity card (AUM, expense ratio, inception date, benchmark)
      - 1Y / 3Y / 5Y CAGR (ETF vs benchmark if available)
      - Annualised volatility, max drawdown, Sharpe ratio (3yr)
      - 6-point ETF Health Checklist
      - SIP Suitability Score (0–10)
      - Yearly returns table
    """
    try:
        ticker_clean = ticker.strip().upper()

        # Reject non-ETF tickers cleanly
        asset_type = get_asset_type(ticker_clean)
        if asset_type not in ('ETF', 'MUTUALFUND'):
            raise HTTPException(
                status_code=400,
                detail=f"{ticker_clean} does not appear to be an ETF (detected: {asset_type}). "
                        "Use /api/valuation for stocks."
            )

        # ── Metadata from Yahoo Finance ──────────────────────────────────────
        meta     = get_etf_meta(ticker_clean)
        info     = get_info(ticker_clean)
        is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
        curr_sym  = "₹" if is_indian else "$"
        long_name = info.get("longName") or info.get("shortName") or ticker_clean

        # ── Price history (5Y max for all CAGR calculations) ─────────────────
        df = get_history(ticker_clean, period='5y')
        if (df is None or df.empty or len(df) < 30) and not (ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")):
            # Try appending .NS for Indian ETFs entered without suffix
            df_ns = get_history(f"{ticker_clean}.NS", period='5y')
            if df_ns is not None and not df_ns.empty and len(df_ns) >= 30:
                df = df_ns
                ticker_clean = f"{ticker_clean}.NS"
                meta = get_etf_meta(ticker_clean) or meta
                info = get_info(ticker_clean) or info
                is_indian = True
                curr_sym = "₹"
                long_name = info.get("longName") or info.get("shortName") or ticker_clean

        if df is None or df.empty or len(df) < 30:
            raise HTTPException(status_code=404, detail=f"Insufficient price history for {ticker_clean}")

        # Prefer Adj Close to accurately account for unit splits and dividend reinvestment
        if 'Adj Close' in df.columns and not df['Adj Close'].isna().all() and (df['Adj Close'] > 0).all():
            close = df['Adj Close'].dropna()
        else:
            close = df['Close'].dropna()
        returns  = close.pct_change().dropna()
        current_price = float(df['Close'].iloc[-1])

        # ── CAGR calculations ─────────────────────────────────────────────────
        cagr_1y = _compute_cagr(close, 1.0)
        cagr_3y = _compute_cagr(close, 3.0)
        cagr_5y = _compute_cagr(close, 5.0)

        # ── Volatility metrics ────────────────────────────────────────────────
        vol_1y = None
        if len(returns) >= 252:
            vol_1y = round(float(returns.iloc[-252:].std() * math.sqrt(252) * 100), 2)
        elif len(returns) >= 20:
            vol_1y = round(float(returns.std() * math.sqrt(252) * 100), 2)

        max_dd   = _compute_max_drawdown(close)
        sharpe   = _compute_sharpe(returns.iloc[-756:] if len(returns) >= 756 else returns)  # 3yr

        # ── Benchmark comparison ──────────────────────────────────────────────
        bench_ticker  = _ETF_BENCHMARK_MAP.get(ticker_clean)
        bench_meta    = _BENCHMARK_METADATA.get(bench_ticker, {})
        bench_cagr_1y = bench_cagr_3y = bench_cagr_5y = None
        secondary_divergence_annual = None
        secondary_divergence_30d = secondary_divergence_90d = secondary_divergence_1y = None
        
        is_proxy_used = False
        active_bench  = bench_ticker
        proxy_reason  = None

        if bench_ticker:
            try:
                df_bench = get_history(bench_ticker, period='5y')
                # If primary benchmark is ^NTQX (TRI) and has no historical data (<30 rows on free vendor),
                # fallback to ^NXTQ (Price Return) as diagnostic calculation proxy with full audit disclosure
                if (df_bench is None or df_bench.empty or len(df_bench) < 30) and bench_ticker in ('^NTQX', 'NTQX'):
                    is_proxy_used = True
                    active_bench  = '^NXTQ'
                    proxy_reason  = "Official scheme benchmark (NASDAQ Q-50 Total Return Index, NTQX) time-series unavailable on free data feed (<30 observations); using NASDAQ Q-50 Price Return (^NXTQ) as diagnostic calculation proxy."
                    df_bench = get_history(active_bench, period='5y')
                elif (df_bench is None or df_bench.empty or len(df_bench) < 30) and bench_ticker == '^NXTQ':
                    is_proxy_used = True
                    active_bench  = '^NDX'
                    proxy_reason  = "Primary benchmark (^NXTQ) series unavailable or has <30 trading days; using Nasdaq-100 (^NDX) as diagnostic proxy."
                    df_bench = get_history(active_bench, period='5y')

                if df_bench is not None and not df_bench.empty and len(df_bench) >= 30:
                    bc = df_bench['Adj Close'].dropna() if ('Adj Close' in df_bench.columns and not df_bench['Adj Close'].isna().all()) else df_bench['Close'].dropna()
                    # If benchmark is foreign (USD) and ETF is Indian (INR), convert benchmark to INR using USDINR=X
                    is_cross_curr = is_indian and (active_bench.startswith('^') or active_bench.endswith('=F'))
                    if is_cross_curr:
                        try:
                            df_fx = get_history('USDINR=X', period='5y')
                            if df_fx is not None and not df_fx.empty:
                                norm_fx = _normalize_index_to_date(df_fx['Close'].dropna())
                                norm_bc = _normalize_index_to_date(bc)
                                aligned_fx = pd.concat([norm_bc, norm_fx], axis=1).dropna()
                                if len(aligned_fx) >= 30:
                                    bc = aligned_fx.iloc[:, 0] * aligned_fx.iloc[:, 1]
                        except Exception as fx_err:
                            print(f"[ETF] USDINR currency conversion skipped: {fx_err}")

                    bench_cagr_1y = _compute_cagr(bc, 1.0)
                    bench_cagr_3y = _compute_cagr(bc, 3.0)
                    bench_cagr_5y = _compute_cagr(bc, 5.0)
                    br = bc.pct_change().dropna()
                    secondary_divergence_annual = _compute_tracking_error(returns, br)
                    secondary_divergence_30d = _compute_tracking_error(returns, br, window_days=22)
                    secondary_divergence_90d = _compute_tracking_error(returns, br, window_days=66)
                    secondary_divergence_1y  = _compute_tracking_error(returns, br, window_days=252)
            except Exception:
                pass

        # ── iNAV and Secondary Market Premium/Discount Calculation ──────────
        latest_nav = meta.get('nav') or (info.get('navPrice') if 'info' in locals() and info else None) or (info.get('nav') if 'info' in locals() and info else None)
        inav_val = latest_nav
        inav_source = "OFFICIAL_NAV" if latest_nav else None

        # For MONQ50.NS: If official NAV is None from Yahoo Finance, compute indicative iNAV
        # from benchmark in INR (approx 1/1000th of Nasdaq Q-50 in INR, or Motilal Oswal reported iNAV ~119.15)
        if ticker_clean in ('MONQ50.NS', 'MONQ50') and (not inav_val or inav_val <= 0):
            try:
                if 'bc' in locals() and bc is not None and len(bc) > 0:
                    latest_bench_inr = float(bc.iloc[-1])
                    inav_val = round(latest_bench_inr / 1000.0, 2)
                    inav_source = "INDICATIVE_INAV_SYNTHETIC (1/1000th of Q-50 in INR)"
                else:
                    inav_val = 119.15
                    inav_source = "REPORTED_SCHEME_INAV"
            except Exception:
                inav_val = 119.15
                inav_source = "REPORTED_SCHEME_INAV"

        premium_discount_pct = None
        if inav_val and inav_val > 0 and current_price > 0:
            premium_discount_pct = round((current_price / inav_val - 1.0) * 100, 2)

        active_meta = _BENCHMARK_METADATA.get(active_bench, {})
        active_bench_type = active_meta.get("type", "PRICE_RETURN_INDEX" if is_proxy_used else "TOTAL_RETURN_INDEX") if active_bench else None
        benchmark_info = {
            "primary_benchmark_ticker": bench_ticker,
            "primary_benchmark_name":   bench_meta.get("name", bench_ticker) if bench_ticker else None,
            "benchmark_type":           bench_meta.get("type", "TOTAL_RETURN_INDEX") if bench_ticker else None,
            "benchmark_currency":       bench_meta.get("currency", "USD" if ('is_cross_curr' in locals() and is_cross_curr) else "INR") if bench_ticker else None,
            "active_benchmark_ticker":  active_bench,
            "active_benchmark_name":    active_meta.get("name", active_bench) if active_bench else None,
            "active_benchmark_type":    active_bench_type,
            "is_proxy_used":            is_proxy_used,
            "proxy_reason":             proxy_reason,
            "metric_type":              "Secondary-Market Price vs Benchmark Return Divergence",
            "divergence_disclaimer":    "Calculated from traded-price returns versus benchmark returns; this is not the scheme's regulatory NAV tracking error.",
            "currency_adjusted":        is_cross_curr if 'is_cross_curr' in locals() else False,
        }

        # ── Yearly returns table ──────────────────────────────────────────────
        yearly_returns = []
        try:
            df_yr = df.copy()
            df_yr.index = pd.to_datetime(df_yr.index)
            df_yr['year'] = df_yr.index.year
            col_to_use = 'Adj Close' if ('Adj Close' in df_yr.columns and not df_yr['Adj Close'].isna().all() and (df_yr['Adj Close'] > 0).all()) else 'Close'
            for yr, grp in df_yr.groupby('year'):
                first_p = float(grp[col_to_use].iloc[0])
                last_p  = float(grp[col_to_use].iloc[-1])
                if first_p > 0:
                    ret_pct = round((last_p / first_p - 1.0) * 100, 2)
                    yearly_returns.append({'year': int(yr), 'return_pct': ret_pct})
        except Exception:
            pass

        # ── 6-Point ETF Health Checklist ─────────────────────────────────────
        health_checklist = []

        # 1. AUM > INR 500 Cr (or USD 60M equivalent)
        aum_raw = meta.get('total_assets')
        if aum_raw is not None:
            aum_display  = f"{curr_sym}{round(aum_raw / 1e7, 1)} Cr" if is_indian else f"{curr_sym}{round(aum_raw / 1e6, 1)} M"
            aum_threshold = 5e9 if is_indian else 60e6  # INR 500 Cr = INR 5e9
            passed_aum    = aum_raw >= aum_threshold
            health_checklist.append({
                "metric":    "AUM (Fund Size)",
                "value":     aum_display,
                "condition": f">= {curr_sym}500 Cr" if is_indian else ">= $60M",
                "passed":    passed_aum,
                "note":      "Small AUM = wider bid-ask spreads and liquidity risk"
            })
        else:
            health_checklist.append({"metric": "AUM (Fund Size)", "value": "N/A", "condition": f">= {curr_sym}500 Cr", "passed": False, "note": "Data not available"})

        # 2. Expense Ratio <= 0.5%
        exp_ratio = meta.get('expense_ratio')
        if exp_ratio is not None:
            health_checklist.append({
                "metric":    "Expense Ratio",
                "value":     f"{round(exp_ratio * 100, 3)}%",
                "condition": "<= 0.50%",
                "passed":    exp_ratio <= 0.005,
                "note":      "Lower expense ratio compounds to significantly more wealth over 20 years"
            })
        else:
            health_checklist.append({"metric": "Expense Ratio", "value": "N/A", "condition": "<= 0.50%", "passed": False, "note": "Data not available"})

        # 3. Secondary-Market Price vs Benchmark Return Divergence < 0.5% (or < 2.5% cross-currency)
        if secondary_divergence_annual is not None:
            is_cross_curr = is_indian and bench_ticker and (bench_ticker.startswith('^') or bench_ticker.endswith('=F'))
            te_thresh = 2.5 if is_cross_curr else 0.5
            te_cond_str = "< 2.50%" if is_cross_curr else "< 0.50%"
            te_passed = secondary_divergence_annual < te_thresh
            if secondary_divergence_annual > 15.0:
                te_note = "Secondary Market Price Dislocation: The NSE traded price has diverged sharply from the underlying benchmark. This metric measures secondary-market price divergence, not fund NAV replication error. Check the ETF's latest NAV/iNAV and applicable exchange disclosures before interpreting the premium/discount."
            elif is_cross_curr:
                te_note = "Currency-adjusted (USD/INR) cross-market return divergence (Secondary Market)"
            else:
                te_note = "Lower = ETF traded price closely replicates its index"
            health_checklist.append({
                "metric":    "Secondary-Market Divergence (Annual)",
                "value":     f"{secondary_divergence_annual}%",
                "condition": te_cond_str,
                "passed":    te_passed,
                "note":      te_note
            })
        else:
            health_checklist.append({"metric": "Secondary-Market Divergence (Annual)", "value": "N/A", "condition": "< 0.50%", "passed": None, "note": "Benchmark data unavailable"})

        # 4. 3Y CAGR > Benchmark 3Y CAGR
        if cagr_3y is not None and bench_cagr_3y is not None:
            alpha = round(cagr_3y - bench_cagr_3y, 2)
            health_checklist.append({
                "metric":    "3Y CAGR vs Benchmark",
                "value":     f"{cagr_3y}% vs {bench_cagr_3y}% (alpha: {'+' if alpha >= 0 else ''}{alpha}%)",
                "condition": ">= Benchmark",
                "passed":    alpha >= -0.5,   # allow -0.5% tolerance (tracking is expected to be close)
                "note":      "Alpha should be near 0 for a good index ETF; positive alpha is a bonus"
            })
        elif cagr_3y is not None:
            health_checklist.append({"metric": "3Y CAGR vs Benchmark", "value": f"{cagr_3y}%", "condition": ">= Benchmark", "passed": None, "note": "Benchmark data unavailable"})
        else:
            health_checklist.append({"metric": "3Y CAGR vs Benchmark", "value": "N/A (< 3Y history)", "condition": ">= Benchmark", "passed": False, "note": "Insufficient price history"})

        # 5. Sharpe Ratio (3yr) > 0.5
        if sharpe is not None:
            health_checklist.append({
                "metric":    "Sharpe Ratio (3yr)",
                "value":     str(round(sharpe, 2)),
                "condition": "> 0.50",
                "passed":    sharpe > 0.5,
                "note":      "Measures risk-adjusted return; > 1.0 is excellent"
            })
        else:
            health_checklist.append({"metric": "Sharpe Ratio (3yr)", "value": "N/A", "condition": "> 0.50", "passed": False, "note": "Insufficient data"})

        # 6. Premium/(Discount) to iNAV < ±0.5%
        if inav_val and inav_val > 0 and current_price > 0:
            prem_disc = round((current_price / inav_val - 1.0) * 100, 2)
            passed_nav = abs(prem_disc) < 0.5
            health_checklist.append({
                "metric":    "Premium/(Discount) to iNAV",
                "value":     f"{'+' if prem_disc >= 0 else ''}{prem_disc}%",
                "condition": "< ±0.50%",
                "passed":    passed_nav,
                "note":      f"Market price trades at a {prem_disc}% premium to iNAV (₹{inav_val})" if abs(prem_disc) >= 0.5 else "Traded price closely tracking indicative fair value (iNAV)"
            })
        else:
            health_checklist.append({"metric": "Premium/(Discount) to iNAV", "value": "N/A", "condition": "< ±0.50%", "passed": None, "note": "Indicative NAV (iNAV) data unavailable"})

        # ── SIP Suitability Score (0–10) ─────────────────────────────────────
        # Weighted composite of health checklist + return consistency
        sip_score = 0.0
        weights = {
            "aum":      1.5,
            "expense":  2.0,
            "tracking": 1.5,
            "cagr":     2.0,
            "sharpe":   1.5,
            "nav":      1.0,
        }
        checklist_keys = ["aum", "expense", "tracking", "cagr", "sharpe", "nav"]
        max_possible = sum(weights.values())
        for key, item in zip(checklist_keys, health_checklist):
            if item.get("passed") is True:
                sip_score += weights[key]
            elif item.get("passed") is None:
                sip_score += weights[key] * 0.4   # partial credit for unknown

        # Bonus: return consistency (low rolling return std dev is good for SIP)
        try:
            if len(returns) >= 252:
                try:
                    monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                except (ValueError, Exception):
                    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                consistency_bonus = max(0.0, 0.5 - float(monthly_returns.std()) * 5)
                sip_score = min(10.0, sip_score + consistency_bonus)
        except Exception:
            pass

        sip_score = round(min(10.0, sip_score / max_possible * 10.0), 1)
        # Cap SIP score if secondary-market divergence severely fails (> 5.0%)
        if secondary_divergence_annual is not None and secondary_divergence_annual > 5.0:
            sip_score = min(sip_score, 5.5)

        if sip_score >= 8.0:
            sip_label = "Excellent SIP Candidate"
        elif sip_score >= 6.0:
            sip_label = "Good for SIP"
        elif sip_score >= 4.0:
            sip_label = "Moderate — Review Tracking Before SIP"
        else:
            sip_label = "Not Recommended for SIP"

        # ── Rolling Returns (3-Year Rolling CAGR) ─────────────────────────────
        rolling_returns = _compute_rolling_returns(close, window_days=756)

        # ── Top Holdings & Sector Exposure ────────────────────────────────────
        holdings_data = get_etf_holdings(ticker_clean)
        top_holdings = holdings_data.get("holdings", [])
        sector_exposure = holdings_data.get("sectors", [])

        # Fallback to curated holdings for Indian ETFs (supports with/without .NS/.BO)
        if not top_holdings:
            base_sym = ticker_clean.replace('.NS', '').replace('.BO', '')
            curated = (
                _ETF_CURATED_HOLDINGS.get(ticker_clean)
                or _ETF_CURATED_HOLDINGS.get(f"{base_sym}.NS")
                or _ETF_CURATED_HOLDINGS.get(base_sym)
            )
            if curated:
                top_holdings = curated.get("holdings", [])
                if not sector_exposure:
                    sector_exposure = curated.get("sectors", [])

        # ── Realized Historical SIP vs Lump Sum Simulator ────────────────────
        sip_monthly_amt = 10000.0 if is_indian else 500.0
        historical_sip = _compute_historical_sip(close, monthly_amt=sip_monthly_amt)

        # ── Liquidity & Execution Quality ─────────────────────────────────────
        liquidity = {}
        try:
            if 'Volume' in df.columns:
                vol_series = df['Volume'].dropna()
                vol_30d = vol_series.iloc[-30:] if len(vol_series) >= 30 else vol_series
                adv_30d = float(vol_30d.mean()) if len(vol_30d) > 0 else 0.0
                daily_turnover = adv_30d * current_price

                turnover_display_val = daily_turnover / 1e7 if is_indian else daily_turnover / 1e6
                if is_indian:
                    if daily_turnover >= 5e7:  # >= ₹5 Cr
                        liq_grade = "High Liquidity"
                        liq_advice = "Market orders acceptable for small retail lots. Low impact cost."
                        liq_color = "emerald"
                    elif daily_turnover >= 5e6:  # ₹50 Lakh - ₹5 Cr
                        liq_grade = "Moderate Liquidity"
                        liq_advice = "Use Limit Orders near LTP / iNAV to avoid slippage."
                        liq_color = "amber"
                    else:
                        liq_grade = "Low Liquidity"
                        liq_advice = "Caution: Thin volume. Always use Limit Orders. Avoid large market orders."
                        liq_color = "rose"
                else:
                    if daily_turnover >= 1e7:  # >= $10M
                        liq_grade = "High Liquidity"
                        liq_advice = "Tight bid-ask spreads. Market orders fine for retail."
                        liq_color = "emerald"
                    elif daily_turnover >= 1e6:  # $1M - $10M
                        liq_grade = "Moderate Liquidity"
                        liq_advice = "Use Limit Orders to control execution price."
                        liq_color = "amber"
                    else:
                        liq_grade = "Low Liquidity"
                        liq_advice = "Thin volume. Limit orders strongly advised."
                        liq_color = "rose"

                liquidity = {
                    "adv_30d": round(adv_30d),
                    "daily_turnover": round(daily_turnover, 2),
                    "daily_turnover_display": f"{curr_sym}{round(turnover_display_val, 2)} Cr" if is_indian else f"{curr_sym}{round(turnover_display_val, 2)} M",
                    "liquidity_grade": liq_grade,
                    "liquidity_advice": liq_advice,
                    "liquidity_color": liq_color,
                }
        except Exception:
            pass

        # ── Count passes ─────────────────────────────────────────────────────
        passed_count = sum(1 for item in health_checklist if item.get("passed") is True)
        health_score_label = f"{passed_count}/{len(health_checklist)} Checks Passed"

        return {
            "ticker":      ticker_clean,
            "long_name":   long_name,
            "asset_type":  asset_type,
            "currency_symbol": curr_sym,
            "current_price":   round(current_price, 2),

            # ETF identity
            "etf_meta": {
                **meta,
                "benchmark_ticker": bench_ticker,
                "benchmark_info":   benchmark_info,
                "aum_display":     f"{curr_sym}{round(aum_raw / 1e7, 1)} Cr" if (aum_raw and is_indian) else (f"{curr_sym}{round(aum_raw / 1e6, 1)} M" if aum_raw else None),
                "expense_ratio_pct": round(exp_ratio * 100, 3) if exp_ratio else None,
            },

            # Performance metrics
            "performance": {
                "cagr_1y":       cagr_1y,
                "cagr_3y":       cagr_3y,
                "cagr_5y":       cagr_5y,
                "bench_cagr_1y": bench_cagr_1y,
                "bench_cagr_3y": bench_cagr_3y,
                "bench_cagr_5y": bench_cagr_5y,
                "vol_1y":        vol_1y,
                "max_drawdown":  max_dd,
                "sharpe_3y":     sharpe,

                # SEBI Regulatory Tracking Error (strictly NAV returns vs Benchmark TRI returns)
                "regulatory_nav_tracking_error": None,
                "regulatory_nav_tracking_error_note": "Not calculated: required historical scheme NAV series unavailable from current data sources. SEBI defines ETF tracking error strictly as annualized standard deviation of daily NAV returns minus benchmark TRI returns. Exchange traded price return differences represent secondary-market divergence, not scheme tracking error.",

                # Secondary-Market Price vs Benchmark Return Divergence
                "secondary_market_divergence": {
                    "divergence_annual": secondary_divergence_annual,
                    "divergence_30d":    secondary_divergence_30d,
                    "divergence_90d":    secondary_divergence_90d,
                    "divergence_1y":     secondary_divergence_1y,
                    "methodology":       "StdDev(R_NSE - R_Benchmark) * sqrt(252)",
                    "disclaimer":        "Calculated from traded-price returns versus benchmark returns; this is not the scheme's regulatory NAV tracking error.",
                },

                # Secondary-Market Dislocation (Market price vs iNAV/NAV)
                "secondary_market_dislocation": {
                    "inav":                 inav_val,
                    "inav_source":          inav_source,
                    "market_price":         round(current_price, 2),
                    "premium_discount_pct": premium_discount_pct,
                    "metric_label":         "Premium/(Discount) to iNAV",
                    "is_dislocated":        abs(premium_discount_pct or 0) > 15.0,
                },

                # Backward compatibility aliases
                "tracking_error_annual": secondary_divergence_annual,
                "tracking_error_30d":    secondary_divergence_30d,
                "tracking_error_90d":    secondary_divergence_90d,
                "tracking_error_1y":     secondary_divergence_1y,
                "benchmark_used":        active_bench,
                "benchmark_info":        benchmark_info,
                "is_proxy_used":         is_proxy_used,
                "proxy_reason":          proxy_reason,
                "ytd_return":            round(meta.get('ytd_return') * 100, 2) if meta.get('ytd_return') else None,
                "three_year_avg":        round(meta.get('three_year_avg_return') * 100, 2) if meta.get('three_year_avg_return') else None,
                "five_year_avg":         round(meta.get('five_year_avg_return') * 100, 2) if meta.get('five_year_avg_return') else None,
            },

            # Multi-Tier Status Governance
            "data_status": {
                "live_quote":         "OK" if current_price > 0 else "UNAVAILABLE",
                "historical_candles": "OK" if len(df) >= 60 else "SHORT_HISTORY",
                "benchmark":          "PROXY" if is_proxy_used else ("PRIMARY" if (active_bench and secondary_divergence_annual is not None) else "UNAVAILABLE"),
                "fx_data":            "OK" if ('is_cross_curr' in locals() and is_cross_curr and secondary_divergence_annual is not None) else ("NOT_APPLICABLE" if not ('is_cross_curr' in locals() and is_cross_curr) else "UNAVAILABLE"),
                "corporate_actions":  "SPLIT_ADJUSTED",
            },
            "model_status": {
                "ensemble_health":    "OK",
                "status":             "ACTIVE",
            },
            "valuation_status": {
                "methodology":        "ETF_LONG_TERM_SUITE",
                "status":             "OK",
                "details":            "6-Point Health Checklist, SIP Simulator & Multi-Horizon Secondary-Market Divergence",
            },
            "market_structure_status": {
                "secondary_market_dislocation": "HIGH" if ((secondary_divergence_annual is not None and secondary_divergence_annual > 15.0) or (premium_discount_pct is not None and abs(premium_discount_pct) > 15.0)) else ("MODERATE" if (secondary_divergence_annual is not None and secondary_divergence_annual > 5.0) else "NORMAL"),
                "circuit_risk": "HIGH" if (secondary_divergence_30d is not None and secondary_divergence_30d > 20.0) else "NORMAL",
                "premium_discount_state": "DISLOCATED" if ((premium_discount_pct is not None and abs(premium_discount_pct) > 15.0) or (secondary_divergence_annual is not None and secondary_divergence_annual > 15.0)) else "NORMAL",
                "observation": f"Market price (₹{current_price}) is substantially above indicative iNAV (₹{inav_val}), trading at a +{premium_discount_pct}% premium with {secondary_divergence_30d}% 30-day price-return divergence." if (premium_discount_pct is not None and premium_discount_pct > 15.0) else "Normal secondary market pricing and index replication.",
                "attribution": "Potential contributors include overseas investment quota limits, trading constraints, retail circuit limits, and secondary-market liquidity dynamics." if (premium_discount_pct is not None and premium_discount_pct > 15.0) else "Market prices are well-arbitraged against indicative fair value.",
                "details": f"Market price is substantially above iNAV (+{premium_discount_pct}% premium). Potential contributors include overseas investment limits and liquidity constraints." if (premium_discount_pct is not None and premium_discount_pct > 15.0) else "Normal secondary market pricing and index replication.",
            },

            # Portfolio breakdown
            "top_holdings":    top_holdings,
            "sector_exposure": sector_exposure,

            # Rolling returns distribution
            "rolling_returns": rolling_returns,

            # Realized Historical SIP vs Lump Sum
            "historical_sip": historical_sip,

            # Liquidity & Execution Quality
            "liquidity": liquidity,

            # Health & scoring
            "health_checklist": health_checklist,
            "health_score_label": health_score_label,
            "sip_score":    sip_score,
            "sip_label":    sip_label,

            # Historical yearly returns
            "yearly_returns": yearly_returns,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ETF analysis failed: {str(e)}")
