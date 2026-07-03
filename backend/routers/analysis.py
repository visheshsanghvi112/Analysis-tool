import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from yf_client import get_history, get_quote, get_info
from peer_data import get_peers

router = APIRouter(prefix="/api", tags=["analysis"])

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
            if len(close) > n:
                return round((close.iloc[-1] / close.iloc[-n] - 1) * 100, 2)
            return None

        ret_1m  = safe_ret(22)
        ret_3m  = safe_ret(66)
        ret_6m  = safe_ret(132)
        ret_1y  = safe_ret(len(close) - 1) if len(close) > 5 else None

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

        high52 = float(close.max())
        pct_from_high = round((current_price - high52) / high52 * 100, 2)

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

        # Fetch underlying data fields
        market_cap = info.get("marketCap")
        shares = info.get("sharesOutstanding") or 0
        eps = info.get("trailingEps")
        book_value = info.get("bookValue")
        pb = info.get("priceToBook")
        pe = info.get("trailingPE")
        long_name = info.get("longName") or ticker_clean

        peg = info.get("pegRatio")
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
        insiders = info.get("heldPercentInsiders")
        institutions = info.get("heldPercentInstitutions")
        rev_growth = info.get("revenueGrowth")
        earn_growth = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
        npm = info.get("profitMargins")
        opm = info.get("operatingMargins")
        gpm = info.get("grossMargins")
        roe = info.get("returnOnEquity")
        roa = info.get("returnOnAssets")
        de = info.get("debtToEquity")
        curr_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        cash = info.get("totalCash") or 0
        debt = info.get("totalDebt") or 0
        fcf = info.get("freeCashflow") or 0
        ocf = info.get("operatingCashflow") or 0
        rev = info.get("totalRevenue") or 0

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
            graham_number = round(math.sqrt(22.5 * eps * book_value), 2)

        # DuPont Analysis Decomposition
        dupont = None
        if roe is not None and npm is not None:
            equity = book_value * shares if (book_value and shares) else None
            if not equity and market_cap and pb:
                equity = market_cap / pb
            
            if equity:
                assets = equity + debt
                asset_turnover = rev / assets if (rev and assets > 0) else (roe / (npm * (assets / equity)) if (npm and assets > 0) else None)
                equity_multiplier = assets / equity if equity > 0 else None
                
                dupont = {
                    "net_profit_margin": round(npm * 100, 2) if npm is not None else None,
                    "asset_turnover": round(asset_turnover, 3) if asset_turnover is not None else None,
                    "equity_multiplier": round(equity_multiplier, 2) if equity_multiplier is not None else None,
                    "calculated_roe": round(roe * 100, 2) if roe is not None else None
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
        fcf_val_str = f"₹{round(fcf/1e9, 2)}B" if fcf else (f"₹{round(ocf/1e9, 2)}B (OCF)" if ocf else "Negative/Zero")
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
            "currency": info.get("currency") or "INR",
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
    initial_capital: float = Query(100000, description="Starting capital in INR"),
):
    """
    Simulates a RSI+MACD momentum strategy on historical data.
    """
    try:
        ticker_clean = ticker.strip().upper()

        df = get_history(ticker_clean, period=period)
        if df is None or df.empty or len(df) < 60:
            raise HTTPException(status_code=400, detail="Insufficient historical data")

        close = df["Close"].copy()
        high  = df["High"].copy()
        low   = df["Low"].copy()

        delta    = close.diff()
        gain     = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        loss     = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        rsi      = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

        ema12    = close.ewm(span=12, adjust=False).mean()
        ema26    = close.ewm(span=26, adjust=False).mean()
        macd     = ema12 - ema26
        signal   = macd.ewm(span=9, adjust=False).mean()

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

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
                if prev_rsi < 35 and curr_rsi >= 35 and macd_bull:
                    in_trade = True
                    entry_px  = float(close.iloc[i])
                    stop_loss = entry_px - 2.0 * float(atr.iloc[i])
                    positions.iloc[i] = 1
                else:
                    positions.iloc[i] = 0
            else:
                curr_px = float(close.iloc[i])
                if curr_rsi >= 65 or macd_bear or curr_px < stop_loss:
                    in_trade = False
                    positions.iloc[i] = 0
                else:
                    positions.iloc[i] = 1

        daily_returns    = close.pct_change().fillna(0)
        strategy_returns = positions.shift(1).fillna(0) * daily_returns

        equity           = (1 + strategy_returns).cumprod() * initial_capital
        bh_equity        = (1 + daily_returns).cumprod() * initial_capital

        try:
            nifty_raw    = get_history("^NSEI", period=period)
            nifty_r      = nifty_raw["Close"].pct_change().fillna(0)
            common       = strategy_returns.index.intersection(nifty_r.index)
            nifty_equity = (1 + nifty_r.loc[common]).cumprod() * initial_capital
        except Exception:
            nifty_equity = None

        total_return   = float((equity.iloc[-1] / initial_capital - 1) * 100)
        bh_return      = float((bh_equity.iloc[-1] / initial_capital - 1) * 100)

        ann_factor     = 252
        strat_ann      = float(strategy_returns.mean() * ann_factor * 100)
        strat_vol      = float(strategy_returns.std() * np.sqrt(ann_factor) * 100)
        rf_daily       = 0.065 / 252
        sharpe         = float((strategy_returns - rf_daily).mean() / (strategy_returns.std() + 1e-9) * np.sqrt(ann_factor))

        roll_max       = equity.cummax()
        drawdown       = (equity - roll_max) / roll_max
        max_dd         = float(drawdown.min() * 100)
        calmar         = (strat_ann / abs(max_dd)) if abs(max_dd) > 0 else 0.0

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
            "trades": trades[-30:],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")
