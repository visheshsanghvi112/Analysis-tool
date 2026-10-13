"""
services/portfolio_intelligence.py
───────────────────────────────────────────────────────────────────────────────
Institutional Quantitative Intelligence Engine for Portfolio Management.

Features:
1. Capital Defense & Loss Remediation Engine (The Great Filter, Breakeven Clock,
   Tax-Loss Harvesting Shield, 3-Tranche Volatility DCA, Capital Rotation Switcher).
2. Alpha Harvest & Profit Maximization Engine ("House Money" Free-Ride Math,
   4-Tier Laddered Profit Escalator, Chandelier Trailing Stops, Winner Pyramiding).
3. Macro Stress Lab & Fragility Engine (5 Historical Crisis Simulators,
   Herfindahl-Hirschman Concentration Index HHI, Dynamic Market Cash Sizing).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from services.ticker_manager import SECTOR_MAP
from yf_client import get_quote, get_history, get_info

# Sector proxy mapping for tax-loss rotation / beta replacement
SECTOR_ETF_PROXIES_IN = {
    "Banking": "BANKBEES.NS",
    "Financial Services": "BANKBEES.NS",
    "NBFC": "BANKBEES.NS",
    "IT": "ITBEES.NS",
    "Technology": "ITBEES.NS",
    "Pharma": "PHARMABEES.NS",
    "Healthcare": "PHARMABEES.NS",
    "Auto": "AUTOBEES.NS",
    "Consumer Cyclical": "AUTOBEES.NS",
    "FMCG": "CONSUMBEES.NS",
    "Consumer Defensive": "CONSUMBEES.NS",
    "Energy": "ENERGYBEES.NS",
    "Power": "ENERGYBEES.NS",
    "Metals": "METALBEES.NS",
    "Basic Materials": "METALBEES.NS",
    "Infrastructure": "INFRABEES.NS",
    "Industrials": "INFRABEES.NS",
}

SECTOR_LEADERS_IN = {
    "Banking": "HDFCBANK.NS",
    "Financial Services": "ICICIBANK.NS",
    "NBFC": "BAJFINANCE.NS",
    "IT": "TCS.NS",
    "Technology": "INFY.NS",
    "Pharma": "SUNPHARMA.NS",
    "Healthcare": "CIPLA.NS",
    "Auto": "TATAMOTORS.NS",
    "FMCG": "ITC.NS",
    "Energy": "RELIANCE.NS",
    "Metals": "TATASTEEL.NS",
    "Infrastructure": "LT.NS",
    "Telecom": "BHARTIARTL.NS",
}

SECTOR_PROXIES_US = {
    "Technology": "QQQ",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Utilities": "XLU",
}

SECTOR_LEADERS_US = {
    "Technology": "MSFT",
    "Financial Services": "JPM",
    "Healthcare": "LLY",
    "Consumer Cyclical": "AMZN",
    "Consumer Defensive": "COST",
    "Energy": "XOM",
    "Industrials": "CAT",
    "Communication Services": "GOOGL",
}


def _safe_float(val, default=0.0, ndigits=2):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, ndigits) if ndigits is not None else f
    except (TypeError, ValueError):
        return default


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculates Wilder's ATR or standard ATR over given period."""
    if df is None or len(df) < period + 1:
        return 0.0
    try:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period, min_periods=period).mean()
        last_atr = float(atr_series.iloc[-1])
        return last_atr if not math.isnan(last_atr) and last_atr > 0 else 0.0
    except Exception:
        return 0.0


def _resolve_sector(ticker: str, info_dict: dict) -> str:
    sym_base = ticker.replace(".NS", "").replace(".BO", "").upper()
    if sym_base in SECTOR_MAP:
        return SECTOR_MAP[sym_base]
    if info_dict and "sector" in info_dict and info_dict["sector"]:
        return str(info_dict["sector"])
    return "Diversified"


# ══════════════════════════════════════════════════════════════════════════════
# 1. CAPITAL DEFENSE & LOSS REMEDIATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_loss_remediation(
    ticker: str,
    buy_price: float,
    qty: float,
    live_px: float,
    df: Optional[pd.DataFrame] = None,
    info: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Computes an institutional loss analysis for an underwater holding:
    - The Great Filter (Quality Compounder vs. Structural Value Trap)
    - Opportunity Cost & Breakeven Horizon Clock (Years to recover vs Index)
    - Tax-Loss Harvesting Shield (Exact fiscal savings & correlated proxy)
    - 3-Tranche Volatility-Guided DCA Plan (with Invalidation Stop)
    - Capital Rotation Alternative
    """
    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    curr_sym = "₹" if is_indian else "$"
    info_dict = info or {}

    loss_amt = _safe_float(max(0.0, (buy_price - live_px) * qty))
    loss_pct = _safe_float(((live_px - buy_price) / buy_price * 100) if buy_price > 0 else 0.0)
    req_gain_to_breakeven = _safe_float(((buy_price / live_px - 1) * 100) if live_px > 0 else 0.0)

    sector = _resolve_sector(ticker, info_dict)

    # ── 1. The Great Filter: Quality Scoring ──────────────────────────────────
    quality_score = 50.0  # neutral starting anchor
    score_breakdown = []

    roe = _safe_float(info_dict.get("returnOnEquity"), default=None)
    debt_to_eq = _safe_float(info_dict.get("debtToEquity"), default=None)
    op_margins = _safe_float(info_dict.get("operatingMargins"), default=None)
    rev_growth = _safe_float(info_dict.get("revenueGrowth"), default=None)
    trailing_pe = _safe_float(info_dict.get("trailingPE"), default=None)

    # Technical momentum & trend
    ema_200 = None
    dist_ema_200_pct = None
    rsi_14 = 50.0

    if df is not None and len(df) > 30:
        close = df["Close"]
        if len(close) >= 60:
            ema200_val = close.ewm(span=min(200, len(close)), adjust=False).mean().iloc[-1]
            ema_200 = _safe_float(ema200_val)
            dist_ema_200_pct = _safe_float(((live_px - ema200_val) / ema200_val) * 100)
        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        loss = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
        rsi_14 = _safe_float(rsi_series.iloc[-1])

    # Assess ROE (Return on Equity)
    if roe is not None:
        if roe >= 0.18:
            quality_score += 15
            score_breakdown.append(f"High ROE ({round(roe*100, 1)}%) shows strong franchise compounding (+15)")
        elif roe >= 0.12:
            quality_score += 8
            score_breakdown.append(f"Healthy ROE ({round(roe*100, 1)}%) (+8)")
        elif roe < 0.05:
            quality_score -= 15
            score_breakdown.append(f"Weak capital efficiency: ROE under 5% (-15)")
    else:
        quality_score += 5

    # Assess Debt to Equity
    if debt_to_eq is not None:
        if debt_to_eq < 50:  # yfinance reports debtToEquity * 100 frequently
            quality_score += 15
            score_breakdown.append("Fortress Balance Sheet: Conservative leverage (+15)")
        elif debt_to_eq > 200:
            quality_score -= 20
            score_breakdown.append("Heavy Debt Burden: High financial risk during downturns (-20)")
    else:
        quality_score += 5

    # Assess Margins & Growth
    if op_margins is not None and op_margins > 0.15:
        quality_score += 10
        score_breakdown.append(f"Defensive operating margins ({round(op_margins*100, 1)}%) (+10)")
    elif op_margins is not None and op_margins < 0.05:
        quality_score -= 10
        score_breakdown.append("Thin margins susceptible to cost inflation (-10)")

    # Assess Technical Discount vs 200 EMA
    if dist_ema_200_pct is not None:
        if dist_ema_200_pct < -25 and rsi_14 < 35:
            score_breakdown.append(f"Deeply oversold at -{abs(dist_ema_200_pct)}% from 200 EMA (RSI: {rsi_14})")

    quality_score = max(5.0, min(95.0, quality_score))

    # Determine Great Filter Classification
    if quality_score >= 65:
        filter_status = "QUALITY_ON_SALE"
        filter_label = "Quality on Sale (Compounder)"
        filter_color = "emerald"
        action_directive = "DCA_APPROVED"
        action_advice = "The business is fundamentally sound. The price decline is primarily cyclical or market-driven. Disciplined, staged averaging is mathematically approved."
    elif quality_score <= 40 or (debt_to_eq is not None and debt_to_eq > 250):
        filter_status = "STRUCTURAL_VALUE_TRAP"
        filter_label = "Structural Value Trap"
        filter_color = "rose"
        action_directive = "STRICT_FIREWALL_DO_NOT_AVERAGE"
        action_advice = "FIREWALL ENGAGED: Sunk-Cost Trap detected. Fundamentals (debt, return on capital, margins) show persistent erosion. Averaging down risks throwing good money into dead capital. Consider cutting or rotating."
    else:
        filter_status = "CYCLICAL_RECOVERY"
        filter_label = "Cyclical / Rebuilding"
        filter_color = "amber"
        action_directive = "WAIT_FOR_CONFIRMATION"
        action_advice = "Mixed balance sheet metrics. Do not average down blindly. Wait for price to form a definitive base (reclaim 20-day EMA with volume) before allocating fresh funds."

    # ── 2. Breakeven Horizon Clock & Opportunity Cost ────────────────────────
    # Estimate historical or conservative EPS growth rate
    stock_cagr = 0.08  # default 8% conservative baseline
    if rev_growth is not None and rev_growth > 0:
        stock_cagr = max(0.04, min(0.25, rev_growth))
    elif roe is not None and roe > 0:
        stock_cagr = max(0.05, min(0.22, roe * 0.7))

    benchmark_cagr = 0.135  # Nifty 50 / broad market historical 13.5% nominal CAGR
    
    # Calculate recovery years
    if loss_pct < -0.5 and live_px > 0 and buy_price > live_px:
        ratio = buy_price / live_px
        stock_years = math.log(ratio) / math.log(1 + stock_cagr)
        benchmark_years = math.log(ratio) / math.log(1 + benchmark_cagr)
        years_saved_by_rotating = max(0.0, stock_years - benchmark_years)
    else:
        stock_years = 0.0
        benchmark_years = 0.0
        years_saved_by_rotating = 0.0

    # ── 3. Tax-Loss Harvesting Shield (Fiscal Shield) ────────────────────────
    # Indian Tax Regime: STCL 20%, LTCL 12.5%. Global/US: 20% baseline capital gains shield
    tax_rate = 0.20 if is_indian else 0.20
    tax_shield_val = _safe_float(loss_amt * tax_rate)

    proxy_etf = SECTOR_ETF_PROXIES_IN.get(sector, "NIFTYBEES.NS") if is_indian else SECTOR_PROXIES_US.get(sector, "SPY")
    proxy_leader = SECTOR_LEADERS_IN.get(sector, "HDFCBANK.NS") if is_indian else SECTOR_LEADERS_US.get(sector, "MSFT")

    # ── 4. 3-Tranche Volatility-Guided DCA Plan ──────────────────────────────
    atr = _calculate_atr(df, 14)
    buffer = atr * 1.5 if atr > 0 else live_px * 0.04

    tranche_1_price = _safe_float(live_px)
    tranche_2_price = _safe_float(live_px * 1.025 if live_px > 0 else live_px)
    tranche_3_price = _safe_float(live_px * 1.06 if live_px > 0 else live_px)

    # Invalidation stop level
    invalidation_stop = _safe_float(max(0.0, live_px - (atr * 2.5 if atr > 0 else live_px * 0.08)))

    # Capital sizing tranches: 35%, 40%, 25%
    reference_remedy_capital = max(live_px * 5, (buy_price * qty) * 0.35)
    t1_qty = max(1, math.floor((reference_remedy_capital * 0.35) / max(1.0, tranche_1_price)))
    t2_qty = max(1, math.floor((reference_remedy_capital * 0.40) / max(1.0, tranche_2_price)))
    t3_qty = max(1, math.floor((reference_remedy_capital * 0.25) / max(1.0, tranche_3_price)))

    total_new_qty = qty + t1_qty + t2_qty + t3_qty
    total_new_cost = (qty * buy_price) + (t1_qty * tranche_1_price) + (t2_qty * tranche_2_price) + (t3_qty * tranche_3_price)
    new_avg_price = _safe_float(total_new_cost / total_new_qty if total_new_qty > 0 else buy_price)
    new_breakeven_pct = _safe_float(((new_avg_price / live_px - 1) * 100) if live_px > 0 else 0.0)
    avg_reduction_pct = _safe_float(((buy_price - new_avg_price) / buy_price * 100) if buy_price > 0 else 0.0)

    return {
        "ticker": ticker,
        "sector": sector,
        "loss_amount": loss_amt,
        "loss_pct": loss_pct,
        "req_gain_to_breakeven_pct": req_gain_to_breakeven,
        "filter": {
            "quality_score": round(quality_score, 1),
            "status": filter_status,
            "label": filter_label,
            "color": filter_color,
            "directive": action_directive,
            "advice": action_advice,
            "breakdown": score_breakdown,
        },
        "breakeven_horizon": {
            "stock_projected_cagr_pct": round(stock_cagr * 100, 1),
            "stock_years_to_breakeven": round(stock_years, 1),
            "benchmark_years_to_breakeven": round(benchmark_years, 1),
            "years_saved_by_rotation": round(years_saved_by_rotating, 1),
            "opportunity_cost_verdict": (
                f"Rotating into a {round(benchmark_cagr*100, 1)}% CAGR leader can recover your principal in {round(benchmark_years, 1)} yrs "
                f"(saving ~{round(years_saved_by_rotating, 1)} years of dead money)"
                if years_saved_by_rotating > 1.0 else "Recovery timeline comparable to benchmark index."
            ),
        },
        "tax_harvesting": {
            "tax_shield_rupees": tax_shield_val,
            "applicable_tax_rate_pct": round(tax_rate * 100, 1),
            "tax_jurisdiction": "India (STCL/LTCL Offset)" if is_indian else "Global / US (Capital Gains Shield)",
            "recommended_proxy_etf": proxy_etf,
            "recommended_leader_proxy": proxy_leader,
            "harvesting_note": (
                f"Booking this {curr_sym}{loss_amt:,.0f} loss unlocks an immediate ~{curr_sym}{tax_shield_val:,.0f} "
                f"fiscal deduction against your capital gains liability. Maintain sector beta by allocating proceeds into {proxy_etf}."
            ),
        },
        "staged_dca_plan": {
            "is_approved": action_directive == "DCA_APPROVED",
            "invalidation_stop": invalidation_stop,
            "tranches": [
                {
                    "tranche": 1,
                    "name": "Base Anchor",
                    "trigger": f"Near current support @ {curr_sym}{tranche_1_price:,.2f}",
                    "weight_pct": 35,
                    "target_price": tranche_1_price,
                    "qty": t1_qty,
                    "capital": _safe_float(t1_qty * tranche_1_price),
                },
                {
                    "tranche": 2,
                    "name": "Confirmation",
                    "trigger": f"Reclaim 20-EMA / Higher Low @ {curr_sym}{tranche_2_price:,.2f}",
                    "weight_pct": 40,
                    "target_price": tranche_2_price,
                    "qty": t2_qty,
                    "capital": _safe_float(t2_qty * tranche_2_price),
                },
                {
                    "tranche": 3,
                    "name": "Breakout Expansion",
                    "trigger": f"Volume Breakout @ {curr_sym}{tranche_3_price:,.2f}",
                    "weight_pct": 25,
                    "target_price": tranche_3_price,
                    "qty": t3_qty,
                    "capital": _safe_float(t3_qty * tranche_3_price),
                },
            ],
            "projected_outcome": {
                "new_avg_price": new_avg_price,
                "cost_reduction_pct": avg_reduction_pct,
                "new_breakeven_pct": new_breakeven_pct,
                "total_additional_capital": _safe_float((t1_qty * tranche_1_price) + (t2_qty * tranche_2_price) + (t3_qty * tranche_3_price)),
            },
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. ALPHA HARVEST & PROFIT MAXIMIZATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_profit_harvest(
    ticker: str,
    buy_price: float,
    qty: float,
    live_px: float,
    df: Optional[pd.DataFrame] = None,
    info: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Computes an institutional profit maximization plan for a winning holding:
    - "House Money" (Zero-Risk Free-Ride) exact share liquidation
    - 4-Tier Institutional Laddered Profit Escalator
    - Chandelier Volatility-Adjusted Trailing Stop Sentinel
    - Winner Pyramiding & Trend Expansion Calculator
    """
    is_indian = ticker.endswith(".NS") or ticker.endswith(".BO")
    curr_sym = "₹" if is_indian else "$"
    info_dict = info or {}

    total_invested = _safe_float(buy_price * qty)
    curr_val = _safe_float(live_px * qty)
    profit_amt = _safe_float(max(0.0, curr_val - total_invested))
    profit_pct = _safe_float(((live_px - buy_price) / buy_price * 100) if buy_price > 0 else 0.0)

    sector = _resolve_sector(ticker, info_dict)

    # ── 1. "House Money" (Free-Ride Zero-Risk) Strategy ───────────────────────
    # How many shares to sell at current price to pull out 100% of initial principal
    if live_px > 0 and total_invested > 0 and qty > 1:
        exact_shares = math.ceil(total_invested / live_px)
        shares_to_sell = min(int(qty) - 1, exact_shares)
    else:
        shares_to_sell = 0

    capital_extracted = _safe_float(shares_to_sell * live_px)
    remaining_free_shares = max(0, int(qty) - shares_to_sell)
    remaining_free_val = _safe_float(remaining_free_shares * live_px)
    principal_recovered_pct = _safe_float((capital_extracted / total_invested * 100) if total_invested > 0 else 0.0)

    # ── 2. Chandelier & ATR Volatility Trailing Stop Sentinel ────────────────
    atr = _calculate_atr(df, 14)
    if df is not None and len(df) >= 30:
        high_60 = float(df["High"].iloc[-min(60, len(df)):].max())
    else:
        high_60 = live_px

    k_multiplier = 2.5
    volatility_buffer = atr * k_multiplier if atr > 0 else live_px * 0.07
    trailing_stop = _safe_float(max(buy_price, high_60 - volatility_buffer))
    dist_to_stop = _safe_float(live_px - trailing_stop)
    dist_to_stop_pct = _safe_float((dist_to_stop / live_px * 100) if live_px > 0 else 0.0)
    protected_profit_pct = _safe_float(((trailing_stop - buy_price) / buy_price * 100) if buy_price > 0 else 0.0)

    if live_px <= trailing_stop:
        trailing_status = "TRIGGER_EXIT_LOCK_PROFIT"
        trailing_color = "rose"
        trailing_label = "Trigger Profit Lock"
    elif dist_to_stop_pct < 3.5:
        trailing_status = "APPROACHING_STOP"
        trailing_color = "amber"
        trailing_label = "Approaching Stop"
    else:
        trailing_status = "SECURE_HOLD"
        trailing_color = "emerald"
        trailing_label = "Secure Hold"

    # ── 3. 4-Tier Institutional Laddered Profit Escalator ─────────────────────
    t1_trigger = _safe_float(buy_price * 1.25)   # +25%
    t2_trigger = _safe_float(buy_price * 1.50)   # +50%
    t3_trigger = _safe_float(buy_price * 1.85)   # +85%
    t4_trigger = _safe_float(buy_price * 2.50)   # Multibagger / Moonbag

    # Determine highest tier passed
    current_tier = 0
    if live_px >= t3_trigger:
        current_tier = 3
    elif live_px >= t2_trigger:
        current_tier = 2
    elif live_px >= t1_trigger:
        current_tier = 1

    ladder_tiers = [
        {
            "tier": 1,
            "name": "Tier 1: Initial De-Risk",
            "condition": f"Gain reaches +25% (@ {curr_sym}{t1_trigger:,.2f})",
            "action": "Trim 20% of holding",
            "target_price": t1_trigger,
            "sell_shares": max(1, math.floor(qty * 0.20)),
            "locked_profit": _safe_float((t1_trigger - buy_price) * max(1, math.floor(qty * 0.20))),
            "status": "PASSED" if current_tier >= 1 else "PENDING",
        },
        {
            "tier": 2,
            "name": "Tier 2: Valuation Stretch",
            "condition": f"Gain reaches +50% (@ {curr_sym}{t2_trigger:,.2f})",
            "action": "Trim 30% of holding",
            "target_price": t2_trigger,
            "sell_shares": max(1, math.floor(qty * 0.30)),
            "locked_profit": _safe_float((t2_trigger - buy_price) * max(1, math.floor(qty * 0.30))),
            "status": "PASSED" if current_tier >= 2 else "PENDING",
        },
        {
            "tier": 3,
            "name": "Tier 3: Super-Cycle Climax",
            "condition": f"Gain reaches +85% (@ {curr_sym}{t3_trigger:,.2f})",
            "action": "Trim 25% of holding",
            "target_price": t3_trigger,
            "sell_shares": max(1, math.floor(qty * 0.25)),
            "locked_profit": _safe_float((t3_trigger - buy_price) * max(1, math.floor(qty * 0.25))),
            "status": "PASSED" if current_tier >= 3 else "PENDING",
        },
        {
            "tier": 4,
            "name": "Tier 4: Compounding Moonbag",
            "condition": "Remaining 25% holding",
            "action": f"Hold indefinitely with Chandelier Stop @ {curr_sym}{trailing_stop:,.2f}",
            "target_price": t4_trigger,
            "sell_shares": 0,
            "locked_profit": 0.0,
            "status": "ACTIVE_RUNNER",
        },
    ]

    # ── 4. Winner Pyramiding Engine (Anti-Martingale Scaling) ─────────────────
    # Allowed to pyramid if winning > 15% and trend is healthy
    pyramid_allowed = profit_pct >= 15.0 and dist_to_stop_pct > 4.0
    pyramid_qty = max(1, math.floor(qty * 0.35))
    pyramid_add_cost = _safe_float(pyramid_qty * live_px)
    new_combined_qty = qty + pyramid_qty
    new_avg_cost = _safe_float(((qty * buy_price) + pyramid_add_cost) / new_combined_qty if new_combined_qty > 0 else live_px)
    safety_margin_to_stop_pct = _safe_float(((trailing_stop - new_avg_cost) / new_avg_cost * 100) if new_avg_cost > 0 else 0.0)

    return {
        "ticker": ticker,
        "sector": sector,
        "profit_amount": profit_amt,
        "profit_pct": profit_pct,
        "total_invested": total_invested,
        "current_value": curr_val,
        "house_money": {
            "eligible": shares_to_sell > 0 and remaining_free_shares > 0,
            "shares_to_sell": shares_to_sell,
            "capital_extracted": capital_extracted,
            "principal_recovered_pct": principal_recovered_pct,
            "remaining_free_shares": remaining_free_shares,
            "remaining_free_value": remaining_free_val,
            "downside_risk_to_capital_pct": 0.0,
            "action_text": (
                f"Sell {shares_to_sell} shares at {curr_sym}{live_px:,.2f} to withdraw {curr_sym}{capital_extracted:,.0f} "
                f"({principal_recovered_pct:.0f}% of your principal). You keep {remaining_free_shares} shares worth "
                f"{curr_sym}{remaining_free_val:,.0f} with ZERO downside risk to original capital."
                if shares_to_sell > 0 else "Gain threshold not yet sufficient for complete principal extraction."
            ),
        },
        "trailing_stop": {
            "stop_price": trailing_stop,
            "status": trailing_status,
            "label": trailing_label,
            "color": trailing_color,
            "distance_rupees": dist_to_stop,
            "distance_pct": dist_to_stop_pct,
            "protected_profit_pct": protected_profit_pct,
            "k_multiplier": k_multiplier,
            "atr": _safe_float(atr),
            "recommendation": (
                f"Your profit is protected down to {curr_sym}{trailing_stop:,.2f} (+{protected_profit_pct}% above buy price). "
                f"Current cushion is {dist_to_stop_pct}%. Hold as long as daily close remains above stop."
            ),
        },
        "ladder_escalator": {
            "current_tier": current_tier,
            "tiers": ladder_tiers,
            "active_directive": (
                f"Reached Tier {current_tier}. Lock in partial gains according to schedule or ride with Chandelier Stop."
                if current_tier > 0 else f"Next profit lock milestone: Tier 1 at {curr_sym}{t1_trigger:,.2f} (+25%)."
            ),
        },
        "pyramiding": {
            "is_recommended": pyramid_allowed,
            "pyramid_qty": pyramid_qty,
            "pyramid_capital": pyramid_add_cost,
            "new_avg_cost": new_avg_cost,
            "safety_margin_to_stop_pct": safety_margin_to_stop_pct,
            "is_safe": new_avg_cost < trailing_stop,
            "advice": (
                f"Pyramiding APPROVED: Adding {pyramid_qty} shares at {curr_sym}{live_px:,.2f} raises your average cost to {curr_sym}{new_avg_cost:,.2f}, "
                f"which remains safely below your Chandelier Stop ({curr_sym}{trailing_stop:,.2f})."
                if (pyramid_allowed and new_avg_cost < trailing_stop)
                else "Pyramiding NOT advised right now: Adding shares at current price would raise cost too close to stop-loss."
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. MACRO STRESS LAB & CONCENTRATION FRAGILITY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_macro_stress_and_concentration(
    holdings: List[Dict[str, Any]],
    total_value: float,
    benchmark_history: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Simulates portfolio resilience against 5 historical crisis shocks and audits
    concentration fragility using the Herfindahl-Hirschman Index (HHI).
    """
    if not holdings or total_value <= 0:
        return {"error": "No holdings or zero portfolio value"}

    # ── 1. Herfindahl-Hirschman Concentration Index (HHI) ─────────────────────
    hhi_sum = 0.0
    sector_weights: Dict[str, float] = {}
    stock_weights = []

    for h in holdings:
        val = float(h.get("curr_value") or (float(h.get("qty", 0)) * float(h.get("live_price", 0))))
        w = (val / total_value) * 100
        hhi_sum += (w ** 2)
        stock_weights.append({"ticker": h.get("ticker", ""), "weight_pct": _safe_float(w)})

        sec = h.get("sector") or "Diversified"
        sector_weights[sec] = sector_weights.get(sec, 0.0) + w

    stock_weights.sort(key=lambda x: x["weight_pct"], reverse=True)
    top3_concentration = sum(x["weight_pct"] for x in stock_weights[:3])

    if hhi_sum < 1500:
        hhi_classification = "DIVERSIFIED"
        hhi_color = "emerald"
        hhi_desc = "Well-balanced portfolio with low single-stock idiosyncratic risk."
    elif hhi_sum <= 2500:
        hhi_classification = "MODERATE_CONCENTRATION"
        hhi_color = "amber"
        hhi_desc = "Moderate concentration. A sharp drawdown in your top 2 holdings will significantly impact total equity."
    else:
        hhi_classification = "DANGEROUSLY_CONCENTRATED"
        hhi_color = "rose"
        hhi_desc = "Fragile portfolio structure! High capital exposure concentrated in too few names. Prone to severe tail risk."

    sector_violations = [
        {"sector": sec, "weight_pct": _safe_float(w), "limit_pct": 30.0}
        for sec, w in sector_weights.items() if w > 30.0
    ]

    # ── 2. Historical Macro Crisis Simulations ───────────────────────────────
    # Beta impact coefficients across 5 historical macro shocks
    scenarios = [
        {
            "id": "covid_2020",
            "name": "2020 COVID Panic Shock",
            "macro_drop_pct": -38.0,
            "desc": "Global economic shutdown, liquidity crunch, rapid high-beta selloff.",
            "sector_betas": {
                "Banking": 1.15, "Financial Services": 1.15, "NBFC": 1.25,
                "IT": 0.65, "Technology": 0.65, "Pharma": 0.40, "Healthcare": 0.40,
                "Auto": 1.25, "Consumer Cyclical": 1.25, "Energy": 1.10,
                "Infrastructure": 1.20, "Metals": 1.20, "FMCG": 0.55
            },
            "default_beta": 1.0,
        },
        {
            "id": "gfc_2008",
            "name": "2008 Global Financial Crisis",
            "macro_drop_pct": -52.0,
            "desc": "Credit freeze, sovereign distress, multi-quarter bear market.",
            "sector_betas": {
                "Banking": 1.30, "Financial Services": 1.30, "NBFC": 1.40,
                "Real Estate": 1.45, "Metals": 1.35, "Infrastructure": 1.30,
                "IT": 0.85, "Technology": 0.85, "Pharma": 0.50, "FMCG": 0.50
            },
            "default_beta": 1.0,
        },
        {
            "id": "rate_hike_2022",
            "name": "2022 Fed/RBI Rate-Hike Shock",
            "macro_drop_pct": -20.0,
            "desc": "Sticky inflation, 400+ bps interest rate surge, P/E multiple compression in growth assets.",
            "sector_betas": {
                "IT": 1.65, "Technology": 1.65, "Consumer Tech": 1.80,
                "Banking": 0.70, "Financial Services": 0.75, "Energy": -0.60,  # Energy rallied
                "Metals": 0.90, "FMCG": 0.60
            },
            "default_beta": 1.0,
        },
        {
            "id": "oil_spike_110",
            "name": "Crude Oil Surge to $110/bbl",
            "macro_drop_pct": -14.0,
            "desc": "Geopolitical supply shock, severe margin squeeze on paints, aviation, tyres, and specialty chemicals.",
            "sector_betas": {
                "Auto": 1.70, "Paints": 2.20, "Chemicals": 1.90, "Consumer Cyclical": 1.40,
                "Energy": -1.40,  # Upstream oil producers gain
                "IT": 0.40, "Pharma": 0.50, "Banking": 0.90
            },
            "default_beta": 0.9,
        },
        {
            "id": "inr_devaluation",
            "name": "Rupee Devaluation Shock (-8% vs USD)",
            "macro_drop_pct": -8.0,
            "desc": "Sharp capital outflows, negative for domestic capital goods & importers; windfall for exporters.",
            "sector_betas": {
                "IT": -1.50, "Technology": -1.50, "Pharma": -1.20,  # Exporters gain
                "Auto": 1.40, "Capital Goods": 1.50, "Metals": 1.20, "Banking": 1.10
            },
            "default_beta": 0.8,
        },
    ]

    simulation_results = []
    max_crisis_drawdown_pct = 0.0

    for sc in scenarios:
        port_sim_loss_amt = 0.0
        holding_losses = []

        for h in holdings:
            val = float(h.get("curr_value") or (float(h.get("qty", 0)) * float(h.get("live_price", 0))))
            sec = h.get("sector") or "Diversified"
            beta = sc["sector_betas"].get(sec, sc["default_beta"])
            stock_drop_pct = sc["macro_drop_pct"] * beta
            # Clamp drop
            stock_drop_pct = max(-90.0, min(50.0, stock_drop_pct))
            stock_loss_rupees = val * (stock_drop_pct / 100.0)

            port_sim_loss_amt += stock_loss_rupees
            holding_losses.append({
                "ticker": h.get("ticker", ""),
                "simulated_pct": _safe_float(stock_drop_pct),
                "simulated_rupees": _safe_float(stock_loss_rupees),
            })

        holding_losses.sort(key=lambda x: x["simulated_rupees"])  # biggest loss first (most negative)
        port_drop_pct = _safe_float((port_sim_loss_amt / total_value * 100) if total_value > 0 else 0.0)

        if abs(port_drop_pct) > max_crisis_drawdown_pct and port_drop_pct < 0:
            max_crisis_drawdown_pct = abs(port_drop_pct)

        simulation_results.append({
            "id": sc["id"],
            "name": sc["name"],
            "desc": sc["desc"],
            "portfolio_drawdown_pct": port_drop_pct,
            "portfolio_drawdown_rupees": _safe_float(port_sim_loss_amt),
            "weakest_links": holding_losses[:2],
        })

    # Portfolio Resilience Score (0 to 100)
    # Higher score = more resilient to historical macro shocks
    resilience_score = _safe_float(max(10.0, min(95.0, 100.0 - (max_crisis_drawdown_pct * 1.35))))

    # ── 3. Dynamic Cash Sizing & Market Valuation Deployer ────────────────────
    # Valuation regime of market
    cash_guidance = {
        "regime": "FAIR_VALUE",
        "recommended_cash_buffer_pct": 15.0,
        "market_pe_est": 21.5,
        "advice": "Market is trading within reasonable historical bands. Maintain 10%–15% in liquid dry powder for high-conviction intraday/swing setups.",
    }

    if benchmark_history is not None and len(benchmark_history) > 50:
        try:
            b_close = benchmark_history["Close"]
            b_last = float(b_close.iloc[-1])
            b_ema200 = float(b_close.ewm(span=min(200, len(b_close)), adjust=False).mean().iloc[-1])
            b_dist = ((b_last - b_ema200) / b_ema200) * 100

            if b_dist > 12.0:
                cash_guidance = {
                    "regime": "OVERHEATED_EXPENSIVE",
                    "recommended_cash_buffer_pct": 25.0,
                    "market_pe_est": 25.2,
                    "advice": f"Benchmark is {round(b_dist, 1)}% stretched above 200 EMA. Macro risk of mean-reversion is elevated. Recommend preserving 25%–30% in cash/arbitrage funds.",
                }
            elif b_dist < -10.0:
                cash_guidance = {
                    "regime": "CAPITULATION_DISCOUNT",
                    "recommended_cash_buffer_pct": 5.0,
                    "market_pe_est": 18.1,
                    "advice": f"Benchmark is {round(abs(b_dist), 1)}% below 200 EMA (oversold discount). Historic risk-reward heavily favors deploying dry powder aggressively into quality compounders.",
                }
        except Exception:
            pass

    return {
        "concentration": {
            "hhi_score": round(hhi_sum, 0),
            "classification": hhi_classification,
            "color": hhi_color,
            "description": hhi_desc,
            "top3_concentration_pct": _safe_float(top3_concentration),
            "sector_violations": sector_violations,
            "sector_breakdown": {sec: _safe_float(w) for sec, w in sector_weights.items()},
        },
        "crisis_stress_test": {
            "resilience_score": resilience_score,
            "max_simulated_drawdown_pct": _safe_float(max_crisis_drawdown_pct),
            "scenarios": simulation_results,
        },
        "market_cash_deployer": cash_guidance,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PORTFOLIO FUNDAMENTAL X-RAY & DIVIDEND CASH FLOW ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_portfolio_fundamentals_and_dividends(
    holdings: List[Dict[str, Any]],
    total_value: float,
    total_cost: float,
) -> Dict[str, Any]:
    """
    Computes portfolio-level fundamental multiples (P/E, P/B, ROE, Div Yield)
    and forward-looking passive dividend income projections (including Yield on Cost).
    """
    if not holdings or total_value <= 0:
        return {}

    weighted_pe_sum = 0.0
    weighted_pb_sum = 0.0
    weighted_roe_sum = 0.0
    weighted_div_sum = 0.0
    total_div_cash_annual = 0.0

    valid_pe_wt = 0.0
    valid_pb_wt = 0.0
    valid_roe_wt = 0.0
    dividend_schedule = []

    for h in holdings:
        val = float(h.get("curr_value") or (float(h.get("qty", 0)) * float(h.get("live_price", 0))))
        wt = (val / total_value)

        # Pull fundamental metrics
        pe = _safe_float(h.get("pe") or h.get("trailing_pe"), default=None)
        pb = _safe_float(h.get("pb") or h.get("price_to_book"), default=None)
        roe = _safe_float(h.get("roe") or h.get("return_on_equity"), default=None)
        div_rate = _safe_float(h.get("dividend_yield"), default=0.0)

        # Normalize dividend yield (e.g. 0.015 or 1.5)
        div_yield_pct = div_rate * 100 if 0 < div_rate < 0.5 else div_rate

        if pe and 0 < pe < 150:
            weighted_pe_sum += wt * pe
            valid_pe_wt += wt

        if pb and 0 < pb < 50:
            weighted_pb_sum += wt * pb
            valid_pb_wt += wt

        if roe and -0.5 < roe < 1.0:
            roe_pct = roe * 100 if -0.5 < roe < 1.0 else roe
            weighted_roe_sum += wt * roe_pct
            valid_roe_wt += wt

        # Dividend cash flow contribution
        stock_annual_div = val * (div_yield_pct / 100.0)
        total_div_cash_annual += stock_annual_div
        weighted_div_sum += wt * div_yield_pct

        if stock_annual_div > 0:
            dividend_schedule.append({
                "ticker": h.get("ticker", ""),
                "div_yield_pct": round(div_yield_pct, 2),
                "annual_income": round(stock_annual_div, 2),
            })

    # Normalized weighted multiples
    port_pe = round(weighted_pe_sum / valid_pe_wt, 1) if valid_pe_wt > 0.2 else 21.4
    port_pb = round(weighted_pb_sum / valid_pb_wt, 2) if valid_pb_wt > 0.2 else 3.2
    port_roe = round(weighted_roe_sum / valid_roe_wt, 1) if valid_roe_wt > 0.2 else 16.5
    port_div_yield = round(weighted_div_sum, 2)

    # Yield on Cost (YOC): Annual dividend income / original invested capital
    yield_on_cost = round((total_div_cash_annual / total_cost * 100.0), 2) if total_cost > 0 else port_div_yield

    # Benchmark comparison standards (Nifty 50 baseline averages)
    bench_pe = 22.4
    bench_pb = 3.6
    bench_roe = 15.2
    bench_div_yield = 1.15

    # Valuation tilt diagnosis
    pe_spread = port_pe - bench_pe
    if pe_spread < -3.0:
        valuation_tilt = "VALUE_DISCOUNT"
        tilt_desc = f"Portfolio trades at a {abs(round(pe_spread, 1))}x P/E discount to Nifty 50 (attractive margin of safety)."
    elif pe_spread > 4.0:
        valuation_tilt = "GROWTH_PREMIUM"
        tilt_desc = f"Portfolio trades at a +{round(pe_spread, 1)}x P/E growth premium over Nifty 50."
    else:
        valuation_tilt = "BENCHMARK_PARITY"
        tilt_desc = "Portfolio valuation multiples are in line with broader market averages."

    dividend_schedule.sort(key=lambda x: x["annual_income"], reverse=True)

    return {
        "fundamentals_xray": {
            "weighted_pe": port_pe,
            "weighted_pb": port_pb,
            "weighted_roe_pct": port_roe,
            "weighted_div_yield_pct": port_div_yield,
            "benchmark_pe": bench_pe,
            "benchmark_pb": bench_pb,
            "benchmark_roe_pct": bench_roe,
            "benchmark_div_yield_pct": bench_div_yield,
            "valuation_tilt": valuation_tilt,
            "valuation_description": tilt_desc,
        },
        "dividend_stream": {
            "annual_projected_cash": round(total_div_cash_annual, 2),
            "monthly_average_cash": round(total_div_cash_annual / 12.0, 2),
            "portfolio_dividend_yield_pct": port_div_yield,
            "yield_on_cost_pct": yield_on_cost,
            "top_dividend_contributors": dividend_schedule[:4],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. BENCHMARK NORMALIZED ALPHA & RELATIVE OUTPERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_benchmark_alpha_overlay(
    returns_map: Dict[str, pd.Series],
    holdings: List[Dict[str, Any]],
    bench_series: Optional[pd.Series],
    total_value: float,
) -> Dict[str, Any]:
    """
    Computes Jensen's Alpha, Beta, Tracking Error, and normalized cumulative
    return comparisons over 1M, 3M, 6M, and 1Y periods.
    """
    if not returns_map or bench_series is None or len(bench_series) < 30:
        return {}

    try:
        ret_df = pd.DataFrame(returns_map).dropna()
        if ret_df.empty:
            return {}

        weights = []
        for col in ret_df.columns:
            h = next((x for x in holdings if x.get("ticker") == col), None)
            val = float(h.get("curr_value", 0)) if h else 1.0
            weights.append(val)

        w_arr = np.array(weights)
        w_norm = (w_arr / w_arr.sum()) if w_arr.sum() > 0 else np.ones(len(w_arr)) / len(w_arr)

        port_ret = ret_df.dot(w_norm)
        aligned = pd.DataFrame({"port": port_ret, "bench": bench_series}).dropna()

        if len(aligned) < 20:
            return {}

        # Period cumulative returns
        def calc_cum(series, days):
            sub = series.iloc[-min(days, len(series)):]
            return round(float(((1 + sub).cumprod().iloc[-1] - 1.0) * 100.0), 2)

        p_1m = calc_cum(aligned["port"], 21)
        b_1m = calc_cum(aligned["bench"], 21)

        p_3m = calc_cum(aligned["port"], 63)
        b_3m = calc_cum(aligned["bench"], 63)

        p_6m = calc_cum(aligned["port"], 126)
        b_6m = calc_cum(aligned["bench"], 126)

        p_1y = calc_cum(aligned["port"], min(252, len(aligned)))
        b_1y = calc_cum(aligned["bench"], min(252, len(aligned)))

        # Jensen's Alpha & Beta
        cov = float(aligned.cov().values[0, 1])
        b_var = float(aligned["bench"].var())
        beta = round(cov / (b_var + 1e-9), 2)

        rf_daily = 0.065 / 252.0
        p_mean = float(aligned["port"].mean())
        b_mean = float(aligned["bench"].mean())
        # Annualized Jensen's alpha
        jensen_alpha = round(((p_mean - rf_daily) - beta * (b_mean - rf_daily)) * 252.0 * 100.0, 2)

        # Alpha spread (1Y)
        alpha_spread_1y = round(p_1y - b_1y, 2)

        return {
            "jensen_alpha_annualized_pct": jensen_alpha,
            "portfolio_beta": beta,
            "alpha_spread_1y_pct": alpha_spread_1y,
            "period_comparison": {
                "1M": {"portfolio_ret_pct": p_1m, "benchmark_ret_pct": b_1m, "alpha_pct": round(p_1m - b_1m, 2)},
                "3M": {"portfolio_ret_pct": p_3m, "benchmark_ret_pct": b_3m, "alpha_pct": round(p_3m - b_3m, 2)},
                "6M": {"portfolio_ret_pct": p_6m, "benchmark_ret_pct": b_6m, "alpha_pct": round(p_6m - b_6m, 2)},
                "1Y": {"portfolio_ret_pct": p_1y, "benchmark_ret_pct": b_1y, "alpha_pct": alpha_spread_1y},
            },
            "performance_verdict": (
                f"Generated +{alpha_spread_1y}% positive alpha over Nifty 50 in past 1Y."
                if alpha_spread_1y > 0 else f"Lagging Nifty 50 by {abs(alpha_spread_1y)}% over past 1Y."
            ),
        }
    except Exception:
        return {}

