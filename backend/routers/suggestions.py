"""
StockIQ Pro — Scrip Suggestions & Reasoning API Router
======================================================

FastAPI endpoints providing real-time, data-driven stock suggestions
with a primary focus on scrips under ₹200 and deep quantitative reasoning.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, Request
from utils.limiter import limiter
from services.suggestions_engine import (
    get_curated_suggestions,
    analyze_single_scrip,
    generate_scrip_basket,
    calculate_custom_position_size,
)

router = APIRouter(prefix="/api/suggestions", tags=["suggestions"])


@router.get("")
@limiter.limit("60/minute")
def get_suggestions(
    request: Request,
    max_price: Optional[float] = Query(200.0, description="Filter stocks below price (e.g., 200, 100, 50, None for all)"),
    setup_type: str = Query("all", description="all, breakout, oversold, pullback, squeeze, high_conviction"),
    sector: str = Query("all", description="Filter by sector"),
    min_conviction: int = Query(50, description="Minimum conviction threshold (0 to 100)"),
    sort_by: str = Query("conviction", description="conviction, risk_reward, rvol, price_asc, price_desc, day_gain, proximity_low"),
    force_refresh: bool = Query(False, description="Set to true to force a fresh market scan"),
) -> Dict[str, Any]:
    """
    Returns curated, data-driven stock suggestions with a primary focus on scrips under ₹200.
    Includes technical setups, entry zones, targets, stop-losses, and deep multi-section reasoning.
    """
    try:
        # If user passes negative or 0 max_price, treat as None (no ceiling)
        effective_max_price = None if (max_price is not None and max_price <= 0) else max_price

        return get_curated_suggestions(
            max_price=effective_max_price,
            setup_type=setup_type,
            sector=sector,
            min_conviction=min_conviction,
            sort_by=sort_by,
            force_refresh=force_refresh,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate stock suggestions: {str(exc)}"
        )


@router.get("/under-200")
@limiter.limit("60/minute")
def get_under_200_suggestions(
    request: Request,
    setup_type: str = Query("all", description="Setup filter"),
    min_conviction: int = Query(50, description="Minimum conviction score"),
    sort_by: str = Query("conviction", description="Sort parameter"),
) -> Dict[str, Any]:
    """
    Quick access endpoint specifically filtered for scrips strictly under ₹200.
    """
    return get_curated_suggestions(
        max_price=200.0,
        setup_type=setup_type,
        sector="all",
        min_conviction=min_conviction,
        sort_by=sort_by,
        force_refresh=False,
    )


@router.get("/evaluate/{ticker}")
@limiter.limit("45/minute")
def evaluate_scrip_suggestion(
    request: Request,
    ticker: str,
) -> Dict[str, Any]:
    """
    Evaluates ANY custom stock ticker on demand and generates the complete
    quantitative suggestion, entry/target plan, and multi-factor reasoning dossier.
    """
    try:
        ticker_clean = ticker.strip().upper()
        if not ticker_clean.endswith(".NS") and not ticker_clean.endswith(".BO") and not ticker_clean.startswith("^"):
            ticker_clean += ".NS"

        result = analyze_single_scrip(ticker_clean)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Unable to generate suggestion analysis for '{ticker_clean}'. Insufficient historical data or invalid scrip."
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate suggestion for '{ticker}': {str(exc)}"
        )


@router.get("/basket")
@limiter.limit("30/minute")
def get_diversified_basket(
    request: Request,
    capital: float = Query(50000.0, description="Total capital to allocate in rupees (e.g. 25000, 50000, 100000)"),
    theme: str = Query("balanced", description="balanced, momentum, value, psu_capex, high_liquidity"),
    max_price: Optional[float] = Query(200.0, description="Max price filter for scrips in basket"),
    max_scrips: int = Query(4, description="Number of diversified scrips in basket (2 to 8)"),
) -> Dict[str, Any]:
    """
    Generates a diversified portfolio basket of high-conviction scrips under ₹200.
    Returns exact share counts, total outlay, cash buffer, and broker-compatible CSV text.
    """
    try:
        effective_max_price = None if (max_price is not None and max_price <= 0) else max_price
        return generate_scrip_basket(
            capital=capital,
            theme=theme,
            max_price=effective_max_price,
            max_scrips=max_scrips,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate diversified basket: {str(exc)}"
        )


@router.get("/stats")
@limiter.limit("60/minute")
def get_suggestion_stats(request: Request) -> Dict[str, Any]:
    """
    Returns high-level statistics across the scanned scrip universe.
    """
    try:
        curated = get_curated_suggestions(max_price=None, force_refresh=False)
        return {
            "total_scanned": curated.get("total_scanned", 0),
            "under_200_count": curated.get("under_200_count", 0),
            "high_conviction_count": curated.get("high_conviction_count", 0),
            "breakout_count": curated.get("breakout_count", 0),
            "btst_count": curated.get("btst_count", 0),
            "golden_count": curated.get("golden_count", 0),
            "dividend_count": curated.get("dividend_count", 0),
            "oversold_count": curated.get("oversold_count", 0),
            "pullback_count": curated.get("pullback_count", 0),
            "squeeze_count": curated.get("squeeze_count", 0),
            "avg_risk_reward": curated.get("avg_risk_reward", 0.0),
            "top_pick": curated.get("top_pick"),
            "timestamp": curated.get("timestamp"),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch suggestion statistics: {str(exc)}"
        )


@router.get("/position-size")
@limiter.limit("45/minute")
def get_position_sizing(
    request: Request,
    ticker: Optional[str] = Query(None, description="Stock ticker, e.g. TATASTEEL.NS"),
    symbol: Optional[str] = Query(None, description="Alternative parameter for stock ticker"),
    capital: float = Query(100000.0, description="Total portfolio trading capital in rupees"),
    risk_pct: float = Query(1.0, description="Risk tolerance percentage (e.g. 1.0 for 1%)"),
    custom_risk_amount: Optional[float] = Query(None, description="Optional fixed risk amount in rupees"),
) -> Dict[str, Any]:
    """
    Calculates position sizing and risk allocation for any ticker given the user's capital.
    Accepts either 'ticker' or 'symbol' query parameter.
    """
    try:
        raw_ticker = ticker or symbol
        if not raw_ticker:
            raise HTTPException(
                status_code=422,
                detail="Either 'ticker' or 'symbol' query parameter must be provided."
            )

        ticker_clean = raw_ticker.strip().upper()
        if not ticker_clean.endswith(".NS") and not ticker_clean.endswith(".BO") and not ticker_clean.startswith("^"):
            ticker_clean += ".NS"

        return calculate_custom_position_size(
            ticker=ticker_clean,
            capital=capital,
            risk_pct=risk_pct,
            custom_risk_amount=custom_risk_amount,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate position sizing for '{ticker or symbol}': {str(exc)}"
        )
