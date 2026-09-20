from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from desk_engine import DeskContext, evaluate_committee
from services.desk_adapter import build_desk_context


router = APIRouter(prefix="/api/desk", tags=["multi-desk"])


@router.post("/evaluate")
def evaluate_deterministic_desk(context: DeskContext) -> Dict[str, Any]:
    payload = context.model_dump(exclude_none=True)
    return evaluate_committee(payload)


@router.get("/evaluate/{ticker}")
def evaluate_ticker_committee(
    ticker: str,
    horizon: str = Query("intraday", description="intraday, swing, or long_term"),
    account_capital: Optional[float] = Query(100000.0, description="Account capital"),
    account_risk_pct: Optional[float] = Query(1.0, description="Account risk percent per trade"),
    direction: str = Query("LONG", description="LONG or SHORT"),
) -> Dict[str, Any]:
    """
    Evaluates the deterministic multi-desk committee for any live ticker.
    Gathers live technicals, fundamentals, and volatility metrics automatically.
    """
    try:
        ctx = build_desk_context(
            ticker=ticker,
            horizon=horizon,
            account_capital=account_capital,
            account_risk_pct=account_risk_pct,
            direction=direction,
        )
        payload = ctx.model_dump(exclude_none=True)
        res = evaluate_committee(payload)
        res["ticker"] = ticker.strip().upper()
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate multi-desk committee for {ticker}: {str(e)}"
        )
