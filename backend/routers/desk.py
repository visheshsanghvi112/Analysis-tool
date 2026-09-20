from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from desk_engine import evaluate_committee


router = APIRouter(prefix="/api/desk", tags=["multi-desk"])


class DeskContext(BaseModel):
    """Canonical evidence snapshot consumed by the deterministic desk engine."""

    model_config = ConfigDict(extra="allow")

    ticker: Optional[str] = None
    horizon: str = "intraday"
    price: Optional[float] = None

    # Technical context
    vwap: Optional[float] = None
    ema9: Optional[float] = None
    ema21: Optional[float] = None
    ema200: Optional[float] = None
    rsi14: Optional[float] = None
    momentum_30d_pct: Optional[float] = None
    rvol: Optional[float] = None
    orb_status: Optional[str] = None
    supertrend_direction: Optional[str] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    price_vs_ema20_atr: Optional[float] = None
    delta_absorption: Optional[bool] = None

    # Fundamental / valuation context
    fair_value: Optional[float] = None
    roe_pct: Optional[float] = None
    revenue_growth_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    pe_percentile: Optional[float] = None
    ev_ebitda_percentile: Optional[float] = None

    # Derivatives / volatility context
    futures_buildup: Optional[str] = None
    pcr_oi: Optional[float] = None
    pcr_volume: Optional[float] = None
    iv_percentile: Optional[float] = None
    iv_rv_spread_pct: Optional[float] = None
    put_call_iv_skew_pct: Optional[float] = None

    # Risk / quality context
    volatility_percentile: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    next_event_days: Optional[float] = None
    price_freshness_sec: Optional[float] = None
    orderbook_available: Optional[bool] = None
    options_available: Optional[bool] = None
    market_open: Optional[bool] = None

    # Optional historical expectancy / sizing context
    historical_win_rate: Optional[float] = None
    historical_avg_win_loss: Optional[float] = None
    account_capital: Optional[float] = None
    account_risk_pct: Optional[float] = Field(default=None, ge=0, le=10)
    entry_price: Optional[float] = None
    direction: str = "LONG"
    stop_atr_multiple: Optional[float] = Field(default=2.0, gt=0, le=10)
    target_r_multiple: Optional[float] = Field(default=2.0, gt=0, le=10)
    target2_r_multiple: Optional[float] = Field(default=3.0, gt=0, le=20)


@router.post("/evaluate")
def evaluate_deterministic_desk(context: DeskContext) -> Dict[str, Any]:
    payload = context.model_dump(exclude_none=True)
    return evaluate_committee(payload)
