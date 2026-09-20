from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from desk_engine import evaluate_committee


router = APIRouter(prefix="/api/desk", tags=["multi-desk"])


class DeskContext(BaseModel):
    """Canonical evidence snapshot consumed by the deterministic desk engine."""

    model_config = ConfigDict(extra="allow")

    ticker: str | None = None
    horizon: str = "intraday"
    price: float | None = None

    # Technical context
    vwap: float | None = None
    ema9: float | None = None
    ema21: float | None = None
    ema200: float | None = None
    rsi14: float | None = None
    momentum_30d_pct: float | None = None
    rvol: float | None = None
    orb_status: str | None = None
    supertrend_direction: str | None = None
    atr: float | None = None
    atr_pct: float | None = None
    price_vs_ema20_atr: float | None = None
    delta_absorption: bool | None = None

    # Fundamental / valuation context
    fair_value: float | None = None
    roe_pct: float | None = None
    revenue_growth_pct: float | None = None
    operating_margin_pct: float | None = None
    debt_to_equity: float | None = None
    pe_percentile: float | None = None
    ev_ebitda_percentile: float | None = None

    # Derivatives / volatility context
    futures_buildup: str | None = None
    pcr_oi: float | None = None
    pcr_volume: float | None = None
    iv_percentile: float | None = None
    iv_rv_spread_pct: float | None = None
    put_call_iv_skew_pct: float | None = None

    # Risk / quality context
    volatility_percentile: float | None = None
    max_drawdown_pct: float | None = None
    next_event_days: float | None = None
    price_freshness_sec: float | None = None
    orderbook_available: bool | None = None
    options_available: bool | None = None
    market_open: bool | None = None

    # Optional historical expectancy / sizing context
    historical_win_rate: float | None = None
    historical_avg_win_loss: float | None = None
    account_capital: float | None = None
    account_risk_pct: float | None = Field(default=None, ge=0, le=10)
    entry_price: float | None = None
    direction: str = "LONG"
    stop_atr_multiple: float | None = Field(default=2.0, gt=0, le=10)
    target_r_multiple: float | None = Field(default=2.0, gt=0, le=10)
    target2_r_multiple: float | None = Field(default=3.0, gt=0, le=20)


@router.post("/evaluate")
def evaluate_deterministic_desk(context: DeskContext) -> Dict[str, Any]:
    payload = context.model_dump(exclude_none=True)
    return evaluate_committee(payload)
