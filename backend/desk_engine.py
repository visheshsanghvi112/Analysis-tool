"""
StockIQ Pro — Deterministic Multi-Desk & Conflict Engine.

Design goals:
- Pure deterministic arithmetic/rules. No LLMs, network calls, or hidden state.
- Consume a canonical structured market snapshot produced by existing data engines.
- Separate directional desks (fundamental/technical/derivatives) from the
  non-directional CRO/risk gate.
- Never manufacture unavailable data: missing inputs reduce confidence and are
  surfaced in the result.
- Return structured evidence so every committee conclusion is auditable and testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Dict, List, Optional


EPS = 1e-12


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "pass", "valid"}:
            return True
        if v in {"false", "0", "no", "fail", "invalid"}:
            return False
    return bool(value)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else value * 100.0 if abs(value) <= 2.0 else value


def _direction_from_points(points: int) -> str:
    if points >= 2:
        return "BULLISH"
    if points <= -2:
        return "BEARISH"
    return "NEUTRAL"


@dataclass(frozen=True)
class DeskResult:
    desk: str
    stance: str
    score: float
    confidence: float
    evidence: List[Dict[str, Any]]
    missing_data: List[str]
    flags: List[str]


@dataclass(frozen=True)
class ConflictResult:
    code: str
    severity: str
    title: str
    condition_met: bool
    evidence: List[str]
    diagnostic: str


@dataclass(frozen=True)
class RiskGateResult:
    state: str
    risk_score: float
    confidence: float
    risk_factors: List[Dict[str, Any]]
    hard_blocks: List[str]
    sizing_cap_pct: float
    missing_data: List[str]


def evaluate_fundamental_desk(ctx: Dict[str, Any]) -> DeskResult:
    fair_value = _num(ctx.get("fair_value"))
    price = _num(ctx.get("price"))
    roe = _pct(_num(ctx.get("roe_pct", ctx.get("roe"))))
    revenue_growth = _pct(_num(ctx.get("revenue_growth_pct", ctx.get("revenue_growth"))))
    op_margin = _pct(_num(ctx.get("operating_margin_pct", ctx.get("operating_margin"))))
    debt_to_equity = _num(ctx.get("debt_to_equity"))
    pe_percentile = _num(ctx.get("pe_percentile"))
    ev_ebitda_percentile = _num(ctx.get("ev_ebitda_percentile"))

    points = 0
    possible = 0
    evidence: List[Dict[str, Any]] = []
    missing: List[str] = []
    flags: List[str] = []

    if fair_value is not None and price is not None and price > 0:
        upside = fair_value / price - 1.0
        possible += 2
        if upside >= 0.15:
            points += 2
            state = "BULLISH"
            detail = f"Fair value implies {upside * 100:.1f}% upside."
        elif upside <= -0.15:
            points -= 2
            state = "BEARISH"
            detail = f"Fair value implies {upside * 100:.1f}% downside."
        else:
            state = "NEUTRAL"
            detail = f"Fair value gap is {upside * 100:.1f}%."
        evidence.append({"factor": "Fair-value gap", "status": state, "value": round(upside * 100, 2), "detail": detail})
    else:
        missing.append("fair_value/price")

    if roe is not None:
        possible += 1
        if roe >= 15:
            points += 1
            evidence.append({"factor": "ROE", "status": "BULLISH", "value": round(roe, 2), "detail": "ROE is at or above 15%."})
        elif roe < 5:
            points -= 1
            evidence.append({"factor": "ROE", "status": "BEARISH", "value": round(roe, 2), "detail": "ROE is below 5%."})
        else:
            evidence.append({"factor": "ROE", "status": "NEUTRAL", "value": round(roe, 2), "detail": "ROE is between 5% and 15%."})
    else:
        missing.append("roe_pct")

    if revenue_growth is not None:
        possible += 1
        if revenue_growth >= 10:
            points += 1
            evidence.append({"factor": "Revenue growth", "status": "BULLISH", "value": round(revenue_growth, 2), "detail": "Growth is at or above 10%."})
        elif revenue_growth <= -5:
            points -= 1
            evidence.append({"factor": "Revenue growth", "status": "BEARISH", "value": round(revenue_growth, 2), "detail": "Revenue is contracting by at least 5%."})
        else:
            evidence.append({"factor": "Revenue growth", "status": "NEUTRAL", "value": round(revenue_growth, 2), "detail": "Growth is positive but below the strong-growth threshold."})
    else:
        missing.append("revenue_growth_pct")

    if op_margin is not None:
        possible += 1
        if op_margin >= 15:
            points += 1
            evidence.append({"factor": "Operating margin", "status": "BULLISH", "value": round(op_margin, 2), "detail": "Operating margin is at or above 15%."})
        elif op_margin < 5:
            points -= 1
            evidence.append({"factor": "Operating margin", "status": "BEARISH", "value": round(op_margin, 2), "detail": "Operating margin is below 5%."})
        else:
            evidence.append({"factor": "Operating margin", "status": "NEUTRAL", "value": round(op_margin, 2), "detail": "Operating margin is between 5% and 15%."})
    else:
        missing.append("operating_margin_pct")

    if debt_to_equity is not None:
        possible += 1
        if debt_to_equity > 250:
            points -= 1
            flags.append("HIGH_LEVERAGE")
            evidence.append({"factor": "Debt / equity", "status": "RISK", "value": round(debt_to_equity, 2), "detail": "Debt-to-equity exceeds 250%."})
        else:
            evidence.append({"factor": "Debt / equity", "status": "OK", "value": round(debt_to_equity, 2), "detail": "No high-leverage flag under the deterministic threshold."})
    else:
        missing.append("debt_to_equity")

    valuation_percentile = max(
        x for x in [pe_percentile, ev_ebitda_percentile] if x is not None
    ) if any(x is not None for x in [pe_percentile, ev_ebitda_percentile]) else None

    if valuation_percentile is not None:
        possible += 1
        if valuation_percentile >= 95:
            flags.append("EXTREME_VALUATION")
            evidence.append({"factor": "Valuation percentile", "status": "RISK", "value": round(valuation_percentile, 1), "detail": "At least one supplied valuation metric is in the top 5% of its reference distribution."})
        elif valuation_percentile >= 80:
            evidence.append({"factor": "Valuation percentile", "status": "ELEVATED", "value": round(valuation_percentile, 1), "detail": "Valuation is elevated relative to its supplied historical distribution."})
        else:
            evidence.append({"factor": "Valuation percentile", "status": "OK", "value": round(valuation_percentile, 1), "detail": "Valuation is below the elevated-percentile threshold."})
    else:
        missing.append("pe_percentile/ev_ebitda_percentile")

    denom = max(possible, 1)
    score = _clamp(50 + (points / denom) * 50, 0, 100)
    confidence = _clamp(100 * (possible / 6.0), 0, 100)
    return DeskResult(
        desk="FUNDAMENTAL",
        stance=_direction_from_points(points),
        score=round(score, 1),
        confidence=round(confidence, 1),
        evidence=evidence,
        missing_data=sorted(set(missing)),
        flags=sorted(set(flags)),
    )


def evaluate_technical_desk(ctx: Dict[str, Any]) -> DeskResult:
    price = _num(ctx.get("price"))
    vwap = _num(ctx.get("vwap"))
    ema9 = _num(ctx.get("ema9"))
    ema21 = _num(ctx.get("ema21"))
    ema200 = _num(ctx.get("ema200"))
    rsi = _num(ctx.get("rsi14", ctx.get("rsi")))
    momentum30 = _pct(_num(ctx.get("momentum_30d_pct", ctx.get("momentum_30d"))))
    rvol = _num(ctx.get("rvol"))
    orb = str(ctx.get("orb_status", "")).upper()
    supertrend = str(ctx.get("supertrend_direction", "")).upper()

    points = 0
    possible = 0
    evidence: List[Dict[str, Any]] = []
    missing: List[str] = []
    flags: List[str] = []

    if price is not None and vwap is not None and vwap > 0:
        possible += 1
        d = (price / vwap - 1.0) * 100
        if d >= 0.25:
            points += 1
            evidence.append({"factor": "VWAP location", "status": "BULLISH", "value": round(d, 2), "detail": f"Price is {d:.2f}% above session VWAP."})
        elif d <= -0.25:
            points -= 1
            evidence.append({"factor": "VWAP location", "status": "BEARISH", "value": round(d, 2), "detail": f"Price is {d:.2f}% below session VWAP."})
        else:
            evidence.append({"factor": "VWAP location", "status": "NEUTRAL", "value": round(d, 2), "detail": "Price is close to session VWAP."})
    else:
        missing.append("price/vwap")

    if ema9 is not None and ema21 is not None:
        possible += 1
        if ema9 > ema21:
            points += 1
            evidence.append({"factor": "EMA 9/21", "status": "BULLISH", "value": round(ema9 - ema21, 4), "detail": "EMA 9 is above EMA 21."})
        elif ema9 < ema21:
            points -= 1
            evidence.append({"factor": "EMA 9/21", "status": "BEARISH", "value": round(ema9 - ema21, 4), "detail": "EMA 9 is below EMA 21."})
        else:
            evidence.append({"factor": "EMA 9/21", "status": "NEUTRAL", "value": 0.0, "detail": "EMA 9 equals EMA 21."})
    else:
        missing.append("ema9/ema21")

    if price is not None and ema200 is not None and ema200 > 0:
        possible += 1
        if price > ema200:
            points += 1
            evidence.append({"factor": "200 EMA anchor", "status": "BULLISH", "value": round((price / ema200 - 1) * 100, 2), "detail": "Price is above the 200 EMA."})
        elif price < ema200:
            points -= 1
            evidence.append({"factor": "200 EMA anchor", "status": "BEARISH", "value": round((price / ema200 - 1) * 100, 2), "detail": "Price is below the 200 EMA."})
        else:
            evidence.append({"factor": "200 EMA anchor", "status": "NEUTRAL", "value": 0.0, "detail": "Price is at the 200 EMA."})
    else:
        missing.append("price/ema200")

    if rsi is not None:
        possible += 1
        if rsi >= 80:
            flags.append("EXTREME_RSI")
            points -= 1
            evidence.append({"factor": "RSI", "status": "EXTENDED", "value": round(rsi, 2), "detail": "RSI is >= 80; trend and extension are separated.")
        elif rsi >= 55:
            points += 1
            evidence.append({"factor": "RSI", "status": "BULLISH_MOMENTUM", "value": round(rsi, 2), "detail": "RSI is above 55 without the extreme-extension flag."})
        elif rsi <= 25:
            flags.append("CAPITULATION_ZONE")
            evidence.append({"factor": "RSI", "status": "OVERSOLD", "value": round(rsi, 2), "detail": "RSI is <= 25; this is a mean-reversion/capitulation condition, not automatically bearish continuation."})
        elif rsi < 45:
            points -= 1
            evidence.append({"factor": "RSI", "status": "BEARISH_MOMENTUM", "value": round(rsi, 2), "detail": "RSI is below 45."})
        else:
            evidence.append({"factor": "RSI", "status": "NEUTRAL", "value": round(rsi, 2), "detail": "RSI is between 45 and 55."})
    else:
        missing.append("rsi14")

    if momentum30 is not None:
        possible += 1
        if momentum30 >= 10:
            points += 1
            evidence.append({"factor": "30D momentum", "status": "BULLISH", "value": round(momentum30, 2), "detail": "30D momentum is >= +10%."})
        elif momentum30 <= -10:
            points -= 1
            evidence.append({"factor": "30D momentum", "status": "BEARISH", "value": round(momentum30, 2), "detail": "30D momentum is <= -10%."})
        else:
            evidence.append({"factor": "30D momentum", "status": "NEUTRAL", "value": round(momentum30, 2), "detail": "30D momentum is inside the neutral band."})
    else:
        missing.append("momentum_30d_pct")

    if orb:
        possible += 1
        if orb in {"BREAKOUT", "BULLISH_BREAKOUT"}:
            points += 1
            evidence.append({"factor": "Opening range", "status": "BULLISH", "value": orb, "detail": "Price is above the opening range."})
        elif orb in {"BREAKDOWN", "BEARISH_BREAKDOWN"}:
            points -= 1
            evidence.append({"factor": "Opening range", "status": "BEARISH", "value": orb, "detail": "Price is below the opening range."})
        else:
            evidence.append({"factor": "Opening range", "status": "NEUTRAL", "value": orb, "detail": "No confirmed opening-range break."})
    else:
        missing.append("orb_status")

    if supertrend:
        possible += 1
        if supertrend in {"BULLISH", "UP", "1"}:
            points += 1
        elif supertrend in {"BEARISH", "DOWN", "-1"}:
            points -= 1
        evidence.append({"factor": "Supertrend", "status": "BULLISH" if supertrend in {"BULLISH", "UP", "1"} else "BEARISH" if supertrend in {"BEARISH", "DOWN", "-1"} else "NEUTRAL", "value": supertrend, "detail": "Directional state from the supplied Supertrend engine."})
    else:
        missing.append("supertrend_direction")

    if rvol is not None:
        if rvol >= 2:
            evidence.append({"factor": "RVOL", "status": "HIGH", "value": round(rvol, 2), "detail": "Volume pace is >= 2x the supplied historical comparator."})
        elif rvol <= 0.5:
            flags.append("LOW_VOLUME_CONFIRMATION")
            evidence.append({"factor": "RVOL", "status": "LOW", "value": round(rvol, 2), "detail": "Volume pace is <= 0.5x the supplied historical comparator."})

    denom = max(possible, 1)
    score = _clamp(50 + (points / denom) * 50, 0, 100)
    confidence = _clamp(100 * (possible / 7.0), 0, 100)
    return DeskResult(
        desk="TECHNICAL",
        stance=_direction_from_points(points),
        score=round(score, 1),
        confidence=round(confidence, 1),
        evidence=evidence,
        missing_data=sorted(set(missing)),
        flags=sorted(set(flags)),
    )


def evaluate_derivatives_desk(ctx: Dict[str, Any]) -> DeskResult:
    futures_buildup = str(ctx.get("futures_buildup", "")).upper()
    pcr_oi = _num(ctx.get("pcr_oi"))
    pcr_volume = _num(ctx.get("pcr_volume"))
    iv_percentile = _num(ctx.get("iv_percentile"))
    iv_rv_spread = _num(ctx.get("iv_rv_spread_pct"))
    option_skew = _num(ctx.get("put_call_iv_skew_pct"))

    points = 0
    possible = 0
    evidence: List[Dict[str, Any]] = []
    missing: List[str] = []
    flags: List[str] = []

    if futures_buildup:
        possible += 1
        mapping = {
            "LONG_BUILDUP": 1,
            "SHORT_COVERING": 1,
            "SHORT_BUILDUP": -1,
            "LONG_UNWINDING": -1,
        }
        p = mapping.get(futures_buildup, 0)
        points += p
        evidence.append({
            "factor": "Futures positioning",
            "status": "BULLISH" if p > 0 else "BEARISH" if p < 0 else "NEUTRAL",
            "value": futures_buildup,
            "detail": "Deterministic price/OI regime label supplied by the derivatives engine."
        })
    else:
        missing.append("futures_buildup")

    # PCR is deliberately contextual, not treated as a standalone directional vote.
    if pcr_oi is not None:
        possible += 1
        if pcr_oi >= 1.30:
            evidence.append({"factor": "PCR OI", "status": "ELEVATED_PUT_OI", "value": round(pcr_oi, 3), "detail": "Put OI is elevated; PCR alone does not establish whether this is hedging or writing."})
        elif pcr_oi <= 0.70:
            evidence.append({"factor": "PCR OI", "status": "ELEVATED_CALL_OI", "value": round(pcr_oi, 3), "detail": "Call OI is elevated; PCR alone does not establish whether this is hedging or writing."})
        else:
            evidence.append({"factor": "PCR OI", "status": "BALANCED", "value": round(pcr_oi, 3), "detail": "PCR is inside the neutral reference band."})
    else:
        missing.append("pcr_oi")

    if pcr_volume is not None:
        evidence.append({"factor": "PCR volume", "status": "OBSERVED", "value": round(pcr_volume, 3), "detail": "Volume PCR supplied as contextual derivatives evidence."})

    if iv_percentile is not None:
        possible += 1
        if iv_percentile >= 85:
            flags.append("HIGH_IV")
            evidence.append({"factor": "IV percentile", "status": "HIGH_RISK", "value": round(iv_percentile, 1), "detail": "IV is >= 85th percentile; premium is expensive relative to its reference window."})
        elif iv_percentile <= 25:
            evidence.append({"factor": "IV percentile", "status": "LOW", "value": round(iv_percentile, 1), "detail": "IV is <= 25th percentile."})
        else:
            evidence.append({"factor": "IV percentile", "status": "NORMAL", "value": round(iv_percentile, 1), "detail": "IV is inside the reference middle range."})
    else:
        missing.append("iv_percentile")

    if iv_rv_spread is not None:
        if iv_rv_spread >= 10:
            flags.append("IV_ABOVE_REALIZED")
            evidence.append({"factor": "IV-RV spread", "status": "ELEVATED", "value": round(iv_rv_spread, 2), "detail": "Implied volatility exceeds realized volatility by >= 10 percentage points."})
        else:
            evidence.append({"factor": "IV-RV spread", "status": "NORMAL", "value": round(iv_rv_spread, 2), "detail": "IV-RV spread is below the elevated reference threshold."})

    if option_skew is not None:
        evidence.append({"factor": "Put-call IV skew", "status": "OBSERVED", "value": round(option_skew, 2), "detail": "Skew supplied for contextual interpretation."})

    denom = max(possible, 1)
    score = _clamp(50 + (points / denom) * 50, 0, 100)
    confidence = _clamp(100 * (possible / 3.0), 0, 100)
    return DeskResult(
        desk="DERIVATIVES",
        stance=_direction_from_points(points),
        score=round(score, 1),
        confidence=round(confidence, 1),
        evidence=evidence,
        missing_data=sorted(set(missing)),
        flags=sorted(set(flags)),
    )


def evaluate_risk_gate(ctx: Dict[str, Any]) -> RiskGateResult:
    volatility_percentile = _num(ctx.get("volatility_percentile"))
    atr_pct = _pct(_num(ctx.get("atr_pct")))
    max_drawdown_pct = _pct(_num(ctx.get("max_drawdown_pct", ctx.get("max_drawdown"))))
    event_days = _num(ctx.get("next_event_days"))
    freshness = _num(ctx.get("price_freshness_sec"))
    orderbook_available = _bool(ctx.get("orderbook_available"))
    options_available = _bool(ctx.get("options_available"))
    market_open = _bool(ctx.get("market_open"))

    risk = 0.0
    factors: List[Dict[str, Any]] = []
    blocks: List[str] = []
    missing: List[str] = []

    if volatility_percentile is not None:
        if volatility_percentile >= 90:
            risk += 30
            factors.append({"factor": "Volatility percentile", "severity": "HIGH", "value": round(volatility_percentile, 1)})
        elif volatility_percentile >= 75:
            risk += 18
            factors.append({"factor": "Volatility percentile", "severity": "ELEVATED", "value": round(volatility_percentile, 1)})
        else:
            factors.append({"factor": "Volatility percentile", "severity": "NORMAL", "value": round(volatility_percentile, 1)})
    else:
        missing.append("volatility_percentile")

    if atr_pct is not None:
        if atr_pct >= 5:
            risk += 20
            factors.append({"factor": "ATR %", "severity": "HIGH", "value": round(atr_pct, 2)})
        elif atr_pct >= 3:
            risk += 10
            factors.append({"factor": "ATR %", "severity": "ELEVATED", "value": round(atr_pct, 2)})
        else:
            factors.append({"factor": "ATR %", "severity": "NORMAL", "value": round(atr_pct, 2)})
    else:
        missing.append("atr_pct")

    if max_drawdown_pct is not None:
        dd = abs(max_drawdown_pct)
        if dd >= 30:
            risk += 20
            factors.append({"factor": "Historical max drawdown", "severity": "HIGH", "value": round(max_drawdown_pct, 2)})
        elif dd >= 20:
            risk += 10
            factors.append({"factor": "Historical max drawdown", "severity": "ELEVATED", "value": round(max_drawdown_pct, 2)})
    else:
        missing.append("max_drawdown_pct")

    if event_days is not None:
        if event_days <= 1:
            risk += 25
            factors.append({"factor": "Next event", "severity": "HIGH", "value": round(event_days, 2)})
            blocks.append("EVENT_WITHIN_1_DAY")
        elif event_days <= 3:
            risk += 12
            factors.append({"factor": "Next event", "severity": "ELEVATED", "value": round(event_days, 2)})
    else:
        missing.append("next_event_days")

    if freshness is not None:
        if freshness > 900:
            risk += 15
            factors.append({"factor": "Price freshness", "severity": "HIGH", "value": round(freshness, 1)})
        elif freshness > 300:
            risk += 7
            factors.append({"factor": "Price freshness", "severity": "ELEVATED", "value": round(freshness, 1)})
        else:
            factors.append({"factor": "Price freshness", "severity": "NORMAL", "value": round(freshness, 1)})
    else:
        missing.append("price_freshness_sec")

    if orderbook_available is False:
        risk += 5
        factors.append({"factor": "Order book", "severity": "MISSING", "value": False})

    if options_available is False:
        factors.append({"factor": "Options data", "severity": "MISSING", "value": False})

    if market_open is False:
        factors.append({"factor": "Market session", "severity": "CLOSED", "value": False})

    risk = _clamp(risk, 0, 100)
    if risk >= 65 or blocks:
        state = "VETO"
        sizing_cap = 0.0
    elif risk >= 40:
        state = "CONDITIONAL"
        sizing_cap = 0.50
    elif risk >= 20:
        state = "CAUTION"
        sizing_cap = 0.75
    else:
        state = "PASS"
        sizing_cap = 1.0

    observed = 0
    for key in (
        volatility_percentile,
        atr_pct,
        max_drawdown_pct,
        event_days,
        freshness,
    ):
        observed += int(key is not None)
    confidence = _clamp(observed / 5.0 * 100, 0, 100)

    return RiskGateResult(
        state=state,
        risk_score=round(risk, 1),
        confidence=round(confidence, 1),
        risk_factors=factors,
        hard_blocks=sorted(set(blocks)),
        sizing_cap_pct=round(sizing_cap, 2),
        missing_data=sorted(set(missing)),
    )


def _conflict_value_trap(ctx: Dict[str, Any]) -> ConflictResult:
    fair_value = _num(ctx.get("fair_value"))
    price = _num(ctx.get("price"))
    ema200 = _num(ctx.get("ema200"))
    momentum = _pct(_num(ctx.get("momentum_30d_pct", ctx.get("momentum_30d"))))

    upside = fair_value / price - 1 if fair_value is not None and price not in (None, 0) else None
    below_ema = price is not None and ema200 not in (None, 0) and price < ema200
    met = bool(upside is not None and upside >= 0.30 and below_ema and momentum is not None and momentum <= -15)

    return ConflictResult(
        code="VALUE_TRAP",
        severity="HIGH" if met else "NONE",
        title="Value Trap",
        condition_met=met,
        evidence=[
            f"Fair-value upside: {upside * 100:.1f}%" if upside is not None else "Fair-value upside unavailable",
            f"Price below 200 EMA: {below_ema}",
            f"30D momentum: {momentum:.1f}%" if momentum is not None else "30D momentum unavailable",
        ],
        diagnostic="Fundamental value appears favorable while long-term price structure and momentum are deteriorating." if met else "Value-trap condition not met.",
    )


def _conflict_parabolic_top(ctx: Dict[str, Any]) -> ConflictResult:
    rsi = _num(ctx.get("rsi14", ctx.get("rsi")))
    pe_pct = _num(ctx.get("pe_percentile"))
    ev_pct = _num(ctx.get("ev_ebitda_percentile"))
    valuation_pct = max(x for x in [pe_pct, ev_pct] if x is not None) if any(x is not None for x in [pe_pct, ev_pct]) else None
    met = bool(rsi is not None and rsi >= 80 and valuation_pct is not None and valuation_pct >= 95)

    return ConflictResult(
        code="PARABOLIC_TOP",
        severity="HIGH" if met else "NONE",
        title="Parabolic Top Risk",
        condition_met=met,
        evidence=[
            f"RSI: {rsi:.1f}" if rsi is not None else "RSI unavailable",
            f"Valuation percentile: {valuation_pct:.1f}" if valuation_pct is not None else "Valuation percentile unavailable",
        ],
        diagnostic="Momentum is extreme while at least one supplied valuation metric is in the top 5% of its reference distribution." if met else "Parabolic-top condition not met.",
    )


def _conflict_iv_crush(ctx: Dict[str, Any]) -> ConflictResult:
    orb = str(ctx.get("orb_status", "")).upper()
    iv = _num(ctx.get("iv_percentile"))
    event_days = _num(ctx.get("next_event_days"))
    met = bool(
        orb in {"BREAKOUT", "BULLISH_BREAKOUT"}
        and iv is not None and iv >= 85
        and event_days is not None and event_days <= 2
    )
    return ConflictResult(
        code="IV_EVENT_TRAP",
        severity="HIGH" if met else "NONE",
        title="Volatility Event Risk",
        condition_met=met,
        evidence=[
            f"ORB status: {orb or 'unavailable'}",
            f"IV percentile: {iv:.1f}" if iv is not None else "IV percentile unavailable",
            f"Next event: {event_days:.2f} days" if event_days is not None else "Next event unavailable",
        ],
        diagnostic="Breakout conditions coincide with unusually expensive implied volatility and a near-term event." if met else "IV-event conflict not met.",
    )


def _conflict_capitulation(ctx: Dict[str, Any]) -> ConflictResult:
    rsi = _num(ctx.get("rsi14", ctx.get("rsi")))
    price_vs_ema20_atr = _num(ctx.get("price_vs_ema20_atr"))
    absorption = _bool(ctx.get("delta_absorption"))
    met = bool(rsi is not None and rsi <= 25 and price_vs_ema20_atr is not None and price_vs_ema20_atr <= -2.5 and absorption is True)

    return ConflictResult(
        code="CAPITULATION_ABSORPTION",
        severity="MEDIUM" if met else "NONE",
        title="Capitulation / Absorption",
        condition_met=met,
        evidence=[
            f"RSI: {rsi:.1f}" if rsi is not None else "RSI unavailable",
            f"Price vs EMA20 in ATRs: {price_vs_ema20_atr:.2f}" if price_vs_ema20_atr is not None else "EMA20 distance unavailable",
            f"Delta absorption: {absorption}",
        ],
        diagnostic="Extreme oversold conditions coincide with a supplied absorption signal; this is a mean-reversion condition, not a guaranteed reversal." if met else "Capitulation-absorption condition not met.",
    )


def build_conflict_matrix(ctx: Dict[str, Any]) -> List[ConflictResult]:
    checks = [
        _conflict_value_trap(ctx),
        _conflict_parabolic_top(ctx),
        _conflict_iv_crush(ctx),
        _conflict_capitulation(ctx),
    ]
    return checks


def calculate_trade_geometry(ctx: Dict[str, Any], risk_gate: RiskGateResult) -> Dict[str, Any]:
    entry = _num(ctx.get("entry_price", ctx.get("price")))
    atr = _num(ctx.get("atr"))
    account = _num(ctx.get("account_capital"))
    risk_pct = _num(ctx.get("account_risk_pct"))
    p_win = _num(ctx.get("historical_win_rate"))
    avg_win_loss = _num(ctx.get("historical_avg_win_loss"))
    direction = str(ctx.get("direction", "LONG")).upper()

    if entry is None or atr is None or atr <= 0:
        return {
            "available": False,
            "reason": "entry_price/price and ATR are required for deterministic trade geometry.",
        }

    atr_multiple = _num(ctx.get("stop_atr_multiple")) or 2.0
    risk_per_share = atr * atr_multiple
    stop = entry - risk_per_share if direction != "SHORT" else entry + risk_per_share
    target1_r = _num(ctx.get("target_r_multiple")) or 2.0
    target2_r = _num(ctx.get("target2_r_multiple")) or 3.0
    target1 = entry + risk_per_share * target1_r if direction != "SHORT" else entry - risk_per_share * target1_r
    target2 = entry + risk_per_share * target2_r if direction != "SHORT" else entry - risk_per_share * target2_r

    kelly_half_pct = None
    if p_win is not None and avg_win_loss is not None and avg_win_loss > 0:
        p = _clamp(p_win / 100 if p_win > 1 else p_win, 0, 1)
        q = 1 - p
        kelly = ((p * avg_win_loss) - q) / avg_win_loss
        kelly_half_pct = max(0.0, 0.5 * kelly * 100)

    sizing_pct = None
    shares = None
    if account is not None and account > 0 and risk_pct is not None and risk_pct > 0:
        base_risk_pct = _clamp(risk_pct, 0, 10)
        sizing_pct = min(base_risk_pct, (kelly_half_pct if kelly_half_pct is not None else base_risk_pct))
        sizing_pct *= risk_gate.sizing_cap_pct
        risk_budget = account * sizing_pct / 100
        shares = int(max(0, risk_budget / max(risk_per_share, EPS)))

    return {
        "available": True,
        "direction": direction,
        "entry_price": round(entry, 4),
        "atr": round(atr, 4),
        "stop_atr_multiple": round(atr_multiple, 3),
        "risk_per_share": round(risk_per_share, 4),
        "stop_loss": round(stop, 4),
        "target1": round(target1, 4),
        "target2": round(target2, 4),
        "target1_r": round(target1_r, 2),
        "target2_r": round(target2_r, 2),
        "half_kelly_pct": round(kelly_half_pct, 3) if kelly_half_pct is not None else None,
        "effective_account_risk_pct": round(sizing_pct, 3) if sizing_pct is not None else None,
        "shares": shares,
        "sizing_cap_from_cro": risk_gate.sizing_cap_pct,
    }


def evaluate_committee(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate the supplied canonical snapshot.

    No network requests are made. Given the same context, output is deterministic.
    """
    normalized = dict(ctx or {})
    fundamentals = evaluate_fundamental_desk(normalized)
    technicals = evaluate_technical_desk(normalized)
    derivatives = evaluate_derivatives_desk(normalized)
    risk = evaluate_risk_gate(normalized)

    conflicts = build_conflict_matrix(normalized)
    active_conflicts = [c for c in conflicts if c.condition_met]

    directional_scores = [
        fundamentals.score,
        technicals.score,
        derivatives.score,
    ]
    observed_confidence = [
        fundamentals.confidence,
        technicals.confidence,
        derivatives.confidence,
    ]

    # Weighting is intentionally transparent and stable.
    # Fundamental matters most for longer-horizon contexts; technical dominates
    # when a snapshot is explicitly marked intraday.
    horizon = str(normalized.get("horizon", "intraday")).lower()
    if horizon == "intraday":
        weights = {"fundamental": 0.15, "technical": 0.60, "derivatives": 0.25}
    elif horizon in {"swing", "short_term"}:
        weights = {"fundamental": 0.30, "technical": 0.50, "derivatives": 0.20}
    else:
        weights = {"fundamental": 0.55, "technical": 0.25, "derivatives": 0.20}

    committee_score = (
        fundamentals.score * weights["fundamental"]
        + technicals.score * weights["technical"]
        + derivatives.score * weights["derivatives"]
    )
    confidence = (
        fundamentals.confidence * weights["fundamental"]
        + technicals.confidence * weights["technical"]
        + derivatives.confidence * weights["derivatives"]
    )

    stance_votes = [
        fundamentals.stance,
        technicals.stance,
        derivatives.stance,
    ]
    bull_votes = stance_votes.count("BULLISH")
    bear_votes = stance_votes.count("BEARISH")

    extension = "EXTENDED" in technicals.flags or "EXTREME_RSI" in technicals.flags
    if risk.state == "VETO":
        action_state = "NO_TRADE"
    elif not normalized.get("price") or confidence < 35:
        action_state = "INSUFFICIENT_DATA"
    elif active_conflicts and any(c.severity == "HIGH" for c in active_conflicts):
        action_state = "WAIT"
    elif bull_votes >= 2 and bear_votes == 0:
        action_state = "WAIT" if extension else "LONG_BIAS"
    elif bear_votes >= 2 and bull_votes == 0:
        action_state = "SHORT_BIAS"
    else:
        action_state = "CONFLICTED"

    if bull_votes >= 2 and bear_votes == 0:
        committee_state = "BULLISH"
    elif bear_votes >= 2 and bull_votes == 0:
        committee_state = "BEARISH"
    else:
        committee_state = "CONFLICTED"

    top_conflicts = sorted(
        active_conflicts,
        key=lambda c: 0 if c.severity == "HIGH" else 1,
    )

    geometry = calculate_trade_geometry(normalized, risk)

    return {
        "engine": "stockiq_deterministic_multidesk",
        "engine_version": "1.0.0",
        "deterministic": True,
        "horizon": horizon,
        "committee_state": committee_state,
        "action_state": action_state,
        "committee_score": round(_clamp(committee_score, 0, 100), 1),
        "committee_confidence": round(_clamp(confidence, 0, 100), 1),
        "weights": weights,
        "desks": {
            "fundamental": asdict(fundamentals),
            "technical": asdict(technicals),
            "derivatives": asdict(derivatives),
        },
        "risk_gate": asdict(risk),
        "conflict_matrix": [asdict(c) for c in conflicts],
        "active_conflicts": [asdict(c) for c in top_conflicts],
        "trade_geometry": geometry,
        "audit": {
            "input_keys": sorted(normalized.keys()),
            "missing_fields_by_desk": {
                "fundamental": fundamentals.missing_data,
                "technical": technicals.missing_data,
                "derivatives": derivatives.missing_data,
                "risk": risk.missing_data,
            },
            "no_hidden_defaults": True,
            "llm_used": False,
            "network_calls": False,
        },
    }


__all__ = [
    "build_conflict_matrix",
    "calculate_trade_geometry",
    "evaluate_committee",
    "evaluate_derivatives_desk",
    "evaluate_fundamental_desk",
    "evaluate_risk_gate",
    "evaluate_technical_desk",
]
