import pytest

from desk_engine import (
    build_conflict_matrix,
    calculate_trade_geometry,
    evaluate_committee,
)


def test_committee_is_deterministic_and_exposes_all_desks():
    ctx = {
        "ticker": "TEST",
        "horizon": "intraday",
        "price": 100,
        "fair_value": 125,
        "roe_pct": 18,
        "revenue_growth_pct": 14,
        "operating_margin_pct": 20,
        "debt_to_equity": 80,
        "price": 100,
        "vwap": 98,
        "ema9": 101,
        "ema21": 99,
        "ema200": 90,
        "rsi14": 61,
        "momentum_30d_pct": 12,
        "rvol": 2.3,
        "orb_status": "BREAKOUT",
        "supertrend_direction": "BULLISH",
        "atr": 1.5,
        "atr_pct": 1.5,
        "volatility_percentile": 45,
        "max_drawdown_pct": -12,
        "price_freshness_sec": 30,
        "market_open": True,
        "orderbook_available": True,
        "options_available": True,
        "futures_buildup": "LONG_BUILDUP",
        "pcr_oi": 1.05,
        "iv_percentile": 40,
    }

    first = evaluate_committee(ctx)
    second = evaluate_committee(ctx)

    assert first == second
    assert set(first["desks"]) == {"fundamental", "technical", "derivatives"}
    assert first["risk_gate"]["state"] in {"PASS", "CAUTION", "CONDITIONAL", "VETO"}
    assert first["action_state"] in {"LONG_BIAS", "SHORT_BIAS", "WAIT", "NO_TRADE", "CONFLICTED", "INSUFFICIENT_DATA"}


def test_value_trap_conflict():
    ctx = {
        "price": 100,
        "fair_value": 140,
        "ema200": 115,
        "momentum_30d_pct": -18,
    }

    conflicts = build_conflict_matrix(ctx)
    trap = next(x for x in conflicts if x.code == "VALUE_TRAP")
    assert trap.condition_met is True
    assert trap.severity == "HIGH"


def test_parabolic_top_conflict():
    ctx = {
        "rsi14": 82,
        "pe_percentile": 97,
    }

    conflicts = build_conflict_matrix(ctx)
    top = next(x for x in conflicts if x.code == "PARABOLIC_TOP")
    assert top.condition_met is True


def test_iv_event_conflict():
    ctx = {
        "orb_status": "BREAKOUT",
        "iv_percentile": 91,
        "next_event_days": 1,
    }

    conflicts = build_conflict_matrix(ctx)
    item = next(x for x in conflicts if x.code == "IV_EVENT_TRAP")
    assert item.condition_met is True


def test_capitulation_requires_absorption():
    base = {
        "rsi14": 20,
        "price_vs_ema20_atr": -3.0,
        "delta_absorption": False,
    }
    assert not next(x for x in build_conflict_matrix(base) if x.code == "CAPITULATION_ABSORPTION").condition_met

    base["delta_absorption"] = True
    assert next(x for x in build_conflict_matrix(base) if x.code == "CAPITULATION_ABSORPTION").condition_met


def test_cro_veto_zeroes_sizing():
    ctx = {
        "price": 100,
        "atr": 2,
        "atr_pct": 6,
        "volatility_percentile": 95,
        "max_drawdown_pct": -35,
        "next_event_days": 0.5,
        "price_freshness_sec": 30,
        "account_capital": 100000,
        "account_risk_pct": 1,
        "historical_win_rate": 60,
        "historical_avg_win_loss": 2,
    }
    result = evaluate_committee(ctx)

    assert result["risk_gate"]["state"] == "VETO"
    assert result["risk_gate"]["sizing_cap_pct"] == 0.0
    assert result["trade_geometry"]["shares"] == 0
    assert result["action_state"] == "NO_TRADE"


def test_half_kelly_and_atr_geometry():
    ctx = {
        "price": 100,
        "atr": 2,
        "volatility_percentile": 30,
        "atr_pct": 2,
        "max_drawdown_pct": -10,
        "next_event_days": 20,
        "price_freshness_sec": 10,
        "account_capital": 100000,
        "account_risk_pct": 1,
        "historical_win_rate": 60,
        "historical_avg_win_loss": 2,
    }
    result = evaluate_committee(ctx)
    geom = result["trade_geometry"]

    assert geom["stop_loss"] == pytest.approx(96)
    assert geom["target1"] == pytest.approx(108)
    assert geom["target2"] == pytest.approx(112)
    assert geom["half_kelly_pct"] == pytest.approx(20.0)
    assert geom["shares"] > 0


def test_missing_data_is_explicit():
    result = evaluate_committee({"price": 100})
    assert result["audit"]["no_hidden_data_fallbacks"] is True
    assert "no_hidden_defaults" not in result["audit"]
    assert "fair_value/price" in result["desks"]["fundamental"]["missing_data"]
    assert "vwap" in str(result["desks"]["technical"]["missing_data"])
    assert "pcr_oi" in result["desks"]["derivatives"]["missing_data"]
