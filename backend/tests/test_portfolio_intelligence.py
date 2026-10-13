import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services.portfolio_intelligence import (
    evaluate_loss_remediation,
    evaluate_profit_harvest,
    evaluate_macro_stress_and_concentration,
)

client = TestClient(app)

def _mock_price_df(n=100, trend="up", start_px=100.0):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="D")
    prices = [start_px]
    step = 0.005 if trend == "up" else -0.005
    for _ in range(n - 1):
        prices.append(prices[-1] * (1.0 + step + np.random.normal(0, 0.01)))
    df = pd.DataFrame({
        "Open": prices,
        "High": [p * 1.02 for p in prices],
        "Low": [p * 0.98 for p in prices],
        "Close": prices,
        "Volume": [100000] * n,
    }, index=dates)
    return df


def test_loss_remediation_quality_compounder():
    df = _mock_price_df(100, trend="down", start_px=150.0)
    info_quality = {
        "returnOnEquity": 0.22,
        "debtToEquity": 15.0,
        "operatingMargins": 0.24,
        "revenueGrowth": 0.15,
        "sector": "Banking",
    }
    res = evaluate_loss_remediation(
        ticker="HDFCBANK.NS",
        buy_price=1600.0,
        qty=50.0,
        live_px=1400.0,
        df=df,
        info=info_quality,
    )
    assert res["loss_pct"] < 0
    assert res["req_gain_to_breakeven_pct"] > 0
    assert res["filter"]["quality_score"] >= 65
    assert res["filter"]["directive"] == "DCA_APPROVED"
    assert res["breakeven_horizon"]["stock_years_to_breakeven"] > 0
    assert res["tax_harvesting"]["tax_shield_rupees"] > 0
    assert len(res["staged_dca_plan"]["tranches"]) == 3
    assert res["staged_dca_plan"]["is_approved"] is True


def test_loss_remediation_structural_value_trap():
    df = _mock_price_df(100, trend="down", start_px=100.0)
    info_trap = {
        "returnOnEquity": 0.01,
        "debtToEquity": 350.0,
        "operatingMargins": 0.02,
        "revenueGrowth": -0.08,
        "sector": "Metals",
    }
    res = evaluate_loss_remediation(
        ticker="TRAPSTOCK.NS",
        buy_price=500.0,
        qty=100.0,
        live_px=250.0,
        df=df,
        info=info_trap,
    )
    assert res["filter"]["status"] == "STRUCTURAL_VALUE_TRAP"
    assert res["filter"]["directive"] == "STRICT_FIREWALL_DO_NOT_AVERAGE"
    assert res["staged_dca_plan"]["is_approved"] is False
    assert res["tax_harvesting"]["tax_shield_rupees"] == (500.0 - 250.0) * 100.0 * 0.20


def test_profit_harvest_house_money_and_tiers():
    df = _mock_price_df(100, trend="up", start_px=500.0)
    info = {"sector": "Auto"}
    # Bought 100 shares at 500 = 50,000 cost. Live price = 1000 = 100,000 value (+100% gain)
    res = evaluate_profit_harvest(
        ticker="TATAMOTORS.NS",
        buy_price=500.0,
        qty=100.0,
        live_px=1000.0,
        df=df,
        info=info,
    )
    assert res["profit_pct"] == 100.0
    assert res["house_money"]["eligible"] is True
    # To recover 50,000 principal at 1000/share, sell exactly 50 shares
    assert res["house_money"]["shares_to_sell"] == 50
    assert res["house_money"]["remaining_free_shares"] == 50
    assert res["house_money"]["capital_extracted"] == 50000.0
    assert res["trailing_stop"]["stop_price"] >= 500.0
    assert len(res["ladder_escalator"]["tiers"]) == 4


def test_macro_stress_and_concentration_hhi():
    holdings = [
        {"ticker": "TCS.NS", "qty": 10, "buy_price": 3000, "live_price": 3500, "curr_value": 35000, "sector": "IT"},
        {"ticker": "INFY.NS", "qty": 20, "buy_price": 1400, "live_price": 1600, "curr_value": 32000, "sector": "IT"},
        {"ticker": "HDFCBANK.NS", "qty": 15, "buy_price": 1500, "live_price": 1600, "curr_value": 24000, "sector": "Banking"},
    ]
    total_val = 91000.0
    res = evaluate_macro_stress_and_concentration(holdings, total_val)

    assert "concentration" in res
    assert res["concentration"]["hhi_score"] > 0
    # IT sector has 35k + 32k = 67k out of 91k (> 70%), so must trigger sector violation
    assert len(res["concentration"]["sector_violations"]) > 0
    assert res["concentration"]["sector_violations"][0]["sector"] == "IT"

    # Crisis stress testing must produce 5 scenarios
    assert len(res["crisis_stress_test"]["scenarios"]) == 5
    for sc in res["crisis_stress_test"]["scenarios"]:
        assert "portfolio_drawdown_pct" in sc
        assert "weakest_links" in sc


def test_portfolio_intelligence_desk_endpoint():
    payload = {
        "holdings": [
            {"ticker": "RELIANCE.NS", "qty": 10, "buy_price": 2400.0},
            {"ticker": "TCS.NS", "qty": 5, "buy_price": 4200.0},
        ]
    }
    response = client.post("/api/portfolio-intelligence-desk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "holdings" in data
    assert "loss_remediation" in data
    assert "alpha_harvest" in data
    assert "macro_stress_lab" in data
    assert "concentration_audit" in data
    assert "market_cash_deployer" in data


def test_portfolio_fundamentals_and_dividends():
    from services.portfolio_intelligence import evaluate_portfolio_fundamentals_and_dividends
    holdings = [
        {"ticker": "ITC.NS", "qty": 100, "buy_price": 400.0, "live_price": 450.0, "curr_value": 45000.0, "pe": 26.5, "pb": 7.8, "roe": 0.28, "dividend_yield": 0.035},
        {"ticker": "HDFCBANK.NS", "qty": 30, "buy_price": 1500.0, "live_price": 1600.0, "curr_value": 48000.0, "pe": 18.2, "pb": 2.4, "roe": 0.16, "dividend_yield": 0.012},
    ]
    res = evaluate_portfolio_fundamentals_and_dividends(holdings, total_value=93000.0, total_cost=85000.0)
    assert "fundamentals_xray" in res
    assert "dividend_stream" in res
    assert res["fundamentals_xray"]["weighted_pe"] > 0
    assert res["dividend_stream"]["annual_projected_cash"] > 0
    assert res["dividend_stream"]["yield_on_cost_pct"] > 0


def test_portfolio_analyze_day_pnl_and_alpha():
    payload = {
        "holdings": [
            {"ticker": "RELIANCE.NS", "qty": 10, "buy_price": 2400.0},
            {"ticker": "INFY.NS", "qty": 20, "buy_price": 1400.0},
        ]
    }
    response = client.post("/api/portfolio-analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "total_day_pnl" in data["summary"]
    assert "total_day_pnl_pct" in data["summary"]
    assert "fundamentals_xray" in data
    assert "dividend_stream" in data
    assert len(data["holdings"]) == 2
    assert "day_change" in data["holdings"][0]
    assert "day_pnl" in data["holdings"][0]

