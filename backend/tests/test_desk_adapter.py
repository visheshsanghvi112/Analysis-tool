from unittest.mock import patch
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from routers.desk import DeskContext
from services.desk_adapter import build_desk_context
from desk_engine import evaluate_committee


client = TestClient(app)


def _create_mock_df(days=250):
    np.random.seed(42)
    dates = pd.date_range(end="2026-09-18", periods=days, freq="B", tz="Asia/Kolkata")
    price = 100.0 + np.cumsum(np.random.randn(days) * 1.5)
    high = price + np.random.rand(days) * 2.0
    low = price - np.random.rand(days) * 2.0
    volume = np.random.randint(500000, 2000000, size=days)

    return pd.DataFrame({
        "Open": price,
        "High": high,
        "Low": low,
        "Close": price,
        "Volume": volume,
    }, index=dates)


@patch("services.desk_adapter.get_quote")
@patch("services.desk_adapter.get_history")
@patch("services.desk_adapter.get_info")
def test_build_desk_context_mocked(mock_info, mock_hist, mock_quote):
    mock_info.return_value = {
        "marketCap": 2000000000000,
        "sharesOutstanding": 15000000000,
        "currentPrice": 175.5,
        "totalRevenue": 380000000000,
        "freeCashflow": 100000000000,
        "netIncomeToCommon": 95000000000,
        "totalCash": 60000000000,
        "totalDebt": 110000000000,
        "returnOnEquity": 0.45,
        "revenueGrowth": 0.09,
        "operatingMargins": 0.30,
        "debtToEquity": 1.45,
        "trailingPE": 28.5,
        "enterpriseToEbitda": 21.0,
        "beta": 1.15,
        "sector": "Technology",
        "industry": "Consumer Electronics",
    }
    mock_hist.return_value = _create_mock_df(250)
    mock_quote.return_value = {"price": 175.5, "dayHigh": 177.0, "dayLow": 174.0}

    ctx = build_desk_context("AAPL", horizon="swing")
    assert isinstance(ctx, DeskContext)
    assert ctx.ticker == "AAPL"
    assert ctx.price == pytest.approx(175.5, rel=1e-2)
    assert ctx.fair_value is not None and ctx.fair_value > 0
    assert ctx.rsi14 is not None
    assert ctx.atr is not None
    assert ctx.roe_pct == pytest.approx(45.0, rel=1e-2)
    assert ctx.revenue_growth_pct == pytest.approx(9.0, rel=1e-2)
    assert ctx.operating_margin_pct == pytest.approx(30.0, rel=1e-2)

    res = evaluate_committee(ctx.model_dump(exclude_none=True))
    assert res["engine"] == "stockiq_deterministic_multidesk"
    assert res["committee_state"] in {"BULLISH", "BEARISH", "CONFLICTED"}
    assert res["action_state"] in {"LONG_BIAS", "SHORT_BIAS", "WAIT", "NO_TRADE", "CONFLICTED"}
    assert res["desks"]["fundamental"]["stance"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert res["trade_geometry"]["available"] is True


@patch("services.desk_adapter.get_history")
@patch("services.desk_adapter.get_info")
def test_build_desk_context_sparse_fallback(mock_info, mock_hist):
    mock_info.return_value = {}
    mock_hist.return_value = pd.DataFrame()

    ctx = build_desk_context("UNKNOWN_TICKER", horizon="intraday")
    assert ctx.ticker == "UNKNOWN_TICKER"
    assert ctx.price is None

    res = evaluate_committee(ctx.model_dump(exclude_none=True))
    assert res["action_state"] == "INSUFFICIENT_DATA"
    assert res["committee_confidence"] < 35


@patch("services.desk_adapter.get_quote")
@patch("services.desk_adapter.get_history")
@patch("services.desk_adapter.get_info")
def test_get_evaluate_ticker_endpoint(mock_info, mock_hist, mock_quote):
    mock_info.return_value = {
        "marketCap": 15000000000000,
        "sharesOutstanding": 6700000000,
        "currentPrice": 2950.0,
        "totalRevenue": 8900000000000,
        "freeCashflow": 450000000000,
        "netIncomeToCommon": 670000000000,
        "totalCash": 200000000000,
        "totalDebt": 3000000000000,
        "returnOnEquity": 0.09,
        "revenueGrowth": 0.12,
        "operatingMargins": 0.16,
        "debtToEquity": 0.40,
        "trailingPE": 24.0,
        "enterpriseToEbitda": 14.5,
        "beta": 0.95,
        "sector": "Energy",
        "industry": "Oil & Gas Refining & Marketing",
    }
    mock_hist.return_value = _create_mock_df(250)
    mock_quote.return_value = {"price": 2950.0}

    response = client.get("/api/desk/evaluate/RELIANCE.NS?horizon=swing&account_capital=500000&account_risk_pct=1.5")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "RELIANCE.NS"
    assert data["engine"] == "stockiq_deterministic_multidesk"
    assert "committee_state" in data
    assert "action_state" in data
    assert "committee_score" in data
    assert "desks" in data
    assert "fundamental" in data["desks"]
    assert "technical" in data["desks"]
    assert "derivatives" in data["desks"]
    assert "conflict_matrix" in data
    assert "risk_gate" in data
    assert "trade_geometry" in data
    assert data["trade_geometry"]["available"] is True
    assert data["trade_geometry"]["entry_price"] > 0
