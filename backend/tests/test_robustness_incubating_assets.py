import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from services.intelligent_news_reader import IntelligentNewsReader
from routers.analysis import _compute_quick_metrics
from routers.portfolio import get_monte_carlo, get_portfolio_metrics
from desk_engine import DeskContext, evaluate_committee, calculate_trade_geometry, evaluate_risk_gate
from yf_client import get_fundamentals_data


def test_brokerage_target_regex_no_comma_bug():
    """Ensure regex matches realistic prices and never produces '₹,' or empty price target."""
    reader = IntelligentNewsReader()
    
    # Text with punctuation after target price
    text_with_comma = "Brokerage sets target price, but maintains neutral stance on the stock."
    catalysts = reader._extract_catalysts(text_with_comma)
    for c in catalysts:
        if c.get("type") == "Brokerage Upgrade":
            assert c["highlight"] != "Institutional Price Target Set at ₹,"
            assert c["highlight"] != "Institutional Price Target Set at ₹"

    # Valid price target
    valid_text = "Goldman Sachs initiates buy rating with a target price of Rs. 2,450 for the stock."
    valid_catalysts = reader._extract_catalysts(valid_text)
    upgrades = [c for c in valid_catalysts if c.get("type") == "Brokerage Upgrade"]
    assert len(upgrades) == 1
    assert "₹2,450" in upgrades[0]["highlight"]


def test_quick_metrics_incubating_asset():
    """Test that _compute_quick_metrics succeeds even with only 12 historical sessions."""
    dates = pd.date_range("2026-09-24", periods=12, freq="B")
    mock_df = pd.DataFrame({
        "Open": np.linspace(1700, 1750, 12),
        "High": np.linspace(1720, 1770, 12),
        "Low": np.linspace(1690, 1740, 12),
        "Close": np.linspace(1710, 1752, 12),
        "Volume": np.full(12, 1000000),
    }, index=dates)

    with patch("routers.analysis.get_history", return_value=mock_df), \
         patch("routers.analysis.get_info", return_value={"shortName": "NSE India", "trailingPE": 41.3, "marketCap": 4000000000000}):
        metrics = _compute_quick_metrics("NSE.BO")
        assert metrics is not None
        assert metrics["ticker"] == "NSE.BO"
        assert metrics["current_price"] == pytest.approx(1752.0)
        assert metrics["annual_vol"] > 0
        assert metrics["pe_ratio"] == 41.3
        assert metrics["company_name"] == "NSE India"


def test_monte_carlo_incubating_asset_no_400():
    """Test that Monte Carlo simulation runs cleanly on 12-day incubating asset."""
    dates = pd.date_range("2026-09-24", periods=12, freq="B")
    mock_df = pd.DataFrame({
        "Open": np.linspace(1700, 1750, 12),
        "High": np.linspace(1720, 1770, 12),
        "Low": np.linspace(1690, 1740, 12),
        "Close": np.linspace(1710, 1752, 12),
        "Volume": np.full(12, 1000000),
    }, index=dates)

    with patch("routers.portfolio.get_history", return_value=mock_df):
        res = get_monte_carlo(ticker="NSE.BO", horizon_days=30, simulations=200, simulation_mode="gbm")
        assert res["ticker"] == "NSE.BO"
        assert res["is_incubation"] is True
        assert len(res["historical"]) == 12
        assert len(res["simulated"]) > 0
        assert res["stats"]["current_price"] == pytest.approx(1752.0)
        assert res["stats"]["expected_price_horizon"] > 0


def test_beta_correlation_vasicek_shrinkage_incubating():
    """Test that stocks with 12 sessions get Vasicek-shrunk Beta and Correlation against Nifty."""
    dates = pd.date_range("2026-09-24", periods=12, freq="B")
    mock_stock = pd.DataFrame({
        "Open": np.linspace(1700, 1750, 12),
        "High": np.linspace(1720, 1770, 12),
        "Low": np.linspace(1690, 1740, 12),
        "Close": [1710, 1720, 1715, 1730, 1740, 1735, 1750, 1745, 1760, 1755, 1765, 1752],
        "Volume": np.full(12, 1000000),
    }, index=dates)

    bench_dates = pd.date_range("2026-01-01", periods=250, freq="B")
    np.random.seed(42)
    bench_rets = 1.0 + np.random.normal(0.0005, 0.01, 250)
    mock_bench = pd.DataFrame({
        "Close": 24000.0 * np.cumprod(bench_rets),
    }, index=bench_dates)

    def mock_hist(t, **kwargs):
        if "^" in t or "NSEI" in t:
            return mock_bench
        return mock_stock

    with patch("routers.portfolio.get_history", side_effect=mock_hist):
        analytics = get_portfolio_metrics(ticker="NSE.BO")
        # Beta must be computed and Vasicek-shrunk, not None/NA
        assert analytics["market_metrics"]["beta"] is not None
        assert analytics["market_metrics"]["correlation"] is not None
        assert 0.0 < analytics["market_metrics"]["beta"] < 3.0


def test_trade_geometry_fallback_atr():
    """Test deterministic trade geometry generates valid levels using fallback ATR when ATR not precomputed."""
    ctx = {
        "price": 1752.0,
        "entry_price": 1752.0,
        "direction": "LONG",
        "account_capital": 100000,
        "account_risk_pct": 1.0,
    }
    gate = MagicMock()
    gate.state = "CLEAR"
    gate.sizing_cap_pct = 1.0

    geom = calculate_trade_geometry(ctx, gate)
    assert geom["available"] is True
    assert geom["stop_loss"] < 1752.0
    assert geom["target1"] > 1752.0
    assert geom["shares"] > 0


def test_committee_dynamic_weight_redistribution_non_fo():
    """Verify non-F&O assets redistribute derivatives weight so confidence is not penalized."""
    ctx = {
        "ticker": "NSE.BO",
        "price": 1752.0,
        "horizon": "swing",
        "derivatives_available": False,
        "options_available": False,
        "fair_value": 1850.0,
        "pe": 41.3,
        "roe": 23.0,
        "operating_margin": 81.5,
        "revenue_growth": 9.4,
        "debt_to_equity": 0.0,
        "rsi": 54.0,
        "ema9": 1740.0,
        "ema21": 1720.0,
        "ema200": 1650.0,
        "momentum_30d": 2.5,
    }
    res = evaluate_committee(ctx)
    # Derivatives confidence is 0
    assert res["desks"]["derivatives"]["confidence"] == 0.0
    # Composite confidence should NOT be artificially dragged below 35
    assert res["committee_confidence"] >= 50.0
    assert res["action_state"] != "INSUFFICIENT_DATA"


def test_financial_institution_solvency_recognition():
    """Verify financial institutions and exchanges do not trigger false High Leverage warnings."""
    mock_quote_summary = {
        "quoteSummary": {
            "result": [{
                "assetProfile": {"sector": "Financial Services", "industry": "Financial Data & Stock Exchanges"},
                "price": {"shortName": "National Stock Exchange", "regularMarketPrice": 1752.0},
                "financialData": {
                    "totalRevenue": {"raw": 150000000000},
                    "netIncomeToCommon": {"raw": 104900000000},
                    "totalCash": {"raw": 200000000000},
                    "totalDebt": {"raw": 50000000000},
                    "debtToEquity": {"raw": 102.0},  # 1.02x
                    "operatingMargins": {"raw": 0.815},
                },
                "defaultKeyStatistics": {
                    "trailingPE": {"raw": 41.3},
                },
                "incomeStatementHistory": {"incomeStatementHistory": []},
                "incomeStatementHistoryQuarterly": {"incomeStatementHistory": []},
            }]
        }
    }

    with patch("yf_client._get", return_value=mock_quote_summary):
        data = get_fundamentals_data("NSE.BO")
        assert data["ratios"]["is_financial"] is True
        # Cons should not include high debt warning
        cons_text = " ".join(data["pros_and_cons"]["cons"])
        assert "Elevated debt-to-equity" not in cons_text
        # Fallback income statements should be synthesized from financialData
        assert len(data["income_annual"]) >= 1
        assert data["income_annual"][0]["net_income"] == 104900000000
