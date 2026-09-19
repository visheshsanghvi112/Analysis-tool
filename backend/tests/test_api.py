import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import FastAPI app from main
from main import app

client = TestClient(app)

# Helper to generate mock stock price dataframes for testing
def get_mock_df():
    dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Open": np.linspace(100, 150, 100),
        "High": np.linspace(102, 152, 100),
        "Low": np.linspace(98, 148, 100),
        "Close": np.linspace(101, 151, 100),
        "Volume": [100000] * 100
    }, index=dates)
    return df

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["using_yf_client"] is True

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "StockIQ Pro API" in data["name"]

@patch("routers.tickers.ensure_ticker_list")
def test_tickers_search(mock_ensure):
    mock_ensure.return_value = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "sector": "Energy"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "sector": "IT"},
    ]
    response = client.get("/api/tickers?q=reliance")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tickers"]) == 1
    assert data["tickers"][0]["symbol"] == "RELIANCE.NS"

@patch("routers.tickers.ensure_ticker_list")
def test_sectors_grouping(mock_ensure):
    mock_ensure.return_value = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "sector": "Energy"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "sector": "IT"},
    ]
    response = client.get("/api/sectors")
    assert response.status_code == 200
    data = response.json()
    assert "sectors" in data
    assert "grouped" in data
    assert len(data["sectors"]) == 2

@patch("routers.tickers.get_quote")
def test_live_price(mock_quote):
    mock_quote.return_value = {
        "price": 2500.50,
        "prevClose": 2480.00,
        "change": 20.50,
        "changePct": 0.83,
        "longName": "Reliance Industries Limited"
    }
    response = client.get("/api/live?ticker=RELIANCE.NS")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert data["price"] == 2500.50

@patch("routers.analysis.get_info")
def test_dcf_valuation_invalid_ticker(mock_info):
    mock_info.return_value = {}
    response = client.get("/api/valuation?ticker=INVALID")
    assert response.status_code == 404

@patch("routers.analysis.get_history")
def test_backtest_endpoint(mock_history):
    mock_history.return_value = get_mock_df()
    response = client.get("/api/backtest?ticker=RELIANCE.NS&period=1y&initial_capital=100000")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "RELIANCE.NS"
    assert "stats" in data
    assert "equity_curves" in data

def test_master_universe_loaded():
    """Verify that the master universe loads thousands of stocks and ETFs."""
    from services.ticker_manager import ensure_ticker_list, SECTOR_MAP
    tickers = ensure_ticker_list()
    assert len(tickers) >= 5000, f"Expected at least 5000 instruments, got {len(tickers)}"
    assert "NIFTYBEES.NS" in [t["symbol"] for t in tickers]
    assert "GOLDBEES.NS" in [t["symbol"] for t in tickers]
    assert "^NSEI" in [t["symbol"] for t in tickers]
    assert len(SECTOR_MAP) >= 5000

def test_etf_search_real():
    """Verify live search for ETFs returns ETF category."""
    response = client.get("/api/tickers?q=niftybees")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tickers"]) >= 1
    symbols = [t["symbol"] for t in data["tickers"]]
    assert "NIFTYBEES.NS" in symbols

def test_bse_code_search_real():
    """Verify live search for 6-digit BSE scrip code."""
    response = client.get("/api/tickers?q=500325")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tickers"]) >= 1
    assert "RELIANCE.BO" in [t["symbol"] for t in data["tickers"]]

@patch("routers.tickers.get_quote")
def test_index_ticker_live_price(mock_quote):
    """Verify index ticker with leading caret is accepted."""
    mock_quote.return_value = {
        "price": 24000.0,
        "prevClose": 23900.0,
        "change": 100.0,
        "changePct": 0.42,
        "longName": "NIFTY 50"
    }
    response = client.get("/api/live?ticker=^NSEI")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "^NSEI"
    assert data["price"] == 24000.0

def test_smart_search_typo_tolerance():
    """Verify common typo 'relaince' correctly resolves to RELIANCE.NS as top result."""
    response = client.get("/api/tickers?q=relaince")
    assert response.status_code == 200
    tickers = response.json()["tickers"]
    assert len(tickers) >= 1
    assert tickers[0]["symbol"] == "RELIANCE.NS"

def test_smart_search_multi_token_space():
    """Verify space-separated query 'tata motors' matches Tata Motors."""
    response = client.get("/api/tickers?q=tata motors")
    assert response.status_code == 200
    tickers = response.json()["tickers"]
    assert len(tickers) >= 1
    assert any("Tata Motors" in t["name"] for t in tickers)

def test_smart_search_financial_alias():
    """Verify financial acronym 'sbi' resolves to State Bank of India (SBIN.NS)."""
    response = client.get("/api/tickers?q=sbi")
    assert response.status_code == 200
    tickers = response.json()["tickers"]
    assert len(tickers) >= 1
    assert tickers[0]["symbol"] == "SBIN.NS"

def test_smart_search_concept_gold_etf():
    """Verify conceptual search 'gold etf' resolves to GOLDBEES.NS."""
    response = client.get("/api/tickers?q=gold etf")
    assert response.status_code == 200
    tickers = response.json()["tickers"]
    assert len(tickers) >= 1
    assert tickers[0]["symbol"] == "GOLDBEES.NS"

def test_empty_ticker_validation():
    """Verify empty ticker query returns 400 Bad Request."""
    response = client.get("/api/advanced-news?ticker=")
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"].lower()

def test_portfolio_optimize_single_holding_validation():
    """Verify optimization requires at least 2 holdings."""
    response = client.post("/api/portfolio-optimize", json={
        "holdings": [{"ticker": "RELIANCE.NS", "qty": 10, "buy_price": 2500}]
    })
    assert response.status_code == 400
    assert "at least 2" in response.json()["detail"].lower()

def test_advanced_news_endpoint_schema():
    """Verify live news reader endpoint returns clean schema."""
    response = client.get("/api/advanced-news?ticker=TCS.NS&company_name=Tata%20Consultancy%20Services")
    assert response.status_code == 200
    data = response.json()
    assert "news_intelligence" in data
    intel = data["news_intelligence"]
    assert intel["status"] in ["live", "active"]
    assert "sentiment" in intel
    assert "total_articles" in intel
    assert "articles" in intel


def test_etf_tracking_error_and_sip_computation():
    from routers.analysis import _compute_tracking_error, _compute_xirr, _compute_historical_sip, _normalize_index_to_date
    import pandas as pd
    import numpy as np

    # Test 1: Cross-timezone alignment for tracking error
    dates_etf = pd.date_range("2024-01-01", periods=100, freq="B", tz="Asia/Kolkata")
    dates_bench = pd.date_range("2024-01-01", periods=100, freq="B", tz="America/New_York")
    
    np.random.seed(42)
    s_etf = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates_etf)
    s_bench = pd.Series(np.random.normal(0.001, 0.01, 100), index=dates_bench)

    te = _compute_tracking_error(s_etf, s_bench)
    assert te is not None
    assert isinstance(te, float)
    assert te > 0

    # Test 2: XIRR computation
    cf = [-1000.0, -1000.0, -1000.0, 3300.0]
    dt = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01"), pd.Timestamp("2023-04-01")]
    xirr = _compute_xirr(cf, dt)
    assert xirr is not None
    assert xirr > 0

    # Test 3: Historical SIP simulation
    price_series = pd.Series(
        np.linspace(100, 200, 300),
        index=pd.date_range("2023-01-01", periods=300, freq="B")
    )
    sip_res = _compute_historical_sip(price_series, monthly_amt=1000.0)
    assert "1Y" in sip_res
    assert sip_res["1Y"]["total_invested"] > 0
    assert sip_res["1Y"]["sip_value"] > sip_res["1Y"]["total_invested"]
    assert sip_res["1Y"]["sip_xirr_pct"] is not None


def test_intraday_session_vwap_reset():
    from routers.intraday import _calculate_vwap_and_bands
    import pandas as pd
    import numpy as np

    # Two distinct trading days with large price jump
    d1 = pd.date_range('2024-01-01 09:15', '2024-01-01 15:30', freq='5min')
    d2 = pd.date_range('2024-01-02 09:15', '2024-01-02 15:30', freq='5min')
    dates = d1.append(d2)

    df = pd.DataFrame({
        'High': [100.0] * len(d1) + [200.0] * len(d2),
        'Low': [100.0] * len(d1) + [200.0] * len(d2),
        'Close': [100.0] * len(d1) + [200.0] * len(d2),
        'Volume': [1000] * len(dates)
    }, index=dates)

    vwap = _calculate_vwap_and_bands(df)['vwap']
    # Day 1 start should be 100.0
    assert abs(vwap[0] - 100.0) < 1e-4
    # Day 2 start MUST reset to 200.0, not carry over Day 1's 100.0
    assert abs(vwap[len(d1)] - 200.0) < 1e-4


def test_bank_valuation_health_checklist():
    from routers.analysis import calculate_dcf
    res = calculate_dcf('HDFCBANK.NS')
    assert res is not None
    assert res['health_score'] >= 7
    # Verify that ROA condition was appropriately set for banks
    roa_item = next(item for item in res['health_checklist'] if item['metric'] == 'Return on Assets (ROA)')
    assert roa_item['condition'] == '>= 1.0%'
    assert roa_item['passed'] is True

