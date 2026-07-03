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
