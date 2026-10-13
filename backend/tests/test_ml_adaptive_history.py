import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from ml_models import StockPredictor
import yf_client

client = TestClient(app)

def _generate_synthetic_ohlcv(num_days: int, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq="B")
    prices = [start_price]
    for _ in range(num_days - 1):
        prices.append(prices[-1] * (1.0 + np.random.normal(0, 0.015)))
    
    return pd.DataFrame({
        "Open": prices,
        "High": [p * 1.02 for p in prices],
        "Low": [p * 0.98 for p in prices],
        "Close": prices,
        "Volume": [250000] * num_days,
    }, index=dates)


def test_ticker_inception_metadata_extraction():
    """Verify get_ticker_inception_metadata correctly computes age and session statistics."""
    mock_meta = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "firstTradeDate": 1700000000,  # Nov 14, 2023
                        "currency": "INR",
                    },
                    "timestamp": [1700000000 + i * 86400 for i in range(100)],
                }
            ]
        }
    }
    with patch("yf_client._get", return_value=mock_meta):
        meta = yf_client.get_ticker_inception_metadata("NEWIPO.NS")
        assert meta["first_trade_date"] is not None
        assert "2023" in meta["first_trade_date"]
        assert meta["calendar_age_days"] > 0
        assert meta["calendar_age_human"] is not None
        assert meta["total_trading_sessions_recorded"] == 100


def test_ml_incubation_period_detection():
    """Verify that an asset with fewer than 50 trading sessions enters the incubation period state."""
    short_df = _generate_synthetic_ohlcv(num_days=25)
    
    with patch("ml_models.get_history", return_value=short_df), \
         patch("ml_models.get_ticker_inception_metadata") as mock_meta:
        mock_meta.return_value = {
            "first_trade_date": "2026-09-01",
            "first_trade_date_formatted": "01 Sep 2026",
            "calendar_age_days": 38,
            "calendar_age_human": "1.3 months",
            "is_newly_listed": True,
            "total_trading_sessions_recorded": 25,
        }
        predictor = StockPredictor()
        success, result = predictor.train("NEWIPO.NS", period="3y")
        
        assert success is False
        assert isinstance(result, dict)
        assert result.get("status") == "INCUBATION_PERIOD"
        assert result.get("error_code") == "ASSET_INCUBATION_WINDOW"
        assert result.get("total_trading_sessions_recorded") == 25
        assert result.get("required_trading_sessions") == 50
        assert result.get("readiness_pct") == 50.0
        assert len(result.get("available_modules", [])) >= 3


def test_ml_auto_adaptation_for_young_stock():
    """Verify that when 5y is requested for a stock with only 120 sessions, ML auto-adapts to max history instead of crashing."""
    medium_df = _generate_synthetic_ohlcv(num_days=120)
    
    with patch("ml_models.get_history", return_value=medium_df), \
         patch("ml_models.get_ticker_inception_metadata") as mock_meta:
        mock_meta.return_value = {
            "first_trade_date": "2025-06-01",
            "first_trade_date_formatted": "01 Jun 2025",
            "calendar_age_days": 180,
            "calendar_age_human": "6.0 months",
            "is_newly_listed": True,
            "total_trading_sessions_recorded": 120,
        }
        predictor = StockPredictor()
        success, metrics = predictor.train("MIDIPO.NS", period="5y")
        
        # Predictor should have auto-adapted and successfully completed training
        assert success is True
        assert isinstance(metrics, dict)
        assert "cv_rmse" in metrics or "mae" in metrics
        pred, err = predictor.predict("MIDIPO.NS")
        
        assert pred is not None
        assert "history_metadata" in pred
        h_meta = pred["history_metadata"]
        assert h_meta.get("is_adapted") is True
        assert h_meta.get("requested_period") == "5y"
        assert "max" in h_meta.get("effective_period")
        assert h_meta.get("calendar_age_human") == "6.0 months"


def test_ml_invalid_custom_date_range_handling():
    """Verify descriptive error explanation when start_date is after end_date."""
    predictor = StockPredictor()
    success, err = predictor.train("TEST.NS", start_date="2025-10-01", end_date="2024-01-01")
    
    assert success is False
    assert isinstance(err, dict)
    assert err.get("error_type") == "INVALID_DATE_RANGE"
    assert "start date" in err.get("message", "").lower()
    assert "precedes" in err.get("suggestion", "").lower()


def test_ml_router_incubation_endpoint_response():
    """Verify /api/ml-predict returns a clean 200 payload with incubation diagnostics."""
    short_df = _generate_synthetic_ohlcv(num_days=20)
    
    with patch("ml_models.get_history", return_value=short_df), \
         patch("routers.ml.get_asset_type", return_value="EQUITY"), \
         patch("ml_models.get_ticker_inception_metadata") as mock_meta:
        mock_meta.return_value = {
            "first_trade_date": "2026-09-10",
            "first_trade_date_formatted": "10 Sep 2026",
            "calendar_age_days": 29,
            "calendar_age_human": "1.0 months",
            "is_newly_listed": True,
            "total_trading_sessions_recorded": 20,
        }
        response = client.get("/api/ml-predict?ticker=BABYIPO.NS&period=3y")
        assert response.status_code == 200
        data = response.json()
        assert data.get("prediction") is None
        assert data.get("incubation") is not None
        assert data["incubation"]["status"] == "INCUBATION_PERIOD"
        assert data["incubation"]["total_trading_sessions_recorded"] == 20
        assert data["incubation"]["readiness_pct"] == 40.0
