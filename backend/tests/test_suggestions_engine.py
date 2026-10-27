"""
Tests for Quantitative Scrip Suggestions & Reasoning Engine
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from services.suggestions_engine import (
    analyze_single_scrip,
    get_curated_suggestions,
    UNDER_200_RADAR_UNIVERSE,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_suggestions_universe_metadata():
    """Verify curated under-200 radar universe has required fields."""
    assert len(UNDER_200_RADAR_UNIVERSE) >= 20
    for item in UNDER_200_RADAR_UNIVERSE:
        assert "symbol" in item
        assert "name" in item
        assert "sector" in item
        assert "catalyst" in item
        assert item["symbol"].endswith(".NS") or item["symbol"].endswith(".BO")


def test_single_scrip_analysis_structure():
    """Verify single scrip analysis returns complete trade plan and reasoning."""
    res = analyze_single_scrip("PNB.NS")
    if res is not None:
        assert res["symbol"] == "PNB.NS"
        assert "trade_plan" in res
        assert "technicals" in res
        assert "reasoning" in res

        # Trade plan fields
        plan = res["trade_plan"]
        assert "entry_range" in plan
        assert "target_1" in plan
        assert "target_2" in plan
        assert "stop_loss" in plan
        assert "risk_reward_ratio" in plan
        assert plan["stop_loss"] < plan["target_1"]

        # Technicals fields
        tech = res["technicals"]
        assert tech["current_price"] > 0
        assert tech["sma20"] > 0
        assert tech["sma50"] > 0
        assert tech["rsi"] >= 0 and tech["rsi"] <= 100
        assert tech["rvol"] > 0

        # Reasoning fields
        reasoning = res["reasoning"]
        assert len(reasoning["primary_thesis"]) > 10
        assert len(reasoning["price_action"]) > 10
        assert len(reasoning["volume_footprint"]) > 10
        assert len(reasoning["momentum_indicators"]) > 10
        assert len(reasoning["levels_rationale"]) > 10
        assert len(reasoning["invalidation_warning"]) > 10


def test_api_suggestions_endpoint(client):
    """Test /api/suggestions route with under 200 filter."""
    response = client.get("/api/suggestions?max_price=200")
    assert response.status_code == 200
    data = response.json()
    assert "total_scanned" in data
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)

    # All returned suggestions should satisfy max_price <= 200
    for s in data["suggestions"]:
        assert s["technicals"]["current_price"] <= 200.0


def test_api_suggestions_under_200_endpoint(client):
    """Test /api/suggestions/under-200 shortcut route."""
    response = client.get("/api/suggestions/under-200")
    assert response.status_code == 200
    data = response.json()
    assert "suggestions" in data
    for s in data["suggestions"]:
        assert s["technicals"]["current_price"] <= 200.0


def test_api_evaluate_ticker_endpoint(client):
    """Test on-demand evaluation of a ticker."""
    response = client.get("/api/suggestions/evaluate/TATASTEEL.NS")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TATASTEEL.NS"
    assert "trade_plan" in data
    assert "reasoning" in data
    assert data["conviction_score"] >= 0
