import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from routers.intraday import (
    _calculate_vwap_and_bands,
    _calculate_rsi,
    _calculate_supertrend,
    _calculate_multi_timeframe,
    _calculate_relative_strength,
    _generate_battle_plan,
    get_intraday_analysis,
    get_options_pcr,
    get_block_deals,
)


def test_live_quote_indicator_synchronization():
    """
    INVARIANT 1: Indicators must be computed on a series synchronized with live quote.
    When live quote moves significantly from the last historical bar close,
    VWAP, RSI, and EMAs must reflect the updated price.
    """
    # 20 bars of synthetic 5m data
    dates = pd.date_range("2026-09-18 09:15", periods=20, freq="5min")
    base_prices = np.linspace(100, 105, 20)
    df = pd.DataFrame({
        "Open": base_prices - 0.2,
        "High": base_prices + 0.5,
        "Low": base_prices - 0.5,
        "Close": base_prices,
        "Volume": np.full(20, 1000)
    }, index=dates)

    # Initial VWAP before live quote update
    vwap_before = _calculate_vwap_and_bands(df)["vwap"][-1]

    # Simulate live quote surge to 120.0
    live_quote_price = 120.0
    df_sync = df.copy()
    last_idx = df_sync.index[-1]
    df_sync.loc[last_idx, "Close"] = live_quote_price
    df_sync.loc[last_idx, "High"] = max(df_sync.loc[last_idx, "High"], live_quote_price)

    vwap_after = _calculate_vwap_and_bands(df_sync)["vwap"][-1]
    rsi_after = _calculate_rsi(df_sync["Close"], 14)[-1]

    # VWAP after must be strictly higher than before
    assert vwap_after > vwap_before, f"Synchronized VWAP {vwap_after} should exceed unsynchronized {vwap_before}"
    assert rsi_after > 50, f"RSI should reflect the sharp surge, got {rsi_after}"


def test_volume_delta_proxy_non_tick_invariant():
    """
    INVARIANT 2: Volume Delta must be explicitly marked as a proxy model
    and never claim tick-level Level 2 order flow.
    """
    dates = pd.date_range("2026-09-18 09:15", periods=10, freq="5min")
    df = pd.DataFrame({
        "Open": [100.0] * 10,
        "High": [102.0] * 10,
        "Low": [99.0] * 10,
        "Close": [101.5] * 10,
        "Volume": [5000] * 10
    }, index=dates)

    with patch("routers.intraday.get_history", return_value=df), \
         patch("routers.intraday.get_quote", return_value={"price": 101.5, "prevClose": 100.0, "name": "TEST"}):
        res = get_intraday_analysis(ticker="RELIANCE.NS", interval="5m", period="1d")

    assert "volume_delta_proxy" in res, "Must return volume_delta_proxy"
    proxy_meta = res["volume_delta_proxy"]
    assert proxy_meta["is_proxy"] is True, "is_proxy must be True"
    assert "Price-Location Volume Delta Proxy" in proxy_meta["methodology"]
    assert "not tick-level" in proxy_meta["methodology"]

    # Verify candle entries also carry proxy flag
    assert res["candles"][-1]["is_delta_proxy"] is True


def test_signal_extension_invariant():
    """
    INVARIANT 3: Overbought assets (RSI >= 70 or VWAP distance >= 1.5%) must NOT
    be classified as an unqualified STRONG BUY. They must be qualified as
    BULLISH (OVEREXTENDED) with elevated chase risk.
    """
    dates = pd.date_range("2026-09-18 09:15", periods=20, freq="5min")
    # Steep surge from 100 to 130
    prices = np.linspace(100, 130, 20)
    df = pd.DataFrame({
        "Open": prices - 0.2,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": np.full(20, 2000)
    }, index=dates)

    with patch("routers.intraday.get_history", return_value=df), \
         patch("routers.intraday.get_quote", return_value={"price": 130.0, "prevClose": 100.0, "name": "TEST"}):
        res = get_intraday_analysis(ticker="RELIANCE.NS", interval="5m", period="1d")

    signals = res["signals"]
    assert signals["extension_state"] == "EXTENDED_OVERBOUGHT", f"Expected EXTENDED_OVERBOUGHT, got {signals['extension_state']}"
    assert "CHASE RISK" in signals["risk_regime"]
    assert signals["overall_bias"] != "STRONG BUY", "Overextended asset must not be rated as unqualified STRONG BUY"
    assert "OVEREXTENDED" in signals["overall_bias"]


def test_battle_plan_rr_mathematical_consistency():
    """
    INVARIANT 4: Battle Plan R:R must be mathematically consistent.
    Target 1 = 1.5R, Target 2 = 2.5R, Weighted R:R = 2.0R.
    """
    plan = _generate_battle_plan(
        ticker="RELIANCE.NS",
        company="Reliance Industries",
        curr_price=2500.0,
        curr_vwap=2490.0,
        supertrend=2470.0,
        supertrend_dir=1,
        pivots={"daily_levels": {"pdh": 2510.0, "pdl": 2460.0, "pdc": 2480.0}},
        orb={"high_15m": 2505.0, "low_15m": 2485.0},
        bias="BUY",
        curr_sym="₹"
    )

    risk = plan["risk_per_share"]
    assert risk > 0
    # Check Target 1 and Target 2 distance in R
    t1_dist = plan["target_1"] - plan["entry_price"]
    t2_dist = plan["target_2"] - plan["entry_price"]
    assert round(t1_dist / risk, 1) == 1.5, f"Target 1 should be 1.5R, got {t1_dist / risk}"
    assert round(t2_dist / risk, 1) == 2.5, f"Target 2 should be 2.5R, got {t2_dist / risk}"
    assert plan["weighted_r"] == 2.0
    assert "1:2.0 (Weighted)" in plan["rr_ratio"]


def test_relative_performance_labeling_and_attribution():
    """
    INVARIANT 5: Relative performance metric must be labeled as relative return spread,
    NOT beta-adjusted alpha, and must NOT claim 'heavy institutional sponsorship'.
    """
    with patch("routers.intraday.get_quote", return_value={"price": 25000.0, "changePct": 0.5}):
        rs = _calculate_relative_strength(stock_change_pct=2.5, is_us=False)

    assert "Beta-Adjusted" not in rs["metric_type"]
    assert rs["metric_type"] == "Intraday Relative Return Spread"
    assert "institutional sponsorship" not in rs["desc"].lower()
    assert "institutional support" not in rs["desc"].lower()
    assert rs["status"] == "OUTPERFORMING"
    assert rs["relative_perf_pct"] == 2.0


def test_options_pcr_provenance_and_model_approximation():
    """
    INVARIANT 6: Options PCR must supply explicit provenance telemetry.
    If using the model approximation fallback, is_model_approximation must be True
    and provenance must be MODEL_ESTIMATE.
    """
    # Mock requests to fail so model fallback triggers
    with patch("requests.Session.get", side_effect=Exception("NSE timeout")), \
         patch("routers.intraday.get_quote", return_value={"price": 24500.0, "changePct": 0.3}):
        res = get_options_pcr(ticker="NIFTY", market="IN")

    assert res["available"] is True
    assert res["provenance"] == "MODEL_ESTIMATE"
    assert res["is_model_approximation"] is True
    assert "Model Approximation" in res["message"]


def test_triple_screen_confluence_scoring():
    """
    INVARIANT 7: Triple Screen must evaluate EMA alignment, RSI, and trend anchors
    to produce a genuine multi-factor score.
    """
    dates = pd.date_range("2026-09-18 09:15", periods=30, freq="5min")
    # Steadily rising prices
    prices = np.linspace(100, 115, 30)
    df = pd.DataFrame({
        "Open": prices - 0.2,
        "High": prices + 0.5,
        "Low": prices - 0.5,
        "Close": prices,
        "Volume": np.full(30, 1000)
    }, index=dates)

    mtf = _calculate_multi_timeframe(df)
    assert mtf["confluence_score"] >= 70, f"Expected high bullish confluence, got {mtf['confluence_score']}"
    assert mtf["confluence_bias"] == "STRONG BULLISH CONFLUENCE"
    assert len(mtf["screens"]) >= 1
    assert "factors" in mtf["screens"][0]
    assert mtf["screens"][0]["factors"]["ema_aligned"] is True


def test_block_deals_twenty_five_crore_circular():
    """
    INVARIANT 8: Block deals must reflect NSE's ₹25 Crore minimum order threshold.
    """
    with patch("requests.Session.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        res = get_block_deals()

    assert "₹25 Cr" in res["note"], f"Note should reflect ₹25 Cr minimum order size: {res['note']}"
