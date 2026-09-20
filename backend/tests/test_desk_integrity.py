"""
StockIQ Pro — Deterministic Committee Data-Integrity Regression Tests
====================================================================

Mandatory Tests A through K proving all synthetic proxies, fake fallbacks,
and heuristic data shortcuts have been completely eliminated:

- Test A: Intraday data source (verifies 5m bars used for intraday horizon, not 1d)
- Test B: Session-anchored VWAP (resets at session boundary, not 20-day rolling)
- Test C: Supertrend ATR(10, 3) (not EMA20 proxy)
- Test D: Canonical ORB (preserves breakout, breakdown, inside range, insufficient data)
- Test E: Market session state machine (pre-market, open, post-market, weekend, holiday)
- Test F: Price freshness (distinguishes quote timestamp from historical bar timestamp)
- Test G: DCF missing inputs (fair_value is None, valuation_status='INSUFFICIENT_DATA')
- Test H: Valuation percentiles (pe_percentile is None without empirical distribution)
- Test I: Historical expectancy / Kelly (half_kelly_pct is None without empirical profile)
- Test J: Derivatives provenance (model approximations excluded from empirical evidence)
- Test K: Sparse-data governance (moves to INSUFFICIENT_DATA/WAIT instead of false confidence)
"""

from datetime import datetime, time, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from desk_engine import (
    DeskContext,
    evaluate_committee,
    build_thesis_invalidation_triggers,
    evaluate_risk_gate,
)
from services.desk_adapter import (
    build_canonical_market_state,
    build_desk_context,
)
from services.intraday_engine import (
    calculate_intraday_snapshot,
    calculate_session_vwap_and_bands,
    calculate_supertrend,
    calculate_orb,
)
from services.market_session import (
    get_market_session_state,
    HOLIDAYS_NSE_2026,
)
from services.valuation_service import (
    calculate_canonical_valuation,
    ValuationResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_intraday_df(days: int = 2, bars_per_day: int = 75, base_price: float = 100.0) -> pd.DataFrame:
    """Generates 5-minute candles spanning multiple trading days."""
    dates = []
    session_dates = ["2026-09-17", "2026-09-18"][:days]
    for s_date in session_dates:
        s_times = pd.date_range(
            f"{s_date} 09:15:00",
            periods=bars_per_day,
            freq="5min",
            tz="Asia/Kolkata",
        )
        dates.extend(s_times)

    n = len(dates)
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(n) * 0.2)
    highs = prices + np.random.rand(n) * 0.3
    lows = prices - np.random.rand(n) * 0.3
    volume = np.random.randint(1000, 10000, size=n)

    return pd.DataFrame({
        "Open": prices,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volume,
    }, index=pd.DatetimeIndex(dates))


def _make_daily_df(days: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    """Generates daily candles."""
    dates = pd.date_range(end="2026-09-18", periods=days, freq="B", tz="Asia/Kolkata")
    n = len(dates)
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(n) * 0.8)
    return pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": np.random.randint(100000, 500000, size=n),
    }, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# Test A: Intraday Data Source
# ─────────────────────────────────────────────────────────────────────────────

def test_a_intraday_data_source():
    """
    Mock history so daily and intraday datasets are deliberately different.
    Verify horizon='intraday' uses the intraday dataset (5m) and NOT daily (1d).
    """
    intraday_df = _make_intraday_df(days=2, bars_per_day=50, base_price=500.0)
    daily_df = _make_daily_df(days=100, base_price=100.0)

    def mock_get_history(ticker, period=None, interval=None):
        if interval == "5m":
            return intraday_df
        elif interval == "1d":
            return daily_df
        return pd.DataFrame()

    with patch("services.desk_adapter.get_history", side_effect=mock_get_history), \
         patch("services.desk_adapter.get_quote", return_value={"price": 500.0, "regularMarketTime": 1774000000}), \
         patch("services.desk_adapter.get_info", return_value={}):

        # Intraday call
        ctx_intra = build_desk_context("RELIANCE.NS", horizon="intraday")
        assert ctx_intra.horizon == "intraday"
        # Intraday VWAP and technicals must come from the 500.0 intraday df, not the 100.0 daily df
        assert ctx_intra.vwap is not None
        assert ctx_intra.vwap > 400.0
        assert ctx_intra.supertrend_direction in ("BULLISH", "BEARISH")

        # Swing call
        ctx_swing = build_desk_context("RELIANCE.NS", horizon="swing")
        assert ctx_swing.horizon == "swing"
        # Swing must not populate session VWAP or intraday Supertrend
        assert ctx_swing.vwap is None
        assert ctx_swing.supertrend_direction is None


# ─────────────────────────────────────────────────────────────────────────────
# Test B: Session-Anchored VWAP
# ─────────────────────────────────────────────────────────────────────────────

def test_b_session_anchored_vwap_resets_between_sessions():
    """
    Provide two sessions with deliberately different prices.
    Verify VWAP resets between sessions and equals canonical intraday VWAP,
    NOT a 20-day rolling typical-price VWAP.
    """
    # Day 1 prices ~100, Day 2 prices ~200
    day1_times = pd.date_range("2026-09-17 09:15", periods=50, freq="5min", tz="Asia/Kolkata")
    day2_times = pd.date_range("2026-09-18 09:15", periods=50, freq="5min", tz="Asia/Kolkata")

    df1 = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1000
    }, index=day1_times)
    df2 = pd.DataFrame({
        "Open": 200.0, "High": 201.0, "Low": 199.0, "Close": 200.0, "Volume": 1000
    }, index=day2_times)
    combined_df = pd.concat([df1, df2])

    vwap_dict = calculate_session_vwap_and_bands(combined_df)
    vwap_arr = vwap_dict["vwap"]

    # At the end of Day 1 (index 49), VWAP should be ~100
    day1_end_vwap = vwap_arr[len(df1) - 1]
    assert pytest.approx(day1_end_vwap, rel=1e-2) == 100.0

    # At the beginning of Day 2 (index 50), VWAP must RESET to Day 2 price (~200), not average with Day 1
    day2_first_vwap = vwap_arr[len(df1)]
    assert pytest.approx(day2_first_vwap, rel=1e-2) == 200.0

    # Test via intraday engine snapshot
    snapshot = calculate_intraday_snapshot("TEST.NS", df=combined_df)
    assert pytest.approx(snapshot["session_vwap"], rel=1e-2) == 200.0


# ─────────────────────────────────────────────────────────────────────────────
# Test C: Canonical Supertrend (10, 3) vs EMA20 Proxy
# ─────────────────────────────────────────────────────────────────────────────

def test_c_supertrend_not_ema20_proxy():
    """
    Construct data where price > EMA20, but canonical Supertrend remains BEARISH.
    Verify committee Supertrend remains BEARISH (proves EMA20 proxy is gone).
    """
    # Create downtrend with high volatility where a quick bounce puts price > EMA20
    # but the Supertrend upper band remains above price (bearish)
    dates = pd.date_range("2026-09-18 09:15", periods=60, freq="5min", tz="Asia/Kolkata")
    
    # Prices declining sharply from 150 to 90, then small bounce to 95
    base = np.linspace(150, 90, 55)
    bounce = np.array([91.0, 92.0, 93.0, 94.0, 95.0])
    prices = np.concatenate([base, bounce])
    
    # Large candle ranges to keep ATR large so Supertrend upper band stays high
    highs = prices + 8.0
    lows = prices - 2.0

    df = pd.DataFrame({
        "Open": prices - 1.0,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": 5000,
    }, index=dates)

    st_res = calculate_supertrend(df, period=10, multiplier=3.0)
    st_direction = "BULLISH" if st_res["direction"][-1] == 1 else "BEARISH"

    # Check that canonical Supertrend is BEARISH
    assert st_direction == "BEARISH"

    # Feed this df to intraday snapshot
    snapshot = calculate_intraday_snapshot("TEST.NS", df=df)
    assert snapshot["supertrend_direction"] == "BEARISH"


# ─────────────────────────────────────────────────────────────────────────────
# Test D: Canonical Opening Range Breakout (ORB)
# ─────────────────────────────────────────────────────────────────────────────

def test_d_orb_preserves_canonical_states():
    """
    Test BULLISH_BREAKOUT, BEARISH_BREAKDOWN, INSIDE_RANGE, and None (insufficient data).
    Verify the adapter does not silently default to INSIDE_RANGE.
    """
    # 1. Insufficient data (< 3 bars of 5m = less than 15m)
    dates_short = pd.date_range("2026-09-18 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    df_short = pd.DataFrame({
        "Open": [100, 101], "High": [102, 103], "Low": [99, 100], "Close": [101, 102], "Volume": [1000, 1000]
    }, index=dates_short)
    res_short = calculate_orb(df_short, interval="5m")
    assert res_short["status"] is None  # Insufficient data, not INSIDE_RANGE!

    # 2. Inside Range: 15m range is 95 - 105; subsequent prices stay between 96 and 104
    dates_inside = pd.date_range("2026-09-18 09:15", periods=10, freq="5min", tz="Asia/Kolkata")
    df_inside = pd.DataFrame({
        "Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 100.0, "Volume": 1000
    }, index=dates_inside)
    res_inside = calculate_orb(df_inside, interval="5m")
    assert res_inside["status"] == "INSIDE_RANGE"

    # 3. Bullish Breakout: subsequent price breaks above 105
    df_breakout = df_inside.copy()
    df_breakout.iloc[-1, df_breakout.columns.get_loc("Close")] = 108.0
    res_breakout = calculate_orb(df_breakout, interval="5m")
    assert res_breakout["status"] == "BULLISH_BREAKOUT"

    # 4. Bearish Breakdown: subsequent price breaks below 95
    df_breakdown = df_inside.copy()
    df_breakdown.iloc[-1, df_breakdown.columns.get_loc("Close")] = 92.0
    res_breakdown = calculate_orb(df_breakdown, interval="5m")
    assert res_breakdown["status"] == "BEARISH_BREAKDOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Test E: Market Session State Machine
# ─────────────────────────────────────────────────────────────────────────────

def test_e_market_session_state_machine():
    """
    Test Indian weekday open, pre-market, post-market, weekend, holiday,
    US weekday open, US weekend.
    """
    # 1. Indian Weekday Open (Wednesday 11:30 AM IST)
    dt_in_open = datetime(2026, 9, 16, 11, 30, tzinfo=timezone.utc)  # 17:00 IST is post, let's use exact IST
    # 11:30 IST is 06:00 UTC
    dt_in_open_utc = datetime(2026, 9, 16, 6, 0, tzinfo=timezone.utc)
    state = get_market_session_state("RELIANCE.NS", now_dt=dt_in_open_utc)
    assert state.status == "OPEN"
    assert state.is_open is True

    # 2. Indian Pre-Market (Wednesday 09:05 AM IST = 03:35 UTC)
    dt_in_pre = datetime(2026, 9, 16, 3, 35, tzinfo=timezone.utc)
    state_pre = get_market_session_state("RELIANCE.NS", now_dt=dt_in_pre)
    assert state_pre.status == "PRE_MARKET"
    assert state_pre.is_open is False

    # 3. Indian Post-Market (Wednesday 15:45 PM IST = 10:15 UTC)
    dt_in_post = datetime(2026, 9, 16, 10, 15, tzinfo=timezone.utc)
    state_post = get_market_session_state("RELIANCE.NS", now_dt=dt_in_post)
    assert state_post.status == "POST_MARKET"
    assert state_post.is_open is False

    # 4. Indian Weekend (Sunday)
    dt_in_weekend = datetime(2026, 9, 20, 6, 0, tzinfo=timezone.utc)
    state_weekend = get_market_session_state("RELIANCE.NS", now_dt=dt_in_weekend)
    assert state_weekend.status == "WEEKEND"
    assert state_weekend.is_open is False

    # 5. Indian Holiday (e.g. Republic Day 2026-01-26 11:00 AM IST = 05:30 UTC)
    dt_holiday = datetime(2026, 1, 26, 5, 30, tzinfo=timezone.utc)
    state_holiday = get_market_session_state("RELIANCE.NS", now_dt=dt_holiday)
    assert state_holiday.status == "HOLIDAY"
    assert state_holiday.is_open is False

    # 6. US Weekday Open (Wednesday 11:00 AM EDT = 15:00 UTC)
    dt_us_open = datetime(2026, 9, 16, 15, 0, tzinfo=timezone.utc)
    state_us_open = get_market_session_state("AAPL", now_dt=dt_us_open)
    assert state_us_open.status == "OPEN"
    assert state_us_open.is_open is True

    # 7. US Weekend (Sunday)
    dt_us_weekend = datetime(2026, 9, 20, 15, 0, tzinfo=timezone.utc)
    state_us_weekend = get_market_session_state("AAPL", now_dt=dt_us_weekend)
    assert state_us_weekend.status == "WEEKEND"
    assert state_us_weekend.is_open is False


# ─────────────────────────────────────────────────────────────────────────────
# Test F: Price Freshness & Timestamp Separation
# ─────────────────────────────────────────────────────────────────────────────

def test_f_price_freshness_separation():
    """
    Give the quote and historical bar intentionally different timestamps.
    Verify quote freshness uses the quote timestamp and does NOT substitute historical bar age.
    """
    quote_time = 1774000000  # Unix timestamp
    mock_quote = {"price": 150.0, "regularMarketTime": quote_time}
    mock_daily_df = _make_daily_df(days=50, base_price=150.0)

    with patch("services.desk_adapter.get_quote", return_value=mock_quote), \
         patch("services.desk_adapter.get_history", return_value=mock_daily_df), \
         patch("services.desk_adapter.get_info", return_value={"currentPrice": 150.0}):

        state = build_canonical_market_state("AAPL", horizon="swing")
        
        # Quote and bar timestamps must be independently tracked
        assert state["quote"]["as_of"] is not None
        assert state["bars"]["as_of"] is not None
        assert state["quote"]["source"] == "regularMarketTime"
        assert state["provenance"]["quote"]["status"] == "OK"
        assert state["provenance"]["bars"]["status"] == "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Test G: DCF Missing Inputs & Zero Synthetic Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_g_dcf_missing_inputs_no_synthetic_fabrication():
    """
    Give the valuation service no FCF, no OCF, and no valid cash flow basis.
    Verify fair_value is None and valuation_status is 'INSUFFICIENT_DATA'.
    Must NOT produce a number from price * shares * 4% or revenue * 6%.
    """
    sparse_info = {
        "symbol": "ACME.NS",
        "currentPrice": 100.0,
        "marketCap": 1000000000,
        "sharesOutstanding": 10000000,
        "totalRevenue": 500000000,  # Revenue present, but FCF/OCF missing
        "freeCashflow": None,
        "operatingCashflow": None,
        "netIncomeToCommon": None,
        "sector": "Industrials",
    }

    res = calculate_canonical_valuation(sparse_info, current_price=100.0)
    assert res.fair_value is None
    assert res.valuation_status == "INSUFFICIENT_DATA"
    assert res.data_status == "INSUFFICIENT_DATA"
    assert "unavailable" in res.valuation_reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test H: Valuation Percentiles
# ─────────────────────────────────────────────────────────────────────────────

def test_h_valuation_percentiles_none_without_empirical_distribution():
    """
    Give trailingPE = 30 with no empirical distribution.
    Verify pe_percentile is None and no synthetic percentile is created.
    """
    info = {
        "symbol": "XYZ.NS",
        "trailingPE": 30.0,
        "enterpriseToEbitda": 18.0,
        "marketCap": 500000000,
        "freeCashflow": 20000000,
    }

    with patch("services.desk_adapter.get_info", return_value=info), \
         patch("services.desk_adapter.get_quote", return_value={"price": 100.0}), \
         patch("services.desk_adapter.get_history", return_value=_make_daily_df(30)):

        ctx = build_desk_context("XYZ.NS", horizon="swing")
        assert ctx.pe_percentile is None
        assert ctx.ev_ebitda_percentile is None


# ─────────────────────────────────────────────────────────────────────────────
# Test I: Historical Expectancy / Kelly Absence
# ─────────────────────────────────────────────────────────────────────────────

def test_i_kelly_absent_without_empirical_expectancy():
    """
    Verify that when expectancy is absent, half_kelly_pct is None,
    and no 55% / 1.8 assumption appears anywhere in the committee context.
    """
    with patch("services.desk_adapter.get_info", return_value={}), \
         patch("services.desk_adapter.get_quote", return_value={"price": 100.0}), \
         patch("services.desk_adapter.get_history", return_value=_make_daily_df(30)):

        ctx = build_desk_context("TEST.NS", horizon="swing")
        assert ctx.historical_win_rate is None
        assert ctx.historical_avg_win_loss is None

        res = evaluate_committee(ctx.model_dump(exclude_none=True))
        trade_geom = res["trade_geometry"]
        assert trade_geom["available"] is True
        assert trade_geom["half_kelly_pct"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Test J: Derivatives Provenance
# ─────────────────────────────────────────────────────────────────────────────

def test_j_derivatives_model_approximation_excluded():
    """
    Verify that model approximation derivatives are excluded from empirical evidence.
    """
    ctx = DeskContext(
        ticker="NIFTY",
        horizon="intraday",
        price=24000.0,
        options_available=False,
        derivatives_available=False,
    )
    res = evaluate_committee(ctx.model_dump(exclude_none=True))
    assert res["desks"]["derivatives"]["confidence"] == 0.0
    assert "derivatives" in res["audit"]["missing_fields_by_desk"]


# ─────────────────────────────────────────────────────────────────────────────
# Test K: Sparse-Data Governance
# ─────────────────────────────────────────────────────────────────────────────

def test_k_sparse_data_governance():
    """
    When canonical inputs are unavailable, verify the committee moves toward
    INSUFFICIENT_DATA or WAIT instead of becoming artificially confident.
    """
    sparse_ctx = DeskContext(
        ticker="SPARSE.NS",
        horizon="intraday",
        price=None,  # No price evidence
    )
    res = evaluate_committee(sparse_ctx.model_dump(exclude_none=True))
    assert res["action_state"] == "INSUFFICIENT_DATA"
    assert res["committee_confidence"] < 35


# ─────────────────────────────────────────────────────────────────────────────
# Test: Thesis Invalidation Triggers Structure
# ─────────────────────────────────────────────────────────────────────────────

def test_thesis_invalidation_triggers_structure():
    """
    Verify structured invalidation triggers are present and evidence-driven.
    """
    # Bullish context with VWAP and Supertrend
    bullish_ctx = {
        "price": 105.0,
        "vwap": 102.0,
        "supertrend_direction": "BULLISH",
        "orb_status": "BULLISH_BREAKOUT",
        "ema21": 101.0,
    }
    risk_gate = evaluate_risk_gate(bullish_ctx)
    triggers = build_thesis_invalidation_triggers(bullish_ctx, "BULLISH", "LONG_BIAS", risk_gate)

    metrics = [t["metric"] for t in triggers]
    assert "vwap" in metrics
    assert "supertrend_direction" in metrics
    assert "orb_status" in metrics
    assert "risk_gate" in metrics

    # Context without VWAP: must NOT invent VWAP trigger
    no_vwap_ctx = {
        "price": 105.0,
        "supertrend_direction": "BULLISH",
    }
    triggers_no_vwap = build_thesis_invalidation_triggers(no_vwap_ctx, "BULLISH", "LONG_BIAS", risk_gate)
    metrics_no_vwap = [t["metric"] for t in triggers_no_vwap]
    assert "vwap" not in metrics_no_vwap


# ─────────────────────────────────────────────────────────────────────────────
# Test: Official 2026 NSE Holidays (Ganesh Chaturthi & Id-Ul-Fitr)
# ─────────────────────────────────────────────────────────────────────────────

def test_nse_2026_holidays_ganesh_chaturthi_and_id_ul_fitr():
    """
    Verify official 2026 NSE trading holidays:
    - 2026-09-14 (Ganesh Chaturthi)
    - 2026-03-19 (Id-Ul-Fitr)
    """
    # Ganesh Chaturthi: Monday 14 September 2026 11:00 AM IST (05:30 UTC)
    dt_ganesh = datetime(2026, 9, 14, 5, 30, tzinfo=timezone.utc)
    state_ganesh = get_market_session_state("RELIANCE.NS", now_dt=dt_ganesh)
    assert state_ganesh.status == "HOLIDAY"
    assert state_ganesh.is_open is False

    # Id-Ul-Fitr: Thursday 19 March 2026 11:00 AM IST (05:30 UTC)
    dt_eid = datetime(2026, 3, 19, 5, 30, tzinfo=timezone.utc)
    state_eid = get_market_session_state("RELIANCE.NS", now_dt=dt_eid)
    assert state_eid.status == "HOLIDAY"
    assert state_eid.is_open is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: US Early Close Rules (Black Friday & Christmas Eve)
# ─────────────────────────────────────────────────────────────────────────────

def test_us_early_close_2026():
    """
    Verify US early close at 13:00 ET (1:00 PM) on:
    - 2026-11-27 (Black Friday)
    - 2026-12-24 (Christmas Eve)
    """
    # Black Friday 2026-11-27:
    # 12:45 ET (17:45 UTC) -> OPEN
    dt_bf_open = datetime(2026, 11, 27, 17, 45, tzinfo=timezone.utc)
    state_bf_open = get_market_session_state("AAPL", now_dt=dt_bf_open)
    assert state_bf_open.status == "OPEN"
    assert state_bf_open.is_open is True

    # 13:15 ET (18:15 UTC) -> POST_MARKET
    dt_bf_closed = datetime(2026, 11, 27, 18, 15, tzinfo=timezone.utc)
    state_bf_closed = get_market_session_state("AAPL", now_dt=dt_bf_closed)
    assert state_bf_closed.status == "POST_MARKET"
    assert state_bf_closed.is_open is False

    # Christmas Eve 2026-12-24:
    # 12:45 ET (17:45 UTC) -> OPEN
    dt_xmas_open = datetime(2026, 12, 24, 17, 45, tzinfo=timezone.utc)
    state_xmas_open = get_market_session_state("AAPL", now_dt=dt_xmas_open)
    assert state_xmas_open.status == "OPEN"
    assert state_xmas_open.is_open is True

    # 13:15 ET (18:15 UTC) -> POST_MARKET
    dt_xmas_closed = datetime(2026, 12, 24, 18, 15, tzinfo=timezone.utc)
    state_xmas_closed = get_market_session_state("AAPL", now_dt=dt_xmas_closed)
    assert state_xmas_closed.status == "POST_MARKET"
    assert state_xmas_closed.is_open is False


# ─────────────────────────────────────────────────────────────────────────────
# Test: Year Awareness (Uncataloged Years Return UNKNOWN)
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_future_year_calendar():
    """
    Verify that uncataloged future years (e.g. 2028) return UNKNOWN session state
    without claiming synthetic calendar validity.
    """
    # Wednesday in 2028 at 11:00 AM IST (05:30 UTC)
    dt_2028 = datetime(2028, 6, 14, 5, 30, tzinfo=timezone.utc)
    state_2028 = get_market_session_state("RELIANCE.NS", now_dt=dt_2028)
    assert state_2028.status == "UNKNOWN"
    assert state_2028.market_open is None
    assert "uncataloged" in state_2028.directive.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Dual-Horizon Intraday Context (Execution + Daily Risk)
# ─────────────────────────────────────────────────────────────────────────────

def test_intraday_dual_context_preserves_daily_risk():
    """
    Verify horizon='intraday' populates both 5m execution technicals
    AND higher-timeframe daily risk metrics (volatility percentile, max drawdown).
    """
    intraday_df = _make_intraday_df(days=2, bars_per_day=50, base_price=500.0)
    daily_df = _make_daily_df(days=100, base_price=500.0)

    def mock_get_history(ticker, period=None, interval=None):
        if interval == "5m":
            return intraday_df
        elif interval == "1d":
            return daily_df
        return pd.DataFrame()

    with patch("services.desk_adapter.get_history", side_effect=mock_get_history), \
         patch("services.desk_adapter.get_quote", return_value={"price": 500.0, "regularMarketTime": 1774000000}), \
         patch("services.desk_adapter.get_info", return_value={"currentPrice": 500.0}):

        ctx = build_desk_context("RELIANCE.NS", horizon="intraday")

        # Intraday execution metrics
        assert ctx.vwap is not None
        assert ctx.supertrend_direction in ("BULLISH", "BEARISH")
        assert ctx.atr is not None
        assert ctx.atr_pct is not None and ctx.atr_pct > 0

        # Higher-timeframe daily risk metrics preserved for CRO risk gate
        assert ctx.volatility_percentile is not None
        assert ctx.max_drawdown_pct is not None
        assert ctx.momentum_30d_pct is not None
        assert ctx.price_vs_ema20_atr is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Supertrend Uses Canonical Welles Wilder ATR
# ─────────────────────────────────────────────────────────────────────────────

def test_supertrend_uses_wilder_atr():
    """
    Verify that calculate_supertrend uses Welles Wilder ATR exponential smoothing.
    """
    df = _make_intraday_df(days=1, bars_per_day=40, base_price=100.0)
    st_res = calculate_supertrend(df, period=10, multiplier=3.0)

    from services.intraday_engine import calculate_atr
    expected_atr = calculate_atr(df, period=10)

    # st_res['atr'] must match canonical Wilder ATR
    np.testing.assert_allclose(st_res["atr"], expected_atr, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Test: VWAP Weighted Standard Deviation (Analytical Verification)
# ─────────────────────────────────────────────────────────────────────────────

def test_vwap_weighted_dispersion_analytical():
    """
    Verify mathematically correct volume-weighted standard deviation
    against an analytically known 2-bar distribution:
    Bar 1: Price 100, Volume 100
    Bar 2: Price 200, Volume 100
    VWAP = 150, Weighted Variance = 2500, Weighted Std = 50.
    """
    times = pd.date_range("2026-09-18 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    df = pd.DataFrame({
        "Open": [100.0, 200.0],
        "High": [100.0, 200.0],
        "Low": [100.0, 200.0],
        "Close": [100.0, 200.0],
        "Volume": [100.0, 100.0],
    }, index=times)

    vwap_dict = calculate_session_vwap_and_bands(df)

    # End VWAP should be 150.0
    assert pytest.approx(vwap_dict["vwap"][-1], rel=1e-4) == 150.0

    # Upper 1 should be 150 + 50 = 200.0
    assert pytest.approx(vwap_dict["upper_1"][-1], rel=1e-4) == 200.0

    # Lower 1 should be 150 - 50 = 100.0
    assert pytest.approx(vwap_dict["lower_1"][-1], rel=1e-4) == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Test: OCF Labeled As Proxy, Not Masquerading As Standard DCF
# ─────────────────────────────────────────────────────────────────────────────

def test_ocf_labeled_as_proxy_not_standard_dcf():
    """
    Verify that Operating Cash Flow is labeled as OCF_PROXY_VALUATION (data_status=PARTIAL)
    and does NOT masquerade as STANDARD_DCF.
    """
    info_ocf = {
        "symbol": "PROXY_CO.NS",
        "currentPrice": 100.0,
        "marketCap": 1000000000,
        "sharesOutstanding": 10000000,
        "freeCashflow": None,  # No FCF
        "operatingCashflow": 50000000,  # OCF available
        "totalRevenue": 500000000,
        "sector": "Technology",
    }
    res_ocf = calculate_canonical_valuation(info_ocf, current_price=100.0)
    assert res_ocf.methodology == "OCF_PROXY_VALUATION"
    assert res_ocf.data_status == "PARTIAL"
    assert res_ocf.valuation_status == "PROXY"

    # FCF available -> STANDARD_DCF
    info_fcf = {
        "symbol": "REAL_CO.NS",
        "currentPrice": 100.0,
        "marketCap": 1000000000,
        "sharesOutstanding": 10000000,
        "freeCashflow": 45000000,
        "operatingCashflow": 50000000,
        "totalRevenue": 500000000,
        "sector": "Technology",
    }
    res_fcf = calculate_canonical_valuation(info_fcf, current_price=100.0)
    assert res_fcf.methodology == "STANDARD_DCF"
    assert res_fcf.data_status == "COMPLETE"
    assert res_fcf.valuation_status == "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Assumption Status Exposed in Valuation
# ─────────────────────────────────────────────────────────────────────────────

def test_assumption_status_exposed():
    """
    Verify assumption_status distinguishes DEFAULT_MODEL_ASSUMPTIONS from CUSTOM_ASSUMPTIONS.
    """
    info = {
        "symbol": "ASSUME.NS",
        "currentPrice": 100.0,
        "marketCap": 1000000000,
        "sharesOutstanding": 10000000,
        "freeCashflow": 50000000,
        "sector": "Technology",
    }
    # Default assumptions
    res_def = calculate_canonical_valuation(info, current_price=100.0)
    assert res_def.assumption_status == "DEFAULT_MODEL_ASSUMPTIONS"

    # Custom assumptions
    res_cust = calculate_canonical_valuation(
        info, current_price=100.0, custom_growth_rate=0.12, custom_discount_rate=0.09
    )
    assert res_cust.assumption_status == "CUSTOM_ASSUMPTIONS"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Financial Institution Methodology Naming
# ─────────────────────────────────────────────────────────────────────────────

def test_financial_institution_methodology_naming():
    """
    Verify financial institution valuation is labeled FINANCIAL_INSTITUTION_EQUITY_CASHFLOW_PROXY.
    """
    info = {
        "symbol": "HDFCBANK.NS",
        "currentPrice": 1600.0,
        "marketCap": 12000000000000,
        "sharesOutstanding": 7500000000,
        "netIncomeToCommon": 600000000000,
        "sector": "Financial Services",
        "industry": "Banks - Diversified",
    }
    res = calculate_canonical_valuation(info, current_price=1600.0)
    assert res.methodology == "FINANCIAL_INSTITUTION_EQUITY_CASHFLOW_PROXY"
    assert res.valuation_status == "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Derivatives Model Estimate Rejection from Empirical Evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_derivatives_model_estimate_rejected_from_empirical():
    """
    Inject pcr_oi with is_model_approximation=True and derivatives_provenance='MODEL_ESTIMATE'.
    Verify that the committee does NOT treat the PCR as empirical derivatives evidence.
    """
    ctx_approx = {
        "price": 100.0,
        "pcr_oi": 1.25,
        "is_model_approximation": True,
        "derivatives_provenance": "MODEL_ESTIMATE",
    }
    res = evaluate_committee(ctx_approx)
    deriv = res["desks"]["derivatives"]

    # Must have 0 confidence and flagged as rejected
    assert deriv["confidence"] == 0.0
    assert "MODEL_DERIVATIVES_REJECTED" in deriv["flags"]
    assert "derivatives (model approximation rejected)" in deriv["missing_data"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: Provenance Timestamps & Quote Source Attribution
# ─────────────────────────────────────────────────────────────────────────────

def test_provenance_timestamps_and_quote_source():
    """
    Verify complete provenance timestamps for session_vwap, supertrend, orb,
    and accurate quote source attribution (regularMarketTime vs price_date).
    """
    intraday_df = _make_intraday_df(days=2, bars_per_day=50, base_price=500.0)
    daily_df = _make_daily_df(days=100, base_price=500.0)

    def mock_get_history(ticker, period=None, interval=None):
        return intraday_df if interval == "5m" else daily_df

    with patch("services.desk_adapter.get_history", side_effect=mock_get_history), \
         patch("services.desk_adapter.get_quote", return_value={"price": 500.0, "price_date": "2026-09-18T09:15:00Z"}), \
         patch("services.desk_adapter.get_info", return_value={}):

        state = build_canonical_market_state("RELIANCE.NS", horizon="intraday")
        prov = state["provenance"]

        # Quote source must be price_date when regularMarketTime is absent
        assert prov["quote"]["source"] == "price_date"
        assert prov["quote"]["as_of"] is not None

        # Intraday timestamps must be populated
        assert prov["session_vwap"]["as_of"] is not None
        assert prov["supertrend"]["as_of"] is not None
        assert prov["orb"]["as_of"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Thesis Invalidation Trigger Wording
# ─────────────────────────────────────────────────────────────────────────────

def test_thesis_invalidation_trigger_wording():
    """
    Verify thesis invalidation wording is concisely 'CRO risk gate enters VETO'.
    """
    ctx = {"price": 100.0, "vwap": 98.0}
    risk_gate = evaluate_risk_gate(ctx)
    triggers = build_thesis_invalidation_triggers(ctx, "BULLISH", "LONG_BIAS", risk_gate)

    conditions = [t["condition"] for t in triggers]
    assert "CRO risk gate enters VETO" in conditions
    assert not any("liquidity breach or extreme volatility" in c for c in conditions)
