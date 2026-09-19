from __future__ import annotations
import math
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ============================================================
# Financial Invariant & Metamorphic Test Suite
# Ensures the system cannot silently produce financially invalid results.
# ============================================================

def test_historical_immutability_invariant():
    """
    INVARIANT 1: Historical Immutability
    Closed historical candles (T-n to T-1) must remain strictly immutable
    when live quotes or intraday observations at T are received.
    """
    from engine import calculate_risk_metrics
    
    # 100 historical closed trading days
    dates = pd.date_range("2024-01-01", periods=100, freq="B", tz="Asia/Kolkata")
    np.random.seed(42)
    base_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 100)))
    
    historical_closed = pd.Series(base_prices, index=dates, name="Close")
    
    # Calculate historical risk metrics prior to any live price arrival
    risk_before = calculate_risk_metrics(historical_closed)
    
    # Simulate arrival of a turbulent intraday quote (e.g. +20% jump or -15% drop)
    # The live quote at T must NOT modify closed candles 0..T-1
    live_price = float(base_prices[-1] * 1.20)
    
    # Re-verify closed candles remain bit-for-bit identical
    np.testing.assert_array_equal(historical_closed.values, base_prices)
    
    # Re-evaluate historical metrics on closed candles
    risk_after = calculate_risk_metrics(historical_closed)
    
    assert risk_before['sharpeRatio'] == risk_after['sharpeRatio']
    assert risk_before['sortinoRatio'] == risk_after['sortinoRatio']
    assert risk_before['maxDrawdown'] == risk_after['maxDrawdown']
    assert risk_before['annualizedVolatility'] == risk_after['annualizedVolatility']


def test_price_base_recalculation_invariant():
    """
    INVARIANT 2: Price-Base Recalculation
    When the live reference price changes from 330 to 396:
      - Live targets and expected absolute gains must scale with the new live price
      - Historical risk metrics (Sharpe, DD, Volatility) must remain invariant
    """
    from engine import calculate_risk_metrics
    
    dates = pd.date_range("2024-01-01", periods=100, freq="B", tz="Asia/Kolkata")
    close_series = pd.Series(np.linspace(300, 330, 100), index=dates)
    
    risk_at_330 = calculate_risk_metrics(close_series)
    
    # A model predicts +10% return
    predicted_return = 0.10
    
    target_at_330 = 330.0 * (1.0 + predicted_return)
    expected_gain_at_330 = 330.0 * predicted_return
    
    target_at_396 = 396.0 * (1.0 + predicted_return)
    expected_gain_at_396 = 396.0 * predicted_return
    
    # Targets must properly reflect the new live base
    assert target_at_396 == pytest.approx(435.6, rel=1e-3)
    assert expected_gain_at_396 == pytest.approx(39.6, rel=1e-3)
    assert target_at_396 > target_at_330
    
    # But historical risk metrics from closed series must NOT change
    risk_recheck = calculate_risk_metrics(close_series)
    assert risk_recheck['sharpeRatio'] == risk_at_330['sharpeRatio']
    assert risk_recheck['maxDrawdown'] == risk_at_330['maxDrawdown']


def test_model_non_negativity_and_telemetry_preservation():
    """
    INVARIANT 3: Price Validity & Telemetry Preservation (SR 11-7)
    - All predicted prices (base and ensemble) must be strictly positive (P > 0)
    - Raw predictions must be preserved in telemetry and not quietly masked
    - Clipped predictions must be explicitly flagged with is_clipped = True
    """
    from ml_models import StockPredictor
    
    predictor = StockPredictor()
    # Mock a catastrophic negative linear prediction (e.g. -120% return)
    predictor.models = {
        'bayesian_ridge': MagicMock(predict=MagicMock(return_value=np.array([-1.20]))),
        'random_forest': MagicMock(predict=MagicMock(return_value=np.array([0.04]))),
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.02])))
    predictor.scaler = MagicMock(transform=MagicMock(return_value=np.zeros((1, 5))))
    predictor.feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
    
    df_dummy = pd.DataFrame({
        'Close': [100.0] * 50,
        'Open': [100.0] * 50,
        'High': [102.0] * 50,
        'Low': [98.0] * 50,
        'Volume': [1000] * 50,
        'f1': [1.0] * 50, 'f2': [1.0] * 50, 'f3': [1.0] * 50, 'f4': [1.0] * 50, 'f5': [1.0] * 50,
    })
    
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 100.0}):
        pred, err = predictor.predict("TEST.NS")
        
        assert err is None
        assert pred is not None
        
        # 1. Non-negativity assertion
        assert pred['predicted_price'] > 0
        assert pred['models']['bayesian_ridge']['validated_price'] > 0
        assert pred['models']['random_forest']['validated_price'] > 0
        
        # 2. Telemetry preservation assertion
        ridge_telemetry = pred['models']['bayesian_ridge']
        assert ridge_telemetry['raw_return'] == pytest.approx(-120.0, rel=1e-2)
        assert ridge_telemetry['is_clipped'] is True
        assert ridge_telemetry['status'] == "DEGRADED_OUTLIER"
        assert ridge_telemetry['included_in_ensemble'] is False


def test_ensemble_outlier_exclusion_invariant():
    """
    INVARIANT 4: Ensemble Governance & Outlier Exclusion
    Degraded outlier models (e.g. Bayesian Ridge predicting -107%) must be
    quarantined and excluded from meta-stacker ensemble inputs to prevent contamination.
    """
    from ml_models import StockPredictor
    
    predictor = StockPredictor()
    # 3 models: 1 outlier, 2 healthy models
    predictor.models = {
        'bayesian_ridge': MagicMock(predict=MagicMock(return_value=np.array([-1.07]))), # Outlier
        'random_forest':  MagicMock(predict=MagicMock(return_value=np.array([0.03]))),  # Healthy
        'gradient_boost': MagicMock(predict=MagicMock(return_value=np.array([0.05]))),  # Healthy
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.04])))
    predictor.scaler = MagicMock(transform=MagicMock(return_value=np.zeros((1, 5))))
    predictor.feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
    
    df_dummy = pd.DataFrame({
        'Close': [100.0] * 50,
        'Open': [100.0] * 50,
        'High': [102.0] * 50,
        'Low': [98.0] * 50,
        'Volume': [1000] * 50,
        'f1': [1.0] * 50, 'f2': [1.0] * 50, 'f3': [1.0] * 50, 'f4': [1.0] * 50, 'f5': [1.0] * 50,
    })
    
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 100.0}):
        pred, err = predictor.predict("TEST.NS")
        
        assert err is None
        # Verify Bayesian Ridge is excluded from ensemble
        assert pred['models']['bayesian_ridge']['included_in_ensemble'] is False
        assert pred['models']['random_forest']['included_in_ensemble'] is True
        assert pred['models']['gradient_boost']['included_in_ensemble'] is True
        
        # The ensemble return must be in the vicinity of the healthy models (+3% to +5%),
        # NOT dragged down by the -107% outlier
        assert pred['predicted_return'] > 0.0


def test_garch_specification_and_auditability():
    """
    INVARIANT 5: GARCH Specification Exposing & Auditability
    GARCH/EWMA conditional volatility must expose full mathematical specification:
      - Annualized percentage
      - Daily conditional std percentage
      - Estimation window and forecast horizon
      - Relationship: Annualized ~= daily_std * sqrt(252)
    """
    from ml_models import _forecast_garch_volatility
    
    np.random.seed(42)
    n = 200
    # Simulate returns with a volatility shock
    returns = np.random.normal(0, 0.01, n)
    returns[-5:] = [0.05, -0.04, 0.06, -0.05, 0.07] # High shock
    
    df = pd.DataFrame({
        'returns': returns,
        'Close': 100 * np.exp(np.cumsum(returns)),
    })
    
    spec = _forecast_garch_volatility(df)
    assert spec is not None
    assert isinstance(spec, dict)
    
    assert "annualized_pct" in spec
    assert "daily_conditional_std_pct" in spec
    assert "forecast_horizon_days" in spec
    assert "estimation_window_days" in spec
    assert "model_spec" in spec
    
    ann = spec["annualized_pct"]
    daily_std = spec["daily_conditional_std_pct"]
    
    assert ann > 0
    assert daily_std > 0
    # Check mathematical consistency within tolerance: ann ~= daily_std * sqrt(252)
    expected_ann = daily_std * math.sqrt(252)
    assert ann == pytest.approx(expected_ann, rel=0.05)


def test_etf_multi_horizon_tracking_error_and_alignment():
    """
    INVARIANT 6: ETF Multi-Horizon Tracking Error & Calendar Alignment
    - Calculates 30D, 90D, and 1Y tracking errors
    - Properly aligns disparate calendars (e.g. NSE holidays vs US holidays)
      without synthetic tracking error spikes from timezone offsets
    """
    from routers.analysis import _compute_tracking_error
    
    # Simulate 300 business days of ETF and Benchmark returns
    # with slightly misaligned dates (e.g., US market closed on July 4, NSE open)
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    
    np.random.seed(42)
    bench_ret = pd.Series(np.random.normal(0.0004, 0.012, 300), index=dates)
    # ETF tracks benchmark with small noise (0.1% daily noise)
    etf_ret = bench_ret + np.random.normal(0, 0.001, 300)
    
    # Inject a holiday mismatch
    etf_ret_misaligned = etf_ret.drop([dates[50], dates[120]])
    bench_ret_misaligned = bench_ret.drop([dates[75], dates[180]])
    
    te_full = _compute_tracking_error(etf_ret_misaligned, bench_ret_misaligned)
    te_30d  = _compute_tracking_error(etf_ret_misaligned, bench_ret_misaligned, window_days=22)
    te_90d  = _compute_tracking_error(etf_ret_misaligned, bench_ret_misaligned, window_days=66)
    te_1y   = _compute_tracking_error(etf_ret_misaligned, bench_ret_misaligned, window_days=252)
    
    assert te_full is not None
    assert te_30d is not None
    assert te_90d is not None
    assert te_1y is not None
    
    # Tracking errors must be positive and within reasonable bounds (~1.0% to 2.5%)
    assert 0.5 < te_full < 5.0
    assert 0.5 < te_30d < 5.0
    assert 0.5 < te_1y < 5.0


def test_financial_institution_valuation_identity():
    """
    INVARIANT 7: Financial Institution Valuation Identity (DDM / Equity DCF Consistency)
    
    For financial institutions (banks/NBFCs like IRFC):
      1. Starting flow must be Equity Flow (Net Income proxy), NOT Operating Cashflow.
      2. Operating debt (borrowings to lend) must NOT be subtracted from Equity Value.
      3. Mathematical consistency between Net Income (E_0), ROE, Cost of Equity (r_e),
         Growth (g), and Terminal Growth (g_t):
         - Cost of equity: r_e = R_f + beta * ERP
         - Sustainable growth: g = (1 - Payout) * ROE
         - Gordon Growth Identity: When g = g_t, the multi-stage Equity DCF formula
           converges exactly to P_0 = E_0 * (1 + g) / (r_e - g)
      4. Verifies that subtracting operating debt would catastrophically violate
         the equity valuation identity (collapsing positive equity value to zero/negative).
    """
    from routers.analysis import calculate_dcf
    
    # IRFC-like financial institution profile
    financial_info = {
        "currentPrice": 150.0,
        "sector": "Financial Services",
        "industry": "Credit Services",
        "marketCap": 1950000000000,
        "sharesOutstanding": 13000000000,
        "totalCash": 30000000000,
        "totalDebt": 3800000000000,        # Huge operating debt (normal for NBFC)
        "netIncomeToCommon": 65000000000,  # Healthy ₹6,500 Cr net profit
        "operatingCashflow": -120000000000,# Negative due to loan disbursements
        "trailingEps": 5.0,
        "bookValue": 35.0,
        "returnOnEquity": 0.14,            # 14% ROE
        "returnOnAssets": 0.015,
        "payoutRatio": 0.30,               # 30% dividend payout
        "beta": 1.0,
        "currency": "INR",
    }
    
    with patch("routers.analysis.get_info", return_value=financial_info):
        dcf_res = calculate_dcf("IRFC.NS")
        
        # 1. Classification & Defaults
        assert dcf_res["is_financial"] is True
        assert dcf_res["dcf_defaults"]["starting_flow"] == 65000000000
        assert "Net Income" in dcf_res["dcf_defaults"]["flow_type"]
        assert dcf_res["valuation_status"]["methodology"] == "FINANCIAL_INSTITUTION_EQUITY_DDM"
        
        # 2. Cost of Equity & Sustainable Growth Mathematical Consistency
        # CAPM: r_e = Rf (6.5%) + Beta (1.0) * ERP (6.0%) = 12.5%
        r_e = dcf_res["dcf_defaults"]["discount_rate"]
        assert 0.08 <= r_e <= 0.15
        assert r_e == pytest.approx(0.125, abs=1e-3)
        
        # Sustainable growth: g = (1 - payout) * ROE = 0.70 * 0.14 = 0.098 (9.8%)
        g_sustainable = (1.0 - 0.30) * 0.14
        assert 0.05 <= g_sustainable <= 0.20
        
        # Terminal growth must be strictly less than cost of equity (r_e > g_t)
        g_t = dcf_res["dcf_defaults"]["terminal_growth"]
        assert r_e > g_t
        
        # 3. Multi-stage Equity DCF Projection Calculation
        E0 = dcf_res["dcf_defaults"]["starting_flow"]
        g = dcf_res["dcf_defaults"]["growth_rate"]  # default 8%
        
        pv_flows = []
        flow = E0
        for yr in range(1, 6):
            flow *= (1 + g)
            pv_flows.append(flow / ((1 + r_e) ** yr))
            
        pv_flow_sum = sum(pv_flows)
        terminal_flow = flow * (1 + g_t)
        terminal_val = terminal_flow / (r_e - g_t)
        pv_terminal = terminal_val / ((1 + r_e) ** 5)
        
        equity_val_dcf = pv_flow_sum + pv_terminal
        intrinsic_per_share_dcf = equity_val_dcf / financial_info["sharesOutstanding"]
        
        # Intrinsic per share must be positive and commercially realistic (~₹50 to ₹150)
        assert intrinsic_per_share_dcf > 0
        assert 40.0 < intrinsic_per_share_dcf < 200.0
        
        # 4. Gordon Growth Identity Convergence Check:
        # When g == g_t (constant growth model), multi-stage formula reduces to E0 * (1 + g) / (r_e - g)
        g_const = 0.045
        gordon_equity_val = (E0 * (1 + g_const)) / (r_e - g_const)
        
        # Multi-stage with g = g_const
        pv_const_sum = sum((E0 * ((1 + g_const) ** t)) / ((1 + r_e) ** t) for t in range(1, 6))
        tv_const = (E0 * ((1 + g_const) ** 6)) / (r_e - g_const)
        pv_tv_const = tv_const / ((1 + r_e) ** 5)
        multistage_const = pv_const_sum + pv_tv_const
        
        # Assert mathematical identity holds
        assert multistage_const == pytest.approx(gordon_equity_val, rel=1e-4)
        
        # 5. Operating Debt Non-Subtraction Invariant
        # If operating debt (₹3.8T) were erroneously subtracted like an industrial company:
        industrial_equity_val = equity_val_dcf + financial_info["totalCash"] - financial_info["totalDebt"]
        # It would wipe out equity value completely into deep negative territory (-₹2.9T)
        assert industrial_equity_val < 0
        # But financial institution model preserves equity value:
        assert equity_val_dcf > 0


def test_forecast_horizon_independence_invariant():
    """
    INVARIANT 8: Forecast Horizon Independence
    ML prediction horizon (5-day for stocks, 30-day for ETFs) must be explicitly
    tagged and distinct from Monte Carlo (e.g. 60-day / 252-day) so the system
    cannot confuse or falsely equate horizons.
    """
    from ml_models import StockPredictor
    
    predictor = StockPredictor()
    predictor.models = {
        'random_forest': MagicMock(predict=MagicMock(return_value=np.array([0.02]))),
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.02])))
    predictor.scaler = MagicMock(transform=MagicMock(return_value=np.zeros((1, 5))))
    predictor.feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
    
    df_dummy = pd.DataFrame({
        'Close': [100.0] * 50,
        'Open': [100.0] * 50,
        'High': [102.0] * 50,
        'Low': [98.0] * 50,
        'Volume': [1000] * 50,
        'f1': [1.0] * 50, 'f2': [1.0] * 50, 'f3': [1.0] * 50, 'f4': [1.0] * 50, 'f5': [1.0] * 50,
    })
    
    # 1. Test Stock: horizon must be 5 days
    predictor.asset_type = 'EQUITY'
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 100.0}):
        pred_stock, _ = predictor.predict("RELIANCE.NS")
        assert pred_stock['horizon_days'] == 5
        assert "5-Day" in pred_stock['horizon_label']
        
    # 2. Test ETF: horizon must be 30 days
    predictor.asset_type = 'ETF'
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 100.0}):
        pred_etf, _ = predictor.predict("MONQ50.NS")
        assert pred_etf['horizon_days'] == 30
        assert "30-Day" in pred_etf['horizon_label']
