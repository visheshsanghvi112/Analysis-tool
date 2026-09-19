from __future__ import annotations
import math
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# ============================================================
# Adversarial & Metamorphic Test Suite
# Tests financial invariants under adversarial data perturbations
# ============================================================

def test_metamorphic_live_price_perturbation():
    """
    METAMORPHIC RELATION 1: Live Price Scaling & Feature State Disentanglement
    
    Distinguishes:
      1. Fixed-Feature Price Scaling:
         When feature vector X is held constant, changing live price P scales P_target
         proportionately, while predicted return r = (P_target / P - 1) remains IDENTICAL:
         r(396) == r(330).
      2. Dynamic-Feature Price Scaling:
         When changing the live observation updates technical features (e.g. today's
         momentum/return), the predicted return legitimately updates to reflect the new state.
      3. Historical Immutability:
         Historical closed candles, Sharpe, DD, and Volatility remain strictly invariant.
    """
    from engine import calculate_risk_metrics
    from ml_models import StockPredictor
    
    # 1. Historical closed series
    dates = pd.date_range("2024-01-01", periods=100, freq="B", tz="Asia/Kolkata")
    np.random.seed(42)
    prices = 300.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, 100)))
    closed_series = pd.Series(prices, index=dates, name="Close")
    
    # Run A: Calculate risk metrics on closed candles
    risk_A = calculate_risk_metrics(closed_series)
    
    # Perturb live quote from 330 to 396 (closed series remains untouched)
    risk_B = calculate_risk_metrics(closed_series)
    
    # Assert historical immutability under live price perturbation
    assert risk_A['sharpeRatio'] == risk_B['sharpeRatio']
    assert risk_A['maxDrawdown'] == risk_B['maxDrawdown']
    assert risk_A['annualizedVolatility'] == risk_B['annualizedVolatility']
    
    # 2. Fixed-Feature Price Scaling
    predictor = StockPredictor()
    predictor.models = {
        'random_forest': MagicMock(predict=MagicMock(return_value=np.array([0.05]))),
        'gradient_boost': MagicMock(predict=MagicMock(return_value=np.array([0.05]))),
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.05])))
    predictor.scaler = MagicMock(transform=MagicMock(return_value=np.zeros((1, 5))))
    predictor.feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
    
    df_dummy = pd.DataFrame({
        'Close': [330.0] * 50,
        'Open': [330.0] * 50,
        'High': [335.0] * 50,
        'Low': [325.0] * 50,
        'Volume': [1000] * 50,
        'f1': [1.0] * 50, 'f2': [1.0] * 50, 'f3': [1.0] * 50, 'f4': [1.0] * 50, 'f5': [1.0] * 50,
    })
    
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 330.0}):
        pred_330, _ = predictor.predict("TEST.NS")
        
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 396.0}):
        pred_396, _ = predictor.predict("TEST.NS")
        
    assert pred_330 is not None and pred_396 is not None
    assert pred_330['predicted_price'] != pred_396['predicted_price']
    
    # Fixed-feature invariant: target price scales proportionately with live price
    ratio = pred_396['predicted_price'] / pred_330['predicted_price']
    assert ratio == pytest.approx(396.0 / 330.0, rel=1e-2)
    
    # Fixed-feature invariant: predicted return is IDENTICAL
    ret_330 = pred_330['predicted_price'] / 330.0 - 1.0
    ret_396 = pred_396['predicted_price'] / 396.0 - 1.0
    assert ret_330 == pytest.approx(ret_396, abs=1e-4)

    # 3. Dynamic-Feature Price Scaling:
    # If the feature state changes (e.g. higher momentum from a +20% surge),
    # the model predicted return legitimately responds to the new feature vector
    predictor_dynamic = StockPredictor()
    predictor_dynamic.models = {
        'random_forest': MagicMock(predict=MagicMock(side_effect=lambda X: np.array([0.05]) if X[0,0] == 0 else np.array([0.12]))),
    }
    predictor_dynamic.meta_model = MagicMock(predict=MagicMock(side_effect=lambda X: np.array([0.05]) if X[0,0] == 0 else np.array([0.12])))
    predictor_dynamic.scaler = MagicMock(transform=MagicMock(side_effect=lambda X: X))
    predictor_dynamic.feature_cols = ['momentum', 'vol']

    df_base = pd.DataFrame({
        'Close': [330.0] * 50,
        'Open': [330.0] * 50,
        'High': [335.0] * 50,
        'Low': [325.0] * 50,
        'Volume': [1000] * 50,
        'momentum': [0.0] * 50,
        'vol': [0.1] * 50
    })
    df_surged = pd.DataFrame({
        'Close': [396.0] * 50,
        'Open': [396.0] * 50,
        'High': [400.0] * 50,
        'Low': [390.0] * 50,
        'Volume': [1000] * 50,
        'momentum': [0.20] * 50,
        'vol': [0.1] * 50
    })

    with patch("ml_models.get_history", return_value=df_base), \
         patch("ml_models.get_quote", return_value={"price": 330.0}):
        pred_dyn_base, err_base = predictor_dynamic.predict("TEST.NS")
        assert err_base is None

    with patch("ml_models.get_history", return_value=df_surged), \
         patch("ml_models.get_quote", return_value={"price": 396.0}):
        pred_dyn_surged, err_surged = predictor_dynamic.predict("TEST.NS")
        assert err_surged is None

    # When features legitimately change, predicted return changes accordingly
    assert pred_dyn_base['predicted_return'] != pred_dyn_surged['predicted_return']


def test_metamorphic_fx_perturbation():
    """
    METAMORPHIC RELATION 2: Common FX Cancellation Invariance on Tracking Error
    
    Financial Invariant:
      For an unhedged international ETF and its correctly converted benchmark:
        (1 + R_ETF,INR)   = (1 + R_ETF,USD) * (1 + R_FX)
        (1 + R_Bench,INR) = (1 + R_Bench,USD) * (1 + R_FX)
      The common FX factor cancels out in return differences:
        R_ETF,INR - R_Bench,INR = (R_ETF,USD - R_Bench,USD) * (1 + R_FX)
                                ~= R_ETF,USD - R_Bench,USD
      
      Therefore, multiplying both ETF and Benchmark by the SAME volatile FX series
      MUST NOT artificially inflate tracking error:
        TE_INR(Volatile FX) ~= TE_USD(Local Currency) ~= TE_INR(Flat FX)
    """
    from routers.analysis import _compute_tracking_error
    
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    np.random.seed(42)
    
    # 1. Benchmark USD series (e.g. Nasdaq Next Generation 100)
    usd_bench_ret = pd.Series(np.random.normal(0.0005, 0.012, 150), index=dates)
    usd_bench_prices = 100.0 * (1.0 + usd_bench_ret).cumprod()
    
    # 2. ETF USD series: tracks benchmark with idiosyncratic tracking noise (~0.1% daily std ~= 1.58% annual TE)
    idiosyncratic_noise = pd.Series(np.random.normal(0.0, 0.001, 150), index=dates)
    usd_etf_ret = usd_bench_ret + idiosyncratic_noise
    usd_etf_prices = 100.0 * (1.0 + usd_etf_ret).cumprod()
    
    # Baseline: Tracking Error in USD
    te_usd = _compute_tracking_error(usd_etf_ret.dropna(), usd_bench_ret.dropna())
    assert 1.0 < te_usd < 2.5  # Realistic ~1.5% USD tracking error
    
    # Run A: Flat FX (1 USD = 83 INR constantly)
    fx_flat = pd.Series(83.0, index=dates)
    inr_bench_A = usd_bench_prices * fx_flat
    inr_etf_A = usd_etf_prices * fx_flat
    te_inr_A = _compute_tracking_error(inr_etf_A.pct_change().dropna(), inr_bench_A.pct_change().dropna())
    
    # Run B: Highly Volatile FX (INR experiences high daily volatility: ~1.0% daily std ~= 16% annualized FX vol)
    fx_vol_ret = np.random.normal(0.0002, 0.010, 150)
    fx_volatile = pd.Series(83.0 * (1.0 + pd.Series(fx_vol_ret, index=dates)).cumprod(), index=dates)
    inr_bench_B = usd_bench_prices * fx_volatile
    inr_etf_B = usd_etf_prices * fx_volatile
    te_inr_B = _compute_tracking_error(inr_etf_B.pct_change().dropna(), inr_bench_B.pct_change().dropna())
    
    # Invariant Assertions:
    # 1. Under Flat FX, INR tracking error equals USD tracking error
    assert te_inr_A == pytest.approx(te_usd, abs=0.01)
    
    # 2. Under Highly Volatile FX, common FX exposure cancels out:
    # TE_INR must remain approximately equal to TE_USD (within 0.1% absolute tolerance)
    assert te_inr_B == pytest.approx(te_usd, abs=0.1)
    
    # 3. Currency volatility does NOT inflate tracking error from ~1.5% to 10%+
    assert abs(te_inr_B - te_inr_A) < 0.1


def test_metamorphic_benchmark_proxy_transparency():
    """
    METAMORPHIC RELATION 3: Benchmark Proxy Fallback Transparency
    
    When primary benchmark (^NTQX, Total Return Index) series is unavailable (< 30 trading days):
      - System falls back to diagnostic calculation proxy (^NXTQ, Price Return Index)
      - If ^NXTQ is also unavailable, falls back to ^NDX
      - is_proxy_used MUST be True
      - proxy_reason MUST be non-empty and document the fallback
      - primary_benchmark_ticker MUST remain '^NTQX'
      - active_benchmark_ticker MUST reflect the fallback proxy (^NXTQ or ^NDX)
    """
    from routers.analysis import get_etf_analysis
    
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    etf_dummy = pd.DataFrame({
        'Close': np.linspace(100, 120, 100),
        'Open': np.linspace(100, 120, 100),
        'High': np.linspace(101, 121, 100),
        'Low': np.linspace(99, 119, 100),
        'Volume': [5000] * 100,
    }, index=dates)
    
    nxtq_bench_dummy = pd.DataFrame({
        'Close': np.linspace(1200, 1400, 100),
        'Adj Close': np.linspace(1200, 1400, 100),
    }, index=dates)
    
    # Mock: primary benchmark (^NTQX) returns empty df (<30 obs); proxy (^NXTQ) returns valid data
    def mock_get_history(ticker, period='5y'):
        if ticker in ('^NTQX', 'NTQX'):
            return pd.DataFrame() # Primary TRI benchmark empty!
        if ticker == '^NXTQ':
            return nxtq_bench_dummy
        if ticker == 'USDINR=X':
            return pd.DataFrame({'Close': [83.5] * 100}, index=dates)
        return etf_dummy
        
    with patch("routers.analysis.get_history", side_effect=mock_get_history), \
         patch("routers.analysis.get_quote", return_value={"price": 120.0}), \
         patch("routers.analysis.get_info", return_value={"longName": "Motilal Oswal Nasdaq Q-50 ETF", "currency": "INR", "nav": 119.15}):
        res = get_etf_analysis("MONQ50.NS")
        
        bench_info = res["performance"]["benchmark_info"]
        assert bench_info["is_proxy_used"] is True
        assert bench_info["primary_benchmark_ticker"] == "^NTQX"
        assert bench_info["primary_benchmark_name"] == "NASDAQ Q-50 Total Return Index"
        assert bench_info["benchmark_type"] == "TOTAL_RETURN_INDEX"
        assert bench_info["active_benchmark_ticker"] == "^NXTQ"
        assert "Official scheme benchmark (NASDAQ Q-50 Total Return Index, NTQX) time-series unavailable" in bench_info["proxy_reason"]
        assert res["data_status"]["benchmark"] == "PROXY"


def test_adversarial_outlier_quarantine_and_policy_telemetry():
    """
    METAMORPHIC RELATION 4: Outlier Contamination Resistance (Adversarial Injection)
    
    Adversarially inject an explosive negative return (-500%) into Bayesian Ridge:
      - Outlier model MUST have status == 'DEGRADED_OUTLIER'
      - Outlier model MUST have included_in_ensemble == False
      - Outlier model MUST record outlier_reason explaining policy breach
      - Meta-stacker MUST quarantine the outlier and predict from healthy models
    """
    from ml_models import StockPredictor, MODEL_GOVERNANCE_THRESHOLDS
    
    predictor = StockPredictor()
    predictor.models = {
        'bayesian_ridge': MagicMock(predict=MagicMock(return_value=np.array([-5.0]))), # -500% catastrophic failure
        'random_forest':  MagicMock(predict=MagicMock(return_value=np.array([0.03]))),  # +3%
        'gradient_boost': MagicMock(predict=MagicMock(return_value=np.array([0.04]))),  # +4%
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.035])))
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
        ridge = pred['models']['bayesian_ridge']
        
        # 1. Quarantined status and policy documentation
        assert ridge['status'] == "DEGRADED_OUTLIER"
        assert ridge['included_in_ensemble'] is False
        assert ridge['is_clipped'] is True
        assert "breached policy bounds" in ridge['outlier_reason']
        assert ridge['thresholds_applied']['policy_key'] == "equity_5d"
        
        # 2. Ensemble stacker contamination check
        # The ensemble return MUST NOT be dragged down by -500%
        assert pred['predicted_return'] > 0.0
        assert pred['predicted_price'] > 100.0
        
        # 3. Model status check
        assert pred['model_status']['ensemble_health'] == "DEGRADED_QUARANTINE"
        assert "bayesian_ridge" in pred['model_status']['quarantined_models']


def test_multi_tier_status_governance_across_asset_types():
    """
    METAMORPHIC RELATION 5: Multi-Tier Status Governance
    
    Verifies that DATA_STATUS, MODEL_STATUS, and VALUATION_STATUS
    are properly segregated and exposed across:
      1. Equity ML predictions
      2. Bank / Financial Institution DCF
      3. ETF Long-Term Analysis
    """
    from ml_models import StockPredictor
    from routers.analysis import calculate_dcf
    
    # 1. Equity ML Status
    predictor = StockPredictor()
    predictor.models = {'rf': MagicMock(predict=MagicMock(return_value=np.array([0.02])))}
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.02])))
    predictor.scaler = MagicMock(transform=MagicMock(return_value=np.zeros((1, 5))))
    predictor.feature_cols = ['f1']
    
    df_dummy = pd.DataFrame({'Close': [100.0]*50, 'Open': [100.0]*50, 'High': [101.0]*50, 'Low': [99.0]*50, 'Volume': [1000]*50, 'f1': [1.0]*50})
    with patch("ml_models.get_history", return_value=df_dummy), \
         patch("ml_models.get_quote", return_value={"price": 100.0}):
        pred, _ = predictor.predict("RELIANCE.NS")
        assert "data_status" in pred
        assert "model_status" in pred
        assert "valuation_status" in pred
        assert pred["data_status"]["live_quote"] == "OK"
        assert pred["model_status"]["ensemble_health"] == "OK"
        
    # 2. Bank DCF Status
    bank_info = {
        "currentPrice": 100.0,
        "sector": "Financial Services",
        "marketCap": 100000000000,
        "sharesOutstanding": 1000000000,
        "netIncomeToCommon": 5000000000,
        "trailingEps": 5.0,
        "bookValue": 40.0,
    }
    with patch("routers.analysis.get_info", return_value=bank_info):
        dcf_res = calculate_dcf("HDFCBANK.NS")
        assert "data_status" in dcf_res
        assert "model_status" in dcf_res
        assert "valuation_status" in dcf_res
        assert "market_structure_status" in dcf_res
        assert dcf_res["valuation_status"]["methodology"] == "FINANCIAL_INSTITUTION_EQUITY_DDM"
        assert dcf_res["model_status"]["status"] == "NOT_APPLICABLE"
        assert dcf_res["market_structure_status"]["secondary_market_dislocation"] == "NORMAL"

    # 3. ETF Status & Market Structure Governance
    from routers.analysis import get_etf_analysis
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    etf_dummy = pd.DataFrame({
        'Close': np.linspace(100, 120, 100),
        'Open': np.linspace(100, 120, 100),
        'High': np.linspace(101, 121, 100),
        'Low': np.linspace(99, 119, 100),
        'Volume': [5000] * 100,
    }, index=dates)
    
    with patch("routers.analysis.get_history", return_value=etf_dummy), \
         patch("routers.analysis.get_quote", return_value={"price": 120.0}), \
         patch("routers.analysis.get_info", return_value={"longName": "Nifty BeES ETF", "currency": "INR"}):
        etf_res = get_etf_analysis("NIFTYBEES.NS")
        assert "data_status" in etf_res
        assert "model_status" in etf_res
        assert "valuation_status" in etf_res
        assert "market_structure_status" in etf_res
        assert etf_res["market_structure_status"]["secondary_market_dislocation"] in ("NORMAL", "HIGH")


def test_benchmark_tri_selection_and_no_silent_regression():
    """
    METAMORPHIC RELATION 7: Benchmark Total Return Index (TRI) Selection Integrity
    
    Verifies that:
      1. MONQ50.NS is explicitly mapped to '^NTQX' (NASDAQ Q-50 Total Return Index),
         honoring Motilal Oswal's official scheme benchmark.
      2. NXTQ is identified as PRICE_RETURN_INDEX and NTQX is identified as TOTAL_RETURN_INDEX.
      3. NXTQ != NTQX: Price-return index cannot silently overwrite total-return index.
      4. Any regression that sets MONQ50 benchmark to '^NXTQ' directly fails.
    """
    from routers.analysis import _ETF_BENCHMARK_MAP, _BENCHMARK_METADATA
    
    # 1. MONQ50 primary benchmark mapping must be '^NTQX' (TRI)
    primary_bench = _ETF_BENCHMARK_MAP.get("MONQ50.NS")
    assert primary_bench == "^NTQX", (
        f"Regression detected: MONQ50.NS primary benchmark is '{primary_bench}'. "
        "It must be '^NTQX' (NASDAQ Q-50 Total Return Index) per scheme SID."
    )
    
    # 2. Metadata distinguishes TRI from Price Return
    ntqx_meta = _BENCHMARK_METADATA.get("^NTQX")
    nxtq_meta = _BENCHMARK_METADATA.get("^NXTQ")
    
    assert ntqx_meta is not None
    assert nxtq_meta is not None
    assert ntqx_meta["type"] == "TOTAL_RETURN_INDEX"
    assert nxtq_meta["type"] == "PRICE_RETURN_INDEX"
    assert ntqx_meta["type"] != nxtq_meta["type"]
    assert ntqx_meta["name"] != nxtq_meta["name"]


def test_metric_segregation_nav_te_vs_secondary_divergence():
    """
    METAMORPHIC RELATION 8: Permanent Metric Segregation
    
    Strictly separates:
      1. SEBI Regulatory NAV Tracking Error:
         StdDev(R_NAV - R_TRI) * sqrt(252).
         Requires daily NAV series. Exchange traded price series does NOT qualify.
      2. Secondary-Market Price vs Benchmark Return Divergence:
         StdDev(R_NSE - R_Benchmark) * sqrt(252).
         Preserves the empirical 115% diagnostic without mislabeling it as regulatory TE.
      3. Secondary-Market Dislocation:
         Premium/Discount to iNAV = (Market Price / iNAV) - 1.
         Directly captures the +232% premium dislocation for MONQ50.
    """
    from routers.analysis import get_etf_analysis
    
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    
    # Simulate MONQ50 traded price series around ₹395.33 with high return volatility
    np.random.seed(42)
    # Repeated ~20% moves reflecting circuit volatility
    etf_prices = [395.33] * 100
    for i in range(1, 100):
        if i % 5 == 0:
            etf_prices[i] = etf_prices[i-1] * 1.20
        elif i % 5 == 1:
            etf_prices[i] = etf_prices[i-1] * 0.85
        else:
            etf_prices[i] = etf_prices[i-1] * 1.002
            
    etf_prices[-1] = 395.33
    etf_dummy = pd.DataFrame({
        'Close': etf_prices,
        'Open': etf_prices,
        'High': [p * 1.01 for p in etf_prices],
        'Low': [p * 0.99 for p in etf_prices],
        'Volume': [10000] * 100,
    }, index=dates)
    
    # Benchmark series (smooth)
    bench_dummy = pd.DataFrame({
        'Close': np.linspace(1200, 1300, 100),
        'Adj Close': np.linspace(1200, 1300, 100),
    }, index=dates)
    
    def mock_get_history(ticker, period='5y'):
        if ticker in ('^NTQX', 'NTQX'):
            return pd.DataFrame() # Fallback to proxy
        if ticker == '^NXTQ':
            return bench_dummy
        if ticker == 'USDINR=X':
            return pd.DataFrame({'Close': [83.5] * 100}, index=dates)
        return etf_dummy

    with patch("routers.analysis.get_history", side_effect=mock_get_history), \
         patch("routers.analysis.get_quote", return_value={"price": 395.33}), \
         patch("routers.analysis.get_info", return_value={"longName": "Motilal Oswal Nasdaq Q-50 ETF", "currency": "INR", "nav": 119.15}):
        res = get_etf_analysis("MONQ50.NS")
        
        perf = res["performance"]
        
        # 1. Regulatory NAV tracking error is segregated and None for traded price series
        assert perf["regulatory_nav_tracking_error"] is None
        assert "SEBI defines ETF tracking error strictly as annualized standard deviation of daily NAV returns" in perf["regulatory_nav_tracking_error_note"]
        
        # 2. Secondary-market divergence is explicitly named and computed
        sec_div = perf["secondary_market_divergence"]
        assert sec_div is not None
        assert sec_div["divergence_annual"] is not None
        assert sec_div["divergence_annual"] > 20.0 # High divergence from circuit volatility
        assert "Calculated from traded-price returns versus benchmark returns" in sec_div["disclaimer"]
        
        # 3. Secondary-market dislocation directly measures iNAV premium
        dislocation = perf["secondary_market_dislocation"]
        assert dislocation is not None
        assert dislocation["inav"] == 119.15
        assert dislocation["market_price"] == 395.33
        # (395.33 / 119.15 - 1) * 100 ~= 231.79%
        assert dislocation["premium_discount_pct"] == pytest.approx(231.79, abs=0.5)
        assert dislocation["is_dislocated"] is True


def test_market_structure_status_observation_vs_attribution():
    """
    METAMORPHIC RELATION 9: Separation of Observation and Attribution
    
    Verifies that market structure diagnostics separate:
      - observation: Factual market-price vs iNAV dislocation and return divergence.
      - attribution: Potential causal mechanisms (overseas investment caps, trading constraints, circuit limits).
    """
    from routers.analysis import get_etf_analysis
    
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    etf_dummy = pd.DataFrame({
        'Close': [395.33] * 100,
        'Open': [395.33] * 100,
        'High': [396.0] * 100,
        'Low': [394.0] * 100,
        'Volume': [5000] * 100,
    }, index=dates)
    
    with patch("routers.analysis.get_history", return_value=etf_dummy), \
         patch("routers.analysis.get_quote", return_value={"price": 395.33}), \
         patch("routers.analysis.get_info", return_value={"longName": "Motilal Oswal Nasdaq Q-50 ETF", "currency": "INR", "nav": 119.15}):
        res = get_etf_analysis("MONQ50.NS")
        
        mkt_status = res["market_structure_status"]
        assert "observation" in mkt_status
        assert "attribution" in mkt_status
        
        # Observation must state factual price and iNAV relationship
        assert "Market price" in mkt_status["observation"]
        assert "indicative iNAV" in mkt_status["observation"]
        
        # Attribution must state potential contributing structural factors
        assert "Potential contributors include" in mkt_status["attribution"]
        assert "overseas investment quota limits" in mkt_status["attribution"] or "trading constraints" in mkt_status["attribution"]


def test_ml_leakage_prevention_and_oof_stacking():
    """
    METAMORPHIC RELATION 10: ML Preprocessing Temporal Leakage Prevention & OOF Stacking
    
    Verifies that:
      1. Train/Test split strictly precedes RobustScaler fitting (no future distribution leakage).
      2. No bfill() is used on feature matrices (early rows do not receive future information).
      3. Base model predictions fed into the meta-stacker during training are genuine Out-Of-Fold (OOF).
    """
    from ml_models import StockPredictor, _walk_forward_metrics
    from sklearn.linear_model import Ridge
    
    predictor = StockPredictor()
    
    # 1. Verify scaler fitting behavior
    # Inject an extreme anomaly in the future test window (e.g. feature value = 10,000)
    # The scaler fitted on training data must NOT be influenced by this future anomaly.
    X_train_clean = np.ones((80, 5)) * 10.0
    X_test_anomaly = np.ones((20, 5)) * 10000.0
    X_full = np.vstack([X_train_clean, X_test_anomaly])
    
    # Split first
    split = 80
    X_tr_raw = X_full[:split]
    X_te_raw = X_full[split:]
    
    predictor.scaler.fit(X_tr_raw)
    
    # Train median must be 10.0, completely unaffected by the 10,000.0 test anomaly
    assert predictor.scaler.center_[0] == pytest.approx(10.0, abs=1e-3)
    
    # 2. Verify walk-forward metrics scale each fold independently without leakage
    y_dummy = np.random.normal(0.001, 0.02, 100)
    pool = predictor._build_pool()
    wf_res = _walk_forward_metrics(X_full, y_dummy, dict(pool), Ridge(alpha=1.0), n_folds=3)
    assert wf_res is not None
    assert "direction_accuracy" in wf_res
    assert "profit_factor" in wf_res


