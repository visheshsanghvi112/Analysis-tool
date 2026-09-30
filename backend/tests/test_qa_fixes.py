import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

def test_financial_dcf_flag():
    from routers.analysis import calculate_dcf
    with patch("routers.analysis.get_info") as mock_info:
        mock_info.return_value = {
            "currentPrice": 150.0,
            "sector": "Financial Services",
            "industry": "Credit Services",
            "marketCap": 2000000000000,
            "sharesOutstanding": 13000000000,
            "totalCash": 50000000000,
            "totalDebt": 3500000000000,  # Huge debt typical of IRFC
            "netIncomeToCommon": 65000000000,
            "trailingEps": 5.0,
            "bookValue": 40.0,
            "returnOnEquity": 0.14,
            "returnOnAssets": 0.015,
            "currency": "INR",
        }
        res = calculate_dcf(ticker="IRFC.NS")
        assert res["is_financial"] is True
        assert res["dcf_defaults"]["starting_flow"] == 65000000000
        assert "Net Income" in res["dcf_defaults"]["flow_type"]

def test_ml_negative_price_floor_and_clip():
    from ml_models import StockPredictor
    predictor = StockPredictor()
    predictor.models = {
        'bayesian_ridge': MagicMock(predict=MagicMock(return_value=np.array([-1.5]))),  # Severe negative
        'random_forest': MagicMock(predict=MagicMock(return_value=np.array([0.05]))),
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([-0.8])))
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
        # Verify Bayesian ridge price is strictly positive despite negative return
        assert pred['models']['bayesian_ridge']['predicted_price'] > 0
        assert pred['predicted_price'] > 0
        # Verify clipping was enforced
        assert pred['models']['bayesian_ridge']['predicted_return'] >= -75.0

def test_backtest_excess_return_label():
    from routers.analysis import run_backtest
    with patch("routers.analysis.get_history") as mock_hist:
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        prices = np.linspace(100, 150, 100)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": [10000] * 100
        }, index=dates)
        mock_hist.return_value = df
        res = run_backtest(ticker="TEST.NS", period="1y")
        assert "excess_return_vs_bh" in res["stats"]
        assert "alpha" in res["stats"]
        assert res["stats"]["excess_return_vs_bh"] == res["stats"]["alpha"]

def test_news_decision_synthesis_crash_dislocation():
    from services.intelligent_news_reader import IntelligentNewsReader
    reader = IntelligentNewsReader()
    text = "Motilal Oswal Nasdaq Q 50 ETF crashes 46% after trading at 235% premium to NAV. SEBI scrutiny tightened around retail overseas trading limits and extreme price dislocation against indicative fair value."
    sentiment = reader._calculate_financial_sentiment(text)
    catalysts = reader._extract_catalysts(text)
    articles = [{"title": text, "snippet": text, "url": "http://example.com"}]
    decision = reader._synthesize_decision(sentiment, catalysts, articles, "MONQ50.NS")
    
    assert sentiment["score"] < -0.5
    assert len(catalysts) >= 1
    assert any("Dislocation" in c["type"] for c in catalysts)
    assert decision["trade_directive"] == "AVOID / HIGH TAIL RISK"
    assert any("Dislocation" in f or "Premium" in f for f in decision["cro_risk_flags"])
    assert decision["committee_vote"] == "BEARISH"

def test_desk_engine_news_cro_veto():
    from desk_engine import DeskContext, evaluate_risk_gate, calculate_trade_geometry
    ctx = DeskContext(
        ticker="MONQ50.NS",
        asset_type="ETF",
        current_price=187.0,
        atr_14=12.5,
        news_sentiment=-1.0,
        news_verdict="STRUCTURAL_RISK_AVOID",
        news_directive="AVOID / HIGH TAIL RISK",
        news_cro_flags=["NAV_DISLOCATION_DISCOUNT_RISK", "CRITICAL_SEVERITY_CATALYST"],
        news_committee_vote="BEARISH"
    )
    
    # Evaluate risk gate
    gate = evaluate_risk_gate(ctx)
    assert gate.state == "VETO"
    assert gate.risk_score == 100.0
    assert gate.sizing_cap_pct == 0.0
    assert any("CRO Risk Flag" in str(r) for r in gate.hard_blocks)
    
    # Check trade geometry blocking
    trade_ctx = {
        "entry_price": 187.0,
        "atr": 12.5,
        "account_capital": 100000,
        "account_risk_pct": 1.0,
        "direction": "LONG",
        "committee_state": "BEARISH"
    }
    geom = calculate_trade_geometry(trade_ctx, gate)
    assert geom["trade_blocked"] is True
    assert geom["shares"] == 0
    assert "CRO Risk Gate VETO" in geom["warning"]

def test_ml_signal_strength_harmonization():
    from ml_models import StockPredictor
    predictor = StockPredictor()
    predictor.models = {
        'xgboost': MagicMock(predict=MagicMock(return_value=np.array([0.15]))),
        'random_forest': MagicMock(
            predict=MagicMock(return_value=np.array([0.15])),
            estimators_=[MagicMock(predict=MagicMock(return_value=np.array([0.15])))]
        ),
    }
    predictor.meta_model = MagicMock(predict=MagicMock(return_value=np.array([0.15])))
    predictor.scaler = MagicMock(transform=MagicMock(return_value=np.zeros((1, 5))))
    predictor.feature_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
    predictor.shap_features = []

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
        # Signal strength must scale with confidence
        assert pred['signal_strength'] <= 100.0
        assert pred['signal_strength'] > 0

