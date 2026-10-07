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


def test_universal_peer_resolution():
    from peer_data import get_peers
    test_cases = [
        ("RELIANCE.NS", "Oil & Gas"),
        ("AAPL", "US Mega-Cap Tech"),
        ("NVDA", "Semiconductors & AI"),
        ("MONQ50.NS", "Index & Sectoral ETFs"),
        ("BEL.NS", "Defence"),
        ("HAL.NS", "Defence"),
        ("IRFC.NS", "Railways"),
        ("ETERNAL.NS", "Internet & Tech"),
        ("SWIGGY.NS", "Internet & Tech"),
        ("PFC.NS", "Power & Energy"),
        ("SPY", "Global & US ETFs"),
    ]
    for sym, expected_sector in test_cases:
        res = get_peers(sym)
        assert res["found"] is True
        assert res["sector"] == expected_sector
        assert len(res["peers"]) >= 3
        # Queried ticker must never be in its own peer list
        assert sym not in res["peers"]
        assert sym.replace(".NS", "") not in res["peers"]


def test_sector_rank_computation_and_aggregates():
    from routers.analysis import sector_rank_endpoint

    dummy_metrics = {
        "RELIANCE.NS": {"ticker": "RELIANCE.NS", "ret_1m": 2.0, "ret_3m": 5.0, "ret_1y": 15.0, "sharpe": 1.2, "annual_vol": 18.0, "rsi": 55.0},
        "ONGC.NS":     {"ticker": "ONGC.NS",     "ret_1m": 4.0, "ret_3m": 12.0, "ret_1y": 25.0, "sharpe": 1.5, "annual_vol": 22.0, "rsi": 62.0},
        "BPCL.NS":     {"ticker": "BPCL.NS",     "ret_1m": -1.0, "ret_3m": -2.0, "ret_1y": 8.0, "sharpe": 0.5, "annual_vol": 25.0, "rsi": 45.0},
        "IOC.NS":      {"ticker": "IOC.NS",      "ret_1m": 0.5, "ret_3m": 1.0, "ret_1y": 10.0, "sharpe": 0.7, "annual_vol": 20.0, "rsi": 48.0},
    }

    with patch("routers.analysis._compute_quick_metrics", side_effect=lambda t: dummy_metrics.get(t)):
        res = sector_rank_endpoint(ticker="RELIANCE.NS")
        assert res["ticker"] == "RELIANCE.NS"
        assert res["sector"] == "Oil & Gas"
        assert len(res["ranked"]) == 4

        # Rank must be sorted descending by score
        scores = [r["score"] for r in res["ranked"]]
        assert scores == sorted(scores, reverse=True)

        # Sector averages must be computed
        avg_3m = res["sector_averages"]["avg_ret_3m"]
        assert avg_3m == pytest.approx((5.0 + 12.0 - 2.0 + 1.0) / 4.0, rel=1e-2)

        # Alpha and tier must be present
        for r in res["ranked"]:
            assert "alpha_3m" in r
            assert "alpha_1y" in r
            assert r["tier"] in ["LEADER", "OUTPERFORMER", "MARKET PERFORMER", "LAGGARD"]
            assert r["ml_signal"] in ["STRONG BUY", "BUY", "HOLD", "SELL"]


def test_sector_chart_endpoint_normalized_returns():
    from routers.analysis import sector_chart_endpoint
    import pandas as pd
    import numpy as np

    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
    df_a = pd.DataFrame({"Close": np.linspace(100, 110, 30)}, index=dates)
    df_b = pd.DataFrame({"Close": np.linspace(200, 240, 30)}, index=dates)

    def mock_hist(t, period="3mo", interval="1d"):
        return df_a if "INFY" in t else df_b

    with patch("routers.analysis.get_history", side_effect=mock_hist):
        res = sector_chart_endpoint(ticker="INFY.NS", period="3mo")
        assert res["ticker"] == "INFY.NS"
        assert res["period"] == "3mo"
        assert "series" in res
        series_tickers = [s["ticker"] for s in res["series"]]
        assert "INFY.NS" in series_tickers
        assert "benchmark" in res
        assert res["benchmark"]["name"] == "IT Benchmark"
        assert len(res["benchmark"]["data"]) == len(res["dates"])

        infy_series = next(s for s in res["series"] if s["ticker"] == "INFY.NS")
        # Base return must start at 0.0%
        assert infy_series["data"][0] == 0.0
        # Final point should match approx +10%
        assert infy_series["data"][-1] == pytest.approx(10.0, rel=1e-1)
        assert "pair_spread" in res


def test_multi_compare_endpoint_and_leaders():
    from routers.analysis import multi_compare_endpoint

    dummy = {
        "TCS.NS": {"ticker": "TCS.NS", "company_name": "Tata Consultancy Services", "current_price": 3500.0, "ret_1m": 2.0, "ret_1y": 20.0, "sharpe": 1.5, "pe_ratio": 28.0, "roe": 45.0, "annual_vol": 16.0},
        "INFY.NS": {"ticker": "INFY.NS", "company_name": "Infosys", "current_price": 1800.0, "ret_1m": 4.0, "ret_1y": 30.0, "sharpe": 1.2, "pe_ratio": 24.0, "roe": 30.0, "annual_vol": 19.0},
        "WIPRO.NS": {"ticker": "WIPRO.NS", "company_name": "Wipro", "current_price": 480.0, "ret_1m": -1.0, "ret_1y": 10.0, "sharpe": 0.8, "pe_ratio": 20.0, "roe": 15.0, "annual_vol": 22.0},
    }

    with patch("routers.analysis._compute_quick_metrics", side_effect=lambda t: dummy.get(t)):
        res = multi_compare_endpoint(ticker="TCS.NS", peers="INFY.NS,WIPRO.NS")
        assert res["queried_ticker"] == "TCS.NS"
        assert len(res["metrics"]) == 3
        assert res["leaders"]["ret_1y"] == "INFY.NS"  # Highest 1y return
        assert res["leaders"]["pe_ratio"] == "WIPRO.NS" # Lowest PE ratio
        assert res["leaders"]["roe"] == "TCS.NS"       # Highest ROE
        assert res["leaders"]["annual_vol"] == "TCS.NS" # Lowest volatility


def test_strict_15_day_news_filtering_and_decay_weighting():
    from services.intelligent_news_reader import IntelligentNewsReader, MAX_NEWS_AGE_DAYS
    reader = IntelligentNewsReader()

    assert MAX_NEWS_AGE_DAYS == 15.0

    # Test exponential recency weighting decay buckets
    w24, b24 = reader._compute_recency_weight(12.0)
    assert w24 == 1.00 and b24 == "24h"

    w48, b48 = reader._compute_recency_weight(48.0)
    assert w48 == 0.85 and b48 == "3d"

    w120, b120 = reader._compute_recency_weight(120.0)
    assert w120 == 0.55 and b120 == "7d"

    w240, b240 = reader._compute_recency_weight(240.0)
    assert w240 == 0.25 and b240 == "15d"

    # Test company name resolver for major tickers
    clean_lt, terms_lt = reader._resolve_company_search_terms("LT.NS")
    assert "Larsen & Toubro" in terms_lt or "Larsen" in clean_lt

    clean_tata, terms_tata = reader._resolve_company_search_terms("TATAMOTORS.NS")
    assert "Tata Motors" in terms_tata or "Tata Motors" in clean_tata


def test_news_decision_engine_buy_or_not_signals():
    from services.intelligent_news_reader import IntelligentNewsReader
    reader = IntelligentNewsReader()

    # Scenario 1: High-conviction Positive Catalyst (Order win, profit surges)
    pos_text = "Larsen & Toubro wins mega Rs 15,000 crore international EPC contract. Record quarterly profit surges 35% with robust order inflows."
    pos_sentiment = reader._calculate_financial_sentiment(pos_text)
    pos_catalysts = reader._extract_catalysts(pos_text)
    pos_articles = [{"title": pos_text, "summary": pos_text, "source": "Economic Times", "recency_bucket": "24h", "sentiment": 0.6}]
    
    pos_decision = reader._synthesize_decision(
        sentiment_res=pos_sentiment,
        catalysts=pos_catalysts,
        articles=pos_articles,
        ticker="LT.NS",
        recency_stats={'within_24h': 1, 'within_3d': 0, 'within_7d': 0, 'within_15d': 0, 'total_recent': 1},
        weighted_score=0.65
    )

    assert pos_decision["action_signal"] in ["STRONG BUY", "BUY"]
    assert pos_decision["decision_score"] >= 65.0
    assert pos_decision["is_buy_vetoed"] is False
    assert "BUY" in pos_decision["executive_rationale"]
    assert len(pos_decision["decision_drivers"]) >= 2

    # Scenario 2: Routine Neutral Coverage (Hold / Watchlist)
    neutral_sentiment = {"score": 0.02, "label": "NEUTRAL", "confidence": 50.0}
    neutral_catalysts = []
    neutral_articles = [{"title": "Markets trade flat today", "summary": "Nifty holds key levels", "source": "Media", "recency_bucket": "3d", "sentiment": 0.0}]
    
    neutral_decision = reader._synthesize_decision(
        sentiment_res=neutral_sentiment,
        catalysts=neutral_catalysts,
        articles=neutral_articles,
        ticker="INFY.NS",
        recency_stats={'within_24h': 0, 'within_3d': 1, 'within_7d': 0, 'within_15d': 0, 'total_recent': 1},
        weighted_score=0.02
    )

    assert neutral_decision["action_signal"] == "HOLD / WATCHLIST"
    assert 42.0 <= neutral_decision["decision_score"] <= 58.0
    assert neutral_decision["is_buy_vetoed"] is False
    assert "HOLD" in neutral_decision["executive_rationale"]

    # Scenario 3: Critical Headwind & Hard Risk Veto (SEBI probe / Lower circuit / Fraud)
    veto_text = "Company promoter summoned under SEBI probe and ED tax raid. Shares locked in lower circuit amid forensic audit."
    veto_sentiment = reader._calculate_financial_sentiment(veto_text)
    veto_catalysts = reader._extract_catalysts(veto_text)
    veto_articles = [{"title": veto_text, "summary": veto_text, "source": "LiveMint", "recency_bucket": "24h", "sentiment": -0.8}]
    
    veto_decision = reader._synthesize_decision(
        sentiment_res=veto_sentiment,
        catalysts=veto_catalysts,
        articles=veto_articles,
        ticker="RISKY.NS",
        recency_stats={'within_24h': 1, 'within_3d': 0, 'within_7d': 0, 'within_15d': 0, 'total_recent': 1},
        weighted_score=-0.75
    )

    assert veto_decision["action_signal"] == "STRONG AVOID / DO NOT BUY"
    assert veto_decision["is_buy_vetoed"] is True
    assert veto_decision["decision_score"] <= 22.0
    assert "DO NOT BUY" in veto_decision["executive_rationale"]
    assert veto_decision["veto_reason"] is not None


def test_engine_signal_harmonization_with_risk_veto():
    from engine import analyze_ticker
    import pandas as pd
    import numpy as np
    from unittest.mock import patch

    # Mock historical candle data that would normally produce a technical BUY
    dates = pd.date_range(end=pd.Timestamp.now(tz="Asia/Kolkata"), periods=100, freq="B")
    t = np.linspace(0, 10, 100)
    close_prices = 100 + t * 4 + np.sin(t * 3) * 2 # realistic upward trend with normal osc
    df = pd.DataFrame({
        "Open": close_prices - 0.5,
        "High": close_prices + 2.0,
        "Low": close_prices - 2.0,
        "Close": close_prices,
        "Volume": [50000] * 100
    }, index=dates)

    # Mock news with active risk veto
    veto_decision = {
        "action_signal": "STRONG AVOID / DO NOT BUY",
        "decision_score": 15.0,
        "is_buy_vetoed": True,
        "veto_reason": "CRO Risk Gate VETO: SEBI PROBE detected in 15-day news flow."
    }

    with patch("engine.get_history", return_value=df), \
         patch("engine.get_quote", return_value={"price": 150.0}), \
         patch("engine.fetch_fundamentals", return_value={}), \
         patch("engine.calculate_relative_strength", return_value={}), \
         patch("engine.fetch_news_sentiment", return_value=(-0.8, 0.5, [], veto_decision)):
        res = analyze_ticker("TEST.NS")
        # When vetoed, technical BUY must be overridden to HOLD
        assert res["summary"]["signal"] != "BUY"
        assert res["summary"]["signal"] in ["HOLD", "SELL"]
        assert any("Risk Gate VETO" in reason for reason in res["summary"]["signalReasons"])
        assert "decision" in res["sentiment"]
        assert res["sentiment"]["decision"]["is_buy_vetoed"] is True



