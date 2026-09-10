import math
import numpy as np
import pandas as pd
import pytest

from engine import (
    calculate_atr,
    calculate_adx,
    calculate_rsi,
    calculate_support_resistance,
    calculate_risk_metrics,
    generate_signal,
)
from ml_models import _walk_forward_metrics, _detect_regime
from news_intelligence import NewsIntelligence
from routers.intraday import (
    _calculate_atr as intraday_atr,
    _calculate_rsi as intraday_rsi,
    _calculate_orb,
    _calculate_pivots,
)
from sklearn.linear_model import Ridge


def test_calculate_atr_and_adx_bounded():
    np.random.seed(42)
    n = 100
    prices = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    highs = prices + np.abs(np.random.randn(n) * 1.0)
    lows = prices - np.abs(np.random.randn(n) * 1.0)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': np.full(n, 100000)
    })

    df['ATR'] = calculate_atr(df, n=14)
    df_adx = calculate_adx(df, n=14)

    assert '+DI' in df_adx.columns
    assert '-DI' in df_adx.columns
    assert 'ADX' in df_adx.columns

    valid_adx = df_adx.dropna(subset=['+DI', '-DI', 'ADX'])
    assert len(valid_adx) > 50

    # Verify strictly bounded in [0, 100]
    assert (valid_adx['+DI'] >= 0.0).all()
    assert (valid_adx['+DI'] <= 100.0).all()
    assert (valid_adx['-DI'] >= 0.0).all()
    assert (valid_adx['-DI'] <= 100.0).all()
    assert (valid_adx['ADX'] >= 0.0).all()
    assert (valid_adx['ADX'] <= 100.0).all()


def test_calculate_support_resistance_price_anchoring():
    np.random.seed(42)
    # Strong uptrend where swing lows are higher than earlier swing highs
    n = 80
    trend = np.linspace(100, 200, n)
    highs = trend + 5.0 + np.sin(np.linspace(0, 10, n)) * 2.0
    lows = trend - 5.0 + np.sin(np.linspace(0, 10, n)) * 2.0
    closes = trend + 1.0

    df = pd.DataFrame({'High': highs, 'Low': lows, 'Close': closes})
    curr_price = closes[-1]

    support, resistance = calculate_support_resistance(df, window=5)

    assert support < curr_price, f"Support ({support}) must be below current price ({curr_price})"
    assert resistance > curr_price, f"Resistance ({resistance}) must be above current price ({curr_price})"


def test_generate_signal_crossover_logic():
    # Test bullish crossover (was below, now above)
    sig, score, reasons = generate_signal(
        rsi=50.0, macd=1.5, signal_line=1.0, close=100.0,
        upper_band=110.0, lower_band=90.0, adx=26.0,
        prev_macd=0.8, prev_signal=1.0
    )
    assert any("bullish crossover" in r for r in reasons)

    # Test bearish crossover (was above, now below)
    sig, score, reasons = generate_signal(
        rsi=50.0, macd=0.8, signal_line=1.0, close=100.0,
        upper_band=110.0, lower_band=90.0, adx=26.0,
        prev_macd=1.2, prev_signal=1.0
    )
    assert any("bearish crossover" in r for r in reasons)

    # Test converged equal state
    sig, score, reasons = generate_signal(
        rsi=50.0, macd=1.0, signal_line=1.0, close=100.0,
        upper_band=110.0, lower_band=90.0, adx=20.0,
        prev_macd=1.0, prev_signal=1.0
    )
    assert any("converged" in r for r in reasons)


def test_calculate_risk_metrics_calmar_cagr():
    # Test steady compounder
    prices = pd.Series([100.0 * (1.001 ** i) for i in range(252)])
    metrics = calculate_risk_metrics(prices)

    assert 'annualizedVolatility' in metrics
    assert 'maxDrawdown' in metrics
    assert 'calmarRatio' in metrics
    assert 'var95_1D' in metrics
    assert 'cvar95_1D' in metrics


def test_walk_forward_metrics_strategy_return():
    X = np.ones((60, 5))
    y = np.random.randn(60) * 0.02

    class DummyModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.full(len(X), 0.01)

    models = {'m': DummyModel()}
    meta = Ridge()

    metrics = _walk_forward_metrics(X, y, models, meta, n_folds=3)
    assert 'direction_accuracy' in metrics
    assert 'profit_factor' in metrics
    assert metrics['profit_factor'] > 0
    assert metrics['profit_factor'] <= 9.99


def test_news_intelligence_sentiment_bounded():
    ni = NewsIntelligence()
    mock_articles = [
        {
            'title': 'Huge quarterly profit beat and guidance upgrade',
            'summary': 'Surge in margins and growth outperforms analyst estimates',
            'published': pd.Timestamp.now(),
            'relevance_score': 0.8
        },
        {
            'title': 'Minor drop in commodity margin',
            'summary': 'Decline in short term volumes',
            'published': pd.Timestamp.now(),
            'relevance_score': 0.2
        }
    ]

    result = ni.analyze_sentiment_advanced(mock_articles)
    assert -1.0 <= result['overall_sentiment'] <= 1.0
    assert 0.0 <= result['confidence'] <= 100.0


def test_intraday_indicators_and_orb():
    # Test intraday Wilder smoothing
    series = pd.Series([100 + i + (i % 3) for i in range(50)])
    rsi = intraday_rsi(series, period=14)
    assert len(rsi) == 50
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()

    df = pd.DataFrame({
        'High': series + 2.0,
        'Low': series - 2.0,
        'Close': series,
    })
    atr = intraday_atr(df, period=14)
    assert len(atr) == 50
    assert (atr >= 0.0).all()

    # Test ORB early session forming status
    early_df = df.iloc[:2]  # only 2 candles
    orb = _calculate_orb(early_df, interval="5m")
    assert orb['status'] == "FORMING_RANGE"
