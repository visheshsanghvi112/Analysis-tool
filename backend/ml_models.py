from __future__ import annotations
# ============================================================
# ML Models for Stock Prediction — by Vishesh Sanghvi
# Upgraded: diverse 6-model stacked ensemble, 40+ features,
# walk-forward financial metrics, SHAP explanations,
# news-sentiment signal fusion, benchmark comparison,
# per-ticker LRU cache, full fallback logic
# ============================================================

import numpy as np
import pandas as pd
import copy
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from yf_client import get_history, get_quote

# ── Optional heavy deps — fail gracefully ──────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Per-ticker LRU model cache (max 20 tickers in memory)
_MODEL_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 20

# ─────────────────────────────────────────────────────────────
# Model Governance Policy Configuration (SR 11-7)
# Centralized, configurable bounds for outlier detection & clipping
# ─────────────────────────────────────────────────────────────
MODEL_GOVERNANCE_THRESHOLDS = {
    "equity_5d": {
        "lower_return": -0.50,
        "upper_return": 1.00,
        "description": "5-Day Equity Return Policy: Bounds return between -50% and +100% to prevent unbounded linear extrapolation.",
    },
    "etf_30d": {
        "lower_return": -0.60,
        "upper_return": 1.20,
        "description": "30-Day ETF Return Policy: Bounds monthly return between -60% and +120% to prevent unconstrained volatility runaway.",
    },
}


# ─────────────────────────────────────────────────────────────
# Feature engineering  (~42 features)
# ─────────────────────────────────────────────────────────────
def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Core price ──
    df['returns']       = df['Close'].pct_change()
    df['log_returns']   = np.log(df['Close'] / df['Close'].shift(1))
    df['price_range']   = (df['High'] - df['Low']) / df['Close']
    df['gap']           = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['close_to_high'] = df['Close'] / df['High']
    df['close_to_low']  = df['Close'] / df['Low']

    # ── MA ratios ──
    for p in [5, 10, 20, 50, 100]:
        window = min(p, len(df))
        ma = df['Close'].rolling(window, min_periods=1).mean()
        df[f'ma_ratio_{p}'] = df['Close'] / ma

    # ── EMA cross & MACD ──
    ema9  = df['Close'].ewm(span=9, min_periods=1).mean()
    ema21 = df['Close'].ewm(span=21, min_periods=1).mean()
    ema12 = df['Close'].ewm(span=12, min_periods=1).mean()
    ema26 = df['Close'].ewm(span=26, min_periods=1).mean()
    df['ema_cross_9_21'] = (ema9 - ema21) / df['Close']
    df['macd']           = (ema12 - ema26) / df['Close']
    df['macd_signal']    = df['macd'].ewm(span=9, min_periods=1).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # ── RSI (14) ──
    delta    = df['Close'].diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
    df['rsi'] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, 1e-9))

    # ── Stochastic %K / %D ──
    w_stoch = min(14, len(df))
    low14  = df['Low'].rolling(w_stoch, min_periods=1).min()
    high14 = df['High'].rolling(w_stoch, min_periods=1).max()
    df['stoch_k'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(min(3, len(df)), min_periods=1).mean()

    # ── Williams %R ──
    df['williams_r'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-9)

    # ── ATR (14) ──
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr']       = tr.ewm(span=14, adjust=False, min_periods=1).mean()
    df['atr_ratio'] = df['atr'] / df['Close']

    # ── Bollinger Bands ──
    w_bb = min(20, len(df))
    bb_mid         = df['Close'].rolling(w_bb, min_periods=1).mean()
    bb_std         = df['Close'].rolling(w_bb, min_periods=2).std()
    df['bb_pos']   = (df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
    df['bb_width'] = (4 * bb_std) / bb_mid

    # ── Rate of Change ──
    for p in [5, 10, 20]:
        df[f'roc_{p}'] = df['Close'].pct_change(p)

    # ── Volume ──
    w_vol = min(20, len(df))
    vol_sma          = df['Volume'].rolling(w_vol, min_periods=1).mean()
    df['vol_ratio']  = df['Volume'] / vol_sma.replace(0, 1)
    df['obv']        = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['obv_ratio']  = df['obv'] / df['obv'].rolling(w_vol, min_periods=1).mean().replace(0, 1)

    # ── Volatility regime ──
    w_v20 = min(20, len(df))
    w_v60 = min(60, len(df))
    df['vol_20']    = df['returns'].rolling(w_v20, min_periods=2).std()
    df['vol_60']    = df['returns'].rolling(w_v60, min_periods=2).std()
    df['vol_regime'] = df['vol_20'] / (df['vol_60'] + 1e-9)

    # ── Distribution moments ──
    df['skew_20'] = df['returns'].rolling(w_v20, min_periods=3).skew()
    df['kurt_20'] = df['returns'].rolling(w_v20, min_periods=4).kurt()

    # ── 52-week proximity ──
    w_52 = min(252, len(df))
    high52          = df['High'].rolling(w_52, min_periods=1).max()
    low52           = df['Low'].rolling(w_52, min_periods=1).min()
    df['pct_52w_h'] = (df['Close'] - high52) / high52
    df['pct_52w_l'] = (df['Close'] - low52)  / low52.replace(0, 1)

    # ── Calendar effects ──
    idx = pd.to_datetime(df.index)
    df['day_of_week'] = idx.dayofweek / 4.0
    df['month']       = idx.month      / 12.0

    # ── Lagged returns ──
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag_{lag}'] = df['returns'].shift(lag)

    return df


FEATURE_COLS = [
    'returns', 'log_returns', 'price_range', 'gap', 'close_to_high', 'close_to_low',
    'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50', 'ma_ratio_100',
    'ema_cross_9_21', 'macd', 'macd_signal', 'macd_histogram',
    'rsi', 'stoch_k', 'stoch_d', 'williams_r',
    'atr_ratio', 'bb_pos', 'bb_width',
    'roc_5', 'roc_10', 'roc_20',
    'vol_ratio', 'obv_ratio',
    'vol_20', 'vol_regime', 'skew_20', 'kurt_20',
    'pct_52w_h', 'pct_52w_l',
    'day_of_week', 'month',
    'ret_lag_1', 'ret_lag_2', 'ret_lag_3', 'ret_lag_5', 'ret_lag_10',
]


# ─────────────────────────────────────────────────────────────
# ETF Feature Engineering (~38 long-term features)
# Used when asset_type == 'ETF' — replaces _create_features()
# Focus: trend following, macro cycles, not short-term momentum
# ─────────────────────────────────────────────────────────────
def _create_etf_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Core price ──
    df['returns']       = df['Close'].pct_change()
    df['log_returns']   = np.log(df['Close'] / df['Close'].shift(1))
    df['price_range']   = (df['High'] - df['Low']) / df['Close']
    df['close_to_high'] = df['Close'] / df['High']
    df['close_to_low']  = df['Close'] / df['Low']
    # Note: no 'gap' feature — gaps are rare/meaningless for ETFs

    # ── Long-term MA ratios (50/100/150/200-day) ──
    for p in [20, 50, 100, 150, 200]:
        window = min(p, len(df))
        ma = df['Close'].rolling(window, min_periods=1).mean()
        df[f'ma_ratio_{p}'] = df['Close'] / ma

    # ── Golden/Death Cross — the primary ETF timing signal ──
    ma50  = df['Close'].rolling(min(50,  len(df)), min_periods=1).mean()
    ma200 = df['Close'].rolling(min(200, len(df)), min_periods=1).mean()
    df['golden_cross']   = (ma50 > ma200).astype(float)   # 1 = bullish regime
    df['cross_dist_pct'] = (ma50 - ma200) / (ma200 + 1e-9) # distance of cross

    # ── Slower EMA cross (26/52 — ETF-appropriate MACD) ──
    ema26 = df['Close'].ewm(span=26, min_periods=1).mean()
    ema52 = df['Close'].ewm(span=52, min_periods=1).mean()
    df['ema_cross_26_52'] = (ema26 - ema52) / (df['Close'] + 1e-9)
    macd_line   = df['Close'].ewm(span=26, min_periods=1).mean() - df['Close'].ewm(span=52, min_periods=1).mean()
    df['macd_slow']        = macd_line / (df['Close'] + 1e-9)
    df['macd_slow_signal'] = df['macd_slow'].ewm(span=18, min_periods=1).mean()

    # ── RSI (21-period — slower, more appropriate for ETFs) ──
    delta    = df['Close'].diff()
    avg_gain = delta.clip(lower=0).ewm(com=20, adjust=False, min_periods=1).mean()
    avg_loss = (-delta).clip(lower=0).ewm(com=20, adjust=False, min_periods=1).mean()
    df['rsi_21'] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, 1e-9))

    # ── ATR (21-period) ──
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr_ratio'] = tr.ewm(span=21, adjust=False, min_periods=1).mean() / (df['Close'] + 1e-9)

    # ── Bollinger Bands (50-period for ETFs) ──
    w_bb = min(50, len(df))
    bb_mid       = df['Close'].rolling(w_bb, min_periods=1).mean()
    bb_std       = df['Close'].rolling(w_bb, min_periods=2).std()
    df['bb_pos']   = (df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
    df['bb_width'] = (4 * bb_std) / (bb_mid + 1e-9)

    # ── Monthly/Quarterly/Semi-annual Rate of Change (ETF momentum) ──
    df['roc_21']  = df['Close'].pct_change(21)   # ~1 month
    df['roc_63']  = df['Close'].pct_change(63)   # ~1 quarter
    df['roc_126'] = df['Close'].pct_change(126)  # ~6 months

    # ── Volume (simpler — ETF volume is less meaningful than for stocks) ──
    w_vol = min(20, len(df))
    vol_sma         = df['Volume'].rolling(w_vol, min_periods=1).mean()
    df['vol_ratio'] = df['Volume'] / vol_sma.replace(0, 1)

    # ── Volatility regime (20d / 60d / 252d annual) ──
    w_20  = min(20,  len(df))
    w_60  = min(60,  len(df))
    w_252 = min(252, len(df))
    df['vol_20']     = df['returns'].rolling(w_20,  min_periods=2).std()
    df['vol_60']     = df['returns'].rolling(w_60,  min_periods=2).std()
    df['vol_252']    = df['returns'].rolling(w_252, min_periods=5).std()
    df['vol_regime'] = df['vol_20'] / (df['vol_60'] + 1e-9)

    # ── Trend strength (ADX proxy: |mean return| / std of returns) ──
    df['adx_proxy'] = df['returns'].rolling(min(21, len(df)), min_periods=3).apply(
        lambda x: abs(x).mean() / (x.std() + 1e-9), raw=True
    )

    # ── Distribution moments (63d — quarterly window) ──
    df['skew_63'] = df['returns'].rolling(min(63, len(df)), min_periods=5).skew()
    df['kurt_63'] = df['returns'].rolling(min(63, len(df)), min_periods=6).kurt()

    # ── 52-week + annual drawdown proximity ──
    w_52 = min(252, len(df))
    high52           = df['High'].rolling(w_52, min_periods=1).max()
    low52            = df['Low'].rolling(w_52,  min_periods=1).min()
    df['pct_52w_h']  = (df['Close'] - high52) / (high52 + 1e-9)
    df['pct_52w_l']  = (df['Close'] - low52)  / (low52.replace(0, 1) + 1e-9)
    df['annual_dd']  = (df['Close'] - high52) / (high52 + 1e-9)   # always <= 0

    # ── Calendar effects (month + quarter-end rebalancing flag) ──
    idx = pd.to_datetime(df.index)
    df['month']       = idx.month / 12.0
    df['quarter_end'] = idx.month.isin([3, 6, 9, 12]).astype(float)

    # ── Long-period lagged returns (weekly / monthly / quarterly) ──
    for lag in [5, 10, 21, 63, 126]:
        df[f'ret_lag_{lag}'] = df['returns'].shift(lag)

    return df


ETF_FEATURE_COLS = [
    'returns', 'log_returns', 'price_range', 'close_to_high', 'close_to_low',
    'ma_ratio_20', 'ma_ratio_50', 'ma_ratio_100', 'ma_ratio_150', 'ma_ratio_200',
    'golden_cross', 'cross_dist_pct',
    'ema_cross_26_52', 'macd_slow', 'macd_slow_signal',
    'rsi_21',
    'atr_ratio', 'bb_pos', 'bb_width',
    'roc_21', 'roc_63', 'roc_126',
    'vol_ratio',
    'vol_20', 'vol_60', 'vol_252', 'vol_regime',
    'adx_proxy',
    'skew_63', 'kurt_63',
    'pct_52w_h', 'pct_52w_l', 'annual_dd',
    'month', 'quarter_end',
    'ret_lag_5', 'ret_lag_10', 'ret_lag_21', 'ret_lag_63', 'ret_lag_126',
]

# Human-readable labels for ETF features (used in SHAP waterfall)
_ETF_FEATURE_LABELS = {
    'returns':          'Daily Returns',
    'log_returns':      'Log Returns',
    'price_range':      'Daily Price Range',
    'close_to_high':    'Close-to-High Ratio',
    'close_to_low':     'Close-to-Low Ratio',
    'ma_ratio_20':      'MA Ratio (20d)',
    'ma_ratio_50':      'MA Ratio (50d)',
    'ma_ratio_100':     'MA Ratio (100d)',
    'ma_ratio_150':     'MA Ratio (150d)',
    'ma_ratio_200':     'MA Ratio (200d)',
    'golden_cross':     'Golden Cross Signal',
    'cross_dist_pct':   'MA Cross Distance',
    'ema_cross_26_52':  'EMA Cross (26/52)',
    'macd_slow':        'MACD (Slow)',
    'macd_slow_signal': 'MACD Signal (Slow)',
    'rsi_21':           'RSI (21)',
    'atr_ratio':        'ATR Ratio (21d)',
    'bb_pos':           'Bollinger Band Position',
    'bb_width':         'Bollinger Band Width',
    'roc_21':           '1-Month Momentum',
    'roc_63':           '3-Month Momentum',
    'roc_126':          '6-Month Momentum',
    'vol_ratio':        'Volume Ratio',
    'vol_20':           '20d Volatility',
    'vol_60':           '60d Volatility',
    'vol_252':          'Annual Volatility',
    'vol_regime':       'Volatility Regime',
    'adx_proxy':        'Trend Strength (ADX)',
    'skew_63':          'Return Skewness (63d)',
    'kurt_63':          'Return Kurtosis (63d)',
    'pct_52w_h':        '52-Week High Proximity',
    'pct_52w_l':        '52-Week Low Proximity',
    'annual_dd':        'Annual Drawdown',
    'month':            'Calendar Month',
    'quarter_end':      'Quarter-End Flag',
    'ret_lag_5':        'Return Lag 1wk',
    'ret_lag_10':       'Return Lag 2wk',
    'ret_lag_21':       'Return Lag 1mo',
    'ret_lag_63':       'Return Lag 1Q',
    'ret_lag_126':      'Return Lag 6mo',
}




# ─────────────────────────────────────────────────────────────
# Market regime detection
# ─────────────────────────────────────────────────────────────
# Market regime detection (Hidden Markov Model)
# ─────────────────────────────────────────────────────────────
def _detect_regime(df: pd.DataFrame) -> str:
    try:
        from hmmlearn import hmm
        # Extract return history (minimum 60 points required to fit HMM stably)
        returns = df['returns'].dropna().values
        if len(returns) < 60:
            return "SIDEWAYS"
        
        # Scale returns by 100 for numerical stability
        X = (returns * 100).reshape(-1, 1)
        
        # Fit Gaussian HMM with 3 states
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42, min_covar=1e-4)
        model.fit(X)
        
        # Predict states
        states = model.predict(X)
        latest_state = states[-1]
        
        # Sort state indices by variance (volatility) to map to Low/Medium/High Volatility
        variances = np.array([model.covars_[i][0][0] for i in range(model.n_components)])
        sorted_indices = np.argsort(variances)
        
        if latest_state == sorted_indices[0]:
            return "LOW_VOLATILITY"
        elif latest_state == sorted_indices[1]:
            return "MEDIUM_VOLATILITY"
        else:
            return "HIGH_VOLATILITY"
    except Exception as e:
        print(f"[ML] HMM Regime detection failed: {e}")
        # Fallback to rule-based volatility regime classification if HMM fails
        try:
            vol_now = df['returns'].tail(20).std()
            vol_long = df['returns'].tail(60).std()
            if vol_now > 1.4 * vol_long or (vol_now * np.sqrt(252) > 0.35):
                return "HIGH_VOLATILITY"
            elif vol_now < 0.8 * vol_long or (vol_now * np.sqrt(252) < 0.16):
                return "LOW_VOLATILITY"
            return "MEDIUM_VOLATILITY"
        except Exception:
            return "MEDIUM_VOLATILITY"


def _forecast_garch_volatility(df: pd.DataFrame) -> dict | None:
    """
    Fits a GARCH(1,1) volatility forecasting model to returns
    and predicts the average annualized volatility for the next 5 days.
    Falls back to EWMA (RiskMetrics lambda=0.94) if arch package is not installed.
    Returns structured specification dictionary.
    """
    try:
        # Scaled returns by 100 for numerical stability
        returns = df['returns'].dropna().values * 100
        if len(returns) < 60:
            return None
        
        try:
            from arch import arch_model
            garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
            res = garch.fit(update_freq=0, disp='off')
            
            forecast = res.forecast(horizon=5)
            # Extract the average variance forecast for the 5-day horizon and annualize it
            mean_variance = float(forecast.variance.iloc[-1].mean())
            daily_cond_sigma = float(np.sqrt(mean_variance))
            annualized_vol = float(daily_cond_sigma * np.sqrt(252))

            return {
                'annualized_pct': round(annualized_vol, 2),
                'daily_conditional_std_pct': round(daily_cond_sigma, 2),
                'forecast_horizon_days': 5,
                'estimation_window_days': len(returns),
                'model_spec': 'GARCH(1,1) Normal (Annualized ×√252)',
            }
        except ImportError:
            # Robust fallback to EWMA (RiskMetrics lambda=0.94)
            decay = 0.94
            sq_ret = (returns - returns.mean()) ** 2
            ewma_var = sq_ret[0]
            for r in sq_ret[1:]:
                ewma_var = decay * ewma_var + (1 - decay) * r
            daily_cond_sigma = float(np.sqrt(ewma_var))
            annualized_vol = float(daily_cond_sigma * np.sqrt(252))

            return {
                'annualized_pct': round(annualized_vol, 2),
                'daily_conditional_std_pct': round(daily_cond_sigma, 2),
                'forecast_horizon_days': 5,
                'estimation_window_days': len(returns),
                'model_spec': 'EWMA (RiskMetrics λ=0.94, Annualized ×√252)',
            }
    except Exception as e:
        print(f"[ML] Volatility forecasting failed: {e}")
        return None



# ─────────────────────────────────────────────────────────────
# Walk-forward financial evaluation
# ─────────────────────────────────────────────────────────────
def _walk_forward_metrics(X_raw: np.ndarray, y: np.ndarray,
                          models: dict, meta: Ridge,
                          n_folds: int = 5) -> dict:
    """
    Returns direction_accuracy, profit_factor, hit_rate,
    avg_win, avg_loss from expanding walk-forward folds.
    Also benchmarks XGBoost-alone vs full ensemble.
    Strictly scales each fold independently (no temporal leakage)
    and uses Out-Of-Fold (OOF) cross-validation for the meta-learner.
    """
    fold_size = max(1, len(X_raw) // (n_folds + 1))
    all_y, all_blend, all_xgb_only = [], [], []

    xgb_model = models.get('xgboost') or models.get('random_forest')

    for i in range(1, n_folds + 1):
        tr_end = fold_size * i
        te_end = min(tr_end + fold_size, len(X_raw))
        if te_end <= tr_end + 5:
            continue

        # 1. Temporal Leakage Prevention: Scale strictly on training fold
        scaler_fold = RobustScaler()
        X_tr = scaler_fold.fit_transform(X_raw[:tr_end])
        X_te = scaler_fold.transform(X_raw[tr_end:te_end])
        y_tr = y[:tr_end]
        y_te = y[tr_end:te_end]

        fold_preds = []
        xgb_preds  = None
        for name, mdl in models.items():
            try:
                mdl.fit(X_tr, y_tr)
                p = mdl.predict(X_te)
                fold_preds.append(p)
                if name in ('xgboost', 'random_forest') and xgb_preds is None:
                    xgb_preds = p
            except Exception:
                pass

        if not fold_preds:
            continue

        meta_in = np.column_stack(fold_preds)

        # 2. Out-of-Fold (OOF) Stacking inside walk-forward fold
        try:
            if len(X_tr) >= 20:
                inner_kf = KFold(n_splits=min(3, max(2, len(X_tr) // 10)), shuffle=False)
                oof_wf = {name: np.zeros(len(X_tr)) for name in models.keys()}
                for inner_tr, inner_val in inner_kf.split(X_tr):
                    for name, mdl in models.items():
                        try:
                            f_m = clone(mdl) if hasattr(mdl, 'get_params') else copy.deepcopy(mdl)
                            f_m.fit(X_tr[inner_tr], y_tr[inner_tr])
                            oof_wf[name][inner_val] = f_m.predict(X_tr[inner_val])
                        except Exception:
                            pass
                meta_in_tr = np.column_stack([oof_wf[name] for name in models.keys()])
                meta.fit(meta_in_tr, y_tr)
                blend = meta.predict(meta_in)
            else:
                blend = np.mean(fold_preds, axis=0)
        except Exception:
            blend = np.mean(fold_preds, axis=0)

        all_y.extend(y_te.tolist())
        all_blend.extend(blend.tolist())
        if xgb_preds is not None:
            all_xgb_only.extend(xgb_preds.tolist())

    if not all_y:
        return {'direction_accuracy': 50.0, 'profit_factor': 1.0,
                'hit_rate': 50.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'ensemble_vs_xgb_delta': 0.0}

    y_arr = np.array(all_y)
    b_arr = np.array(all_blend)

    correct    = np.sign(b_arr) == np.sign(y_arr)
    dir_acc    = float(np.mean(correct) * 100)

    strat_ret = np.sign(b_arr) * y_arr
    wins = strat_ret[strat_ret > 0]
    loses = np.abs(strat_ret[strat_ret < 0])

    gross_win  = float(wins.sum())  if len(wins)  else 0.0
    gross_loss = float(loses.sum()) if len(loses) else 0.0
    profit_fac = (gross_win / gross_loss) if gross_loss > 0 else (9.99 if gross_win > 0 else 1.0)
    profit_fac = min(profit_fac, 9.99)

    avg_win  = float(wins.mean())  if len(wins)  else 0.0
    avg_loss = float(-loses.mean()) if len(loses) else 0.0

    # Benchmark: ensemble direction acc - xgb-alone direction acc
    xgb_delta = 0.0
    if all_xgb_only:
        x_arr  = np.array(all_xgb_only[:len(y_arr)])
        xgb_da = float(np.mean(np.sign(x_arr) == np.sign(y_arr)) * 100)
        xgb_delta = round(dir_acc - xgb_da, 1)

    return {
        'direction_accuracy':     round(dir_acc, 1),
        'profit_factor':          round(profit_fac, 2),
        'hit_rate':               round(dir_acc, 1),
        'avg_win_pct':            round(avg_win * 100, 3),
        'avg_loss_pct':           round(avg_loss * 100, 3),
        'ensemble_vs_xgb_delta':  xgb_delta,   # +ve means ensemble better
    }


# ─────────────────────────────────────────────────────────────
# Human-readable feature label map
# ─────────────────────────────────────────────────────────────
_FEATURE_LABELS = {
    'rsi':            'RSI (14)',
    'macd':           'MACD Line',
    'macd_histogram': 'MACD Histogram',
    'macd_signal':    'MACD Signal',
    'bb_pos':         'Bollinger Band Position',
    'bb_width':       'Bollinger Band Width',
    'atr_ratio':      'ATR Ratio',
    'vol_regime':     'Volatility Regime',
    'vol_20':         '20d Volatility',
    'vol_60':         '60d Volatility',
    'stoch_k':        'Stochastic %K',
    'stoch_d':        'Stochastic %D',
    'williams_r':     'Williams %R',
    'ema_cross_9_21': 'EMA Cross (9/21)',
    'returns':        'Daily Returns',
    'log_returns':    'Log Returns',
    'obv_ratio':      'OBV Ratio',
    'vol_ratio':      'Volume Ratio',
    'pct_52w_h':      '52-Week High Proximity',
    'pct_52w_l':      '52-Week Low Proximity',
    'skew_20':        'Return Skewness (20d)',
    'kurt_20':        'Return Kurtosis (20d)',
    'price_range':    'Daily Price Range',
    'gap':            'Opening Gap',
    'close_to_high':  'Close-to-High Ratio',
    'close_to_low':   'Close-to-Low Ratio',
    'roc_5':          'Rate of Change (5d)',
    'roc_10':         'Rate of Change (10d)',
    'roc_20':         'Rate of Change (20d)',
    'ret_lag_1':      'Return Lag 1d',
    'ret_lag_2':      'Return Lag 2d',
    'ret_lag_3':      'Return Lag 3d',
    'ret_lag_5':      'Return Lag 5d',
    'ret_lag_10':     'Return Lag 10d',
    'day_of_week':    'Day of Week',
    'month':          'Calendar Month',
    'ma_ratio_5':     'MA Ratio (5d)',
    'ma_ratio_10':    'MA Ratio (10d)',
    'ma_ratio_20':    'MA Ratio (20d)',
    'ma_ratio_50':    'MA Ratio (50d)',
    'ma_ratio_100':   'MA Ratio (100d)',
}


# ─────────────────────────────────────────────────────────────
# SHAP explanation — returns signed values for waterfall chart
# ─────────────────────────────────────────────────────────────
def _shap_top_features(model, X_sample: np.ndarray, feature_names: list, top_n: int = 8) -> list:
    """
    Returns a list of dicts sorted by |shap_value| desc:
      { name, label, value (signed), abs_value }
    Value sign: positive pushes toward BUY, negative toward SELL.
    Falls back to unsigned feature_importances_ if SHAP unavailable.
    """
    def _make_entry(name, signed_val, signed: bool = True):
        return {
            'name':      name,
            'label':     _FEATURE_LABELS.get(name, name.replace('_', ' ').title()),
            'value':     round(float(signed_val), 5),
            'abs_value': round(abs(float(signed_val)), 5),
            'is_signed': signed,
        }

    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X_sample)     # shape: (n_samples, n_features)
            # If single sample, squeeze to 1-D
            if vals.ndim == 2:
                signed = vals[0]
            else:
                signed = vals
            top_idx = np.argsort(np.abs(signed))[-top_n:][::-1]
            return [_make_entry(feature_names[i], signed[i], signed=True) for i in top_idx]
        except Exception as e:
            print(f"[SHAP] TreeExplainer failed: {e}")

    # Fallback: unsigned importances (no direction)
    try:
        imp     = model.feature_importances_
        top_idx = np.argsort(imp)[-top_n:][::-1]
        return [_make_entry(feature_names[i], imp[i], signed=False) for i in top_idx]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# StockPredictor
# ─────────────────────────────────────────────────────────────
class StockPredictor:
    def __init__(self):
        self.models       = {}
        self.meta_model   = Ridge(alpha=1.0)
        self.scaler       = RobustScaler()
        self.feature_cols = []
        self.ticker       = None
        self.period       = None
        self.start_date   = None
        self.end_date     = None
        self.train_meta   = {}
        self.shap_features = {}
        self.asset_type   = 'EQUITY'  # 'ETF' or 'EQUITY'

    # ── Build diverse model pool ────────────────────────────
    def _build_pool(self) -> dict:
        pool = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1),
            # Linear model → regularized to prevent collinearity runaway on correlated features
            'bayesian_ridge': BayesianRidge(alpha_1=1e-2, lambda_1=1e-2),
        }
        if HAS_XGB:
            pool['xgboost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0, n_jobs=-1)
        if HAS_LGB:
            pool['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, n_jobs=-1)
        return pool

    # ── Train ───────────────────────────────────────────────
    def train(self, ticker, period='2y', start_date=None, end_date=None, asset_type='EQUITY'):
        try:
            self.asset_type = asset_type
            is_etf = (asset_type == 'ETF')

            # ETFs need longer history to capture full market cycles
            default_period = '5y' if is_etf else period
            actual_period  = default_period if period == '2y' and is_etf else period

            df = get_history(ticker, period=actual_period,
                             start_date=start_date, end_date=end_date)

            # ETFs need >= 252 trading days (1 full year) to learn trend cycles
            min_days = 252 if is_etf else 120
            if df is None or df.empty or len(df) < min_days:
                return False, f"Insufficient historical data (need >={min_days} trading days for {'ETF' if is_etf else 'stock'})"

            # ── Select feature pipeline based on asset type ──
            feature_fn   = _create_etf_features if is_etf else _create_features
            feature_cols = ETF_FEATURE_COLS    if is_etf else FEATURE_COLS

            df_feat = feature_fn(df)
            avail   = [c for c in feature_cols if c in df_feat.columns]
            self.feature_cols = avail

            # 1. Temporal Leakage Prevention: dropna() without bfill()
            min_clean = 150 if is_etf else 80
            df_clean = df_feat[avail + ['Close']].dropna()
            if len(df_clean) < min_clean:
                return False, "Too many NaN values after feature engineering"

            # ETF: 30-day forward return; Stock: 5-day
            horizon = 30 if is_etf else 5
            n       = len(df_clean) - horizon
            X_raw   = df_clean[avail].values[:n]
            y       = np.array([
                (df_clean['Close'].iloc[i + horizon] - df_clean['Close'].iloc[i])
                / df_clean['Close'].iloc[i]
                for i in range(n)
            ])

            # 2. Temporal Leakage Prevention: Train/Test Split BEFORE Scaling
            split = int(len(X_raw) * 0.80)
            X_tr_raw, X_te_raw = X_raw[:split], X_raw[split:]
            y_tr, y_te = y[:split], y[split:]

            # Fit scaler strictly on training split
            self.scaler.fit(X_tr_raw)
            X_tr = self.scaler.transform(X_tr_raw)
            X_te = self.scaler.transform(X_te_raw)

            # 3. Genuine Out-Of-Fold (OOF) Stacking on X_tr
            pool = self._build_pool()
            n_splits = min(5, max(2, len(X_tr) // 20))
            kf = KFold(n_splits=n_splits, shuffle=False)
            oof_preds = {name: np.zeros(len(X_tr)) for name in pool.keys()}

            for tr_idx, val_idx in kf.split(X_tr):
                X_fold_tr, y_fold_tr = X_tr[tr_idx], y_tr[tr_idx]
                X_fold_val = X_tr[val_idx]
                for name, mdl in pool.items():
                    try:
                        fold_mdl = clone(mdl) if hasattr(mdl, 'get_params') else copy.deepcopy(mdl)
                        fold_mdl.fit(X_fold_tr, y_fold_tr)
                        oof_preds[name][val_idx] = fold_mdl.predict(X_fold_val)
                    except Exception as fold_err:
                        print(f"[ML OOF] {name} fold failed: {fold_err}")

            # 4. Train final base models on FULL X_tr for test evaluation and production inference
            self.models = {}
            te_outs = []
            valid_model_names = []

            for name, mdl in pool.items():
                try:
                    mdl.fit(X_tr, y_tr)
                    self.models[name] = mdl
                    te_outs.append(mdl.predict(X_te))
                    valid_model_names.append(name)
                except Exception as exc:
                    print(f"[ML] {name} skipped: {exc}")

            if not self.models:
                return False, "All models failed to train"

            # 5. Ridge Meta-Stacker fit on genuine Out-Of-Fold (OOF) predictions
            meta_tr = np.column_stack([oof_preds[name] for name in valid_model_names])
            meta_te = np.column_stack(te_outs)
            self.meta_model.fit(meta_tr, y_tr)
            meta_preds = self.meta_model.predict(meta_te)

            mae   = float(mean_absolute_error(y_te, meta_preds))
            rmse  = float(np.sqrt(mean_squared_error(y_te, meta_preds)))

            # 6. Walk-forward financial metrics (passes X_raw to eliminate scaler leakage)
            wf = _walk_forward_metrics(X_raw, y, dict(pool), Ridge(alpha=1.0))

            # ── SHAP on best tree model ──
            shap_model = (self.models.get('xgboost')
                          or self.models.get('random_forest')
                          or next(iter(self.models.values())))
            self.shap_features = _shap_top_features(shap_model, X_te[:100], avail)

            self.ticker, self.period = ticker, actual_period
            self.start_date, self.end_date = start_date, end_date
            self.train_meta = {
                'mae':  round(mae, 5),
                'rmse': round(rmse, 5),
                'models_used': list(self.models.keys()),
                'n_train': len(X_tr),
                'n_test':  len(X_te),
                'asset_type': asset_type,
                'horizon_days': horizon,
                **wf,
            }
            return True, self.train_meta

        except Exception as exc:
            return False, f"Training error: {exc}"

    # ── Predict ─────────────────────────────────────────────
    def predict(self, ticker, news_sentiment: float = 0.0):
        """
        news_sentiment: float in [-1, +1]. Passed from news module.
        For ETFs, news_sentiment is always 0.0 (no company-specific news).
        ETFs use ACCUMULATE/AVOID signal language and 30-day horizon.
        """
        try:
            if not self.models:
                return None, "Models not trained"

            is_etf = (self.asset_type == 'ETF')

            # ETFs use 1yr of recent data to capture trend context
            recent_period = '1y' if is_etf else '6mo'
            df = get_history(ticker, period=recent_period)
            if df is None or df.empty:
                return None, "No recent price data"

            # ── Synchronize with live quote if available ──
            live_price = None
            try:
                live_q = get_quote(ticker)
                if live_q and live_q.get('price') and float(live_q['price']) > 0:
                    live_price = float(live_q['price'])
                    # If live quote price differs from last candle by >0.2%, sync latest bar
                    if abs(float(df['Close'].iloc[-1]) - live_price) / float(df['Close'].iloc[-1]) > 0.002:
                        now_dt = pd.Timestamp.now(tz='Asia/Kolkata')
                        last_dt = df.index[-1]
                        if now_dt.date() > last_dt.date():
                            today_bar = pd.DataFrame({
                                'Open':   [live_q.get('prevClose') or live_price],
                                'High':   [max(live_q.get('dayHigh') or live_price, live_price)],
                                'Low':    [min(live_q.get('dayLow') or live_price, live_price)],
                                'Close':  [live_price],
                                'Volume': [live_q.get('volume') or df['Volume'].iloc[-1]],
                            }, index=[now_dt.floor('D')])
                            df = pd.concat([df, today_bar])
                        else:
                            df.loc[df.index[-1], 'Close'] = live_price
                            if live_q.get('dayHigh'):
                                df.loc[df.index[-1], 'High'] = max(df['High'].iloc[-1], float(live_q['dayHigh']))
                            if live_q.get('dayLow'):
                                df.loc[df.index[-1], 'Low'] = min(df['Low'].iloc[-1], float(live_q['dayLow']))
                            if live_q.get('volume'):
                                df.loc[df.index[-1], 'Volume'] = int(live_q['volume'])
            except Exception as e:
                print(f"[ML] Live quote sync skipped: {e}")

            # Use the matching feature pipeline
            feature_fn = _create_etf_features if is_etf else _create_features
            df_feat    = feature_fn(df)
            avail      = [c for c in self.feature_cols if c in df_feat.columns]
            df_clean   = df_feat[avail].dropna()

            min_recent = 30 if is_etf else 15
            if len(df_clean) < min_recent:
                return None, "Insufficient recent data"

            X_latest = self.scaler.transform(df_clean.tail(1).values)

            # ── Base predictions & Model Governance (SR 11-7) ──
            current_price = live_price if (live_price is not None and live_price > 0) else float(df['Close'].iloc[-1])

            # Centralized governance policy bounds
            policy_key = "etf_30d" if is_etf else "equity_5d"
            policy = MODEL_GOVERNANCE_THRESHOLDS[policy_key]
            lower_bound = policy["lower_return"]
            upper_bound = policy["upper_return"]

            base_preds = []
            eligible_base_preds = []
            model_telemetry = {}

            for name, mdl in self.models.items():
                try:
                    p_raw = float(mdl.predict(X_latest)[0])
                    is_outlier = (p_raw < lower_bound) or (p_raw > upper_bound)
                    p_val = float(np.clip(p_raw, lower_bound, upper_bound))
                    is_clipped = (abs(p_raw - p_val) > 1e-4)

                    raw_price = round(current_price * (1.0 + p_raw), 2)
                    val_price = round(max(0.01, current_price * (1.0 + p_val)), 2)

                    status = "DEGRADED_OUTLIER" if is_outlier else "OK"
                    is_eligible = not is_outlier

                    if is_eligible:
                        eligible_base_preds.append(p_val)
                    base_preds.append(p_val)

                    model_telemetry[name] = {
                        'raw_return': round(p_raw * 100, 2),
                        'validated_return': round(p_val * 100, 2),
                        'raw_price': raw_price,
                        'validated_price': val_price,
                        'predicted_return': round(p_val * 100, 2),
                        'predicted_price': val_price,
                        'is_clipped': is_clipped,
                        'status': status,
                        'included_in_ensemble': is_eligible,
                        'thresholds_applied': {
                            'policy_key': policy_key,
                            'lower_pct': round(lower_bound * 100, 1),
                            'upper_pct': round(upper_bound * 100, 1),
                        },
                        'outlier_reason': (
                            f"Raw return ({round(p_raw * 100, 2)}%) breached policy bounds "
                            f"[{round(lower_bound * 100, 1)}%, {round(upper_bound * 100, 1)}%]"
                            if is_outlier else None
                        ),
                    }
                except Exception as exc:
                    print(f"[ML] Base model {name} failed: {exc}")

            if not base_preds:
                return None, "All model predictions failed"

            # ── Meta prediction with Outlier Exclusion ──
            if len(eligible_base_preds) >= 2:
                if len(eligible_base_preds) == len(self.models):
                    meta_in = np.array(base_preds).reshape(1, -1)
                    predicted_return = float(self.meta_model.predict(meta_in)[0])
                else:
                    # An outlier was excluded from stacking: use robust median of eligible models
                    predicted_return = float(np.median(eligible_base_preds))
            elif eligible_base_preds:
                predicted_return = float(eligible_base_preds[0])
            else:
                predicted_return = float(np.median(base_preds))

            # ETF 30d returns can be larger; widen clip window
            clip_max = 0.35 if is_etf else 0.20
            predicted_return = float(np.clip(predicted_return, -clip_max, clip_max))

            # ── Sentiment fusion ──
            # ETFs: NO news sentiment — irrelevant for index/commodity ETFs
            # Stocks: 80% ML + 20% news nudge (original logic)
            if is_etf:
                fused_return    = predicted_return
                news_sentiment  = 0.0   # zero out for clean reporting
            else:
                sentiment_nudge = np.clip(news_sentiment, -1.0, 1.0) * 0.004
                fused_return    = predicted_return * 0.80 + sentiment_nudge * 0.20
                fused_return    = float(np.clip(fused_return, -0.20, 0.20))

            # ── Confidence from model disagreement ──
            model_std = float(np.std(base_preds))
            rf_std    = 0.0
            if 'random_forest' in self.models:
                tree_preds = [
                    t.predict(X_latest)[0]
                    for t in self.models['random_forest'].estimators_
                ]
                rf_std = float(np.std(tree_preds))
            combined_std = model_std * 0.6 + rf_std * 0.4
            confidence   = float(np.clip(90.0 - combined_std * 2000.0, 15.0, 95.0))

            # ── Regime ──
            regime = _detect_regime(df_feat)

            # ── Volatility Forecast (GARCH) ──
            garch_vol = _forecast_garch_volatility(df_feat)

            # ── Stability ──
            if combined_std > 0.025:
                stability  = "HIGH UNCERTAINTY"
                confidence = max(10.0, confidence - 20.0)
            elif combined_std > 0.012:
                stability = "MODERATE VOLATILITY"
            else:
                stability = "STABLE CONSENSUS"

            # ── Signal thresholds & labels — diverge by asset type ──
            if is_etf:
                # ETF: larger threshold (slower moves), long-term language
                base_thresh = 0.05 if regime == "HIGH_VOLATILITY" else 0.03
                if   fused_return >  base_thresh * 1.5: signal = "STRONG ACCUMULATE"
                elif fused_return >  base_thresh:        signal = "ACCUMULATE"
                elif fused_return < -base_thresh * 1.5:  signal = "AVOID / REDUCE"
                elif fused_return < -base_thresh:        signal = "CAUTION"
                else:                                    signal = "HOLD / SIP"
                horizon_days = 30
            else:
                # Stock: original short-term BUY/SELL signals
                base_thresh = 0.030 if regime == "HIGH_VOLATILITY" else 0.018
                if (news_sentiment > 0.3 and fused_return > 0) or \
                   (news_sentiment < -0.3 and fused_return < 0):
                    base_thresh *= 0.85
                if   fused_return >  base_thresh * 1.5: signal = "STRONG BUY"
                elif fused_return >  base_thresh:        signal = "BUY"
                elif fused_return < -base_thresh * 1.5:  signal = "STRONG SELL"
                elif fused_return < -base_thresh:        signal = "SELL"
                else:                                    signal = "HOLD"
                horizon_days = 5

            signal_strength = float(np.clip(abs(fused_return) * 1000, 0, 100))

            # ── Prices ──
            current_price   = live_price if (live_price is not None and live_price > 0) else float(df['Close'].iloc[-1])
            predicted_price = round(max(0.01, current_price * (1 + fused_return)), 2)

            # ── ATR-based risk/reward ──
            atr_col = 'atr_ratio' if is_etf else 'atr'
            if atr_col in df_feat.columns:
                atr_raw = float(df_feat[atr_col].iloc[-1])
                atr = atr_raw * current_price if is_etf else atr_raw
            else:
                atr = current_price * 0.02
            expected_gain = abs(fused_return) * current_price
            risk_reward   = round(expected_gain / (atr + 1e-9), 2)

            # ── SHAP labels: use ETF-specific map for ETF tickers ──
            label_map = _ETF_FEATURE_LABELS if is_etf else _FEATURE_LABELS

            # ── SHAP top features ──
            live_shap = []
            if self.models:
                best = (self.models.get('xgboost')
                        or self.models.get('random_forest')
                        or next(iter(self.models.values())))
                # Override labels in _shap_top_features via monkey-patch not ideal;
                # instead inject labels post-hoc
                live_shap_raw = _shap_top_features(best, X_latest, avail)
                live_shap = [
                    {**f, 'label': label_map.get(f['name'], f['label'])}
                    for f in live_shap_raw
                ]

            top_features = live_shap if live_shap else [
                {**f, 'label': label_map.get(f['name'], f['label'])}
                for f in self.shap_features
            ]

            return {
                'predicted_return':       round(fused_return * 100, 2),
                'predicted_price':        predicted_price,
                'current_price':          round(current_price, 2),
                'signal':                 signal,
                'signal_strength':        round(signal_strength, 1),
                'confidence':             round(confidence, 1),
                'stability':              stability,
                'regime':                 regime,
                'garch_volatility':       garch_vol.get('annualized_pct') if isinstance(garch_vol, dict) else garch_vol,
                'garch_spec':             garch_vol if isinstance(garch_vol, dict) else None,
                'asset_type':             self.asset_type,
                # Walk-forward financial metrics
                'direction_accuracy':     self.train_meta.get('direction_accuracy', 50.0),
                'profit_factor':          self.train_meta.get('profit_factor', 1.0),
                'hit_rate':               self.train_meta.get('hit_rate', 50.0),
                'avg_win_pct':            self.train_meta.get('avg_win_pct', 0.0),
                'avg_loss_pct':           self.train_meta.get('avg_loss_pct', 0.0),
                'ensemble_vs_xgb_delta':  self.train_meta.get('ensemble_vs_xgb_delta', 0.0),
                # Explainability
                'top_features':           top_features,
                'shap_powered':           HAS_SHAP,
                # Sentiment fusion
                'news_sentiment_used':    round(news_sentiment, 3),
                'ml_raw_return':          round(predicted_return * 100, 2),
                # Risk
                'risk_reward_ratio':      risk_reward,
                'prediction_horizon_days': horizon_days,
                'horizon_days':           horizon_days,
                'horizon_label':          f"{horizon_days}-Day Outlook",
                'models_used':            list(self.models.keys()),
                'timestamp':              datetime.now().isoformat(),
                'models':                 model_telemetry,
                # ── Multi-Tier Status Governance ──
                'data_status': {
                    'live_quote':         'OK' if (live_price is not None and live_price > 0) else 'UNAVAILABLE',
                    'historical_candles': 'OK' if len(df_clean) >= 30 else 'SHORT_HISTORY',
                    'corporate_actions':  'SPLIT_ADJUSTED',
                },
                'model_status': {
                    'ensemble_health':    'OK' if not [k for k, v in model_telemetry.items() if not v['included_in_ensemble']] else ('DEGRADED_QUARANTINE' if eligible_base_preds else 'CRITICAL'),
                    'active_models_count': len(eligible_base_preds),
                    'total_models_count':  len(self.models),
                    'quarantined_models':  [k for k, v in model_telemetry.items() if not v['included_in_ensemble']],
                    'governance_policy':   policy,
                },
                'valuation_status': {
                    'methodology':        'ETF_30D_TACTICAL_ENSEMBLE' if is_etf else 'EQUITY_5D_TACTICAL_ENSEMBLE',
                    'horizon_days':       horizon_days,
                    'status':             'OK',
                },
            }, None

        except Exception as exc:
            return None, f"Prediction error: {exc}"


# ─────────────────────────────────────────────────────────────
# Public API — per-ticker LRU cache
# asset_type is included in cache key so ETF/EQUITY are cached separately
# ─────────────────────────────────────────────────────────────
def _cache_key(ticker, period, start_date, end_date, asset_type='EQUITY'):
    return f"{ticker}|{period}|{start_date}|{end_date}|{asset_type}"


def get_ml_prediction(ticker, period='2y', start_date=None, end_date=None,
                      news_sentiment: float = 0.0, asset_type: str = 'EQUITY'):
    global _MODEL_CACHE
    key = _cache_key(ticker, period, start_date, end_date, asset_type)

    if key not in _MODEL_CACHE:
        predictor = StockPredictor()
        ok, result = predictor.train(
            ticker, period=period,
            start_date=start_date, end_date=end_date,
            asset_type=asset_type
        )
        if not ok:
            return None, result
        if len(_MODEL_CACHE) >= _CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
        _MODEL_CACHE[key] = predictor
    else:
        _MODEL_CACHE.move_to_end(key)

    return _MODEL_CACHE[key].predict(ticker, news_sentiment=news_sentiment)


def retrain_model(ticker, period='2y', start_date=None, end_date=None,
                  asset_type: str = 'EQUITY'):
    key = _cache_key(ticker, period, start_date, end_date, asset_type)
    _MODEL_CACHE.pop(key, None)
    return get_ml_prediction(ticker, period=period,
                             start_date=start_date, end_date=end_date,
                             asset_type=asset_type)