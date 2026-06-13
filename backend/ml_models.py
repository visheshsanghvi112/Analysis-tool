# ============================================================
# ML Models for Stock Prediction — by Vishesh Sanghvi
# Upgraded: diverse 6-model stacked ensemble, 40+ features,
# walk-forward financial metrics, SHAP explanations,
# news-sentiment signal fusion, benchmark comparison,
# per-ticker LRU cache, full fallback logic
# ============================================================

import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from yf_client import get_history

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
        # Fallback to simple rule-based classification if HMM fails
        try:
            closes = df['Close'].tail(60)
            vol_now = df['returns'].tail(20).std()
            vol_long = df['returns'].tail(60).std()
            if vol_now > 1.8 * vol_long: return "HIGH_VOLATILITY"
            price = closes.iloc[-1]
            ma20 = closes.rolling(20).mean().iloc[-1]
            ma50 = closes.rolling(min(50, len(closes))).mean().iloc[-1]
            if price > ma20 > ma50: return "TRENDING_UP"
            if price < ma20 < ma50: return "TRENDING_DOWN"
            return "SIDEWAYS"
        except:
            return "UNKNOWN"


def _forecast_garch_volatility(df: pd.DataFrame) -> float:
    """
    Fits a GARCH(1,1) volatility forecasting model to returns
    and predicts the average annualized volatility for the next 5 days.
    """
    try:
        from arch import arch_model
        # Scaled returns by 100 for numerical stability
        returns = df['returns'].dropna().values * 100
        if len(returns) < 60:
            return None
        
        garch = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        res = garch.fit(update_freq=0, disp='off')
        
        forecast = res.forecast(horizon=5)
        # Extract the average variance forecast for the 5-day horizon and annualize it
        mean_variance = forecast.variance.iloc[-1].mean()
        annualized_vol = float(np.sqrt(mean_variance) * np.sqrt(252))
        return round(annualized_vol, 2)
    except Exception as e:
        print(f"[ML] GARCH volatility forecasting failed: {e}")
        return None



# ─────────────────────────────────────────────────────────────
# Walk-forward financial evaluation
# ─────────────────────────────────────────────────────────────
def _walk_forward_metrics(X: np.ndarray, y: np.ndarray,
                          models: dict, meta: Ridge,
                          n_folds: int = 5) -> dict:
    """
    Returns direction_accuracy, profit_factor, hit_rate,
    avg_win, avg_loss from expanding walk-forward folds.
    Also benchmarks XGBoost-alone vs full ensemble.
    """
    fold_size = max(1, len(X) // (n_folds + 1))
    all_y, all_blend, all_xgb_only = [], [], []

    xgb_model = models.get('xgboost') or models.get('random_forest')

    for i in range(1, n_folds + 1):
        tr_end = fold_size * i
        te_end = min(tr_end + fold_size, len(X))
        if te_end <= tr_end + 5:
            continue
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_te, y_te = X[tr_end:te_end], y[tr_end:te_end]

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
        try:
            meta.fit(np.column_stack([mdl.predict(X_tr) for mdl in models.values()
                                      if hasattr(mdl, 'predict')]), y_tr)
            blend = meta.predict(meta_in)
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

    wins  = y_arr[correct]
    loses = y_arr[~correct]
    gross_win  = float(wins.sum())  if len(wins)  else 0.0
    gross_loss = float(loses.sum()) if len(loses) else 0.0
    profit_fac = (gross_win / abs(gross_loss)) if gross_loss < 0 else float('inf')
    profit_fac = min(profit_fac, 9.99)

    avg_win  = float(wins.mean())  if len(wins)  else 0.0
    avg_loss = float(loses.mean()) if len(loses) else 0.0

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
# SHAP explanation (top-5 features)
# ─────────────────────────────────────────────────────────────
def _shap_top_features(model, X_sample: np.ndarray, feature_names: list) -> dict:
    if not HAS_SHAP:
        return {}
    try:
        explainer = shap.TreeExplainer(model)
        vals      = explainer.shap_values(X_sample)
        mean_abs  = np.abs(vals).mean(axis=0)
        top_idx   = np.argsort(mean_abs)[-5:][::-1]
        return {feature_names[i]: round(float(mean_abs[i]), 4) for i in top_idx}
    except Exception:
        # Fallback: raw importance if tree model
        try:
            imp     = model.feature_importances_
            top_idx = np.argsort(imp)[-5:][::-1]
            return {feature_names[i]: round(float(imp[i]), 4) for i in top_idx}
        except Exception:
            return {}


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
            # Linear model → breaks tree correlation, adds diversity
            'bayesian_ridge': BayesianRidge(),
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
    def train(self, ticker, period='2y', start_date=None, end_date=None):
        try:
            df = get_history(ticker, period=period,
                             start_date=start_date, end_date=end_date)
            if df is None or df.empty or len(df) < 120:
                return False, "Insufficient historical data (need ≥120 trading days)"

            df_feat = _create_features(df)
            avail   = [c for c in FEATURE_COLS if c in df_feat.columns]
            self.feature_cols = avail

            df_clean = df_feat[avail + ['Close']].ffill().bfill().dropna()
            if len(df_clean) < 80:
                return False, "Too many NaN values after feature engineering"

            horizon = 5
            n       = len(df_clean) - horizon
            X_raw   = df_clean[avail].values[:n]
            y       = np.array([
                (df_clean['Close'].iloc[i + horizon] - df_clean['Close'].iloc[i])
                / df_clean['Close'].iloc[i]
                for i in range(n)
            ])

            X = self.scaler.fit_transform(X_raw)
            split = int(len(X) * 0.80)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

            # ── Train each base model independently ──
            pool = self._build_pool()
            self.models = {}
            tr_outs, te_outs = [], []

            for name, mdl in pool.items():
                try:
                    mdl.fit(X_tr, y_tr)
                    tr_outs.append(mdl.predict(X_tr))
                    te_outs.append(mdl.predict(X_te))
                    self.models[name] = mdl
                except Exception as exc:
                    print(f"[ML] {name} skipped: {exc}")

            if not self.models:
                return False, "All models failed to train"

            # ── Ridge meta-stacker ──
            meta_tr = np.column_stack(tr_outs)
            meta_te = np.column_stack(te_outs)
            self.meta_model.fit(meta_tr, y_tr)
            meta_preds = self.meta_model.predict(meta_te)

            mae   = float(mean_absolute_error(y_te, meta_preds))
            rmse  = float(np.sqrt(mean_squared_error(y_te, meta_preds)))

            # ── Walk-forward financial metrics ──
            wf = _walk_forward_metrics(X, y, dict(pool), Ridge(alpha=1.0))

            # ── SHAP on best tree model ──
            shap_model = (self.models.get('xgboost')
                          or self.models.get('random_forest')
                          or next(iter(self.models.values())))
            self.shap_features = _shap_top_features(shap_model, X_tr[:200], avail)

            self.ticker, self.period = ticker, period
            self.start_date, self.end_date = start_date, end_date
            self.train_meta = {
                'mae':  round(mae, 5),
                'rmse': round(rmse, 5),
                'models_used': list(self.models.keys()),
                'n_train': len(X_tr),
                'n_test':  len(X_te),
                **wf,
            }
            return True, self.train_meta

        except Exception as exc:
            return False, f"Training error: {exc}"

    # ── Predict ─────────────────────────────────────────────
    def predict(self, ticker, news_sentiment: float = 0.0):
        """
        news_sentiment: float in [-1, +1]. Passed from news module.
        Adjusts signal thresholds based on fundamental signal direction.
        """
        try:
            if not self.models:
                return None, "Models not trained"

            df = get_history(ticker, period='6mo')
            if df is None or df.empty:
                return None, "No recent price data"

            df_feat  = _create_features(df)
            avail    = [c for c in self.feature_cols if c in df_feat.columns]
            df_clean = df_feat[avail].ffill().bfill().dropna()

            if len(df_clean) < 15:
                return None, "Insufficient recent data"

            X_latest = self.scaler.transform(df_clean.tail(1).values)

            # ── Base predictions ──
            base_preds  = []
            model_outs  = {}
            for name, mdl in self.models.items():
                try:
                    p = float(mdl.predict(X_latest)[0])
                    base_preds.append(p)
                    model_outs[name] = p
                except Exception:
                    pass

            if not base_preds:
                return None, "All model predictions failed"

            # ── Meta prediction ──
            meta_in        = np.array(base_preds).reshape(1, -1)
            predicted_return = float(self.meta_model.predict(meta_in)[0])
            predicted_return = float(np.clip(predicted_return, -0.20, 0.20))

            # ── News sentiment fusion ──
            # Weight: 80% ML, 20% sentiment direction
            # Only nudges; doesn't override strong ML signal
            sentiment_nudge  = np.clip(news_sentiment, -1.0, 1.0) * 0.004
            fused_return     = predicted_return * 0.80 + sentiment_nudge * 0.20
            fused_return     = float(np.clip(fused_return, -0.20, 0.20))

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

            # ── Regime-aware + sentiment-aware thresholds ──
            # HMM Regimes: LOW_VOLATILITY, MEDIUM_VOLATILITY, HIGH_VOLATILITY
            base_thresh = 0.030 if regime == "HIGH_VOLATILITY" else 0.018
            # If sentiment strongly agrees with ML direction, lower bar slightly
            if (news_sentiment > 0.3 and fused_return > 0) or \
               (news_sentiment < -0.3 and fused_return < 0):
                base_thresh *= 0.85

            if   fused_return >  base_thresh * 1.5: signal = "STRONG BUY"
            elif fused_return >  base_thresh:        signal = "BUY"
            elif fused_return < -base_thresh * 1.5:  signal = "STRONG SELL"
            elif fused_return < -base_thresh:        signal = "SELL"
            else:                                    signal = "HOLD"

            signal_strength = float(np.clip(abs(fused_return) * 1000, 0, 100))

            # ── Prices ──
            current_price   = float(df['Close'].iloc[-1])
            predicted_price = round(current_price * (1 + fused_return), 2)

            # ── ATR-based risk/reward ──
            atr = float(df_feat['atr'].iloc[-1]) if 'atr' in df_feat.columns \
                  else current_price * 0.02
            expected_gain = abs(fused_return) * current_price
            risk_reward   = round(expected_gain / (atr + 1e-9), 2)

            # ── SHAP top features (live explanation on latest point) ──
            live_shap = {}
            if HAS_SHAP and self.models:
                best = (self.models.get('xgboost')
                        or self.models.get('random_forest')
                        or next(iter(self.models.values())))
                live_shap = _shap_top_features(best, X_latest, avail)

            top_features = live_shap or self.shap_features

            return {
                'predicted_return':       round(fused_return * 100, 2),
                'predicted_price':        predicted_price,
                'current_price':          round(current_price, 2),
                'signal':                 signal,
                'signal_strength':        round(signal_strength, 1),
                'confidence':             round(confidence, 1),
                'stability':              stability,
                'regime':                 regime,
                'garch_volatility':       garch_vol,
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
                'prediction_horizon_days': 5,
                'models_used':            list(self.models.keys()),
                'timestamp':              datetime.now().isoformat(),
                'models': {
                    name: {
                        'predicted_return': round(p * 100, 2),
                        'predicted_price':  round(current_price * (1 + p), 2),
                    }
                    for name, p in model_outs.items()
                },
            }, None

        except Exception as exc:
            return None, f"Prediction error: {exc}"


# ─────────────────────────────────────────────────────────────
# Public API — per-ticker LRU cache
# ─────────────────────────────────────────────────────────────
def _cache_key(ticker, period, start_date, end_date):
    return f"{ticker}|{period}|{start_date}|{end_date}"


def get_ml_prediction(ticker, period='2y', start_date=None, end_date=None,
                      news_sentiment: float = 0.0):
    global _MODEL_CACHE
    key = _cache_key(ticker, period, start_date, end_date)

    if key not in _MODEL_CACHE:
        predictor = StockPredictor()
        ok, result = predictor.train(
            ticker, period=period,
            start_date=start_date, end_date=end_date
        )
        if not ok:
            return None, result
        if len(_MODEL_CACHE) >= _CACHE_MAX:
            _MODEL_CACHE.popitem(last=False)
        _MODEL_CACHE[key] = predictor
    else:
        _MODEL_CACHE.move_to_end(key)

    return _MODEL_CACHE[key].predict(ticker, news_sentiment=news_sentiment)


def retrain_model(ticker, period='2y', start_date=None, end_date=None):
    key = _cache_key(ticker, period, start_date, end_date)
    _MODEL_CACHE.pop(key, None)
    return get_ml_prediction(ticker, period=period,
                             start_date=start_date, end_date=end_date)