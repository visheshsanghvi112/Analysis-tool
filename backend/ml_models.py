# ============================================================
# ML Models for Stock Prediction — by Vishesh Sanghvi
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = []
        
    def create_features(self, df):
        """Create technical indicators as features for ML model"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_change'] = df['Close'] - df['Open']
        df['price_range'] = df['High'] - df['Low']
        df['gap'] = df['Open'] - df['Close'].shift(1)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'MA_{period}'] = df['Close'].rolling(period).mean()
            df[f'MA_ratio_{period}'] = df['Close'] / df[f'MA_{period}']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13).mean()
        avg_loss = loss.ewm(com=13).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_price_trend'] = df['Volume'] * df['returns']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(60).mean()
        
        # Market microstructure
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_to_high'] = df['Close'] / df['High']
        df['close_to_low'] = df['Close'] / df['Low']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        return df
    
    def prepare_sequences(self, df, sequence_length=10, prediction_horizon=5):
        """Create sequences for time series prediction"""
        features = [
            'returns', 'log_returns', 'price_change', 'price_range', 'gap',
            'MA_ratio_5', 'MA_ratio_10', 'MA_ratio_20', 'MA_ratio_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'BB_position',
            'volume_ratio', 'volume_price_trend', 'volatility', 'volatility_ratio',
            'high_low_ratio', 'close_to_high', 'close_to_low',
            'returns_lag_1', 'returns_lag_2', 'returns_lag_3', 'returns_lag_5'
        ]
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        self.feature_columns = available_features
        
        # Clean data
        df_clean = df[available_features + ['Close']].dropna()
        
        if len(df_clean) < sequence_length + prediction_horizon:
            return None, None, None
        
        # Normalize features
        feature_data = self.scaler.fit_transform(df_clean[available_features])
        
        X, y = [], []
        for i in range(sequence_length, len(df_clean) - prediction_horizon + 1):
            X.append(feature_data[i-sequence_length:i])
            # Predict future returns
            future_price = df_clean['Close'].iloc[i + prediction_horizon - 1]
            current_price = df_clean['Close'].iloc[i - 1]
            future_return = (future_price - current_price) / current_price
            y.append(future_return)
        
        return np.array(X), np.array(y), df_clean
    
    def train_model(self, ticker, days_back=730):
        """Train prediction model on historical data"""
        try:
            # Fetch extended historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty or len(df) < 100:
                return False, "Insufficient data for training"
            
            # Create features
            df_features = self.create_features(df)
            
            # Prepare sequences
            X, y, df_clean = self.prepare_sequences(df_features)
            
            if X is None:
                return False, "Could not create training sequences"
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Flatten sequences for Random Forest
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_flat, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_flat)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Calculate accuracy (% predictions within 2% of actual)
            accuracy = np.mean(np.abs(y_pred - y_test) < 0.02) * 100
            
            return True, {
                'mae': mae,
                'rmse': rmse, 
                'accuracy': accuracy,
                'samples_trained': len(X_train),
                'samples_tested': len(X_test)
            }
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def predict(self, ticker, prediction_horizon=5):
        """Make prediction for future price movement"""
        try:
            if self.model is None:
                return None, "Model not trained"
            
            # Fetch recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return None, "No recent data available"
            
            # Create features
            df_features = self.create_features(df)
            
            # Get latest sequence
            available_features = [f for f in self.feature_columns if f in df_features.columns]
            df_clean = df_features[available_features].dropna()
            
            if len(df_clean) < 10:
                return None, "Insufficient recent data"
            
            # Normalize latest data
            latest_data = self.scaler.transform(df_clean.tail(10))
            
            # Reshape for prediction
            X_pred = latest_data.reshape(1, -1)
            
            # Make prediction
            predicted_return = self.model.predict(X_pred)[0]
            
            # Calculate prediction confidence based on feature importance
            feature_importance = self.model.feature_importances_
            confidence = min(90, max(30, 70 + (np.std(feature_importance) * 100)))
            
            # Convert return to price target
            current_price = df['Close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_return)
            
            # Generate signal
            if predicted_return > 0.02:  # > 2% expected return
                signal = "STRONG BUY"
                signal_strength = min(100, abs(predicted_return) * 1000)
            elif predicted_return > 0.005:  # > 0.5% expected return
                signal = "BUY"
                signal_strength = min(100, abs(predicted_return) * 1000)
            elif predicted_return < -0.02:  # < -2% expected return
                signal = "STRONG SELL"
                signal_strength = min(100, abs(predicted_return) * 1000)
            elif predicted_return < -0.005:  # < -0.5% expected return
                signal = "SELL"
                signal_strength = min(100, abs(predicted_return) * 1000)
            else:
                signal = "HOLD"
                signal_strength = 50
            
            return {
                'predicted_return': round(predicted_return * 100, 2),  # Convert to percentage
                'predicted_price': round(predicted_price, 2),
                'current_price': round(current_price, 2),
                'signal': signal,
                'signal_strength': round(signal_strength, 1),
                'confidence': round(confidence, 1),
                'prediction_horizon_days': prediction_horizon,
                'timestamp': datetime.now().isoformat()
            }, None
            
        except Exception as e:
            return None, f"Prediction failed: {str(e)}"

# Global predictor instance
stock_predictor = StockPredictor()

def get_ml_prediction(ticker):
    """Get ML prediction for a stock ticker"""
    global stock_predictor
    
    # Train model if not already trained
    if stock_predictor.model is None:
        success, result = stock_predictor.train_model(ticker)
        if not success:
            return None, result
    
    # Make prediction
    prediction, error = stock_predictor.predict(ticker)
    return prediction, error

def retrain_model(ticker):
    """Force retrain the model with latest data"""
    global stock_predictor
    stock_predictor.model = None  # Reset model
    return stock_predictor.train_model(ticker)