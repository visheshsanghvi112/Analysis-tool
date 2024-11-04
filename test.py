# Importing necessary libraries
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # Importing make_subplots
from textblob import TextBlob
import requests

# Function to calculate support and resistance levels
def calculate_support_resistance(data):
    resistance = data['Close'].max()
    support = data['Close'].min()
    return support, resistance

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_levels(data):
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    diff = max_price - min_price
    first_level = max_price - 0.236 * diff
    second_level = max_price - 0.382 * diff
    third_level = max_price - 0.618 * diff
    return first_level, second_level, third_level

# Function to calculate correlation heatmap
def correlation_heatmap(data):
    correlation = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title("Correlation Heatmap of Indicators")
    plt.show()

# Function to fetch news sentiment
def fetch_news_sentiment(ticker):
    # Example API call (you need to use a valid news API)
    # response = requests.get(f"https://newsapi.org/v2/everything?q={ticker}&apiKey=YOUR_API_KEY")
    # articles = response.json()['articles']
    
    # For this example, let's use dummy articles
    articles = [
        {"title": "HDFC Bank reports strong quarterly results", "description": "HDFC Bank's Q3 profits exceeded expectations."},
        {"title": "Market reacts positively to HDFC Bank's strategy", "description": "Investors show confidence in HDFC Bank's new initiatives."},
        {"title": "Concerns about HDFC Bank's loan growth", "description": "Analysts express caution regarding loan growth in the next quarter."},
    ]
    
    sentiment_score = 0
    subjectivity_score = 0
    for article in articles:
        analysis = TextBlob(article['title'] + " " + article['description'])
        sentiment_score += analysis.sentiment.polarity
        subjectivity_score += analysis.sentiment.subjectivity
    
    avg_sentiment = sentiment_score / len(articles)
    avg_subjectivity = subjectivity_score / len(articles)
    return avg_sentiment, avg_subjectivity

# Function to perform an in-depth stock analysis
def detailed_stock_analysis(tickers, start_date=None, end_date=None):
    if start_date is None or end_date is None:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)  # Default to the past 3 months

    for ticker in tickers:
        try:
            # Fetch stock data from Yahoo Finance
            stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

            # Technical indicators
            stock_data['20 Day MA'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['50 Day MA'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['100 Day MA'] = stock_data['Close'].rolling(window=100).mean()

            # Bollinger Bands
            stock_data['20 Day STD'] = stock_data['Close'].rolling(window=20).std()
            stock_data['Upper Band'] = stock_data['20 Day MA'] + (stock_data['20 Day STD'] * 2)
            stock_data['Lower Band'] = stock_data['20 Day MA'] - (stock_data['20 Day STD'] * 2)

            # Relative Strength Index (RSI)
            delta = stock_data['Close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            stock_data['RSI'] = 100 - (100 / (1 + rs))

            # Moving Average Convergence Divergence (MACD)
            stock_data['12 EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
            stock_data['26 EMA'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['MACD'] = stock_data['12 EMA'] - stock_data['26 EMA']
            stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

            # Average True Range (ATR)
            high_low = stock_data['High'] - stock_data['Low']
            high_close = np.abs(stock_data['High'] - stock_data['Close'].shift())
            low_close = np.abs(stock_data['Low'] - stock_data['Close'].shift())
            true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
            stock_data['ATR'] = true_range.rolling(window=14).mean()

            # Volume Weighted Average Price (VWAP)
            stock_data['VWAP'] = (stock_data['Volume'] * (stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 3).cumsum() / stock_data['Volume'].cumsum()

            # On-Balance Volume (OBV)
            stock_data['OBV'] = (np.sign(stock_data['Close'].diff()) * stock_data['Volume']).fillna(0).cumsum()

            # Average Directional Index (ADX)
            def calculate_adx(df, n=14):
                df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
                df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
                df['+DI'] = 100 * (df['+DM'].rolling(n).sum() / df['ATR'])
                df['-DI'] = 100 * (df['-DM'].rolling(n).sum() / df['ATR'])
                df['DX'] = (np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
                df['ADX'] = df['DX'].rolling(n).mean()
                return df

            stock_data = calculate_adx(stock_data)

            # Calculate support and resistance levels
            support, resistance = calculate_support_resistance(stock_data)
            print(f"Ticker: {ticker} - Support Level: {support:.2f}, Resistance Level: {resistance:.2f}")

            # Calculate Fibonacci levels
            fib_levels = calculate_fibonacci_levels(stock_data)
            print(f"Fibonacci Levels for {ticker}: {fib_levels}")

            # Fetch news sentiment
            avg_sentiment, avg_subjectivity = fetch_news_sentiment(ticker)
            sentiment_label = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
            print(f"Average News Sentiment for {ticker}: {avg_sentiment:.2f} - {sentiment_label}, Subjectivity: {avg_subjectivity:.2f}")

            # Plotting with Plotly for interactive visualization
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

            # Candlestick chart
            fig.add_trace(go.Candlestick(x=stock_data.index,
                                          open=stock_data['Open'],
                                          high=stock_data['High'],
                                          low=stock_data['Low'],
                                          close=stock_data['Close'],
                                          name='Candlestick'), row=1, col=1)
            
            # Adding moving averages
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['20 Day MA'], mode='lines', name='20 Day MA', line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['50 Day MA'], mode='lines', name='50 Day MA', line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['100 Day MA'], mode='lines', name='100 Day MA', line=dict(color='green', width=1)), row=1, col=1)

            # Adding Bollinger Bands
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper Band'], mode='lines', name='Upper Band', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower Band'], mode='lines', name='Lower Band', line=dict(color='red', dash='dash')), row=1, col=1)

            # Add RSI plot
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=2, col=1)

            # Add MACD plot
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='teal')), row=3, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal Line'], mode='lines', name='Signal Line', line=dict(color='orange')), row=3, col=1)

            fig.update_layout(title=f"{ticker} - Stock Price with Technical Indicators", xaxis_title="Date", yaxis_title="Price", template='plotly_white')
            fig.show()

            # Correlation Heatmap
            correlation_heatmap(stock_data[['Close', '20 Day MA', '50 Day MA', 'Upper Band', 'Lower Band', 'RSI', 'MACD', 'ATR', 'VWAP', 'OBV', 'ADX']])
            
            # Summary Statistics
            print(f"\n--- Summary Statistics for {ticker} ---")
            print(stock_data.describe())

        except Exception as e:
            print(f"An error occurred while processing {ticker}: {e}")

# User input for tickers and date range
tickers_input = input("Enter ticker symbols separated by commas (e.g., 'HDFCBANK.NS, TATAMOTORS.NS'): ")
tickers = [ticker.strip() for ticker in tickers_input.split(',')]
start_date_input = input("Enter start date (YYYY-MM-DD) or press Enter for the last 3 months: ")
end_date_input = input("Enter end date (YYYY-MM-DD) or press Enter for today: ")

# Adjust the date input to proper format if needed
start_date = start_date_input if start_date_input else None
end_date = end_date_input if end_date_input else None

# Perform detailed analysis on the provided tickers
detailed_stock_analysis(tickers, start_date, end_date)
