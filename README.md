# Stock Analysis Tool

## Overview

This Stock Analysis Tool is a comprehensive Python application that fetches historical stock data from Yahoo Finance, calculates various technical indicators, performs sentiment analysis on related news articles, and provides insights on support and resistance levels. It employs advanced visualization techniques using Plotly for interactive charts and includes a correlation heatmap for technical indicators.

## Features

- **Historical Data Fetching**: Retrieve stock data for specified tickers over custom date ranges.
- **Technical Indicators**: Calculate and visualize moving averages, Bollinger Bands, RSI, MACD, ATR, VWAP, OBV, and ADX.
- **Sentiment Analysis**: Fetch and analyze news sentiment related to the stock, providing a sentiment score and subjectivity.
- **Support and Resistance Levels**: Automatically calculate key support and resistance levels based on historical price data.
- **Interactive Visualization**: Display candlestick charts and technical indicators using Plotly for an interactive user experience.
- **Correlation Heatmap**: Visualize the correlation between various technical indicators.

## Requirements

To run this application, you will need the following libraries:

- `yfinance`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `textblob`
- `requests`

You can install these libraries using pip:

```bash
pip install yfinance numpy pandas matplotlib seaborn plotly textblob requests

git clone https://github.com/yourusername/stock-analysis-tool.git
cd stock-analysis-tool
python test.py


### Instructions for Creating a GitHub Repository

1. Go to GitHub and log in to your account.
2. Click on the **New** button to create a new repository.
3. Fill in the repository name (e.g., `stock-analysis-tool`), description, and choose visibility (public/private).
4. Click **Create repository**.
5. Follow the instructions to push your local code to GitHub (you may need to initialize a git repository in your project directory if you haven't already).

Feel free to modify the README as per your preferences!
Example
When prompted, enter ticker symbols (e.g., HDFCBANK.NS) and specify start and end dates for analysis. The script will output various statistics, visualizations, and insights.
