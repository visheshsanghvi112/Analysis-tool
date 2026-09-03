import os
import json
import requests
import io
import pandas as pd

# Global ticker list and status
TICKER_LIST = []
_ticker_list_loaded = False

def _load_local_tickers():
    """Attempts to load tickers from bundled master JSON files."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "tickers.json"),
        os.path.join(os.path.dirname(__file__), "..", "tickers.json"),
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public", "tickers.json"),
        "backend/data/tickers.json",
        "backend/tickers.json",
        "frontend/public/tickers.json"
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return data
            except Exception as e:
                print(f"Error reading local tickers from {abs_path}: {e}")
    return []

# Eagerly load master ticker list on module import
TICKER_LIST = _load_local_tickers()
if TICKER_LIST:
    _ticker_list_loaded = True
    print(f"Instantly loaded {len(TICKER_LIST)} instruments (Equities, ETFs, Indices, Global) from master database.")

def ensure_ticker_list():
    """
    Returns the complete master list of all Indian Equities (NSE & BSE),
    ETFs, Benchmark Indices, and Global assets.
    """
    global TICKER_LIST, _ticker_list_loaded
    if _ticker_list_loaded and len(TICKER_LIST) > 0:
        return TICKER_LIST

    # Fallback to local files if not loaded
    TICKER_LIST = _load_local_tickers()
    if TICKER_LIST:
        _ticker_list_loaded = True
        return TICKER_LIST

    # Ultimate fallback: attempt download if local file was somehow missing
    try:
        url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        if res.ok:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = df.columns.str.strip()
            temp_list = []
            for _, row in df.iterrows():
                raw_symbol = str(row['SYMBOL']).strip()
                symbol = raw_symbol + ".NS"
                name = str(row['NAME OF COMPANY']).strip()
                temp_list.append({"symbol": symbol, "name": name, "sector": "Others", "type": "Equity"})
            TICKER_LIST = temp_list
            _ticker_list_loaded = True
            print(f"Fallback: Loaded {len(TICKER_LIST)} NSE tickers from network.")
    except Exception as e:
        print(f"Error loading ticker list: {e}")
        _ticker_list_loaded = False

    return TICKER_LIST

