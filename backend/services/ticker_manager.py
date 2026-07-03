import requests
import io
import pandas as pd

# Global ticker list and status
TICKER_LIST = []
_ticker_list_loaded = False

# NSE sector/industry mapping (bhavcopy mapping best-effort)
SECTOR_MAP = {
    "RELIANCE": "Energy", "TCS": "IT", "HDFCBANK": "Banking", "INFY": "IT",
    "ICICIBANK": "Banking", "HINDUNILVR": "FMCG", "ITC": "FMCG", "SBIN": "Banking",
    "BAJFINANCE": "NBFC", "BHARTIARTL": "Telecom", "KOTAKBANK": "Banking",
    "LT": "Infrastructure", "AXISBANK": "Banking", "ASIANPAINT": "FMCG",
    "MARUTI": "Auto", "TITAN": "Consumer", "WIPRO": "IT", "HCLTECH": "IT",
    "TECHM": "IT", "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "ONGC": "Energy", "NTPC": "Power", "POWERGRID": "Power",
    "COALINDIA": "Mining", "JSWSTEEL": "Metals", "TATASTEEL": "Metals",
    "HINDALCO": "Metals", "ADANIENT": "Diversified", "ADANIPORTS": "Logistics",
    "ULTRACEMCO": "Cement", "GRASIM": "Cement", "BAJAJFINSV": "NBFC",
    "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto", "EICHERMOT": "Auto",
    "TATAMOTORS": "Auto", "M&M": "Auto", "NESTLEIND": "FMCG", "BRITANNIA": "FMCG",
    "DABUR": "FMCG", "GODREJCP": "FMCG", "PIDILITIND": "Chemicals",
    "BERGEPAINT": "FMCG", "INDUSINDBK": "Banking", "FEDERALBNK": "Banking",
    "BANDHANBNK": "Banking", "IDFCFIRSTB": "Banking", "PNB": "Banking",
    "BANKBARODA": "Banking", "CANBK": "Banking", "UNIONBANK": "Banking",
    "HDFCLIFE": "Insurance", "SBILIFE": "Insurance", "ICICIPRULI": "Insurance",
    "MUTHOOTFIN": "NBFC", "CHOLAFIN": "NBFC", "RECLTD": "NBFC", "PFC": "NBFC",
    "ZOMATO": "Consumer Tech", "PAYTM": "Fintech", "NYKAA": "Consumer Tech",
    "POLICYBZR": "Fintech", "DELHIVERY": "Logistics",
}

def ensure_ticker_list():
    """Loads the NSE ticker list if not already loaded."""
    global TICKER_LIST, _ticker_list_loaded
    if _ticker_list_loaded:
        return TICKER_LIST
    _ticker_list_loaded = True
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
                sector = SECTOR_MAP.get(raw_symbol, None)
                entry = {"symbol": symbol, "name": name}
                if sector:
                    entry["sector"] = sector
                temp_list.append(entry)
            TICKER_LIST = temp_list
            print(f"Loaded {len(TICKER_LIST)} NSE tickers successfully.")
        else:
            print("Failed to download NSE tickers, using empty list")
    except Exception as e:
        print(f"Error loading ticker list: {e}")
        _ticker_list_loaded = False
    return TICKER_LIST
