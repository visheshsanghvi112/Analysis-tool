import requests
import csv
import io
import json

# Comprehensive mapping of major NSE symbols to sectors
SECTOR_MAP = {
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "INDUSINDBK": "Banking", "FEDERALBNK": "Banking", "BANDHANBNK": "Banking",
    "IDFCFIRSTB": "Banking", "PNB": "Banking", "BANKBARODA": "Banking", "CANBK": "Banking",
    "UNIONBANK": "Banking", "IOB": "Banking", "CENTRALBK": "Banking", "MAHABANK": "Banking",
    "UCOBANK": "Banking", "BANKINDIA": "Banking", "YESBANK": "Banking", "RBLBANK": "Banking",
    "KARURVYSYA": "Banking", "CUB": "Banking", "DCBBANK": "Banking", "SOUTHBANK": "Banking",
    "J&KBANK": "Banking", "PSB": "Banking", "BOB": "Banking",

    # IT / Software
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT", "LTIM": "IT", "LTI": "IT",
    "MPHASIS": "IT", "COFORGE": "IT", "PERSISTENT": "IT", "TATAELXSI": "IT", "KPITTECH": "IT",
    "CYIENT": "IT", "SONATSOFTW": "IT", "ZENSARTECH": "IT", "MASTEK": "IT",
    "NIITLTD": "IT", "BIRLASOFT": "IT", "ECLERX": "IT", "INTELLECT": "IT", "FSL": "IT",
    "NEWGEN": "IT", "QUESS": "IT",

    # Energy / Oil & Gas
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "IOC": "Energy", "NTPC": "Energy",
    "ADANIGREEN": "Energy", "POWERGRID": "Energy", "TATAPOWER": "Energy", "GAIL": "Energy",
    "PETRONET": "Energy", "HINDPETRO": "Energy", "MGL": "Energy", "IGL": "Energy", "GUJGASLTD": "Energy",
    "CESC": "Energy", "TORNTPOWER": "Energy", "JSWENERGY": "Energy", "SJVN": "Energy", "NHPC": "Energy",
    "OIL": "Energy", "MRPL": "Energy", "CHENNPETRO": "Energy", "ADANIPOWER": "Energy",

    # Pharma & Healthcare
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma", "DIVISLAB": "Pharma",
    "BIOCON": "Pharma", "AUROPHARMA": "Pharma", "LUPIN": "Pharma", "TORNTPHARM": "Pharma",
    "ALKEM": "Pharma", "IPCALAB": "Pharma", "GLENMARK": "Pharma", "ZYDUSLIFE": "Pharma",
    "CADILAHC": "Pharma", "ABBOTINDIA": "Pharma", "PFIZER": "Pharma", "GLAXO": "Pharma",
    "NATCOPHARM": "Pharma", "LALPATHLAB": "Pharma", "METROPOLIS": "Pharma", "THYROCARE": "Pharma",
    "APOLLOHOSP": "Pharma", "MAXHEALTH": "Pharma", "FORTIS": "Pharma", "SYNGENE": "Pharma",
    "LAURUSLABS": "Pharma", "GRANULES": "Pharma", "JBCHEPHARM": "Pharma", "MARKSANS": "Pharma",

    # Automobile
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto", "HEROMOTOCO": "Auto",
    "EICHERMOT": "Auto", "ASHOKLEY": "Auto", "TVSMOTOR": "Auto", "BOSCHLTD": "Auto", "BHARATFORG": "Auto",
    "MOTHERSON": "Auto", "APOLLOTYRE": "Auto", "MRF": "Auto", "CEATLTD": "Auto", "EXIDEIND": "Auto",
    "AMARAJABAT": "Auto", "TIINDIA": "Auto", "SUPRAJIT": "Auto", "CRAFTSMAN": "Auto", "SONACOMS": "Auto",

    # FMCG & Consumer Goods
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "BRITANNIA": "FMCG", "GODREJCP": "FMCG", "COLPAL": "FMCG", "EMAMILTD": "FMCG", "RADICO": "FMCG",
    "UBL": "FMCG", "UNITDSPR": "FMCG", "TATACONSUM": "FMCG", "VBL": "FMCG", "JUBLFOOD": "FMCG",
    "DEVYANI": "FMCG", "WESTLIFE": "FMCG", "BALRAMCHIN": "FMCG", "TRIDENT": "FMCG", "KRBL": "FMCG",
    "AVANTIFEED": "FMCG", "HATSUN": "FMCG", "BECTORFOOD": "FMCG",

    # Finance / NBFC / Insurance
    "BAJFINANCE": "Finance", "BAJAJFINSV": "Finance", "HDFCLIFE": "Finance", "SBILIFE": "Finance",
    "ICICIGI": "Finance", "MUTHOOTFIN": "Finance", "CHOLAFIN": "Finance", "RECLTD": "Finance",
    "PFC": "Finance", "M&MFIN": "Finance", "SHRIRAMFIN": "Finance", "MANAPPURAM": "Finance",
    "LICHSGFIN": "Finance", "CANFINHOME": "Finance", "HOMEFIRST": "Finance", "AAVAS": "Finance",
    "ICICIPRULI": "Finance", "HDFCAMC": "Finance", "NIPPONLIFE": "Finance", "NAUKRI": "Finance",
    "L&TFH": "Finance", "IBULHSGFIN": "Finance", "HUDCO": "Finance", "CREDITACC": "Finance",
    "CRISIL": "Finance", "ICRA": "Finance", "CDSL": "Finance", "BSE": "Finance", "MCX": "Finance",

    # Infrastructure / Cement / Construction
    "LT": "Infra", "ULTRACEMCO": "Infra", "ADANIPORTS": "Infra", "GRASIM": "Infra",
    "SHREECEM": "Infra", "SIEMENS": "Infra", "ABB": "Infra", "HAVELLS": "Infra", "POLYCAB": "Infra",
    "KEI": "Infra", "CUMMINSIND": "Infra", "THERMAX": "Infra", "VOLTAS": "Infra", "BLUESTARCO": "Infra",
    "INOXWIND": "Infra", "GPPL": "Infra", "CONCOR": "Infra", "KEC": "Infra", "IRCON": "Infra",
    "RVNL": "Infra", "NCC": "Infra", "DILIPBUILD": "Infra", "ENGINERSIN": "Infra", "AMBUJACEM": "Infra",
    "ACC": "Infra", "JKCEMENT": "Infra", "RAMCOCEM": "Infra", "HEIDELBERG": "Infra",

    # Metals & Mining
    "JSWSTEEL": "Metals", "TATASTEEL": "Metals", "HINDALCO": "Metals", "COALINDIA": "Metals",
    "VEDL": "Metals", "NMDC": "Metals", "SAIL": "Metals", "NATIONALUM": "Metals", "MOIL": "Metals",
    "APLAPOLLO": "Metals", "RATNAMANI": "Metals", "JSWISPL": "Metals", "HINDZINC": "Metals",
    "HINDCOPPER": "Metals", "WELCORP": "Metals", "JSL": "Metals", "GPIL": "Metals",

    # Telecom
    "BHARTIARTL": "Telecom", "VODAFONEIDEA": "Telecom", "TATACOMM": "Telecom", "STLTECH": "Telecom",
    "HFCL": "Telecom", "TEJASNET": "Telecom", "ROUTE": "Telecom", "TANLA": "Telecom", "TTML": "Telecom",

    # Consumer Tech & Retail
    "ZOMATO": "Consumer Tech", "NYKAA": "Consumer Tech", "PAYTM": "Consumer Tech",
    "POLICYBZR": "Consumer Tech", "DELHIVERY": "Consumer Tech", "CARTRADE": "Consumer Tech",
    "INDIAMART": "Consumer Tech", "JUSTDIAL": "Consumer Tech", "IRCTC": "Consumer Tech",
    "EASEMYTRIP": "Consumer Tech", "MAPMYINDIA": "Consumer Tech",
    "DMART": "Consumer Tech", "TRENT": "Consumer Tech", "ADANIENT": "Consumer Tech",
    "BATAINDIA": "Consumer Tech", "RELAXO": "Consumer Tech", "METROBRAND": "Consumer Tech",
}

import sys
import os

# Delegate directly to the master ticker builder
sys.path.insert(0, os.path.dirname(__file__))
from scripts.build_master_tickers import main

if __name__ == "__main__":
    main()

