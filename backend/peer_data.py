"""
peer_data.py — Curated sector peer map for NSE India.
Maps every major ticker to its real industry competitors.
Used by /api/peers and /api/sector-rank endpoints.
"""

SECTOR_PEERS = {
    # ── Oil & Gas ──────────────────────────────────────────────────────────
    "RELIANCE.NS":  {"sector": "Oil & Gas",      "peers": ["ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "PETRONET.NS"]},
    "ONGC.NS":      {"sector": "Oil & Gas",      "peers": ["RELIANCE.NS", "BPCL.NS", "IOC.NS", "OIL.NS", "GAIL.NS"]},
    "BPCL.NS":      {"sector": "Oil & Gas",      "peers": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "HPCL.NS"]},
    "IOC.NS":       {"sector": "Oil & Gas",      "peers": ["BPCL.NS", "HPCL.NS", "ONGC.NS", "RELIANCE.NS"]},
    "HPCL.NS":      {"sector": "Oil & Gas",      "peers": ["BPCL.NS", "IOC.NS", "ONGC.NS"]},
    "GAIL.NS":      {"sector": "Oil & Gas",      "peers": ["IGL.NS", "MGL.NS", "PETRONET.NS", "ONGC.NS"]},
    "PETRONET.NS":  {"sector": "Oil & Gas",      "peers": ["GAIL.NS", "IGL.NS", "MGL.NS", "RELIANCE.NS"]},
    "IGL.NS":       {"sector": "Oil & Gas",      "peers": ["MGL.NS", "GAIL.NS", "PETRONET.NS"]},
    "MGL.NS":       {"sector": "Oil & Gas",      "peers": ["IGL.NS", "GAIL.NS", "PETRONET.NS"]},
    "OIL.NS":       {"sector": "Oil & Gas",      "peers": ["ONGC.NS", "RELIANCE.NS", "BPCL.NS"]},

    # ── IT ────────────────────────────────────────────────────────────────
    "TCS.NS":       {"sector": "IT",             "peers": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"]},
    "INFY.NS":      {"sector": "IT",             "peers": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"]},
    "WIPRO.NS":     {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"]},
    "HCLTECH.NS":   {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"]},
    "TECHM.NS":     {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "LTIM.NS"]},
    "LTIM.NS":      {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "MPHASIS.NS"]},
    "MPHASIS.NS":   {"sector": "IT",             "peers": ["LTIM.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]},
    "PERSISTENT.NS":{"sector": "IT",             "peers": ["MPHASIS.NS", "LTIM.NS", "WIPRO.NS", "TECHM.NS"]},
    "COFORGE.NS":   {"sector": "IT",             "peers": ["MPHASIS.NS", "PERSISTENT.NS", "LTIM.NS"]},

    # ── Banking ───────────────────────────────────────────────────────────
    "HDFCBANK.NS":  {"sector": "Banking",        "peers": ["ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]},
    "ICICIBANK.NS": {"sector": "Banking",        "peers": ["HDFCBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]},
    "SBIN.NS":      {"sector": "Banking",        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "BANKBARODA.NS", "CANARABANK.NS"]},
    "KOTAKBANK.NS": {"sector": "Banking",        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]},
    "AXISBANK.NS":  {"sector": "Banking",        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS"]},
    "INDUSINDBK.NS":{"sector": "Banking",        "peers": ["AXISBANK.NS", "KOTAKBANK.NS", "ICICIBANK.NS", "FEDERALBNK.NS"]},
    "BANKBARODA.NS":{"sector": "Banking",        "peers": ["SBIN.NS", "CANARABANK.NS", "UNIONBANK.NS", "ICICIBANK.NS"]},
    "CANARABANK.NS":{"sector": "Banking",        "peers": ["SBIN.NS", "BANKBARODA.NS", "UNIONBANK.NS"]},
    "FEDERALBNK.NS":{"sector": "Banking",        "peers": ["INDUSINDBK.NS", "RBLBANK.NS", "BANDHANBNK.NS"]},
    "RBLBANK.NS":   {"sector": "Banking",        "peers": ["FEDERALBNK.NS", "INDUSINDBK.NS", "BANDHANBNK.NS"]},
    "BANDHANBNK.NS":{"sector": "Banking",        "peers": ["RBLBANK.NS", "FEDERALBNK.NS", "INDUSINDBK.NS"]},

    # ── FMCG ──────────────────────────────────────────────────────────────
    "HINDUNILVR.NS":{"sector": "FMCG",           "peers": ["ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"]},
    "ITC.NS":       {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS"]},
    "NESTLEIND.NS": {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS", "DABUR.NS"]},
    "BRITANNIA.NS": {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "MARICO.NS"]},
    "DABUR.NS":     {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "ITC.NS", "MARICO.NS", "GODREJCP.NS"]},
    "MARICO.NS":    {"sector": "FMCG",           "peers": ["DABUR.NS", "HINDUNILVR.NS", "GODREJCP.NS"]},
    "GODREJCP.NS":  {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "DABUR.NS", "MARICO.NS", "ITC.NS"]},
    "COLPAL.NS":    {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "DABUR.NS", "MARICO.NS"]},

    # ── Auto ──────────────────────────────────────────────────────────────
    "MARUTI.NS":    {"sector": "Auto",           "peers": ["TATAMOTORS.NS", "M&M.NS", "HYUNDAI.NS", "BAJAJ-AUTO.NS"]},
    "TATAMOTORS.NS":{"sector": "Auto",           "peers": ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS", "EICHERMOT.NS"]},
    "M&M.NS":       {"sector": "Auto",           "peers": ["MARUTI.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS"]},
    "HEROMOTOCO.NS":{"sector": "Auto",           "peers": ["BAJAJ-AUTO.NS", "TVSMOTORS.NS", "EICHERMOT.NS", "M&M.NS"]},
    "BAJAJ-AUTO.NS":{"sector": "Auto",           "peers": ["HEROMOTOCO.NS", "TVSMOTORS.NS", "EICHERMOT.NS"]},
    "EICHERMOT.NS": {"sector": "Auto",           "peers": ["BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "TVSMOTORS.NS"]},
    "TVSMOTORS.NS": {"sector": "Auto",           "peers": ["HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"]},
    "ASHOKLEY.NS":  {"sector": "Auto",           "peers": ["TATAMOTORS.NS", "TIINDIA.NS", "M&M.NS"]},
    "MOTHERSON.NS": {"sector": "Auto",           "peers": ["BOSCHLTD.NS", "TIINDIA.NS", "BALKRISIND.NS"]},
    "BALKRISIND.NS":{"sector": "Auto",           "peers": ["MOTHERSON.NS", "CEATLTD.NS", "MRF.NS"]},
    "MRF.NS":       {"sector": "Auto",           "peers": ["CEATLTD.NS", "APOLLOTYRE.NS", "BALKRISIND.NS"]},
    "APOLLOTYRE.NS":{"sector": "Auto",           "peers": ["MRF.NS", "CEATLTD.NS", "BALKRISIND.NS"]},

    # ── Pharma ────────────────────────────────────────────────────────────
    "SUNPHARMA.NS": {"sector": "Pharma",         "peers": ["DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS"]},
    "DRREDDY.NS":   {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"]},
    "CIPLA.NS":     {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "LUPIN.NS", "AUROPHARMA.NS"]},
    "DIVISLAB.NS":  {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS"]},
    "LUPIN.NS":     {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "AUROPHARMA.NS"]},
    "AUROPHARMA.NS":{"sector": "Pharma",         "peers": ["CIPLA.NS", "LUPIN.NS", "DRREDDY.NS", "SUNPHARMA.NS"]},
    "BIOCON.NS":    {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "AUROPHARMA.NS", "CIPLA.NS"]},
    "TORNTPHARM.NS":{"sector": "Pharma",         "peers": ["LUPIN.NS", "CIPLA.NS", "SUNPHARMA.NS"]},
    "IPCALAB.NS":   {"sector": "Pharma",         "peers": ["LUPIN.NS", "CIPLA.NS", "AUROPHARMA.NS"]},

    # ── Telecom ───────────────────────────────────────────────────────────
    "BHARTIARTL.NS":{"sector": "Telecom",        "peers": ["VODAFONEIDEA.NS", "TATACOMM.NS", "INDUSTOWER.NS"]},
    "VODAFONEIDEA.NS":{"sector": "Telecom",      "peers": ["BHARTIARTL.NS", "TATACOMM.NS", "INDUSTOWER.NS"]},
    "TATACOMM.NS":  {"sector": "Telecom",        "peers": ["BHARTIARTL.NS", "INDUSTOWER.NS"]},
    "INDUSTOWER.NS":{"sector": "Telecom",        "peers": ["BHARTIARTL.NS", "TATACOMM.NS"]},

    # ── Metals & Mining ───────────────────────────────────────────────────
    "TATASTEEL.NS": {"sector": "Metals",         "peers": ["JSWSTEEL.NS", "HINDALCO.NS", "SAIL.NS", "JSPL.NS", "VEDL.NS"]},
    "JSWSTEEL.NS":  {"sector": "Metals",         "peers": ["TATASTEEL.NS", "SAIL.NS", "JSPL.NS", "NMDC.NS"]},
    "HINDALCO.NS":  {"sector": "Metals",         "peers": ["VEDL.NS", "NATIONALUM.NS", "TATASTEEL.NS", "JSWSTEEL.NS"]},
    "VEDL.NS":      {"sector": "Metals",         "peers": ["HINDALCO.NS", "NATIONALUM.NS", "TATASTEEL.NS"]},
    "SAIL.NS":      {"sector": "Metals",         "peers": ["TATASTEEL.NS", "JSWSTEEL.NS", "JSPL.NS"]},
    "JSPL.NS":      {"sector": "Metals",         "peers": ["TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS"]},
    "NMDC.NS":      {"sector": "Metals",         "peers": ["SAIL.NS", "JSWSTEEL.NS", "COALINDIA.NS"]},
    "COALINDIA.NS": {"sector": "Metals",         "peers": ["NMDC.NS", "VEDL.NS", "SAIL.NS"]},

    # ── Power / Energy ────────────────────────────────────────────────────
    "NTPC.NS":      {"sector": "Power",          "peers": ["POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "TORNTPOWER.NS", "CESC.NS"]},
    "POWERGRID.NS": {"sector": "Power",          "peers": ["NTPC.NS", "TATAPOWER.NS", "ADANIGREEN.NS"]},
    "TATAPOWER.NS": {"sector": "Power",          "peers": ["NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "TORNTPOWER.NS"]},
    "ADANIGREEN.NS":{"sector": "Power",          "peers": ["NTPC.NS", "TATAPOWER.NS", "POWERGRID.NS"]},
    "TORNTPOWER.NS":{"sector": "Power",          "peers": ["NTPC.NS", "TATAPOWER.NS", "CESC.NS"]},
    "CESC.NS":      {"sector": "Power",          "peers": ["TORNTPOWER.NS", "NTPC.NS", "TATAPOWER.NS"]},

    # ── Real Estate ───────────────────────────────────────────────────────
    "DLF.NS":       {"sector": "Real Estate",    "peers": ["GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS"]},
    "GODREJPROP.NS":{"sector": "Real Estate",    "peers": ["DLF.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS"]},
    "OBEROIRLTY.NS":{"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "BRIGADE.NS"]},
    "PRESTIGE.NS":  {"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "BRIGADE.NS"]},
    "BRIGADE.NS":   {"sector": "Real Estate",    "peers": ["DLF.NS", "PRESTIGE.NS", "GODREJPROP.NS"]},
    "PHOENIXLTD.NS":{"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS"]},

    # ── Cement ────────────────────────────────────────────────────────────
    "ULTRACEMCO.NS":{"sector": "Cement",         "peers": ["AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "JKCEMENT.NS", "DALMIABHARAT.NS"]},
    "AMBUJACEM.NS": {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "ACC.NS", "SHREECEM.NS", "JKCEMENT.NS"]},
    "ACC.NS":       {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS"]},
    "SHREECEM.NS":  {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "AMBUJACEM.NS", "DALMIABHARAT.NS"]},
    "JKCEMENT.NS":  {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS"]},
    "DALMIABHARAT.NS":{"sector": "Cement",       "peers": ["SHREECEM.NS", "ULTRACEMCO.NS", "AMBUJACEM.NS"]},

    # ── Finance / NBFC ────────────────────────────────────────────────────
    "BAJFINANCE.NS":{"sector": "Finance",        "peers": ["BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "SBICARD.NS"]},
    "BAJAJFINSV.NS":{"sector": "Finance",        "peers": ["BAJFINANCE.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS"]},
    "CHOLAFIN.NS":  {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "MUTHOOTFIN.NS", "L&TFH.NS"]},
    "MUTHOOTFIN.NS":{"sector": "Finance",        "peers": ["BAJFINANCE.NS", "CHOLAFIN.NS", "MANAPPURAM.NS"]},
    "SBICARD.NS":   {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "HDFCAMC.NS", "NIPPONLIFE.NS"]},
    "HDFCAMC.NS":   {"sector": "Finance",        "peers": ["NIPPONLIFE.NS", "SBICARD.NS", "BAJFINANCE.NS"]},
    "MANAPPURAM.NS":{"sector": "Finance",        "peers": ["MUTHOOTFIN.NS", "CHOLAFIN.NS", "BAJFINANCE.NS"]},

    # ── Insurance ─────────────────────────────────────────────────────────
    "SBILIFE.NS":   {"sector": "Insurance",      "peers": ["HDFCLIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS", "LICI.NS"]},
    "HDFCLIFE.NS":  {"sector": "Insurance",      "peers": ["SBILIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS"]},
    "ICICIGI.NS":   {"sector": "Insurance",      "peers": ["SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"]},
    "ICICIPRULI.NS":{"sector": "Insurance",      "peers": ["SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS"]},
    "LICI.NS":      {"sector": "Insurance",      "peers": ["SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS"]},

    # ── Capital Goods / Infra ─────────────────────────────────────────────
    "LT.NS":        {"sector": "Capital Goods",  "peers": ["SIEMENS.NS", "ABB.NS", "BHEL.NS", "CUMMINSIND.NS", "THERMAX.NS"]},
    "SIEMENS.NS":   {"sector": "Capital Goods",  "peers": ["LT.NS", "ABB.NS", "BHEL.NS", "CUMMINSIND.NS"]},
    "ABB.NS":       {"sector": "Capital Goods",  "peers": ["LT.NS", "SIEMENS.NS", "BHEL.NS"]},
    "BHEL.NS":      {"sector": "Capital Goods",  "peers": ["LT.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS"]},
    "CUMMINSIND.NS":{"sector": "Capital Goods",  "peers": ["SIEMENS.NS", "ABB.NS", "BHEL.NS", "LT.NS"]},
    "THERMAX.NS":   {"sector": "Capital Goods",  "peers": ["LT.NS", "SIEMENS.NS", "CUMMINSIND.NS"]},
    "ADANIPORTS.NS":{"sector": "Capital Goods",  "peers": ["LT.NS", "IRB.NS", "POLYCAB.NS"]},
    "POLYCAB.NS":   {"sector": "Capital Goods",  "peers": ["HAVELLS.NS", "KEI.NS", "CUMMINSIND.NS"]},

    # ── Consumer Durables ─────────────────────────────────────────────────
    "TITAN.NS":     {"sector": "Consumer",       "peers": ["HAVELLS.NS", "VOLTAS.NS", "BLUESTARCO.NS", "CROMPTON.NS"]},
    "HAVELLS.NS":   {"sector": "Consumer",       "peers": ["TITAN.NS", "VOLTAS.NS", "CROMPTON.NS", "VGUARD.NS"]},
    "VOLTAS.NS":    {"sector": "Consumer",       "peers": ["HAVELLS.NS", "BLUESTARCO.NS", "CROMPTON.NS", "TITAN.NS"]},
    "BLUESTARCO.NS":{"sector": "Consumer",       "peers": ["VOLTAS.NS", "HAVELLS.NS", "CROMPTON.NS"]},
    "CROMPTON.NS":  {"sector": "Consumer",       "peers": ["HAVELLS.NS", "VOLTAS.NS", "VGUARD.NS"]},
    "VGUARD.NS":    {"sector": "Consumer",       "peers": ["CROMPTON.NS", "HAVELLS.NS", "VOLTAS.NS"]},

    # ── Aviation ──────────────────────────────────────────────────────────
    "INDIGO.NS":    {"sector": "Aviation",       "peers": ["SPICEJET.NS"]},
    "SPICEJET.NS":  {"sector": "Aviation",       "peers": ["INDIGO.NS"]},

    # ── Paint ─────────────────────────────────────────────────────────────
    "ASIANPAINT.NS":{"sector": "Paint",          "peers": ["BERGEPAINT.NS", "KANSAINER.NS", "INDIGO.NS", "AKZONOBL.NS"]},
    "BERGEPAINT.NS":{"sector": "Paint",          "peers": ["ASIANPAINT.NS", "KANSAINER.NS", "AKZONOBL.NS"]},
    "KANSAINER.NS": {"sector": "Paint",          "peers": ["ASIANPAINT.NS", "BERGEPAINT.NS", "AKZONOBL.NS"]},

    # ── Chemicals ─────────────────────────────────────────────────────────
    "PIDILITIND.NS":{"sector": "Chemicals",      "peers": ["ATUL.NS", "DEEPAKNTR.NS", "TATACHEM.NS"]},
    "TATACHEM.NS":  {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "ATUL.NS", "DEEPAKNTR.NS"]},
    "DEEPAKNTR.NS": {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "TATACHEM.NS", "ATUL.NS"]},
    "ATUL.NS":      {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "TATACHEM.NS", "DEEPAKNTR.NS"]},

    # ── Hospitality ───────────────────────────────────────────────────────
    "IHCL.NS":      {"sector": "Hospitality",    "peers": ["LEMONTREE.NS", "EIH.NS", "CHALET.NS"]},
    "LEMONTREE.NS": {"sector": "Hospitality",    "peers": ["IHCL.NS", "EIH.NS", "CHALET.NS"]},
    "EIH.NS":       {"sector": "Hospitality",    "peers": ["IHCL.NS", "LEMONTREE.NS", "CHALET.NS"]},

    # ── E-Commerce / Internet ─────────────────────────────────────────────
    "ZOMATO.NS":    {"sector": "Internet",       "peers": ["SWIGGY.NS", "NYKAA.NS", "DMART.NS"]},
    "NYKAA.NS":     {"sector": "Internet",       "peers": ["ZOMATO.NS", "MAMAEARTH.NS", "SWIGGY.NS"]},
    "POLICYBZR.NS": {"sector": "Internet",       "peers": ["NYKAA.NS", "ZOMATO.NS"]},

    # ── Retail ────────────────────────────────────────────────────────────
    "DMART.NS":     {"sector": "Retail",         "peers": ["TRENT.NS", "ABFRL.NS", "SHOPERSTOP.NS"]},
    "TRENT.NS":     {"sector": "Retail",         "peers": ["DMART.NS", "ABFRL.NS", "SHOPERSTOP.NS"]},
}


def get_peers(ticker: str) -> dict:
    """Return sector name and peer list for a given ticker."""
    data = SECTOR_PEERS.get(ticker)
    if data:
        return {
            "sector": data["sector"],
            "peers": data["peers"],
            "found": True,
        }
    # Fallback: unknown ticker
    return {"sector": "Unknown", "peers": [], "found": False}


def get_all_sector_members(ticker: str) -> list[str]:
    """Return all tickers in the same sector as the given ticker."""
    data = SECTOR_PEERS.get(ticker)
    if not data:
        return []
    sector = data["sector"]
    return [t for t, d in SECTOR_PEERS.items() if d["sector"] == sector and t != ticker]
