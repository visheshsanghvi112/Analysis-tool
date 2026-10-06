"""
peer_data.py — Universal sector peer map and dynamic peer resolution engine.
Maps major Indian and global equities, ETFs, and sectoral themes to real industry peers.
Provides multi-tier dynamic fallback so any queried ticker has authentic sector context and peers.
Used by /api/peers, /api/peer-compare, and /api/sector-rank endpoints.
"""

# Curated high-precision peer clusters
SECTOR_PEERS = {
    # ── Oil & Gas / Energy (India) ──────────────────────────────────────────
    "RELIANCE.NS":   {"sector": "Oil & Gas",      "peers": ["ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "PETRONET.NS"]},
    "ONGC.NS":       {"sector": "Oil & Gas",      "peers": ["RELIANCE.NS", "BPCL.NS", "IOC.NS", "OIL.NS", "GAIL.NS"]},
    "BPCL.NS":       {"sector": "Oil & Gas",      "peers": ["RELIANCE.NS", "ONGC.NS", "IOC.NS", "HPCL.NS"]},
    "IOC.NS":        {"sector": "Oil & Gas",      "peers": ["BPCL.NS", "HPCL.NS", "ONGC.NS", "RELIANCE.NS"]},
    "HPCL.NS":       {"sector": "Oil & Gas",      "peers": ["BPCL.NS", "IOC.NS", "ONGC.NS"]},
    "GAIL.NS":       {"sector": "Oil & Gas",      "peers": ["IGL.NS", "MGL.NS", "PETRONET.NS", "ONGC.NS"]},
    "PETRONET.NS":   {"sector": "Oil & Gas",      "peers": ["GAIL.NS", "IGL.NS", "MGL.NS", "RELIANCE.NS"]},
    "IGL.NS":        {"sector": "Oil & Gas",      "peers": ["MGL.NS", "GAIL.NS", "PETRONET.NS"]},
    "MGL.NS":        {"sector": "Oil & Gas",      "peers": ["IGL.NS", "GAIL.NS", "PETRONET.NS"]},
    "OIL.NS":        {"sector": "Oil & Gas",      "peers": ["ONGC.NS", "RELIANCE.NS", "BPCL.NS"]},

    # ── IT & Software Services (India) ──────────────────────────────────────
    "TCS.NS":        {"sector": "IT",             "peers": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"]},
    "INFY.NS":       {"sector": "IT",             "peers": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"]},
    "WIPRO.NS":      {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"]},
    "HCLTECH.NS":    {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"]},
    "TECHM.NS":      {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "LTIM.NS"]},
    "LTIM.NS":       {"sector": "IT",             "peers": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "MPHASIS.NS"]},
    "MPHASIS.NS":    {"sector": "IT",             "peers": ["LTIM.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]},
    "PERSISTENT.NS": {"sector": "IT",             "peers": ["MPHASIS.NS", "LTIM.NS", "WIPRO.NS", "TECHM.NS"]},
    "COFORGE.NS":    {"sector": "IT",             "peers": ["MPHASIS.NS", "PERSISTENT.NS", "LTIM.NS"]},
    "TATAELXSI.NS":  {"sector": "IT",             "peers": ["KPITTECH.NS", "LTTS.NS", "PERSISTENT.NS", "COFORGE.NS"]},
    "LTTS.NS":       {"sector": "IT",             "peers": ["TATAELXSI.NS", "KPITTECH.NS", "PERSISTENT.NS"]},
    "KPITTECH.NS":   {"sector": "IT",             "peers": ["TATAELXSI.NS", "LTTS.NS", "COFORGE.NS"]},

    # ── Banking (India) ─────────────────────────────────────────────────────
    "HDFCBANK.NS":   {"sector": "Banking",        "peers": ["ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]},
    "ICICIBANK.NS":  {"sector": "Banking",        "peers": ["HDFCBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]},
    "SBIN.NS":       {"sector": "Banking",        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "BANKBARODA.NS", "CANARABANK.NS"]},
    "KOTAKBANK.NS":  {"sector": "Banking",        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"]},
    "AXISBANK.NS":   {"sector": "Banking",        "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "INDUSINDBK.NS"]},
    "INDUSINDBK.NS": {"sector": "Banking",        "peers": ["AXISBANK.NS", "KOTAKBANK.NS", "ICICIBANK.NS", "FEDERALBNK.NS"]},
    "BANKBARODA.NS": {"sector": "Banking",        "peers": ["SBIN.NS", "CANARABANK.NS", "PNB.NS", "UNIONBANK.NS", "ICICIBANK.NS"]},
    "CANARABANK.NS": {"sector": "Banking",        "peers": ["SBIN.NS", "BANKBARODA.NS", "PNB.NS", "UNIONBANK.NS"]},
    "PNB.NS":        {"sector": "Banking",        "peers": ["SBIN.NS", "BANKBARODA.NS", "CANARABANK.NS", "UNIONBANK.NS"]},
    "UNIONBANK.NS":  {"sector": "Banking",        "peers": ["SBIN.NS", "BANKBARODA.NS", "CANARABANK.NS", "PNB.NS"]},
    "FEDERALBNK.NS": {"sector": "Banking",        "peers": ["INDUSINDBK.NS", "IDFCFIRSTB.NS", "RBLBANK.NS", "BANDHANBNK.NS"]},
    "IDFCFIRSTB.NS": {"sector": "Banking",        "peers": ["FEDERALBNK.NS", "INDUSINDBK.NS", "RBLBANK.NS", "BANDHANBNK.NS"]},
    "RBLBANK.NS":    {"sector": "Banking",        "peers": ["FEDERALBNK.NS", "IDFCFIRSTB.NS", "INDUSINDBK.NS", "BANDHANBNK.NS"]},
    "BANDHANBNK.NS": {"sector": "Banking",        "peers": ["RBLBANK.NS", "FEDERALBNK.NS", "INDUSINDBK.NS"]},

    # ── Defence & Aerospace (India) ─────────────────────────────────────────
    "HAL.NS":        {"sector": "Defence",        "peers": ["BEL.NS", "BDL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "GRSE.NS", "DATAPATTNS.NS"]},
    "BEL.NS":        {"sector": "Defence",        "peers": ["HAL.NS", "BDL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "DATAPATTNS.NS", "PARAS.NS"]},
    "BDL.NS":        {"sector": "Defence",        "peers": ["HAL.NS", "BEL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "DATAPATTNS.NS"]},
    "MAZDOCK.NS":    {"sector": "Defence",        "peers": ["COCHINSHIP.NS", "GRSE.NS", "HAL.NS", "BEL.NS", "BDL.NS"]},
    "COCHINSHIP.NS": {"sector": "Defence",        "peers": ["MAZDOCK.NS", "GRSE.NS", "HAL.NS", "BEL.NS"]},
    "GRSE.NS":       {"sector": "Defence",        "peers": ["MAZDOCK.NS", "COCHINSHIP.NS", "HAL.NS", "BEL.NS"]},
    "DATAPATTNS.NS": {"sector": "Defence",        "peers": ["PARAS.NS", "HAL.NS", "BEL.NS", "BDL.NS"]},
    "PARAS.NS":      {"sector": "Defence",        "peers": ["DATAPATTNS.NS", "HAL.NS", "BEL.NS"]},
    "SOLARINDS.NS":  {"sector": "Defence",        "peers": ["HAL.NS", "BEL.NS", "BDL.NS"]},

    # ── Railways (India) ────────────────────────────────────────────────────
    "IRFC.NS":       {"sector": "Railways",       "peers": ["RVNL.NS", "IRCON.NS", "IRCTC.NS", "RAILTEL.NS", "RITES.NS", "TITAGARH.NS"]},
    "RVNL.NS":       {"sector": "Railways",       "peers": ["IRFC.NS", "IRCON.NS", "IRCTC.NS", "RAILTEL.NS", "RITES.NS"]},
    "IRCON.NS":      {"sector": "Railways",       "peers": ["RVNL.NS", "IRFC.NS", "RITES.NS", "RAILTEL.NS"]},
    "IRCTC.NS":      {"sector": "Railways",       "peers": ["IRFC.NS", "RVNL.NS", "RITES.NS", "RAILTEL.NS"]},
    "RAILTEL.NS":    {"sector": "Railways",       "peers": ["IRFC.NS", "RVNL.NS", "IRCON.NS", "RITES.NS"]},
    "RITES.NS":      {"sector": "Railways",       "peers": ["IRCON.NS", "RVNL.NS", "IRFC.NS", "RAILTEL.NS"]},
    "TITAGARH.NS":   {"sector": "Railways",       "peers": ["JWL.NS", "BEML.NS", "IRFC.NS", "RVNL.NS"]},
    "JWL.NS":        {"sector": "Railways",       "peers": ["TITAGARH.NS", "BEML.NS", "IRFC.NS", "RVNL.NS"]},
    "BEML.NS":       {"sector": "Railways",       "peers": ["TITAGARH.NS", "JWL.NS", "BEL.NS", "RVNL.NS"]},

    # ── Power & Green Energy (India) ────────────────────────────────────────
    "PFC.NS":        {"sector": "Power & Energy", "peers": ["RECLTD.NS", "IREDA.NS", "POWERGRID.NS", "NTPC.NS", "TATAPOWER.NS"]},
    "RECLTD.NS":     {"sector": "Power & Energy", "peers": ["PFC.NS", "IREDA.NS", "POWERGRID.NS", "NTPC.NS", "TATAPOWER.NS"]},
    "IREDA.NS":      {"sector": "Power & Energy", "peers": ["PFC.NS", "RECLTD.NS", "SUZLON.NS", "ADANIGREEN.NS", "TATAPOWER.NS"]},
    "SUZLON.NS":     {"sector": "Power & Energy", "peers": ["IREDA.NS", "WAAREEENER.NS", "PREMIERENE.NS", "TATAPOWER.NS", "ADANIGREEN.NS"]},
    "WAAREEENER.NS": {"sector": "Power & Energy", "peers": ["SUZLON.NS", "PREMIERENE.NS", "IREDA.NS", "TATAPOWER.NS"]},
    "PREMIERENE.NS": {"sector": "Power & Energy", "peers": ["WAAREEENER.NS", "SUZLON.NS", "IREDA.NS", "TATAPOWER.NS"]},
    "NTPC.NS":       {"sector": "Power & Energy", "peers": ["POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "NHPC.NS", "TORNTPOWER.NS"]},
    "POWERGRID.NS":  {"sector": "Power & Energy", "peers": ["NTPC.NS", "TATAPOWER.NS", "ADANIGREEN.NS", "PFC.NS"]},
    "TATAPOWER.NS":  {"sector": "Power & Energy", "peers": ["NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS", "SUZLON.NS", "TORNTPOWER.NS"]},
    "ADANIGREEN.NS": {"sector": "Power & Energy", "peers": ["NTPC.NS", "TATAPOWER.NS", "POWERGRID.NS", "SUZLON.NS"]},
    "NHPC.NS":       {"sector": "Power & Energy", "peers": ["SJVN.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS"]},
    "SJVN.NS":       {"sector": "Power & Energy", "peers": ["NHPC.NS", "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS"]},
    "TORNTPOWER.NS": {"sector": "Power & Energy", "peers": ["NTPC.NS", "TATAPOWER.NS", "CESC.NS"]},
    "CESC.NS":       {"sector": "Power & Energy", "peers": ["TORNTPOWER.NS", "NTPC.NS", "TATAPOWER.NS"]},

    # ── Internet & Consumer Tech / Quick Commerce (India) ───────────────────
    "ETERNAL.NS":    {"sector": "Internet & Tech","peers": ["SWIGGY.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS"]},
    "SWIGGY.NS":     {"sector": "Internet & Tech","peers": ["ETERNAL.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS"]},
    "ZOMATO.NS":     {"sector": "Internet & Tech","peers": ["SWIGGY.NS", "ETERNAL.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS"]},
    "NYKAA.NS":      {"sector": "Internet & Tech","peers": ["ETERNAL.NS", "SWIGGY.NS", "ZOMATO.NS", "PAYTM.NS", "POLICYBZR.NS"]},
    "PAYTM.NS":      {"sector": "Internet & Tech","peers": ["POLICYBZR.NS", "ETERNAL.NS", "SWIGGY.NS", "ZOMATO.NS", "NYKAA.NS"]},
    "POLICYBZR.NS":  {"sector": "Internet & Tech","peers": ["PAYTM.NS", "ETERNAL.NS", "SWIGGY.NS", "NYKAA.NS", "ZOMATO.NS"]},
    "DELHIVERY.NS":  {"sector": "Internet & Tech","peers": ["ETERNAL.NS", "SWIGGY.NS", "BLUEDART.NS", "NYKAA.NS"]},
    "NAUKRI.NS":     {"sector": "Internet & Tech","peers": ["ETERNAL.NS", "POLICYBZR.NS", "PAYTM.NS", "ZOMATO.NS"]},

    # ── FMCG (India) ────────────────────────────────────────────────────────
    "HINDUNILVR.NS": {"sector": "FMCG",           "peers": ["ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"]},
    "ITC.NS":        {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "GODREJCP.NS"]},
    "NESTLEIND.NS":  {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "ITC.NS", "BRITANNIA.NS", "DABUR.NS", "VARUN.NS"]},
    "BRITANNIA.NS":  {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "MARICO.NS", "VARUN.NS"]},
    "DABUR.NS":      {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "ITC.NS", "MARICO.NS", "GODREJCP.NS"]},
    "MARICO.NS":     {"sector": "FMCG",           "peers": ["DABUR.NS", "HINDUNILVR.NS", "GODREJCP.NS", "COLPAL.NS"]},
    "GODREJCP.NS":   {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "DABUR.NS", "MARICO.NS", "ITC.NS"]},
    "COLPAL.NS":     {"sector": "FMCG",           "peers": ["HINDUNILVR.NS", "DABUR.NS", "MARICO.NS"]},
    "VARUN.NS":      {"sector": "FMCG",           "peers": ["NESTLEIND.NS", "BRITANNIA.NS", "ITC.NS", "HINDUNILVR.NS"]},

    # ── Auto & Mobility (India) ─────────────────────────────────────────────
    "MARUTI.NS":     {"sector": "Auto",           "peers": ["TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HYUNDAI.NS"]},
    "TATAMOTORS.NS": {"sector": "Auto",           "peers": ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"]},
    "M&M.NS":        {"sector": "Auto",           "peers": ["MARUTI.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"]},
    "HEROMOTOCO.NS": {"sector": "Auto",           "peers": ["BAJAJ-AUTO.NS", "TVSMOTORS.NS", "EICHERMOT.NS", "M&M.NS"]},
    "BAJAJ-AUTO.NS": {"sector": "Auto",           "peers": ["HEROMOTOCO.NS", "TVSMOTORS.NS", "EICHERMOT.NS", "M&M.NS"]},
    "EICHERMOT.NS":  {"sector": "Auto",           "peers": ["BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "TVSMOTORS.NS", "TATAMOTORS.NS"]},
    "TVSMOTORS.NS":  {"sector": "Auto",           "peers": ["HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"]},
    "ASHOKLEY.NS":   {"sector": "Auto",           "peers": ["TATAMOTORS.NS", "M&M.NS", "EICHERMOT.NS"]},
    "MOTHERSON.NS":  {"sector": "Auto Ancillary", "peers": ["BOSCHLTD.NS", "BALKRISIND.NS", "MRF.NS", "ENDURANCE.NS"]},
    "BOSCHLTD.NS":   {"sector": "Auto Ancillary", "peers": ["MOTHERSON.NS", "BALKRISIND.NS", "MRF.NS"]},
    "BALKRISIND.NS": {"sector": "Auto Ancillary", "peers": ["MOTHERSON.NS", "MRF.NS", "APOLLOTYRE.NS", "CEATLTD.NS"]},
    "MRF.NS":        {"sector": "Auto Ancillary", "peers": ["APOLLOTYRE.NS", "BALKRISIND.NS", "CEATLTD.NS"]},
    "APOLLOTYRE.NS": {"sector": "Auto Ancillary", "peers": ["MRF.NS", "BALKRISIND.NS", "CEATLTD.NS"]},

    # ── Pharma & Healthcare (India) ─────────────────────────────────────────
    "SUNPHARMA.NS":  {"sector": "Pharma",         "peers": ["DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "AUROPHARMA.NS", "MANKIND.NS"]},
    "DRREDDY.NS":    {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "MANKIND.NS"]},
    "CIPLA.NS":      {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "LUPIN.NS", "AUROPHARMA.NS", "MANKIND.NS"]},
    "DIVISLAB.NS":   {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "LUPIN.NS"]},
    "LUPIN.NS":      {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "AUROPHARMA.NS"]},
    "AUROPHARMA.NS": {"sector": "Pharma",         "peers": ["CIPLA.NS", "LUPIN.NS", "DRREDDY.NS", "SUNPHARMA.NS"]},
    "MANKIND.NS":    {"sector": "Pharma",         "peers": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "TORNTPHARM.NS"]},
    "TORNTPHARM.NS": {"sector": "Pharma",         "peers": ["MANKIND.NS", "LUPIN.NS", "CIPLA.NS", "SUNPHARMA.NS"]},
    "APOLLOHOSP.NS": {"sector": "Healthcare",     "peers": ["MAXHEALTH.NS", "FORTIS.NS", "MEDANTA.NS"]},
    "MAXHEALTH.NS":  {"sector": "Healthcare",     "peers": ["APOLLOHOSP.NS", "FORTIS.NS", "MEDANTA.NS"]},
    "FORTIS.NS":     {"sector": "Healthcare",     "peers": ["APOLLOHOSP.NS", "MAXHEALTH.NS", "MEDANTA.NS"]},

    # ── Metals & Mining (India) ─────────────────────────────────────────────
    "TATASTEEL.NS":  {"sector": "Metals",         "peers": ["JSWSTEEL.NS", "HINDALCO.NS", "SAIL.NS", "JSPL.NS", "VEDL.NS", "COALINDIA.NS"]},
    "JSWSTEEL.NS":   {"sector": "Metals",         "peers": ["TATASTEEL.NS", "SAIL.NS", "JSPL.NS", "NMDC.NS", "HINDALCO.NS"]},
    "HINDALCO.NS":   {"sector": "Metals",         "peers": ["VEDL.NS", "NATIONALUM.NS", "TATASTEEL.NS", "JSWSTEEL.NS"]},
    "VEDL.NS":       {"sector": "Metals",         "peers": ["HINDALCO.NS", "NATIONALUM.NS", "TATASTEEL.NS", "COALINDIA.NS"]},
    "SAIL.NS":       {"sector": "Metals",         "peers": ["TATASTEEL.NS", "JSWSTEEL.NS", "JSPL.NS", "NMDC.NS"]},
    "JSPL.NS":       {"sector": "Metals",         "peers": ["TATASTEEL.NS", "JSWSTEEL.NS", "SAIL.NS"]},
    "NMDC.NS":       {"sector": "Metals",         "peers": ["SAIL.NS", "JSWSTEEL.NS", "COALINDIA.NS", "TATASTEEL.NS"]},
    "COALINDIA.NS":  {"sector": "Metals",         "peers": ["NMDC.NS", "VEDL.NS", "SAIL.NS", "TATASTEEL.NS"]},
    "NATIONALUM.NS": {"sector": "Metals",         "peers": ["HINDALCO.NS", "VEDL.NS", "TATASTEEL.NS"]},

    # ── Real Estate & Infrastructure Developers (India) ─────────────────────
    "DLF.NS":        {"sector": "Real Estate",    "peers": ["GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "BRIGADE.NS", "LODHA.NS"]},
    "GODREJPROP.NS": {"sector": "Real Estate",    "peers": ["DLF.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS", "LODHA.NS"]},
    "OBEROIRLTY.NS": {"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "BRIGADE.NS", "PRESTIGE.NS"]},
    "PRESTIGE.NS":   {"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "BRIGADE.NS", "OBEROIRLTY.NS"]},
    "BRIGADE.NS":    {"sector": "Real Estate",    "peers": ["DLF.NS", "PRESTIGE.NS", "GODREJPROP.NS", "SOBHA.NS"]},
    "PHOENIXLTD.NS": {"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS"]},
    "LODHA.NS":      {"sector": "Real Estate",    "peers": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS"]},

    # ── Cement & Building Materials (India) ─────────────────────────────────
    "ULTRACEMCO.NS": {"sector": "Cement",         "peers": ["AMBUJACEM.NS", "ACC.NS", "SHREECEM.NS", "JKCEMENT.NS", "DALMIABHARAT.NS"]},
    "AMBUJACEM.NS":  {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "ACC.NS", "SHREECEM.NS", "JKCEMENT.NS"]},
    "ACC.NS":        {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS"]},
    "SHREECEM.NS":   {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "AMBUJACEM.NS", "DALMIABHARAT.NS"]},
    "JKCEMENT.NS":   {"sector": "Cement",         "peers": ["ULTRACEMCO.NS", "AMBUJACEM.NS", "SHREECEM.NS"]},
    "DALMIABHARAT.NS":{"sector": "Cement",        "peers": ["SHREECEM.NS", "ULTRACEMCO.NS", "AMBUJACEM.NS"]},

    # ── NBFC & Financial Services (India) ───────────────────────────────────
    "BAJFINANCE.NS": {"sector": "Finance",        "peers": ["BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "SBICARD.NS", "SHRIRAMFIN.NS"]},
    "BAJAJFINSV.NS": {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "SHRIRAMFIN.NS"]},
    "CHOLAFIN.NS":   {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "MUTHOOTFIN.NS", "SHRIRAMFIN.NS"]},
    "MUTHOOTFIN.NS": {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "CHOLAFIN.NS", "MANAPPURAM.NS"]},
    "SHRIRAMFIN.NS": {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "CHOLAFIN.NS", "BAJAJFINSV.NS"]},
    "SBICARD.NS":    {"sector": "Finance",        "peers": ["BAJFINANCE.NS", "HDFCAMC.NS", "NIPPONLIFE.NS"]},
    "HDFCAMC.NS":    {"sector": "Asset Mgmt",     "peers": ["NIPPONLIFE.NS", "UTIAMC.NS", "SBICARD.NS"]},
    "NIPPONLIFE.NS": {"sector": "Asset Mgmt",     "peers": ["HDFCAMC.NS", "UTIAMC.NS", "SBICARD.NS"]},

    # ── Insurance (India) ───────────────────────────────────────────────────
    "SBILIFE.NS":    {"sector": "Insurance",      "peers": ["HDFCLIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS", "LICI.NS"]},
    "HDFCLIFE.NS":   {"sector": "Insurance",      "peers": ["SBILIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS", "LICI.NS"]},
    "ICICIGI.NS":    {"sector": "Insurance",      "peers": ["SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS"]},
    "ICICIPRULI.NS": {"sector": "Insurance",      "peers": ["SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "LICI.NS"]},
    "LICI.NS":       {"sector": "Insurance",      "peers": ["SBILIFE.NS", "HDFCLIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS"]},

    # ── Capital Goods & Infrastructure Engineering (India) ──────────────────
    "LT.NS":         {"sector": "Capital Goods",  "peers": ["SIEMENS.NS", "ABB.NS", "BHEL.NS", "CUMMINSIND.NS", "THERMAX.NS"]},
    "SIEMENS.NS":    {"sector": "Capital Goods",  "peers": ["LT.NS", "ABB.NS", "BHEL.NS", "CUMMINSIND.NS"]},
    "ABB.NS":        {"sector": "Capital Goods",  "peers": ["LT.NS", "SIEMENS.NS", "BHEL.NS"]},
    "BHEL.NS":       {"sector": "Capital Goods",  "peers": ["LT.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS"]},
    "CUMMINSIND.NS": {"sector": "Capital Goods",  "peers": ["SIEMENS.NS", "ABB.NS", "BHEL.NS", "LT.NS"]},
    "THERMAX.NS":    {"sector": "Capital Goods",  "peers": ["LT.NS", "SIEMENS.NS", "CUMMINSIND.NS"]},
    "ADANIPORTS.NS": {"sector": "Infrastructure", "peers": ["LT.NS", "IRB.NS", "GMRINFRA.NS", "POLYCAB.NS"]},
    "POLYCAB.NS":    {"sector": "Electricals",    "peers": ["HAVELLS.NS", "KEI.NS", "RRKABEL.NS", "CUMMINSIND.NS"]},
    "KEI.NS":        {"sector": "Electricals",    "peers": ["POLYCAB.NS", "HAVELLS.NS", "RRKABEL.NS"]},

    # ── Consumer Durables & Retail (India) ──────────────────────────────────
    "TITAN.NS":      {"sector": "Consumer",       "peers": ["TRENT.NS", "KALYANKJIL.NS", "HAVELLS.NS", "VOLTAS.NS"]},
    "TRENT.NS":      {"sector": "Retail",         "peers": ["DMART.NS", "TITAN.NS", "ABFRL.NS", "VEDANTFASH.NS"]},
    "DMART.NS":      {"sector": "Retail",         "peers": ["TRENT.NS", "ABFRL.NS", "SHOPERSTOP.NS"]},
    "HAVELLS.NS":    {"sector": "Consumer Durables","peers": ["TITAN.NS", "VOLTAS.NS", "CROMPTON.NS", "POLYCAB.NS"]},
    "VOLTAS.NS":     {"sector": "Consumer Durables","peers": ["HAVELLS.NS", "BLUESTARCO.NS", "CROMPTON.NS", "DAIKIN"]},
    "BLUESTARCO.NS": {"sector": "Consumer Durables","peers": ["VOLTAS.NS", "HAVELLS.NS", "CROMPTON.NS"]},
    "CROMPTON.NS":   {"sector": "Consumer Durables","peers": ["HAVELLS.NS", "VOLTAS.NS", "VGUARD.NS"]},

    # ── Chemicals (India) ───────────────────────────────────────────────────
    "PIDILITIND.NS": {"sector": "Chemicals",      "peers": ["ATUL.NS", "DEEPAKNTR.NS", "SRF.NS", "NAVINFLUOR.NS"]},
    "DEEPAKNTR.NS":  {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "SRF.NS", "TATACHEM.NS", "ATUL.NS"]},
    "SRF.NS":        {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "NAVINFLUOR.NS", "DEEPAKNTR.NS", "FLUOROCHEM.NS"]},
    "TATACHEM.NS":   {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "ATUL.NS", "DEEPAKNTR.NS"]},
    "ATUL.NS":       {"sector": "Chemicals",      "peers": ["PIDILITIND.NS", "DEEPAKNTR.NS", "SRF.NS"]},

    # ── Telecom (India) ─────────────────────────────────────────────────────
    "BHARTIARTL.NS": {"sector": "Telecom",        "peers": ["VODAFONEIDEA.NS", "TATACOMM.NS", "INDUSTOWER.NS"]},
    "VODAFONEIDEA.NS":{"sector": "Telecom",       "peers": ["BHARTIARTL.NS", "TATACOMM.NS", "INDUSTOWER.NS"]},
    "TATACOMM.NS":   {"sector": "Telecom",        "peers": ["BHARTIARTL.NS", "INDUSTOWER.NS"]},
    "INDUSTOWER.NS": {"sector": "Telecom",        "peers": ["BHARTIARTL.NS", "TATACOMM.NS"]},

    # ── Indian Index & Thematic ETFs ────────────────────────────────────────
    "MONQ50.NS":     {"sector": "Index & Sectoral ETFs", "peers": ["MON100.NS", "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "ITBEES.NS", "MID150BEES.NS"]},
    "MON100.NS":     {"sector": "Index & Sectoral ETFs", "peers": ["MONQ50.NS", "NIFTYBEES.NS", "BANKBEES.NS", "ITBEES.NS", "JUNIORBEES.NS"]},
    "NIFTYBEES.NS":  {"sector": "Index & Sectoral ETFs", "peers": ["BANKBEES.NS", "JUNIORBEES.NS", "MID150BEES.NS", "ITBEES.NS", "MON100.NS"]},
    "BANKBEES.NS":   {"sector": "Index & Sectoral ETFs", "peers": ["NIFTYBEES.NS", "JUNIORBEES.NS", "ITBEES.NS", "CPSEETF.NS", "MID150BEES.NS"]},
    "JUNIORBEES.NS": {"sector": "Index & Sectoral ETFs", "peers": ["NIFTYBEES.NS", "MID150BEES.NS", "BANKBEES.NS", "MONQ50.NS"]},
    "MID150BEES.NS": {"sector": "Index & Sectoral ETFs", "peers": ["JUNIORBEES.NS", "NIFTYBEES.NS", "BANKBEES.NS", "MONQ50.NS"]},
    "ITBEES.NS":     {"sector": "Index & Sectoral ETFs", "peers": ["NIFTYBEES.NS", "BANKBEES.NS", "MON100.NS", "MONQ50.NS"]},
    "GOLDBEES.NS":   {"sector": "Commodity ETFs",        "peers": ["SILVERBEES.NS", "NIFTYBEES.NS", "BANKBEES.NS"]},
    "SILVERBEES.NS": {"sector": "Commodity ETFs",        "peers": ["GOLDBEES.NS", "NIFTYBEES.NS", "BANKBEES.NS"]},
    "CPSEETF.NS":    {"sector": "Index & Sectoral ETFs", "peers": ["NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS"]},

    # ── US Mega-Cap Tech & Platforms ────────────────────────────────────────
    "AAPL":          {"sector": "US Mega-Cap Tech", "peers": ["MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]},
    "MSFT":          {"sector": "US Mega-Cap Tech", "peers": ["AAPL", "GOOGL", "AMZN", "META", "NVDA", "ORCL"]},
    "GOOGL":         {"sector": "US Mega-Cap Tech", "peers": ["MSFT", "AAPL", "META", "AMZN", "NVDA"]},
    "GOOG":          {"sector": "US Mega-Cap Tech", "peers": ["MSFT", "AAPL", "META", "AMZN", "NVDA"]},
    "AMZN":          {"sector": "US Mega-Cap Tech", "peers": ["MSFT", "GOOGL", "AAPL", "META", "WMT"]},
    "META":          {"sector": "US Mega-Cap Tech", "peers": ["GOOGL", "MSFT", "AAPL", "AMZN", "NFLX"]},
    "TSLA":          {"sector": "US Mega-Cap Tech", "peers": ["AAPL", "NVDA", "MSFT", "AMZN"]},
    "NFLX":          {"sector": "US Mega-Cap Tech", "peers": ["DIS", "WBD", "META", "AMZN", "AAPL"]},

    # ── US Semiconductors & AI Hardware ─────────────────────────────────────
    "NVDA":          {"sector": "Semiconductors & AI", "peers": ["AMD", "AVGO", "QCOM", "INTC", "TSM", "ARM", "MU"]},
    "AMD":           {"sector": "Semiconductors & AI", "peers": ["NVDA", "INTC", "AVGO", "QCOM", "ARM", "TSM"]},
    "INTC":          {"sector": "Semiconductors & AI", "peers": ["AMD", "NVDA", "QCOM", "AVGO", "TSM"]},
    "AVGO":          {"sector": "Semiconductors & AI", "peers": ["NVDA", "QCOM", "AMD", "TSM", "TXN"]},
    "QCOM":          {"sector": "Semiconductors & AI", "peers": ["AVGO", "NVDA", "AMD", "INTC", "ARM"]},
    "TSM":           {"sector": "Semiconductors & AI", "peers": ["NVDA", "AMD", "AVGO", "QCOM", "INTC", "ASML"]},
    "ARM":           {"sector": "Semiconductors & AI", "peers": ["NVDA", "QCOM", "AMD", "INTC", "AVGO"]},
    "MU":            {"sector": "Semiconductors & AI", "peers": ["NVDA", "AMD", "INTC", "AVGO"]},

    # ── US Enterprise Software & Cloud ──────────────────────────────────────
    "ORCL":          {"sector": "US Enterprise Software", "peers": ["MSFT", "CRM", "NOW", "PLTR", "ADBE"]},
    "CRM":           {"sector": "US Enterprise Software", "peers": ["ORCL", "NOW", "MSFT", "ADBE", "SNOW"]},
    "NOW":           {"sector": "US Enterprise Software", "peers": ["CRM", "ORCL", "MSFT", "PANW", "CRWD"]},
    "PLTR":          {"sector": "US Enterprise Software", "peers": ["NOW", "CRM", "ORCL", "MSFT", "SNOW"]},
    "ADBE":          {"sector": "US Enterprise Software", "peers": ["CRM", "MSFT", "ORCL", "NOW"]},
    "SNOW":          {"sector": "US Enterprise Software", "peers": ["PLTR", "CRM", "NOW", "ORCL"]},
    "CRWD":          {"sector": "Cybersecurity",          "peers": ["PANW", "FTNT", "NET", "ZS"]},
    "PANW":          {"sector": "Cybersecurity",          "peers": ["CRWD", "FTNT", "NET", "ZS"]},

    # ── US Global Indices & ETFs ────────────────────────────────────────────
    "SPY":           {"sector": "Global & US ETFs", "peers": ["QQQ", "VOO", "IVV", "VTI", "IWM", "DIA"]},
    "QQQ":           {"sector": "Global & US ETFs", "peers": ["SPY", "VOO", "IWM", "VGT", "DIA"]},
    "VOO":           {"sector": "Global & US ETFs", "peers": ["SPY", "IVV", "VTI", "QQQ", "IWM"]},
    "IVV":           {"sector": "Global & US ETFs", "peers": ["SPY", "VOO", "VTI", "QQQ"]},
    "VTI":           {"sector": "Global & US ETFs", "peers": ["VOO", "SPY", "IWM", "QQQ"]},
    "IWM":           {"sector": "Global & US ETFs", "peers": ["SPY", "QQQ", "VOO", "VTI"]},
    "DIA":           {"sector": "Global & US ETFs", "peers": ["SPY", "QQQ", "VOO", "IWM"]},
    "GLD":           {"sector": "Commodity ETFs",   "peers": ["SLV", "IAU", "SPY"]},
    "SLV":           {"sector": "Commodity ETFs",   "peers": ["GLD", "IAU", "SPY"]},
}

# Sector-level archetype bellwethers used for dynamic fallbacks
ARCHETYPE_PEERS = {
    "Energy":            {"sector": "Oil & Gas / Energy",        "peers": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "BPCL.NS", "TATAPOWER.NS"]},
    "IT":                {"sector": "Information Technology",    "peers": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"]},
    "Banking":           {"sector": "Banking",                   "peers": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "BANKBARODA.NS"]},
    "Finance":           {"sector": "Financial Services",        "peers": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PFC.NS", "IRFC.NS"]},
    "Pharma":            {"sector": "Pharma & Healthcare",       "peers": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "MANKIND.NS"]},
    "Auto":              {"sector": "Automobile & Mobility",     "peers": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"]},
    "Metals":            {"sector": "Metals & Mining",           "peers": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS", "SAIL.NS"]},
    "FMCG":              {"sector": "FMCG & Consumer Goods",     "peers": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"]},
    "Consumer Tech":     {"sector": "Internet & Consumer Tech",  "peers": ["ETERNAL.NS", "SWIGGY.NS", "ZOMATO.NS", "NYKAA.NS", "PAYTM.NS", "POLICYBZR.NS"]},
    "Infra":             {"sector": "Infrastructure & Defence",  "peers": ["LT.NS", "HAL.NS", "BEL.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS"]},
    "Telecom":           {"sector": "Telecom Services",          "peers": ["BHARTIARTL.NS", "VODAFONEIDEA.NS", "TATACOMM.NS", "INDUSTOWER.NS"]},
    "ETF":               {"sector": "Index & Sectoral ETFs",     "peers": ["NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "MON100.NS", "MONQ50.NS", "ITBEES.NS"]},
    "Global":            {"sector": "US Mega-Cap Tech",          "peers": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]},
}

# Universal market fallbacks
DEFAULT_INDIAN_PEERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "LT.NS"]
DEFAULT_US_PEERS     = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]


def get_peers(ticker: str) -> dict:
    """
    Return canonical sector name and curated peer list for any given ticker.
    Guarantees that a valid sector and non-empty peer list is ALWAYS returned.
    """
    if not ticker:
        return {"sector": "General Equities", "peers": DEFAULT_INDIAN_PEERS, "found": False}

    ticker_clean = ticker.strip().upper()
    raw_sym = ticker_clean.replace(".NS", "").replace(".BO", "").replace("^", "")

    # 1. Exact match in SECTOR_PEERS
    if ticker_clean in SECTOR_PEERS:
        data = SECTOR_PEERS[ticker_clean]
        peers = [p for p in data["peers"] if p != ticker_clean]
        return {"sector": data["sector"], "peers": peers, "found": True}

    # 2. Match without exchange suffix (or vice-versa)
    if raw_sym in SECTOR_PEERS:
        data = SECTOR_PEERS[raw_sym]
        peers = [p for p in data["peers"] if p != ticker_clean and p != raw_sym]
        return {"sector": data["sector"], "peers": peers, "found": True}

    ns_candidate = f"{raw_sym}.NS"
    if ns_candidate in SECTOR_PEERS:
        data = SECTOR_PEERS[ns_candidate]
        peers = [p for p in data["peers"] if p != ticker_clean and p != ns_candidate]
        return {"sector": data["sector"], "peers": peers, "found": True}

    # 3. Check ETF heuristics
    is_etf = any(k in raw_sym for k in [
        "BEES", "ETF", "MON100", "MONQ50", "MAFANG", "CPSE", "GOLD", "SILVER",
        "MID150", "JUNIOR", "NV20", "Q50", "SPY", "QQQ", "VOO", "IVV", "VTI"
    ])
    if is_etf:
        if ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO") or "BEES" in raw_sym:
            peers = [p for p in ARCHETYPE_PEERS["ETF"]["peers"] if p != ticker_clean]
            return {"sector": "Index & Sectoral ETFs", "peers": peers, "found": True}
        else:
            peers = [p for p in ARCHETYPE_PEERS["Global"]["peers"] if p != ticker_clean]
            return {"sector": "Global & US ETFs", "peers": ["SPY", "QQQ", "VOO", "IWM", "DIA"], "found": True}

    # 4. Check master SECTOR_MAP from ticker_manager
    try:
        from services.ticker_manager import SECTOR_MAP
        mapped_sec = SECTOR_MAP.get(ticker_clean) or SECTOR_MAP.get(raw_sym)
        if mapped_sec and mapped_sec in ARCHETYPE_PEERS:
            arch = ARCHETYPE_PEERS[mapped_sec]
            peers = [p for p in arch["peers"] if p != ticker_clean and p != raw_sym]
            return {"sector": arch["sector"], "peers": peers, "found": True}
    except Exception:
        pass

    # 5. Check fundamental quote profile from Yahoo Finance
    try:
        from yf_client import get_info
        inf = get_info(ticker_clean)
        yf_sec = inf.get("sector")
        yf_ind = inf.get("industry")
        if yf_sec or yf_ind:
            label = yf_ind or yf_sec
            is_us = not (ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO"))
            # Match to nearest archetype
            s_low = f"{yf_sec} {yf_ind}".lower()
            if "technol" in s_low or "software" in s_low:
                peers = [p for p in (ARCHETYPE_PEERS["Global"]["peers"] if is_us else ARCHETYPE_PEERS["IT"]["peers"]) if p != ticker_clean]
                return {"sector": label, "peers": peers, "found": True}
            elif "bank" in s_low or "financial" in s_low:
                peers = [p for p in ARCHETYPE_PEERS["Banking"]["peers"] if p != ticker_clean]
                return {"sector": label, "peers": peers, "found": True}
            elif "defense" in s_low or "aerospace" in s_low:
                peers = [p for p in SECTOR_PEERS["BEL.NS"]["peers"] if p != ticker_clean]
                return {"sector": "Defence & Aerospace", "peers": peers, "found": True}
            elif "pharma" in s_low or "health" in s_low:
                peers = [p for p in ARCHETYPE_PEERS["Pharma"]["peers"] if p != ticker_clean]
                return {"sector": label, "peers": peers, "found": True}
            elif "energy" in s_low or "oil" in s_low or "power" in s_low:
                peers = [p for p in ARCHETYPE_PEERS["Energy"]["peers"] if p != ticker_clean]
                return {"sector": label, "peers": peers, "found": True}
            elif "auto" in s_low:
                peers = [p for p in ARCHETYPE_PEERS["Auto"]["peers"] if p != ticker_clean]
                return {"sector": label, "peers": peers, "found": True}
            else:
                fallback_pool = DEFAULT_US_PEERS if is_us else DEFAULT_INDIAN_PEERS
                peers = [p for p in fallback_pool if p != ticker_clean]
                return {"sector": label, "peers": peers, "found": True}
    except Exception:
        pass

    # 6. Universal Fallback
    is_us = not (ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO") or ticker_clean.startswith("^"))
    if is_us:
        peers = [p for p in DEFAULT_US_PEERS if p != ticker_clean]
        return {"sector": "US Market Equities", "peers": peers, "found": True}
    else:
        peers = [p for p in DEFAULT_INDIAN_PEERS if p != ticker_clean]
        return {"sector": "Nifty Benchmark Equities", "peers": peers, "found": True}


def get_all_sector_members(ticker: str) -> list[str]:
    """Return all tickers in the same sector as the given ticker."""
    info = get_peers(ticker)
    sector = info.get("sector", "")
    peers = info.get("peers", [])
    members = set(peers)

    # Also include any matching tickers in curated SECTOR_PEERS
    for sym, val in SECTOR_PEERS.items():
        if val.get("sector") == sector and sym != ticker.upper().strip():
            members.add(sym)

    return sorted(list(members))
