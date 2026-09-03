import requests
import csv
import io
import json
import os
import re

def main():
    print("=" * 60)
    print("StockIQ Pro — Master Ticker & ETF Universe Generator")
    print("=" * 60)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    bse_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Referer': 'https://www.bseindia.com/'
    }

    # 1. Fetch Nifty Total Market for high-fidelity industry mapping
    print("Step 1: Downloading Nifty Total Market industry taxonomy...")
    tm_map = {}
    try:
        tm_res = requests.get('https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv', headers=headers, timeout=15)
        if tm_res.ok:
            for row in csv.DictReader(io.StringIO(tm_res.text)):
                s = row.get('Symbol', '').strip()
                ind = row.get('Industry', '').strip()
                if s and ind:
                    tm_map[s] = ind
        print(f"  Mapped {len(tm_map)} stocks from Nifty Total Market.")
    except Exception as e:
        print(f"  Warning: Could not load Total Market index: {e}")

    IND_TO_SECTOR = {
        'Financial Services': 'Finance',
        'Automobile and Auto Components': 'Auto',
        'Information Technology': 'IT',
        'Fast Moving Consumer Goods': 'FMCG',
        'Healthcare': 'Pharma',
        'Oil Gas & Consumable Fuels': 'Energy',
        'Metals & Mining': 'Metals',
        'Construction': 'Infra',
        'Construction Materials': 'Infra',
        'Power': 'Energy',
        'Telecommunication': 'Telecom',
        'Consumer Durables': 'Consumer Tech',
        'Chemicals': 'Pharma',
        'Services': 'Others',
        'Capital Goods': 'Infra',
        'Textiles': 'FMCG',
        'Utilities': 'Energy',
        'Realty': 'Infra',
        'Consumer Services': 'Consumer Tech',
        'Media, Entertainment & Publication': 'Others',
        'Diversified': 'Others',
    }

    KNOWN_SECTOR_MAP = {
        # Banking
        'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
        'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking', 'FEDERALBNK': 'Banking', 'BANDHANBNK': 'Banking',
        'IDFCFIRSTB': 'Banking', 'PNB': 'Banking', 'BANKBARODA': 'Banking', 'CANBK': 'Banking',
        'UNIONBANK': 'Banking', 'IOB': 'Banking', 'CENTRALBK': 'Banking', 'MAHABANK': 'Banking',
        'UCOBANK': 'Banking', 'BANKINDIA': 'Banking', 'YESBANK': 'Banking', 'RBLBANK': 'Banking',
        'KARURVYSYA': 'Banking', 'CUB': 'Banking', 'DCBBANK': 'Banking', 'SOUTHBANK': 'Banking',
        'J&KBANK': 'Banking', 'PSB': 'Banking', 'AUBANK': 'Banking',
        # IT
        'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT', 'TECHM': 'IT', 'LTIM': 'IT',
        'MPHASIS': 'IT', 'COFORGE': 'IT', 'PERSISTENT': 'IT', 'TATAELXSI': 'IT', 'KPITTECH': 'IT',
        'CYIENT': 'IT', 'SONATSOFTW': 'IT', 'ZENSARTECH': 'IT', 'MASTEK': 'IT', 'BIRLASOFT': 'IT',
        # Energy
        'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy', 'IOC': 'Energy', 'NTPC': 'Energy',
        'ADANIGREEN': 'Energy', 'POWERGRID': 'Energy', 'TATAPOWER': 'Energy', 'GAIL': 'Energy',
        'PETRONET': 'Energy', 'HINDPETRO': 'Energy', 'MGL': 'Energy', 'IGL': 'Energy', 'GUJGASLTD': 'Energy',
        'CESC': 'Energy', 'TORNTPOWER': 'Energy', 'JSWENERGY': 'Energy', 'SJVN': 'Energy', 'NHPC': 'Energy',
        'OIL': 'Energy', 'MRPL': 'Energy', 'CHENNPETRO': 'Energy', 'ADANIPOWER': 'Energy',
        # Auto
        'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto', 'BAJAJ-AUTO': 'Auto', 'HEROMOTOCO': 'Auto',
        'EICHERMOT': 'Auto', 'ASHOKLEY': 'Auto', 'TVSMOTOR': 'Auto', 'BOSCHLTD': 'Auto', 'BHARATFORG': 'Auto',
        'MOTHERSON': 'Auto', 'APOLLOTYRE': 'Auto', 'MRF': 'Auto', 'CEATLTD': 'Auto', 'EXIDEIND': 'Auto',
        # Pharma
        'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'DIVISLAB': 'Pharma',
        'BIOCON': 'Pharma', 'AUROPHARMA': 'Pharma', 'LUPIN': 'Pharma', 'TORNTPHARM': 'Pharma',
        'ALKEM': 'Pharma', 'IPCALAB': 'Pharma', 'GLENMARK': 'Pharma', 'ZYDUSLIFE': 'Pharma',
        'APOLLOHOSP': 'Pharma', 'MAXHEALTH': 'Pharma', 'FORTIS': 'Pharma', 'SYNGENE': 'Pharma',
        # FMCG
        'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
        'BRITANNIA': 'FMCG', 'GODREJCP': 'FMCG', 'COLPAL': 'FMCG', 'EMAMILTD': 'FMCG', 'RADICO': 'FMCG',
        'VBL': 'FMCG', 'TATACONSUM': 'FMCG',
        # Metals
        'JSWSTEEL': 'Metals', 'TATASTEEL': 'Metals', 'HINDALCO': 'Metals', 'COALINDIA': 'Metals',
        'VEDL': 'Metals', 'NMDC': 'Metals', 'SAIL': 'Metals', 'NATIONALUM': 'Metals', 'JSL': 'Metals',
        # Consumer Tech
        'ZOMATO': 'Consumer Tech', 'ETERNAL': 'Consumer Tech', 'NYKAA': 'Consumer Tech', 'PAYTM': 'Consumer Tech',
        'POLICYBZR': 'Consumer Tech', 'DELHIVERY': 'Consumer Tech', 'CARTRADE': 'Consumer Tech',
        'INDIAMART': 'Consumer Tech', 'IRCTC': 'Consumer Tech', 'EASEMYTRIP': 'Consumer Tech', 'MAPMYINDIA': 'Consumer Tech',
        'DMART': 'Consumer Tech', 'TRENT': 'Consumer Tech', 'NAUKRI': 'Consumer Tech',
    }

    def resolve_sector(sym, name, industry=''):
        raw = sym.upper().replace('.NS', '').replace('.BO', '')
        if raw in KNOWN_SECTOR_MAP:
            return KNOWN_SECTOR_MAP[raw]
        if raw in tm_map:
            mapped_ind = tm_map[raw]
            if 'Bank' in mapped_ind:
                return 'Banking'
            return IND_TO_SECTOR.get(mapped_ind, 'Others')
        
        n = (name + ' ' + industry).lower()
        if 'bees' in sym.lower() or 'etf' in sym.lower() or 'etf' in n or 'bees' in n or 'mutual fund' in n or 'index fund' in n:
            return 'ETF'
        if 'bank' in n and 'financial' not in n and 'investment' not in n: return 'Banking'
        if any(k in n for k in ['software', 'technology', 'infotech', 'consultancy services', 'systems', 'digital solutions']): return 'IT'
        if any(k in n for k in ['pharma', 'laborator', 'healthcare', 'hospital', 'biotech', 'drug', 'medic']): return 'Pharma'
        if any(k in n for k in ['automobile', 'auto ', 'motors', 'tyre', 'automotive', 'axle']): return 'Auto'
        if any(k in n for k in ['food', 'beverag', 'sugar', 'brewer', 'distiller', 'dairy', 'fmcg', 'tea', 'coffee']): return 'FMCG'
        if any(k in n for k in ['power', 'energy', 'oil & gas', 'oil gas', 'petro', 'solar', 'renewable', 'fuel']): return 'Energy'
        if any(k in n for k in ['construct', 'cement', 'infra', 'engineering', 'realty', 'real estate', 'housing dev']): return 'Infra'
        if any(k in n for k in ['metal', 'steel', 'iron', 'aluminium', 'copper', 'mining', 'minerals', 'ore', 'zinc']): return 'Metals'
        if any(k in n for k in ['telecom', 'cellular', 'telecommunication', 'broadband']): return 'Telecom'
        if any(k in n for k in ['finance', 'financial', 'insurance', 'housing fin', 'capital', 'investment', 'nbfc', 'securities', 'credit']): return 'Finance'
        if any(k in n for k in ['online', 'digital', 'tech', 'retail', 'platform', 'e-commerce']): return 'Consumer Tech'
        return 'Others'

    master_list = []
    seen_symbols = set()
    isin_to_symbol = {}

    # =========================================================================
    # 2. Benchmark Indices (NSE, BSE, US / Global)
    # =========================================================================
    print("Step 2: Adding Benchmark Indices & Global Assets...")
    indices = [
        {"symbol": "^NSEI",       "name": "NIFTY 50 (NSE Benchmark Index)", "sector": "Indices", "type": "Index"},
        {"symbol": "^NSEBANK",    "name": "NIFTY BANK (Banking Sector Benchmark Index)", "sector": "Indices", "type": "Index"},
        {"symbol": "^BSESN",      "name": "S&P BSE SENSEX (BSE 30 Benchmark Index)", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXIT",      "name": "NIFTY IT Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXAUTO",    "name": "NIFTY AUTO Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXFMCG",    "name": "NIFTY FMCG Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXPHARMA",  "name": "NIFTY PHARMA Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXMETAL",   "name": "NIFTY METAL Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXREALTY",  "name": "NIFTY REALTY Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXENERGY",  "name": "NIFTY ENERGY Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXINFRA",   "name": "NIFTY INFRA Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CNXMEDIA",   "name": "NIFTY MEDIA Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^CRSLDX",     "name": "NIFTY 500 Broad Market Index", "sector": "Indices", "type": "Index"},
        {"symbol": "BSE-BANK.BO", "name": "BSE BANKEX Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^IXIC",       "name": "NASDAQ Composite Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^GSPC",       "name": "S&P 500 Index", "sector": "Indices", "type": "Index"},
        {"symbol": "^DJI",        "name": "Dow Jones Industrial Average", "sector": "Indices", "type": "Index"},
    ]
    for idx in indices:
        if idx["symbol"] not in seen_symbols:
            master_list.append(idx)
            seen_symbols.add(idx["symbol"])
    print(f"  Added {len(indices)} benchmark indices.")

    # =========================================================================
    # 3. Top Global / US Equities
    # =========================================================================
    print("Step 3: Adding Global / US Equities...")
    global_assets = [
        {"symbol": "AAPL",  "name": "Apple Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "NVDA",  "name": "NVIDIA Corporation", "sector": "Global", "type": "Global Equity"},
        {"symbol": "MSFT",  "name": "Microsoft Corporation", "sector": "Global", "type": "Global Equity"},
        {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)", "sector": "Global", "type": "Global Equity"},
        {"symbol": "AMZN",  "name": "Amazon.com Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "TSLA",  "name": "Tesla Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "META",  "name": "Meta Platforms Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "AMD",   "name": "Advanced Micro Devices Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "NFLX",  "name": "Netflix Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "BRK-B", "name": "Berkshire Hathaway Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "PLTR",  "name": "Palantir Technologies Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "ARM",   "name": "Arm Holdings plc", "sector": "Global", "type": "Global Equity"},
        {"symbol": "SMCI",  "name": "Super Micro Computer Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "COIN",  "name": "Coinbase Global Inc.", "sector": "Global", "type": "Global Equity"},
        {"symbol": "SPY",   "name": "SPDR S&P 500 ETF Trust", "sector": "Global", "type": "Global ETF"},
        {"symbol": "QQQ",   "name": "Invesco QQQ Trust (NASDAQ 100)", "sector": "Global", "type": "Global ETF"},
    ]
    for g in global_assets:
        if g["symbol"] not in seen_symbols:
            master_list.append(g)
            seen_symbols.add(g["symbol"])
    print(f"  Added {len(global_assets)} global assets.")

    # =========================================================================
    # 4. NSE Exchange Traded Funds (ETFs)
    # =========================================================================
    print("Step 4: Downloading All NSE ETFs...")
    try:
        nse_etf_res = requests.get('https://archives.nseindia.com/content/equities/eq_etfseclist.csv', headers=headers, timeout=15)
        if nse_etf_res.ok:
            reader = csv.DictReader(io.StringIO(nse_etf_res.text))
            reader.fieldnames = [f.strip() for f in reader.fieldnames]
            etf_count = 0
            for row in reader:
                sym_raw = row.get('Symbol', '').strip()
                if not sym_raw:
                    continue
                symbol = f"{sym_raw}.NS"
                sec_name = row.get('SecurityName', '').strip()
                underlying = row.get('Underlying', '').strip()
                isin = row.get('ISINNumber', '').strip()
                
                # Format a user-friendly ETF name
                clean_name = sec_name
                if underlying and underlying.lower() not in sec_name.lower():
                    clean_name = f"{sec_name} ({underlying})"

                item = {
                    "symbol": symbol,
                    "name": clean_name,
                    "sector": "ETF",
                    "type": "ETF",
                }
                if isin:
                    item["isin"] = isin
                    isin_to_symbol[isin] = symbol
                
                if symbol not in seen_symbols:
                    master_list.append(item)
                    seen_symbols.add(symbol)
                    etf_count += 1
            print(f"  Successfully added {etf_count} NSE ETFs.")
        else:
            print(f"  Warning: ETF fetch returned status {nse_etf_res.status_code}")
    except Exception as e:
        print(f"  Error fetching NSE ETFs: {e}")

    # =========================================================================
    # 5. NSE Equities (Mainboard + Series EQ, BE, BZ)
    # =========================================================================
    print("Step 5: Downloading All NSE Equities...")
    try:
        nse_eq_res = requests.get('https://archives.nseindia.com/content/equities/EQUITY_L.csv', headers=headers, timeout=15)
        if nse_eq_res.ok:
            reader = csv.DictReader(io.StringIO(nse_eq_res.text))
            reader.fieldnames = [f.strip() for f in reader.fieldnames]
            eq_count = 0
            for row in reader:
                sym_raw = row.get('SYMBOL', '').strip()
                if not sym_raw:
                    continue
                symbol = f"{sym_raw}.NS"
                name = row.get('NAME OF COMPANY', '').strip()
                isin = row.get('ISIN NUMBER', '').strip()
                series = row.get('SERIES', '').strip()

                # Handle popular corporate renames for search discoverability
                if sym_raw == 'ETERNAL':
                    name = "Eternal Limited (Formerly Zomato Limited)"
                elif sym_raw == 'LTIM':
                    name = "LTIMindtree Limited (Formerly L&T Infotech & Mindtree)"
                elif sym_raw == 'ZYDUSLIFE':
                    name = "Zydus Lifesciences Limited (Formerly Cadila Healthcare)"
                elif sym_raw == 'MOTHERSON':
                    name = "Samvardhana Motherson International Ltd (Formerly Motherson Sumi)"
                elif sym_raw == 'LTF':
                    name = "L&T Finance Holdings Limited"

                sector = resolve_sector(sym_raw, name)

                item = {
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "type": "Equity",
                }
                if isin:
                    item["isin"] = isin
                    isin_to_symbol[isin] = symbol

                if symbol not in seen_symbols:
                    master_list.append(item)
                    seen_symbols.add(symbol)
                    eq_count += 1
            print(f"  Successfully added {eq_count} NSE Equities.")
        else:
            print(f"  Warning: NSE Equities fetch returned status {nse_eq_res.status_code}")
    except Exception as e:
        print(f"  Error fetching NSE Equities: {e}")

    # =========================================================================
    # 6. BSE Equities & Exclusively Listed BSE Companies
    # =========================================================================
    print("Step 6: Downloading Active BSE Equities...")
    try:
        bse_res = requests.get(
            'https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&Industry=&Segment=Equity&Status=Active',
            headers=bse_headers,
            timeout=15
        )
        if bse_res.ok:
            bse_scrips = bse_res.json()
            bse_added = 0
            for row in bse_scrips:
                scrip_id = row.get('scrip_id', '').strip()
                scrip_cd = row.get('SCRIP_CD', '').strip()
                scrip_name = row.get('Scrip_Name', '').strip()
                issuer_name = row.get('Issuer_Name', '').strip() or scrip_name
                isin = row.get('ISIN_NUMBER', '').strip()

                if not scrip_id and not scrip_cd:
                    continue

                # Primary symbol on BSE
                primary_bse_sym = f"{scrip_id}.BO" if scrip_id else f"{scrip_cd}.BO"

                # Check if this company is already in NSE list by ISIN
                if isin and isin in isin_to_symbol:
                    # Company is dual-listed on NSE and BSE.
                    # We can also register the .BO ticker if distinct
                    if primary_bse_sym not in seen_symbols:
                        nse_peer = isin_to_symbol[isin]
                        item = {
                            "symbol": primary_bse_sym,
                            "name": f"{issuer_name} (BSE)",
                            "sector": resolve_sector(scrip_id, issuer_name),
                            "type": "Equity (BSE)",
                            "bse_code": scrip_cd
                        }
                        if isin:
                            item["isin"] = isin
                        master_list.append(item)
                        seen_symbols.add(primary_bse_sym)
                        bse_added += 1
                else:
                    # Company is EXCLUSIVELY listed on BSE!
                    if primary_bse_sym not in seen_symbols:
                        item = {
                            "symbol": primary_bse_sym,
                            "name": issuer_name,
                            "sector": resolve_sector(scrip_id, issuer_name),
                            "type": "Equity (BSE)",
                            "bse_code": scrip_cd
                        }
                        if isin:
                            item["isin"] = isin
                            isin_to_symbol[isin] = primary_bse_sym
                        master_list.append(item)
                        seen_symbols.add(primary_bse_sym)
                        bse_added += 1
            print(f"  Successfully added {bse_added} BSE Equities.")
        else:
            print(f"  Warning: BSE API returned status {bse_res.status_code}")
    except Exception as e:
        print(f"  Error fetching BSE Equities: {e}")

    # =========================================================================
    # 7. Add Former / Historic Aliases for Instant Search Discoverability
    # =========================================================================
    print("Step 7: Adding Historical / Rename Aliases...")
    aliases = [
        {"symbol": "ETERNAL.NS", "name": "Zomato Limited (Now Eternal Limited)", "sector": "Consumer Tech", "type": "Equity Alias"},
        {"symbol": "LTIM.NS",    "name": "Mindtree Limited (Merged into LTIMindtree)", "sector": "IT", "type": "Equity Alias"},
        {"symbol": "HDFCBANK.NS","name": "HDFC Ltd (Housing Development Finance Corp - Merged into HDFC Bank)", "sector": "Banking", "type": "Equity Alias"},
        {"symbol": "IDFCFIRSTB.NS", "name": "IDFC Limited (Merged into IDFC FIRST Bank)", "sector": "Banking", "type": "Equity Alias"},
        {"symbol": "ZYDUSLIFE.NS",  "name": "Cadila Healthcare Limited (Now Zydus Lifesciences)", "sector": "Pharma", "type": "Equity Alias"},
        {"symbol": "MOTHERSON.NS",  "name": "Motherson Sumi Systems Limited (Now Samvardhana Motherson)", "sector": "Auto", "type": "Equity Alias"},
        {"symbol": "LTF.NS",        "name": "L&T Finance Holdings (L&TFH)", "sector": "Finance", "type": "Equity Alias"},
    ]
    alias_count = 0
    for a in aliases:
        # Check if already added or add as search entry
        master_list.append(a)
        alias_count += 1
    print(f"  Added {alias_count} search alias entries.")

    # Sort master list: Indices & Global first, then ETFs, then equities
    # Within each category, NSE (.NS) ranks before BSE (.BO)
    def sort_key(item):
        sec = item.get("sector", "")
        sym = item.get("symbol", "")
        exch_prio = 1 if sym.endswith(".BO") else 0
        if sec == "Indices":
            return (0, exch_prio, sym)
        if sec == "Global":
            return (1, exch_prio, sym)
        if sec == "ETF":
            return (2, exch_prio, sym)
        return (3, exch_prio, sym)

    master_list.sort(key=sort_key)

    total_count = len(master_list)
    print("=" * 60)
    print(f"Master Instrument Compilation Complete: {total_count} TOTAL INSTRUMENTS")
    
    # Calculate Sector Distribution
    sector_counts = {}
    for item in master_list:
        sec = item.get("sector", "Others")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
    
    print("Sector Breakdown:")
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sec:15}: {cnt}")
    print("=" * 60)

    # 8. Write to Targets:
    # A. backend/data/tickers.json & backend/tickers.json
    # B. frontend/public/tickers.json

    targets = [
        "backend/data/tickers.json",
        "backend/tickers.json",
        "frontend/public/tickers.json"
    ]

    for target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(master_list, f, indent=2, ensure_ascii=False)
        size_kb = os.path.getsize(target) / 1024
        print(f"Wrote {total_count} records to {target} ({size_kb:.1f} KB)")

    print("SUCCESS: All stocks, ETFs, indices, and global equities generated successfully!")

if __name__ == "__main__":
    main()
