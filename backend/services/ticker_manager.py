import os
import json
import requests
import io
import pandas as pd

# Global ticker list and status
TICKER_LIST = []
_ticker_list_loaded = False

_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_SERVICE_DIR, "..", ".."))

def _load_local_tickers():
    """Attempts to load tickers from bundled master JSON files."""
    candidates = [
        os.path.join(REPO_ROOT, "backend", "data", "tickers.json"),
        os.path.join(REPO_ROOT, "backend", "tickers.json"),
        os.path.join(REPO_ROOT, "frontend", "public", "tickers.json"),
        os.path.join(_SERVICE_DIR, "..", "data", "tickers.json"),
        os.path.join(_SERVICE_DIR, "..", "tickers.json"),
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

# Master SECTOR_MAP mapping symbol and raw_symbol to sector name for all instruments
SECTOR_MAP = {}
for item in TICKER_LIST:
    sym = item.get("symbol", "")
    raw = sym.replace(".NS", "").replace(".BO", "").replace("^", "")
    sec = item.get("sector", "Others")
    SECTOR_MAP[sym] = sec
    SECTOR_MAP[raw] = sec

import time
import threading
from datetime import datetime

_sync_lock = threading.Lock()
_periodic_sync_started = False

def get_sync_metadata():
    """Reads sync metadata file, or returns default if not found."""
    candidates = [
        os.path.join(REPO_ROOT, "backend", "data", "sync_metadata.json"),
        os.path.join(_SERVICE_DIR, "..", "data", "sync_metadata.json"),
        "backend/data/sync_metadata.json"
    ]
    for p in candidates:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            try:
                with open(abs_p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {
        "last_synced_at": datetime.now().isoformat(),
        "last_synced_epoch": time.time(),
        "last_added_count": 0,
        "total_instruments": len(TICKER_LIST),
        "sync_interval_hours": 48,
        "last_status": "initialized"
    }

def save_sync_metadata(data):
    """Persists sync metadata atomically to prevent corrupted reads."""
    target = os.path.join(REPO_ROOT, "backend", "data", "sync_metadata.json")
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp_target = f"{target}.tmp"
        with open(tmp_target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_target, target)
    except Exception as e:
        print(f"[TickerSync] Error saving sync metadata: {e}")

def sync_latest_listings_if_needed(force: bool = False, min_interval_hours: int = 48) -> dict:
    """
    Checks if min_interval_hours (default 48 hours / alternate day) has passed since
    the last check. If so, or if force=True, downloads the latest exchange lists,
    adds any newly listed stocks/ETFs/SMEs/REITs to TICKER_LIST and SECTOR_MAP,
    and updates JSON files on disk.
    """
    global TICKER_LIST, SECTOR_MAP, _ticker_list_loaded

    with _sync_lock:
        meta = get_sync_metadata()
        last_epoch = meta.get("last_synced_epoch", 0)
        now_epoch = time.time()
        elapsed_hours = (now_epoch - last_epoch) / 3600.0

        # Debounce rapid force triggers (< 30 seconds) to avoid spamming exchange endpoints
        if force and (now_epoch - last_epoch) < 30 and meta.get("last_status") == "success":
            return {
                "synced": False,
                "reason": "Sync completed moments ago (< 30s). Throttled to prevent redundant exchange traffic.",
                "last_synced_at": meta.get("last_synced_at"),
                "total_instruments": len(TICKER_LIST),
                "remaining_hours": round(min_interval_hours, 1)
            }

        if not force and elapsed_hours < min_interval_hours:
            remaining_hours = round(min_interval_hours - elapsed_hours, 1)
            return {
                "synced": False,
                "reason": f"Already up to date. Next scheduled sync in {remaining_hours}h.",
                "last_synced_at": meta.get("last_synced_at"),
                "total_instruments": len(TICKER_LIST),
                "remaining_hours": remaining_hours
            }

        print(f"[TickerSync] Running automated stock sync (elapsed: {elapsed_hours:.1f}h)...")
        prev_count = len(TICKER_LIST)
        prev_symbols = {item.get("symbol") for item in TICKER_LIST}

        try:
            import sys
            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)
            from scripts.build_master_tickers import main as run_ticker_builder
            
            build_result = run_ticker_builder()

            # Reload updated tickers into RAM
            fresh_list = _load_local_tickers()
            new_symbols = []
            if fresh_list and len(fresh_list) > 0:
                fresh_symbols = {item.get("symbol") for item in fresh_list}
                new_symbols = sorted(list(fresh_symbols - prev_symbols))

                # In-place update guarantees any references held by callers are updated
                TICKER_LIST.clear()
                TICKER_LIST.extend(fresh_list)
                _ticker_list_loaded = True
                
                # Rebuild SECTOR_MAP in place
                SECTOR_MAP.clear()
                for item in TICKER_LIST:
                    sym = item.get("symbol", "")
                    raw = sym.replace(".NS", "").replace(".BO", "").replace("^", "")
                    sec = item.get("sector", "Others")
                    SECTOR_MAP[sym] = sec
                    SECTOR_MAP[raw] = sec

            added_count = len(new_symbols)
            now_iso = datetime.now().isoformat()
            
            meta.update({
                "last_synced_at": now_iso,
                "last_synced_epoch": now_epoch,
                "last_added_count": added_count,
                "total_instruments": len(TICKER_LIST),
                "sync_interval_hours": min_interval_hours,
                "last_status": "success",
                "new_symbols_sample": new_symbols[:10]
            })
            save_sync_metadata(meta)
            
            print(f"[TickerSync] Auto-sync complete! Instruments in catalog: {len(TICKER_LIST)} (+{added_count} new)")
            return {
                "synced": True,
                "new_added": added_count,
                "new_symbols": new_symbols[:20],
                "total_instruments": len(TICKER_LIST),
                "synced_at": now_iso
            }
        except Exception as e:
            print(f"[TickerSync] Error during auto-sync: {e}")
            meta["last_status"] = f"error: {str(e)}"
            save_sync_metadata(meta)
            return {
                "synced": False,
                "error": str(e),
                "total_instruments": len(TICKER_LIST)
            }

def start_periodic_ticker_sync(interval_hours: int = 48, check_frequency_seconds: int = 3600):
    """
    Spawns a background daemon thread that checks every hour whether
    the alternate day (48h) sync interval has elapsed, automatically updating the catalog.
    """
    global _periodic_sync_started
    if _periodic_sync_started:
        return None
    _periodic_sync_started = True

    def _worker():
        time.sleep(5)
        print(f"[TickerSync] Background auto-sync scheduler initialized (cadence: every {interval_hours}h).")
        while True:
            try:
                sync_latest_listings_if_needed(force=False, min_interval_hours=interval_hours)
            except Exception as e:
                print(f"[TickerSync] Background worker error: {e}")
            time.sleep(check_frequency_seconds)

    thread = threading.Thread(target=_worker, name="TickerAutoSyncWorker", daemon=True)
    thread.start()
    return thread

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

