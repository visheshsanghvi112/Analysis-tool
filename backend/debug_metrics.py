"""
Full integration test for all 3 peer endpoints.
Tests: real data, edge cases, winner logic, score validity, sector coverage.
"""
import sys
sys.path.insert(0, 'd:/Analysis-tool/backend')
import requests
import json

BASE = "http://localhost:8000"

PASS = []
FAIL = []

def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  --  {detail}")

def get(path):
    r = requests.get(BASE + path, timeout=30)
    return r.status_code, r.json() if r.status_code == 200 else {}

print("\n" + "="*60)
print("  TEST 1: /api/peers — known tickers")
print("="*60)

for ticker, expected_sector in [
    ("RELIANCE.NS", "Oil & Gas"),
    ("TCS.NS", "IT"),
    ("HDFCBANK.NS", "Banking"),
    ("SUNPHARMA.NS", "Pharma"),
    ("TATAMOTORS.NS", "Auto"),
    ("ZOMATO.NS", "Internet"),
]:
    code, d = get(f"/api/peers?ticker={ticker}")
    check(f"{ticker} status=200", code == 200)
    check(f"{ticker} found=True", d.get("found") == True, f"found={d.get('found')}")
    check(f"{ticker} sector correct", d.get("sector") == expected_sector, f"got={d.get('sector')}")
    check(f"{ticker} has peers", len(d.get("peers", [])) > 0, f"peers={d.get('peers')}")
    print(f"       -> sector={d.get('sector')}, peers={d.get('peers')}")

print("\n" + "="*60)
print("  TEST 2: /api/peers — unknown ticker (edge case)")
print("="*60)

code, d = get("/api/peers?ticker=FAKESTK.NS")
check("unknown ticker status=200", code == 200)
check("unknown ticker found=False", d.get("found") == False, f"found={d.get('found')}")
check("unknown ticker empty peers", d.get("peers") == [], f"peers={d.get('peers')}")
print(f"       -> {d}")

print("\n" + "="*60)
print("  TEST 3: /api/peer-compare — HDFCBANK vs ICICIBANK")
print("="*60)

code, d = get("/api/peer-compare?ticker=HDFCBANK.NS&peer=ICICIBANK.NS")
check("peer-compare status=200", code == 200)
m_a = d.get("metrics_a", {})
m_b = d.get("metrics_b", {})
w   = d.get("winners", {})

# Metrics completeness
for field in ["current_price", "ret_1m", "ret_3m", "ret_6m", "ret_1y", "sharpe", "annual_vol", "rsi"]:
    check(f"metrics_a.{field} not null", m_a.get(field) is not None, f"val={m_a.get(field)}")
    check(f"metrics_b.{field} not null", m_b.get(field) is not None, f"val={m_b.get(field)}")

# RSI range
rsi_a, rsi_b = m_a.get("rsi", 0), m_b.get("rsi", 0)
check("rsi_a in [0,100]", 0 <= rsi_a <= 100, f"rsi_a={rsi_a}")
check("rsi_b in [0,100]", 0 <= rsi_b <= 100, f"rsi_b={rsi_b}")

# Sharpe is finite
sharpe_a = m_a.get("sharpe")
check("sharpe_a is float", isinstance(sharpe_a, float), f"sharpe={sharpe_a}")

# Winner logic
check("winners has ret_1m", "ret_1m" in w)
check("winner is one of the tickers or None", w.get("ret_1m") in ["HDFCBANK.NS","ICICIBANK.NS","tie",None])
check("annual_vol winner valid (lower is better)", w.get("annual_vol") in ["HDFCBANK.NS","ICICIBANK.NS","tie",None])

print(f"\n  HDFCBANK: price={m_a.get('current_price')}, 1Y={m_a.get('ret_1y')}%, sharpe={m_a.get('sharpe')}, rsi={m_a.get('rsi')}")
print(f"  ICICIBANK: price={m_b.get('current_price')}, 1Y={m_b.get('ret_1y')}%, sharpe={m_b.get('sharpe')}, rsi={m_b.get('rsi')}")
print(f"  Winners: {w}")

print("\n" + "="*60)
print("  TEST 4: /api/peer-compare — same ticker (edge case)")
print("="*60)

code, d = get("/api/peer-compare?ticker=TCS.NS&peer=TCS.NS")
check("same ticker status 200 or 404", code in [200, 404])
print(f"       -> status={code}")

print("\n" + "="*60)
print("  TEST 5: /api/sector-rank — IT sector (7+ peers)")
print("="*60)

code, d = get("/api/sector-rank?ticker=TCS.NS")
check("sector-rank status=200", code == 200)
ranked = d.get("ranked", [])
insights = d.get("insights", {})

check("sector is IT", d.get("sector") == "IT", f"sector={d.get('sector')}")
check("at least 3 peers ranked", len(ranked) >= 3, f"ranked={len(ranked)}")
check("all scores 0-100", all(0 <= m.get("score",0) <= 100 for m in ranked))
check("ranks are sequential", [m["rank"] for m in ranked] == list(range(1, len(ranked)+1)))
check("scores descending", all(ranked[i]["score"] >= ranked[i+1]["score"] for i in range(len(ranked)-1)))

# TCS rank exists
tcs_rank = insights.get("queried_rank")
check("TCS has a rank", tcs_rank is not None, f"rank={tcs_rank}")
check("insights has best_momentum", insights.get("best_momentum") is not None)
check("insights has best_risk_adj", insights.get("best_risk_adj") is not None)
check("insights has lowest_vol", insights.get("lowest_vol") is not None)

print(f"\n  Sector: {d.get('sector')}, Total peers: {insights.get('total_peers')}")
print(f"  TCS rank: #{tcs_rank} of {len(ranked)}")
print(f"  Best momentum: {insights.get('best_momentum')}")
print(f"  Best Sharpe: {insights.get('best_risk_adj')}")
print(f"  Lowest Vol: {insights.get('lowest_vol')}")
print("\n  Leaderboard:")
for m in ranked:
    sym = m["ticker"].replace(".NS","")
    bar = "#" * int(m["score"] / 5)
    print(f"    #{m['rank']} {sym:<12} score={m['score']:5.1f}  {bar}")

print("\n" + "="*60)
print("  TEST 6: /api/sector-rank — Banking sector")
print("="*60)

code, d = get("/api/sector-rank?ticker=HDFCBANK.NS")
check("banking sector-rank 200", code == 200)
ranked = d.get("ranked", [])
check("banking has 4+ peers", len(ranked) >= 4, f"ranked={len(ranked)}")

print(f"\n  Banking Leaderboard:")
for m in ranked:
    sym = m["ticker"].replace(".NS","")
    print(f"    #{m['rank']} {sym:<15} score={m['score']:5.1f}  ret_3m={m.get('ret_3m')}%  rsi={m.get('rsi')}")

print("\n" + "="*60)
print(f"  SUMMARY: {len(PASS)} passed, {len(FAIL)} failed")
print("="*60)
if FAIL:
    print("  FAILURES:")
    for f in FAIL:
        print(f"    - {f}")
else:
    print("  ALL TESTS PASSED")
