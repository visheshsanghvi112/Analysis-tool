"""
capital_allocator.py  — v3 (with fresh market buys support)
─────────────────────────────────────────────────────────────
Upgraded Smart Capital Allocation Engine for StockIQ Pro.

Key improvements over v2:
  - Added "mode" parameter: "recovery" (averaging down) or "market_buys" (fresh market buys).
  - Standardized scoring and logic to handle fresh buy opportunities (qty=0).
  - Rich action text custom-tailored for fresh buys vs averaging down.
"""

import numpy as np
from yf_client import get_quote, get_history, get_info
from news_intelligence import get_advanced_news_analysis

# Curated list of top Nifty 50 leaders representing general market sentiment
CURATED_MARKET_LEADERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS"
]

# ── Sector P/E benchmarks (India) ────────────────────────────────────────────
SECTOR_PE = {
    "Banking": 12, "NBFC": 18, "IT": 28, "Pharma": 22, "FMCG": 40,
    "Auto": 18, "Energy": 10, "Power": 14, "Metals": 10, "Cement": 20,
    "Insurance": 30, "Telecom": 25, "Consumer Tech": 50, "Fintech": 40,
    "Infrastructure": 15, "Logistics": 22, "Mining": 8, "Chemicals": 25,
    "Consumer": 35, "Diversified": 20,
}
DEFAULT_SECTOR_PE = 22

def _rsi_series(close):
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

def _macd(close):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def _compute_stock_profile(ticker: str, buy_price: float, qty: float) -> dict:
    """
    Builds a rich signal profile for one holding using yfinance data.
    """
    try:
        live_px = float((get_quote(ticker) or {}).get("price") or buy_price)
    except Exception:
        live_px = buy_price

    pnl_pct = (live_px - buy_price) / buy_price * 100 if buy_price > 0 else 0.0
    in_loss  = pnl_pct < -0.5

    # Defaults
    rsi_now = rsi_10d = None
    macd_crossover = False
    rsi_divergence = False
    above_ema200   = None
    pct_from_ema200 = None
    vol_spike = False
    vol_3m    = None
    momentum  = None
    pe_ratio  = None
    sector    = None

    try:
        df = get_history(ticker, period="1y")
        if df is not None and len(df) > 60:
            close = df["Close"]
            vol   = df.get("Volume")

            # ── RSI ───────────────────────────────────────────────────────────
            rsi = _rsi_series(close)
            rsi_now = round(float(rsi.iloc[-1]), 1)
            rsi_10d = round(float(rsi.iloc[-10]), 1) if len(rsi) > 10 else rsi_now

            # ── MACD crossover (bullish = MACD crossed above signal in last 3 days)
            macd_line, sig_line = _macd(close)
            if len(macd_line) > 3:
                prev_diff = float(macd_line.iloc[-3]) - float(sig_line.iloc[-3])
                curr_diff = float(macd_line.iloc[-1]) - float(sig_line.iloc[-1])
                macd_crossover = (prev_diff < 0) and (curr_diff >= 0)

            # ── RSI bullish divergence: price lower low, RSI higher low ───────
            if len(close) > 10:
                price_lower = close.iloc[-1] < close.iloc[-10]
                rsi_higher  = rsi_now > rsi_10d
                rsi_divergence = bool(price_lower and rsi_higher)

            # ── 200-day EMA ───────────────────────────────────────────────────
            if len(close) >= 150:
                ema200 = close.ewm(span=200, min_periods=150, adjust=False).mean()
                e200   = float(ema200.iloc[-1])
                above_ema200    = bool(close.iloc[-1] > e200)
                pct_from_ema200 = round((float(close.iloc[-1]) / e200 - 1) * 100, 2)

            # ── Volume spike: today > 1.5× 20-day avg ────────────────────────
            if vol is not None and len(vol) > 20:
                avg_vol = float(vol.rolling(20).mean().iloc[-1])
                if avg_vol > 0:
                    vol_spike = float(vol.iloc[-1]) > 1.5 * avg_vol

            # ── 3M volatility & 1M momentum ──────────────────────────────────
            r3m  = close.pct_change().dropna()
            vol_3m   = round(float(r3m.std() * np.sqrt(252) * 100), 2)
            if len(close) > 22:
                momentum = round(float((close.iloc[-1] / close.iloc[-22] - 1) * 100), 2)
    except Exception:
        pass

    # ── P/E from yfinance info ────────────────────────────────────────────────
    try:
        info     = get_info(ticker)
        pe_ratio = info.get("trailingPE")
        sector   = info.get("sector")
        if pe_ratio:
            pe_ratio = round(float(pe_ratio), 1)
    except Exception:
        pass

    # ── Sentiment ─────────────────────────────────────────────────────────────
    sentiment = 0.0
    try:
        nd = get_advanced_news_analysis(ticker, max_articles=5)
        sentiment = float(nd.get("aggregate", {}).get("avg_sentiment", 0))
    except Exception:
        pass

    gain_to_be = round(((buy_price / live_px) - 1) * 100, 2) if in_loss else 0.0

    return {
        "ticker":           ticker,
        "live_price":       round(live_px, 2),
        "buy_price":        round(buy_price, 2),
        "qty":              qty,
        "pnl_pct":          round(pnl_pct, 2),
        "in_loss":          in_loss,
        "cost_basis":       round(qty * buy_price, 2),
        "rsi":              rsi_now,
        "macd_crossover":   macd_crossover,
        "rsi_divergence":   rsi_divergence,
        "above_ema200":     above_ema200,
        "pct_from_ema200":  pct_from_ema200,
        "vol_spike":        vol_spike,
        "volatility_3m":    vol_3m,
        "momentum_1m":      momentum,
        "sentiment":        round(sentiment, 3),
        "pe_ratio":         pe_ratio,
        "sector":           sector,
        "gain_to_breakeven": gain_to_be,
    }

def _recovery_score(p: dict, horizon_days: int) -> float:
    """
    0–100 composite score. Higher = better capital deployment candidate.
    """
    score = 45.0  # baseline

    rsi  = p.get("rsi") or 50
    sent = p.get("sentiment") or 0.0
    mom  = p.get("momentum_1m") or 0.0
    vol  = p.get("volatility_3m") or 30.0
    loss = abs(p.get("pnl_pct") or 0.0)

    # ── 1. 200-day EMA ────────────────────────────────────────────────────────
    above_200 = p.get("above_ema200")
    pct_200   = p.get("pct_from_ema200") or 0.0
    if above_200 is True:
        score += 15
    elif above_200 is False:
        score -= 25
        if pct_200 < -15:
            score -= 10

    # ── 2. MACD Crossover ─────────────────────────────────────────────────────
    if p.get("macd_crossover"):
        score += 20

    # ── 3. RSI ────────────────────────────────────────────────────────────────
    if rsi < 30:
        score += 20
    elif rsi < 40:
        score += 12
    elif rsi < 50:
        score += 5
    elif rsi > 70:
        score -= 20
    elif rsi > 60:
        score -= 8

    # ── 4. RSI Bullish Divergence ─────────────────────────────────────────────
    if p.get("rsi_divergence"):
        score += 18

    # ── 5. Volume Spike ───────────────────────────────────────────────────────
    if p.get("vol_spike"):
        score += 12

    # ── 6. P/E vs Sector ──────────────────────────────────────────────────────
    pe = p.get("pe_ratio")
    if pe and pe > 0:
        sec_pe = SECTOR_PE.get(p.get("sector") or "", DEFAULT_SECTOR_PE)
        pe_ratio_vs_sector = pe / sec_pe
        if pe_ratio_vs_sector < 0.7:
            score += 15
        elif pe_ratio_vs_sector < 1.0:
            score += 7
        elif pe_ratio_vs_sector > 2.0:
            score -= 20
        elif pe_ratio_vs_sector > 1.5:
            score -= 10

    # ── 7. Momentum ───────────────────────────────────────────────────────────
    if -2 < mom < 5:
        score += 8
    elif mom >= 5:
        score += 3
    elif mom < -10:
        score -= 12

    # ── 8. Loss depth (only if there is an existing loss) ─────────────────────
    if loss > 0:
        if loss > 30 and horizon_days < 180:
            score -= 20
        elif loss > 30 and horizon_days >= 365:
            score += 8
        elif loss > 15 and horizon_days < 90:
            score -= 10
        elif loss < 10:
            score += 6
    else:
        # Fresh buys: reward low volatility + short-horizon combination
        score += 5

    # ── 9. Volatility vs Horizon ──────────────────────────────────────────────
    rf = _horizon_risk_factor(horizon_days)
    if vol > 50 and rf < 0.6:
        score -= 12
    elif vol < 20:
        score += 6

    # ── 10. Sentiment ─────────────────────────────────────────────────────────
    if sent > 0.3:
        score += 8
    elif sent > 0.1:
        score += 4
    elif sent < -0.3:
        score -= 15
    elif sent < -0.1:
        score -= 7

    return round(max(0.0, min(100.0, score)), 1)

def _horizon_risk_factor(horizon_days: int) -> float:
    if horizon_days <= 30:   return 0.4
    if horizon_days <= 90:   return 0.6
    if horizon_days <= 180:  return 0.75
    if horizon_days <= 365:  return 0.90
    return 1.0

def _signal_label(p: dict) -> str:
    bullish = 0
    rsi = p.get("rsi") or 50
    if rsi < 40:
        bullish += 1
    if p.get("macd_crossover"):
        bullish += 2
    if p.get("rsi_divergence"):
        bullish += 2
    if p.get("above_ema200"):
        bullish += 1
    if p.get("vol_spike"):
        bullish += 1
    if (p.get("sentiment") or 0) > 0.1:
        bullish += 1

    if bullish >= 4:
        return "STRONG_BUY"
    elif bullish >= 2:
        return "ACCUMULATE"
    return "WAIT"

def _recovery_timeline(loss_pct, p: dict) -> str:
    gain_needed = abs(loss_pct) / (1 - abs(loss_pct) / 100) if abs(loss_pct) < 99 else abs(loss_pct)
    if gain_needed <= 0:
        return "Immediate (Fresh Buy)"
    vol = p.get("volatility_3m") or 30.0
    monthly_upside = max(1.5, vol / 12 * 0.8)
    months = gain_needed / monthly_upside
    if months <= 1:   return "2–4 weeks"
    if months <= 2:   return "1–2 months"
    if months <= 4:   return "2–4 months"
    if months <= 6:   return "4–6 months"
    if months <= 12:  return "6–12 months"
    return "12+ months"

def _confidence_band(score, horizon_days):
    conf = min(95.0, score + min(10, horizon_days / 365 * 10))
    label = "HIGH" if conf >= 70 else "MODERATE" if conf >= 50 else "LOW"
    return round(conf), label

def _tranche_plan(live_px: float, alloc_amt: float, qty: float, buy_price: float, horizon_days: int) -> list:
    if horizon_days <= 90:
        splits = [0.60, 0.40]
        conditions = [
            f"Deploy now at ₹{live_px:,.0f} (current price)",
            f"Deploy if price falls another 4–6% to ≈₹{live_px * 0.95:,.0f}",
        ]
    else:
        splits = [0.50, 0.30, 0.20]
        conditions = [
            f"Deploy now at ₹{live_px:,.0f} (RSI/MACD signal)",
            f"Deploy if price dips 5–8% to ≈₹{live_px * 0.93:,.0f} (better average)",
            f"Deploy when price reclaims 200-day EMA (trend confirmation)",
        ]

    plan = []
    for i, (split, cond) in enumerate(zip(splits, conditions)):
        t_amt = round(alloc_amt * split, 0)
        shares = max(0, int(t_amt / live_px)) if live_px > 0 else 0
        plan.append({
            "tranche":   i + 1,
            "pct":       int(split * 100),
            "amount":    t_amt,
            "shares":    shares,
            "condition": cond,
        })
    return plan

def allocate_capital(holdings: list, floating_capital: float, horizon_days: int, mode: str = "recovery") -> dict:
    if floating_capital <= 0 or horizon_days <= 0:
        return {"error": "Invalid inputs"}

    # If fresh market buys mode, ignore user holdings and scan curated market leaders
    if mode == "market_buys":
        target_list = [{"ticker": ticker, "buy_price": 0.0, "qty": 0.0} for ticker in CURATED_MARKET_LEADERS]
    else:
        target_list = holdings

    if not target_list:
        return {
            "floating_capital": round(floating_capital, 2),
            "horizon_days": horizon_days,
            "no_loss_positions": True,
            "message": "No holdings provided for analysis.",
            "suggestions": [],
            "summary": {
                "total_allocated": 0, "unallocated_cash": round(floating_capital, 2),
                "positions_addressed": 0, "overall_confidence": "N/A",
                "horizon_label": _horizon_label(horizon_days),
            }
        }

    profiles = []
    for h in target_list:
        p = _compute_stock_profile(h["ticker"].strip().upper(), h["buy_price"], h["qty"])
        profiles.append(p)

    # Filter based on mode
    if mode == "recovery":
        eligible_profiles = [p for p in profiles if p["in_loss"]]
    else:
        eligible_profiles = profiles  # all market leaders are candidates

    if not eligible_profiles:
        return {
            "floating_capital": round(floating_capital, 2),
            "horizon_days": horizon_days,
            "no_loss_positions": True,
            "message": "All your holdings are currently in profit! No averaging down needed.",
            "suggestions": [],
            "summary": {
                "total_allocated": 0, "unallocated_cash": round(floating_capital, 2),
                "positions_addressed": 0, "overall_confidence": "N/A",
                "horizon_label": _horizon_label(horizon_days),
            }
        }

    for p in eligible_profiles:
        p["recovery_score"] = _recovery_score(p, horizon_days)

    # Sort candidates by score descending
    eligible_profiles.sort(key=lambda x: x["recovery_score"], reverse=True)

    # Determine thresholds
    threshold = 38.0 if mode == "recovery" else 42.0
    eligible = [p for p in eligible_profiles if p["recovery_score"] >= threshold]
    if not eligible:
        eligible = [eligible_profiles[0]]  # Fallback to the top candidate

    elig_scores = np.array([p["recovery_score"] for p in eligible], dtype=float)
    exp_s   = np.exp((elig_scores - elig_scores.max()) / 5.0)
    weights = exp_s / exp_s.sum()

    buffer_pct = 0.20 if horizon_days < 90 else 0.12 if horizon_days < 180 else 0.08
    deployable = floating_capital * (1 - buffer_pct)
    reserve    = floating_capital * buffer_pct

    suggestions = []
    total_allocated = 0.0

    for i, (p, w) in enumerate(zip(eligible, weights)):
        alloc_amt = round(deployable * float(w), 2)
        live_px   = p["live_price"]

        shares = max(1 if i == 0 else 0, int(alloc_amt / live_px)) if live_px > 0 else 0
        actual = round(shares * live_px, 2)

        signal              = _signal_label(p)
        conf_pct, conf_lbl  = _confidence_band(p["recovery_score"], horizon_days)
        rec_time            = _recovery_timeline(abs(p["pnl_pct"]), p)
        tranches            = _tranche_plan(live_px, actual, p["qty"], p["buy_price"], horizon_days)

        priority_map = {0: ("🔥 Top Pick", "emerald"), 1: ("⚡ Strong Opportunity", "indigo"),
                        2: ("📊 Moderate Opportunity", "amber")}
        pl, pc = priority_map.get(i, ("🔵 Consider", "slate"))

        # Signals explanation
        active = []
        if p.get("macd_crossover"): active.append("MACD crossover ✓")
        if p.get("rsi_divergence"):  active.append("RSI divergence ✓")
        if p.get("above_ema200"):    active.append("above 200 EMA ✓")
        if p.get("vol_spike"):       active.append("volume spike ✓")
        if p.get("rsi") and p["rsi"] < 40: active.append(f"RSI oversold ({p['rsi']}) ✓")
        sigs = ", ".join(active) if active else "mixed signals"

        if mode == "recovery":
            total_qty = p["qty"] + shares
            new_avg   = (p["qty"] * p["buy_price"] + shares * live_px) / total_qty if total_qty > 0 else live_px
            new_be    = round(((new_avg / live_px) - 1) * 100, 2) if live_px > 0 else 0.0
            avg_red   = round(((p["buy_price"] - new_avg) / p["buy_price"]) * 100, 2) if p["buy_price"] > 0 else 0.0

            if signal == "STRONG_BUY":
                action_text = (f"Strong setup — {sigs}. Buying {shares} shares at ₹{live_px:,.0f} "
                               f"cuts your avg from ₹{p['buy_price']:,.0f} → ₹{new_avg:,.1f} (-{avg_red}%). "
                               f"New break-even: +{new_be}% (was +{p['gain_to_breakeven']}%).")
            elif signal == "ACCUMULATE":
                action_text = (f"Moderate setup — {sigs}. Partial entry of {shares} shares lowers "
                               f"avg cost by {avg_red}%. Est. recovery: {rec_time}.")
            else:
                action_text = (f"Weak signals ({sigs}). Small position of {shares} shares only. "
                               f"Wait for MACD crossover or 200 EMA reclaim before committing more.")
        else:
            # Fresh buys
            new_avg = live_px
            new_be = 0.0
            avg_red = 0.0
            if signal == "STRONG_BUY":
                action_text = (f"Premium fresh entry setup — {sigs}. Strong momentum and low risk path. "
                               f"Deploy first tranche of {shares} shares at ₹{live_px:,.0f}.")
            elif signal == "ACCUMULATE":
                action_text = (f"Solid accumulation setup — {sigs}. High sector resilience. "
                               f"Start building position with {shares} shares at ₹{live_px:,.0f}.")
            else:
                action_text = (f"Neutral setup — {sigs}. Enter with caution. Buy {shares} shares and "
                               f"deploy rest only after MACD crossover confirms.")

        suggestions.append({
            "ticker": p["ticker"], "priority_rank": i + 1,
            "priority_label": pl, "priority_color": pc,
            "allocated_amount": actual, "shares_to_buy": shares,
            "allocation_weight_pct": round(float(w) * 100, 1),
            "live_price": live_px, "buy_price": p["buy_price"],
            "current_pnl_pct": p["pnl_pct"],
            "current_gain_to_be": p["gain_to_breakeven"],
            "new_avg_price": round(new_avg, 2),
            "new_gain_to_be": new_be, "avg_cost_reduction_pct": avg_red,
            "rsi": p["rsi"], "sentiment": p["sentiment"],
            "momentum_1m": p.get("momentum_1m"),
            "macd_crossover": p.get("macd_crossover"),
            "rsi_divergence": p.get("rsi_divergence"),
            "above_ema200": p.get("above_ema200"),
            "pct_from_ema200": p.get("pct_from_ema200"),
            "vol_spike": p.get("vol_spike"),
            "pe_ratio": p.get("pe_ratio"),
            "sector": p.get("sector"),
            "signal": signal, "recovery_score": p["recovery_score"],
            "confidence_pct": conf_pct, "confidence_label": conf_lbl,
            "estimated_recovery": rec_time,
            "action_text": action_text,
            "tranches": tranches,
        })
        total_allocated += actual

    skipped = [p for p in eligible_profiles if not any(s["ticker"] == p["ticker"] for s in suggestions)]
    skip_notes = [{"ticker": sp["ticker"], "pnl_pct": sp["pnl_pct"],
                   "recovery_score": sp.get("recovery_score", 0),
                   "reason": _skip_reason(sp, horizon_days)} for sp in skipped]

    avg_conf = round(float(np.mean([s["confidence_pct"] for s in suggestions])), 0) if suggestions else 0

    return {
        "floating_capital": round(floating_capital, 2),
        "horizon_days": horizon_days, "horizon_label": _horizon_label(horizon_days),
        "deployable_capital": round(deployable, 2),
        "reserve_buffer": round(reserve, 2), "buffer_reason": _buffer_reason(horizon_days),
        "suggestions": suggestions, "skipped_positions": skip_notes,
        "summary": {
            "total_allocated": round(total_allocated, 2),
            "unallocated_cash": round(floating_capital - total_allocated, 2),
            "positions_addressed": len(suggestions), "positions_skipped": len(skip_notes),
            "avg_confidence": avg_conf,
            "overall_confidence": "HIGH" if avg_conf >= 70 else "MODERATE" if avg_conf >= 50 else "LOW",
            "horizon_label": _horizon_label(horizon_days),
            "risk_profile": _risk_profile_label(horizon_days),
        },
        "strategy_note": _strategy_note(horizon_days, len(suggestions), total_allocated, floating_capital),
    }

# ── Label helpers ─────────────────────────────────────────────────────────────

def _horizon_label(d):
    if d <= 30:  return "Short-term (≤1 month)"
    if d <= 90:  return "Short-term (1–3 months)"
    if d <= 180: return "Medium-term (3–6 months)"
    if d <= 365: return "Medium-long (6–12 months)"
    return "Long-term (>1 year)"

def _risk_profile_label(d):
    if d <= 30:  return "Conservative"
    if d <= 90:  return "Moderate-Conservative"
    if d <= 180: return "Moderate"
    if d <= 365: return "Moderate-Aggressive"
    return "Aggressive"

def _skip_reason(p: dict, horizon_days: int) -> str:
    rsi  = p.get("rsi") or 50
    sent = p.get("sentiment") or 0.0
    loss = abs(p.get("pnl_pct") or 0.0)
    if p.get("above_ema200") is False:
        return "Price is below 200-day EMA — structural downtrend. Risk of further losses is high."
    if rsi > 65:
        return "RSI overbought — poor entry timing. Wait for RSI to fall below 45."
    if sent < -0.3:
        return "Strong negative news sentiment — high risk of continued downside pressure."
    if loss > 30 and horizon_days < 90:
        return f"Deep {loss:.0f}% loss needs 12+ months to recover. Your {horizon_days}d horizon is too short."
    return f"Bullish signal strength too low ({p.get('recovery_score', 0)}) — wait for MACD or RSI confirmation."

def _buffer_reason(d):
    if d <= 30:  return "20% reserve — short-horizon safety buffer"
    if d <= 90:  return "12% reserve — for opportunistic dip-buying if market falls further"
    return "8% reserve — SIP-style staggered entry buffer"

def _strategy_note(horizon_days, n, alloc, capital):
    pct = round(alloc / capital * 100) if capital > 0 else 0
    if horizon_days <= 30:
        return (f"Conservative: {pct}% deployed across {n} top setups. "
                f"Short horizon — only high-signal opportunities qualify.")
    if horizon_days <= 180:
        return (f"Medium-term accumulation: {pct}% across {n} positions. "
                f"Stagger entry using the tranche plan. Reassess in 4 weeks.")
    return (f"Long-term compounding: {pct}% across {n} positions. "
            f"200 EMA + MACD filters ensure you're buying structurally sound setups.")
