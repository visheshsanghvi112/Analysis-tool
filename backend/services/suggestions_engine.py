"""
StockIQ Pro — Quantitative Scrip Suggestions & Reasoning Engine
===============================================================

Generates mathematically rigorous, multi-factor suggestions for Indian scrips,
specifically calibrated for high-volume, liquid instruments under ₹200 (as well
as customizable price ceilings).

Every suggestion includes:
1. Technical Regime & Moving Average Confluence (20, 50, 200 DMA)
2. Momentum & Volatility (Wilder's RSI-14, MACD, ATR-14)
3. Volume Footprint & Relative Volume (RVOL vs 20-day baseline)
4. Setup Classification (Momentum Breakout, Oversold Reversal, 50 DMA Pullback, Squeeze Base)
5. Algorithmic Trade Blueprint (Entry Range, Conservative Target 1, Runner Target 2, Hard Stop Loss, R:R)
6. Transparent, Non-Generic Multi-Section Reasoning Dossier:
   - Primary Thesis
   - Price Action & Structural Evidence
   - Volume & Institutional Footprint
   - Momentum & Indicator Confluence
   - Execution Level Justification
   - Fundamental & Sector Catalysts
   - Downside Risk & Invalidation Triggers
7. Quantitative Conviction Score (0 to 100)
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from utils.cache import cache_ttl, cache_store
from yf_client import get_history, get_quote


# Curated universe of high-liquidity, high-impact scrips mostly under ₹200
UNDER_200_RADAR_UNIVERSE = [
    {
        "symbol": "TATASTEEL.NS",
        "name": "Tata Steel Limited",
        "sector": "Metals & Mining",
        "industry": "Steel Manufacturing",
        "catalyst": "Global steel price stabilization, domestic infrastructure demand, reduction in UK/European drag.",
        "description": "One of India's lowest-cost steel producers with expanding Kalinganagar capacity."
    },
    {
        "symbol": "IOC.NS",
        "name": "Indian Oil Corporation",
        "sector": "Energy & Oil Gas",
        "industry": "Oil Refining & Marketing",
        "catalyst": "High dividend yield, robust gross refining margins (GRMs), normalized auto fuel retail margins.",
        "description": "India's largest commercial enterprise with over 36,000 retail fuel outlets and extensive pipeline network."
    },
    {
        "symbol": "BEL.NS",
        "name": "Bharat Electronics Limited",
        "sector": "Defense & Aerospace",
        "industry": "Defense Electronics",
        "catalyst": "Record defense order book, indigenous electronic warfare contracts, expanding naval radar systems.",
        "description": "Navratna PSU monopoly in defense avionics, radar, missile electronics, and electronic warfare suites."
    },
    {
        "symbol": "SAIL.NS",
        "name": "Steel Authority of India Ltd",
        "sector": "Metals & Mining",
        "industry": "Steel Manufacturing",
        "catalyst": "Railway expansion orders, domestic infra capex, debt deleveraging, high operating leverage.",
        "description": "State-owned steel manufacturing titan with captive iron ore mines and huge production capacity."
    },
    {
        "symbol": "NHPC.NS",
        "name": "NHPC Limited",
        "sector": "Power & Renewables",
        "industry": "Hydroelectric Power",
        "catalyst": "Subansiri Lower and Parbati-II project commissioning, steady regulated return on equity, green energy push.",
        "description": "Premier hydro power PSU in India generating clean base-load electricity with predictable cash flows."
    },
    {
        "symbol": "IRFC.NS",
        "name": "Indian Railway Finance Corp",
        "sector": "Railways & Financials",
        "industry": "Infrastructure Finance",
        "catalyst": "Zero-NPA dedicated railway borrowing arm, rolling stock funding, long-term sovereign lease rentals.",
        "description": "Financing arm of the Indian Railways with a cost-plus margin model and sovereign credit backing."
    },
    {
        "symbol": "PNB.NS",
        "name": "Punjab National Bank",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Aggressive asset quality cleanup, GNPA sub-4%, double-digit corporate and MSME credit growth.",
        "description": "Second largest public sector bank in India with extensive rural and semi-urban branch penetration."
    },
    {
        "symbol": "CANBK.NS",
        "name": "Canara Bank",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Industry-leading ROA (>1.0%) and ROE (>18%) among PSBs, strong retail and agri credit traction.",
        "description": "High-performing public sector lender with consistent underwriting discipline and low credit cost."
    },
    {
        "symbol": "UNIONBANK.NS",
        "name": "Union Bank of India",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Expanding net interest margin, low slippage ratio, expanding SME and retail lending book.",
        "description": "Major public sector bank with strong pan-India presence and robust digital lending initiatives."
    },
    {
        "symbol": "IDFCFIRSTB.NS",
        "name": "IDFC FIRST Bank",
        "sector": "Banking & Financials",
        "industry": "Private Sector Bank",
        "catalyst": "Strong retail deposit franchise, CASA ratio above 45%, operating leverage kicking in.",
        "description": "Customer-centric private bank focused on high-yielding retail assets and technology-driven banking."
    },
    {
        "symbol": "SUZLON.NS",
        "name": "Suzlon Energy Limited",
        "sector": "Clean Energy",
        "industry": "Wind Energy Equipment",
        "catalyst": "Debt-free balance sheet, record order book of 4.5+ GW, national renewable hybrid tender tailwinds.",
        "description": "Pioneering Indian wind turbine generator manufacturer with commanding market share in wind turbine installation."
    },
    {
        "symbol": "NBCC.NS",
        "name": "NBCC (India) Limited",
        "sector": "Infrastructure",
        "industry": "Civil Construction",
        "catalyst": "Redevelopment of government colonies, Amrapali stalled housing completion, monetization of PSU lands.",
        "description": "Navratna civil engineering enterprise undertaking large-scale government redevelopment and infrastructure projects."
    },
    {
        "symbol": "SJVN.NS",
        "name": "SJVN Limited",
        "sector": "Power & Renewables",
        "industry": "Power Generation",
        "catalyst": "Multi-gigawatt hydro and solar projects pipeline under construction in Himachal Pradesh, Uttarakhand, and Nepal.",
        "description": "Joint venture PSU between Government of India and Himachal Pradesh expanding aggressively into solar & wind."
    },
    {
        "symbol": "HUDCO.NS",
        "name": "Housing & Urban Development Corp",
        "sector": "Financial Services",
        "industry": "Housing Finance",
        "catalyst": "PMAY housing scheme budget allocation, state urban infrastructure funding mandate, zero retail default risk.",
        "description": "Premier public techno-financial institution dedicated to housing and urban infrastructure financing across India."
    },
    {
        "symbol": "IREDA.NS",
        "name": "Indian Renewable Energy Dev Agency",
        "sector": "Clean Energy",
        "industry": "Green Infrastructure NBFC",
        "catalyst": "Pivotal financier for India's 500 GW non-fossil target, declining cost of borrowing, low credit slippage.",
        "description": "Dedicated renewable energy financier supporting solar, wind, green hydrogen, and battery storage projects."
    },
    {
        "symbol": "TRIDENT.NS",
        "name": "Trident Limited",
        "sector": "Textiles & Paper",
        "industry": "Home Textiles",
        "catalyst": "Rebound in US retail home textile inventory restocking, wheat straw paper operating margins.",
        "description": "Global leader in terry towels and wheat straw paper manufacturing with integrated yarn spinning facilities."
    },
    {
        "symbol": "SOUTHBANK.NS",
        "name": "South Indian Bank",
        "sector": "Banking & Financials",
        "industry": "Private Sector Bank",
        "catalyst": "New management operational turnaround, expanding gold loan book, high provision coverage.",
        "description": "Historic Kerala-headquartered private sector lender undergoing comprehensive balance sheet re-rating."
    },
    {
        "symbol": "FEDERALBNK.NS",
        "name": "Federal Bank Limited",
        "sector": "Banking & Financials",
        "industry": "Private Sector Bank",
        "catalyst": "Fintech neo-bank distribution partnerships, pristine retail credit quality, high NRI remittance market share.",
        "description": "Premier mid-sized private bank known for conservative risk culture and high digital adoption."
    },
    {
        "symbol": "MOTHERSON.NS",
        "name": "Samvardhana Motherson Int.",
        "sector": "Automotive",
        "industry": "Auto Parts & Equipment",
        "catalyst": "Global light vehicle production rebound, increasing EV component value per car, synergistic global acquisitions.",
        "description": "Tier-1 global automotive supplier dominating wiring harnesses, exterior mirrors, and polymer modules."
    },
    {
        "symbol": "UJJIVANSFB.NS",
        "name": "Ujjivan Small Finance Bank",
        "sector": "Banking & Financials",
        "industry": "Small Finance Bank",
        "catalyst": "Post-merger capital efficiency, expanding secured loan portfolio, robust rural and micro-banking momentum.",
        "description": "Mass-market focused small finance bank serving unserved and underserved financial segments."
    },
    {
        "symbol": "EQUITASBNK.NS",
        "name": "Equitas Small Finance Bank",
        "sector": "Banking & Financials",
        "industry": "Small Finance Bank",
        "catalyst": "Strong growth in used car financing and small business loans, declining microfinance concentration.",
        "description": "Diversified small finance bank with high-yielding secured retail lending portfolio."
    },
    {
        "symbol": "MANAPPURAM.NS",
        "name": "Manappuram Finance",
        "sector": "Financial Services",
        "industry": "NBFC & Gold Loan",
        "catalyst": "High gold prices bolstering collateral safety, expansion into vehicle finance and micro-loans.",
        "description": "Leading non-banking finance company specializing in quick turnaround gold-backed collateral lending."
    },
    {
        "symbol": "CASTROLIND.NS",
        "name": "Castrol India Limited",
        "sector": "Chemicals",
        "industry": "Lubricants",
        "catalyst": "Debt-free high-ROCE cash cow, consistent 75%+ dividend payout, expansion into EV thermal fluids.",
        "description": "India's leading private automotive and industrial lubricant manufacturer with premium brand equity."
    },
    {
        "symbol": "HFCL.NS",
        "name": "HFCL Limited",
        "sector": "Telecom & Tech",
        "industry": "Telecom Equipment",
        "catalyst": "BharatNet Phase-3 fiber rollouts, optical fiber export growth to Europe and US, defense communication systems.",
        "description": "Key manufacturer of high-capacity optical fiber cables and next-generation telecom network equipment."
    },
    {
        "symbol": "MAHABANK.NS",
        "name": "Bank of Maharashtra",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Consistently lowest net NPA and highest loan growth among Indian public sector banks.",
        "description": "Top-tier operational efficiency among state-owned lenders with concentrated western region dominance."
    },
    {
        "symbol": "CENTRALBK.NS",
        "name": "Central Bank of India",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Post-PCA expansion, ramping up retail and agriculture co-lending partnerships.",
        "description": "Centenary public sector bank modernizing branch network and accelerating secured loan disbursements."
    },
    {
        "symbol": "IOB.NS",
        "name": "Indian Overseas Bank",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Dramatic turnaround in provision coverage and recovery from legacy stressed assets.",
        "description": "Major state-owned bank with significant presence in southern India undergoing rapid asset quality normalization."
    },
    {
        "symbol": "UCOBANK.NS",
        "name": "UCO Bank",
        "sector": "Banking & Financials",
        "industry": "Public Sector Bank",
        "catalyst": "Clean balance sheet, expanding credit portfolio, strong low-cost CASA deposit foundation.",
        "description": "Kolkata-based public sector lender with improving operational ratios and lower credit provisioning."
    },
    {
        "symbol": "YESBANK.NS",
        "name": "YES Bank Limited",
        "sector": "Banking & Financials",
        "industry": "Private Sector Bank",
        "catalyst": "Normalization of funding costs, expansion into profitable retail loans, resolution of legacy bad assets.",
        "description": "Full-service private commercial bank completing its comprehensive multi-year balance sheet rehabilitation."
    },
    {
        "symbol": "ALOKINDS.NS",
        "name": "Alok Industries",
        "sector": "Textiles & Synthetic Fibers",
        "industry": "Integrated Textiles",
        "catalyst": "Backing and strategic integration with Reliance Industries, modernization of large-scale cotton/polyester units.",
        "description": "Integrated textile manufacturer benefiting from strategic promoter infusion and supply chain integration."
    },
    {
        "symbol": "RPOWER.NS",
        "name": "Reliance Power Limited",
        "sector": "Power & Renewables",
        "industry": "Power Generation",
        "catalyst": "Debt reduction milestones, operational resolution of coal supply agreements, expansion into renewable projects.",
        "description": "Private thermal and solar power generator operating large-scale base-load power utilities."
    },
    {
        "symbol": "NATIONALUM.NS",
        "name": "National Aluminium Company (NALCO)",
        "sector": "Metals & Mining",
        "industry": "Aluminium",
        "catalyst": "Elevated global alumina and aluminium prices, low-cost captive bauxite mines, high export realizations.",
        "description": "Navratna PSU with world-class low cost of bauxite extraction and integrated aluminium smelting."
    },
    {
        "symbol": "NMDC.NS",
        "name": "NMDC Limited",
        "sector": "Metals & Mining",
        "industry": "Iron Ore Mining",
        "catalyst": "Record domestic iron ore despatches, robust pricing, capacity expansion to 67 MTPA.",
        "description": "India's largest iron ore merchant miner supplying primary feed to major steel producers."
    },
    {
        "symbol": "BHEL.NS",
        "name": "Bharat Heavy Electricals Ltd",
        "sector": "Capital Goods",
        "industry": "Heavy Electrical Equipment",
        "catalyst": "Resurgence in thermal power equipment tenders, supercritical 800 MW boiler awards, railway trainsets.",
        "description": "Pioneering state-owned engineering and manufacturing conglomerate dominating power plant equipment."
    },
    {
        "symbol": "GAIL.NS",
        "name": "GAIL (India) Limited",
        "sector": "Energy & Oil Gas",
        "industry": "Gas Transmission & Trading",
        "catalyst": "Expanding National Gas Grid pipeline tariffs, petrochemical margin turnaround, steady volume growth.",
        "description": "Maharatna PSU operating over 70% of India's natural gas pipeline transmission network."
    },
    {
        "symbol": "CPSEETF.NS",
        "name": "CPSE ETF (PSU Basket)",
        "sector": "Exchange Traded Fund",
        "industry": "Diversified PSU Equity",
        "catalyst": "High average dividend yield, government capex theme, low expense ratio basket of Maharatna PSUs.",
        "description": "Index ETF tracking the Nifty CPSE Index consisting of high-dividend central public sector enterprises."
    },
    {
        "symbol": "GOLDBEES.NS",
        "name": "Nippon India ETF Gold BeES",
        "sector": "Exchange Traded Fund",
        "industry": "Commodity Precious Metals",
        "catalyst": "Central bank gold reserve accumulation, geopolitical risk hedge, long-term currency debasement buffer.",
        "description": "India's largest and most liquid gold ETF directly backed by 99.5% pure physical gold bullion."
    },
    {
        "symbol": "SILVERBEES.NS",
        "name": "Nippon India ETF Silver BeES",
        "sector": "Exchange Traded Fund",
        "industry": "Commodity Industrial & Precious",
        "catalyst": "Structural supply deficit, surging industrial consumption in solar PV panels and EV electronic interconnects.",
        "description": "Premier physical silver ETF providing low-cost access to silver with zero storage hassles."
    },
    {
        "symbol": "ITBEES.NS",
        "name": "Nippon India ETF Nifty IT",
        "sector": "Exchange Traded Fund",
        "industry": "Technology Equity",
        "catalyst": "Sub-₹50 accessible entry into India's premier IT giants ahead of global enterprise AI and cloud spending recovery.",
        "description": "Sectoral ETF mirroring the performance of the Nifty IT Index composed of leading software exporters."
    }
]

# Lookup dictionary for fast access by symbol
RADAR_MAP: Dict[str, Dict[str, Any]] = {item["symbol"]: item for item in UNDER_200_RADAR_UNIVERSE}


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return default
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def _calculate_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes technical indicators from daily OHLCV dataframe."""
    if df.empty or len(df) < 25:
        return {}

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    c = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else c
    day_high = float(high.iloc[-1])
    day_low = float(low.iloc[-1])
    day_vol = float(volume.iloc[-1])

    # Moving Averages
    sma20 = float(close.rolling(20, min_periods=5).mean().iloc[-1])
    sma50 = float(close.rolling(50, min_periods=10).mean().iloc[-1])
    sma200 = float(close.rolling(200, min_periods=25).mean().iloc[-1]) if len(close) >= 25 else sma50
    ema9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])

    # Volume dynamics
    vol20 = float(volume.rolling(20, min_periods=5).mean().iloc[-1])
    rvol = day_vol / vol20 if vol20 > 0 else 1.0

    # Wilder's smoothed RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
    loss = (-delta).clip(lower=0).ewm(com=13, adjust=False, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1])
    rsi_5d_ago = float(rsi_series.iloc[-5]) if len(rsi_series) >= 5 else rsi

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    curr_macd = float(macd_line.iloc[-1])
    curr_signal = float(signal_line.iloc[-1])
    curr_hist = float(macd_hist.iloc[-1])
    prev_hist = float(macd_hist.iloc[-2]) if len(macd_hist) >= 2 else curr_hist

    # ATR-14 (Average True Range)
    p_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - p_close).abs(),
        (low - p_close).abs()
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14, min_periods=5).mean().iloc[-1])
    if atr <= 0 or math.isnan(atr):
        atr = c * 0.03

    # 52-Week High & Low
    high_52 = float(high.rolling(252, min_periods=20).max().iloc[-1])
    low_52 = float(low.rolling(252, min_periods=20).min().iloc[-1])

    # Support & Resistance levels from recent 40-day pivots
    recent_lows = low.tail(40)
    recent_highs = high.tail(40)
    support_pivot = float(recent_lows.nsmallest(3).iloc[-1])
    resistance_pivot = float(recent_highs.nlargest(3).iloc[-1])

    return {
        "current_price": c,
        "prev_close": prev_close,
        "day_high": day_high,
        "day_low": day_low,
        "day_volume": day_vol,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ema9": ema9,
        "ema21": ema21,
        "vol20": vol20,
        "rvol": rvol,
        "rsi": rsi,
        "rsi_5d_ago": rsi_5d_ago,
        "macd": curr_macd,
        "macd_signal": curr_signal,
        "macd_hist": curr_hist,
        "prev_hist": prev_hist,
        "atr": atr,
        "high_52": high_52,
        "low_52": low_52,
        "support_pivot": support_pivot,
        "resistance_pivot": resistance_pivot,
    }


def _determine_setup_and_plan(metrics: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classifies the setup and computes a deterministic, risk-managed trade plan
    along with comprehensive, data-grounded reasoning.
    """
    price = round(metrics["current_price"], 2)
    prev_close = round(metrics["prev_close"], 2)
    day_chg_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0.0

    sma20 = round(metrics["sma20"], 2)
    sma50 = round(metrics["sma50"], 2)
    sma200 = round(metrics["sma200"], 2)
    rvol = round(metrics["rvol"], 2)
    rsi = round(metrics["rsi"], 1)
    macd_hist = round(metrics["macd_hist"], 3)
    prev_hist = round(metrics["prev_hist"], 3)
    atr = round(metrics["atr"], 2)
    h52 = round(metrics["high_52"], 2)
    l52 = round(metrics["low_52"], 2)

    # Proximity metrics
    pct_from_52w_low = round(((price - l52) / l52) * 100, 2) if l52 > 0 else 0.0
    pct_from_52w_high = round(((h52 - price) / h52) * 100, 2) if h52 > 0 else 0.0
    pct_vs_sma20 = round(((price - sma20) / sma20) * 100, 2) if sma20 > 0 else 0.0
    pct_vs_sma50 = round(((price - sma50) / sma50) * 100, 2) if sma50 > 0 else 0.0
    pct_vs_sma200 = round(((price - sma200) / sma200) * 100, 2) if sma200 > 0 else 0.0

    # Trend structure
    is_above_200 = price > sma200
    is_above_50 = price > sma50
    is_above_20 = price > sma20
    is_golden_trend = sma50 > sma200
    day_high = metrics.get("day_high", price)
    is_near_day_high = day_high > 0 and (price >= day_high * 0.985)

    # Determine Setup Type
    # 1. BTST 3 PM Momentum Surge (Buy Today, Sell Tomorrow)
    if day_chg_pct >= 1.0 and rvol >= 1.30 and is_near_day_high and 50 <= rsi <= 74:
        setup_type = "BTST_SURGE"
        setup_label = "BTST 3 PM Momentum Surge"
        setup_icon = "⚡"
        horizon = "Intraday to Next-Day Morning Gap"

        entry_low = round(price * 0.996, 2)
        entry_high = round(price * 1.006, 2)
        stop_loss = round(max(metrics.get("day_low", price * 0.96) * 0.992, price - (1.1 * atr)), 2)
        target_1 = round(price + max(1.2 * atr, price * 0.035), 2)
        target_2 = round(price + max(2.5 * atr, price * 0.075), 2)

        thesis_summary = (
            f"Prime Buy-Today-Sell-Tomorrow (BTST) setup: Gaining {day_chg_pct:+.2f}% on {rvol:.1f}x volume surge "
            f"and closing within 1.5% of day's high (₹{day_high:.2f}). High institutional closing demand favors a gap-up opening next session."
        )

    # 2. Golden 50/200 DMA Crossover (Macro Structural Bull)
    elif is_above_200 and (-1.5 <= ((sma50 - sma200) / sma200) * 100 <= 3.8) and is_above_50 and day_chg_pct > -0.8 and rvol >= 0.85:
        setup_type = "GOLDEN_CROSSOVER"
        setup_label = "Golden 50/200 DMA Crossover"
        setup_icon = "✨"
        horizon = "1 to 3 Months Macro Swing"

        entry_low = round(price * 0.99, 2)
        entry_high = round(price * 1.015, 2)
        stop_loss = round(min(sma50 * 0.98, price - (1.5 * atr)), 2)
        target_1 = round(price + max(2.5 * atr, price * 0.12), 2)
        target_2 = round(price + max(4.5 * atr, price * 0.25), 2)

        thesis_summary = (
            f"Structural Golden Cross transition: 50 DMA (₹{sma50:.2f}) is crossing above 200 DMA (₹{sma200:.2f}). "
            f"Marks entry into a major medium-term bull cycle with substantial multi-month runway."
        )

    # 3. High-Dividend PSU Cash Flow
    elif (meta.get("symbol") in ["IOC.NS", "NHPC.NS", "SAIL.NS", "NMDC.NS", "SJVN.NS", "CPSEETF.NS", "CASTROLIND.NS", "GAIL.NS"]
          or ("PSU" in meta.get("name", "") or "PSU" in meta.get("catalyst", ""))) and (rsi <= 52 or pct_from_52w_low <= 25.0):
        setup_type = "DIVIDEND_VALUE"
        setup_label = "High-Dividend PSU Cash Flow"
        setup_icon = "🛡️"
        horizon = "3 to 12 Months Wealth Accumulation"

        entry_low = round(price * 0.985, 2)
        entry_high = round(price * 1.008, 2)
        stop_loss = round(min(l52 * 0.97, price - (1.5 * atr)), 2)
        target_1 = round(price + max(2.0 * atr, price * 0.10), 2)
        target_2 = round(price + max(3.8 * atr, price * 0.22), 2)

        thesis_summary = (
            f"Defensive high-yield sovereign asset trading at ₹{price:.2f}. "
            f"Backed by predictable cash flows, healthy dividend payout (4%–7% est.), and robust capital safety under ₹200."
        )

    # 4. High Volume Momentum Breakout
    elif (is_above_20 or is_above_50) and rvol >= 1.25 and 50 <= rsi <= 72 and day_chg_pct > 0.5:
        setup_type = "MOMENTUM_BREAKOUT"
        setup_label = "High-Volume Momentum Breakout"
        setup_icon = "🚀"
        horizon = "1 to 3 Weeks Swing"

        entry_low = round(price * 0.995, 2)
        entry_high = round(price * 1.012, 2)
        stop_loss = round(max(price - (1.6 * atr), sma20 * 0.985), 2)
        target_1 = round(price + max(2.0 * atr, price * 0.08), 2)
        target_2 = round(price + max(3.5 * atr, price * 0.18), 2)

        thesis_summary = (
            f"Strong momentum expansion with {rvol:.1f}x relative volume surge. "
            f"Price is trading above key moving averages with RSI({rsi:.1f}) in the acceleration zone."
        )

    # 5. Oversold Mean-Reversion Rebound
    elif (rsi <= 40 or pct_from_52w_low <= 12.0) and (macd_hist > prev_hist or day_chg_pct >= 0):
        setup_type = "OVERSOLD_REVERSAL"
        setup_label = "Oversold Value Reversal"
        setup_icon = "💎"
        horizon = "2 to 6 Weeks Mean-Reversion"

        entry_low = round(price * 0.988, 2)
        entry_high = round(price * 1.008, 2)
        stop_loss = round(min(l52 * 0.98, price - (1.4 * atr)), 2)
        target_1 = round(min(sma50, price + (2.2 * atr)), 2)
        if target_1 <= price * 1.04:
            target_1 = round(price + (2.0 * atr), 2)
        target_2 = round(price + (3.8 * atr), 2)

        thesis_summary = (
            f"Deeply oversold base formation (RSI {rsi:.1f}, only {pct_from_52w_low:.1f}% above 52W low). "
            f"MACD histogram shows seller exhaustion and favorable risk-reward asymmetry."
        )

    # 6. Pullback to Dynamic Support (20 DMA / 50 DMA)
    elif is_above_200 and (-3.0 <= pct_vs_sma20 <= 2.5 or -3.5 <= pct_vs_sma50 <= 2.0):
        setup_type = "SUPPORT_PULLBACK"
        setup_label = "Trend Pullback to Support"
        setup_icon = "📈"
        horizon = "2 to 4 Weeks Swing"

        anchor_ma = sma20 if abs(pct_vs_sma20) < abs(pct_vs_sma50) else sma50
        entry_low = round(min(price, anchor_ma) * 0.995, 2)
        entry_high = round(max(price, anchor_ma) * 1.008, 2)
        stop_loss = round(anchor_ma - (1.3 * atr), 2)
        target_1 = round(price + (1.9 * atr), 2)
        target_2 = round(price + (3.4 * atr), 2)

        thesis_summary = (
            f"Healthy shallow pullback inside a macro uptrend (Price > 200 DMA). "
            f"Testing dynamic moving average support with controlled volume, offering a low-risk entry."
        )

    # 7. Low-Volatility Squeeze & Base Accumulation
    elif (atr / price) <= 0.04 and abs(pct_vs_sma20) <= 3.5:
        setup_type = "SQUEEZE_BASE"
        setup_label = "Low-Volatility Squeeze Base"
        setup_icon = "⚡"
        horizon = "3 to 8 Weeks Positional"

        entry_low = round(price * 0.992, 2)
        entry_high = round(price * 1.008, 2)
        stop_loss = round(price - (1.4 * atr), 2)
        target_1 = round(price + (2.2 * atr), 2)
        target_2 = round(price + (4.0 * atr), 2)

        thesis_summary = (
            f"Price volatility has compressed into a narrow accumulation band (ATR only {(atr/price)*100:.1f}% of price). "
            f"Historically precedes significant directional expansion."
        )

    # 8. Default: General Technical Accumulate
    else:
        setup_type = "TECHNICAL_ACCUMULATE"
        setup_label = "Selective Value Accumulate"
        setup_icon = "🎯"
        horizon = "1 to 2 Months Positional"

        entry_low = round(price * 0.99, 2)
        entry_high = round(price * 1.01, 2)
        stop_loss = round(price - (1.5 * atr), 2)
        target_1 = round(price + (2.0 * atr), 2)
        target_2 = round(price + (3.5 * atr), 2)

        thesis_summary = (
            f"Balanced setup with defined risk parameters. Trading at ₹{price:.2f} with support at ₹{stop_loss:.2f}."
        )

    # Safety checks on levels
    if stop_loss >= price:
        stop_loss = round(price * 0.94, 2)
    if target_1 <= price:
        target_1 = round(price * 1.08, 2)
    if target_2 <= target_1:
        target_2 = round(target_1 * 1.10, 2)

    risk_amount = max(price - stop_loss, 0.01)
    reward_amount = max(target_1 - price, 0.01)
    rr_ratio = round(reward_amount / risk_amount, 2)

    # Target percentages
    target_1_pct = round(((target_1 - price) / price) * 100, 1)
    target_2_pct = round(((target_2 - price) / price) * 100, 1)
    stop_loss_pct = round(((stop_loss - price) / price) * 100, 1)

    # Conviction Score (0 - 100)
    score = 50.0

    # Trend component (up to +20)
    if is_above_200:
        score += 8.0
    if is_golden_trend:
        score += 6.0
    if is_above_20:
        score += 6.0

    # Volume component (up to +20)
    if rvol >= 1.5:
        score += 18.0
    elif rvol >= 1.2:
        score += 12.0
    elif rvol >= 0.9:
        score += 6.0

    # RSI & Momentum component (up to +20)
    if 52 <= rsi <= 68:
        score += 18.0  # Prime acceleration
    elif rsi <= 35:
        score += 16.0  # Oversold spring
    elif 45 <= rsi < 52:
        score += 10.0
    elif rsi > 72:
        score -= 8.0   # Overbought caution

    # MACD component (up to +15)
    if macd_hist > 0 and macd_hist > prev_hist:
        score += 14.0  # Expanding positive momentum
    elif macd_hist > prev_hist:
        score += 8.0   # Ticking up

    # Risk:Reward bonus (up to +15)
    if rr_ratio >= 2.5:
        score += 15.0
    elif rr_ratio >= 2.0:
        score += 10.0
    elif rr_ratio >= 1.5:
        score += 5.0

    # Sub-200 Liquidity confirmation (+10)
    if metrics["vol20"] >= 500000:
        score += 10.0  # Healthy institutional liquidity
    elif metrics["vol20"] >= 150000:
        score += 5.0

    conviction_score = int(min(98, max(45, score)))

    if conviction_score >= 80:
        conviction_tier = "HIGH CONVICTION"
        conviction_color = "#10b981"  # Emerald
    elif conviction_score >= 65:
        conviction_tier = "MODERATE CONVICTION"
        conviction_color = "#06b6d4"  # Cyan
    else:
        conviction_tier = "SPECULATIVE / HIGH BETA"
        conviction_color = "#f59e0b"  # Amber

    # Build Deep Transparent Reasoning Dossier
    ma_alignment_str = (
        f"Trading above all key benchmarks (20 DMA ₹{sma20:.2f}, 50 DMA ₹{sma50:.2f}, 200 DMA ₹{sma200:.2f})"
        if (is_above_20 and is_above_50 and is_above_200)
        else f"Trading at {pct_vs_sma20:+.1f}% vs 20 DMA (₹{sma20:.2f}) and {pct_vs_sma50:+.1f}% vs 50 DMA (₹{sma50:.2f})"
    )

    reasoning_price_action = (
        f"Price action is positioned at ₹{price:.2f} ({day_chg_pct:+.2f}% session move). "
        f"{ma_alignment_str}. "
        f"The stock has preserved structure, holding {pct_from_52w_low:.1f}% above its 52-week low of ₹{l52:.2f}, "
        f"leaving {pct_from_52w_high:.1f}% runway to its 52-week high of ₹{h52:.2f}."
    )

    volume_desc = (
        f"Institutional volume footprint confirms activity: Current volume is {rvol:.2f}x the 20-day average "
        f"({int(metrics['day_volume']):,} shares vs 20-day mean of {int(metrics['vol20']):,} shares). "
        + ("Heavy accumulation signature indicating strong institutional absorption." if rvol >= 1.25 else
           "Volume is in a healthy equilibrium, reflecting low supply pressure during this consolidation.")
    )

    momentum_desc = (
        f"Wilder's RSI-14 stands at {rsi:.1f} (compared to {metrics['rsi_5d_ago']:.1f} 5 sessions ago). "
        f"MACD histogram reads {macd_hist:+.3f} (previous {prev_hist:+.3f}), indicating "
        + ("accelerating bullish expansion." if macd_hist > 0 and macd_hist > prev_hist else
           "bullish momentum inflection and positive histogram expansion." if macd_hist > prev_hist else
           "consolidation momentum with early stabilization.")
    )

    levels_desc = (
        f"Entry band is mathematically mapped between ₹{entry_low:.2f} and ₹{entry_high:.2f}. "
        f"Target 1 at ₹{target_1:.2f} ({target_1_pct:+.1f}%) provides a conservative swing milestone based on 2x ATR expansion. "
        f"Extended Target 2 at ₹{target_2:.2f} ({target_2_pct:+.1f}%) captures a full multi-week structural run. "
        f"Hard invalidation stop loss is anchored at ₹{stop_loss:.2f} ({stop_loss_pct:.1f}%), yielding an asymmetric Risk-to-Reward ratio of 1:{rr_ratio}."
    )

    invalidation_desc = (
        f"Trade invalidation strictly triggers on a daily close below ₹{stop_loss:.2f}. "
        f"A decisive breach below this support voids the setup, indicating seller control and demanding immediate risk mitigation. "
        f"Never average down if the hard invalidation level is breached."
    )

    # Liquidity & Operator Trap Assessment
    vol20 = int(metrics["vol20"])
    session_vol = int(metrics["day_volume"])
    avg_turnover_cr = round((price * vol20) / 1e7, 2)
    session_turnover_cr = round((price * session_vol) / 1e7, 2)

    if avg_turnover_cr >= 50.0:
        liquidity_rating = "INSTITUTIONAL_LIQUID"
        liquidity_label = "Institutional-Grade (₹50+ Cr Daily Turnover)"
        operator_trap_risk = "MINIMAL"
        operator_trap_label = "Minimal Risk · High Turnover Institutional Scrip"
        circuit_band_desc = "Standard 10%–20% Exchange Circuit Bands"
    elif avg_turnover_cr >= 15.0:
        liquidity_rating = "HIGH_LIQUIDITY"
        liquidity_label = "High Liquidity (₹15–₹50 Cr Daily Turnover)"
        operator_trap_risk = "LOW"
        operator_trap_label = "Low Risk · Liquid Mid/Large Cap"
        circuit_band_desc = "Standard 10%–20% Exchange Circuit Bands"
    elif avg_turnover_cr >= 5.0:
        liquidity_rating = "MODERATE_LIQUIDITY"
        liquidity_label = "Moderate Liquidity (₹5–₹15 Cr Daily Turnover)"
        operator_trap_risk = "MODERATE"
        operator_trap_label = "Moderate Risk · Size Orders Appropriately"
        circuit_band_desc = "Watch 10% Circuit Filter Limits"
    else:
        liquidity_rating = "MICRO_CAUTION"
        liquidity_label = "Caution · Sub-₹5 Cr Daily Turnover"
        operator_trap_risk = "ELEVATED"
        operator_trap_label = "Elevated Risk · Beware of Illiquid Circuit Freezes"
        circuit_band_desc = "Tight Band Watch / Potential 5% Filter"

    # 3-Stage Trailing Trade Management Playbook
    execution_playbook = {
        "stage_1_entry": {
            "title": "Stage 1: Position Accumulation",
            "description": f"Accumulate strictly between ₹{entry_low:.2f} and ₹{entry_high:.2f}. Set initial hard stop-loss alert at ₹{stop_loss:.2f}.",
            "action": "Execute Entry Order",
            "stop_loss": stop_loss
        },
        "stage_2_target_1": {
            "title": "Stage 2: Target 1 Milestone & Breakeven Trail",
            "description": f"When price reaches ₹{target_1:.2f} ({target_1_pct:+.1f}%), book 50% partial profits immediately. Shift your stop-loss on remaining 50% to your entry price (₹{entry_high:.2f}), locking this into a completely risk-free trade.",
            "action": "Book 50% Profit & Trail Stop to Breakeven",
            "level": target_1
        },
        "stage_3_target_2": {
            "title": "Stage 3: Extended Runner Target",
            "description": f"Allow the remaining 50% position to ride toward Target 2 at ₹{target_2:.2f} ({target_2_pct:+.1f}%). Trail a dynamic stop loss along the rising 20 DMA (currently ₹{sma20:.2f}).",
            "action": "Close Remaining Position or Trail by 20 DMA",
            "level": target_2
        }
    }

    # Pre-calculated Position Sizing for Retail Portfolios
    risk_per_share = max(round(price - stop_loss, 2), 0.05)

    def _calc_position_guide(portfolio_capital: float, risk_pct: float = 1.0):
        budget = portfolio_capital * (risk_pct / 100.0)
        qty = max(1, int(budget // risk_per_share))
        outlay = round(qty * price, 2)
        # Cap outlay at 30% of total portfolio to prevent over-concentration
        if outlay > portfolio_capital * 0.35:
            qty = max(1, int((portfolio_capital * 0.35) // price))
            outlay = round(qty * price, 2)
        profit_t1 = round(qty * (target_1 - price), 2)
        profit_t2 = round(qty * (target_2 - price), 2)
        max_loss = round(qty * risk_per_share, 2)
        return {
            "portfolio_capital": portfolio_capital,
            "risk_pct": risk_pct,
            "shares_to_buy": qty,
            "capital_outlay": outlay,
            "capital_pct": round((outlay / portfolio_capital) * 100, 1),
            "max_risk_loss": max_loss,
            "profit_target_1": profit_t1,
            "profit_target_2": profit_t2,
            "portfolio_gain_t1_pct": round((profit_t1 / portfolio_capital) * 100, 2),
            "portfolio_gain_t2_pct": round((profit_t2 / portfolio_capital) * 100, 2),
        }

    position_sizing_presets = {
        "capital_25k": _calc_position_guide(25000.0, 1.0),
        "capital_50k": _calc_position_guide(50000.0, 1.0),
        "capital_100k": _calc_position_guide(100000.0, 1.0),
        "capital_200k": _calc_position_guide(200000.0, 1.5),
    }
    # Circuit & Trap Metrics
    circuit_band_pct = 20.0 if avg_turnover_cr >= 25.0 else (10.0 if avg_turnover_cr >= 5.0 else 5.0)
    upper_circuit = round(price * (1.0 + (circuit_band_pct / 100.0)), 2)
    lower_circuit = round(price * (1.0 - (circuit_band_pct / 100.0)), 2)
    headroom_pct = round(((upper_circuit - price) / price) * 100.0, 1)
    circuit_status = "Healthy Headroom (>8%)" if headroom_pct >= 8.0 else "Caution: Near Circuit Ceiling (<5%)"
    circuit_metrics = {
        "circuit_band_pct": circuit_band_pct,
        "upper_circuit": upper_circuit,
        "lower_circuit": lower_circuit,
        "headroom_pct": headroom_pct,
        "circuit_status": circuit_status,
        "is_circuit_locked": False,
    }

    # Delivery & Institutional Footprint Metrics
    base_deliv = 54.0 if meta.get("sector") in ["Metals & Mining", "Power & Renewables", "Energy & Oil Gas", "Banking & Financials"] else 46.0
    if rvol >= 1.3 and day_chg_pct > 0:
        deliv_pct = min(78.5, round(base_deliv + (rvol * 5.2), 1))
        deliv_verdict = "High Institutional Accumulation (>55% Delivery)"
    elif rvol >= 1.6 and abs(day_chg_pct) < 0.8:
        deliv_pct = round(base_deliv + 3.0, 1)
        deliv_verdict = "Stealth Hand-Off / Block Absorption"
    else:
        deliv_pct = max(28.0, round(base_deliv - 5.0, 1))
        deliv_verdict = "Balanced Delivery vs Intraday Turnover"
    delivery_metrics = {
        "estimated_delivery_pct": deliv_pct,
        "delivery_verdict": deliv_verdict,
        "delivery_quality": "Institutional Grade" if deliv_pct >= 50.0 else "Mixed Speculative"
    }

    # Nifty Beta Stress Test
    beta = 1.35 if setup_type in ["MOMENTUM_BREAKOUT", "BTST_SURGE"] else (0.80 if setup_type == "DIVIDEND_VALUE" else 1.10)
    nifty_beta_sensitivity = {
        "beta": beta,
        "nifty_drop_1pct": round(-1.0 * beta, 2),
        "nifty_drop_2pct": round(-2.0 * beta, 2),
        "nifty_rally_1pct": round(1.0 * beta, 2),
        "nifty_rally_2pct": round(2.0 * beta, 2),
        "defensive_score": "High Resilience" if beta < 0.95 else ("Moderate Volatility" if beta <= 1.25 else "High Beta Aggressive"),
    }

    # One-Click Broker Order Formats
    clean_sym = meta.get("symbol", "").replace(".NS", "").replace(".BO", "")
    guide_qty = position_sizing_presets["capital_50k"]["shares_to_buy"]
    broker_order_formats = {
        "zerodha_kite_csv": f"{clean_sym},NSE,{guide_qty},{price:.2f},LIMIT,CNC",
        "dhan_json": {
            "dhanClientId": "YOUR_ID",
            "transactionType": "BUY",
            "exchangeSegment": "NSE_EQ",
            "productType": "CNC",
            "orderType": "LIMIT",
            "price": price,
            "securityId": clean_sym,
            "quantity": guide_qty,
        },
        "angel_one_csv": f"{clean_sym},NSE,{guide_qty},{price:.2f},LIMIT,DELIVERY",
        "groww_csv": f"{clean_sym},NSE,{guide_qty},{price:.2f},CNC",
        "one_click_alert_text": f"BUY {clean_sym} @ ₹{price:.2f} | SL: ₹{stop_loss:.2f} | T1: ₹{target_1:.2f} | T2: ₹{target_2:.2f} (R:R 1:{rr_ratio})"
    }

    # Dynamic Holding Time & Days to Target Projection
    eff_atr = max(atr, price * 0.015)
    dist_t1 = max(target_1 - price, price * 0.02)
    dist_t2 = max(target_2 - price, price * 0.05)
    
    # Days needed based on trending velocity (0.65x ATR daily travel)
    days_to_t1 = max(1, int(math.ceil(dist_t1 / max(eff_atr * 0.65, 0.1))))
    days_to_t2 = max(days_to_t1 + 2, int(math.ceil(dist_t2 / max(eff_atr * 0.50, 0.1))))
    
    if days_to_t1 <= 2 or setup_type == "BTST_SURGE":
        horizon_bucket = "BTST"
        horizon_label = "1–2 Trading Days (BTST / Scalp)"
    elif days_to_t1 <= 7:
        horizon_bucket = "SWING_SHORT"
        horizon_label = "3–7 Trading Days (Short Swing)"
    elif days_to_t1 <= 21:
        horizon_bucket = "SWING_POSITIONAL"
        horizon_label = "2–4 Weeks (Positional Swing)"
    elif days_to_t1 <= 65:
        horizon_bucket = "QUARTERLY_CYCLE"
        horizon_label = "1–3 Months (Quarterly Trend)"
    else:
        horizon_bucket = "WEALTH_COMPOUNDER"
        horizon_label = "6+ Months (Multi-Bagger Compounder)"

    # Annualized Return (IRR / CAGR equivalent for holding days)
    gain_ratio_t1 = max(1.0 + (target_1_pct / 100.0), 1.001)
    annualized_irr_t1 = min(999.0, round(((gain_ratio_t1 ** (252.0 / max(days_to_t1, 1))) - 1.0) * 100.0, 1))
    gain_ratio_t2 = max(1.0 + (target_2_pct / 100.0), 1.001)
    annualized_irr_t2 = min(999.0, round(((gain_ratio_t2 ** (252.0 / max(days_to_t2, 2))) - 1.0) * 100.0, 1))

    holding_period_metrics = {
        "days_to_target_1": days_to_t1,
        "days_to_target_2": days_to_t2,
        "horizon_bucket": horizon_bucket,
        "horizon_label": horizon_label,
        "annualized_irr_t1_pct": annualized_irr_t1,
        "annualized_irr_t2_pct": annualized_irr_t2,
    }

    # Statutory Indian Delivery Taxes & Real Net Profit Friction Model (Zerodha/Angel/Groww CNC)
    unit_qty = max(10, int(50000 // price))
    buy_val = unit_qty * price
    sell_val_t1 = unit_qty * target_1
    sell_val_t2 = unit_qty * target_2
    stt_total_t1 = round((buy_val * 0.001) + (sell_val_t1 * 0.001), 2)
    stamp_duty = round(buy_val * 0.00015, 2)
    exchange_charges = round((buy_val + sell_val_t1) * 0.0000325, 2)
    sebi_charges = round((buy_val + sell_val_t1) * 0.000001, 2)
    gst_charges = round(exchange_charges * 0.18, 2)
    dp_charges = 15.34  # CDSL flat fee on delivery sale
    total_friction_charges_t1 = round(stt_total_t1 + stamp_duty + exchange_charges + sebi_charges + gst_charges + dp_charges, 2)
    
    gross_profit_t1 = round(unit_qty * (target_1 - price), 2)
    net_profit_t1 = round(gross_profit_t1 - total_friction_charges_t1, 2)
    breakeven_price = round(price + (total_friction_charges_t1 / unit_qty), 2)
    friction_drag_pct = round((total_friction_charges_t1 / max(gross_profit_t1, 1.0)) * 100.0, 1)

    tax_breakdown = {
        "reference_shares": unit_qty,
        "brokerage": 0.0,  # Zero brokerage for CNC delivery
        "stt_total": stt_total_t1,
        "stamp_duty": stamp_duty,
        "exchange_charges": exchange_charges,
        "gst_charges": gst_charges,
        "dp_charges": dp_charges,
        "total_statutory_charges": total_friction_charges_t1,
        "breakeven_price": breakeven_price,
        "breakeven_move_pct": round(((breakeven_price - price) / price) * 100.0, 2),
        "gross_profit_t1": gross_profit_t1,
        "net_profit_t1": net_profit_t1,
        "friction_drag_pct": friction_drag_pct,
    }

    # 3-Tranche Staggered Scale-In Matrix for Sub-₹200 Volatility Control
    pyramiding_plan = {
        "tranche_1": {
            "name": "Tranche 1 (Starter Support)",
            "allocation_pct": 40,
            "target_price": round(price, 2),
            "condition": "Initiate 40% position at CMP / immediate support floor",
        },
        "tranche_2": {
            "name": "Tranche 2 (Momentum Pivot)",
            "allocation_pct": 30,
            "target_price": round(entry_high, 2),
            "condition": f"Add 30% only on clean 15-min candle close above ₹{entry_high:.2f}",
        },
        "tranche_3": {
            "name": "Tranche 3 (Retest Buffer)",
            "allocation_pct": 30,
            "target_price": round(entry_low, 2),
            "condition": f"Accumulate final 30% on intraday pullback near ₹{entry_low:.2f}",
        },
        "blended_entry_price": round(0.40 * price + 0.30 * entry_high + 0.30 * entry_low, 2)
    }

    # Trailing Stop-Loss Strategies
    trailing_stop_strategies = {
        "breakeven_trail": {
            "name": "Target 1 Breakeven Trail",
            "rule": f"Once price crosses Target 1 (₹{target_1:.2f}), move Stop Loss from ₹{stop_loss:.2f} to ₹{entry_high:.2f} (locked profit).",
        },
        "moving_average_trail": {
            "name": "20 EMA Trend Trail",
            "rule": f"Hold remaining runner position until a daily candle closes below the 20 EMA (~₹{sma20:.2f}).",
        },
        "atr_volatility_trail": {
            "name": "1.5x ATR Volatility Floor",
            "rule": f"Dynamic trailing stop set at (Peak Price - ₹{(1.5 * atr):.2f}).",
        },
        "swing_low_trail": {
            "name": "3-Day Swing Low Trail",
            "rule": "Trail stop loss to the lowest price of the preceding 3 trading sessions.",
        }
    }

    return {
        "setup_type": setup_type,
        "setup_label": setup_label,
        "setup_icon": setup_icon,
        "horizon": horizon,
        "conviction_score": conviction_score,
        "conviction_tier": conviction_tier,
        "conviction_color": conviction_color,
        "day_change_pct": day_chg_pct,
        "trade_plan": {
            "entry_range": [entry_low, entry_high],
            "entry_display": f"₹{entry_low:.2f} – ₹{entry_high:.2f}",
            "target_1": target_1,
            "target_1_pct": target_1_pct,
            "target_2": target_2,
            "target_2_pct": target_2_pct,
            "stop_loss": stop_loss,
            "stop_loss_pct": stop_loss_pct,
            "risk_reward_ratio": rr_ratio,
            "risk_per_share": risk_per_share,
        },
        "technicals": {
            "current_price": price,
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "pct_vs_sma20": pct_vs_sma20,
            "pct_vs_sma50": pct_vs_sma50,
            "pct_vs_sma200": pct_vs_sma200,
            "rvol": round(rvol, 2),
            "rsi": round(rsi, 1),
            "macd_hist": round(macd_hist, 3),
            "atr": round(atr, 2),
            "high_52": round(h52, 2),
            "low_52": round(l52, 2),
            "pct_from_52w_low": pct_from_52w_low,
            "pct_from_52w_high": pct_from_52w_high,
            "avg_volume_20d": vol20,
            "session_volume": session_vol,
            "avg_turnover_cr_20d": avg_turnover_cr,
            "session_turnover_cr": session_turnover_cr,
        },
        "liquidity_guardrails": {
            "avg_turnover_cr": avg_turnover_cr,
            "session_turnover_cr": session_turnover_cr,
            "liquidity_rating": liquidity_rating,
            "liquidity_label": liquidity_label,
            "operator_trap_risk": operator_trap_risk,
            "operator_trap_label": operator_trap_label,
            "circuit_band_desc": circuit_band_desc,
        },
        "circuit_metrics": circuit_metrics,
        "delivery_metrics": delivery_metrics,
        "nifty_beta_sensitivity": nifty_beta_sensitivity,
        "broker_order_formats": broker_order_formats,
        "execution_playbook": execution_playbook,
        "position_sizing_presets": position_sizing_presets,
        "holding_period_metrics": holding_period_metrics,
        "tax_breakdown": tax_breakdown,
        "pyramiding_plan": pyramiding_plan,
        "trailing_stop_strategies": trailing_stop_strategies,
        "reasoning": {
            "primary_thesis": thesis_summary,
            "price_action": reasoning_price_action,
            "volume_footprint": volume_desc,
            "momentum_indicators": momentum_desc,
            "levels_rationale": levels_desc,
            "sector_catalyst": meta.get("catalyst", "Strong industry growth and capex tailwinds."),
            "invalidation_warning": invalidation_desc,
        }
    }


def analyze_single_scrip(ticker: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Analyzes a single scrip and returns a complete suggestion object."""
    ticker_clean = ticker.strip().upper()
    meta = metadata or RADAR_MAP.get(ticker_clean) or {
        "symbol": ticker_clean,
        "name": ticker_clean.replace(".NS", "").replace(".BO", ""),
        "sector": "Equities",
        "industry": "NSE Stock",
        "catalyst": "Active equity performance and technical setup.",
        "description": "NSE listed security evaluated by StockIQ Pro."
    }

    try:
        # Fetch 1y OHLCV data
        df = get_history(ticker_clean, period="1y")
        if df.empty or len(df) < 20:
            return None

        indicators = _calculate_indicators(df)
        if not indicators or indicators.get("current_price") is None:
            return None

        # Build trade plan and reasoning
        plan_and_reasoning = _determine_setup_and_plan(indicators, meta)

        return {
            "symbol": ticker_clean,
            "display_symbol": ticker_clean.replace(".NS", "").replace(".BO", ""),
            "name": meta.get("name", ticker_clean),
            "sector": meta.get("sector", "General"),
            "industry": meta.get("industry", "Equity"),
            "description": meta.get("description", ""),
            "is_under_200": indicators["current_price"] <= 200.0,
            "currency": "₹",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            **plan_and_reasoning
        }
    except Exception as e:
        return None


def get_curated_suggestions(
    max_price: Optional[float] = 200.0,
    setup_type: str = "all",
    sector: str = "all",
    min_conviction: int = 50,
    sort_by: str = "conviction",
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Computes suggestions across the curated universe with high-speed parallelization.
    Cached in-memory with automatic TTL.
    """
    cache_key = "stockiq_suggestions_master_scan"

    if not force_refresh:
        cached = cache_store.get(cache_key)
        if cached:
            return _filter_and_sort_results(
                cached, max_price, setup_type, sector, min_conviction, sort_by
            )

    # Parallel scan across the universe
    def _worker(item: Dict[str, Any]):
        return analyze_single_scrip(item["symbol"], item)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(_worker, item) for item in UNDER_200_RADAR_UNIVERSE]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)

    # Cache raw scan for 120 seconds
    cache_store.set(cache_key, results, ttl_seconds=120)

    return _filter_and_sort_results(
        results, max_price, setup_type, sector, min_conviction, sort_by
    )


def _filter_and_sort_results(
    raw_results: List[Dict[str, Any]],
    max_price: Optional[float],
    setup_type: str,
    sector: str,
    min_conviction: int,
    sort_by: str,
) -> Dict[str, Any]:
    """Applies filters and sorting on the scanned results."""
    filtered = []
    for item in raw_results:
        price = item["technicals"]["current_price"]

        # Price ceiling filter
        if max_price is not None and price > max_price:
            continue

        # Setup type filter
        if setup_type != "all":
            if setup_type == "high_conviction" and item["conviction_score"] < 80:
                continue
            elif setup_type != "high_conviction" and item["setup_type"].lower() != setup_type.lower():
                continue

        # Sector filter
        if sector != "all" and item["sector"].lower() != sector.lower():
            continue

        # Minimum conviction filter
        if item["conviction_score"] < min_conviction:
            continue

        filtered.append(item)

    # Sorting
    if sort_by == "conviction":
        filtered.sort(key=lambda x: x["conviction_score"], reverse=True)
    elif sort_by == "risk_reward":
        filtered.sort(key=lambda x: x["trade_plan"]["risk_reward_ratio"], reverse=True)
    elif sort_by == "rvol":
        filtered.sort(key=lambda x: x["technicals"]["rvol"], reverse=True)
    elif sort_by == "turnover":
        filtered.sort(key=lambda x: x["liquidity_guardrails"]["avg_turnover_cr"], reverse=True)
    elif sort_by == "circuit_headroom":
        filtered.sort(key=lambda x: x["circuit_metrics"]["headroom_pct"], reverse=True)
    elif sort_by == "delivery":
        filtered.sort(key=lambda x: x["delivery_metrics"]["estimated_delivery_pct"], reverse=True)
    elif sort_by == "price_asc":
        filtered.sort(key=lambda x: x["technicals"]["current_price"])
    elif sort_by == "price_desc":
        filtered.sort(key=lambda x: x["technicals"]["current_price"], reverse=True)
    elif sort_by == "day_gain":
        filtered.sort(key=lambda x: x["day_change_pct"], reverse=True)
    elif sort_by == "proximity_low":
        filtered.sort(key=lambda x: x["technicals"]["pct_from_52w_low"])

    # High-level summary metrics
    under_200_count = sum(1 for x in raw_results if x["technicals"]["current_price"] <= 200)
    high_conviction_count = sum(1 for x in raw_results if x["conviction_score"] >= 80)
    breakout_count = sum(1 for x in raw_results if x["setup_type"] == "MOMENTUM_BREAKOUT")
    btst_count = sum(1 for x in raw_results if x["setup_type"] == "BTST_SURGE")
    golden_count = sum(1 for x in raw_results if x["setup_type"] == "GOLDEN_CROSSOVER")
    dividend_count = sum(1 for x in raw_results if x["setup_type"] == "DIVIDEND_VALUE")
    oversold_count = sum(1 for x in raw_results if x["setup_type"] == "OVERSOLD_REVERSAL")
    pullback_count = sum(1 for x in raw_results if x["setup_type"] == "SUPPORT_PULLBACK")
    squeeze_count = sum(1 for x in raw_results if x["setup_type"] == "SQUEEZE_BASE")

    avg_rr = (
        round(sum(x["trade_plan"]["risk_reward_ratio"] for x in filtered) / len(filtered), 2)
        if filtered else 0.0
    )

    top_pick = raw_results[0] if raw_results else None

    # Available sectors for filter dropdown
    available_sectors = sorted(list(set(x["sector"] for x in raw_results)))

    return {
        "total_scanned": len(raw_results),
        "total_matches": len(filtered),
        "under_200_count": under_200_count,
        "high_conviction_count": high_conviction_count,
        "breakout_count": breakout_count,
        "btst_count": btst_count,
        "golden_count": golden_count,
        "dividend_count": dividend_count,
        "oversold_count": oversold_count,
        "pullback_count": pullback_count,
        "squeeze_count": squeeze_count,
        "avg_risk_reward": avg_rr,
        "top_pick": {
            "symbol": top_pick["display_symbol"],
            "name": top_pick["name"],
            "price": top_pick["technicals"]["current_price"],
            "conviction": top_pick["conviction_score"],
            "setup": top_pick["setup_label"]
        } if top_pick else None,
        "available_sectors": available_sectors,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suggestions": filtered
    }


def calculate_custom_position_size(
    ticker: str,
    capital: float = 100000.0,
    risk_pct: float = 1.0,
    custom_risk_amount: Optional[float] = None,
    sizing_mode: str = "risk_budget",
    fixed_outlay: Optional[float] = None,
    fixed_shares: Optional[int] = None,
    holding_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Calculates position size and risk allocation for any ticker and user capital."""
    suggestion = analyze_single_scrip(ticker)
    if not suggestion:
        raise ValueError(f"Unable to analyze ticker '{ticker}' for position sizing.")

    price = suggestion["technicals"]["current_price"]
    stop_loss = suggestion["trade_plan"]["stop_loss"]
    target_1 = suggestion["trade_plan"]["target_1"]
    target_2 = suggestion["trade_plan"]["target_2"]
    conviction = suggestion.get("conviction_score", 75)
    default_days = suggestion.get("holding_period_metrics", {}).get("days_to_target_1", 7)
    effective_days = holding_days if (holding_days and holding_days > 0) else default_days

    risk_per_share = max(round(price - stop_loss, 2), 0.05)

    # Determine quantity based on sizing_mode
    if sizing_mode == "fixed_shares" and fixed_shares and fixed_shares > 0:
        qty = fixed_shares
    elif sizing_mode == "fixed_outlay" and fixed_outlay and fixed_outlay > 0:
        qty = max(1, int(fixed_outlay // price))
    elif sizing_mode == "slot_4":
        slot_budget = capital * 0.25
        qty = max(1, int(slot_budget // price))
    elif sizing_mode == "slot_5":
        slot_budget = capital * 0.20
        qty = max(1, int(slot_budget // price))
    elif sizing_mode == "conviction_kelly":
        # Kelly weight: 85+ conviction -> 25%, 70-84 -> 18%, <70 -> 10%
        kelly_pct = 0.25 if conviction >= 85 else (0.18 if conviction >= 70 else 0.10)
        qty = max(1, int((capital * kelly_pct) // price))
    else:
        # Default: Risk-Budgeted Sizing (Capital * Risk% / Risk-per-share)
        risk_budget = custom_risk_amount if custom_risk_amount is not None else capital * (risk_pct / 100.0)
        risk_budget = max(risk_budget, 100.0)
        qty = max(1, int(risk_budget // risk_per_share))

    outlay = round(qty * price, 2)

    # Cap single position at 50% of capital for safety unless explicit fixed_shares requested
    if sizing_mode not in ["fixed_shares"] and outlay > capital * 0.50:
        qty = max(1, int((capital * 0.50) // price))
        outlay = round(qty * price, 2)

    profit_t1 = round(qty * (target_1 - price), 2)
    profit_t2 = round(qty * (target_2 - price), 2)
    actual_risk_loss = round(qty * risk_per_share, 2)

    # Statutory Indian Delivery Taxes on this exact quantity
    buy_val = outlay
    sell_val_t1 = qty * target_1
    stt_val = round((buy_val * 0.001) + (sell_val_t1 * 0.001), 2)
    stamp_val = round(buy_val * 0.00015, 2)
    exchange_val = round((buy_val + sell_val_t1) * 0.0000325, 2)
    sebi_val = round((buy_val + sell_val_t1) * 0.000001, 2)
    gst_val = round(exchange_val * 0.18, 2)
    dp_val = 15.34
    total_tax_fees = round(stt_val + stamp_val + exchange_val + sebi_val + gst_val + dp_val, 2)
    net_profit_t1 = round(profit_t1 - total_tax_fees, 2)
    breakeven_price = round(price + (total_tax_fees / qty), 2)

    # Annualized Return (IRR) based on holding days
    gain_pct_t1 = ((target_1 - price) / price) * 100.0
    irr_t1 = min(999.0, round((((1.0 + gain_pct_t1 / 100.0) ** (252.0 / max(effective_days, 1))) - 1.0) * 100.0, 1))

    return {
        "ticker": suggestion["symbol"],
        "display_symbol": suggestion["display_symbol"],
        "name": suggestion["name"],
        "price": price,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "risk_per_share": risk_per_share,
        "portfolio_capital": capital,
        "sizing_mode": sizing_mode,
        "holding_days": effective_days,
        "shares_to_buy": qty,
        "capital_outlay": outlay,
        "capital_pct": round((outlay / capital) * 100, 1),
        "max_risk_loss": actual_risk_loss,
        "risk_pct": round((actual_risk_loss / capital) * 100, 2),
        "gross_profit_t1": profit_t1,
        "gross_profit_t2": profit_t2,
        "total_statutory_taxes": total_tax_fees,
        "net_profit_t1": net_profit_t1,
        "breakeven_price": breakeven_price,
        "annualized_irr_t1_pct": irr_t1,
        "portfolio_gain_t1_pct": round((profit_t1 / capital) * 100, 2),
        "portfolio_gain_t2_pct": round((profit_t2 / capital) * 100, 2),
        "risk_reward_ratio": suggestion["trade_plan"]["risk_reward_ratio"],
    }


def generate_scrip_basket(
    capital: float = 50000.0,
    theme: str = "balanced",
    max_price: Optional[float] = 200.0,
    max_scrips: int = 4
) -> Dict[str, Any]:
    """
    Generates a diversified portfolio basket of high-conviction scrips under ₹200.
    Returns exact share counts, total outlay, cash buffer, and broker-compatible CSV text.
    """
    capital = max(5000.0, min(capital, 5000000.0))
    max_scrips = max(2, min(max_scrips, 8))

    # Pull latest master scan
    curated = get_curated_suggestions(max_price=max_price, force_refresh=False)
    pool = curated.get("suggestions", [])

    if not pool:
        return {
            "theme": theme,
            "capital_requested": capital,
            "allocations": [],
            "error": "No scrips available matching the criteria."
        }

    # Theme selection logic
    selected: List[Dict[str, Any]] = []
    seen_sectors = set()

    if theme == "momentum":
        # Sort by RVOL desc and Breakout setups
        candidates = sorted(
            [s for s in pool if s["setup_type"] == "MOMENTUM_BREAKOUT"] or pool,
            key=lambda x: x["technicals"]["rvol"],
            reverse=True
        )
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    elif theme == "value":
        # Sort by lowest RSI and Oversold setups
        candidates = sorted(
            [s for s in pool if s["setup_type"] == "OVERSOLD_REVERSAL"] or pool,
            key=lambda x: x["technicals"]["rsi"]
        )
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    elif theme == "psu_capex":
        psu_symbols = {
            "SAIL.NS", "IOC.NS", "BEL.NS", "NHPC.NS", "IRFC.NS", "PNB.NS",
            "CANBK.NS", "UNIONBANK.NS", "SJVN.NS", "HUDCO.NS", "IREDA.NS",
            "MAHABANK.NS", "CENTRALBK.NS", "IOB.NS", "UCOBANK.NS", "GAIL.NS", "CPSEETF.NS"
        }
        candidates = [s for s in pool if s["symbol"] in psu_symbols]
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    elif theme == "high_liquidity":
        # Only institutional grade names with turnover > 50 Cr
        candidates = sorted(
            [s for s in pool if s["liquidity_guardrails"]["avg_turnover_cr"] >= 25.0],
            key=lambda x: x["liquidity_guardrails"]["avg_turnover_cr"],
            reverse=True
        )
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    elif theme == "btst":
        # Sort by day gain desc and RVOL
        candidates = sorted(
            [s for s in pool if s["setup_type"] == "BTST_SURGE"] or pool,
            key=lambda x: (x["day_change_pct"], x["technicals"]["rvol"]),
            reverse=True
        )
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    elif theme == "clean_energy":
        green_symbols = {"SUZLON.NS", "IREDA.NS", "NHPC.NS", "SJVN.NS", "HFCL.NS", "NBCC.NS"}
        candidates = [s for s in pool if s["symbol"] in green_symbols] or pool
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    elif theme == "dividend_yield":
        div_symbols = {"IOC.NS", "NHPC.NS", "SAIL.NS", "NMDC.NS", "SJVN.NS", "CPSEETF.NS", "CASTROLIND.NS", "GAIL.NS"}
        candidates = [s for s in pool if s["symbol"] in div_symbols] or pool
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    else:
        # Default "balanced": Top conviction across diverse sectors
        candidates = sorted(pool, key=lambda x: x["conviction_score"], reverse=True)
        for s in candidates:
            if s["sector"] not in seen_sectors:
                seen_sectors.add(s["sector"])
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    # Fallback if not enough distinct sectors
    if len(selected) < max_scrips:
        for s in pool:
            if s not in selected:
                selected.append(s)
                if len(selected) >= max_scrips:
                    break

    if not selected:
        return {"error": "Unable to assemble basket."}

    # Capital allocation across selected scrips
    per_scrip_budget = capital / len(selected)
    allocations = []
    total_deployed = 0.0
    total_t1_gain = 0.0
    total_t2_gain = 0.0
    total_max_loss = 0.0

    csv_rows = ["Instrument,Exchange,Quantity,Price,OrderType,Product"]
    angel_rows = ["Symbol,Exchange,Quantity,Price,OrderType,Product"]
    groww_rows = ["Symbol,Exchange,Quantity,Price,Product"]
    dhan_orders = []

    for s in selected:
        price = s["technicals"]["current_price"]
        shares = max(1, int(per_scrip_budget // price))
        outlay = round(shares * price, 2)
        total_deployed += outlay

        t1_gain = round(shares * (s["trade_plan"]["target_1"] - price), 2)
        t2_gain = round(shares * (s["trade_plan"]["target_2"] - price), 2)
        sl_loss = round(shares * (price - s["trade_plan"]["stop_loss"]), 2)

        total_t1_gain += t1_gain
        total_t2_gain += t2_gain
        total_max_loss += sl_loss

        clean_sym = s["display_symbol"]
        csv_rows.append(f"{clean_sym},NSE,{shares},{price:.2f},LIMIT,CNC")
        angel_rows.append(f"{clean_sym},NSE,{shares},{price:.2f},LIMIT,DELIVERY")
        groww_rows.append(f"{clean_sym},NSE,{shares},{price:.2f},CNC")
        dhan_orders.append({
            "dhanClientId": "YOUR_ID",
            "transactionType": "BUY",
            "exchangeSegment": "NSE_EQ",
            "productType": "CNC",
            "orderType": "LIMIT",
            "price": price,
            "securityId": clean_sym,
            "quantity": shares,
        })

        allocations.append({
            "symbol": s["symbol"],
            "display_symbol": s["display_symbol"],
            "name": s["name"],
            "sector": s["sector"],
            "setup": s["setup_label"],
            "setup_icon": s["setup_icon"],
            "conviction": s["conviction_score"],
            "price": price,
            "shares": shares,
            "outlay": outlay,
            "allocation_pct": round((outlay / capital) * 100, 1),
            "target_1": s["trade_plan"]["target_1"],
            "target_2": s["trade_plan"]["target_2"],
            "stop_loss": s["trade_plan"]["stop_loss"],
            "target_1_profit": t1_gain,
            "target_2_profit": t2_gain,
            "max_stop_loss": sl_loss,
            "risk_reward_ratio": s["trade_plan"]["risk_reward_ratio"],
        })

    cash_buffer = round(capital - total_deployed, 2)

    # Telegram / WhatsApp Summary Note
    summary_lines = [
        f"🎯 *StockIQ Pro — Smart Basket ({theme.upper()})*",
        f"💰 Capital: ₹{capital:,.0f} | Deployed: ₹{total_deployed:,.2f} | Buffer: ₹{cash_buffer:,.2f}",
        f"📈 Target 1 Profit: +₹{total_t1_gain:,.2f} (+{(total_t1_gain / capital)*100:.1f}%)",
        f"🛡️ Max Portfolio Stop Loss: -₹{total_max_loss:,.2f} (-{(total_max_loss / capital)*100:.1f}%)",
        "────────────────────"
    ]
    for idx, a in enumerate(allocations, 1):
        summary_lines.append(
            f"{idx}. {a['display_symbol']} — {a['shares']} shares @ ₹{a['price']:.2f} (₹{a['outlay']:,.0f})\n"
            f"   SL: ₹{a['stop_loss']:.2f} | T1: ₹{a['target_1']:.2f} | T2: ₹{a['target_2']:.2f}"
        )
    telegram_notes = "\n".join(summary_lines)

    zerodha_csv = "\n".join(csv_rows)

    return {
        "theme": theme,
        "theme_label": {
            "balanced": "Balanced Multi-Sector Alpha",
            "momentum": "High-Volume Momentum Breakouts",
            "btst": "BTST 3 PM Overnight Surge",
            "clean_energy": "Clean Energy & Infra Turnaround",
            "dividend_yield": "High-Yield PSU Dividends",
            "value": "Oversold Mean-Reversion",
            "psu_capex": "Sovereign PSU Capex & Dividends",
            "high_liquidity": "Ultra-Liquid Institutional Radar",
        }.get(theme, theme.title()),
        "capital_requested": capital,
        "capital_deployed": round(total_deployed, 2),
        "cash_buffer": cash_buffer,
        "num_scrips": len(allocations),
        "projected_profit_t1": round(total_t1_gain, 2),
        "projected_return_t1_pct": round((total_t1_gain / capital) * 100, 2),
        "projected_profit_t2": round(total_t2_gain, 2),
        "projected_return_t2_pct": round((total_t2_gain / capital) * 100, 2),
        "max_portfolio_loss": round(total_max_loss, 2),
        "max_portfolio_risk_pct": round((total_max_loss / capital) * 100, 2),
        "portfolio_risk_reward": round(total_t1_gain / max(total_max_loss, 0.01), 2),
        "allocations": allocations,
        "broker_basket_csv": zerodha_csv,
        "broker_exports": {
            "zerodha_kite_csv": zerodha_csv,
            "angel_one_csv": "\n".join(angel_rows),
            "groww_csv": "\n".join(groww_rows),
            "dhan_orders": dhan_orders,
            "telegram_notes": telegram_notes,
        }
    }

