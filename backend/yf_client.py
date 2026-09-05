# ============================================================
# Yahoo Finance direct REST client — bypasses the yfinance
# Python library which gets blocked on Vercel serverless IPs.
# Uses Yahoo's undocumented but stable v8/v10 JSON endpoints.
# ============================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
from utils.cache import cache_ttl

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
})

_CHART_URL  = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_QUOTE_URL  = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"


def _get(url, params=None, retries=2):
    """GET with simple retry logic."""
    for attempt in range(retries + 1):
        try:
            r = _SESSION.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and attempt < retries:
                time.sleep(1.5)
                continue
        except Exception:
            if attempt < retries:
                time.sleep(1)
    return None


@cache_ttl(seconds=300)
def get_history(ticker: str, period: str = None, interval: str = "1d", start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns OHLCV DataFrame for the given ticker.
    period:   1d 5d 1mo 3mo 6mo 1y 2y 5y
    interval: 1m 5m 15m 1h 1d 1wk 1mo
    start_date: YYYY-MM-DD string
    end_date: YYYY-MM-DD string
    """
    params = {"interval": interval, "includeAdjustedClose": "true"}
    if start_date and end_date:
        try:
            # Parse YYYY-MM-DD strings to epoch timestamps
            dt_start = datetime.strptime(start_date, "%Y-%m-%d")
            dt_end = datetime.strptime(end_date, "%Y-%m-%d")
            params["period1"] = int(dt_start.replace(tzinfo=timezone.utc).timestamp())
            params["period2"] = int(dt_end.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            if period:
                params["range"] = period
            else:
                params["range"] = "1y"
    else:
        if period:
            params["range"] = period
        else:
            params["range"] = "1y"

    data = _get(
        _CHART_URL.format(ticker=ticker),
        params=params,
    )
    if not data:
        return pd.DataFrame()

    try:
        result = data["chart"]["result"]
        if not result:
            return pd.DataFrame()
        r = result[0]
        timestamps = r.get("timestamp", [])
        quote = r["indicators"]["quote"][0]
        adjclose_list = r["indicators"].get("adjclose", [{}])
        adjclose = adjclose_list[0].get("adjclose", []) if adjclose_list else []

        df = pd.DataFrame({
            "Open":   quote.get("open",   []),
            "High":   quote.get("high",   []),
            "Low":    quote.get("low",    []),
            "Close":  quote.get("close",  []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata"))

        if adjclose:
            df["Adj Close"] = adjclose

        df.index.name = "Date"
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


@cache_ttl(seconds=60)
def get_quote(ticker: str) -> dict:
    """
    Returns a live price snapshot dict using the chart meta endpoint.
    Fields: price, prevClose, dayHigh, dayLow, volume, marketCap,
            change, changePct, timestamp
    """
    data = _get(
        _CHART_URL.format(ticker=ticker),
        params={"interval": "1d", "range": "2d"},
    )
    if not data:
        return {}

    try:
        result = data["chart"]["result"]
        if not result:
            return {}
        meta = result[0]["meta"]

        price      = meta.get("regularMarketPrice")
        prev_close = meta.get("chartPreviousClose") or meta.get("previousClose")
        day_high   = meta.get("regularMarketDayHigh")
        day_low    = meta.get("regularMarketDayLow")
        volume     = meta.get("regularMarketVolume")
        mkt_cap    = meta.get("marketCap")

        change = change_pct = None
        if price and prev_close and prev_close != 0:
            change     = round(price - prev_close, 2)
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        gmtoffset = meta.get("gmtoffset")
        market_time = meta.get("regularMarketTime")
        price_date_str = ""
        if market_time is not None:
            if gmtoffset is not None:
                tz = timezone(timedelta(seconds=gmtoffset))
                dt = datetime.fromtimestamp(market_time, tz)
            else:
                dt = datetime.fromtimestamp(market_time, timezone.utc)
            tz_name = meta.get("timezone", "")
            price_date_str = dt.strftime("%Y-%m-%d %H:%M:%S") + (f" {tz_name}" if tz_name else "")

        curr_code = meta.get("currency")
        is_in = ticker.endswith(".NS") or ticker.endswith(".BO") or curr_code == "INR"
        curr_sym = "₹" if is_in else ("$" if curr_code == "USD" else (curr_code or "$"))

        return {
            "price":      round(price, 2) if price else None,
            "prevClose":  round(prev_close, 2) if prev_close else None,
            "dayHigh":    round(day_high, 2) if day_high else None,
            "dayLow":     round(day_low, 2) if day_low else None,
            "volume":     int(volume) if volume else None,
            "marketCap":  mkt_cap,
            "change":     change,
            "changePct":  change_pct,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "price_date": price_date_str,
            "regularMarketTime": market_time,
            "longName":   meta.get("longName"),
            "fiftyTwoWeekHigh": meta.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow":  meta.get("fiftyTwoWeekLow"),
            "currency":   curr_code,
            "currency_symbol": curr_sym,
        }
    except Exception:
        return {}


_CRUMB = None

def _ensure_crumb():
    global _CRUMB
    if _CRUMB is not None:
        return _CRUMB
    try:
        # Fetch fc.yahoo.com to set cookies
        _SESSION.get("https://fc.yahoo.com", headers={"Accept": "*/*"}, timeout=10)
        # Fetch crumb with overridden Accept header (since it returns text/plain, which causes 406 with Accept: application/json)
        r = _SESSION.get("https://query2.finance.yahoo.com/v1/test/getcrumb", headers={"Accept": "*/*"}, timeout=10)
        if r.status_code == 200:
            _CRUMB = r.text.strip()
            print(f"Initialized Yahoo Finance crumb: {_CRUMB}")
            return _CRUMB
        else:
            print(f"Crumb request failed with status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Failed to fetch Yahoo Finance crumb: {e}")
    return None


@cache_ttl(seconds=3600)
def get_info(ticker: str) -> dict:
    """
    Returns fundamental info via quoteSummary (price + defaultKeyStatistics
    + financialData modules).
    """
    global _CRUMB
    crumb = _ensure_crumb()
    params = {"modules": "price,defaultKeyStatistics,financialData,summaryDetail"}
    if crumb:
        params["crumb"] = crumb

    data = _get(
        _QUOTE_URL.format(ticker=ticker),
        params=params,
    )

    # If unauthorized, try resetting crumb and retrying once
    if not data and _CRUMB is not None:
        _CRUMB = None
        crumb = _ensure_crumb()
        if crumb:
            params["crumb"] = crumb
            data = _get(
                _QUOTE_URL.format(ticker=ticker),
                params=params,
            )

    if not data:
        return {}

    try:
        res = data.get("quoteSummary", {}).get("result", [])
        if not res:
            return {}

        out = {}
        for module in res[0].values():
            if isinstance(module, dict):
                for k, v in module.items():
                    if isinstance(v, dict) and "raw" in v:
                        out[k] = v["raw"]
                    elif not isinstance(v, dict):
                        out[k] = v
        return out
    except Exception:
        return {}


@cache_ttl(seconds=3600)
def get_fundamentals_data(ticker: str) -> dict:
    """
    Fetches rich fundamental data for long-term investor analysis:
    - 5-year annual income statement (revenue, net income, EPS)
    - Dividend history, yield, payout ratio, ex-dividend date
    - Ownership breakdown: promoter/insider %, FII/institutions %, retail %
    - Earnings per quarter (last 8 quarters) for trend visualization
    Uses Yahoo Finance quoteSummary with multiple modules.
    """
    global _CRUMB
    crumb = _ensure_crumb()

    modules = ",".join([
        "incomeStatementHistory",
        "incomeStatementHistoryQuarterly",
        "earningsHistory",
        "summaryDetail",
        "defaultKeyStatistics",
        "financialData",
        "majorHoldersBreakdown",
        "insiderHolders",
        "calendarEvents",
        "price",
    ])

    params = {"modules": modules}
    if crumb:
        params["crumb"] = crumb

    def _fetch():
        return _get(_QUOTE_URL.format(ticker=ticker), params=params)

    data = _fetch()
    if not data and _CRUMB is not None:
        _CRUMB = None
        crumb2 = _ensure_crumb()
        if crumb2:
            params["crumb"] = crumb2
        data = _fetch()

    if not data:
        return {}

    try:
        result = data.get("quoteSummary", {}).get("result", [])
        if not result:
            return {}
        r = result[0]

        def raw(d, key):
            v = d.get(key, {})
            if isinstance(v, dict):
                return v.get("raw")
            return v

        # ── 1. Annual income statement (last 5 years) ────────────────────────
        annual_stmts = r.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])
        income_annual = []
        for stmt in annual_stmts:
            end_date = raw(stmt, "endDate")
            year = datetime.fromtimestamp(end_date).year if end_date else None
            income_annual.append({
                "year": year,
                "revenue": raw(stmt, "totalRevenue"),
                "net_income": raw(stmt, "netIncome"),
                "gross_profit": raw(stmt, "grossProfit"),
                "ebit": raw(stmt, "ebit"),
                "eps": raw(stmt, "dilutedEps"),
            })
        income_annual = list(reversed(income_annual))  # oldest first

        # ── 2. Quarterly earnings (last 8 quarters) ──────────────────────────
        quarterly_stmts = r.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", [])
        income_quarterly = []
        for stmt in quarterly_stmts[:8]:
            end_date = raw(stmt, "endDate")
            label = datetime.fromtimestamp(end_date).strftime("%b '%y") if end_date else None
            income_quarterly.append({
                "quarter": label,
                "revenue": raw(stmt, "totalRevenue"),
                "net_income": raw(stmt, "netIncome"),
                "eps": raw(stmt, "dilutedEps"),
            })
        income_quarterly = list(reversed(income_quarterly))  # oldest first

        # ── 3. Dividend data ─────────────────────────────────────────────────
        sd = r.get("summaryDetail", {})
        ks = r.get("defaultKeyStatistics", {})
        calendar = r.get("calendarEvents", {})

        div_yield = raw(sd, "dividendYield")
        div_rate   = raw(sd, "dividendRate")
        payout_ratio = raw(sd, "payoutRatio")
        five_yr_avg_yield = raw(sd, "fiveYearAvgDividendYield")

        # Ex-dividend date
        ex_div_ts = raw(sd, "exDividendDate")
        ex_div_date = None
        if ex_div_ts:
            try:
                ex_div_date = datetime.fromtimestamp(ex_div_ts).strftime("%d %b %Y")
            except Exception:
                pass

        # Last split info
        last_split_factor = raw(ks, "lastSplitFactor")
        last_split_date_ts = raw(ks, "lastSplitDate")
        last_split_date = None
        if last_split_date_ts:
            try:
                last_split_date = datetime.fromtimestamp(last_split_date_ts).strftime("%d %b %Y")
            except Exception:
                pass

        # ── 4. Dividend history from v8 chart (with events) ─────────────────
        div_history = []
        try:
            chart_data = _get(
                _CHART_URL.format(ticker=ticker),
                params={"interval": "1d", "range": "5y", "events": "dividends"}
            )
            if chart_data:
                events = chart_data.get("chart", {}).get("result", [{}])[0].get("events", {})
                dividends_raw = events.get("dividends", {})
                for ts_str, div_info in dividends_raw.items():
                    ts = int(ts_str)
                    amount = div_info.get("amount", 0)
                    div_history.append({
                        "date": datetime.fromtimestamp(ts).strftime("%d %b %Y"),
                        "year": datetime.fromtimestamp(ts).year,
                        "amount": round(float(amount), 2),
                    })
                div_history = sorted(div_history, key=lambda x: x["date"])
        except Exception:
            pass

        # Aggregate dividends per year for bar chart
        div_by_year = {}
        for d in div_history:
            yr = d["year"]
            div_by_year[yr] = round(div_by_year.get(yr, 0) + d["amount"], 2)
        div_annual = [{"year": yr, "dividend": amt} for yr, amt in sorted(div_by_year.items())]

        # ── 5. Ownership / shareholding breakdown ────────────────────────────
        mhb = r.get("majorHoldersBreakdown", {})
        insiders_pct    = raw(mhb, "insidersPercentHeld")   # promoter / insider group
        institutions_pct = raw(mhb, "institutionsPercentHeld")  # FII + DII
        retail_pct = None
        if insiders_pct is not None and institutions_pct is not None:
            retail_pct = max(0.0, 1.0 - insiders_pct - institutions_pct)

        # Top institutional holders
        inst_holders_raw = r.get("insiderHolders", {}).get("holders", [])
        top_insiders = []
        for h in inst_holders_raw[:5]:
            top_insiders.append({
                "name": h.get("name", ""),
                "relation": h.get("relation", ""),
                "shares": raw(h, "shares"),
                "transaction": h.get("transactionDescription", ""),
                "date": raw(h, "latestTransDate"),
            })

        # ── 6. Price CAGR from history ────────────────────────────────────────
        price_cagr = {}
        try:
            price_module = r.get("price", {})
            current_price = raw(price_module, "regularMarketPrice")
            if current_price:
                for years, period in [(1, "1y"), (3, "3y"), (5, "5y")]:
                    hist = get_history(ticker, period=period)
                    if hist is not None and not hist.empty and len(hist) > 10:
                        start_price = float(hist["Close"].iloc[0])
                        if start_price > 0:
                            cagr = ((current_price / start_price) ** (1 / years) - 1) * 100
                            price_cagr[f"{years}y"] = round(cagr, 2)
        except Exception:
            pass

        # ── 7. Key ratios ─────────────────────────────────────────────────────
        fd = r.get("financialData", {})
        total_cash = raw(fd, "totalCash")
        total_debt = raw(fd, "totalDebt")
        net_debt = (total_debt - total_cash) if (total_debt is not None and total_cash is not None) else None
        ebitda = raw(fd, "ebitda")
        net_debt_to_ebitda = None
        if net_debt is not None and ebitda and ebitda > 0:
            net_debt_to_ebitda = round(net_debt / ebitda, 2)

        ratios = {
            "pe_ratio": raw(sd, "trailingPE") or raw(ks, "trailingPE"),
            "forward_pe": raw(sd, "forwardPE") or raw(ks, "forwardPE"),
            "peg_ratio": raw(ks, "pegRatio"),
            "pb_ratio": raw(ks, "priceToBook") or raw(sd, "priceToBook"),
            "enterprise_to_ebitda": raw(ks, "enterpriseToEbitda"),
            "roe": raw(fd, "returnOnEquity"),
            "roa": raw(fd, "returnOnAssets"),
            "debt_to_equity": raw(fd, "debtToEquity"),
            "total_cash": total_cash,
            "total_debt": total_debt,
            "net_debt": net_debt,
            "ebitda": ebitda,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "current_ratio": raw(fd, "currentRatio"),
            "revenue_growth": raw(fd, "revenueGrowth"),
            "earnings_growth": raw(fd, "earningsGrowth"),
            "gross_margins": raw(fd, "grossMargins"),
            "profit_margins": raw(fd, "profitMargins"),
            "operating_margins": raw(fd, "operatingMargins"),
        }

        # ── 8. 3-Year Compounded Growth (Screener.in style) ───────────────────
        sales_cagr_3y = None
        profit_cagr_3y = None
        if len(income_annual) >= 4:
            r_start = income_annual[0].get("revenue")
            r_end = income_annual[-1].get("revenue")
            if r_start and r_end and r_start > 0 and r_end > 0:
                sales_cagr_3y = round(((r_end / r_start) ** (1.0 / 3.0) - 1.0) * 100.0, 1)

            p_start = income_annual[0].get("net_income")
            p_end = income_annual[-1].get("net_income")
            if p_start and p_end and p_start > 0 and p_end > 0:
                profit_cagr_3y = round(((p_end / p_start) ** (1.0 / 3.0) - 1.0) * 100.0, 1)
        elif len(income_annual) >= 2:
            yrs = max(1, len(income_annual) - 1)
            r_start = income_annual[0].get("revenue")
            r_end = income_annual[-1].get("revenue")
            if r_start and r_end and r_start > 0 and r_end > 0:
                sales_cagr_3y = round(((r_end / r_start) ** (1.0 / yrs) - 1.0) * 100.0, 1)

            p_start = income_annual[0].get("net_income")
            p_end = income_annual[-1].get("net_income")
            if p_start and p_end and p_start > 0 and p_end > 0:
                profit_cagr_3y = round(((p_end / p_start) ** (1.0 / yrs) - 1.0) * 100.0, 1)

        # ── 9. Automated Pros & Cons Digest (Screener.in style) ──────────────
        pros = []
        cons = []

        de_val = ratios.get("debt_to_equity")
        de = (de_val / 100.0) if (de_val is not None and de_val > 2.0) else de_val
        peg = ratios.get("peg_ratio")
        opm = ratios.get("operating_margins")
        promoter = round(insiders_pct * 100, 1) if insiders_pct is not None else None
        div_y = round(div_yield * 100, 2) if div_yield else None
        earn_growth = ratios.get("earnings_growth")
        pe = ratios.get("pe_ratio")

        # Pros Evaluation
        if net_debt is not None and net_debt <= 0:
            pros.append("Company is virtually debt-free with net cash reserves.")
        elif de is not None and de <= 0.6:
            pros.append(f"Prudent balance sheet with low financial leverage (D/E: {de:.2f}x).")

        if peg is not None and 0 < peg < 1.0:
            pros.append(f"PEG ratio of {peg:.2f} indicates earnings growth is trading at an attractive multiple.")

        if opm is not None and opm >= 0.12:
            pros.append(f"Strong operating profitability margin of {opm * 100:.1f}% reflecting institutional pricing power.")

        if sales_cagr_3y is not None and sales_cagr_3y >= 10.0:
            pros.append(f"Healthy medium-term revenue compounding of +{sales_cagr_3y:.1f}% p.a. over the past 3 years.")

        if promoter is not None and promoter >= 50.0:
            pros.append(f"High promoter commitment with {promoter:.1f}% equity skin-in-the-game.")

        if net_debt_to_ebitda is not None and 0 < net_debt_to_ebitda <= 1.5:
            pros.append(f"Net debt is conservative and fully repayable within {net_debt_to_ebitda:.1f} years of operating EBITDA.")

        if div_y is not None and div_y >= 1.0:
            pros.append(f"Provides tangible shareholder cash returns with a {div_y:.2f}% dividend yield.")

        # Cons Evaluation
        if earn_growth is not None and earn_growth < -0.05:
            cons.append(f"Recent quarterly net earnings contracted by {abs(earn_growth * 100):.1f}% YoY.")

        if profit_cagr_3y is not None and profit_cagr_3y <= 3.0 and (sales_cagr_3y or 0) > 8.0:
            cons.append(f"3-year profit compounding has lagged revenue ({profit_cagr_3y:+.1f}% p.a.), signaling margin pressure.")

        if pe is not None and pe >= 35.0:
            cons.append(f"Stock trades at a rich valuation multiple of {pe:.1f}x P/E, requiring sustained high execution.")

        if peg is not None and peg >= 2.2:
            cons.append(f"Growth is priced at a premium with a PEG ratio of {peg:.2f}.")

        if de is not None and de > 1.2:
            cons.append(f"Elevated debt-to-equity ratio of {de:.2f}x increases vulnerability to rising interest rates.")

        if promoter is not None and promoter < 30.0:
            cons.append(f"Low promoter holding ({promoter:.1f}%) exposes the stock to potential ownership dilution.")

        # Fallback guarantees
        if not pros:
            pros.append("Established market footprint and operational presence in its sector.")
            if opm is not None and opm > 0:
                pros.append(f"Maintains positive operating margins of {opm * 100:.1f}%.")

        if not cons:
            if div_y is None or div_y == 0:
                cons.append("Company does not currently pay dividends, retaining cash for operations.")
            cons.append("Subject to broader industry cyclicality and macroeconomic market conditions.")

        return {
            "ticker": ticker,
            "income_annual": income_annual,
            "income_quarterly": income_quarterly,
            "sales_cagr_3y": sales_cagr_3y,
            "profit_cagr_3y": profit_cagr_3y,
            "net_debt_to_ebitda": net_debt_to_ebitda,
            "pros_and_cons": {
                "pros": pros[:4],
                "cons": cons[:4],
            },
            "dividend": {
                "yield_pct": round(div_yield * 100, 2) if div_yield else None,
                "rate": div_rate,
                "payout_ratio_pct": round(payout_ratio * 100, 1) if payout_ratio else None,
                "five_yr_avg_yield_pct": round(five_yr_avg_yield, 2) if five_yr_avg_yield else None,
                "ex_dividend_date": ex_div_date,
                "last_split_factor": last_split_factor,
                "last_split_date": last_split_date,
                "history": div_history[-20:],  # last 20 payments
                "annual_totals": div_annual[-6:],  # last 6 years
            },
            "ownership": {
                "promoter_pct": round(insiders_pct * 100, 2) if insiders_pct is not None else None,
                "institutions_pct": round(institutions_pct * 100, 2) if institutions_pct is not None else None,
                "retail_pct": round(retail_pct * 100, 2) if retail_pct is not None else None,
                "top_insiders": top_insiders,
            },
            "price_cagr": price_cagr,
            "ratios": ratios,
        }
    except Exception as e:
        print(f"[FUNDAMENTALS] Error parsing data for {ticker}: {e}")
        return {}

