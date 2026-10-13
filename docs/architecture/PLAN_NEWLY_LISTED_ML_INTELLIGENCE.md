# StockIQ Pro — Intelligent Handling of Newly Listed Stocks & Adaptive ML History

## 1. Executive Summary & Vision Alignment
When a user analyzes an equity or ETF in the **AI Trend Outlook / ML Prediction** section:
- If an asset was listed recently (e.g. 1 year ago or a few weeks/months ago), requesting **3Y, 5Y, or 10Y** data historically caused abrupt failures (`Insufficient historical data`).
- Even when an asset had sufficient data to train (e.g., 1.2 years of data), users were misled into thinking 3Y/5Y was analyzed, without knowing the asset's true listing inception.
- If an asset was brand new (e.g., IPO 2 weeks ago), the system threw a raw, blunt error without explaining **why** machine learning requires incubation or **when** the asset was listed.

### The Core Vision
1. **Intelligent Auto-Adaptation to Available History**:
   Whatever the maximum period of the asset is on the exchange, the AI engine must automatically adapt to that maximum span and successfully deliver predictive analytics, rather than crashing on an arbitrary 3Y/5Y selection.
2. **Transparent Inception & History Tracking**:
   The system must explicitly detect and state the asset's **First Trade Date / Listing Inception** (e.g., *"Listed on Nov 13, 2024 • 11 months of trading history on NSE"*), so users understand the asset's maturity profile.
3. **Institutional, Explanatory Error & Incubation Handling**:
   If an asset is in its initial incubation window (<50 trading sessions), the system must **not** display a generic red error. Instead, it must render an informative **Asset Incubation & Readiness Tracker** that explains:
   - Why machine learning needs ~50 sessions (feature extraction, rolling volatility, walk-forward validation without overfitting).
   - The exact number of sessions recorded vs required.
   - An estimated date of AI activation.
   - Immediate alternative modules available right now (Live Technicals, VWAP, Order Flow, News Catalysts).
4. **Structured Error Handling on All Parameters**:
   All parameters (custom date ranges, invalid horizons, holidays) must return structured, contextual diagnostic objects with clear explanations and recovery suggestions.

---

## 2. End-to-End Architectural Blueprint

```mermaid
graph TD
    UserReq[User Requests Period: 3Y / 5Y / 10Y / Max / Custom] --> API[/api/ml-predict]
    
    API --> Inception[Asset Inception & History Inspector]
    Inception -->|Fetch Chart Meta & Range| YF[Yahoo Finance Chart / Exchange Engine]
    
    YF --> CheckCandles{Candles Available?}
    
    CheckCandles -->|< 50 Sessions| Incubation[Incubation Mode: Return Structured Explanatory Diagnostics]
    CheckCandles -->|50 to Requested Sessions| AdaptiveTrain[Adaptive Training: Auto-Fit to Max Inception Window]
    CheckCandles -->|>= Requested Sessions| StandardTrain[Standard Training: Use Requested Window]
    
    Incubation --> FE_Incubation[Frontend: Asset Incubation & Readiness Card]
    
    AdaptiveTrain --> ML_Engine[6-Model Stacked Ensemble + SHAP]
    StandardTrain --> ML_Engine
    
    ML_Engine --> Telemetry[Attach History Metadata & Inception Telemetry]
    Telemetry --> FE_Success[Frontend: Render AI Forecast + Inception Adaptation Pill]
```

---

## 3. Detailed Component Plan

### A. Backend History & Inception Inspector (`backend/yf_client.py`)
- Enhance `get_history()` and add `get_ticker_inception_metadata(ticker)`:
  - Extract `firstTradeDate` epoch from Yahoo Finance chart `meta`.
  - Calculate `first_trade_date` (YYYY-MM-DD), `last_trade_date`, `total_trading_days`, and `calendar_age_human` (e.g. "8 months", "1.4 years", "12 years").
  - Cache inception metadata in memory for sub-millisecond retrieval.

### B. Adaptive Training Engine in ML Models (`backend/ml_models.py`)
- **Dynamic Thresholding**:
  - Standard Equity: 120 trading days.
  - Adaptive Young Equity: Down to 50 trading days (dynamically scales rolling window indicators like 5d/10d/20d MA without crashing on 50d/100d MA).
  - Standard ETF: 252 trading days (1 full year).
  - Adaptive Young ETF: Down to 80 trading days (with warning flag).
- **Auto-Period Scaling**:
  - If requested period is `'5y'` (approx 1,260 trading days) but the asset has only 280 trading days:
    - Automatically train on all 280 trading days (`effective_period = 'max'`).
    - Tag response with `is_adapted: true`, `requested_period: '5y'`, `effective_period: 'max (1.1Y)'`, and `adaptation_note: 'Requested 5Y period auto-adapted to full 1.1Y of trading history since IPO on 13 Nov 2024.'`.

### C. Explanatory Diagnostic Error Response (`backend/routers/ml.py`)
- Replace generic `raise HTTPException(status_code=400, detail=str(error))` with structured responses:
  - When sessions < 50:
    ```json
    {
      "status": "INCUBATION_PERIOD",
      "error_code": "INSUFFICIENT_HISTORY_NEW_LISTING",
      "ticker": "EXAMPLE.NS",
      "title": "Newly Listed Asset in Incubation Window",
      "first_trade_date": "2026-09-18",
      "total_trading_days": 16,
      "required_trading_days": 50,
      "readiness_pct": 32.0,
      "explanation": "This asset commenced exchange trading on 18 Sep 2026 and has only 16 sessions recorded. Supervised ML ensembles require at least 50 trading sessions to compute multi-period moving averages, volatility clustering, and walk-forward validation without severe lookahead bias.",
      "estimated_activation": "Nov 2026",
      "available_modules": ["Technical Screener", "VWAP & Intraday Desk", "News Catalyst Reader", "Order Flow Analysis"]
    }
    ```
  - For Custom Date errors:
    - Explain exact issue (e.g. *"Start date is later than end date"* or *"Zero trading sessions took place in the specified date range"*).

### D. Upgraded Frontend UI (`frontend/src/app/components/MLPrediction.js`)
1. **Adaptive History Pill & Listing Info**:
   - In the header of the AI forecast card, display a prominent metadata pill:
     - 🗓️ `Listed 13 Nov 2024 • 11 months history`
     - ⚡ `Auto-adapted from requested 3Y to maximum available 11M history`
2. **Period Selector Intelligence**:
   - If asset history is less than an option (e.g. stock is 1.5Y old, but buttons show `3Y`, `5Y`), indicate `3Y (Max: 1.5Y)` or dynamically disable periods beyond `Max`, or show tooltips.
3. **Institutional Incubation Card (Replacing harsh error)**:
   - When status is `INCUBATION_PERIOD`:
     - Beautiful visual incubator card with animated gauge / progress bar.
     - Displays listing date, elapsed sessions, and sessions needed.
     - Plain-English explanation.
     - Actionable buttons to jump to **Intraday VWAP Desk** or **News Catalysts Desk**.
4. **Resilient Parameter Error Card**:
   - For any other parameter issue, display an intuitive troubleshooting card with single-click "Reset to Max Available" or "Adjust Dates" buttons.

---

## 4. Verification & Validation Matrix
1. **Recent IPO Test**: Test with stocks listed within the last 1–2 years (e.g., `SWIGGY.NS`, `HYUNDAI.NS`, `IREDA.NS`, `TATATECH.NS`). Verify that selecting `3Y`, `5Y`, or `10Y` auto-adapts to maximum available data, displays inception info, and renders predictions cleanly.
2. **Ultra-New Listing Test**: Simulate or test a stock with <50 trading days. Verify that the system renders the structured Incubation & Readiness card instead of an unhandled error.
3. **Custom Date Edge Cases**: Test invalid start/end dates. Verify friendly explanatory diagnostics.
4. **Automated Unit Tests**: Add pytest test cases in `backend/tests/` to verify adaptive training thresholds, inception telemetry, and structured error schemas.
5. **Full Suite Health**: Maintain 100% pass rate across the full 114+ test suite and ensure clean Next.js build compilation.
