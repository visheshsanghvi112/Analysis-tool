from fastapi import APIRouter, Query, HTTPException
from ml_models import get_ml_prediction, retrain_model
try:
    from news_intelligence import get_advanced_news_analysis
except ImportError:
    get_advanced_news_analysis = None
from yf_client import get_asset_type, get_info

router = APIRouter(prefix="/api", tags=["ml"])

@router.get("/ml-predict")
def get_ml_prediction_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    period: str = Query("2y", description="Training data time period, e.g., 1y, 2y, 5y"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Returns ML-powered price prediction with confidence intervals, diverse stacked ensemble,
    walk-forward stats, and news sentiment fusion.

    For ETF tickers: uses long-term feature engineering (Golden Cross, quarterly momentum,
    annual volatility), 30-day prediction horizon, ACCUMULATE/AVOID signal language,
    and skips news sentiment (irrelevant for index/commodity ETFs).
    """
    try:
        ticker_clean = ticker.strip().upper()

        # ── Auto-detect asset type (ETF, EQUITY, INDEX) ──────────────────────
        # This is the single source of truth that drives everything downstream.
        asset_type = get_asset_type(ticker_clean)

        # ── News sentiment: only meaningful for individual equities ───────────
        # Index ETFs (NIFTYBEES, GOLDBEES, etc.) have no company-specific news.
        news_sentiment = 0.0
        if asset_type == 'EQUITY':
            try:
                from services.intelligent_news_reader import intelligent_news_reader
                news_res = intelligent_news_reader.fetch_live_stock_news(ticker_clean)
                if news_res and "sentiment" in news_res and "overall_sentiment" in news_res["sentiment"]:
                    news_sentiment = float(news_res["sentiment"]["overall_sentiment"])
            except Exception:
                try:
                    news_res = get_advanced_news_analysis(ticker_clean)
                    if news_res and "sentiment" in news_res and "overall_sentiment" in news_res["sentiment"]:
                        news_sentiment = float(news_res["sentiment"]["overall_sentiment"])
                except Exception:
                    pass

        # ── Auto-upgrade period for ETFs ──────────────────────────────────────
        # ETFs need 5Y history to learn full market cycles.
        # If client sent "1y" or the stock default "2y", silently upgrade to "5y".
        effective_period = period
        if asset_type == 'ETF' and period in ('1y', '2y'):
            effective_period = '5y'

        prediction, error = get_ml_prediction(
            ticker_clean,
            period=effective_period,
            start_date=start_date,
            end_date=end_date,
            news_sentiment=news_sentiment,
            asset_type=asset_type,
        )

        if error:
            raise HTTPException(status_code=400, detail=error)

        if prediction:
            is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
            prediction["currency_symbol"] = "₹" if is_indian else "$"

            # Street Analyst Consensus (for individual equities)
            if asset_type == 'EQUITY':
                try:
                    info = get_info(ticker_clean)
                    target_mean = info.get("targetMeanPrice")
                    target_high = info.get("targetHighPrice")
                    target_low  = info.get("targetLowPrice")
                    num_analysts = info.get("numberOfAnalystOpinions")
                    recommendation = info.get("recommendationKey")
                    rec_mean = info.get("recommendationMean")

                    def _safe_pos_float(val):
                        try:
                            f = float(val)
                            import math
                            return round(f, 2) if (f > 0 and not (math.isnan(f) or math.isinf(f))) else None
                        except (TypeError, ValueError):
                            return None

                    tm = _safe_pos_float(target_mean)
                    if tm is not None:
                        num_opinions = None
                        try:
                            if num_analysts is not None and int(num_analysts) > 0:
                                num_opinions = int(num_analysts)
                        except (TypeError, ValueError):
                            pass

                        prediction["analyst_consensus"] = {
                            "target_mean": tm,
                            "target_high": _safe_pos_float(target_high),
                            "target_low":  _safe_pos_float(target_low),
                            "num_analysts": num_opinions,
                            "recommendation": str(recommendation).replace("_", " ").upper() if recommendation else None,
                            "recommendation_mean": _safe_pos_float(rec_mean),
                        }
                except Exception:
                    pass

        return {
            "ticker": ticker_clean,
            "period": effective_period,
            "start_date": start_date,
            "end_date": end_date,
            "asset_type": asset_type,   # exposed so frontend can adapt its UI
            "prediction": prediction,
            "disclaimer": "Predictions are for educational purposes only. Not financial advice."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")


@router.post("/api/retrain-model")  # keep the /api prefix or relative path
@router.post("/retrain-model")
def retrain_ml_model(
    ticker: str = Query(..., description="Stock ticker to retrain model for"),
    period: str = Query("2y", description="Training data time period, e.g., 1y, 2y, 5y"),
    start_date: str = Query(None, description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(None, description="End date in YYYY-MM-DD format")
):
    """
    Force retrain the ML model with latest data for improved accuracy.
    """
    try:
        ticker_clean = ticker.strip().upper()
        asset_type   = get_asset_type(ticker_clean)
        effective_period = '5y' if (asset_type == 'ETF' and period == '2y') else period

        prediction, error = retrain_model(
            ticker_clean,
            period=effective_period,
            start_date=start_date,
            end_date=end_date,
            asset_type=asset_type,
        )

        if error:
            raise HTTPException(status_code=400, detail=error)

        return {
            "ticker": ticker_clean,
            "period": effective_period,
            "asset_type": asset_type,
            "status": "success",
            "prediction": prediction,
            "metrics": {
                "direction_accuracy": prediction.get("direction_accuracy"),
                "profit_factor": prediction.get("profit_factor"),
                "hit_rate": prediction.get("hit_rate"),
                "models_used": prediction.get("models_used"),
            } if prediction else None,
            "message": "Model retrained successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")



