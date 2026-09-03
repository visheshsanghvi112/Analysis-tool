from fastapi import APIRouter, Query, HTTPException
from ml_models import get_ml_prediction, retrain_model
from news_intelligence import get_advanced_news_analysis

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
    """
    try:
        ticker_clean = ticker.strip().upper()
        
        # Resolve sentiment score for fusion via live intelligent news reader
        news_sentiment = 0.0
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

        prediction, error = get_ml_prediction(
            ticker_clean, 
            period=period, 
            start_date=start_date, 
            end_date=end_date, 
            news_sentiment=news_sentiment
        )
        
        if error:
            raise HTTPException(status_code=400, detail=error)
        
        return {
            "ticker": ticker_clean,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
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
        success, result = retrain_model(ticker_clean, period=period, start_date=start_date, end_date=end_date)
        
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        return {
            "ticker": ticker_clean,
            "period": period,
            "status": "success",
            "metrics": result,
            "message": "Model retrained successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")
