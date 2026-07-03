from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from news_intelligence import get_advanced_news_analysis

router = APIRouter(prefix="/api", tags=["news"])

@router.get("/advanced-news")
def get_advanced_news_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    company_name: Optional[str] = Query(None, description="Company name for better news matching")
):
    """
    Returns advanced news intelligence with AI sentiment analysis, 
    breaking news detection, and market impact scoring.
    """
    try:
        ticker_clean = ticker.strip().upper()
        news_analysis = get_advanced_news_analysis(ticker_clean, company_name)
        
        return {
            "ticker": ticker_clean,
            "news_intelligence": news_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")
