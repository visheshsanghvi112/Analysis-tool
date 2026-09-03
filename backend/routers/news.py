from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from services.intelligent_news_reader import intelligent_news_reader
from news_intelligence import get_advanced_news_analysis

router = APIRouter(prefix="/api", tags=["news"])

@router.get("/advanced-news")
def get_advanced_news_endpoint(
    ticker: str = Query(..., description="Stock ticker symbol, e.g., HDFCBANK.NS"),
    company_name: Optional[str] = Query(None, description="Company name for better news matching")
):
    """
    Returns 100% live news intelligence with Scrapling deep article reading,
    corporate catalyst extraction, and domain-aware financial sentiment.
    """
    try:
        ticker_clean = ticker.strip().upper()
        if not ticker_clean:
            raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty")
        # 1. Primary: Use Scrapling-powered live deep reader
        try:
            news_analysis = intelligent_news_reader.fetch_live_stock_news(ticker_clean, company_name)
        except Exception:
            # Fallback to legacy news intelligence if unexpected error
            news_analysis = get_advanced_news_analysis(ticker_clean, company_name)
        
        return {
            "ticker": ticker_clean,
            "news_intelligence": news_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")
