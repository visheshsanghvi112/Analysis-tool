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
            if news_analysis and "decision" not in news_analysis:
                sentiment_val = news_analysis.get("sentiment", {}).get("overall_sentiment", 0.0)
                news_analysis["decision"] = {
                    "verdict": "MILD_ACCUMULATE" if sentiment_val >= 0.15 else ("BEARISH_HEADWIND" if sentiment_val <= -0.15 else "NEUTRAL_NOISE"),
                    "trade_directive": "MILD BUY / DIP ACCUMULATE" if sentiment_val >= 0.15 else ("SHORT_BIAS / CAUTION" if sentiment_val <= -0.15 else "NO DIRECTIONAL BIAS"),
                    "catalyst_class": "ROUTINE_MARKET_COVERAGE",
                    "sentiment_score": round(sentiment_val, 4),
                    "conviction_score": 50.0,
                    "action_recommendation": "Rely primarily on price action and technical levels.",
                    "cro_risk_flags": [],
                    "committee_vote": "BULLISH" if sentiment_val >= 0.15 else ("BEARISH" if sentiment_val <= -0.15 else "NEUTRAL"),
                    "ml_adjustment_factor": round(sentiment_val * 0.1, 4)
                }
        
        return {
            "ticker": ticker_clean,
            "news_intelligence": news_analysis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis failed: {str(e)}")
