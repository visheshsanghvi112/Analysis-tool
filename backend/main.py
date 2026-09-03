import os
import threading
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from utils.limiter import limiter
from services.ticker_manager import ensure_ticker_list

# Import all routers from our routers package
from routers import tickers, ml, news, portfolio, analysis

app = FastAPI(
    title="StockIQ Pro API",
    description="Professional Stock Analysis Platform API by Vishesh Sanghvi",
    version="2.0.0"
)

# Rate Limiting configuration
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
if os.getenv("ENVIRONMENT") == "development" or not os.getenv("ALLOWED_ORIGINS"):
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload NSE tickers on startup to avoid delays on initial client queries
@app.on_event("startup")
def startup_event():
    threading.Thread(target=ensure_ticker_list, daemon=True).start()

# Root and health checks
@app.get("/")
def read_root():
    return {
        "name": "StockIQ Pro API",
        "description": "Professional Stock Analysis Platform API",
        "version": "2.0.0",
        "author": "Vishesh Sanghvi",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-routers",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "using_yf_client": True
    }

# Register all APIRouters
app.include_router(tickers.router)
app.include_router(ml.router)
app.include_router(news.router)
app.include_router(portfolio.router)
app.include_router(analysis.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
