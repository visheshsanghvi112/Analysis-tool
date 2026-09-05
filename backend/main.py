import os
import logging
import threading
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from utils.limiter import limiter
from services.ticker_manager import ensure_ticker_list

# Import all routers from our routers package
from routers import tickers, ml, news, portfolio, analysis, intraday

logger = logging.getLogger("stockiq")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure 7,954 master ticker universe is pre-warmed
    threading.Thread(target=ensure_ticker_list, daemon=True).start()
    yield
    # Shutdown logic (if any cleanup is needed in the future)

app = FastAPI(
    title="StockIQ Pro API",
    description="Professional Stock Analysis Platform API by Vishesh Sanghvi",
    version="2.4.0",
    lifespan=lifespan
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

# Global defensive error handler for unexpected server errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An internal error occurred while processing the request.",
            "detail": str(exc),
            "path": request.url.path
        }
    )

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
app.include_router(intraday.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
