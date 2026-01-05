from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from app.api.endpoints import router as api_router
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Algorithmic Trading Analytics API",
    description="Advanced quantitative frameworks for institutional trading",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Algorithmic Trading Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "analysis": "/api/v1/analyze",
            "instruments": "/api/v1/instruments",
            "metrics": "/api/v1/metrics"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-15T00:00:00Z"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )