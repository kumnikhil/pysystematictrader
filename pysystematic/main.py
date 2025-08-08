import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager
import asyncio
from typing import Dict
import uvicorn
import logging
import datetime as dt
from src.utils.utils_kite import Kite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store kite instance
kite_obj = None
instruments_cache = None
last_refresh = None

load_dotenv(find_dotenv())
os.environ['API_KEY'] =  os.getenv('API_KEY')
os.environ['API_SECRET'] = os.getenv('API_SECRET')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and cleanup on shutdown"""
    global kite_obj, instruments_cache, last_refresh
    
    # Startup
    logger.info("Starting FastAPI application...")
    try:
        kite_obj = Kite()
        instruments_cache = kite_obj.instruments_df.to_dict(orient="records")
        last_refresh = dt.datetime.now()
        logger.info("Kite session initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Kite session: {e}")
        # Continue without Kite for now
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")

app = FastAPI(
    title="PySysTrader API",
    description="Trading API for developing strategies",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware for better performance
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=Dict[str, str])
async def home():
    """Health check endpoint"""
    return {
        'message': 'Landing point of pysystematic trader app for testing and developing strategies',
        'status': 'healthy',
        'timestamp': dt.datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global kite_obj, last_refresh
    return {
        'status': 'healthy',
        'kite_initialized': kite_obj is not None,
        'last_refresh': last_refresh.isoformat() if last_refresh else None,
        'timestamp': dt.datetime.now().isoformat()
    }


async def refresh_kite_session():
    """Async function to refresh Kite session"""
    global kite_obj, instruments_cache, last_refresh
    logger.info("Refreshing Kite session...")
    kite_obj = Kite()
    instruments_cache = kite_obj.instruments_df.to_dict(orient="records")
    last_refresh = dt.datetime.now()
    logger.info("Kite session refreshed successfully")
    return True
    
@app.get("/refresh_kite")
async def refresh_kite(background_tasks: BackgroundTasks):
    """Refresh Kite session in background"""
    if kite_obj is None:
        raise HTTPException(status_code=503, detail="Kite session not initialized")
    
    # Add refresh task to background
    background_tasks.add_task(refresh_kite_session)
    
    return {
        'message': 'Kite session refresh initiated',
        'status': 'processing'
    }

@app.get("/refresh_kite_sync")
async def refresh_kite_sync():
    """Synchronous refresh endpoint (use sparingly)"""
    success = await refresh_kite_session()
    
    if success:
        return {
            'message': 'Kite session refreshed successfully',
            'timestamp': last_refresh.isoformat()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh Kite session")


@app.get("/get_instruments")
async def get_instruments():
    """Get cached instruments data"""
    global instruments_cache, kite_obj
    
    if kite_obj is None:
        raise HTTPException(status_code=503, detail="Kite session not initialized")
    
    if instruments_cache is None:
        # Try to get fresh data if cache is empty
        try:
            instruments_cache = kite_obj.instruments_df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to get instruments: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve instruments")
    
    return {
        'instruments': instruments_cache,
        'count': len(instruments_cache),
        'last_refresh': last_refresh.isoformat() if last_refresh else None
    }


if __name__ == "__main__":
    # Production configuration
    uvicorn.run(
        "main:app",  # Replace "main" with your filename
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker to maintain shared state, or implement proper state management for multiple workers
        reload=True,  # Set to True only in development
        access_log=True,
        log_level="info"
    )