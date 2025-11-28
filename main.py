from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from routers import forecast
from middleware.auth import check_wms_auth
from core.model_loader import load_model_artifacts
from core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print("🚀 Starting up application...")
    load_model_artifacts()
    yield
    # Shutdown: Cleanup if needed
    print("🛑 Shutting down application...")

app = FastAPI(
    title="Demand Forecast API",
    description="A FastAPI microservice to forecast stock demand using an LSTM model.",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.middleware("http")(check_wms_auth)

# Include routers
app.include_router(forecast.router)

@app.get("/")
async def root():
    """Root endpoint - API welcome message."""
    return {
        "message": "Welcome to the Demand Forecast API",
        "status": "running",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )