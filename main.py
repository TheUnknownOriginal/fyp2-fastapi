"""
FastAPI Prediction Service for LSTM Demand Forecasting
Provides endpoints for 1-day, 7-day, and 30-day demand predictions
Supports single and multiple item predictions
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import io
from datetime import datetime
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from database import get_db, Prediction, create_tables

# ============================================================================
# Initialize FastAPI App
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print("Starting up application...")
    # Create tables if they don't exist
    try:
        create_tables()
        print("[OK] Database tables verified")
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
        
    await load_model_and_scalers()
    yield
    # Shutdown: Cleanup if needed
    print("Shutting down application...")

app = FastAPI(
    title="LSTM Demand Forecasting API",
    description="Multi-horizon demand forecasting for warehouse-to-supermarket operations",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# Global Variables (Load on Startup)
# ============================================================================
MODEL = None
FEATURE_SCALER = None
TARGET_SCALER = None
ITEM_ID_MAPPING = None
FEATURE_NAMES = None

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================
class PredictionRequest(BaseModel):
    item_ids: List[str] = Field(..., description="List of item IDs to predict (can be single item)")
    csv_data: Optional[str] = Field(None, description="CSV data as string (optional if using file upload)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "item_ids": ["2010048801", "1090092001"],
                "csv_data": None
            }
        }

class SinglePrediction(BaseModel):
    item_id: str
    item_desc: Optional[str] = None
    predicted_demand: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    last_30_days_avg: Optional[float] = None

class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    forecast_horizon: str
    model_used: str
    timestamp: str
    total_items: int
    mape: float = Field(..., description="Model's Mean Absolute Percentage Error")

class SavePredictionRequest(BaseModel):
    item_id: str
    item_desc: Optional[str] = None
    predicted_demand: float
    forecast_horizon: str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    model_version: str = "v1.0"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_items: int
    timestamp: str

# ============================================================================
# Startup Event - Load Model and Scalers
# ============================================================================
async def load_model_and_scalers():
    """Load model and preprocessing tools on startup"""
    global MODEL, FEATURE_SCALER, TARGET_SCALER, ITEM_ID_MAPPING, FEATURE_NAMES
    
    print("="*70)
    print("LOADING LSTM MODEL AND SCALERS")
    print("="*70)
    
    try:
        # Load best model
        model_path = "models/large_batch.keras"
        MODEL = tf.keras.models.load_model(model_path)
        print(f"[OK] Loaded model: {model_path}")
        print(f"  Parameters: {MODEL.count_params():,}")
        
        # Load scalers
        with open('preprocessed/scaler.pkl', 'rb') as f:
            FEATURE_SCALER = pickle.load(f)
        print("[OK] Loaded feature scaler")
        
        with open('preprocessed/target_scaler.pkl', 'rb') as f:
            TARGET_SCALER = pickle.load(f)
        print("[OK] Loaded target scaler")
        
        with open('preprocessed/item_id_mapping.pkl', 'rb') as f:
            mapping = pickle.load(f)
            # Ensure all keys are strings and stripped for consistent matching
            ITEM_ID_MAPPING = {str(k).strip(): v for k, v in mapping.items()}
        print(f"[OK] Loaded item mapping ({len(ITEM_ID_MAPPING)} items)")
        
        # Load feature names
        data = np.load('preprocessed/data.npz')
        FEATURE_NAMES = list(data['feature_names'])
        print(f"[OK] Loaded feature names ({len(FEATURE_NAMES)} features)")
        
        print("="*70)
        print("MODEL READY FOR PREDICTIONS")
        print("="*70)
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        raise

# ============================================================================
# Helper Functions
# ============================================================================
def preprocess_data(df: pd.DataFrame, item_ids: List[str]) -> Dict:
    """Preprocess data for prediction"""
    
    # Convert item_ids to strings and strip whitespace
    item_ids_clean = [str(id).strip() for id in item_ids]
    
    # Convert DataFrame item_id to string and strip whitespace
    df['item_id'] = df['item_id'].astype(str).str.strip()
    
    # Filter for requested items
    df_filtered = df[df['item_id'].isin(item_ids_clean)].copy()
    
    if df_filtered.empty:
        raise ValueError(f"No data found for items: {item_ids}")
    
    # Ensure date column is datetime
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered = df_filtered.sort_values(['item_id', 'date'])
    
    # Add temporal features
    df_filtered['day_of_week'] = df_filtered['date'].dt.dayofweek
    df_filtered['is_weekend'] = (df_filtered['day_of_week'] >= 5).astype(int)
    df_filtered['month'] = df_filtered['date'].dt.month
    df_filtered['week_of_year'] = df_filtered['date'].dt.isocalendar().week
    df_filtered['day_of_month'] = df_filtered['date'].dt.day
    
    # Add lagged features
    for lag in [1, 7, 30]:
        df_filtered[f'qty_ordered_lag_{lag}'] = (
            df_filtered.groupby('item_id')['qty_ordered']
            .shift(lag)
            .fillna(0)  # Handle cases with less history than the lag
        )
    
    # Add rolling features
    for window in [7, 30]:
        df_filtered[f'qty_ordered_roll_mean_{window}'] = (
            df_filtered.groupby('item_id')['qty_ordered']
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df_filtered[f'qty_ordered_roll_std_{window}'] = (
            df_filtered.groupby('item_id')['qty_ordered']
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    
    # Add days since last order
    df_filtered['days_since_last_order'] = (
        df_filtered.groupby('item_id')['date'].diff().dt.days.fillna(0)
    )
    
    # Encode item IDs
    df_filtered['item_id_encoded'] = df_filtered['item_id'].map(ITEM_ID_MAPPING)
    
    # Drop rows with NaN
    df_filtered = df_filtered.dropna()
    
    return df_filtered

def create_sequences(df: pd.DataFrame, seq_length: int = 30) -> Dict:
    """Create sequences for each item"""
    sequences = {}
    
    for item_id in df['item_id'].unique():
        item_data = df[df['item_id'] == item_id].copy()
        
        if len(item_data) < seq_length:
            print(f"⚠️ Warning: Item {item_id} has only {len(item_data)} days of data (need {seq_length})")
            continue
        
        # Get last 30 days
        last_30_days = item_data.tail(seq_length)
        
        # Extract features
        sequence = last_30_days[FEATURE_NAMES].values
        
        # Scale features
        sequence_scaled = FEATURE_SCALER.transform(sequence)
        
        # Get item encoding
        item_id_encoded = last_30_days['item_id_encoded'].iloc[0]
        
        # Get item description
        item_desc = last_30_days['item_desc'].iloc[0] if 'item_desc' in last_30_days.columns else None
        
        # Calculate last 30 days average
        last_30_avg = last_30_days['qty_ordered'].mean()
        
        sequences[item_id] = {
            'sequence': sequence_scaled,
            'item_id_encoded': item_id_encoded,
            'item_desc': item_desc,
            'last_30_avg': last_30_avg
        }
    
    return sequences

def make_predictions(sequences: Dict, horizon: str) -> List[SinglePrediction]:
    """Make predictions for all items"""
    predictions = []
    
    # Prepare batch inputs
    X_seq_batch = []
    X_item_batch = []
    item_ids_batch = []
    
    for item_id, data in sequences.items():
        X_seq_batch.append(data['sequence'])
        X_item_batch.append(data['item_id_encoded'])
        item_ids_batch.append(item_id)
    
    if not item_ids_batch:
        raise HTTPException(
            status_code=400, 
            detail="None of the requested items have enough historical data (need at least 30 days) to generate a prediction."
        )
    
    # Convert to numpy arrays
    X_seq_batch = np.array(X_seq_batch)
    X_item_batch = np.array(X_item_batch)
    
    # Make predictions
    try:
        model_predictions = MODEL.predict([X_seq_batch, X_item_batch], verbose=0)
        # Handle different model output formats
        if isinstance(model_predictions, list):
            pred_1d, pred_7d, pred_30d = model_predictions
        else:
            # If model returns a single array with 3 values per prediction
            pred_1d = model_predictions[:, 0:1]
            pred_7d = model_predictions[:, 1:2]
            pred_30d = model_predictions[:, 2:3]
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    
    # Select appropriate horizon
    if horizon == "1d":
        preds = pred_1d
        horizon_idx = 0
    elif horizon == "7d":
        preds = pred_7d
        horizon_idx = 1
    else:  # 30d
        preds = pred_30d
        horizon_idx = 2
    
    # Inverse transform predictions
    for i, item_id in enumerate(item_ids_batch):
        # Create array for inverse transform
        if horizon_idx == 0:
            transform_array = np.array([[preds[i][0], 0, 0]])
        elif horizon_idx == 1:
            transform_array = np.array([[0, preds[i][0], 0]])
        else:
            transform_array = np.array([[0, 0, preds[i][0]]])
        
        # Inverse transform
        pred_original = TARGET_SCALER.inverse_transform(transform_array)[0, horizon_idx]
        
        # Calculate confidence interval (±20% as approximation)
        ci_lower = pred_original * 0.8
        ci_upper = pred_original * 1.2
        
        predictions.append(SinglePrediction(
            item_id=item_id,
            item_desc=sequences[item_id]['item_desc'],
            predicted_demand=float(pred_original),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            last_30_days_avg=float(sequences[item_id]['last_30_avg'])
        ))
    
    return predictions

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LSTM Demand Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_1d": "/predict/1-day",
            "predict_7d": "/predict/7-day",
            "predict_30d": "/predict/30-day",
            "predict_all": "/predict/all-horizons"
        },
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        available_items=len(ITEM_ID_MAPPING) if ITEM_ID_MAPPING else 0,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/1-day", response_model=PredictionResponse)
async def predict_1_day(
    item_ids: str = Form(..., description="Comma-separated item IDs"),
    file: UploadFile = File(..., description="CSV file with historical data")
):
    """Predict 1-day demand for specified items"""
    try:
        # Parse item IDs
        item_id_list = [id.strip() for id in item_ids.split(',')]
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess data
        df_processed = preprocess_data(df, item_id_list)
        
        # Create sequences
        sequences = create_sequences(df_processed)
        
        # Make predictions
        predictions = make_predictions(sequences, "1d")
        
        return PredictionResponse(
            predictions=predictions,
            forecast_horizon="1-day",
            model_used="large_batch",
            timestamp=datetime.now().isoformat(),
            total_items=len(predictions),
            mape=228.2  # From model results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/7-day", response_model=PredictionResponse)
async def predict_7_day(
    item_ids: str = Form(..., description="Comma-separated item IDs"),
    file: UploadFile = File(..., description="CSV file with historical data")
):
    """Predict 7-day demand for specified items"""
    try:
        # Parse item IDs
        item_id_list = [id.strip() for id in item_ids.split(',')]
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess data
        df_processed = preprocess_data(df, item_id_list)
        
        # Create sequences
        sequences = create_sequences(df_processed)
        
        # Make predictions
        predictions = make_predictions(sequences, "7d")
        
        return PredictionResponse(
            predictions=predictions,
            forecast_horizon="7-day",
            model_used="large_batch",
            timestamp=datetime.now().isoformat(),
            total_items=len(predictions),
            mape=29.9  # From model results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/30-day", response_model=PredictionResponse)
async def predict_30_day(
    item_ids: str = Form(..., description="Comma-separated item IDs"),
    file: UploadFile = File(..., description="CSV file with historical data")
):
    """Predict 30-day demand for specified items"""
    try:
        # Parse item IDs
        item_id_list = [id.strip() for id in item_ids.split(',')]
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess data
        df_processed = preprocess_data(df, item_id_list)
        
        # Create sequences
        sequences = create_sequences(df_processed)
        
        # Make predictions
        predictions = make_predictions(sequences, "30d")
        
        return PredictionResponse(
            predictions=predictions,
            forecast_horizon="30-day",
            model_used="large_batch",
            timestamp=datetime.now().isoformat(),
            total_items=len(predictions),
            mape=17.1  # From model results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/all-horizons")
async def predict_all_horizons(
    item_ids: str = Form(..., description="Comma-separated item IDs"),
    file: UploadFile = File(..., description="CSV file with historical data")
):
    """Predict all horizons (1-day, 7-day, 30-day) for specified items"""
    try:
        # Parse item IDs
        item_id_list = [id.strip() for id in item_ids.split(',')]
        
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess data
        df_processed = preprocess_data(df, item_id_list)
        
        # Create sequences
        sequences = create_sequences(df_processed)
        
        # Make predictions for all horizons
        predictions_1d = make_predictions(sequences, "1d")
        predictions_7d = make_predictions(sequences, "7d")
        predictions_30d = make_predictions(sequences, "30d")
        
        # Combine results
        combined_predictions = []
        for i, item_id in enumerate(item_id_list):
            if item_id in sequences:
                combined_predictions.append({
                    "item_id": item_id,
                    "item_desc": sequences[item_id]['item_desc'],
                    "predictions": {
                        "1_day": {
                            "demand": predictions_1d[i].predicted_demand,
                            "confidence_interval": [
                                predictions_1d[i].confidence_interval_lower,
                                predictions_1d[i].confidence_interval_upper
                            ]
                        },
                        "7_day": {
                            "demand": predictions_7d[i].predicted_demand,
                            "confidence_interval": [
                                predictions_7d[i].confidence_interval_lower,
                                predictions_7d[i].confidence_interval_upper
                            ],
                            "reorder_point": predictions_7d[i].predicted_demand * 1.35  # With 35% safety stock
                        },
                        "30_day": {
                            "demand": predictions_30d[i].predicted_demand,
                            "confidence_interval": [
                                predictions_30d[i].confidence_interval_lower,
                                predictions_30d[i].confidence_interval_upper
                            ],
                            "recommended_order_qty": predictions_30d[i].predicted_demand
                        }
                    },
                    "last_30_days_avg": sequences[item_id]['last_30_avg']
                })
        
        return {
            "predictions": combined_predictions,
            "model_used": "large_batch",
            "timestamp": datetime.now().isoformat(),
            "total_items": len(combined_predictions),
            "model_accuracy": {
                "1_day_mape": 228.2,
                "7_day_mape": 29.9,
                "30_day_mape": 17.1
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items", response_model=Dict)
async def list_available_items():
    """List all available items in the model"""
    if ITEM_ID_MAPPING is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Load item mapping CSV for descriptions
    try:
        item_df = pd.read_csv('preprocessed/item_mapping.csv')
        items = item_df.to_dict('records')
    except:
        items = [{"item_id": k, "item_id_encoded": v} for k, v in ITEM_ID_MAPPING.items()]
    
    return {
        "total_items": len(items),
        "items": items
    }

@app.post("/predict/save")
async def save_prediction(
    prediction: SavePredictionRequest,
    db: Session = Depends(get_db)
):
    """Save a prediction to the database"""
    try:
        # Create new prediction record
        db_prediction = Prediction(
            item_id=prediction.item_id,
            item_desc=prediction.item_desc,
            predicted_demand=prediction.predicted_demand,
            forecast_horizon=prediction.forecast_horizon,
            confidence_lower=prediction.confidence_lower,
            confidence_upper=prediction.confidence_upper,
            model_version=prediction.model_version,
            prediction_date=datetime.utcnow()
        )
        
        # Add and commit
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        return {
            "status": "success",
            "message": "Prediction saved successfully",
            "id": db_prediction.id
        }
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save prediction: {str(e)}")

# ============================================================================
# Run with: uvicorn api_prediction:app --reload --port 8000
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
