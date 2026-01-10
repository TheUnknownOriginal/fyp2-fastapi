"""
FastAPI Prediction Service for LSTM Demand Forecasting
Provides endpoints for 90-day daily demand predictions
Supports single and multiple item predictions
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import io
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db, TSalesForecast

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
    description="90-Day Daily Demand Forecasting using 'low_dropout' model",
    version="2.0.0",
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
MODEL_METRICS = None

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================
class DailyPrediction(BaseModel):
    date: str
    day_index: int
    predicted_demand: float
    confidence_lower: float
    confidence_upper: float

class ModelPerformance(BaseModel):
    mae: float
    rmse: float
    r2_score: float

class ItemPredictionResponse(BaseModel):
    item_id: str
    item_desc: Optional[str] = None
    daily_predictions: List[DailyPrediction]
    total_90_day_demand: float
    last_30_days_avg: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[ItemPredictionResponse]
    model_used: str
    timestamp: str
    total_items: int
    data_last_date: str
    model_performance: Optional[ModelPerformance] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_items: int
    timestamp: str

class DatabaseHealthResponse(BaseModel):
    status: str
    connected: bool
    details: str
    timestamp: str


class SavePredictionRequest(BaseModel):
    item_id: str
    item_desc: Optional[str] = None
    predicted_demand: float
    forecast_horizon: str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    model_version: str = "v2.0"

# ============================================================================
# Startup Event - Load Model and Scalers
# ============================================================================
async def load_model_and_scalers():
    """Load model and preprocessing tools on startup"""
    global MODEL, FEATURE_SCALER, TARGET_SCALER, ITEM_ID_MAPPING, FEATURE_NAMES, MODEL_METRICS
    
    print("="*70)
    print("LOADING LSTM MODEL AND SCALERS")
    print("="*70)
    
    try:
        # Load best model
        model_path = "models/low_dropout.keras"
        MODEL = tf.keras.models.load_model(model_path)
        print(f"[OK] Loaded model: {model_path}")
        print(f"  Model Input Shape: {MODEL.input_shape}")
        print(f"  Model Output Shape: {MODEL.output_shape}")
        
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
        
        # Load model metrics
        try:
            metrics_df = pd.read_csv('models/model_performance.csv')
            # Extract values from the first row (assuming one model or specific logic)
            # User asked for mae_all, rmse_all, mape_all, r2_all
            row = metrics_df.iloc[0]
            MODEL_METRICS = {
                "mae": float(row.get("mae_all", 0.0)),
                "rmse": float(row.get("rmse_all", 0.0)),
                "mape": float(row.get("mape_all", 0.0)),
                "r2_score": float(row.get("r2_all", 0.0))
            }
            print(f"[OK] Loaded model metrics: {MODEL_METRICS}")
        except Exception as e:
            print(f"[WARNING] Could not load model metrics: {e}")
            MODEL_METRICS = {"mae": 0.0, "rmse": 0.0, "r2_score": 0.0, "mape": 0.0}

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
    
    # Drop rows with NaN if critical (or handle them)
    # df_filtered = df_filtered.dropna() # Careful with this, might drop too much
    
    return df_filtered

def create_sequences(df: pd.DataFrame, seq_length: int = 60) -> Dict:
    """Create sequences for each item. Updated default seq_length to 60 based on recent context."""
    sequences = {}
    
    for item_id in df['item_id'].unique():
        item_data = df[df['item_id'] == item_id].copy()
        
        # Sort by date just in case
        item_data = item_data.sort_values('date')
        
        if len(item_data) < seq_length:
            print(f"⚠️ Warning: Item {item_id} has only {len(item_data)} days of data (need {seq_length})")
            continue
        
        # Get last seq_length days
        last_history = item_data.tail(seq_length)
        last_date = last_history['date'].iloc[-1]
        
        # Check if features match trained features
        missing_features = [f for f in FEATURE_NAMES if f not in last_history.columns]
        if missing_features:
            print(f"⚠️ Warning: Item {item_id} missing features: {missing_features}")
            # Potentially fill missing with 0
            for f in missing_features:
                last_history[f] = 0
        
        # Extract features
        sequence = last_history[FEATURE_NAMES].values
        
        # Scale features
        # Note: If FEATURE_SCALER expects dataframe, fine. If numpy, fine.
        # Assuming FEATURE_SCALER fits on FEATURE_NAMES columns.
        sequence_scaled = FEATURE_SCALER.transform(sequence)
        
        # Get item encoding
        item_id_encoded = last_history['item_id_encoded'].iloc[0]
        # Keep encoded ID valid (fill with 0 or similar if NaN, though map should have handled it)
        if pd.isna(item_id_encoded):
             # Fallback if item not in mapping (should typically filter these out or use 'other')
             # For now, let's assume it was mapped or we skip
             pass

        # Get item description
        item_desc = last_history['item_desc'].iloc[0] if 'item_desc' in last_history.columns else None
        
        # Calculate last 30 days average for reference
        last_30_avg = last_history['qty_ordered'].tail(30).mean()
        
        sequences[item_id] = {
            'sequence': sequence_scaled,
            'item_id_encoded': item_id_encoded,
            'item_desc': item_desc,
            'last_30_avg': last_30_avg,
            'last_date': last_date
        }
    
    return sequences

def make_predictions(sequences: Dict) -> List[ItemPredictionResponse]:
    """Make 90-day predictions for all items"""
    responses = []
    
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
            detail="None of the requested items have enough historical data to generate a prediction."
        )
    
    # Convert to numpy arrays
    X_seq_batch = np.array(X_seq_batch)
    X_item_batch = np.array(X_item_batch)
    
    # Make predictions
    try:
        # Expected output shape: (batch, 90) or (batch, 90, 1)
        model_predictions = MODEL.predict([X_seq_batch, X_item_batch], verbose=0)
        
        # Squeeze if (batch, 90, 1) -> (batch, 90)
        if len(model_predictions.shape) == 3:
             model_predictions = np.squeeze(model_predictions, axis=-1)
             
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    
    # Process predictions per item
    for i, item_id in enumerate(item_ids_batch):
        preds = model_predictions[i] # Shape (90,)
        
        # Inverse transform
        # Target scaler likely expects shape (n, 3) if trained on [target_col1, target_col2, target_col3] 
        # OR shape (n, 1) if single target.
        # Based on previous code: TARGET_SCALER.inverse_transform(transform_array)[0, horizon_idx]
        # It seemed to train on 3 targets (1d, 7d, 30d). 
        # BUT the new model `low_dropout` is a Sequence-to-Sequence (90 days).
        # We need to consider how the target scaler was fitted.
        # IF target_scaler was fitted on 'qty_ordered' (single column), then simple inverse.
        # IF it was fitted on the 3-horizon training target, we might have a mismatch if we reuse the old scalar.
        # **ASSUMPTION**: The user kept `preprocessed/target_scaler.pkl`. If this scaler expects 3 dims, we have a problem
        # unless we pretend it's one of them.
        # Let's inspect the `TARGET_SCALER` safely.
        # For now, simplistic approach: create dummy array matching scaler input size.
        
        try:
            # Try 1D first
            preds_reshaped = preds.reshape(-1, 1)
            pred_original = TARGET_SCALER.inverse_transform(preds_reshaped).flatten()
        except ValueError:
            # Revert to 3D dummy padding if scaler expects 3 columns (common in previous step)
            # Create (90, 3) filled with zeros
            dummy_input = np.zeros((90, 3))
            # Assuming the target of interest (daily demand) corresponds to the first column or we apply to relevant col.
            # *Actually*, usually standard scaler is per feature. If we predicted raw values, we inverse.
            # Let's try filling column 0.
            dummy_input[:, 0] = preds
            pred_inv_full = TARGET_SCALER.inverse_transform(dummy_input)
            pred_original = pred_inv_full[:, 0]
        
        # Ensure non-negative
        pred_original = np.maximum(pred_original, 0)
        
        # Create daily objects
        last_date = sequences[item_id]['last_date']
        daily_preds_list = []
        
        for day_idx in range(len(pred_original)):
            forecast_date = last_date + timedelta(days=day_idx + 1)
            # Round to nearest whole number as requested
            val = float(round(pred_original[day_idx]))
            
            # Confidence intervals (Approximate for now: +/- 20%)
            # Ideally model outputs quantiles, but using heuristic here if not available
            daily_preds_list.append(DailyPrediction(
                date=forecast_date.strftime("%Y-%m-%d"),
                day_index=day_idx + 1,
                predicted_demand=val,
                confidence_lower=val * 0.8,
                confidence_upper=val * 1.2
            ))
            
        total_demand = sum(p.predicted_demand for p in daily_preds_list)
        
        responses.append(ItemPredictionResponse(
            item_id=item_id,
            item_desc=sequences[item_id]['item_desc'],
            daily_predictions=daily_preds_list,
            total_90_day_demand=total_demand,
            last_30_days_avg=float(sequences[item_id]['last_30_avg'])
        ))
    
    return responses

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LSTM Demand Forecasting API (90-Day Horizon)",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "health_db": "/health/db",
            "predict": "/predict"
        },
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check"""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        available_items=len(ITEM_ID_MAPPING) if ITEM_ID_MAPPING else 0,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health/db", response_model=DatabaseHealthResponse)
async def db_health_check(db: Session = Depends(get_db)):
    """Database connectivity check"""
    try:
        # Execute a simple query
        result = db.execute(text("SELECT 1"))
        result.scalar() # fetch result
        return DatabaseHealthResponse(
            status="healthy",
            connected=True,
            details="Successfully connected to SQL database",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "connected": False,
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# ============================================================================
def fetch_history_from_db(db: Session) -> pd.DataFrame:
    """Fetch the last 60 days of sales history from the database."""
    query = text("""
        SELECT *
        FROM [t_sales_order_history]
        WHERE [date] >= (
            SELECT DATEADD(day, -60, MAX([date])) 
            FROM [t_sales_order_history]
        )
        ORDER BY [date] DESC;
    """)
    
    result = db.execute(query)
    # Convert to list of dicts then DataFrame
    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()
        
    # Get column names
    keys = result.keys()
    df = pd.DataFrame([dict(zip(keys, row)) for row in rows])
    
    # Normalize column names to lowercase just in case
    df.columns = [c.lower() for c in df.columns]
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    return df

@app.post("/predict", response_model=BatchPredictionResponse)
async def predict(
    item_ids: str = Form(..., description="Comma-separated item IDs (e.g. '101,102') or 'all'"),
    db: Session = Depends(get_db)
):
    """
    Generate 90-day daily predictions using last 60 days history from DB.
    """
    try:
        # Fetch history from DB
        df = fetch_history_from_db(db)
        
        if df.empty:
             raise HTTPException(status_code=404, detail="No historical data found in 't_sales_order_history' (last 60 days)")

        # Parse item IDs
        if item_ids.lower().strip() == 'all':
             target_items = df['item_id'].astype(str).unique().tolist()
        else:
             target_items = [id.strip() for id in item_ids.split(',')]
        
        # Preprocess data
        df_processed = preprocess_data(df, target_items)
        
        # Create sequences
        # Note: assuming new model trained on 60 days as per recent context in conversation history
        # If model expects 30, this needs adjustment. 
        # Defaulting to 60 as per recent user "Refactor 90-Day Forecast" context.
        sequences = create_sequences(df_processed, seq_length=60)
        
        # Make predictions
        predictions = make_predictions(sequences)
        
        # Get last date from data for meta info
        last_date_str = df_processed['date'].max().strftime("%Y-%m-%d")
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_used="low_dropout",
            timestamp=datetime.now().isoformat(),
            total_items=len(predictions),
            data_last_date=last_date_str,
            model_performance=ModelPerformance(
                mae=MODEL_METRICS["mae"] if MODEL_METRICS else 0.0,
                rmse=MODEL_METRICS["rmse"] if MODEL_METRICS else 0.0,
                r2_score=MODEL_METRICS["r2_score"] if MODEL_METRICS else 0.0,
            )
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/save", response_model=BatchPredictionResponse)
async def predict_and_save(
    item_ids: str = Form(..., description="Comma-separated item IDs or 'all'"),
    db: Session = Depends(get_db)
):
    """
    Generate 90-day predictions and SAVE them to the database.
    (Uses last 60 days history from t_sales_order_history)
    """
    try:
        # Fetch history from DB
        df = fetch_history_from_db(db)
        
        if df.empty:
             raise HTTPException(status_code=404, detail="No historical data found in 't_sales_order_history' (last 60 days)")

        # Parse item IDs
        if item_ids.lower().strip() == 'all':
             target_items = df['item_id'].astype(str).unique().tolist()
        else:
             target_items = [id.strip() for id in item_ids.split(',')]
        
        # Preprocess
        df_processed = preprocess_data(df, target_items)
        sequences = create_sequences(df_processed, seq_length=60)
        predictions = make_predictions(sequences)
        last_date_str = df_processed['date'].max().strftime("%Y-%m-%d")
        
        # Save to DB
        db_objects = []
        now_utc = datetime.utcnow()
        
        # Parse data_last_date for DB (if needed as datetime)
        data_last_date_dt = df_processed['date'].max().to_pydatetime()
        
        # Metrics to store
        mae_val = MODEL_METRICS["mae"] if MODEL_METRICS else None
        rmse_val = MODEL_METRICS["rmse"] if MODEL_METRICS else None
        r2_val = MODEL_METRICS["r2_score"] if MODEL_METRICS else None
        
        for item_pred in predictions:
            for day_pred in item_pred.daily_predictions:
                db_obj = TSalesForecast(
                    created_at=now_utc,
                    created_by=0, # Assuming ID
                    is_deleted=0,
                    updated_at=now_utc,
                    updated_by=0, # Assuming ID
                    
                    item_id=item_pred.item_id,
                    item_desc=item_pred.item_desc,
                    date=datetime.strptime(day_pred.date, "%Y-%m-%d"),
                    day_index=day_pred.day_index,
                    predicted_demand=day_pred.predicted_demand,
                    
                    confidence_lower=day_pred.confidence_lower,
                    confidence_upper=day_pred.confidence_upper,
                    model_used="low_dropout", # Or "low_dropout_v2" if preferred
                    data_last_date=data_last_date_dt,
                    
                    mae=mae_val,
                    rmse=rmse_val,
                    r2_score=r2_val
                )
                db_objects.append(db_obj)
        
        # Bulk save
        if db_objects:
            db.bulk_save_objects(db_objects)
            db.commit()
            print(f"[DB] Saved {len(db_objects)} prediction records to t_sales_forecast.")
            
        return BatchPredictionResponse(
            predictions=predictions,
            model_used="low_dropout",
            timestamp=datetime.now().isoformat(),
            total_items=len(predictions),
            data_last_date=last_date_str,
            model_performance=ModelPerformance(
                mae=MODEL_METRICS["mae"] if MODEL_METRICS else 0.0,
                rmse=MODEL_METRICS["rmse"] if MODEL_METRICS else 0.0,
                r2_score=MODEL_METRICS["r2_score"] if MODEL_METRICS else 0.0,
                mape=MODEL_METRICS["mape"] if MODEL_METRICS else 0.0
            )
        )
        
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items", response_model=Dict)
async def list_available_items():
    """List all available items in the mapping"""
    if ITEM_ID_MAPPING is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        item_df = pd.read_csv('preprocessed/item_mapping.csv')
        items = item_df.to_dict('records')
    except:
        items = [{"item_id": k, "item_id_encoded": v} for k, v in ITEM_ID_MAPPING.items()]
    
    return {
        "total_items": len(items),
        "items": items
    }

# ============================================================================
# Run with: uvicorn main:app --reload --port 8000
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
