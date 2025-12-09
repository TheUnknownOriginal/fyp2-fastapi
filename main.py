from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta

# ============================================================================
# Load Model & Artifacts on Startup
# ============================================================================
class ModelArtifacts:
    """Container for all model artifacts"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.le_item = None
        self.df_original = None
        self.feature_cols = None
        self.metadata = None
        self.seq_len = 30

artifacts = ModelArtifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    print("🚀 Starting up application...")
    await load_model_artifacts()
    yield
    # Shutdown: Cleanup if needed
    print("🛑 Shutting down application...")

app = FastAPI(
    title="Warehouse Predictive Restocking API",
    description="LSTM-based demand forecasting for intelligent inventory management",
    version="1.0.0",
    lifespan=lifespan
)

async def load_model_artifacts():
    """Load all model artifacts when API starts"""
    try:
        print("🔄 Loading model artifacts...")
        
        # Load LSTM model
        artifacts.model = tf.keras.models.load_model("final_lstm_model.keras")
        print("✓ Model loaded")
        
        # Load preprocessors
        artifacts.scaler = joblib.load("scaler.pkl")
        artifacts.le_item = joblib.load("label_encoder.pkl")
        artifacts.df_original = joblib.load("df_original.pkl")
        print("✓ Preprocessors loaded")
        
        # Load configuration
        with open("feature_cols.json", "r") as f:
            artifacts.feature_cols = json.load(f)
        
        with open("model_metadata.json", "r") as f:
            artifacts.metadata = json.load(f)
            artifacts.seq_len = artifacts.metadata["seq_length"]
        
        print("✅ All artifacts loaded successfully!")
        print(f"   - Model MAPE: {artifacts.metadata['mape']:.2f}%")
        print(f"   - Items supported: {artifacts.metadata['num_items']}")
        
    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}")
        raise

# ============================================================================
# Request/Response Models
# ============================================================================
class PredictionRequest(BaseModel):
    item_id: str
    forecast_days: int = 7  # Default: predict next 7 days
    
class ReorderRecommendation(BaseModel):
    item_id: str
    current_stock: float
    predicted_demand_7days: float
    predicted_demand_14days: float
    predicted_demand_30days: float
    reorder_point: float
    days_until_stockout: int
    recommended_order_quantity: float
    safety_stock: float
    lead_time_days: int
    confidence_score: float  # Based on model MAPE
    
class BatchPredictionRequest(BaseModel):
    item_ids: List[str]
    forecast_days: int = 7

# ============================================================================
# Core Prediction Function
# ============================================================================
def predict_demand(item_id: str, days_ahead: int = 1) -> List[float]:
    """
    Predict demand for an item for next N days
    
    Args:
        item_id: Item identifier (e.g., 'ITEM0001')
        days_ahead: Number of days to forecast (default: 1)
    
    Returns:
        List of predicted quantities for each day
    """
    try:
        # Get item index
        item_idx = artifacts.le_item.transform([item_id])[0]
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Item '{item_id}' not found in training data")
    
    # Get last 30 days of data (scaled)
    df_scaled = artifacts.df_original.copy()
    df_scaled[artifacts.feature_cols] = artifacts.scaler.transform(
        df_scaled[artifacts.feature_cols]
    )
    
    item_data = df_scaled[df_scaled["item_id_encoded"] == item_idx].tail(artifacts.seq_len)
    
    if len(item_data) < artifacts.seq_len:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient data for '{item_id}'. Need {artifacts.seq_len} days, have {len(item_data)}"
        )
    
    # Predict iteratively for multiple days
    predictions = []
    current_sequence = item_data[artifacts.feature_cols].values.copy()
    
    for day in range(days_ahead):
        # Prepare input
        seq_input = np.expand_dims(current_sequence, axis=0)
        item_input = np.array([[item_idx]])
        
         # Predict next day
        pred_scaled = artifacts.model.predict(
            {"seq_input": seq_input, "item_input": item_input},
            verbose=0
        )[0][0]
        # Predict next day - try named-input dict first, fallback to single-array input
        try:
            pred_out = artifacts.model.predict(
                {"seq_input": seq_input, "item_input": item_input},
                verbose=0
            )
            pred_scaled = pred_out[0][0]
        except Exception:
            # Fallback: many models accept a single array input (seq only)
            pred_out = artifacts.model.predict(seq_input, verbose=0)
            # handle shapes like (1,1) or (1, n)
            pred_scaled = pred_out.reshape(-1)[0]
        
        # Inverse transform
        pred_qty = artifacts.scaler.inverse_transform(
            np.array([[pred_scaled] + [0] * (len(artifacts.feature_cols) - 1)])
        )[0][0]
        
        predictions.append(max(0, pred_qty))  # Ensure non-negative
        
        # Update sequence for next prediction (shift window)
        # For simplicity, we'll just update qty_sold and keep other features constant
        new_row = current_sequence[-1].copy()
        new_row[0] = pred_scaled  # Update qty_sold (first feature)
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    return predictions

# ============================================================================
# Reorder Point Calculation
# ============================================================================
def calculate_reorder_point(
    item_id: str,
    current_stock: float,
    lead_time_days: int,
    safety_factor: float = 1.5  # 1.5x safety stock for uncertainty
) -> ReorderRecommendation:
    """
    Calculate intelligent reorder point based on predicted demand
    
    Formula:
        Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
        Safety Stock = Z-score × σ × sqrt(Lead Time)
    """
    # Get predictions for different horizons
    pred_7days = predict_demand(item_id, days_ahead=7)
    pred_14days = predict_demand(item_id, days_ahead=14)
    pred_30days = predict_demand(item_id, days_ahead=30)
    
    total_7day = sum(pred_7days)
    total_14day = sum(pred_14days)
    total_30day = sum(pred_30days)
    
    avg_daily_demand = total_30day / 30
    
    # Calculate safety stock (1.5 standard deviations for ~93% service level)
    demand_std = np.std(pred_30days)
    safety_stock = safety_factor * demand_std * np.sqrt(lead_time_days)
    
    # Reorder point
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
    
    # Days until stockout
    if avg_daily_demand > 0:
        days_until_stockout = int(current_stock / avg_daily_demand)
    else:
        days_until_stockout = 999  # Very high number if no demand
    
    # Recommended order quantity (Economic Order Quantity simplified)
    # Order enough to cover lead time + review period (7 days)
    recommended_qty = avg_daily_demand * (lead_time_days + 7) + safety_stock - current_stock
    recommended_qty = max(0, recommended_qty)  # Don't order if overstocked
    
    # Confidence score (inverse of MAPE)
    confidence = max(0, 100 - artifacts.metadata['mape'])
    
    return ReorderRecommendation(
        item_id=item_id,
        current_stock=current_stock,
        predicted_demand_7days=round(total_7day, 2),
        predicted_demand_14days=round(total_14day, 2),
        predicted_demand_30days=round(total_30day, 2),
        reorder_point=round(reorder_point, 2),
        days_until_stockout=days_until_stockout,
        recommended_order_quantity=round(recommended_qty, 2),
        safety_stock=round(safety_stock, 2),
        lead_time_days=lead_time_days,
        confidence_score=round(confidence, 2)
    )

# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
async def root():
    """Health check endpoint"""
    if not artifacts.metadata or not artifacts.le_item:
        return {"status": "starting", "service": "Warehouse Predictive Restocking API", "model_loaded": False}
    return {
        "status": "online",
        "service": "Warehouse Predictive Restocking API",
        "model_accuracy": f"{100 - artifacts.metadata['mape']:.2f}%",
        "items_supported": artifacts.metadata['num_items']
    }

@app.post("/predict/demand", response_model=dict)
async def predict_item_demand(request: PredictionRequest):
    """
    Predict demand for a single item over the next N days
    
    Example request:
    ```json
    {
        "item_id": "ITEM0001",
        "forecast_days": 7
    }
    ```
    """
    predictions = predict_demand(request.item_id, request.forecast_days)
    
    return {
        "item_id": request.item_id,
        "forecast_days": request.forecast_days,
        "predictions": [round(p, 2) for p in predictions],
        "total_demand": round(sum(predictions), 2),
        "average_daily_demand": round(sum(predictions) / len(predictions), 2),
        "confidence": round(100 - artifacts.metadata['mape'], 2)
    }

@app.post("/reorder/recommendation", response_model=ReorderRecommendation)
async def get_reorder_recommendation(
    item_id: str,
    current_stock: float,
    lead_time_days: int = 7
):
    """
    Get intelligent reorder point recommendation
    
    Example: `/reorder/recommendation?item_id=ITEM0001&current_stock=50&lead_time_days=7`
    
    Returns reorder point, safety stock, and recommended order quantity
    """
    return calculate_reorder_point(item_id, current_stock, lead_time_days)

@app.post("/reorder/batch", response_model=List[ReorderRecommendation])
async def batch_reorder_recommendations(
    items: List[dict]  # [{"item_id": "ITEM0001", "current_stock": 50, "lead_time_days": 7}]
):
    """
    Get reorder recommendations for multiple items at once
    
    Example request:
    ```json
    [
        {"item_id": "ITEM0001", "current_stock": 50, "lead_time_days": 7},
        {"item_id": "ITEM0002", "current_stock": 30, "lead_time_days": 5}
    ]
    ```
    """
    recommendations = []
    for item in items:
        try:
            rec = calculate_reorder_point(
                item["item_id"],
                item["current_stock"],
                item.get("lead_time_days", 7)
            )
            recommendations.append(rec)
        except Exception as e:
            print(f"Error processing {item['item_id']}: {e}")
            continue
    
    return recommendations

@app.get("/items/critical", response_model=List[dict])
async def get_critical_items(
    threshold_days: int = 7
):
    """
    Get list of items that need immediate restocking (stockout within threshold days)
    
    Example: `/items/critical?threshold_days=7`
    """
    # This would ideally connect to your warehouse database
    # For now, return items from training data
    critical_items = []
    
    for item_id in artifacts.le_item.classes_[:10]:  # Sample first 10 items
        try:
            # Get current stock from your database (mocked here)
            current_stock = 50  # Replace with actual DB query
            lead_time = 7
            
            rec = calculate_reorder_point(item_id, current_stock, lead_time)
            
            if rec.days_until_stockout <= threshold_days:
                critical_items.append({
                    "item_id": item_id,
                    "days_until_stockout": rec.days_until_stockout,
                    "current_stock": current_stock,
                    "recommended_order_qty": rec.recommended_order_quantity,
                    "urgency": "HIGH" if rec.days_until_stockout <= 3 else "MEDIUM"
                })
        except:
            continue
    
    return sorted(critical_items, key=lambda x: x["days_until_stockout"])

@app.get("/model/info")
async def get_model_info():
    """Get information about the deployed model"""
    if not artifacts.metadata or not artifacts.le_item:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded yet")
    return {
        "model_metadata": artifacts.metadata,
        "supported_items": list(artifacts.le_item.classes_),
        "features_used": artifacts.feature_cols,
        "sequence_length": artifacts.seq_len
    }

# ============================================================================
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ============================================================================