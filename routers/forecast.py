from fastapi import APIRouter, HTTPException
import numpy as np
from models.schemas import ForecastRequest
from core.model_loader import get_model

router = APIRouter(
    prefix="/api",
    tags=["forecast"]
)

@router.post("/predict")
async def predict(request: ForecastRequest):
    """
    Predict future demand based on historical sales data.
    
    - **item_id**: Unique identifier for the item
    - **history**: List of historical sales values
    - **steps_ahead**: Number of future time steps to predict (default: 7)
    """
    model = get_model()
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded on server. Please ensure the model file exists."
        )
    
    try:
        # Prepare input
        input_data = np.array(request.history).reshape((1, len(request.history), 1))
        
        # Predict sequentially for the requested steps
        preds = []
        current_input = input_data.copy()
        
        for _ in range(request.steps_ahead):
            next_val = model.predict(current_input, verbose=0)[0, -1, 0]
            preds.append(float(next_val))  # Convert to Python float for JSON serialization
            
            # Append and shift window
            current_input = np.append(current_input[:, 1:, :], [[[next_val]]], axis=1)

        return {
            "item_id": request.item_id,
            "forecast": preds,
            "steps_ahead": request.steps_ahead,
            "status": "success"
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Check if the model is loaded and API is healthy."""
    model = get_model()
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }