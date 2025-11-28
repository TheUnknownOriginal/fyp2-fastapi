from typing import List
from pydantic import BaseModel


# class ForecastRequest(BaseModel):
#     item_id: str
#     history: list  # historical sales (e.g., [120, 130, 128, 135, ...])
#     steps_ahead: int = 7  # how many days to forecast (default = 7)

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