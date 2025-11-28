import tensorflow as tf
import os
from core.config import settings
from models.schemas import ModelArtifacts
import joblib
import json

artifacts = ModelArtifacts()

async def load_model_artifacts():
    """Load the LSTM model on application startup."""
    global artifacts
    
    if os.path.exists(settings.MODEL_PATH):
        try:
            artifacts.model = tf.keras.models.load_model(settings.MODEL_PATH)
            print(f"✅ Model loaded successfully from {settings.MODEL_PATH}")

            #Load preprocessors
            # Example: Load scaler, label encoder, etc. if saved separately
            artifacts.scaler = joblib.load(settings.SCALER_PATH) if os.path.exists(settings.SCALER_PATH) else None
            artifacts.le_item = joblib.load(settings.LABEL_ENCODER_PATH) if os.path.exists(settings.LABEL_ENCODER_PATH) else None
            artifacts.df_original = joblib.load(settings.DF_ORIGINAL_PATH) if os.path.exists(settings.DF_ORIGINAL_PATH) else None

            #Load configuration
            with open(settings.feature_cols_path, 'r') as f:
                artifacts.feature_cols = json.load(f)

            with open(settings.metadata_path, 'r') as f:
                artifacts.metadata = json.load(f)
                artifacts.seq_len = artifacts.metadata.get["seq_length"]

            print("✅ All artifacts loaded successfully!")
            print(f"   - Model MAPE: {artifacts.metadata['mape']:.2f}%")
            print(f"   - Items supported: {artifacts.metadata['num_items']}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            artifacts.model = None
    else:
        artifacts.model = None
        print(f"⚠️  Model not found at {settings.MODEL_PATH}. Train and save your LSTM model first.")

def get_model():
    """Get the loaded model instance."""
    return artifacts.model