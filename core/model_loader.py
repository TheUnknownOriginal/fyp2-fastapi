import tensorflow as tf
import os
from core.config import settings

model = None

def load_model():
    """Load the LSTM model on application startup."""
    global model
    
    if os.path.exists(settings.MODEL_PATH):
        try:
            model = tf.keras.models.load_model(settings.MODEL_PATH)
            print(f"✅ Model loaded successfully from {settings.MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            model = None
    else:
        model = None
        print(f"⚠️  Model not found at {settings.MODEL_PATH}. Train and save your LSTM model first.")

def get_model():
    """Get the loaded model instance."""
    return model