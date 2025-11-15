import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    WMS_API_KEY: str = os.getenv("WMS_API_KEY", "test123")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/lstm_model.h5")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

settings = Settings()