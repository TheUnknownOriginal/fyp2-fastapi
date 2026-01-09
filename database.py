import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Get database configuration from environment variables
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")

# Construct connection string
# Using pyodbc connection string format
DATABASE_URL = f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}?driver={DB_DRIVER}"

# Create SQLAlchemy engine
# echo=True will log all SQL statements (useful for debugging)
engine = create_engine(DATABASE_URL, echo=False)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class Prediction(Base):
    """
    SQLAlchemy model for storing predictions.
    Adjust tablename and columns as needed to match your actual database schema.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    item_id = Column(String(50), nullable=False, index=True)
    item_desc = Column(String(255), nullable=True)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    forecast_horizon = Column(String(20), nullable=False) # '1-day', '7-day', '30-day'
    predicted_demand = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Optional: raw input data or additional metadata
    notes = Column(Text, nullable=True)

def get_db():
    """
    Dependency to get a database session.
    Yields a session and closes it after request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """
    Create all tables defined in models.
    """
    Base.metadata.create_all(bind=engine)
