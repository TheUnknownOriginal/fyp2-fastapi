"""
Database Testing Script
Run SQL queries to test database connectivity and operations
"""

from sqlalchemy import text
from database import SessionLocal, engine, Prediction, create_tables
from datetime import datetime

def test_connection():
    """Test basic database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✓ Database connection successful!")
            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def create_sample_prediction():
    """Insert a sample prediction record"""
    db = SessionLocal()
    try:
        sample = Prediction(
            item_id="TEST001",
            item_desc="Test Item",
            predicted_demand=150.5,
            forecast_horizon="7-day",
            confidence_lower=135.0,
            confidence_upper=165.0,
            model_version="v1.0",
            prediction_date=datetime.utcnow()
        )
        db.add(sample)
        db.commit()
        db.refresh(sample)
        print(f"✓ Sample prediction created with ID: {sample.id}")
        return sample.id
    except Exception as e:
        print(f"✗ Failed to create sample: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def query_all_predictions():
    """Query all predictions from database"""
    db = SessionLocal()
    try:
        predictions = db.query(Prediction).all()
        print(f"\n{'='*70}")
        print(f"Total predictions in database: {len(predictions)}")
        print(f"{'='*70}")
        
        for pred in predictions:
            print(f"\nID: {pred.id}")
            print(f"  Item: {pred.item_id} - {pred.item_desc}")
            print(f"  Demand: {pred.predicted_demand}")
            print(f"  Horizon: {pred.forecast_horizon}")
            print(f"  Confidence: [{pred.confidence_lower}, {pred.confidence_upper}]")
            print(f"  Date: {pred.prediction_date}")
            print(f"  Model: {pred.model_version}")
        
        return predictions
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return []
    finally:
        db.close()

def query_by_item_id(item_id: str):
    """Query predictions for a specific item"""
    db = SessionLocal()
    try:
        predictions = db.query(Prediction).filter(Prediction.item_id == item_id).all()
        print(f"\nFound {len(predictions)} prediction(s) for item {item_id}")
        for pred in predictions:
            print(f"  - {pred.forecast_horizon}: {pred.predicted_demand}")
        return predictions
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return []
    finally:
        db.close()

def execute_raw_sql(sql_query: str):
    """Execute a raw SQL query"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            
            # Try to fetch results if it's a SELECT query
            if sql_query.strip().upper().startswith("SELECT"):
                rows = result.fetchall()
                print(f"\nQuery returned {len(rows)} row(s):")
                for row in rows:
                    print(f"  {row}")
                return rows
            else:
                connection.commit()
                print(f"✓ Query executed successfully")
                return None
    except Exception as e:
        print(f"✗ Query failed: {e}")
        return None

def insert_sales_forecast():
    """Insert into t_sales_forecast"""
    print("\nInserting into t_sales_forecast...")
    val = input("Enter value for demand_1_day: ").strip()
    
    sql = """
INSERT INTO t_sales_forecast(
    id, created_at, created_by, is_deleted, updated_at, updated_by, demand_1_day
)
VALUES(
    dbo.GenerateHighConcurrencySnowflakeID(), GETDATE(), 1, 0, GETDATE(), 1, :val
)
"""
    try:
        with engine.connect() as connection:
            connection.execute(text(sql), {"val": val})
            connection.commit()
            print("✓ Insert successful")
    except Exception as e:
        print(f"✗ Insert failed: {e}")

def delete_all_predictions():
    """Delete all predictions (use with caution!)"""
    db = SessionLocal()
    try:
        count = db.query(Prediction).delete()
        db.commit()
        print(f"✓ Deleted {count} prediction(s)")
        return count
    except Exception as e:
        print(f"✗ Delete failed: {e}")
        db.rollback()
        return 0
    finally:
        db.close()

def main():
    """Main testing menu"""
    print("="*70)
    print("DATABASE TESTING SCRIPT")
    print("="*70)
    
    # Test connection first
    if not test_connection():
        print("\n⚠️  Cannot proceed without database connection.")
        print("Please check your .env file and database credentials.")
        return
    
    # Create tables if they don't exist
    try:
        create_tables()
        print("✓ Database tables verified/created")
    except Exception as e:
        print(f"✗ Table creation failed: {e}")
        return
    
    while True:
        print("\n" + "="*70)
        print("MENU:")
        print("  1. Query all predictions")
        print("  2. Query by item ID")
        print("  3. Create sample prediction")
        print("  4. Execute raw SQL")
        print("  5. Delete all predictions")
        print("  6. Insert into t_sales_forecast")
        print("  7. Exit")
        print("="*70)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            query_all_predictions()
        
        elif choice == "2":
            item_id = input("Enter item ID: ").strip()
            query_by_item_id(item_id)
        
        elif choice == "3":
            create_sample_prediction()
        
        elif choice == "4":
            print("\nExamples:")
            print("  SELECT * FROM predictions")
            print("  SELECT item_id, predicted_demand FROM predictions WHERE forecast_horizon = '7-day'")
            print("  SELECT COUNT(*) FROM predictions")
            sql = input("\nEnter SQL query: ").strip()
            if sql:
                execute_raw_sql(sql)
        
        elif choice == "5":
            confirm = input("Are you sure you want to delete ALL predictions? (yes/no): ").strip().lower()
            if confirm == "yes":
                delete_all_predictions()
            else:
                print("Cancelled.")
        
        elif choice == "6":
            insert_sales_forecast()
            
        elif choice == "7":
            print("\nExiting...")
            break        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
