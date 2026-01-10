
import requests
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"
LOG_FILE = "verify_log.txt"

def log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def test_health():
    log(f"Testing Health Check ({BASE_URL}/health)...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            log("[OK] Health check passed")
            log(str(response.json()))
            return True
        else:
            log(f"[FAIL] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        log(f"[FAIL] Health check error: {e}")
        return False

def test_db_health():
    log(f"Testing DB Health Check ({BASE_URL}/health/db)...")
    try:
        response = requests.get(f"{BASE_URL}/health/db")
        if response.status_code == 200:
            log("[OK] DB Health check passed")
            log(str(response.json()))
            return True
        else:
            log(f"[FAIL] DB Health check failed: {response.status_code}")
            return False
    except Exception as e:
        log(f"[FAIL] DB Health check error: {e}")
        return False

def test_prediction():
    log(f"Testing Prediction ({BASE_URL}/predict)...")
    
    # generate dummy data for an item (need 60 days)
    try:
        items_resp = requests.get(f"{BASE_URL}/items")
        if items_resp.status_code == 200:
            items = items_resp.json()['items']
            if items:
                test_item_id = str(items[0]['item_id'])
                log(f"Using test item: {test_item_id}")
            else:
                test_item_id = "TEST_ITEM"
        else:
            test_item_id = "TEST_ITEM"
    except:
        test_item_id = "TEST_ITEM"

    # Create dummy CSV
    dates = [datetime.now() - timedelta(days=x) for x in range(70)]
    dates.reverse() # Oldest to newest
    
    df = pd.DataFrame({
        'date': dates,
        'item_id': [test_item_id] * 70,
        'qty_ordered': np.random.randint(0, 20, 70),
        'item_desc': ['Test Description'] * 70
    })
    
    # Save to CSV string
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_val = csv_buffer.getvalue()
    
    # files = {
    #     'file': ('test_data.csv', csv_val, 'text/csv')
    # }
    data = {
        'item_ids': test_item_id
    }
    
    try:
        # response = requests.post(f"{BASE_URL}/predict", files=files, data=data)
        response = requests.post(f"{BASE_URL}/predict", data=data) 
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions'][0]['daily_predictions']
            
            log(f"[OK] Prediction successful. Received {len(predictions)} daily predictions.")
            if len(predictions) == 90:
                log("[OK] Verified 90-day horizon.")
            else:
                log(f"[FAIL] Expected 90 predictions, got {len(predictions)}")
            
            # Check for rounding
            val = predictions[0]['predicted_demand']
            if float(val).is_integer():
                log(f"[OK] Prediction value is rounded (e.g. {val}).")
            else:
                 log(f"[FAIL] Prediction value is NOT rounded (e.g. {val}).")
            
            # Check model performance metrics
            if 'model_performance' in result:
                perf = result['model_performance']
                log(f"[OK] Model Performance found: MAE={perf.get('mae')}, RMSE={perf.get('rmse')}, R2={perf.get('r2_score')}")
            else:
                 log("[FAIL] 'model_performance' key missing in response")
                
            log(f"First prediction: {predictions[0]}")
            return True
        else:
            log(f"[FAIL] Prediction failed: {response.text}")
            return False
    except Exception as e:
        log(f"[FAIL] Prediction error: {e}")
        return False

def test_prediction_save():
    log(f"Testing Prediction & Save ({BASE_URL}/predict/save)...")
    
    # Use same setup as test_prediction. We can mostly duplicate logic or check DB.
    # Ideally we'd verify row count increase, but simple 200 OK check is a good start.
    
    # test_item_id = "TEST_SAVE_ITEM" # This caused embedding error
    
    # Get valid item
    try:
        items_resp = requests.get(f"{BASE_URL}/items")
        if items_resp.status_code == 200:
            items = items_resp.json()['items']
            if items:
                test_item_id = str(items[0]['item_id'])
                log(f"Using test item for save: {test_item_id}")
            else:
                 test_item_id = "100660" # Fallback to known item
        else:
            test_item_id = "100660"
    except:
        test_item_id = "100660"
    
     # Create dummy CSV
    dates = [datetime.now() - timedelta(days=x) for x in range(70)]
    dates.reverse()
    
    df = pd.DataFrame({
        'date': dates,
        'item_id': [test_item_id] * 70,
        'qty_ordered': np.random.randint(5, 25, 70),
        'item_desc': ['Save Test Description'] * 70
    })
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_val = csv_buffer.getvalue()
    
    files = {'file': ('test_save.csv', csv_val, 'text/csv')}
    data = {'item_ids': test_item_id}
    
    try:
        response = requests.post(f"{BASE_URL}/predict/save", files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            predictions = result['predictions'][0]['daily_predictions']
            log(f"[OK] Save request successful. Received {len(predictions)} predictions.")
            if len(predictions) == 90:
                log("[OK] Verified 90-day horizon.")
            return True
        else:
            log(f"[FAIL] Save request failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log(f"[FAIL] Save error: {e}")
        return False

if __name__ == "__main__":
    # Clear log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Starting Verification\n")
        
    log("Waiting for server to be ready...")
    # Retries
    for i in range(5):
        if test_health():
            break
        time.sleep(2)
        
    test_db_health()
    test_prediction()
    # test_prediction_save()
