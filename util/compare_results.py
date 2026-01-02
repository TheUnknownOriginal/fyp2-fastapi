import pandas as pd
import requests
import argparse
import sys
import os
from datetime import datetime

def compare_results(item_id=None, horizon="7d"):
    # API Endpoint
    api_url = "http://127.0.0.1:8000/predict/all-horizons"
    
    # Files
    train_file = "data/store_orders_train.csv"
    actual_may_file = "store_orders_may_2025.csv"

    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        return
    if not os.path.exists(actual_may_file):
        print(f"Error: May 2025 file not found at {actual_may_file}")
        return

    print(f"Loading actual May 2025 data from {actual_may_file}...")
    try:
        df_actual = pd.read_csv(actual_may_file)
        # Ensure date parsing
        df_actual['date'] = pd.to_datetime(df_actual['date'])
    except Exception as e:
        print(f"Error reading {actual_may_file}: {e}")
        return
        
    # Filter actuals based on horizon
    if horizon == "7d":
        start_date = pd.Timestamp("2025-05-01")
        end_date = pd.Timestamp("2025-05-07")
        print(f"Filtering actual data for first 7 days ({start_date.date()} to {end_date.date()})...")
        df_actual = df_actual[(df_actual['date'] >= start_date) & (df_actual['date'] <= end_date)]
        pred_key = "7_day"
        header_pred_col = "Pred (7d)"
    else: # 30d
        # Assuming May file is appropriate for 30d comparison (ignoring 31st day or handled loosely)
        print("Using full available actual data for 30-day comparison...")
        pred_key = "30_day"
        header_pred_col = "Pred (30d)"

    # Determine items to predict
    if item_id:
        # Check if item_id exists in filtered actual data
        if str(item_id) not in df_actual['item_id'].astype(str).values:
            print(f"Warning: Item {item_id} not found in actual data (date filtered).")
        target_items = [str(item_id)]
    else:
        # Use all items present in the now-filtered May 2025 data
        target_items = df_actual['item_id'].astype(str).unique().tolist()
        print(f"Found {len(target_items)} unique items in actual data for selected horizon.")

    if not target_items:
        print("No items found to compare.")
        return

    # Prepare API payload
    item_ids_str = ",".join(target_items)
    
    print(f"Sending request to {api_url} for {len(target_items)} items...")
    
    try:
        with open(train_file, 'rb') as f:
            files = {'file': ('store_orders_train.csv', f, 'text/csv')}
            data = {'item_ids': item_ids_str}
            response = requests.post(api_url, files=files, data=data)
            
        if response.status_code != 200:
            print(f"API Error ({response.status_code}): {response.text}")
            return
            
        api_response = response.json()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Is the server running? (uv run uvicorn main:app --reload)")
        return
    except Exception as e:
        print(f"Error calling API: {e}")
        return

    predictions = api_response.get('predictions', [])
    
    # Prepare comparison table
    print("\n" + "="*100)
    print(f"{'Item ID':<15} | {'Desc (Trunc)':<20} | {'Actual (' + horizon + ')':<15} | {header_pred_col:<15} | {'Diff':<10} | {'Error %':<10}")
    print("="*100)
    
    total_abs_error = 0
    total_actual = 0
    count = 0

    for pred in predictions:
        p_item_id = str(pred['item_id'])
        item_desc = pred.get('item_desc', 'N/A')
        # Truncate desc for display
        if item_desc and len(item_desc) > 18:
            item_desc = item_desc[:17] + "."
        
        # Get predicted demand
        try:
            pred_demand = pred['predictions'][pred_key]['demand']
        except KeyError:
            print(f"Warning: No {pred_key} prediction for item {p_item_id}")
            continue
            
        # Calculate actual sum
        item_actual_rows = df_actual[df_actual['item_id'].astype(str) == p_item_id]
        if item_actual_rows.empty:
            actual_demand = 0
        else:
            actual_demand = item_actual_rows['qty_ordered'].sum()
            
        diff = pred_demand - actual_demand
        
        if actual_demand != 0:
            error_pct = (abs(diff) / actual_demand) * 100
        else:
            error_pct = 0.0 if diff == 0 else float('inf')
            
        print(f"{p_item_id:<15} | {str(item_desc):<20} | {actual_demand:<15.2f} | {pred_demand:<15.2f} | {diff:<10.2f} | {error_pct:<10.2f}%")
        
        # Aggregate metrics
        if actual_demand > 0:
            total_abs_error += abs(diff)
            total_actual += actual_demand
            count += 1
            
    print("="*100)
    
    # Calculate overall MAPE for this batch
    if total_actual > 0:
        wmape = (total_abs_error / total_actual) * 100
        print(f"\nOverall WMAPE (Weighted Mean Absolute Percentage Error) for this set: {wmape:.2f}%")
    else:
        print("\nNo valid actual data to calculate aggregate error.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predictions against actual May 2025 data.")
    parser.add_argument("--item_id", type=str, help="Specific item ID to compare (optional).")
    parser.add_argument("--horizon", type=str, choices=["7d", "30d"], default="7d", help="Forecast horizon to compare (default: 30d)")
    
    args = parser.parse_args()
    
    compare_results(args.item_id, args.horizon)
