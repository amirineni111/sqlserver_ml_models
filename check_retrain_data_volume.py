#!/usr/bin/env python3
"""
Quick diagnostic script to check data volume for weekly retrain
This helps identify why retraining is taking too long
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

def check_data_volume():
    """Check how much data we're loading"""
    
    print("=" * 70)
    print("WEEKLY RETRAIN DATA VOLUME CHECK")
    print("=" * 70)
    
    db = SQLServerConnection()
    
    # Check 1: Total rows in source tables
    print("\n[CHECK 1] Source table sizes:")
    
    query1 = """
    SELECT 
        COUNT(*) as total_rows,
        MIN(trading_date) as earliest_date,
        MAX(trading_date) as latest_date,
        COUNT(DISTINCT ticker) as unique_tickers
    FROM dbo.nasdaq_100_hist_data
    """
    
    result1 = db.execute_query(query1)
    print(f"  nasdaq_100_hist_data:")
    print(f"    Total rows: {result1['total_rows'].iloc[0]:,}")
    print(f"    Date range: {result1['earliest_date'].iloc[0]} to {result1['latest_date'].iloc[0]}")
    print(f"    Tickers: {result1['unique_tickers'].iloc[0]}")
    
    # Check 2: What the NEW ultra-fast query would load (3 months)
    print("\n[CHECK 2] NEW ultra-fast query (3 months, TOP 100000):")
    
    query2 = """
    SELECT TOP 100000
        COUNT(*) as row_count,
        MIN(h.trading_date) as earliest_date,
        MAX(h.trading_date) as latest_date,
        COUNT(DISTINCT h.ticker) as ticker_count
    FROM dbo.nasdaq_100_hist_data h
    INNER JOIN dbo.nasdaq_100_rsi_signals r 
        ON h.ticker = r.ticker AND h.trading_date = r.trading_date
    WHERE h.trading_date >= DATEADD(MONTH, -3, GETDATE())
      AND r.rsi_trade_signal IS NOT NULL
    """
    
    result2 = db.execute_query(query2)
    print(f"    Rows: {result2['row_count'].iloc[0]:,}")
    print(f"    Date range: {result2['earliest_date'].iloc[0]} to {result2['latest_date'].iloc[0]}")
    print(f"    Tickers: {result2['ticker_count'].iloc[0]}")
    
    # Check 3: What the OLD query would load (from 2025-04-01)
    print("\n[CHECK 3] OLD query (from 2025-04-01, no limit):")
    
    query3 = """
    SELECT 
        COUNT(*) as row_count,
        MIN(h.trading_date) as earliest_date,
        MAX(h.trading_date) as latest_date,
        COUNT(DISTINCT h.ticker) as ticker_count
    FROM dbo.nasdaq_100_hist_data h
    INNER JOIN dbo.nasdaq_100_rsi_signals r 
        ON h.ticker = r.ticker AND h.trading_date = r.trading_date
    WHERE h.trading_date >= '2025-04-01'
      AND r.rsi_trade_signal IS NOT NULL
    """
    
    result3 = db.execute_query(query3)
    old_rows = result3['row_count'].iloc[0]
    new_rows = result2['row_count'].iloc[0]
    
    print(f"    Rows: {old_rows:,}")
    print(f"    Date range: {result3['earliest_date'].iloc[0]} to {result3['latest_date'].iloc[0]}")
    print(f"    Tickers: {result3['ticker_count'].iloc[0]}")
    
    # Calculate improvement
    if old_rows > 0:
        reduction_pct = ((old_rows - new_rows) / old_rows) * 100
        print(f"\n[IMPROVEMENT] Data reduction: {reduction_pct:.1f}% ({old_rows - new_rows:,} fewer rows)")
    
    # Estimate training time
    print(f"\n[TIME ESTIMATE]")
    print(f"  OLD query rows: {old_rows:,}")
    print(f"  NEW query rows: {new_rows:,}")
    print(f"  With 50 estimators + vectorized ops:")
    print(f"    OLD estimate: {old_rows / 100:.0f} seconds = {old_rows / 6000:.1f} minutes")
    print(f"    NEW estimate: {new_rows / 100:.0f} seconds = {new_rows / 6000:.1f} minutes")
    
    print("=" * 70)

if __name__ == "__main__":
    try:
        check_data_volume()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
