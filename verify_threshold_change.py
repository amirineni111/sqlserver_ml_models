"""
Verify NASDAQ Confidence Threshold Change Impact
Queries ml_trading_predictions to check impact of lowering threshold from 70% to 60%
"""
import pyodbc
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from nasdaq_config import HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD

# Load environment variables
load_dotenv()

def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_replacements = {
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '📊': '[DATA]',
            '🎯': '[TARGET]',
            '⚠️': '[WARN]',
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

def get_connection():
    """Get database connection using .env configuration"""
    server = os.getenv('SQL_SERVER', 'localhost')
    database = os.getenv('SQL_DATABASE', 'stockdata_db')
    username = os.getenv('SQL_USERNAME')
    password = os.getenv('SQL_PASSWORD')
    driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password}"
    )
    return pyodbc.connect(conn_str)

def analyze_threshold_impact():
    """Analyze threshold impact on historical predictions"""
    safe_print("="*80)
    safe_print(f"NASDAQ ML Confidence Threshold Analysis - {datetime.now()}")
    safe_print("="*80)
    safe_print(f"New High Confidence Threshold: {HIGH_CONFIDENCE_THRESHOLD:.0%}")
    safe_print(f"New Medium Confidence Threshold: {MEDIUM_CONFIDENCE_THRESHOLD:.0%}")
    safe_print("="*80)
    
    conn = get_connection()
    
    # Query recent predictions (last 5 trading days)
    query = """
    SELECT TOP 10000
        trading_date,
        ticker,
        predicted_signal,
        confidence_percentage / 100.0 AS confidence,
        CASE 
            WHEN confidence_percentage / 100.0 >= 0.70 THEN 1 
            ELSE 0 
        END AS old_high_confidence,
        CASE 
            WHEN confidence_percentage / 100.0 >= ? THEN 1 
            ELSE 0 
        END AS new_high_confidence
    FROM ml_trading_predictions
    WHERE trading_date >= DATEADD(day, -10, GETDATE())
    ORDER BY trading_date DESC
    """
    
    df = pd.read_sql(query, conn, params=[HIGH_CONFIDENCE_THRESHOLD])
    conn.close()
    
    if df.empty:
        safe_print("⚠️  No predictions found in database")
        return
    
    # Summary statistics
    total_predictions = len(df)
    old_high_conf = df['old_high_confidence'].sum()
    new_high_conf = df['new_high_confidence'].sum()
    newly_unlocked = new_high_conf - old_high_conf
    
    avg_confidence = df['confidence'].mean()
    median_confidence = df['confidence'].median()
    
    safe_print("\n📊 THRESHOLD IMPACT ANALYSIS")
    safe_print("-"*80)
    safe_print(f"Total Predictions (last 10 days): {total_predictions}")
    safe_print(f"Average Confidence: {avg_confidence:.1%}")
    safe_print(f"Median Confidence: {median_confidence:.1%}")
    safe_print("")
    safe_print(f"OLD High-Confidence Count (≥70%): {old_high_conf} ({old_high_conf/total_predictions:.1%})")
    safe_print(f"NEW High-Confidence Count (≥{HIGH_CONFIDENCE_THRESHOLD:.0%}): {new_high_conf} ({new_high_conf/total_predictions:.1%})")
    safe_print(f"✅ Newly Unlocked Signals: {newly_unlocked} ({newly_unlocked/total_predictions:.1%})")
    safe_print("")
    
    # Daily breakdown
    daily_summary = df.groupby('trading_date').agg({
        'ticker': 'count',
        'old_high_confidence': 'sum',
        'new_high_confidence': 'sum'
    }).rename(columns={'ticker': 'total'})
    daily_summary['unlocked'] = daily_summary['new_high_confidence'] - daily_summary['old_high_confidence']
    
    safe_print("\n📅 DAILY BREAKDOWN")
    safe_print("-"*80)
    safe_print(f"{'Date':<12} {'Total':>6} {'Old HC':>7} {'New HC':>7} {'Unlocked':>9}")
    safe_print("-"*80)
    
    for date, row in daily_summary.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        safe_print(f"{date_str:<12} {int(row['total']):>6} {int(row['old_high_confidence']):>7} {int(row['new_high_confidence']):>7} {int(row['unlocked']):>9}")
    
    safe_print("\n" + "="*80)
    safe_print("🎯 RECOMMENDATION")
    safe_print("="*80)
    
    if newly_unlocked > 0:
        pct_increase = (newly_unlocked / old_high_conf * 100) if old_high_conf > 0 else float('inf')
        safe_print(f"✅ Threshold change unlocks {newly_unlocked} additional high-confidence signals")
        safe_print(f"✅ This is a {pct_increase:.0f}% increase in actionable predictions")
        safe_print(f"✅ Median confidence ({median_confidence:.1%}) validates threshold appropriateness")
    else:
        safe_print("⚠️  No additional signals unlocked - threshold may need further tuning")
    
    safe_print("="*80)

if __name__ == "__main__":
    try:
        analyze_threshold_impact()
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
