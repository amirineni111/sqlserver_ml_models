"""
Quick Data Check Script - Console Safe Version

This script quickly checks if new data is available in the database
and shows a summary of current data status with Windows console compatibility.
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace emoji with text equivalents for Windows console
        emoji_replacements = {
            '🚀': '[START]',
            '📅': '[DATE]',
            '📊': '[DATA]',
            '📈': '[STATS]',
            '🔍': '[INFO]',
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '⚠️': '[WARN]',
            '🎯': '[TARGET]',
            '🕒': '[TIME]',
            '💡': '[TIP]',
            '🔗': '[LINK]',
            '📋': '[LIST]',
            '🤖': '[MODEL]',
            '🔄': '[RETRAIN]',
            '🔧': '[TROUBLESHOOT]',
            '🟢': '[BUY]',
            '🔴': '[SELL]',
            '⚪': '[NEUTRAL]',
            '💻': '[COMMAND]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

def check_data_status():
    """Check current data status in the database"""
    safe_print("🔍 Checking data status...")
    print("=" * 60)
    
    try:
        db = SQLServerConnection()
        
        # Overall data summary
        summary_query = """
        SELECT 
            COUNT(*) as total_records,
            MIN(trading_date) as earliest_date,
            MAX(trading_date) as latest_date,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT trading_date) as trading_days
        FROM dbo.nasdaq_100_hist_data
        """
        
        summary = db.execute_query(summary_query).iloc[0]
        
        safe_print("📊 OVERALL DATA SUMMARY")
        print(f"  Total Records: {summary['total_records']:,}")
        print(f"  Date Range: {summary['earliest_date']} to {summary['latest_date']}")
        print(f"  Unique Tickers: {summary['unique_tickers']}")
        print(f"  Trading Days: {summary['trading_days']}")
        
        # Recent data check
        recent_query = """
        SELECT 
            COUNT(*) as records_7d,
            MAX(trading_date) as latest_date
        FROM dbo.nasdaq_100_hist_data 
        WHERE trading_date >= DATEADD(day, -7, GETDATE())
        """
        
        recent = db.execute_query(recent_query).iloc[0]
        
        safe_print(f"📅 RECENT DATA (Last 7 Days)")
        print(f"  Records: {recent['records_7d']:,}")
        print(f"  Latest Date: {recent['latest_date']}")
        
        # Check for RSI signals
        rsi_query = """
        SELECT 
            COUNT(*) as total_signals,
            COUNT(CASE WHEN rsi_trade_signal = 'Oversold (Buy)' THEN 1 END) as buy_signals,
            COUNT(CASE WHEN rsi_trade_signal = 'Overbought (Sell)' THEN 1 END) as sell_signals,
            MAX(trading_date) as latest_signal_date
        FROM dbo.nasdaq_100_rsi_signals
        WHERE trading_date >= DATEADD(day, -30, GETDATE())
        """
        
        rsi_data = db.execute_query(rsi_query).iloc[0]
        
        safe_print(f"🎯 RSI SIGNALS (Last 30 Days)")
        print(f"  Total Signals: {rsi_data['total_signals']:,}")
        print(f"  Buy Signals: {rsi_data['buy_signals']:,}")
        print(f"  Sell Signals: {rsi_data['sell_signals']:,}")
        print(f"  Latest Signal: {rsi_data['latest_signal_date']}")
        
        # Data freshness assessment
        latest_date = pd.to_datetime(summary['latest_date'])
        days_old = (datetime.now().date() - latest_date.date()).days
        
        safe_print(f"🕒 DATA FRESHNESS ASSESSMENT")
        if days_old == 0:
            safe_print(f"  ✅ Data is current (today)")
        elif days_old <= 3:
            safe_print(f"  ✅ Data is recent ({days_old} days old)")
        elif days_old <= 7:
            safe_print(f"  ⚠️ Data is somewhat old ({days_old} days old)")
        else:
            print(f"  Data is stale ({days_old} days old)")
        
        # Output data age for automation script
        print(f"Data is {days_old} days old")
        print(f"Latest date: {summary['latest_date']}")
        print(f"Total records: {summary['total_records']:,}")
        print("Database connection successful")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ Error checking data status: {e}")
        safe_print("🔧 TROUBLESHOOTING")
        print("  1. Check database connection")
        print("  2. Verify SQL Server is running")
        print("  3. Check .env configuration")
        return False

def main():
    """Main function"""
    safe_print("🚀 Data Status Check")
    safe_print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = check_data_status()
    
    if success:
        print("\n" + "=" * 60)
        safe_print("✅ Data status check complete!")
        safe_print("🔗 Next Actions:")
        print("  • Retrain model: python retrain_model.py")
        print("  • Get predictions: python predict_trading_signals.py --batch")
        print("  • Full analysis: jupyter lab --notebook-dir=notebooks")
    else:
        safe_print("❌ Data check failed. Please resolve issues before retraining.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
