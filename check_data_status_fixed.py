"""
Quick Data Check Script - Fixed Version

This script quickly checks if new data is available in the database
and shows a summary of current data status.
"""

import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

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
        
        safe_print(f"\n📅 RECENT DATA (Last 7 Days)")
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
        
        safe_print(f"\n🎯 RSI SIGNALS (Last 30 Days)")
        print(f"  Total Signals: {rsi_data['total_signals']:,}")
        print(f"  Buy Signals: {rsi_data['buy_signals']:,}")
        print(f"  Sell Signals: {rsi_data['sell_signals']:,}")
        print(f"  Latest Signal: {rsi_data['latest_signal_date']}")
        
        # Check last model training date
        try:
            import pickle
            with open('data/model_results.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            last_training = model_data.get('training_timestamp', 'Unknown')
            best_model = model_data.get('best_model_name', 'Unknown')
            
            if 'model_results' in model_data and best_model in model_data['model_results']:
                f1_score = model_data['model_results'][best_model]['f1_score']
                safe_print(f"\n🤖 CURRENT MODEL STATUS")
                print(f"  Last Training: {last_training}")
                print(f"  Best Model: {best_model}")                print(f"  F1-Score: {f1_score:.3f}")
            
        except FileNotFoundError:
            safe_print(f"\n🤖 CURRENT MODEL STATUS")
            safe_print(f"  ❌ No trained model found")
        
        # Data freshness assessment
        latest_date = pd.to_datetime(summary['latest_date'])
        days_old = (datetime.now().date() - latest_date.date()).days
        
        safe_print(f"\n🕒 DATA FRESHNESS ASSESSMENT")
        if days_old == 0:
            safe_print(f"  ✅ Data is current (today)")
            needs_retrain = False
        elif days_old <= 3:
            safe_print(f"  ✅ Data is recent ({days_old} days old)")
            needs_retrain = False
        elif days_old <= 7:
            safe_print(f"  ⚠️ Data is somewhat old ({days_old} days old)")
            needs_retrain = True
        else:
            safe_print(f"  🔴 Data is stale ({days_old} days old)")
            needs_retrain = True
        
        # Recommendations
        safe_print(f"\n💡 RECOMMENDATIONS")
        if needs_retrain:
            safe_print("  🔄 Retraining recommended due to data age")
            print("  💻 Run: python retrain_model.py --backup-old")
        else:
            safe_print("  ✅ Model is up-to-date with current data")
            print("  📊 Optional: python predict_trading_signals.py --batch")
        
        # Sample recent data
        sample_query = """
        SELECT TOP 5
            h.trading_date,
            h.ticker,
            h.close_price,
            r.RSI,
            r.rsi_trade_signal
        FROM dbo.nasdaq_100_hist_data h
        INNER JOIN dbo.nasdaq_100_rsi_signals r 
            ON h.ticker = r.ticker AND h.trading_date = r.trading_date
        ORDER BY h.trading_date DESC, h.ticker
        """
          try:
            sample_data = db.execute_query(sample_query)
            
            safe_print(f"\n📋 SAMPLE RECENT DATA")
            for _, row in sample_data.iterrows():
                signal_emoji = "🟢" if "Buy" in str(row['rsi_trade_signal']) else "🔴" if "Sell" in str(row['rsi_trade_signal']) else "⚪"
                rsi_val = row['RSI'] if pd.notna(row['RSI']) else 0.0
                close_val = row['close_price'] if pd.notna(row['close_price']) else 0.0
                safe_print(f"  {signal_emoji} {row['trading_date'].strftime('%Y-%m-%d')} {row['ticker']} ${close_val:.2f} RSI:{rsi_val:.1f}")
        except Exception as sample_error:
            safe_print(f"\n📋 SAMPLE RECENT DATA")
            safe_print(f"  ⚠️ Could not load sample data: {sample_error}")
        
        return True
        
    except Exception as e:
        safe_print(f"❌ Error checking data status: {e}")
        safe_print("\n🔧 TROUBLESHOOTING")
        print("  1. Check database connection")
        print("  2. Verify SQL Server is running")
        print("  3. Check .env configuration")
        return False

def safe_print(text):
    """Print text with safe encoding handling."""
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
            '📋': '[LIST]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)

def main():
    """Main function"""
    safe_print("🚀 Data Status Check")
    safe_print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = check_data_status()
      if success:
        print("\n" + "=" * 60)
        safe_print("✅ Data status check complete!")
        safe_print("\n🔗 Next Actions:")
        print("  • Retrain model: python retrain_model.py")
        print("  • Get predictions: python predict_trading_signals.py --batch")
        print("  • Full analysis: jupyter lab --notebook-dir=notebooks")
    else:
        safe_print("\n❌ Data check failed. Please resolve issues before retraining.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
