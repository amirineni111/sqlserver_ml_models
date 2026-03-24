"""Temporary script to check NSE + Forex view schemas."""
import sys
sys.path.insert(0, 'src')
from database.connection import SQLServerConnection
db = SQLServerConnection()

tables = ['nse_500_ema_sma_view', 'nse_500_ema_sma_data', 'nse_500_macd', 'nse_500_macd_data',
          'nse_500_sma_signals', 'nse_500_macd_signals']
for t in tables:
    print(f"\n=== {t} columns ===")
    try:
        df2 = db.execute_query(f'SELECT TOP 1 * FROM dbo.[{t}]')
        for col in df2.columns:
            print(f"  {col}")
    except Exception as e:
        print(f"  ERROR: {e}")
