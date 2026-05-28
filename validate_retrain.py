"""
Post-retrain validation — check alpha feature selection and buy signal recovery.
Run this after weekly_retrain_ultra_fast.py completes.

Usage:
    python validate_retrain.py
"""

import json
import os
import sys
import pyodbc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
ALPHA_FEATURES = [
    'alpha_vs_nasdaq_5d', 'alpha_vs_nasdaq_20d',
    'alpha_vs_sector_5d', 'rs_rank_52wk',
    'fundamental_quality_score', 'pe_vs_sector_median',
]

conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={os.getenv('SQL_SERVER', r'192.168.86.28\MSSQLSERVER01')};"
    f"DATABASE={os.getenv('SQL_DATABASE', 'stockdata_db')};"
    f"UID={os.getenv('SQL_USERNAME', 'remote_user')};"
    f"PWD={os.getenv('SQL_PASSWORD', 'YourStrongPassword123!')};"
    f"TrustServerCertificate=yes;"
)

# ── 1. Verify selected_features.json ────────────────────────────────────────
print("=" * 60)
print("1. CHECKING SELECTED FEATURES")
print("=" * 60)

features_path = os.path.join(os.path.dirname(__file__), 'data', 'selected_features.json')
if not os.path.exists(features_path):
    print("  ERROR: selected_features.json not found — retrain may not have completed")
    sys.exit(1)

with open(features_path) as f:
    selected = json.load(f)

print(f"  Total features selected: {len(selected)}")
print(f"  Selected: {selected}")
print()

alpha_selected = [f for f in ALPHA_FEATURES if f in selected]
alpha_missing  = [f for f in ALPHA_FEATURES if f not in selected]

print(f"  Alpha features in selected set ({len(alpha_selected)}/{len(ALPHA_FEATURES)}):")
for f in ALPHA_FEATURES:
    mark = "✓" if f in selected else "✗"
    print(f"    {mark} {f}")

if not alpha_selected:
    print()
    print("  WARNING: No alpha features made it through selection.")
    print("  This means their mutual_info_classif scores were below")
    print("  the top-25 stock-specific features. The model may still")
    print("  behave differently due to updated training data.")
else:
    print(f"\n  Good: {len(alpha_selected)} alpha features selected!")

# ── 2. Check model file timestamps ──────────────────────────────────────────
print()
print("=" * 60)
print("2. MODEL FILE TIMESTAMPS")
print("=" * 60)

data_dir = os.path.join(os.path.dirname(__file__), 'data')
model_files = [
    'best_model_gradient_boosting.joblib',
    'scaler.joblib',
    'sector_encoder.joblib',
    'selected_features.json',
]
for fname in model_files:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        import datetime
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
        print(f"  {fname}: {mtime:%Y-%m-%d %H:%M:%S}")
    else:
        print(f"  MISSING: {fname}")

# ── 3. Run test prediction (no DB write) ────────────────────────────────────
print()
print("=" * 60)
print("3. TEST PREDICTION (sample 5 tickers, no DB write)")
print("=" * 60)

try:
    from predict_trading_signals import TradingSignalPredictor
    predictor = TradingSignalPredictor()
    predictor.load_model()

    # Predict for a small sample
    with pyodbc.connect(conn_str) as conn:
        tickers_df = pd.read_sql(
            "SELECT TOP 20 ticker FROM dbo.nasdaq_top100 ORDER BY ticker", conn
        )
    tickers = tickers_df['ticker'].tolist()

    results = predictor.predict_signals(tickers)

    if results is not None and not results.empty:
        print(f"  Rows returned (before filter): {len(results)}")
        print(f"  Buy signals: {(results['predicted_signal'] == 'Up').sum()}")
        print(f"  Max buy_prob: {results.get('up_probability', results.get('buy_probability', pd.Series([0]))).max():.4f}")
        print(f"  Mean buy_prob: {results.get('up_probability', results.get('buy_probability', pd.Series([0]))).mean():.4f}")
        print()
        # Show top-5 by buy probability
        prob_col = 'up_probability' if 'up_probability' in results.columns else 'buy_probability'
        if prob_col in results.columns:
            top5 = results.nlargest(5, prob_col)[['ticker', prob_col, 'predicted_signal']]
            print("  Top 5 by buy probability:")
            print(top5.to_string(index=False))
    else:
        print("  No results returned from predict_signals()")

except Exception as e:
    print(f"  ERROR in test prediction: {e}")
    import traceback
    traceback.print_exc()

# ── 4. Check recent ml_trading_predictions for anomalies ────────────────────
print()
print("=" * 60)
print("4. RECENT ml_trading_predictions AUDIT")
print("=" * 60)

try:
    with pyodbc.connect(conn_str) as conn:
        df_audit = pd.read_sql("""
            SELECT
                trading_date,
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN predicted_signal IN ('Buy','Up') THEN 1 ELSE 0 END) AS buy_count,
                SUM(CASE WHEN predicted_signal IN ('Sell','Down') THEN 1 ELSE 0 END) AS sell_count,
                ROUND(MAX(buy_probability), 4) AS max_buy_prob,
                ROUND(AVG(buy_probability), 4) AS avg_buy_prob,
                ROUND(AVG(confidence_percentage)/100.0, 4) AS avg_confidence
            FROM dbo.ml_trading_predictions
            WHERE trading_date >= DATEADD(day, -10, CAST(GETDATE() AS DATE))
            GROUP BY trading_date
            ORDER BY trading_date DESC
        """, conn)

    if df_audit.empty:
        print("  No recent predictions found")
    else:
        print(df_audit.to_string(index=False))
        last_day = df_audit.iloc[0]
        if last_day['buy_count'] == 0:
            print(f"\n  ALERT: Most recent day ({last_day['trading_date']}) still has 0 buys!")
            print("  Run predict_trading_signals.py to generate new predictions with retrained model.")
        else:
            print(f"\n  OK: Most recent day has {int(last_day['buy_count'])} buy signals")

except Exception as e:
    print(f"  DB error: {e}")

print()
print("=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
