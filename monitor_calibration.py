"""
Calibration monitoring for the NASDAQ ML model.

Answers the question nothing in the pipeline asked before: *does a higher
confidence actually mean a higher win rate?* Writes one row per
(model_version, signal, confidence bucket) to dbo.ml_calibration_monitor so the
trend is queryable over time instead of being re-derived by hand.

Also emits a monotonicity check per signal side. As of Sep 2026 the Buy side is
cleanly monotonic (47.5% -> 57.3% across buckets over 150 days) while the Sell
side is flat (~51%), meaning Sell confidence carries no usable information — the
kind of divergence this table exists to surface early.

Usage:
    python monitor_calibration.py                # score last 150d, write a snapshot
    python monitor_calibration.py --window 90
    python monitor_calibration.py --dry-run      # print, don't write
    python monitor_calibration.py --actionable-only
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from sqlalchemy import text

sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection  # noqa: E402

from model_version import load_model_version  # noqa: E402

MONITOR_TABLE = 'ml_calibration_monitor'

# Bucket edges as (label, lower_inclusive, upper_exclusive). Chosen to straddle
# the gates in nasdaq_config: BUY_MIN 0.55, HIGH_CONFIDENCE 0.58.
BUCKETS = [
    ('<52',   0.00, 0.52),
    ('52-55', 0.52, 0.55),
    ('55-58', 0.55, 0.58),
    ('58-60', 0.58, 0.60),
    ('60+',   0.60, 1.01),
]

CREATE_TABLE_SQL = f"""
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{MONITOR_TABLE}')
BEGIN
    CREATE TABLE {MONITOR_TABLE} (
        calibration_id INT IDENTITY(1,1) PRIMARY KEY,
        snapshot_date DATE NOT NULL,
        model_version VARCHAR(50),
        window_days INT NOT NULL,
        actionable_only BIT NOT NULL,
        predicted_signal VARCHAR(50) NOT NULL,
        confidence_bucket VARCHAR(20) NOT NULL,
        bucket_low FLOAT,
        bucket_high FLOAT,
        n_predictions INT NOT NULL,
        n_correct INT NOT NULL,
        win_rate FLOAT,
        avg_confidence FLOAT,
        is_monotonic BIT,
        created_at DATETIME DEFAULT GETDATE(),
        INDEX IDX_calib_snapshot (snapshot_date),
        INDEX IDX_calib_model (model_version)
    )
END
"""

OUTCOMES_SQL = """
SELECT predicted_signal, confidence, correct, is_actionable, model_version
FROM dbo.ml_prediction_outcomes
WHERE correct IS NOT NULL
  AND confidence IS NOT NULL
  AND trading_date >= DATEADD(day, -:window_days, CAST(GETDATE() AS DATE))
"""

# Minimum rows behind a bucket before it counts toward the monotonicity verdict —
# thin tails are noise, not drift.
MIN_BUCKET_SAMPLES = 100


def _bucket_of(conf):
    for label, lo, hi in BUCKETS:
        if lo <= conf < hi:
            return label
    return BUCKETS[-1][0]


def compute_calibration(window_days=150, actionable_only=False, db=None):
    """Return a DataFrame of per-signal, per-bucket win rates over the window."""
    db = db or SQLServerConnection()
    df = db.execute_query(OUTCOMES_SQL, params={'window_days': window_days})
    if df.empty:
        return pd.DataFrame()

    if actionable_only:
        df = df[df['is_actionable'].fillna(True).astype(bool)]
        if df.empty:
            return pd.DataFrame()

    df['confidence_bucket'] = df['confidence'].astype(float).apply(_bucket_of)
    edges = {label: (lo, hi) for label, lo, hi in BUCKETS}

    grouped = df.groupby(['predicted_signal', 'confidence_bucket']).agg(
        n_predictions=('correct', 'size'),
        n_correct=('correct', 'sum'),
        avg_confidence=('confidence', 'mean'),
    ).reset_index()
    grouped['win_rate'] = grouped['n_correct'] / grouped['n_predictions']
    grouped['bucket_low'] = grouped['confidence_bucket'].map(lambda b: edges[b][0])
    grouped['bucket_high'] = grouped['confidence_bucket'].map(lambda b: edges[b][1])

    # Monotonic = win rate never falls as confidence rises, per signal side.
    order = {label: i for i, (label, _, _) in enumerate(BUCKETS)}
    grouped['_ord'] = grouped['confidence_bucket'].map(order)
    grouped = grouped.sort_values(['predicted_signal', '_ord'])
    grouped['is_monotonic'] = False
    for signal, side in grouped.groupby('predicted_signal'):
        solid = side[side['n_predictions'] >= MIN_BUCKET_SAMPLES]
        mono = bool(solid['win_rate'].is_monotonic_increasing) if len(solid) >= 2 else False
        grouped.loc[grouped['predicted_signal'] == signal, 'is_monotonic'] = mono

    return grouped.drop(columns='_ord')


def save_snapshot(grouped, window_days, actionable_only, db=None):
    """Append today's calibration snapshot to ml_calibration_monitor."""
    db = db or SQLServerConnection()
    engine = db.get_sqlalchemy_engine()
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))

    out = grouped.copy()
    out['snapshot_date'] = datetime.now().date()
    out['window_days'] = window_days
    out['actionable_only'] = bool(actionable_only)
    # A snapshot mixes models when the window spans a retrain; record the version
    # in force when it ran so snapshots can be compared across retrains.
    _, version = load_model_version()
    out['model_version'] = version

    cols = ['snapshot_date', 'model_version', 'window_days', 'actionable_only',
            'predicted_signal', 'confidence_bucket', 'bucket_low', 'bucket_high',
            'n_predictions', 'n_correct', 'win_rate', 'avg_confidence', 'is_monotonic']
    out[cols].to_sql(MONITOR_TABLE, engine, if_exists='append', index=False)
    return len(out)


def print_report(grouped):
    if grouped.empty:
        print("[CALIB] No scored outcomes in window")
        return
    for signal, side in grouped.groupby('predicted_signal'):
        mono = bool(side['is_monotonic'].iloc[0])
        flag = "monotonic" if mono else "NOT monotonic - confidence is not informative"
        print(f"\n  {signal} ({flag})")
        for _, r in side.iterrows():
            bar = '#' * int(round(r['win_rate'] * 40))
            print(f"    {r['confidence_bucket']:>7}  n={int(r['n_predictions']):>7,}  "
                  f"win={r['win_rate']:6.1%}  {bar}")


def main():
    p = argparse.ArgumentParser(description='NASDAQ ML calibration monitor')
    p.add_argument('--window', type=int, default=150, help='Lookback days (default: 150)')
    p.add_argument('--actionable-only', action='store_true',
                   help='Restrict to is_actionable=1 rows')
    p.add_argument('--dry-run', action='store_true', help='Print without writing')
    args = p.parse_args()

    db = SQLServerConnection()
    grouped = compute_calibration(args.window, args.actionable_only, db)
    if grouped.empty:
        print("[CALIB] No data to report")
        return 0

    print(f"[CALIB] Calibration over last {args.window} days"
          f"{' (actionable only)' if args.actionable_only else ''}")
    print_report(grouped)

    if args.dry_run:
        print("\n[CALIB] --dry-run: nothing written")
        return 0

    n = save_snapshot(grouped, args.window, args.actionable_only, db)
    print(f"\n[CALIB] Wrote {n} rows to {MONITOR_TABLE}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
