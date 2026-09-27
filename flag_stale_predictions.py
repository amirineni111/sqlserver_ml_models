"""
One-time cleanup: flag historical stale / duplicate rows in ml_trading_predictions.

Before Sep 2026 two pipeline bugs wrote rows that downstream consumers (dashboard,
agentic AI top-5 report) still read as live signals:

  stale_data     A ticker whose price feed stopped was re-predicted every day with its
                 last trading_date (up to 49 times, stamped with each new
                 model_version). ~3,700 of these were scored: 1.3% Buy accuracy.
  duplicate_run  Re-runs appended a second copy of the same (ticker, trading_date);
                 the earliest row is kept, later copies are flagged.

Rows are FLAGGED, not deleted: is_actionable=0, high_confidence=0,
signal_strength='Suppressed', suppression_reason set — the same contract every
consumer already filters on. The predictor/export fixes stop new ones.

Usage:
    python flag_stale_predictions.py            # dry run: counts only
    python flag_stale_predictions.py --apply    # write the flags
"""

import argparse
import os
import sys

from sqlalchemy import text

sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection  # noqa: E402
from evaluate_predictions import STALE_LAG_DAYS  # noqa: E402

# One CTE only — SQL Server does not allow nesting WITH inside WITH.
RANKED_CTE = """
WITH ranked AS (
    SELECT prediction_id,
           DATEDIFF(day, trading_date, CAST(run_timestamp AS DATE)) AS lag_days,
           ROW_NUMBER() OVER (PARTITION BY ticker, trading_date
                              ORDER BY prediction_id) AS rn
    FROM dbo.ml_trading_predictions
)
"""
REASON = f"CASE WHEN r.lag_days > {STALE_LAG_DAYS} THEN 'stale_data' ELSE 'duplicate_run' END"
TARGET_WHERE = f"(r.lag_days > {STALE_LAG_DAYS} OR r.rn > 1)"

SUMMARY_SQL = RANKED_CTE + f"""
SELECT {REASON} AS reason, COUNT(*) AS n_rows,
       SUM(CASE WHEN ISNULL(p.is_actionable, 1) = 1 THEN 1 ELSE 0 END) AS currently_actionable,
       SUM(CASE WHEN p.high_confidence = 1 THEN 1 ELSE 0 END) AS currently_high_conf,
       MIN(p.trading_date) AS first_date, MAX(p.trading_date) AS last_date
FROM ranked r JOIN dbo.ml_trading_predictions p ON p.prediction_id = r.prediction_id
WHERE {TARGET_WHERE}
GROUP BY {REASON}
"""

APPLY_SQL = RANKED_CTE + f"""
UPDATE p SET
    is_actionable = 0,
    high_confidence = 0,
    signal_strength = 'Suppressed',
    suppression_reason = {REASON}
FROM dbo.ml_trading_predictions p
JOIN ranked r ON r.prediction_id = p.prediction_id
WHERE {TARGET_WHERE}
  AND ISNULL(p.suppression_reason, '') NOT IN ('stale_data', 'duplicate_run')
"""

# Keep the outcomes table consistent so suppressed-row stats exclude them too
APPLY_OUTCOMES_SQL = """
UPDATE o SET is_actionable = 0, suppression_reason = p.suppression_reason
FROM dbo.ml_prediction_outcomes o
JOIN dbo.ml_trading_predictions p ON p.prediction_id = o.prediction_id
WHERE p.suppression_reason IN ('stale_data', 'duplicate_run')
  AND ISNULL(o.suppression_reason, '') NOT IN ('stale_data', 'duplicate_run')
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--apply', action='store_true', help='write the flags (default: dry run)')
    args = ap.parse_args()

    db = SQLServerConnection()
    print(db.execute_query(SUMMARY_SQL).to_string(index=False))
    if not args.apply:
        print("\n[DRY RUN] No changes written. Re-run with --apply to flag these rows.")
        return
    with db.get_sqlalchemy_engine().begin() as conn:
        n_pred = conn.execute(text(APPLY_SQL)).rowcount
        n_out = conn.execute(text(APPLY_OUTCOMES_SQL)).rowcount
    print(f"\n[APPLIED] Flagged {n_pred:,} prediction rows and {n_out:,} outcome rows.")


if __name__ == '__main__':
    main()
