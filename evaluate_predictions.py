"""
Live Prediction Accuracy Tracker

Compares predictions in ml_trading_predictions against realized 5-day forward
returns from nasdaq_100_hist_data and stores the outcomes in
ml_prediction_outcomes. This is the feedback loop the pipeline was missing:
every model/threshold change is ultimately judged on this table, not on
offline test-set metrics.

A prediction is evaluable once the history table has 5 more trading rows for
that ticker after the prediction date (LEAD over actual rows = trading days,
so weekends/holidays are handled without a calendar table).

Correctness rule (mirrors the 5-day direction training target):
    bullish signal (Buy/Up)   correct if 5-day forward return > 0
    bearish signal (Sell/Down) correct if 5-day forward return <= 0

Skill rule (Sep 2026): raw direction accuracy swings 20-80% with the tape, so each
outcome also records the universe's average 5-day return for the same trading_date.
    beat_universe = Buy outperformed the universe / Sell underperformed it
This is the metric that separates model skill from market beta.

Usage:
    python evaluate_predictions.py                 # evaluate new + print summary
    python evaluate_predictions.py --days-back 365 # backfill window for price LEAD scan
    python evaluate_predictions.py --summary-only  # no writes, just rolling accuracy
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection

from model_version import load_model_version  # noqa: E402
from nasdaq_config import MIN_STOCK_PRICE  # noqa: E402

OUTCOMES_TABLE = 'ml_prediction_outcomes'

CREATE_TABLE_SQL = f"""
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{OUTCOMES_TABLE}')
BEGIN
    CREATE TABLE {OUTCOMES_TABLE} (
        outcome_id INT IDENTITY(1,1) PRIMARY KEY,
        prediction_id INT NOT NULL,
        ticker VARCHAR(10) NOT NULL,
        trading_date DATE NOT NULL,
        predicted_signal VARCHAR(50) NOT NULL,
        confidence FLOAT,
        entry_close FLOAT,
        realized_close_5d FLOAT,
        realized_return_5d FLOAT,
        realized_date DATE,
        correct BIT,
        model_version VARCHAR(50),
        is_actionable BIT,
        suppression_reason VARCHAR(100),
        universe_return_5d FLOAT,
        universe_pct_up_5d FLOAT,
        excess_return_5d FLOAT,
        beat_universe BIT,
        evaluated_at DATETIME DEFAULT GETDATE(),
        INDEX IDX_outcome_pred (prediction_id),
        INDEX IDX_outcome_ticker_date (ticker, trading_date)
    )
END
"""

# Suppressed signals are now stored (is_actionable=0) instead of dropped, so
# outcomes cover the full ticker universe — needed to re-derive the reliability
# thresholds. Headline accuracy stays actionable-only (see get_rolling_accuracy).
MIGRATE_COLUMNS_SQL = f"""
IF COL_LENGTH('{OUTCOMES_TABLE}', 'is_actionable') IS NULL
    ALTER TABLE {OUTCOMES_TABLE} ADD is_actionable BIT;
IF COL_LENGTH('{OUTCOMES_TABLE}', 'suppression_reason') IS NULL
    ALTER TABLE {OUTCOMES_TABLE} ADD suppression_reason VARCHAR(100);
IF COL_LENGTH('ml_trading_predictions', 'is_actionable') IS NULL
    ALTER TABLE ml_trading_predictions ADD is_actionable BIT;
IF COL_LENGTH('ml_trading_predictions', 'suppression_reason') IS NULL
    ALTER TABLE ml_trading_predictions ADD suppression_reason VARCHAR(100);
IF COL_LENGTH('ml_trading_predictions', 'model_version') IS NULL
    ALTER TABLE ml_trading_predictions ADD model_version VARCHAR(50);
IF COL_LENGTH('{OUTCOMES_TABLE}', 'universe_return_5d') IS NULL
    ALTER TABLE {OUTCOMES_TABLE} ADD universe_return_5d FLOAT;
IF COL_LENGTH('{OUTCOMES_TABLE}', 'universe_pct_up_5d') IS NULL
    ALTER TABLE {OUTCOMES_TABLE} ADD universe_pct_up_5d FLOAT;
IF COL_LENGTH('{OUTCOMES_TABLE}', 'excess_return_5d') IS NULL
    ALTER TABLE {OUTCOMES_TABLE} ADD excess_return_5d FLOAT;
IF COL_LENGTH('{OUTCOMES_TABLE}', 'beat_universe') IS NULL
    ALTER TABLE {OUTCOMES_TABLE} ADD beat_universe BIT;
"""

# A prediction is only valid for the session it was made on. Before Sep 2026 a
# ticker whose feed stopped was re-predicted daily with its last (stale)
# trading_date — ~3,700 rows scoring 1.3% Buy accuracy / -18% return. The
# predictor now drops them; this lag cap keeps the historical ones out of every
# accuracy read. 4 calendar days covers Fri->Mon plus a holiday.
STALE_LAG_DAYS = 4

# Append to any query over ml_prediction_outcomes aliased `o`: excludes stale
# re-inserts and keeps only the first scored row per (ticker, trading_date), so
# historical duplicate runs aren't double-counted.
CLEAN_OUTCOMES_FILTER = f"""
  AND EXISTS (SELECT 1 FROM dbo.ml_trading_predictions sp
              WHERE sp.prediction_id = o.prediction_id
                AND DATEDIFF(day, sp.trading_date, CAST(sp.run_timestamp AS DATE)) <= {STALE_LAG_DAYS})
  AND o.outcome_id = (SELECT MIN(o2.outcome_id) FROM dbo.ml_prediction_outcomes o2
                      WHERE o2.ticker = o.ticker AND o2.trading_date = o.trading_date)
"""

# LEAD(close, 5) over actual history rows gives the close exactly 5 TRADING
# days later. Prices are VARCHAR in nasdaq_100_hist_data — always CAST.
# realized_date is capped at 12 calendar days: a ticker with a feed gap would
# otherwise LEAD across the gap and book a months-long return as "5 days".
# `uni` is the equal-weight universe (same $ floor as the predictor) per date.
PRICE_CTES = f"""
WITH px AS (
    SELECT ticker, trading_date,
           CAST(close_price AS FLOAT) AS entry_close,
           LEAD(CAST(close_price AS FLOAT), 5) OVER (
               PARTITION BY ticker ORDER BY trading_date) AS realized_close_5d,
           LEAD(trading_date, 5) OVER (
               PARTITION BY ticker ORDER BY trading_date) AS realized_date
    FROM dbo.nasdaq_100_hist_data
    WHERE ISNUMERIC(close_price) = 1
      AND trading_date >= DATEADD(day, -:days_back, CAST(GETDATE() AS DATE))
),
pxv AS (
    SELECT * FROM px
    WHERE realized_close_5d IS NOT NULL AND entry_close > 0
      AND DATEDIFF(day, trading_date, realized_date) <= 12
),
uni AS (
    SELECT trading_date,
           AVG((realized_close_5d - entry_close) / entry_close * 100) AS universe_return_5d,
           AVG(CASE WHEN realized_close_5d > entry_close THEN 1.0 ELSE 0.0 END) AS universe_pct_up_5d
    FROM pxv
    WHERE entry_close >= {MIN_STOCK_PRICE}
    GROUP BY trading_date
)
"""

PENDING_PREDICTIONS_SQL = PRICE_CTES + f"""
SELECT p.prediction_id, p.run_timestamp, p.ticker, p.trading_date,
       p.predicted_signal, p.confidence,
       p.is_actionable, p.suppression_reason, p.model_version,
       px.entry_close, px.realized_close_5d, px.realized_date,
       uni.universe_return_5d, uni.universe_pct_up_5d
FROM dbo.ml_trading_predictions p
INNER JOIN pxv px
    ON px.ticker = p.ticker AND px.trading_date = p.trading_date
LEFT JOIN uni ON uni.trading_date = p.trading_date
LEFT JOIN dbo.ml_prediction_outcomes o
    ON o.prediction_id = p.prediction_id
WHERE o.prediction_id IS NULL
  AND DATEDIFF(day, p.trading_date, CAST(p.run_timestamp AS DATE)) <= {STALE_LAG_DAYS}
  AND NOT EXISTS (SELECT 1 FROM dbo.ml_prediction_outcomes o2
                  WHERE o2.ticker = p.ticker AND o2.trading_date = p.trading_date)
"""

# Fills the skill columns on outcomes scored before they existed. Only touches
# rows where universe_return_5d IS NULL, so it is a no-op once caught up.
BACKFILL_UNIVERSE_SQL = PRICE_CTES + """
UPDATE o SET
    universe_return_5d = uni.universe_return_5d,
    universe_pct_up_5d = uni.universe_pct_up_5d,
    excess_return_5d = o.realized_return_5d - uni.universe_return_5d,
    beat_universe = CASE
        WHEN (o.predicted_signal LIKE '%Buy%' OR o.predicted_signal LIKE '%Up%')
            THEN CASE WHEN o.realized_return_5d > uni.universe_return_5d THEN 1 ELSE 0 END
        ELSE CASE WHEN o.realized_return_5d < uni.universe_return_5d THEN 1 ELSE 0 END END
FROM dbo.ml_prediction_outcomes o
INNER JOIN uni ON uni.trading_date = o.trading_date
WHERE o.universe_return_5d IS NULL AND o.realized_return_5d IS NOT NULL
"""


def classify_signal(signal):
    """Map a predicted_signal string to 'bullish'/'bearish' (or None)."""
    s = str(signal).lower()
    if 'buy' in s or 'up' in s:
        return 'bullish'
    if 'sell' in s or 'down' in s:
        return 'bearish'
    return None


def _load_model_version():
    """Best-effort (timestamp, version) of the CURRENT model artifacts.

    Only a fallback now: predictions written since Sep 2026 carry their own
    model_version, which is authoritative. See model_version.py.
    """
    return load_model_version()


def ensure_outcomes_table(db):
    engine = db.get_sqlalchemy_engine()
    with engine.begin() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.execute(text(MIGRATE_COLUMNS_SQL))


def evaluate_new_predictions(db=None, days_back=120):
    """Score all evaluable, not-yet-scored predictions. Returns rows written."""
    db = db or SQLServerConnection()
    ensure_outcomes_table(db)

    backfill_universe_metrics(db, days_back=days_back)

    df = db.execute_query(PENDING_PREDICTIONS_SQL, params={'days_back': days_back})
    if df.empty:
        print("[EVAL] No new predictions ready for evaluation")
        return 0
    # Historical duplicate runs of the same (ticker, trading_date): score the first
    df = df.sort_values('prediction_id').drop_duplicates(['ticker', 'trading_date'])

    df['direction'] = df['predicted_signal'].map(classify_signal)
    df = df[df['direction'].notna()].copy()
    if df.empty:
        print("[EVAL] No predictions with recognizable Buy/Sell signals")
        return 0

    df['realized_return_5d'] = (
        (df['realized_close_5d'] - df['entry_close']) / df['entry_close'] * 100
    )
    df['correct'] = np.where(
        df['direction'] == 'bullish',
        df['realized_return_5d'] > 0,
        df['realized_return_5d'] <= 0,
    ).astype(int)
    df['excess_return_5d'] = df['realized_return_5d'] - df['universe_return_5d']
    df['beat_universe'] = np.where(
        df['direction'] == 'bullish',
        df['excess_return_5d'] > 0,
        df['excess_return_5d'] < 0,
    ).astype(float)
    df.loc[df['universe_return_5d'].isna(), 'beat_universe'] = np.nan

    # Prefer the version stamped on the prediction row at write time (exact).
    # Fall back to the run_timestamp heuristic only for rows written before
    # prediction-time stamping existed (Sep 2026) — those carry NULL.
    if 'model_version' not in df.columns:
        df['model_version'] = None
    unstamped = df['model_version'].isna()
    if unstamped.any():
        train_ts, version = _load_model_version()
        if train_ts and version:
            try:
                trained_at = pd.to_datetime(train_ts, format='%Y%m%d_%H%M%S')
                df.loc[unstamped & (pd.to_datetime(df['run_timestamp']) >= trained_at),
                       'model_version'] = version
            except Exception:
                pass

    # Rows written before the suppression flag existed had survived the old
    # drop-style filters, so NULL means actionable.
    df['is_actionable'] = df['is_actionable'].fillna(True).astype(bool)

    out_cols = ['prediction_id', 'ticker', 'trading_date', 'predicted_signal',
                'confidence', 'entry_close', 'realized_close_5d',
                'realized_return_5d', 'realized_date', 'correct', 'model_version',
                'is_actionable', 'suppression_reason',
                'universe_return_5d', 'universe_pct_up_5d', 'excess_return_5d', 'beat_universe']
    out = df[out_cols]

    engine = db.get_sqlalchemy_engine()
    out.to_sql(OUTCOMES_TABLE, engine, if_exists='append', index=False)

    actionable = out[out['is_actionable']]
    acc = actionable['correct'].mean() if len(actionable) else float('nan')
    print(f"[EVAL] Scored {len(out):,} predictions "
          f"({out['trading_date'].min()} .. {out['trading_date'].max()}), "
          f"actionable accuracy={acc:.1%} ({len(actionable):,} actionable, "
          f"{len(out) - len(actionable):,} suppressed)")
    return len(out)


def backfill_universe_metrics(db, days_back=120):
    """Populate universe/excess/beat columns on already-scored outcomes."""
    engine = db.get_sqlalchemy_engine()
    with engine.begin() as conn:
        n = conn.execute(text(BACKFILL_UNIVERSE_SQL), {'days_back': days_back}).rowcount
    if n and n > 0:
        print(f"[EVAL] Backfilled universe/excess-return metrics on {n:,} outcomes")
    return n


ROLLING_SQL = """
WITH recent AS (
    SELECT DISTINCT TOP (:n_days) trading_date
    FROM dbo.ml_prediction_outcomes
    ORDER BY trading_date DESC
)
SELECT o.predicted_signal, o.confidence, o.correct, o.trading_date,
       o.is_actionable, o.excess_return_5d, o.beat_universe, t.sector
FROM dbo.ml_prediction_outcomes o
INNER JOIN recent r ON o.trading_date = r.trading_date
LEFT JOIN dbo.nasdaq_top100 t ON t.ticker = o.ticker
WHERE 1 = 1
""" + CLEAN_OUTCOMES_FILTER


def get_rolling_accuracy(db=None, n_days=20):
    """Accuracy over the last n_days *trading* days of evaluated outcomes.

    All stats are computed on actionable outcomes only (NULL = pre-flag rows,
    which were all actionable), so the MIN_LIVE_ACCURACY retrain gate keeps
    measuring tradeable signals. Full-universe counts are reported separately.

    Returns a dict (json-serializable) or None when no outcomes exist yet.
    """
    db = db or SQLServerConnection()
    try:
        ensure_outcomes_table(db)
        df = db.execute_query(ROLLING_SQL, params={'n_days': n_days})
    except Exception as e:
        print(f"[EVAL] Rolling accuracy unavailable: {e}")
        return None
    if df.empty:
        return None

    df['is_actionable'] = df['is_actionable'].fillna(True).astype(bool)
    df_all = df
    df = df[df['is_actionable']].copy()
    if df.empty:
        return None

    df['direction'] = df['predicted_signal'].map(classify_signal)
    buys = df[df['direction'] == 'bullish']
    sells = df[df['direction'] == 'bearish']

    buckets = {}
    for lo, hi, label in [(0.0, 0.55, '<55%'), (0.55, 0.60, '55-60%'),
                          (0.60, 0.67, '60-67%'), (0.67, 1.01, '>=67%')]:
        band = df[(df['confidence'] >= lo) & (df['confidence'] < hi)]
        if len(band) > 0:
            buckets[label] = {'n': int(len(band)),
                              'accuracy': round(float(band['correct'].mean()), 4)}

    by_sector = {}
    if 'sector' in df.columns:
        for sector, grp in df[df['sector'].notna()].groupby('sector'):
            if len(grp) >= 10:
                by_sector[sector] = {'n': int(len(grp)),
                                     'accuracy': round(float(grp['correct'].mean()), 4)}

    suppressed = df_all[~df_all['is_actionable']]

    def _beat_rate(frame):
        frame = frame[frame['beat_universe'].notna()]
        return round(float(frame['beat_universe'].astype(float).mean()), 4) if len(frame) else None

    skill = df[df['excess_return_5d'].notna()]
    directional_excess = np.where(
        skill['predicted_signal'].map(classify_signal) == 'bullish',
        skill['excess_return_5d'], -skill['excess_return_5d'])

    return {
        'window_trading_days': n_days,
        'n_predictions': int(len(df)),
        'n_suppressed': int(len(suppressed)),
        'suppressed_accuracy': round(float(suppressed['correct'].mean()), 4) if len(suppressed) else None,
        'accuracy': round(float(df['correct'].mean()), 4),
        'buy_accuracy': round(float(buys['correct'].mean()), 4) if len(buys) else None,
        'buy_n': int(len(buys)),
        'sell_accuracy': round(float(sells['correct'].mean()), 4) if len(sells) else None,
        'sell_n': int(len(sells)),
        # Skill vs the market: share of signals on the right side of the universe
        # average, and mean 5d excess return in the signal's direction (Sell flipped).
        'beat_universe_rate': _beat_rate(df),
        'buy_beat_universe_rate': _beat_rate(buys),
        'sell_beat_universe_rate': _beat_rate(sells),
        'avg_directional_excess_pct': round(float(directional_excess.mean()), 4) if len(skill) else None,
        'by_confidence': buckets,
        'by_sector': by_sector,
        'first_date': str(df['trading_date'].min()),
        'last_date': str(df['trading_date'].max()),
    }


def print_summary(db=None):
    db = db or SQLServerConnection()
    for window in (20, 60):
        stats = get_rolling_accuracy(db, n_days=window)
        if stats is None:
            print(f"[EVAL] No outcomes yet for {window}-day window")
            continue
        print(f"\n[EVAL] Rolling {window} trading days "
              f"({stats['first_date']} .. {stats['last_date']}):")
        print(f"  Overall: {stats['accuracy']:.1%} ({stats['n_predictions']:,} predictions)")
        if stats['buy_accuracy'] is not None:
            print(f"  Buy:     {stats['buy_accuracy']:.1%} ({stats['buy_n']:,})")
        if stats['sell_accuracy'] is not None:
            print(f"  Sell:    {stats['sell_accuracy']:.1%} ({stats['sell_n']:,})")
        if stats['beat_universe_rate'] is not None:
            print(f"  Beat universe: {stats['beat_universe_rate']:.1%} "
                  f"(Buy {stats['buy_beat_universe_rate'] or 0:.1%} / "
                  f"Sell {stats['sell_beat_universe_rate'] or 0:.1%}), "
                  f"avg directional excess {stats['avg_directional_excess_pct']:+.2f}%")
        for label, b in stats['by_confidence'].items():
            print(f"  Confidence {label}: {b['accuracy']:.1%} ({b['n']:,})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ML predictions vs realized 5-day returns')
    parser.add_argument('--days-back', type=int, default=120,
                        help='Price history window for the forward-return scan (default: 120)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Skip evaluation, just print rolling accuracy')
    args = parser.parse_args()

    db = SQLServerConnection()
    if not args.summary_only:
        evaluate_new_predictions(db, days_back=args.days_back)
        # Re-derive reliability thresholds from the freshly-scored outcomes so the gates
        # track the current model's realized accuracy (Layer 2 feedback loop). Best-effort:
        # imported lazily to avoid a circular import and never block evaluation.
        try:
            from derive_thresholds import derive_and_save
            derive_and_save(db=db)
        except Exception as e:  # noqa: BLE001
            print(f"[EVAL] Threshold re-derivation skipped: {e}")
    print_summary(db)


if __name__ == '__main__':
    main()
