"""
Outcome-driven threshold derivation (Layer 2 of the Jun 2026 signal-suppression fix).

The reliability gates in nasdaq_config.py used to be hardcoded constants tuned by hand
against a particular model calibration. When the Jun 21 2026 retrain switched to
sigmoid/Platt calibration the model's confidence range collapsed to ~0.50-0.60, the old
0.67 buy gate became unreachable, and 100% of Buy signals were silently suppressed
(buy_signals = 0 every day). See [[nasdaq-accuracy-overhaul-june-2026]].

This module re-derives the gates from realized accuracy in ml_prediction_outcomes — the
actual feedback loop — so they always track the deployed model's real confidence
distribution. It writes data/derived_thresholds.json, which nasdaq_config.py loads at
import time (falling back to the hardcoded defaults when the file is absent/malformed).

Philosophy: BALANCED. The buy gate is the lowest confidence cutoff whose realized
accuracy clears a modest target (default 53%, the accuracy-curve inflection) with enough
samples — surfacing a useful volume of buys while keeping each meaningfully above
coin-flip. Suppressed rows are still written downstream (is_actionable=0), so lowering a
gate adds flagged rows rather than dropping anything.

Usage:
    python derive_thresholds.py                 # derive over default window, write JSON
    python derive_thresholds.py --window 365     # wider history
    python derive_thresholds.py --dry-run        # print, don't write
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import SQLServerConnection  # noqa: E402
from evaluate_predictions import classify_signal, ensure_outcomes_table  # noqa: E402

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'derived_thresholds.json')

# Tuning knobs (overridable via CLI / kwargs)
DEFAULT_WINDOW_DAYS = 180     # trailing calendar days of outcomes to learn from
BUY_ACC_TARGET = 0.53         # balanced inflection: first tier meaningfully > 0.50
HIGH_CONF_ACC_TARGET = 0.58   # tier approaching the genuinely-good (~60% acc) band
MIN_SAMPLES = 200             # min scored signals behind any chosen cutoff
SECTOR_MIN_SAMPLES = 150      # min per-sector samples to emit a sector override
ACTIONABLE_BUY_FLOOR = 0.01   # guardrail: warn if buy gate keeps < 1% of buys actionable

# Hardcoded fallbacks (mirror nasdaq_config.py defaults) used when data is insufficient.
FALLBACK = {
    'buy_min_confidence': 0.55,
    'high_confidence_threshold': 0.58,
    'sell_max_confidence': 0.99,  # post Jun-23 retrain: sell calibration is monotonic, don't suppress
    'sector_confidence_overrides': {'Technology': 0.57, 'Healthcare': 0.57},
}

# Only learn from outcomes produced by the CURRENT model. model_version is stamped at
# evaluation time (unreliable for old predictions), so we instead lower-bound the
# prediction trading_date at the deployed model's training date — those predictions used
# the current calibration. Mixing the prior (isotonic) model's wider-spread outcomes is
# exactly what would mis-derive the gates (the bug this layer prevents).
OUTCOMES_SQL = """
SELECT o.predicted_signal, o.confidence, o.correct, o.trading_date,
       o.is_actionable, t.sector
FROM dbo.ml_prediction_outcomes o
LEFT JOIN dbo.nasdaq_top100 t ON t.ticker = o.ticker
WHERE o.correct IS NOT NULL
  AND o.confidence IS NOT NULL
  AND o.trading_date >= DATEADD(day, -:window_days, CAST(GETDATE() AS DATE))
  AND o.trading_date >= :model_since
"""


def _current_model_date():
    """Training date (date string 'YYYY-MM-DD') of the deployed model, or '1900-01-01'."""
    try:
        with open(os.path.join('data', 'training_metadata.pkl'), 'rb') as f:
            ts = pickle.load(f).get('training_timestamp')  # e.g. '20260621_103041'
        if ts:
            return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
    except Exception:
        pass
    return '1900-01-01'


def _lowest_cutoff_for_accuracy(df, acc_target, min_samples):
    """Lowest confidence cutoff C such that rows with confidence >= C reach acc_target
    with >= min_samples behind it. Returns (cutoff, n, accuracy) or None.

    Sweeps candidate cutoffs at the observed confidences (descending): as C drops we add
    lower-confidence rows, so accuracy generally falls — we want the smallest C still
    above target (maximizes volume at acceptable quality).
    """
    if df.empty:
        return None
    candidates = np.sort(np.unique(np.round(df['confidence'].values, 4)))  # ascending
    # Collect every cutoff whose band (confidence >= c) clears target with enough samples,
    # then take the SMALLEST such cutoff to maximize volume at acceptable quality.
    qualifying = []
    for c in candidates:
        band = df[df['confidence'] >= c]
        if len(band) >= min_samples and float(band['correct'].mean()) >= acc_target:
            qualifying.append((round(float(c), 4), int(len(band)), round(float(band['correct'].mean()), 4)))
    return min(qualifying, key=lambda t: t[0]) if qualifying else None


def _highest_cutoff_keeping_accuracy(df, acc_floor, min_samples):
    """Highest cutoff H such that rows with confidence <= H still hold accuracy >= floor.
    Used for the Sell cap (suppress high-confidence sells when they're inversely
    calibrated). Returns cutoff or None.
    """
    if df.empty:
        return None
    candidates = np.sort(np.unique(np.round(df['confidence'].values, 4)))
    best = None
    for c in candidates:
        band = df[df['confidence'] <= c]
        if len(band) < min_samples:
            continue
        if float(band['correct'].mean()) >= acc_floor:
            best = round(float(c), 4)
    return best


def derive(db=None, window_days=DEFAULT_WINDOW_DAYS, model_since=None):
    """Compute thresholds from ml_prediction_outcomes. Returns (result_dict, warnings).

    Only outcomes from predictions dated on/after the current model's training date are
    used (model_since); pass model_since='1900-01-01' to learn across all model versions.
    """
    db = db or SQLServerConnection()
    ensure_outcomes_table(db)
    if model_since is None:
        model_since = _current_model_date()
    df = db.execute_query(OUTCOMES_SQL, params={'window_days': window_days,
                                                'model_since': model_since})

    warnings = []
    result = dict(FALLBACK)
    meta = {
        'derived_at': datetime.now().isoformat(timespec='seconds'),
        'window_days': window_days,
        'model_since': model_since,
        'n_outcomes': 0 if df is None else int(len(df)),
        'source': 'ml_prediction_outcomes',
    }

    if df is None or len(df) < MIN_SAMPLES:
        warnings.append(
            f"Only {0 if df is None else len(df)} current-model outcomes since {model_since} "
            f"(need >= {MIN_SAMPLES}); using hardcoded fallback thresholds until more accrue.")
        result['_meta'] = meta
        result['_warnings'] = warnings
        return result, warnings

    df['correct'] = df['correct'].astype(float)
    df['direction'] = df['predicted_signal'].map(classify_signal)
    buys = df[df['direction'] == 'bullish'].copy()
    sells = df[df['direction'] == 'bearish'].copy()

    # --- Buy gate (balanced inflection) ---
    buy_hit = _lowest_cutoff_for_accuracy(buys, BUY_ACC_TARGET, MIN_SAMPLES)
    if buy_hit:
        result['buy_min_confidence'] = buy_hit[0]
        meta['buy_gate'] = {'cutoff': buy_hit[0], 'n': buy_hit[1], 'accuracy': buy_hit[2]}
    else:
        warnings.append(
            f"No buy cutoff reaches {BUY_ACC_TARGET:.0%} accuracy with >= {MIN_SAMPLES} "
            f"samples; keeping fallback buy_min_confidence={result['buy_min_confidence']}.")

    # --- High-confidence flag tier ---
    high_hit = _lowest_cutoff_for_accuracy(buys, HIGH_CONF_ACC_TARGET, MIN_SAMPLES)
    if high_hit:
        # never below the buy gate
        result['high_confidence_threshold'] = max(high_hit[0], result['buy_min_confidence'])
        meta['high_conf_gate'] = {'cutoff': high_hit[0], 'n': high_hit[1], 'accuracy': high_hit[2]}
    else:
        result['high_confidence_threshold'] = max(result['high_confidence_threshold'],
                                                   result['buy_min_confidence'])
        warnings.append(
            f"No buy cutoff reaches {HIGH_CONF_ACC_TARGET:.0%}; high_confidence_threshold "
            f"set to {result['high_confidence_threshold']} (fallback/buy-gate).")

    # --- Sell cap (sells are inversely calibrated / near-random) ---
    sell_cap = _highest_cutoff_keeping_accuracy(sells, 0.50, MIN_SAMPLES)
    if sell_cap is not None:
        result['sell_max_confidence'] = sell_cap
        meta['sell_cap'] = {'cutoff': sell_cap}
    else:
        warnings.append(
            f"Sells never hold >= 50% accuracy with >= {MIN_SAMPLES} samples (no edge); "
            f"keeping fallback sell_max_confidence={result['sell_max_confidence']}.")

    # --- Per-sector buy overrides (only where the sector lags the base gate) ---
    sector_overrides = {}
    if 'sector' in buys.columns:
        for sector, grp in buys[buys['sector'].notna()].groupby('sector'):
            if len(grp) < SECTOR_MIN_SAMPLES:
                continue
            hit = _lowest_cutoff_for_accuracy(grp, BUY_ACC_TARGET, SECTOR_MIN_SAMPLES)
            if hit and hit[0] > result['buy_min_confidence']:
                sector_overrides[str(sector)] = hit[0]
    if sector_overrides:
        result['sector_confidence_overrides'] = sector_overrides
        meta['sector_overrides_basis'] = {k: v for k, v in sector_overrides.items()}
    # else: keep fallback sector overrides

    # --- Guardrail: would the buy gate strand almost all buys? ---
    if len(buys) > 0:
        kept = float((buys['confidence'] >= result['buy_min_confidence']).mean())
        meta['actionable_buy_fraction'] = round(kept, 4)
        if kept < ACTIONABLE_BUY_FLOOR:
            warnings.append(
                f"GUARDRAIL: derived buy_min_confidence={result['buy_min_confidence']} keeps "
                f"only {kept:.2%} of buys actionable (< {ACTIONABLE_BUY_FLOOR:.0%}). The model's "
                f"confidence range may have shifted — review calibration before trusting signals.")

    result['_meta'] = meta
    result['_warnings'] = warnings
    return result, warnings


def derive_and_save(db=None, window_days=DEFAULT_WINDOW_DAYS, output_path=OUTPUT_PATH,
                    dry_run=False, model_since=None):
    """Derive thresholds and write the JSON consumed by nasdaq_config.py.

    Safe to call at the end of a retrain: failures are logged and swallowed so a bad
    derivation never blocks the pipeline (config falls back to hardcoded defaults).
    """
    try:
        result, warnings = derive(db=db, window_days=window_days, model_since=model_since)
    except Exception as e:  # noqa: BLE001
        print(f"[THRESHOLDS] Derivation failed ({e}); leaving existing thresholds untouched.")
        return None

    for w in warnings:
        print(f"[THRESHOLDS][WARN] {w}")
    print("[THRESHOLDS] Derived: "
          f"buy_min_confidence={result['buy_min_confidence']}, "
          f"high_confidence_threshold={result['high_confidence_threshold']}, "
          f"sell_max_confidence={result['sell_max_confidence']}, "
          f"sector_overrides={result['sector_confidence_overrides']}")

    if dry_run:
        print("[THRESHOLDS] --dry-run: not writing file.")
        return result

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, output_path)  # atomic
    print(f"[THRESHOLDS] Wrote {output_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Derive reliability thresholds from ml_prediction_outcomes")
    ap.add_argument('--window', type=int, default=DEFAULT_WINDOW_DAYS, help='trailing days of outcomes')
    ap.add_argument('--dry-run', action='store_true', help='print results without writing JSON')
    ap.add_argument('--all-models', action='store_true',
                    help='learn across all model versions (default: current model only)')
    args = ap.parse_args()
    derive_and_save(window_days=args.window, dry_run=args.dry_run,
                    model_since='1900-01-01' if args.all_models else None)


if __name__ == '__main__':
    main()
