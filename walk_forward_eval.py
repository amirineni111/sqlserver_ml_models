"""
Walk-Forward Evaluation Harness (offline)

Simulates production: train on everything up to a cutoff, predict the next
~month, roll forward. This is the gate for model/threshold changes — a change
ships only if it beats the baseline here, because walk-forward accuracy is the
closest offline proxy for realized live accuracy.

Reuses UltraFastWeeklyRetrainer's data loading, feature engineering and
dataset prep (including its train-slice-only feature selection — folds are
constrained to test AFTER the selection cutoff so selection never sees a
fold's test window).

Not part of the daily/weekly schedule — run manually:
    python walk_forward_eval.py                          # 6 folds, ~21 trading days each
    python walk_forward_eval.py --folds 4 --test-window 21
    python walk_forward_eval.py --dead-zone 0.5          # sweep label dead-zone
    python walk_forward_eval.py --compare-gb             # also run legacy GradientBoosting
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from model_calibration import SigmoidCalibratedClassifier
from weekly_retrain_ultra_fast import UltraFastWeeklyRetrainer


def build_models(compare_gb=False):
    """Candidate models with production hyperparameters."""
    models = {
        'Hist Gradient Boosting': lambda: HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.1, max_leaf_nodes=31,
            l2_regularization=1.0, min_samples_leaf=20,
            early_stopping=True, validation_fraction=0.1,
            random_state=42
        ),
    }
    if compare_gb:
        models['Gradient Boosting'] = lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, min_samples_split=10, min_samples_leaf=5,
            subsample=0.8, random_state=42
        )
    return models


def evaluate_fold(model_factory, X, y, y_return, dates, train_dates, cal_dates,
                  test_dates, dead_zone_pct):
    """Train + calibrate on one fold, return test metrics."""
    date_vals = dates.values
    train_mask = np.isin(date_vals, train_dates)
    cal_mask = np.isin(date_vals, cal_dates)
    test_mask = np.isin(date_vals, test_dates)

    # Label dead-zone: drop near-zero movers from train/cal only
    dead_zone = np.abs(y_return.values) < dead_zone_pct
    train_mask = train_mask & ~dead_zone
    cal_mask = cal_mask & ~dead_zone

    X_train, X_cal, X_test = X[train_mask], X[cal_mask], X[test_mask]
    y_train, y_cal, y_test = y[train_mask], y[cal_mask], y[test_mask]
    test_dead_zone = dead_zone[test_mask]

    if len(X_train) < 500 or len(X_test) == 0 or len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    # Class balance * time recency weights (rows are date-sorted)
    class_weights = compute_sample_weight('balanced', y_train)
    pos = np.arange(len(y_train)) / len(y_train)
    time_weights = np.exp(1.2 * (pos - 1))
    sample_weights = class_weights * time_weights
    sample_weights = sample_weights / sample_weights.mean()

    model = model_factory()
    try:
        model.fit(X_train_s, y_train, sample_weight=sample_weights)
    except TypeError:
        model.fit(X_train_s, y_train)

    # Platt calibration on the held-out calibration slice
    calibrated = model
    if len(X_cal) > 50 and len(np.unique(y_cal)) == 2:
        raw_cal = model.predict_proba(X_cal_s)[:, 1]
        platt = LogisticRegression(solver='lbfgs', max_iter=1000)
        platt.fit(SigmoidCalibratedClassifier.log_odds(raw_cal), y_cal)
        calibrated = SigmoidCalibratedClassifier(model, platt, model.classes_)

    preds = calibrated.predict(X_test_s)
    probs = calibrated.predict_proba(X_test_s)
    conf = probs.max(axis=1)

    ex_dz = ~test_dead_zone
    metrics = {
        'n_test': int(len(y_test)),
        'accuracy': round(float(accuracy_score(y_test, preds)), 4),
        'accuracy_ex_dead_zone': (
            round(float(accuracy_score(y_test[ex_dz], preds[ex_dz])), 4)
            if ex_dz.sum() > 0 else None
        ),
        'f1_weighted': round(float(f1_score(y_test, preds, average='weighted',
                                            zero_division=0)), 4),
        'test_start': str(test_dates.min().astype('datetime64[D]')),
        'test_end': str(test_dates.max().astype('datetime64[D]')),
        'by_confidence': {},
        'by_side': {},
    }

    for lo, hi, label in [(0.0, 0.55, '<55%'), (0.55, 0.60, '55-60%'),
                          (0.60, 0.67, '60-67%'), (0.67, 1.01, '>=67%')]:
        band = (conf >= lo) & (conf < hi)
        if band.sum() >= 10:
            metrics['by_confidence'][label] = {
                'n': int(band.sum()),
                'accuracy': round(float(accuracy_score(y_test[band], preds[band])), 4),
            }

    # 'Down' encodes to 0, 'Up' to 1 (LabelEncoder, alphabetical)
    for cls, label in [(1, 'buy'), (0, 'sell')]:
        side = preds == cls
        if side.sum() >= 10:
            metrics['by_side'][label] = {
                'n': int(side.sum()),
                'accuracy': round(float(accuracy_score(y_test[side], preds[side])), 4),
            }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Walk-forward evaluation of the NASDAQ ML pipeline')
    parser.add_argument('--folds', type=int, default=6, help='Number of walk-forward folds')
    parser.add_argument('--test-window', type=int, default=21,
                        help='Test window length in trading days (default ~1 month)')
    parser.add_argument('--days-back', type=int, default=730, help='Training data window')
    parser.add_argument('--dead-zone', type=float, default=None,
                        help='Label dead-zone %% override (default: nasdaq_config.LABEL_DEAD_ZONE_PCT)')
    parser.add_argument('--compare-gb', action='store_true',
                        help='Also evaluate the legacy GradientBoostingClassifier')
    args = parser.parse_args()

    from nasdaq_config import LABEL_DEAD_ZONE_PCT
    dead_zone_pct = args.dead_zone if args.dead_zone is not None else LABEL_DEAD_ZONE_PCT

    print("=" * 70)
    print(f"[WALK-FORWARD] folds={args.folds}, test_window={args.test_window} trading days, "
          f"dead_zone={dead_zone_pct}%")
    print("=" * 70)

    retrainer = UltraFastWeeklyRetrainer(backup_old=False, days_back=args.days_back)
    df = retrainer.load_training_data()
    df = retrainer.create_target_variable(df)
    df_features = retrainer.engineer_features_vectorized(df)
    X, y, y_return, encoder, feature_cols, dates = retrainer.prepare_ml_dataset(df_features)

    unique_dates = np.sort(pd.unique(dates.values))
    n_dates = len(unique_dates)
    purge = 5

    # Feature selection used data through the 60th-percentile date; folds whose
    # test windows start before that would leak through feature selection.
    selection_cutoff = unique_dates[int(0.60 * n_dates)]
    needed = args.folds * args.test_window
    available = (unique_dates > selection_cutoff).sum()
    if needed > available:
        max_folds = max(available // args.test_window, 1)
        print(f"[WARN] Only {available} trading days after the feature-selection cutoff "
              f"({selection_cutoff.astype('datetime64[D]')}); reducing folds "
              f"{args.folds} -> {max_folds}")
        args.folds = max_folds

    models = build_models(compare_gb=args.compare_gb)
    results = {name: [] for name in models}

    for i in range(args.folds):
        end = n_dates - (args.folds - 1 - i) * args.test_window
        test_dates = unique_dates[end - args.test_window:end]
        train_pool = unique_dates[:end - args.test_window - purge]

        # Last 15% of the training pool (behind its own purge gap) calibrates
        cal_n = max(int(0.15 * len(train_pool)), 10)
        cal_dates = train_pool[-cal_n:]
        train_dates = train_pool[:-(cal_n + purge)]

        print(f"\n[FOLD {i + 1}/{args.folds}] test "
              f"{test_dates.min().astype('datetime64[D]')} .. {test_dates.max().astype('datetime64[D]')} "
              f"(train: {len(train_dates)} days, cal: {len(cal_dates)} days)")

        for name, factory in models.items():
            m = evaluate_fold(factory, X, y, y_return, dates,
                              train_dates, cal_dates, test_dates, dead_zone_pct)
            if m is None:
                print(f"  {name}: skipped (insufficient data)")
                continue
            results[name].append(m)
            print(f"  {name}: acc={m['accuracy']:.3f} "
                  f"(ex-dead-zone={m['accuracy_ex_dead_zone']}) "
                  f"f1={m['f1_weighted']:.3f} n={m['n_test']:,}")
            for label, b in m['by_confidence'].items():
                print(f"    conf {label}: {b['accuracy']:.3f} ({b['n']:,})")

    # Aggregate
    print("\n" + "=" * 70)
    print("[WALK-FORWARD SUMMARY]")
    summary = {'run_at': datetime.now().isoformat(), 'dead_zone_pct': dead_zone_pct,
               'folds': args.folds, 'test_window': args.test_window, 'models': {}}
    for name, fold_metrics in results.items():
        if not fold_metrics:
            continue
        accs = [m['accuracy'] for m in fold_metrics]
        summary['models'][name] = {
            'mean_accuracy': round(float(np.mean(accs)), 4),
            'std_accuracy': round(float(np.std(accs)), 4),
            'folds': fold_metrics,
        }
        print(f"  {name}: mean acc={np.mean(accs):.3f} (+/- {np.std(accs):.3f}) "
              f"over {len(fold_metrics)} folds")

    out_dir = Path('data')
    out_path = out_dir / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] Results -> {out_path}")


if __name__ == '__main__':
    main()
