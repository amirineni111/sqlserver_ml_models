"""
Layer 3 diagnostic: why does the NASDAQ model predict "Up" ~99% of the time, and did
sigmoid/Platt calibration collapse the confidence spread?

Read-only. Loads the deployed model + today's engineered features for the full ticker
universe and compares RAW (base-estimator) vs CALIBRATED P(Up):
  - If the Up-bias is already in the RAW probs  -> model/feature problem (train/serve skew
    or a genuinely bullish feature regime), calibration is not the culprit.
  - If RAW is balanced but CALIBRATED is skewed/compressed -> calibration shifted the
    operating point; fix the calibration / decision threshold.

It also reports the P(Up) decision threshold that would balance Buy/Sell, which is the
basis for the decision-threshold fix in weekly_retrain_ultra_fast.py.

Usage:  python diagnose_model_bias.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'src'))
from predict_trading_signals import TradingSignalPredictor  # noqa: E402


def _pct(arr):
    q = np.percentile(arr, [1, 10, 25, 50, 75, 90, 99])
    return {f'p{p}': round(float(v), 4) for p, v in zip([1, 10, 25, 50, 75, 90, 99], q)}


def main():
    pred = TradingSignalPredictor()
    model = pred.model
    classes = list(pred.target_encoder.classes_)
    up_idx = classes.index('Up') if 'Up' in classes else 1
    print(f"Model classes: {classes} (Up index = {up_idx})")
    print(f"Calibrator: {type(model).__name__}, base = {type(getattr(model, 'base_model', model)).__name__}")

    # Latest engineered features for every ticker (same path predict_signals uses)
    df = pred.get_latest_data(days_back=120)
    feats = pred.engineer_features(df)
    latest = feats.groupby('ticker').tail(1).copy()
    X = latest[pred.feature_columns].copy()
    X_scaled = pred.scaler.transform(X)
    n = len(X_scaled)
    print(f"\nScored {n} tickers on {latest['trading_date'].max()}")

    cal_up = model.predict_proba(X_scaled)[:, up_idx]
    base = getattr(model, 'base_model', None)
    raw_up = base.predict_proba(X_scaled)[:, up_idx] if base is not None else None

    print(f"\n=== Predicted class balance (argmax @ 0.5) ===")
    pred_up = (cal_up > 0.5).mean()
    print(f"  CALIBRATED: Up={pred_up:.1%}  Down={1 - pred_up:.1%}")
    if raw_up is not None:
        praw = (raw_up > 0.5).mean()
        print(f"  RAW base:   Up={praw:.1%}  Down={1 - praw:.1%}")

    print(f"\n=== P(Up) distribution (percentiles) ===")
    print(f"  CALIBRATED: mean={cal_up.mean():.4f}  {_pct(cal_up)}")
    if raw_up is not None:
        print(f"  RAW base:   mean={raw_up.mean():.4f}  {_pct(raw_up)}")
        print(f"  Spread (p90-p10): raw={np.percentile(raw_up,90)-np.percentile(raw_up,10):.4f}  "
              f"calibrated={np.percentile(cal_up,90)-np.percentile(cal_up,10):.4f}")

    print(f"\n=== P(Up) decision threshold needed for a balanced split (on CALIBRATED) ===")
    for target_up in (0.50, 0.55, 0.60):
        thr = float(np.quantile(cal_up, 1 - target_up))
        print(f"  to call {target_up:.0%} of tickers 'Up' -> threshold P(Up) >= {thr:.4f}")

    print("\nInterpretation:")
    if raw_up is not None:
        if (raw_up > 0.5).mean() > 0.9:
            print("  - Up-bias is present in the RAW model -> not a calibration artifact;")
            print("    investigate train/serve feature drift and class handling (Layer 3 step 1/3).")
        else:
            print("  - RAW is balanced but calibrated is skewed -> calibration/operating-point")
            print("    issue; prefer decision-threshold tuning on P(Up) (Layer 3 step 2).")


if __name__ == '__main__':
    main()
