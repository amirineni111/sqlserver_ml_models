"""
Model version identity for the NASDAQ prediction pipeline.

A single source for "which model produced this row". The string is
``{training_timestamp}@{git_commit}`` (e.g. ``20260830_143033@48db759``), read
from ``data/training_metadata.pkl`` written by the retrain.

Historically this lived in evaluate_predictions.py and was stamped only at
*evaluation* time, which left 98.9% of ml_prediction_outcomes rows with a NULL
model_version and starved derive_thresholds.py of current-model outcomes.
Predictions now stamp their own version at write time (see export_to_database.py)
and evaluation carries that value through, so attribution is exact rather than
inferred from run_timestamp.
"""

import os
import pickle

METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'training_metadata.pkl')

# VARCHAR(50) in ml_trading_predictions / ml_prediction_outcomes
MAX_VERSION_LEN = 50


def load_model_version(metadata_path=None):
    """Return (training_timestamp, version_string) for the deployed model.

    Both are None when the metadata file is missing or unreadable — callers must
    treat an unknown version as NULL rather than failing the run.
    """
    try:
        with open(metadata_path or METADATA_PATH, 'rb') as f:
            meta = pickle.load(f)
        ts = meta.get('training_timestamp')
        if not ts:
            return None, None
        commit = meta.get('git_commit')
        version = f"{ts}@{commit}" if commit else str(ts)
        return ts, version[:MAX_VERSION_LEN]
    except Exception:
        return None, None
