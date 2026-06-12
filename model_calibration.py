"""Shared probability calibration utilities for NASDAQ ML pipeline.

Defines IsotonicCalibratedClassifier at module level so joblib can
serialize/deserialize it correctly across different entry-point scripts.
"""
import numpy as np


class IsotonicCalibratedClassifier:
    """Probability calibrator using isotonic regression on a held-out set.

    Replaces CalibratedClassifierCV(cv='prefit') which was removed in
    sklearn 1.6+. Defined at module level (not inside a class or function)
    so it is picklable by joblib regardless of which script is __main__.

    DEPRECATED for new training runs: isotonic calibration produced
    inversely-calibrated sell probabilities (high-confidence sells were
    LESS accurate than low-confidence ones). Kept only so joblib can
    deserialize models trained before the switch to Platt scaling.
    """

    def __init__(self, base_model, calibrator, classes):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = classes

    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        cal_probs_pos = self.calibrator.predict(raw_probs)
        cal_probs_pos = np.clip(cal_probs_pos, 0.0, 1.0)
        return np.column_stack([1.0 - cal_probs_pos, cal_probs_pos])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class SigmoidCalibratedClassifier:
    """Probability calibrator using Platt scaling on a held-out set.

    Fits a logistic regression on the base model's log-odds. Unlike
    isotonic regression, the sigmoid is a smooth monotonic map that does
    not overfit small calibration sets — this fixes the inverted sell
    calibration observed with isotonic (Sell@70% conf = 43% accuracy vs
    Sell@50-54% conf = 53-57% accuracy).
    """

    def __init__(self, base_model, calibrator, classes):
        self.base_model = base_model
        self.calibrator = calibrator  # LogisticRegression fit on log-odds
        self.classes_ = classes

    @staticmethod
    def log_odds(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def predict_proba(self, X):
        raw_probs = self.base_model.predict_proba(X)[:, 1]
        cal_probs_pos = self.calibrator.predict_proba(self.log_odds(raw_probs))[:, 1]
        return np.column_stack([1.0 - cal_probs_pos, cal_probs_pos])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
