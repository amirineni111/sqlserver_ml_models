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
