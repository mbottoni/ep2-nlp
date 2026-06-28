"""Evaluation metrics for the regression and classification tasks.

Uses numpy + scikit-learn only (no torch), so it is testable standalone.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE, R2 and Pearson correlation for vowel-density regression."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson = float("nan")
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": pearson,
    }


def classification_metrics(y_true, y_pred, num_classes: int = 3) -> dict:
    """Accuracy plus per-class accuracy and the confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.divide(np.diag(cm), cm.sum(axis=1))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_accuracy": np.nan_to_num(per_class).tolist(),
        "confusion_matrix": cm.tolist(),
    }
