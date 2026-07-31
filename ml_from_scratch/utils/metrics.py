"""Model evaluation metrics."""

from typing import Any

import numpy as np


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Return the mean squared error regression loss."""
    y_true_checked, y_pred_checked = _check_targets(y_true, y_pred)
    return float(np.mean((y_true_checked - y_pred_checked) ** 2))


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Return the mean absolute error regression loss."""
    y_true_checked, y_pred_checked = _check_targets(y_true, y_pred)
    return float(np.mean(np.abs(y_true_checked - y_pred_checked)))


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Return the coefficient of determination."""
    y_true_checked, y_pred_checked = _check_targets(y_true, y_pred)
    residual_sum_of_squares = np.sum((y_true_checked - y_pred_checked) ** 2)
    total_sum_of_squares = np.sum((y_true_checked - np.mean(y_true_checked)) ** 2)

    if total_sum_of_squares == 0:
        return 1.0 if residual_sum_of_squares == 0 else 0.0

    return float(1 - residual_sum_of_squares / total_sum_of_squares)


def _check_targets(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    y_true_checked = np.asarray(y_true, dtype=float)
    y_pred_checked = np.asarray(y_pred, dtype=float)

    if y_true_checked.ndim == 2 and y_true_checked.shape[1] == 1:
        y_true_checked = y_true_checked.ravel()

    if y_pred_checked.ndim == 2 and y_pred_checked.shape[1] == 1:
        y_pred_checked = y_pred_checked.ravel()

    if y_true_checked.ndim != 1 or y_pred_checked.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D array-like objects.")

    if y_true_checked.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one value.")

    if y_true_checked.shape != y_pred_checked.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    return y_true_checked, y_pred_checked
