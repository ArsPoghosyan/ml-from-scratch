"""Input validation helpers."""

from typing import Any

import numpy as np


def check_array(X: Any) -> np.ndarray:
    """Convert input features to a 2D NumPy array."""
    array = np.asarray(X, dtype=float)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    if array.ndim != 2:
        raise ValueError("X must be a 1D or 2D array-like object.")

    if array.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")

    if array.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")

    return array


def check_X_y(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate feature and target arrays for supervised estimators."""
    X_checked = check_array(X)
    y_checked = np.asarray(y, dtype=float)

    if y_checked.ndim == 2 and y_checked.shape[1] == 1:
        y_checked = y_checked.ravel()

    if y_checked.ndim != 1:
        raise ValueError("y must be a 1D array-like object.")

    if y_checked.shape[0] == 0:
        raise ValueError("y must contain at least one target value.")

    if X_checked.shape[0] != y_checked.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    return X_checked, y_checked
