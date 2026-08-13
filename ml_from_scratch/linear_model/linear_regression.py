"""Linear regression implementation."""

from typing import Any

import numpy as np

from ml_from_scratch.base import BaseEstimator, RegressorMixin
from ml_from_scratch.utils.validation import check_array, check_X_y


class LinearRegression(RegressorMixin, BaseEstimator):
    """Ordinary least squares linear regression."""

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X: Any, y: Any = None) -> "LinearRegression":
        """Fit the linear model using least squares."""
        if y is None:
            raise ValueError("y is required for LinearRegression.fit.")

        X_checked, y_checked = check_X_y(X, y)
        design_matrix = self._add_intercept_column(X_checked)

        weights, *_ = np.linalg.lstsq(design_matrix, y_checked, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights

        self.n_features_in_ = X_checked.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for input samples."""
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        return X_checked @ self.coef_ + self.intercept_

    def _add_intercept_column(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X

        return np.c_[np.ones(X.shape[0]), X]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_") or not hasattr(self, "intercept_"):
            raise ValueError("This LinearRegression instance is not fitted yet.")
