"""Logistic regression implementation."""

from typing import Any

import numpy as np

from ml_from_scratch.base import BaseEstimator, ClassifierMixin
from ml_from_scratch.utils.validation import check_array, check_X_y


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression trained with gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        tol: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.tol = tol

    def fit(self, X: Any, y: Any = None) -> "LogisticRegression":
        """Fit the model to binary classification data."""
        if y is None:
            raise ValueError("y is required for LogisticRegression.fit.")

        X_checked, y_checked = check_X_y(X, y)
        self.classes_ = np.unique(y_checked)

        if self.classes_.shape[0] != 2:
            raise ValueError("LogisticRegression supports exactly two classes.")

        y_binary = (y_checked == self.classes_[1]).astype(float)
        n_samples, n_features = X_checked.shape

        self.coef_ = np.zeros(n_features, dtype=float)
        self.intercept_ = 0.0
        self.loss_history_ = []

        previous_loss = None
        for iteration in range(self.max_iter):
            linear_output = X_checked @ self.coef_
            if self.fit_intercept:
                linear_output = linear_output + self.intercept_

            probabilities = self._sigmoid(linear_output)
            errors = probabilities - y_binary

            self.coef_ -= self.learning_rate * (X_checked.T @ errors) / n_samples
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * float(np.mean(errors))
            else:
                self.intercept_ = 0.0

            updated_probabilities = self._sigmoid(
                X_checked @ self.coef_ + self.intercept_
            )
            loss = self._binary_cross_entropy(y_binary, updated_probabilities)
            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if previous_loss is not None and abs(previous_loss - loss) < self.tol:
                break

            previous_loss = loss

        self.n_features_in_ = n_features
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class membership probabilities."""
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        linear_output = X_checked @ self.coef_ + self.intercept_
        positive_class_prob = self._sigmoid(linear_output)
        return np.column_stack((1.0 - positive_class_prob, positive_class_prob))

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for input samples."""
        probabilities = self.predict_proba(X)
        predicted_positive = probabilities[:, 1] >= 0.5
        return np.where(predicted_positive, self.classes_[1], self.classes_[0])

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        clipped_values = np.clip(values, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped_values))

    def _binary_cross_entropy(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> float:
        epsilon = np.finfo(float).eps
        y_prob = np.clip(y_prob, epsilon, 1.0 - epsilon)
        return float(
            -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
        )

    def _check_is_fitted(self) -> None:
        if (
            not hasattr(self, "coef_")
            or not hasattr(self, "intercept_")
            or not hasattr(self, "classes_")
            or not hasattr(self, "n_features_in_")
        ):
            raise ValueError("This LogisticRegression instance is not fitted yet.")
