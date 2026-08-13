"""Base estimator interfaces used by project models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEstimator(ABC):
    """Common interface for estimators.

    Estimators should learn parameters in `fit` and use them in `predict`.
    The methods intentionally mirror the familiar scikit-learn API.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any | None = None) -> BaseEstimator:
        """Fit the estimator to training data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict outputs for input data."""
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """Return public constructor-style parameters."""
        return {
            name: value
            for name, value in self.__dict__.items()
            if not name.endswith("_") and not name.startswith("_")
        }


class RegressorMixin:
    """Marker mixin for regression estimators."""

    estimator_type = "regressor"


class ClassifierMixin:
    """Marker mixin for classification estimators."""

    estimator_type = "classifier"
