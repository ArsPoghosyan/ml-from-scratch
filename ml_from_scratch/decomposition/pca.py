"""Principal component analysis implementation."""
from __future__ import annotations

from typing import Any

import numpy as np

from ml_from_scratch.base import BaseEstimator
from ml_from_scratch.utils.validation import check_array


class PCA(BaseEstimator):
    """Principal component analysis using singular value decomposition."""

    def __init__(
        self,
        n_components: int | None = None,
        whiten: bool = False,
        center: bool = True,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.center = center

    def fit(self, X: Any, y: Any = None) -> PCA:
        """Fit PCA by learning the principal axes of the input data."""
        X_checked = check_array(X)
        n_samples, n_features = X_checked.shape

        if n_samples < 2:
            raise ValueError("PCA requires at least 2 samples.")

        if not isinstance(self.whiten, bool):
            raise TypeError("whiten must be a boolean.")

        if not isinstance(self.center, bool):
            raise TypeError("center must be a boolean.")

        if isinstance(self.n_components, bool) or not isinstance(
            self.n_components, (type(None), int, np.integer)
        ):
            raise TypeError("n_components must be an integer or None.")

        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = int(self.n_components)

        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

        if n_components > min(n_samples, n_features):
            raise ValueError(
                "n_components cannot be greater than min(n_samples, n_features)."
            )

        if self.center:
            mean = X_checked.mean(axis=0)
        else:
            mean = np.zeros(n_features, dtype=float)

        X_centered = X_checked - mean
        _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)

        explained_variance = (singular_values**2) / (n_samples - 1)
        total_variance = float(explained_variance.sum())

        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples
        self.n_components_ = n_components
        self.mean_ = mean
        self.components_ = vt[:n_components].copy()
        self.singular_values_ = singular_values[:n_components].copy()
        self.explained_variance_ = explained_variance[:n_components].copy()
        if total_variance > 0.0:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros(n_components, dtype=float)

        if self.whiten:
            scale = np.sqrt(self.explained_variance_)
            scale[scale == 0.0] = 1.0
            self.whiten_scale_ = scale

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Project samples onto the learned principal component axes."""
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        X_centered = X_checked - self.mean_
        transformed = X_centered @ self.components_.T

        if self.whiten:
            transformed = transformed / self.whiten_scale_

        return transformed

    def inverse_transform(self, X: Any) -> np.ndarray:
        """Reconstruct data from the PCA representation."""
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_components_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        reconstructed = X_checked
        if self.whiten:
            reconstructed = reconstructed * self.whiten_scale_

        return reconstructed @ self.components_ + self.mean_

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the model and return the transformed data."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: Any) -> np.ndarray:
        """Alias for transform to satisfy the project estimator interface."""
        return self.transform(X)

    def _check_is_fitted(self) -> None:
        required_attributes = (
            "components_",
            "mean_",
            "n_features_in_",
            "n_components_",
            "explained_variance_",
        )
        if any(not hasattr(self, attribute) for attribute in required_attributes):
            raise ValueError("This PCA instance is not fitted yet.")
