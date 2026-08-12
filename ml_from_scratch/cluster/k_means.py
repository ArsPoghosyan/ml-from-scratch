"""K-means clustering implementation."""

from typing import Any, Optional

import numpy as np

from ml_from_scratch.base import BaseEstimator
from ml_from_scratch.utils.validation import check_array


class KMeans(BaseEstimator):
    """K-means clustering with multiple random restarts."""

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "KMeans":
        """Compute k-means clustering."""
        X_checked = check_array(X)
        n_samples, n_features = X_checked.shape

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than the number of samples.")

        if self.n_init < 1:
            raise ValueError("n_init must be at least 1.")

        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        rng = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

        for _ in range(self.n_init):
            centers = self._initialize_centroids(X_checked, rng)

            for iteration in range(self.max_iter):
                labels = self._assign_clusters(X_checked, centers)
                new_centers = self._update_centroids(X_checked, labels, centers, rng)

                shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
                centers = new_centers

                if shift <= self.tol:
                    break

            labels = self._assign_clusters(X_checked, centers)
            inertia = self._compute_inertia(X_checked, labels, centers)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()
                best_labels = labels.copy()
                best_n_iter = iteration + 1

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = best_n_iter
        self.n_features_in_ = n_features
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Assign each sample to the nearest centroid."""
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        return self._assign_clusters(X_checked, self.cluster_centers_)

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit the model and return cluster labels."""
        self.fit(X, y)
        return self.labels_

    def _initialize_centroids(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        if self.init == "random":
            indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[indices].copy()

        if self.init == "k-means++":
            return self._initialize_centroids_kmeans_plus_plus(X, rng)

        raise ValueError("init must be either 'random' or 'k-means++'.")

    def _initialize_centroids_kmeans_plus_plus(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        first_index = rng.integers(0, X.shape[0])
        centroids[0] = X[first_index]

        closest_distances_sq = np.sum((X - centroids[0]) ** 2, axis=1)

        for centroid_index in range(1, self.n_clusters):
            if np.allclose(closest_distances_sq.sum(), 0.0):
                remaining_indices = rng.choice(
                    X.shape[0], size=self.n_clusters - centroid_index, replace=False
                )
                centroids[centroid_index:] = X[remaining_indices]
                break

            probabilities = closest_distances_sq / closest_distances_sq.sum()
            chosen_index = rng.choice(X.shape[0], p=probabilities)
            centroids[centroid_index] = X[chosen_index]

            new_distances_sq = np.sum((X - centroids[centroid_index]) ** 2, axis=1)
            closest_distances_sq = np.minimum(closest_distances_sq, new_distances_sq)

        return centroids

    def _assign_clusters(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances_sq = self._squared_distances(X, centers)
        return np.argmin(distances_sq, axis=1)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        previous_centers: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        new_centers = np.empty_like(previous_centers)

        for cluster_index in range(self.n_clusters):
            cluster_points = X[labels == cluster_index]

            if cluster_points.shape[0] == 0:
                new_centers[cluster_index] = X[rng.integers(0, X.shape[0])]
            else:
                new_centers[cluster_index] = cluster_points.mean(axis=0)

        return new_centers

    def _compute_inertia(
        self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray
    ) -> float:
        distances_sq = self._squared_distances(X, centers)
        return float(np.sum(distances_sq[np.arange(X.shape[0]), labels]))

    def _squared_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "cluster_centers_") or not hasattr(self, "labels_"):
            raise ValueError("This KMeans instance is not fitted yet.")
