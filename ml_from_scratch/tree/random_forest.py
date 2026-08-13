"""Random forest implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_from_scratch.base import BaseEstimator, ClassifierMixin, RegressorMixin
from ml_from_scratch.tree.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from ml_from_scratch.utils.validation import check_array, check_X_y


@dataclass
class _ForestMember:
    estimator: BaseEstimator
    feature_indices: np.ndarray


class _BaseRandomForest(BaseEstimator):
    """Shared bootstrap-and-aggregate logic for forest estimators."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_features: Any = None,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> _BaseRandomForest:
        if y is None:
            raise ValueError("y is required for RandomForest.fit.")

        X_checked, y_checked = check_X_y(X, y)
        self.n_features_in_ = X_checked.shape[1]
        self.estimators_ = []

        rng = self._get_rng()
        for _ in range(self.n_estimators):
            feature_indices = self._sample_feature_indices(rng, self.n_features_in_)
            sample_indices = self._sample_row_indices(rng, X_checked.shape[0])

            estimator = self._make_base_estimator()
            estimator.fit(
                X_checked[sample_indices][:, feature_indices],
                y_checked[sample_indices],
            )
            self.estimators_.append(
                _ForestMember(
                    estimator=estimator,
                    feature_indices=np.asarray(feature_indices, dtype=int),
                )
            )

        return self

    def _get_rng(self) -> np.random.Generator:
        if isinstance(self.random_state, np.random.Generator):
            return self.random_state

        return np.random.default_rng(self.random_state)

    def _sample_row_indices(
        self, rng: np.random.Generator, n_samples: int
    ) -> np.ndarray:
        if self.bootstrap:
            return rng.integers(0, n_samples, size=n_samples)

        return np.arange(n_samples)

    def _sample_feature_indices(
        self, rng: np.random.Generator, n_features: int
    ) -> np.ndarray:
        n_selected = self._resolve_max_features(n_features)
        return np.sort(rng.choice(n_features, size=n_selected, replace=False))

    def _resolve_max_features(self, n_features: int) -> int:
        max_features = self.max_features

        if max_features is None:
            return n_features

        if isinstance(max_features, str):
            if max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))

            if max_features == "log2":
                return max(1, int(np.log2(n_features)))

            raise ValueError("max_features must be None, 'sqrt', 'log2', int, or float.")

        if isinstance(max_features, int):
            if max_features < 1:
                raise ValueError("max_features must be at least 1.")
            return min(max_features, n_features)

        if isinstance(max_features, float):
            if not 0.0 < max_features <= 1.0:
                raise ValueError("max_features as a float must be in the interval (0, 1].")
            return max(1, int(max_features * n_features))

        raise TypeError("max_features must be None, a string, int, or float.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "estimators_") or not hasattr(self, "n_features_in_"):
            raise ValueError("This RandomForest instance is not fitted yet.")


class RandomForestClassifier(ClassifierMixin, _BaseRandomForest):
    """Random forest classifier built from decision trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_features: Any = "sqrt",
        criterion: str = "gini",
        random_state: int | None = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            max_features=max_features,
            random_state=random_state,
        )
        self.criterion = criterion

    def fit(self, X: Any, y: Any = None) -> RandomForestClassifier:
        if y is None:
            raise ValueError("y is required for RandomForestClassifier.fit.")

        X_checked, y_checked = check_X_y(X, y)
        self.classes_ = np.unique(y_checked)
        self.n_features_in_ = X_checked.shape[1]
        self.estimators_ = []

        rng = self._get_rng()
        for _ in range(self.n_estimators):
            feature_indices = self._sample_feature_indices(rng, self.n_features_in_)
            sample_indices = self._sample_row_indices(rng, X_checked.shape[0])

            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                criterion=self.criterion,
            )
            estimator.fit(
                X_checked[sample_indices][:, feature_indices],
                y_checked[sample_indices],
            )
            self.estimators_.append(
                _ForestMember(
                    estimator=estimator,
                    feature_indices=np.asarray(feature_indices, dtype=int),
                )
            )

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        probabilities = np.zeros((X_checked.shape[0], self.classes_.shape[0]))
        for member in self.estimators_:
            tree_proba = member.estimator.predict_proba(
                X_checked[:, member.feature_indices]
            )
            class_positions = np.searchsorted(self.classes_, member.estimator.classes_)
            probabilities[:, class_positions] += tree_proba

        return probabilities / len(self.estimators_)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]

    def _check_is_fitted(self) -> None:
        if (
            not hasattr(self, "estimators_")
            or not hasattr(self, "n_features_in_")
            or not hasattr(self, "classes_")
        ):
            raise ValueError("This RandomForestClassifier instance is not fitted yet.")


class RandomForestRegressor(RegressorMixin, _BaseRandomForest):
    """Random forest regressor built from decision trees."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_features: Any = 1.0,
        random_state: int | None = None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            max_features=max_features,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any = None) -> RandomForestRegressor:
        if y is None:
            raise ValueError("y is required for RandomForestRegressor.fit.")

        X_checked, y_checked = check_X_y(X, y)
        self.n_features_in_ = X_checked.shape[1]
        self.estimators_ = []

        rng = self._get_rng()
        for _ in range(self.n_estimators):
            feature_indices = self._sample_feature_indices(rng, self.n_features_in_)
            sample_indices = self._sample_row_indices(rng, X_checked.shape[0])

            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            estimator.fit(
                X_checked[sample_indices][:, feature_indices],
                y_checked[sample_indices],
            )
            self.estimators_.append(
                _ForestMember(
                    estimator=estimator,
                    feature_indices=np.asarray(feature_indices, dtype=int),
                )
            )

        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        predictions = [
            member.estimator.predict(X_checked[:, member.feature_indices])
            for member in self.estimators_
        ]
        return np.mean(predictions, axis=0)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "estimators_") or not hasattr(self, "n_features_in_"):
            raise ValueError("This RandomForestRegressor instance is not fitted yet.")
