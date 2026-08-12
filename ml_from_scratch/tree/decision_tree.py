"""Decision tree implementations."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ml_from_scratch.base import BaseEstimator, ClassifierMixin, RegressorMixin
from ml_from_scratch.utils.validation import check_X_y, check_array


@dataclass
class _TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    prediction: Optional[float] = None
    class_counts: Optional[np.ndarray] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _BaseDecisionTree(BaseEstimator):
    """Shared tree-building logic for classifier and regressor variants."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        criterion: str = "gini",
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion

    def fit(self, X: Any, y: Any = None) -> "_BaseDecisionTree":
        if y is None:
            raise ValueError("y is required for DecisionTree.fit.")

        X_checked, y_checked = check_X_y(X, y)
        self.n_features_in_ = X_checked.shape[1]
        self.tree_ = self._grow_tree(X_checked, y_checked, depth=0)
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        return np.asarray([self._predict_row(row, self.tree_) for row in X_checked])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        if self._should_stop(y, depth):
            return self._make_leaf(y)

        split = self._best_split(X, y)
        if split is None:
            return self._make_leaf(y)

        feature_index, threshold, left_mask, right_mask, gain = split
        if gain < self.min_impurity_decrease:
            return self._make_leaf(y)

        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return _TreeNode(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Optional[tuple[int, float, np.ndarray, np.ndarray, float]]:
        current_impurity = self._impurity(y)
        best_gain = 0.0
        best_split: Optional[tuple[int, float, np.ndarray, np.ndarray, float]] = None
        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            if unique_values.shape[0] < 2:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                left_impurity = self._impurity(left_y)
                right_impurity = self._impurity(right_y)
                weighted_impurity = (
                    (left_y.shape[0] / n_samples) * left_impurity
                    + (right_y.shape[0] / n_samples) * right_impurity
                )
                gain = current_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_split = (
                        feature_index,
                        float(threshold),
                        left_mask,
                        right_mask,
                        gain,
                    )

        return best_split

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if self.max_depth is not None and depth >= self.max_depth:
            return True

        if y.shape[0] < self.min_samples_split:
            return True

        if np.unique(y).shape[0] == 1:
            return True

        return False

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        current = node
        while not current.is_leaf:
            assert current.feature_index is not None
            assert current.threshold is not None
            assert current.left is not None
            assert current.right is not None

            if row[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right

        assert current.prediction is not None
        return current.prediction

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "tree_") or not hasattr(self, "n_features_in_"):
            raise ValueError("This DecisionTree instance is not fitted yet.")

    def _make_leaf(self, y: np.ndarray) -> _TreeNode:
        raise NotImplementedError

    def _impurity(self, y: np.ndarray) -> float:
        raise NotImplementedError


class DecisionTreeClassifier(ClassifierMixin, _BaseDecisionTree):
    """Decision tree classifier using greedy impurity minimization."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        criterion: str = "gini",
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
        )

    def fit(self, X: Any, y: Any = None) -> "DecisionTreeClassifier":
        if y is None:
            raise ValueError("y is required for DecisionTreeClassifier.fit.")

        if self.criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be either 'gini' or 'entropy'.")

        X_checked, y_checked = check_X_y(X, y)
        self.classes_ = np.unique(y_checked)
        self.n_features_in_ = X_checked.shape[1]
        self.tree_ = self._grow_tree(X_checked, y_checked, depth=0)
        return self

    def predict(self, X: Any) -> np.ndarray:
        encoded = super().predict(X)
        return encoded.astype(self.classes_.dtype, copy=False)

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_checked = check_array(X)

        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data."
            )

        probabilities = [
            self._predict_proba_row(row, self.tree_) for row in X_checked
        ]
        return np.vstack(probabilities)

    def _predict_proba_row(self, row: np.ndarray, node: _TreeNode) -> np.ndarray:
        current = node
        while not current.is_leaf:
            assert current.feature_index is not None
            assert current.threshold is not None
            assert current.left is not None
            assert current.right is not None

            if row[current.feature_index] <= current.threshold:
                current = current.left
            else:
                current = current.right

        if current.class_counts is None:
            raise ValueError("Leaf node is missing class counts.")

        total = current.class_counts.sum()
        if total == 0:
            return np.full(self.classes_.shape[0], 1.0 / self.classes_.shape[0])

        return current.class_counts / total

    def _make_leaf(self, y: np.ndarray) -> _TreeNode:
        counts = np.array([np.sum(y == class_label) for class_label in self.classes_])
        prediction = float(self.classes_[np.argmax(counts)])
        return _TreeNode(prediction=prediction, class_counts=counts.astype(float))

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "gini":
            probabilities = np.array(
                [np.mean(y == class_label) for class_label in self.classes_]
            )
            return float(1.0 - np.sum(probabilities**2))

        if self.criterion == "entropy":
            probabilities = np.array(
                [np.mean(y == class_label) for class_label in self.classes_]
            )
            probabilities = probabilities[probabilities > 0]
            return float(-np.sum(probabilities * np.log2(probabilities)))

        raise ValueError("criterion must be either 'gini' or 'entropy'.")

    def _check_is_fitted(self) -> None:
        if (
            not hasattr(self, "tree_")
            or not hasattr(self, "n_features_in_")
            or not hasattr(self, "classes_")
        ):
            raise ValueError("This DecisionTreeClassifier instance is not fitted yet.")


class DecisionTreeRegressor(RegressorMixin, _BaseDecisionTree):
    """Decision tree regressor using greedy variance reduction."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            criterion="squared_error",
        )

    def _make_leaf(self, y: np.ndarray) -> _TreeNode:
        return _TreeNode(prediction=float(np.mean(y)))

    def _impurity(self, y: np.ndarray) -> float:
        return float(np.mean((y - np.mean(y)) ** 2))
