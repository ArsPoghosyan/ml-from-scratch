"""Fit and plot a simple random forest example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_from_scratch.tree import RandomForestClassifier


def _make_demo_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)

    class_zero = rng.normal(loc=(-2.0, -1.0), scale=0.8, size=(60, 2))
    class_one = rng.normal(loc=(2.0, 2.0), scale=0.9, size=(60, 2))

    X = np.vstack((class_zero, class_one))
    y = np.concatenate((np.zeros(class_zero.shape[0]), np.ones(class_one.shape[0])))
    return X, y


def main() -> None:
    X, y = _make_demo_data()

    model = RandomForestClassifier(
        n_estimators=25,
        max_depth=4,
        bootstrap=True,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    accuracy = float(np.mean(predictions == y))

    print(f"Training accuracy: {accuracy:.3f}")
    print(f"Mean predicted probability for class 1: {probabilities.mean():.3f}")

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_probabilities = model.predict_proba(grid_points)[:, 1].reshape(grid_x.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x, grid_y, grid_probabilities, levels=25, cmap="RdBu", alpha=0.4)
    plt.contour(grid_x, grid_y, grid_probabilities, levels=[0.5], colors="black")
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        label="Class 0",
        edgecolors="white",
        alpha=0.9,
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Class 1",
        edgecolors="white",
        alpha=0.9,
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Random Forest From Scratch")
    plt.legend()
    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
