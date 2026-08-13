"""Fit and plot a simple k-means example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_from_scratch.cluster import KMeans


def _make_demo_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)

    cluster_a = rng.normal(loc=(-2.0, -1.5), scale=0.55, size=(50, 2))
    cluster_b = rng.normal(loc=(2.0, 2.5), scale=0.6, size=(50, 2))
    cluster_c = rng.normal(loc=(4.5, -1.0), scale=0.5, size=(50, 2))

    X = np.vstack((cluster_a, cluster_b, cluster_c))
    y = np.concatenate(
        (
            np.zeros(cluster_a.shape[0], dtype=int),
            np.ones(cluster_b.shape[0], dtype=int),
            np.full(cluster_c.shape[0], 2, dtype=int),
        )
    )
    return X, y


def main() -> None:
    X, _ = _make_demo_data()

    model = KMeans(n_clusters=3, init="k-means++", n_init=8, random_state=42)
    labels = model.fit_predict(X)

    print(f"Inertia: {model.inertia_:.3f}")
    print(f"Iterations: {model.n_iter_}")
    print(f"Cluster centers:\n{model.cluster_centers_}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=28, alpha=0.85)
    plt.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        c="black",
        s=140,
        marker="x",
        linewidths=2.5,
        label="Centroids",
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means From Scratch")
    plt.legend()
    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
