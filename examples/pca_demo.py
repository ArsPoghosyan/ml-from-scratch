"""Fit and plot a simple PCA example."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_from_scratch.decomposition import PCA


def _make_demo_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)

    latent = rng.normal(size=(200, 2))
    transform = np.array([[3.0, 1.8], [1.2, 0.7]])
    noise = rng.normal(scale=0.25, size=(200, 2))
    return latent @ transform + noise


def main() -> None:
    X = _make_demo_data()

    model = PCA(n_components=2)
    X_transformed = model.fit_transform(X)
    X_reconstructed = model.inverse_transform(X_transformed)
    reconstruction_error = float(np.mean((X - X_reconstructed) ** 2))

    print(f"Explained variance ratio: {model.explained_variance_ratio_}")
    print(f"Reconstruction MSE: {reconstruction_error:.4f}")

    center = model.mean_
    principal_direction = model.components_[0]
    direction_length = 3.0 * np.sqrt(model.explained_variance_[0])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=20, alpha=0.75, label="Original data")
    plt.plot(
        [center[0] - direction_length * principal_direction[0], center[0] + direction_length * principal_direction[0]],
        [center[1] - direction_length * principal_direction[1], center[1] + direction_length * principal_direction[1]],
        color="black",
        linewidth=2.0,
        label="First principal axis",
    )
    plt.axis("equal")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Space")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(
        X_transformed[:, 0],
        X_transformed[:, 1],
        s=20,
        alpha=0.75,
        label="PCA coordinates",
    )
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    plt.axis("equal")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Space")
    plt.legend()

    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
