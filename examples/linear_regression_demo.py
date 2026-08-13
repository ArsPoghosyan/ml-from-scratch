"""Fit and plot a simple linear regression example."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_from_scratch.linear_model import LinearRegression
from ml_from_scratch.utils import mean_absolute_error, mean_squared_error, r2_score


def main() -> None:
    rng = np.random.default_rng(seed=42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3.0 * X.ravel() + 2.0 + rng.normal(0, 2.0, size=X.shape[0])

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    print(f"MSE: {mean_squared_error(y, predictions):.3f}")
    print(f"MAE: {mean_absolute_error(y, predictions):.3f}")
    print(f"R2: {r2_score(y, predictions):.3f}")

    plt.scatter(X.ravel(), y, label="Training data", alpha=0.75)
    plt.plot(X.ravel(), predictions, color="black", label="Prediction")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression From Scratch")
    plt.legend()
    plt.tight_layout()

    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
