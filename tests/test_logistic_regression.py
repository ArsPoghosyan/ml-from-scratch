import numpy as np
import pytest

from ml_from_scratch.linear_model import LogisticRegression


def test_logistic_regression_fits_simple_binary_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = LogisticRegression(learning_rate=0.5, max_iter=5000)
    fitted = model.fit(X, y)

    assert fitted is model
    np.testing.assert_array_equal(model.predict(X), y)


def test_logistic_regression_predict_proba_returns_two_columns():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = LogisticRegression(learning_rate=0.5, max_iter=5000).fit(X, y)
    probabilities = model.predict_proba([[1.5], [2.5]])

    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))


def test_logistic_regression_rejects_multiclass_targets():
    model = LogisticRegression()

    with pytest.raises(ValueError, match="exactly two classes"):
        model.fit([[0], [1], [2]], [0, 1, 2])
