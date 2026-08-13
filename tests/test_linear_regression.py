import numpy as np
import pytest
from ml_from_scratch.linear_model import LinearRegression


def test_linear_regression_fits_simple_line_data():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression()
    fitted = model.fit(X, y)

    assert fitted is model


def test_linear_regression_prediction_shape():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression().fit(X, y)
    predictions = model.predict([[5], [6]])

    assert predictions.shape == (2,)


def test_linear_regression_learns_coefficient_and_intercept():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression().fit(X, y)

    assert model.coef_ == pytest.approx(np.array([2.0]))
    assert model.intercept_ == pytest.approx(1.0)


def test_linear_regression_predicts_expected_values():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression().fit(X, y)

    np.testing.assert_allclose(model.predict([[5], [6]]), np.array([11, 13]))
