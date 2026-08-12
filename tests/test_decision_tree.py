import numpy as np
import pytest

from ml_from_scratch.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_decision_tree_classifier_fits_simple_split():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = DecisionTreeClassifier(max_depth=1)
    fitted = model.fit(X, y)

    assert fitted is model
    np.testing.assert_array_equal(model.predict(X), y)


def test_decision_tree_classifier_predict_proba_sums_to_one():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    probabilities = model.predict_proba([[0.5], [2.5]])

    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))


def test_decision_tree_regressor_fits_piecewise_constant_data():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 1.0, 4.0, 4.0])

    model = DecisionTreeRegressor(max_depth=1)
    model.fit(X, y)

    np.testing.assert_allclose(model.predict(X), y)


def test_decision_tree_predict_rejects_feature_mismatch():
    model = DecisionTreeClassifier(max_depth=1).fit([[0.0], [1.0]], [0.0, 1.0])

    with pytest.raises(ValueError, match="different number of features"):
        model.predict([[0.0, 1.0]])
