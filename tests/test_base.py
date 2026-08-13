import pytest
from ml_from_scratch.base import BaseEstimator, ClassifierMixin, RegressorMixin


def test_base_estimator_requires_fit_and_predict():
    with pytest.raises(TypeError):
        BaseEstimator()


def test_get_params_returns_public_non_fitted_attributes():
    class DummyEstimator(BaseEstimator):
        def __init__(self, learning_rate=0.1):
            self.learning_rate = learning_rate
            self.coef_ = None
            self._cache = {}

        def fit(self, X, y=None):
            return self

        def predict(self, X):
            return X

    estimator = DummyEstimator(learning_rate=0.5)

    assert estimator.get_params() == {"learning_rate": 0.5}


def test_estimator_mixins_define_estimator_type():
    assert RegressorMixin.estimator_type == "regressor"
    assert ClassifierMixin.estimator_type == "classifier"
