import pytest
from ml_from_scratch.utils import mean_absolute_error, mean_squared_error, r2_score


def test_mean_squared_error_returns_known_value():
    assert mean_squared_error([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)


def test_mean_absolute_error_returns_known_value():
    assert mean_absolute_error([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)


def test_r2_score_returns_known_value():
    assert r2_score([1, 2, 3], [1, 2, 5]) == pytest.approx(-1.0)


def test_r2_score_returns_one_for_perfect_prediction():
    assert r2_score([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        mean_squared_error([1, 2, 3], [1, 2])
