import numpy as np
import pytest
from ml_from_scratch.utils.validation import check_array, check_X_y


def test_check_array_converts_1d_input_to_2d_float_array():
    X = check_array([1, 2, 3])

    assert X.shape == (3, 1)
    assert X.dtype == float


def test_check_array_rejects_more_than_two_dimensions():
    with pytest.raises(ValueError, match="1D or 2D"):
        check_array(np.zeros((2, 2, 2)))


def test_check_X_y_returns_checked_arrays():
    X, y = check_X_y([[1], [2], [3]], [2, 4, 6])

    assert X.shape == (3, 1)
    assert y.shape == (3,)


def test_check_X_y_rejects_mismatched_sample_counts():
    with pytest.raises(ValueError, match="same number of samples"):
        check_X_y([[1], [2]], [1])
