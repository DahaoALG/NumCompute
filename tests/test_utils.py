import numpy as np
import pytest

from numcompute.utils import check_array, iter_batches


def test_check_array_converts_to_float64_without_copying_float64():
    assert check_array([[1, 2], [3, 4]]).dtype == np.float64
    X = np.ones((3, 2))
    assert check_array(X) is X


def test_check_array_errors():
    with pytest.raises(ValueError, match="2-D"):
        check_array([1.0, 2.0], ndim=2)
    with pytest.raises(ValueError, match="empty"):
        check_array(np.zeros((0, 3)))
    with pytest.raises(TypeError):
        check_array([["a"]])


def test_iter_batches_covers_all_rows_as_views():
    X = np.arange(10).reshape(10, 1)
    batches = list(iter_batches(X, 4))
    assert [b.shape[0] for b in batches] == [4, 4, 2]
    assert np.shares_memory(batches[0], X)
    np.testing.assert_array_equal(np.vstack(batches), X)
    with pytest.raises(ValueError):
        list(iter_batches(X, 0))
