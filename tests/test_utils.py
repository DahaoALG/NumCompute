from itertools import count

import numpy as np
import pytest

from numcompute.utils import check_array, iter_batches, iter_line_batches


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


# ---- Jose: iterating line batches ----

def test_iter_line_batches_sizes_and_empty_input():
    """Full batches first, a shorter last one, nothing for an empty input."""
    assert list(iter_line_batches(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(iter_line_batches([], 3)) == []


def test_iter_line_batches_is_lazy_on_infinite_iterators():
    """Only one batch is read at a time, so an endless source still works."""
    batches = iter_line_batches(count(), 3)

    assert next(batches) == [0, 1, 2]
    assert next(batches) == [3, 4, 5]  # continues where the last batch stopped


@pytest.mark.parametrize("batch_size, error", [(0, ValueError), (-2, ValueError), (2.5, TypeError), ("3", TypeError)])
def test_iter_line_batches_rejects_bad_batch_size(batch_size, error):
    """batch_size must be an integer >= 1."""
    with pytest.raises(error, match="batch_size"):
        list(iter_line_batches(range(5), batch_size))
