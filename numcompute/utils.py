"""Shared helpers. Each member owns the functions in their own block."""
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["iter_line_batches", "check_array", "iter_batches"]


def _as_numeric_array(values: ArrayLike) -> np.ndarray:
    """Return a validated one-dimensional real numeric array."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("values must contain real numeric values")
    return array

# ---- Jose: iterating line batches ----
def iter_line_batches(items: Iterable, batch_size: int) -> Iterator[list]:
    """Yield consecutive batches of at most ``batch_size`` items.

    Works on any iterable, including ones that can only be read once
    (open files, generators), without loading the whole input into memory.

    Parameters
    ----------
    items : iterable
        Source of items, consumed lazily.
    batch_size : int
        Maximum number of items per batch. Must be a positive integer.

    Returns
    -------
    Iterator[list]
        Lists of length ``batch_size``; the last one may be shorter.
        Nothing is yielded for an empty input.

    Raises
    ------
    TypeError
        If ``batch_size`` is not an integer.
    ValueError
        If ``batch_size < 1``.

    Notes
    -----
    Time Complexity: O(n: number of items), over all batches.
    Space Complexity: O(b: batch_size), only one batch is held at a time.

    Examples
    --------
    >>> list(iter_line_batches(range(5), 2))
    [[0, 1], [2, 3], [4]]
    """
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be greater or equal than 1, got {batch_size}")

    it = iter(items)                          # one iterator: each islice continues from the last
    batch = list(islice(it, batch_size))
    while batch:
        yield batch
        batch = list(islice(it, batch_size))


# ---- Dat: validation and batching ----

def check_array(X, ndim: int = None) -> np.ndarray:
    """Convert X to a float64 array and check it is non-empty with the expected ndim.

    Args: X array_like of numbers; ndim required dimensions, None accepts any.
    Returns: float64 ndarray, same shape as X.
    Raises: TypeError if X is not numeric; ValueError if X is empty or ndim is wrong.
    Complexity: O(n) time; copies only when the dtype changes.
    """
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("X must be numeric")

    if X.size == 0:
        raise ValueError("X is empty")
    if ndim is not None and X.ndim != ndim:
        raise ValueError(f"X must be {ndim}-D, got shape {X.shape}; use X.reshape(-1, 1) for one feature")
    return X


def iter_batches(X: np.ndarray, batch_size: int):
    """Yield X in consecutive row batches (views, no copies).

    Args: X array of shape (n, ...); batch_size rows per batch, the last may be shorter.
    Returns: generator of arrays with at most batch_size rows.
    Raises: ValueError if batch_size is not positive.
    Complexity: O(n / batch_size) iterations, O(1) extra space.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X = np.asarray(X)
    for start in range(0, X.shape[0], batch_size):
        yield X[start:start + batch_size]


if __name__ == "__main__":
    assert check_array([[1, 2]]).dtype == float

    sizes = [batch.shape[0] for batch in iter_batches(np.zeros((10, 1)), 4)]
    assert sizes == [4, 4, 2]

    try:
        check_array([1.0], ndim=2)
        assert False, "1-D input should be rejected"
    except ValueError:
        pass

    print("utils ok")
