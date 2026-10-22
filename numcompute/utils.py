"""Shared helpers. Each member owns the functions in their own block."""
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["iter_line_batches", "check_array", "iter_batches"]


def _as_numeric_array(values: ArrayLike) -> np.ndarray:
    """Validate a one-dimensional real numeric sequence without changing its dtype.

    Parameters
    ----------
    values : array_like, shape (n,)
        Real numeric values; empty input and NaNs are allowed.

    Returns
    -------
    ndarray, shape (n,)
        Input array or an array converted from the sequence.

    Raises
    ------
    TypeError
        If values are nonnumeric or complex.
    ValueError
        If values are not one-dimensional or cannot form a regular array.

    Notes
    -----
    Time/space: O(n) when conversion allocates, otherwise O(1).
    """
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
    """Convert input to a nonempty floating-point array and validate its dimensions.

    Parameters
    ----------
    X : array_like
        Values convertible to float; NaN and infinity are not rejected.
    ndim : int or None, default None
        Required number of dimensions; None allows any nonempty shape.

    Returns
    -------
    ndarray of float64, same shape as X
        May share storage with an existing float64 array.

    Raises
    ------
    TypeError
        If X cannot be converted to floating-point data.
    ValueError
        If X is empty or has a different number of dimensions.

    Notes
    -----
    Time/space: O(N) if conversion allocates, otherwise O(1), N=X.size.
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
    """Yield consecutive row slices, including a shorter final batch.

    Parameters
    ----------
    X : array_like, shape (n, ...)
        Input with a row axis; converted with np.asarray.
    batch_size : int
        Positive maximum row count per batch.

    Returns
    -------
    iterator of ndarray, shape (b, ...)
        Views of the converted array, with b <= batch_size. Empty input
        with a row axis yields no batches.

    Raises
    ------
    ValueError
        If batch_size is not positive.
    TypeError
        If batch_size is not usable as a range step.
    IndexError
        If X is scalar and has no row axis.

    Notes
    -----
    Time: O(ceil(n / batch_size)) iterations after conversion. Extra space
    is O(1) for array input; converting a sequence may allocate its full size.
    Validation occurs when the iterator is consumed.
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
