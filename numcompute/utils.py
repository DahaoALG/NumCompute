"""Shared helpers. Each member owns the functions in their own block."""
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "iter_line_batches",
    "check_array",
    "iter_batches",
    "logsumexp",
    "softmax",
    "sigmoid",
    "pairwise_distances",
]


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


# ---- Jose: numerically stable activations and distances ----

def _safe_max(x: np.ndarray, axis) -> np.ndarray:
    """Per-slice maximum used as a shift, replaced by 0 where it is not finite.

    Parameters
    ----------
    x : ndarray of float
        Input values.
    axis : int or None
        Axis of the maximum; the result keeps it with length 1.

    Returns
    -------
    ndarray of float
        Shift that broadcasts against x. A slice that is all -inf, contains
        +inf or contains NaN gets 0, so subtracting it never creates inf - inf.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N / size of the reduced axis) for the result.
    """
    shift = np.max(x, axis=axis, keepdims=True)
    return np.where(np.isfinite(shift), shift, 0.0)


def logsumexp(x: ArrayLike, axis: int = None):
    """Compute log(sum(exp(x))) without overflow or underflow.

    Uses the max-shift identity log(sum(exp(x))) = m + log(sum(exp(x - m))),
    with m = max(x), so the largest term is exp(0) = 1.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Values converted to float, e.g. log-probabilities or logits.
    axis : int or None, default None
        Axis to reduce; None reduces all values.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the input shape with the
        axis removed. A slice of only -inf gives -inf, +inf gives +inf and
        NaN propagates.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for the shifted temporary array.

    Examples
    --------
    >>> logsumexp([1000.0, 1000.0])          # the naive formula overflows to inf
    1000.6931471805599
    """
    x = check_array(x)
    shift = _safe_max(x, axis)
    with np.errstate(divide="ignore"):         # log(0) = -inf is the right answer
        out = np.log(np.sum(np.exp(x - shift), axis=axis, keepdims=True)) + shift

    if axis is None:
        return float(out.reshape(()))
    return np.squeeze(out, axis=axis)


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert scores to probabilities that sum to 1 along an axis.

    Subtracts the maximum of each slice before exponentiating (max-shifted
    softmax), so large scores do not overflow. The result does not change
    when a constant is added to every score in a slice.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Scores (logits) converted to float; for (n, k) input each row is one
        sample's k class scores.
    axis : int, default -1
        Axis along which the probabilities sum to 1 (the last axis by default).

    Returns
    -------
    ndarray of float, same shape as x
        Values in [0, 1]. A slice containing NaN or +inf, or made only of
        -inf, gives NaN.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for the exponentials and the output.

    Examples
    --------
    >>> softmax([1000.0, 1000.0])            # naive exp(1000) overflows
    array([0.5, 0.5])
    """
    x = check_array(x)
    exp = np.exp(x - _safe_max(x, axis))
    with np.errstate(invalid="ignore"):       # 0/0 and inf/inf give NaN, as documented
        return exp / np.sum(exp, axis=axis, keepdims=True)


def sigmoid(x: ArrayLike):
    """Compute the logistic function 1 / (1 + exp(-x)) without overflow.

    Uses z = exp(-|x|), which never exceeds 1: x >= 0 gives 1 / (1 + z) and
    x < 0 gives z / (1 + z), the same value written in a safe form.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Values converted to float.

    Returns
    -------
    float or ndarray of float
        A Python float for scalar input; otherwise the same shape as x.
        Values are in [0, 1]; NaN propagates.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for temporaries and the output.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> float(sigmoid([-1000.0])[0])          # naive exp(1000) overflows
    0.0
    """
    x = check_array(x)
    z = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    if out.ndim == 0:
        return float(out)
    return out


def pairwise_distances(X: ArrayLike, Y: ArrayLike = None) -> np.ndarray:
    """Euclidean distance between every row of X and every row of Y.

    Expands ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y, so the work is one
    matrix product instead of an (n, m, d) array of differences. Both inputs
    are first centred on X's column means (distances do not change under a
    shared shift), which limits cancellation for data far from the origin;
    rounding below zero is clipped before the square root.

    Parameters
    ----------
    X : numeric array_like, shape (n, d)
        First set of points, one per row.
    Y : numeric array_like, shape (m, d), or None, default None
        Second set of points; None compares X with itself.

    Returns
    -------
    ndarray of float, shape (n, m)
        Entry [i, j] is the distance between X[i] and Y[j]. With Y=None the
        result is (n, n), symmetric, with an exact zero diagonal.

    Raises
    ------
    TypeError
        If X or Y cannot be converted to numeric data.
    ValueError
        If X or Y is empty or not 2-D, or they have different feature counts.

    Notes
    -----
    Time Complexity: O(n * m * d).
    Space Complexity: O(n * m + (n + m) * d).

    Examples
    --------
    >>> pairwise_distances([[0.0, 0.0], [3.0, 4.0]])
    array([[0., 5.],
           [5., 0.]])
    """
    X = check_array(X, ndim=2)
    same = Y is None
    Y = X if same else check_array(Y, ndim=2)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y need the same number of features, got {X.shape[1]} and {Y.shape[1]}")

    centre = X.mean(axis=0)
    Xc, Yc = X - centre, Y - centre
    sq = (Xc ** 2).sum(axis=1)[:, None] + (Yc ** 2).sum(axis=1)[None, :] - 2.0 * (Xc @ Yc.T)
    np.maximum(sq, 0.0, out=sq)               # rounding can leave tiny negatives
    if same:
        np.fill_diagonal(sq, 0.0)
    return np.sqrt(sq)


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
