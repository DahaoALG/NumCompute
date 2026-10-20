"""Sorting, selection, top-k, and binary-search algorithms.

All functions accept one-dimensional real numeric sequences or NumPy arrays.
Input arrays are never modified. NaNs, when present, are ordered after every
non-NaN value.
"""

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from numcompute.utils import _as_numeric_array

__all__ = [
    "quicksort",
    "stable_sort",
    "lexsort",
    "argpartition",
    "topk",
    "top_k",
    "quickselect",
    "binary_search",
]


def _is_nan(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _less(left: object, right: object) -> bool:
    """Compare real numbers with NaNs consistently ordered last."""
    left_is_nan = _is_nan(left)
    right_is_nan = _is_nan(right)
    if left_is_nan:
        return False
    if right_is_nan:
        return True
    return bool(np.less(np.asarray(left), np.asarray(right)))


def _check_kth(kth: int, size: int) -> int:
    if isinstance(kth, (bool, np.bool_)) or not isinstance(kth, Integral):
        raise TypeError("kth must be an integer")
    index = int(kth)
    if index < 0 or index >= size:
        raise ValueError(f"kth must be in the range [0, {size - 1}]")
    return index


def _median_of_three(first: object, middle: object, last: object) -> object:
    if _less(first, middle):
        if _less(middle, last):
            return middle
        return last if _less(first, last) else first
    if _less(first, last):
        return first
    return last if _less(middle, last) else middle


def _partition_indices(
    values: np.ndarray, indices: list[int], low: int, high: int, pivot: object
) -> tuple[int, int]:
    """Three-way partition indices around pivot; return the equal interval."""
    less_end = low
    current = low
    greater_start = high
    while current <= greater_start:
        value = values[indices[current]]
        if _less(value, pivot):
            indices[less_end], indices[current] = indices[current], indices[less_end]
            less_end += 1
            current += 1
        elif _less(pivot, value):
            indices[current], indices[greater_start] = (
                indices[greater_start],
                indices[current],
            )
            greater_start -= 1
        else:
            current += 1
    return less_end, greater_start


def _select_indices(values: np.ndarray, indices: list[int], kth: int) -> None:
    """Rearrange indices in place so the value at sorted rank kth is in place."""
    low = 0
    high = len(indices) - 1
    while low <= high:
        middle = (low + high) // 2
        pivot = _median_of_three(
            values[indices[low]], values[indices[middle]], values[indices[high]]
        )
        equal_start, equal_end = _partition_indices(
            values, indices, low, high, pivot
        )
        if kth < equal_start:
            high = equal_start - 1
        elif kth > equal_end:
            low = equal_end + 1
        else:
            return


def quicksort(values: ArrayLike) -> np.ndarray:
    """Return a copy sorted in ascending order with the handwritten quicksort.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to sort.

    Returns
    -------
    ndarray of shape (n,)
        Sorted copy; ties need not retain input order and NaNs are last.

    Raises
    ------
    TypeError
        If values are not real numeric values.
    ValueError
        If values are not one-dimensional.

    Notes
    -----
    Average time is O(n log n), worst-case time is O(n²), and auxiliary space
    is O(log n), excluding the O(n) returned copy.
    """
    result = _as_numeric_array(values).copy()
    if result.size < 2:
        return result

    pending = [(0, result.size - 1)]
    while pending:
        low, high = pending.pop()
        while low < high:
            middle = (low + high) // 2
            pivot = _median_of_three(result[low], result[middle], result[high])
            less_end = low
            current = low
            greater_start = high
            while current <= greater_start:
                if _less(result[current], pivot):
                    result[less_end], result[current] = (
                        result[current],
                        result[less_end],
                    )
                    less_end += 1
                    current += 1
                elif _less(pivot, result[current]):
                    result[current], result[greater_start] = (
                        result[greater_start],
                        result[current],
                    )
                    greater_start -= 1
                else:
                    current += 1

            left = (low, less_end - 1)
            right = (greater_start + 1, high)
            if left[1] - left[0] < right[1] - right[0]:
                if right[0] < right[1]:
                    pending.append(right)
                low, high = left
            else:
                if left[0] < left[1]:
                    pending.append(left)
                low, high = right
    return result


def stable_sort(values: ArrayLike) -> np.ndarray:
    """Return a stable ascending sort of a one-dimensional numeric sequence.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to sort.

    Returns
    -------
    ndarray of shape (n,)
        Stably sorted copy with the input dtype preserved. NaNs are last.

    Raises
    ------
    TypeError
        If values are not real numeric values.
    ValueError
        If values are not one-dimensional.

    Notes
    -----
    Time is O(n log n) and auxiliary space is O(n), excluding the O(n)
    returned copy.
    """
    array = _as_numeric_array(values)
    return np.sort(array, kind="stable")


def lexsort(keys: Sequence[ArrayLike]) -> np.ndarray:
    """Return indices that sort rows by multiple keys.

    The last key is the primary key, following :func:`numpy.lexsort`.

    Parameters
    ----------
    keys : sequence of array-like of shape (n,)
        Non-empty sequence of real numeric sort keys, all with equal length.

    Returns
    -------
    ndarray of shape (n,), dtype intp
        Indices that order the input by the supplied keys. Input arrays are
        not modified.

    Raises
    ------
    TypeError
        If a key contains non-real-numeric values.
    ValueError
        If keys is empty, a key is not one-dimensional, or key lengths differ.

    Notes
    -----
    For p keys of length n, time is O(p n log n) and auxiliary space is O(n).
    """
    if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)) or not keys:
        raise ValueError("keys must be a non-empty sequence of arrays")
    arrays = [_as_numeric_array(key) for key in keys]
    size = arrays[0].size
    if any(key.size != size for key in arrays[1:]):
        raise ValueError("all sort keys must have the same length")
    return np.lexsort(tuple(arrays))


def argpartition(values: ArrayLike, kth: int) -> np.ndarray:
    """Return indices partitioned around the zero-based sorted rank ``kth``.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to partition.
    kth : int
        Zero-based sorted position to place at ``result[kth]``.

    Returns
    -------
    ndarray of shape (n,), dtype intp
        Indices such that values before ``kth`` are no greater than the
        selected value and values after it are no less. Within-partition order
        is unspecified; NaNs are ordered last.

    Raises
    ------
    TypeError
        If values are not real numeric values or ``kth`` is not an integer.
    ValueError
        If values are not one-dimensional or ``kth`` is out of range.

    Notes
    -----
    Time is O(n) and auxiliary space is O(n), excluding the returned indices.
    """
    array = _as_numeric_array(values)
    index = _check_kth(kth, array.size)
    return np.argpartition(array, index)


def top_k(
    values: ArrayLike,
    k: int,
    largest: bool = True,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` smallest or largest values in sorted order.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to select from.
    k : int
        Number of values to return, from zero through ``n``.
    largest : bool, default=True
        If true, return values descending; otherwise return them ascending.
    return_indices : bool, default=False
        If true, also return the selected indices in the same order as values.

    Returns
    -------
    ndarray of shape (k,) or (ndarray of shape (k,), ndarray of shape (k,))
        Selected values, and optionally their indices. NaNs follow all
        non-NaN values in either direction. Ties need not retain input order.

    Raises
    ------
    TypeError
        If values are not real numeric values, ``k`` is not an integer, or
        ``largest``/``return_indices`` are not booleans.
    ValueError
        If values are not one-dimensional or ``k`` is outside [0, n].

    Notes
    -----
    Average time is O(n + k log k) and auxiliary space is O(n), excluding
    returned values and optional indices.
    """
    array = _as_numeric_array(values)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be a boolean")
    if not isinstance(return_indices, (bool, np.bool_)):
        raise TypeError("return_indices must be a boolean")
    count = int(k)
    if count < 0 or count > array.size:
        raise ValueError(f"k must be in the range [0, {array.size}]")
    if count == 0:
        empty = array[:0].copy()
        if return_indices:
            return empty, np.empty(0, dtype=np.intp)
        return empty

    if np.issubdtype(array.dtype, np.inexact):
        non_nan_indices = np.flatnonzero(~np.isnan(array))
    else:
        non_nan_indices = np.arange(array.size)
    nan_indices = np.flatnonzero(np.isnan(array)) if np.issubdtype(
        array.dtype, np.inexact
    ) else np.empty(0, dtype=np.intp)

    take = min(count, non_nan_indices.size)
    if take == 0:
        selected_indices = nan_indices[:count]
        selected = array[selected_indices].copy()
        if return_indices:
            return selected, selected_indices.copy()
        return selected

    if largest:
        rank = non_nan_indices.size - take
        local_indices = np.argpartition(array[non_nan_indices], rank)[rank:]
    else:
        local_indices = np.argpartition(array[non_nan_indices], take - 1)[:take]
    chosen = non_nan_indices[local_indices]

    selected_order = np.argsort(array[chosen], kind="stable")
    if largest:
        selected_order = selected_order[::-1]
    selected_indices = chosen[selected_order]
    selected = array[selected_indices].copy()
    if take < count:
        selected_indices = np.concatenate(
            (selected_indices, nan_indices[: count - take])
        )
        selected = array[selected_indices].copy()
    if return_indices:
        return selected, selected_indices
    return selected


def topk(
    values: ArrayLike,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return the top ``k`` values, optionally with their input indices.

    This spec-facing spelling defaults to returning both values and indices.
    Use :func:`top_k` for the backward-compatible API, whose default returns
    values only.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to select from.
    k : int
        Number of values to return, from zero through ``n``.
    largest : bool, default=True
        If true, return values descending; otherwise return them ascending.
    return_indices : bool, default=True
        If true, also return selected indices in the same order as values.

    Returns
    -------
    ndarray of shape (k,) or (ndarray of shape (k,), ndarray of shape (k,))
        Selected values, and optionally their input indices.

    Raises
    ------
    TypeError
        If the input values, ``k``, or boolean options have invalid types.
    ValueError
        If values are not one-dimensional or ``k`` is outside [0, n].

    Notes
    -----
    Selection uses NumPy's ``argpartition`` and takes O(n + k log k) average
    time with O(n) auxiliary space.
    """
    return top_k(values, k, largest=largest, return_indices=return_indices)


def quickselect(values: ArrayLike, kth: int) -> np.generic:
    """Return the value at zero-based sorted rank ``kth`` without full sorting.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to select from.
    kth : int
        Zero-based position in the fully sorted sequence.

    Returns
    -------
    numpy scalar
        Value at sorted position ``kth``; NaNs follow all numeric values.

    Raises
    ------
    TypeError
        If values are not real numeric values or ``kth`` is not an integer.
    ValueError
        If values are not one-dimensional or ``kth`` is out of range.

    Notes
    -----
    Average time is O(n), worst-case time is O(n²), and auxiliary space is
    O(n) for the selection index array.
    """
    array = _as_numeric_array(values)
    index = _check_kth(kth, array.size)
    indices = list(range(array.size))
    _select_indices(array, indices, index)
    return array[indices[index]]


def binary_search(
    sorted_array: ArrayLike, x: float
) -> tuple[int, bool]:
    """Return the insertion index of ``x`` and whether it is present.

    Parameters
    ----------
    sorted_array : array-like of shape (n,)
        Values already sorted in ascending order, with NaNs at the end.
        Sortedness is not checked.
    x : real numeric scalar
        Value to locate.

    Returns
    -------
    tuple[int, bool]
        ``(insertion_index, found)``. The insertion index is the first
        position where ``x`` may be inserted without breaking sort order.
        For a present value (including NaN), it is the first matching index.

    Raises
    ------
    TypeError
        If the array or search value is not real numeric.
    ValueError
        If ``sorted_array`` is not one-dimensional.

    Notes
    -----
    Time is O(log n) and auxiliary space is O(1).
    """
    array = _as_numeric_array(sorted_array)
    target_array = np.asarray(x)
    if target_array.ndim != 0 or not np.issubdtype(
        target_array.dtype, np.number
    ) or np.issubdtype(target_array.dtype, np.complexfloating):
        raise TypeError("x must be a real numeric scalar")
    value = target_array[()]
    index = int(np.searchsorted(array, value, side="left"))
    if index == array.size:
        return index, False
    if np.issubdtype(array.dtype, np.inexact) and np.isnan(array[index]):
        target_is_nan = np.issubdtype(target_array.dtype, np.inexact) and np.isnan(
            value
        )
        return index, bool(target_is_nan)
    return index, bool(array[index] == value)
