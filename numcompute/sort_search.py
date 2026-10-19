"""Sorts, argpartition/top-k, quickselect, and binary search.

Task allocation: Faris.
"""

"""Sorting, selection, top-k, and binary-search algorithms.

All functions accept one-dimensional real numeric sequences or NumPy arrays.
Input arrays are never modified. NaNs, when present, are ordered after every
non-NaN value.
"""

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

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
    """Return a copy of *values* sorted in ascending order using quicksort.

    Ties are not guaranteed to retain their original order. NaNs are placed at
    the end. Average time complexity is O(n log n); worst-case time is O(n²).
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


def merge_sort(values: ArrayLike) -> np.ndarray:
    """Return a stably sorted copy of *values* using bottom-up merge sort.

    NaNs are placed at the end. Time complexity is O(n log n); auxiliary
    storage is O(n).
    """
    array = _as_numeric_array(values)
    source = array.tolist()
    size = len(source)
    if size < 2:
        return array.copy()

    width = 1
    while width < size:
        merged: list[object] = []
        for start in range(0, size, 2 * width):
            middle = min(start + width, size)
            end = min(start + 2 * width, size)
            left = start
            right = middle
            while left < middle and right < end:
                if _less(source[right], source[left]):
                    merged.append(source[right])
                    right += 1
                else:
                    merged.append(source[left])
                    left += 1
            merged.extend(source[left:middle])
            merged.extend(source[right:end])
        source = merged
        width *= 2

    return np.asarray(source, dtype=array.dtype)


def argpartition(values: ArrayLike, kth: int) -> np.ndarray:
    """Return indices partitioned around the zero-based sorted rank ``kth``.

    The value at ``result[kth]`` is the same as the value at that rank in a
    fully sorted array. Every value before it is less than or equal to it, and
    every value after it is greater than or equal to it. The order within
    either partition is unspecified. NaNs are ordered last.

    Average time complexity is O(n); worst-case time is O(n²).
    """
    array = _as_numeric_array(values)
    index = _check_kth(kth, array.size)
    indices = list(range(array.size))
    _select_indices(array, indices, index)
    return np.asarray(indices, dtype=np.intp)


def top_k(values: ArrayLike, k: int, largest: bool = True) -> np.ndarray:
    """Return the ``k`` smallest or largest values in sorted order.

    The result is descending when ``largest`` is true and ascending otherwise.
    NaNs are placed after all non-NaN results in either direction. Ties have
    no guaranteed order. ``k`` may be any integer from zero through the input
    length. Average time complexity is O(n + k log k).
    """
    array = _as_numeric_array(values)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise TypeError("k must be an integer")
    count = int(k)
    if count < 0 or count > array.size:
        raise ValueError(f"k must be in the range [0, {array.size}]")
    if count == 0:
        return array[:0].copy()

    if np.issubdtype(array.dtype, np.floating):
        nan_mask = np.isnan(array)
        nan_indices = np.flatnonzero(nan_mask).tolist()
        non_nan_indices = np.flatnonzero(~nan_mask).tolist()
    else:
        nan_indices = []
        non_nan_indices = list(range(array.size))

    take = min(count, len(non_nan_indices))
    if take == 0:
        return array[nan_indices[:count]].copy()

    if largest:
        rank = len(non_nan_indices) - take
        candidates = non_nan_indices.copy()
        _select_indices(array, candidates, rank)
        chosen = candidates[rank:]
    else:
        candidates = non_nan_indices.copy()
        _select_indices(array, candidates, take - 1)
        chosen = candidates[:take]

    selected = merge_sort(array[chosen])
    if largest:
        selected = selected[::-1]
    if take < count:
        selected = np.concatenate((selected, array[nan_indices[: count - take]]))
    return selected


def quickselect(values: ArrayLike, kth: int) -> np.generic:
    """Return the value at zero-based sorted rank ``kth`` without full sorting.

    NaNs are ordered after every non-NaN value. Average time complexity is
    O(n); worst-case time is O(n²).
    """
    array = _as_numeric_array(values)
    index = _check_kth(kth, array.size)
    indices = list(range(array.size))
    _select_indices(array, indices, index)
    return array[indices[index]]


def binary_search(sorted_values: ArrayLike, target: float) -> int:
    """Return the first index of ``target`` in ascending ``sorted_values``.

    Return ``-1`` if the target is absent. The input must already be sorted
    according to the module's ordering (with NaNs at the end); it is not
    modified or checked for sortedness. If the target is NaN, the first NaN
    index is returned. Time complexity is O(log n).
    """
    array = _as_numeric_array(sorted_values)
    target_array = np.asarray(target)
    if target_array.ndim != 0 or not np.issubdtype(
        target_array.dtype, np.number
    ) or np.issubdtype(target_array.dtype, np.complexfloating):
        raise TypeError("target must be a real numeric scalar")
    value = target_array[()]

    low = 0
    high = array.size
    while low < high:
        middle = (low + high) // 2
        if _less(array[middle], value):
            low = middle + 1
        else:
            high = middle
    if low < array.size and not _less(value, array[low]) and not _less(
        array[low], value
    ):
        return low
    return -1
