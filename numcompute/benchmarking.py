"""Benchmark: time a Python-loop version against a NumPy version.

Each comparison first checks that both versions give the same result, then
times them on the same inputs. The functions take no arguments: bind the
inputs with a lambda, e.g. ``compare("mse", lambda: mse_loop(a, b), lambda: mse(a, b))``.
"""

import os
import platform
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# All that can be imported from this module
__all__ = [
    "time_function",
    "compare",
    "environment_info",
    "format_table",
]


def time_function(
    func: Callable[[], Any],
    repeat: int = 7,
    number: int = 1,
    warmup: int = 1,
) -> np.ndarray:
    """Time a function that takes no arguments.

    Parameters
    ----------
    func : callable
        Function to time, called as ``func()``. Bind its inputs with a lambda.
    repeat : int, default 7
        Number of measurements. Must be >= 1.
    number : int, default 1
        Calls per measurement. Must be >= 1. Raise it for functions so fast that
        one call is close to the clock's resolution.
    warmup : int, default 1
        Untimed calls made first (first-call costs such as memory allocation).
        Must be >= 0.

    Returns
    -------
    numpy.ndarray, shape (repeat,)
        Seconds per call for each measurement.

    Raises
    ------
    TypeError
        If ``func`` is not callable.
    ValueError
        If ``repeat < 1``, ``number < 1`` or ``warmup < 0``.

    Notes
    -----
    Time Complexity: O(warmup + repeat * number) calls of ``func``.
    Space Complexity: O(repeat).
    The loops here are the timing itself, not a computation to vectorise.

    Examples
    --------
    >>> times = time_function(lambda: sum(range(100)), repeat=3)
    >>> times.shape
    (3,)
    """
    # Check everything that could go wrong first
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func).__name__}")
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    if number < 1:
        raise ValueError(f"number must be >= 1, got {number}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    # Warmup runs
    for _ in range(warmup):
        func()

    # Time the function per se and store the values
    times = np.empty(repeat)
    for r in range(repeat):
        start = time.perf_counter()
        for _ in range(number):
            func()
        times[r] = (time.perf_counter() - start) / number   # seconds per call
    
    return times


def compare(
    case: str,
    loop_func: Callable[[], Any],
    vec_func: Callable[[], Any],
    repeat: int = 7,
    number: int = 1,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, str | float]:
    """Check that a loop version and a vectorised version agree, then time both.

    Parameters
    ----------
    case : str
        Description of what is compared, e.g. ``"metrics.mse n=100,000"``.
    loop_func : callable
        Python-loop version, called as ``loop_func()``.
    vec_func : callable
        Vectorised version, called as ``vec_func()``.
    repeat : int, default 7
        Number of measurements. Must be >= 1.
    number : int, default 1
        Calls per measurement. Must be >= 1. Raise it for functions so fast that
        one call is close to the clock's resolution.
    rtol, atol : float
        Tolerances for the result check (numeric results only; text must match exactly).

    Returns
    -------
    dict
        ``case``; ``loop_s`` and ``vec_s``, the median seconds per call;
        ``speedup`` = ``loop_s / vec_s`` (> 1 means the vectorised version is faster).

    Raises
    ------
    TypeError
        If a function is not callable.
    ValueError
        If the two results differ, or ``repeat`` / ``number`` is invalid.

    Notes
    -----
    Time Complexity: O(repeat * number + 1) calls of each function.
    Space Complexity: O(repeat) plus the size of the two results.
    The median is used because one slow measurement (OS noise) does not move it.

    Examples
    --------
    >>> row = compare("sum", lambda: sum(range(1000)), lambda: np.arange(1000).sum())
    >>> sorted(row)
    ['case', 'loop_s', 'speedup', 'vec_s']
    """
    if not callable(loop_func) or not callable(vec_func):
        raise TypeError("loop_func and vec_func must be callable")

    # Check first, so a fast but wrong version never reaches the table
    if not _same_result(loop_func(), vec_func(), rtol, atol):
        raise ValueError(f"{case}: the loop and vectorised results differ")

    # The check calls above already warmed both functions up
    loop_s = float(np.median(time_function(loop_func, repeat, number, warmup=0)))
    vec_s = float(np.median(time_function(vec_func, repeat, number, warmup=0)))
    
    speedup = loop_s / vec_s if vec_s > 0 else float("inf")
    
    return {"case": case, "loop_s": loop_s, "vec_s": vec_s, "speedup": speedup}


def environment_info() -> dict[str, str]:
    """Describe the machine and software, so timings can be reproduced and compared.

    Returns
    -------
    dict[str, str]
        Keys ``python``, ``numpy``, ``os``, ``cpu`` and ``cpu_count``.

    Notes
    -----
    Time Complexity: O(1).
    Space Complexity: O(1).

    Examples
    --------
    >>> sorted(environment_info())
    ['cpu', 'cpu_count', 'numpy', 'os', 'python']
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cpu_count": str(os.cpu_count()),
    }


def format_table(rows: Sequence[dict[str, str | float]]) -> str:
    """Format :func:`compare` results as a Markdown table, one row per case.

    Parameters
    ----------
    rows : sequence of dict
        Results of :func:`compare`. An optional ``"notes"`` entry fills the
        Notes column (empty if missing).

    Returns
    -------
    str
        Columns Operation; NumCompute result (median ms per call);
        Baseline/result (the loop's median ms per call, then the speed-up
        ``loop / NumCompute`` in parentheses); Notes.

    Notes
    -----
    Time Complexity: O(k: number of rows).
    Space Complexity: O(k).

    Examples
    --------
    >>> row = {"case": "mse", "loop_s": 0.01, "vec_s": 0.0002, "speedup": 50.0}
    >>> print(format_table([row]))
    | Operation | NumCompute result | Baseline/result | Notes |
    |---|---:|---:|---|
    | mse | 0.200 ms | 10.000 ms (50.0x) |  |
    """
    lines = ["| Operation | NumCompute result | Baseline/result | Notes |",
             "|---|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['vec_s'] * 1e3:,.3f} ms "
                     f"| {row['loop_s'] * 1e3:,.3f} ms ({row['speedup']:,.1f}x) "
                     f"| {row.get('notes', '')} |")
    return "\n".join(lines)


def _same_result(a: Any, b: Any, rtol: float, atol: float) -> bool:
    """True if two results have the same shape and values (NaN equals NaN)."""
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.shape != b.shape:
        return False
    
    # Numbers may differ by rounding (loop and NumPy add in a different order)
    if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
        return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    
    # Text must match exactly
    return bool(np.array_equal(a, b))
