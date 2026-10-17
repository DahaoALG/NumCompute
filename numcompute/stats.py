"""Descriptive statistics, quantiles, histogram and streaming (Welford) statistics.

axis follows NumPy: None reduces everything to one float, axis=0 on (n, d) gives (d,).
skip_nan=False lets NaN propagate like NumPy; skip_nan=True ignores NaN values.
"""

import numpy as np

from .utils import check_array


def mean(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Mean of x along axis.

    Args: x array of any shape; axis None reduces all elements.
    Returns: float when axis is None, else array with that axis removed.
    Raises: TypeError if x is not numeric; ValueError if x is empty.
    Complexity: O(n) time, O(1) extra space.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmean(x, axis=axis)
    else:
        result = np.mean(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def median(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Median of x along axis (average of the two middle values when n is even).

    Args: x array of any shape; axis None reduces all elements.
    Returns: float when axis is None, else array with that axis removed.
    Raises: TypeError if x is not numeric; ValueError if x is empty.
    Complexity: O(n) average time (np.partition), O(n) space.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmedian(x, axis=axis)
    else:
        result = np.median(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def std(x: np.ndarray, axis: int = None, ddof: int = 0, skip_nan: bool = False):
    """Standard deviation of x along axis.

    Args: x array of any shape; ddof 0 for population, 1 for the sample estimate.
    Returns: float when axis is None, else array with that axis removed.
    Raises: ValueError if fewer than ddof + 1 values are available (would divide by zero).
    Complexity: O(n) time, O(n) space for the centred copy.
    """
    x = check_array(x)
    if skip_nan:
        count = np.sum(~np.isnan(x), axis=axis)
    else:
        count = x.size if axis is None else x.shape[axis]
    if np.any(count <= ddof):
        raise ValueError(f"std needs more than ddof={ddof} values")

    if skip_nan:
        result = np.nanstd(x, axis=axis, ddof=ddof)
    else:
        result = np.std(x, axis=axis, ddof=ddof)

    if axis is None:
        return float(result)
    return result


def min(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Minimum of x along axis. Named min to match the spec; it shadows the builtin inside this module.

    Args: x array of any shape; axis None reduces all elements.
    Returns: float when axis is None, else array with that axis removed.
    Raises: TypeError if x is not numeric; ValueError if x is empty.
    Complexity: O(n) time, O(1) extra space.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmin(x, axis=axis)
    else:
        result = np.min(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def max(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Maximum of x along axis. See min.

    Args: x array of any shape; axis None reduces all elements.
    Returns: float when axis is None, else array with that axis removed.
    Raises: TypeError if x is not numeric; ValueError if x is empty.
    Complexity: O(n) time, O(1) extra space.
    """
    x = check_array(x)
    if skip_nan:
        result = np.nanmax(x, axis=axis)
    else:
        result = np.max(x, axis=axis)

    if axis is None:
        return float(result)
    return result


def quantile(x: np.ndarray, q, axis: int = None, method: str = "linear", skip_nan: bool = False):
    """Quantile(s) of x along axis.

    Args: q one probability or a sequence of them, each in [0, 1];
          method 'linear' | 'lower' | 'higher' | 'midpoint' | 'nearest'.
    Returns: float for one q and axis None; otherwise an array, with a leading axis of len(q) for a sequence.
    Raises: ValueError if q is outside [0, 1] or method is unknown.
    Complexity: O(n) average time, O(n) space.
    """
    x = check_array(x)
    q = np.asarray(q, dtype=float)
    if q.size == 0 or np.any(q < 0) or np.any(q > 1):
        raise ValueError("q must lie in [0, 1]")
    if method not in ("linear", "lower", "higher", "midpoint", "nearest"):
        raise ValueError(f"unknown method {method!r}")

    if skip_nan:
        result = np.nanquantile(x, q, axis=axis, method=method)
    else:
        result = np.quantile(x, q, axis=axis, method=method)

    if axis is None and q.ndim == 0:
        return float(result)
    return result


def histogram(x: np.ndarray, bins=10, range: tuple = None, density: bool = False):
    """Histogram of the flattened data. NaN is always dropped because it cannot be binned.

    Args: bins number of equal-width bins or explicit edges; range (low, high) when bins is a number.
    Returns: (counts of shape (b,), edges of shape (b + 1,)).
    Raises: ValueError if no non-NaN value is left.
    Complexity: O(n log b) time, O(n + b) space.
    """
    x = check_array(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("histogram needs at least one non-NaN value")
    return np.histogram(x, bins=bins, range=range, density=density)


class RunningStats:
    """Welford streaming mean / variance / min / max, updated one batch at a time.

    Args: skip_nan True drops rows that contain NaN before updating.
    Attributes: n rows seen; mean, m2 (sum of squared deviations), min, max of shape (d,).
    Raises: ValueError if a batch has a different number of features; RuntimeError if var() is called before data.
    Complexity: update O(b * d) time for b rows; O(d) memory however many rows are seen.
    """

    def __init__(self, skip_nan: bool = False):
        self.skip_nan = skip_nan
        self.n = 0

    def update(self, x: np.ndarray) -> "RunningStats":
        """Fold a scalar, a (b,) vector of one feature, or a (b, d) batch into the running statistics."""
        x = np.asarray(x, dtype=float)
        if x.ndim < 2:
            x = x.reshape(-1, 1)
        if x.ndim > 2:
            raise ValueError(f"batch must be at most 2-D, got shape {x.shape}")
        if self.skip_nan:
            x = x[~np.isnan(x).any(axis=1)]

        b, d = x.shape
        if b == 0:
            return self
        if self.n == 0:
            self.mean = np.zeros(d)
            self.m2 = np.zeros(d)
            self.min = np.full(d, np.inf)
            self.max = np.full(d, -np.inf)
        if d != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} features, got {d}")

        # merge the stored statistics with this batch (Chan et al. formula)
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - self.mean
        n_new = self.n + b

        self.mean = self.mean + delta * b / n_new
        self.m2 = self.m2 + batch_m2 + delta ** 2 * self.n * b / n_new
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))
        self.n = n_new
        return self

    def var(self, ddof: int = 0) -> np.ndarray:
        """Running variance per feature. Raises ValueError when n <= ddof."""
        if self.n == 0:
            raise RuntimeError("no data yet; call update() first")
        if self.n <= ddof:
            raise ValueError(f"var needs more than ddof={ddof} rows, seen {self.n}")
        return self.m2 / (self.n - ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Running standard deviation per feature, sqrt(var(ddof))."""
        return np.sqrt(self.var(ddof))


if __name__ == "__main__":
    from .utils import iter_batches

    rng = np.random.default_rng(0)
    X = rng.normal(1e6, 1, (1000, 3))
    X[5, 1] = np.nan

    assert np.isnan(mean(X, axis=0)[1])
    assert np.allclose(mean(X, axis=0, skip_nan=True), np.nanmean(X, axis=0))
    assert isinstance(mean(X[:, 0]), float)
    assert quantile(X, [0.25, 0.5], axis=0, skip_nan=True).shape == (2, 3)
    assert histogram(X[:, 1], bins=5)[0].sum() == 999

    running = RunningStats(skip_nan=True)
    for batch in iter_batches(X, 64):
        running.update(batch)
    clean = X[~np.isnan(X).any(axis=1)]
    assert running.n == 999
    assert np.allclose(running.mean, clean.mean(axis=0))
    assert np.allclose(running.var(), clean.var(axis=0))

    print("stats ok")
