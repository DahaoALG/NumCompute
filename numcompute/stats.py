"""Descriptive statistics, quantiles, histogram and streaming (Welford) statistics.

axis follows NumPy: None reduces everything to one float, axis=0 on (n, d) gives (d,).
skip_nan=False lets NaN propagate like NumPy; skip_nan=True ignores NaN values.
"""

import numpy as np

from .utils import check_array


def mean(x: np.ndarray, axis: int = None, skip_nan: bool = False):
    """Compute the mean along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Time: O(N), N=x.size. Space: O(N) in the worst case for conversion,
    NaN handling and the output; float inputs may avoid a conversion copy.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
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
    """Compute the median along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Median has a conservative O(N * log(N)) time bound and O(N) space.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
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
    """Compute the standard deviation along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    ddof : int, default 0
        Divisor is the number of included values minus ddof.
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        Float for axis=None; otherwise the input shape with axis removed.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).
        Also raised when any reduced group has no more than ddof values.

    Notes
    -----
    Time and worst-case space: O(N), N=x.size, including conversion and
    centred/NaN temporary arrays.
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
    """Compute the minimum along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Time: O(N), N=x.size. Space: O(N) in the worst case for conversion,
    NaN handling and the output; float inputs may avoid a conversion copy.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
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
    """Compute the maximum along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Data converted to float.
    axis : int or None, default None
        Reduction axis; None reduces all values. On (n, d), axis=0 gives (d,).
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the reduced array with
        the selected axis removed (possibly a zero-dimensional array scalar).

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid (including NumPy axis errors).

    Notes
    -----
    Time: O(N), N=x.size. Space: O(N) in the worst case for conversion,
    NaN handling and the output; float inputs may avoid a conversion copy.
    All-NaN slices return NaN with NumPy's warning when skip_nan=True.
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
    """Compute quantiles along an axis.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Input data converted to float.
    q : float or 1-D array_like
        One or more probabilities in [0, 1].
    axis : int or None, default None
        Axis to reduce; None flattens the data.
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        Quantile interpolation method; default 'linear'.
    skip_nan : bool, default False
        Ignore NaNs if true, otherwise propagate them.

    Returns
    -------
    float or ndarray
        Scalar q and axis=None return a float. Otherwise scalar q removes
        the reduced axis; a sequence prepends an axis of length len(q).

    Raises
    ------
    TypeError
        If x or q cannot be converted to numbers.
    ValueError
        If x/q is empty, q is invalid, method is unsupported or axis is invalid.

    Notes
    -----
    Conservative time: O(N * log(N) + output_size); space O(N + output_size),
    where N=x.size. NumPy partition-based selection can be faster.
    NaN handling follows NumPy; all-NaN slices may emit a warning.
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
    """Bin flattened data after discarding NaN values.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Input flattened to one dimension.
    bins : int or sequence of edges, default 10
        Number of equal-width bins or explicit increasing edges.
    range : pair of numbers or None, default None
        Bounds for automatically chosen edges; values outside are ignored.
    density : bool, default False
        If true, normalize counts so the density integral is one.

    Returns
    -------
    tuple of (ndarray, ndarray)
        Counts or densities of shape (b,) and bin edges of shape (b+1,).
        All bins except the last exclude the right endpoint.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data or bins has invalid type.
    ValueError
        If x is empty, all values are NaN, or bins/range is invalid.

    Notes
    -----
    Conservative time: O(N * log(N) + N * log(b) + b); space O(N + b),
    N=x.size. NumPy uses faster paths for equal-width bins.
    """
    x = check_array(x).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("histogram needs at least one non-NaN value")
    return np.histogram(x, bins=bins, range=range, density=density)


class RunningStats:
    """Accumulate per-feature streaming moments, minima and maxima by batches.

    Parameters
    ----------
    skip_nan : bool, default False
        Drop entire rows containing NaN when true. Otherwise NaN propagates
        into the affected feature statistics.

    Attributes
    ----------
    n : int
        Number of retained rows, initially zero.
    mean : ndarray, shape (d,)
        Running means, created after the first retained batch.
    m2 : ndarray, shape (d,)
        Running sums of squared deviations from the means.
    min, max : ndarray, shape (d,)
        Running minima and maxima.

    Raises
    ------
    ValueError
        If a batch has invalid dimensions or a different feature count, or
        variance/standard deviation is requested with n <= ddof.
    RuntimeError
        If variance/standard deviation is requested before any retained data.

    Notes
    -----
    Uses Chan/Welford moment merging. Updating b rows and d features takes
    O(b*d) time and temporary space; persistent state is O(d), independent
    of the total number of rows.
    """

    def __init__(self, skip_nan: bool = False):
        """Initialize an empty streaming accumulator.

        Parameters
        ----------
        skip_nan : bool, default False
            If true, discard each incoming row containing any NaN. Otherwise
            NaN propagates into the statistics for its feature.

        Returns
        -------
        None
            Set n=0; feature arrays are created by the first retained batch.

        Notes
        -----
        Time and space: O(1).
        """
        self.skip_nan = skip_nan
        self.n = 0

    def update(self, x: np.ndarray) -> "RunningStats":
        """Merge an incoming batch into the stored statistics.

        Parameters
        ----------
        x : numeric scalar or array_like, shape (b,) or (b, d)
            A scalar is one sample with one feature. A 1-D array is b samples of
            one feature, not one sample of b features. Later batches must use d
            features. With skip_nan=True, discard complete rows containing NaN.

        Returns
        -------
        RunningStats
            The updated instance (self). An empty or fully discarded batch does
            not change state. n counts retained rows; feature statistics are (d,).

        Raises
        ------
        TypeError
            If x cannot be converted to a numeric array.
        ValueError
            If conversion fails, x has more than two dimensions, or a nonempty
            batch has a different feature count.

        Notes
        -----
        Time: O(b * d). Temporary space: O(b * d); persistent state O(d).
        Combines batch moments with stored moments using the Chan/Welford formula.
        """
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
        """Return the accumulated per-feature variance.

        Parameters
        ----------
        ddof : int, default 0
            Degrees-of-freedom correction: denominator is n - ddof.
            Use 0 for population variance or 1 for a sample estimate.

        Returns
        -------
        ndarray, shape (d,)
            One value per feature over retained rows.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated.
        ValueError
            If n <= ddof.

        Notes
        -----
        Time and output space: O(d).
        """
        if self.n == 0:
            raise RuntimeError("no data yet; call update() first")
        if self.n <= ddof:
            raise ValueError(f"var needs more than ddof={ddof} rows, seen {self.n}")
        return self.m2 / (self.n - ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        """Return the accumulated per-feature standard deviation.

        Parameters
        ----------
        ddof : int, default 0
            Degrees-of-freedom correction: denominator is n - ddof.
            Use 0 for population variance or 1 for a sample estimate.

        Returns
        -------
        ndarray, shape (d,)
            One value per feature over retained rows.

        Raises
        ------
        RuntimeError
            If no samples have been accumulated.
        ValueError
            If n <= ddof.

        Notes
        -----
        Time and output space: O(d).
        """
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
