"""Ranking with tie handling and percentile utilities.

NaNs receive NaN ranks and are omitted from percentile calculations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from numcompute.utils import _as_numeric_array

__all__ = ["rank", "rankdata", "percentile_rank", "percentile"]

RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
PercentileRankKind = Literal["rank", "weak", "strict", "mean"]
PercentileInterpolation = Literal[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]


def rank(
    values: ArrayLike,
    method: RankMethod = "average",
    ascending: bool = True,
) -> np.ndarray:
    """Assign one-based ranks to a one-dimensional sequence.

    Parameters
    ----------
    values : array-like of shape (n,)
        Real numeric values to rank. NaNs receive NaN ranks.
    method : {"average", "min", "max", "dense", "ordinal"}, default="average"
        Tie handling: ``average`` assigns the mean occupied rank; ``min`` and
        ``max`` assign the lowest or highest occupied rank; ``dense`` assigns
        consecutive ranks to distinct values; ``ordinal`` breaks ties by
        preserving their input order.
    ascending : bool, default=True
        If true, smaller values receive smaller ranks. If false, larger values
        receive smaller ranks.

    Returns
    -------
    ndarray of shape (n,), dtype float
        One-based ranks in the original input order. NaN elements receive NaN.

    Raises
    ------
    TypeError
        If values are not real numeric values or ``ascending`` is not boolean.
    ValueError
        If values are not one-dimensional or ``method`` is unsupported.

    Notes
    -----
    Time complexity is O(n log n), dominated by sorting. Space complexity is
    O(n).
    """
    array = _as_numeric_array(values)
    if method not in {"average", "min", "max", "dense", "ordinal"}:
        raise ValueError(
            "method must be 'average', 'min', 'max', 'dense', or 'ordinal'"
        )
    if not isinstance(ascending, (bool, np.bool_)):
        raise TypeError("ascending must be a boolean")

    result = np.full(array.size, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(~np.isnan(array))
    valid_values = array[valid_indices]
    if not valid_values.size:
        return result
    _, inverse, counts = np.unique(
        valid_values, return_inverse=True, return_counts=True
    )

    if method == "ordinal":
        if ascending:
            order = np.argsort(valid_values, kind="stable")
        else:
            order = np.argsort(-inverse, kind="stable")
        result[valid_indices[order]] = np.arange(1, valid_values.size + 1)
        return result

    if ascending:
        group_counts = counts
    else:
        group_counts = counts[::-1]
    maximum_ranks = np.cumsum(group_counts)
    minimum_ranks = maximum_ranks - group_counts + 1
    if method == "average":
        group_ranks = (minimum_ranks + maximum_ranks) / 2
    elif method == "min":
        group_ranks = minimum_ranks
    elif method == "max":
        group_ranks = maximum_ranks
    else:
        group_ranks = np.arange(1, counts.size + 1)
    if not ascending:
        group_ranks = group_ranks[::-1]
    result[valid_indices] = group_ranks[inverse]
    return result


# Keep the earlier API name as a compatibility alias.
rankdata = rank


def percentile_rank(
    values: ArrayLike,
    scores: ArrayLike,
    kind: PercentileRankKind = "mean",
) -> float | np.ndarray:
    """Compute the percentile rank of each score relative to a sample.

    Percentile ranks range from 0 to 100. NaNs in ``values`` are omitted, and
    NaN scores produce NaN results. If the sample has no non-NaN values, all
    results are NaN.

    Parameters
    ----------
    values : array-like of shape (n,)
        Reference sample.
    scores : real number or array-like
        Score or scores whose percentile ranks are requested.
    kind : {"rank", "weak", "strict", "mean"}, default="mean"
        ``strict`` uses the percentage of sample values strictly below each
        score; ``weak`` uses values less than or equal; ``mean`` averages those
        two percentages. ``rank`` uses the average one-based rank of matching
        sample values, divided by the non-NaN sample size. If there are no
        matches, it is equivalent to ``weak``.

    Returns
    -------
    float or ndarray
        A scalar for a scalar score; otherwise an array matching ``scores``.

    Raises
    ------
    TypeError
        If the sample or scores are not real numeric values, or kind is not a
        string.
    ValueError
        If the sample or scores are not one-dimensional, or kind is
        unsupported.

    Notes
    -----
    Time complexity is O(n log n + m log n), where n is the sample length and
    m is the number of scores. Space complexity is O(n + m).
    """
    sample = _as_numeric_array(values)
    scalar_score: np.ndarray = np.asarray(scores)
    scalar = scalar_score.ndim == 0
    if scalar:
        score_array: np.ndarray = scalar_score.reshape(1)
    elif scalar_score.ndim == 1:
        score_array = scalar_score
    else:
        raise ValueError("scores must be a scalar or one-dimensional sequence")

    if not np.issubdtype(score_array.dtype, np.number) or np.issubdtype(
        score_array.dtype, np.complexfloating
    ):
        raise TypeError("scores must contain real numeric values")
    if kind not in {"rank", "weak", "strict", "mean"}:
        raise ValueError("kind must be 'rank', 'weak', 'strict', or 'mean'")

    valid_sample = np.sort(sample[~np.isnan(sample)])
    result = np.full(score_array.size, np.nan, dtype=np.float64)
    if valid_sample.size:
        valid_scores = ~np.isnan(score_array)
        lower = np.searchsorted(valid_sample, score_array[valid_scores], side="left")
        upper = np.searchsorted(valid_sample, score_array[valid_scores], side="right")
        count = valid_sample.size
        strict_percent = lower / count * 100
        weak_percent = upper / count * 100
        if kind == "strict":
            values_percent = strict_percent
        elif kind == "weak":
            values_percent = weak_percent
        elif kind == "mean":
            values_percent = (strict_percent + weak_percent) / 2
        else:
            matched = upper > lower
            values_percent = weak_percent.copy()
            values_percent[matched] = (
                (lower[matched] + upper[matched] + 1) / 2 / count * 100
            )
        result[valid_scores] = values_percent

    return float(result[0]) if scalar else result


def percentile(
    values: ArrayLike,
    q: float | Sequence[float] | np.ndarray,
    interpolation: PercentileInterpolation = "linear",
) -> float | np.ndarray:
    """Compute one or more percentiles of a one-dimensional numeric sample.

    NaNs in ``values`` are omitted. Percentile positions use NumPy's
    ``percentile`` interpolation methods.

    Parameters
    ----------
    values : array-like of shape (n,)
        Numeric sample. At least one non-NaN value is required.
    q : float or array-like
        Percentile or percentiles in the closed interval [0, 100].
    interpolation : str, default="linear"
        NumPy percentile interpolation method, such as ``"linear"``,
        ``"lower"``, ``"higher"``, ``"midpoint"``, or ``"nearest"``.

    Returns
    -------
    float or ndarray
        A scalar for scalar ``q``; otherwise an array matching ``q``.

    Raises
    ------
    TypeError
        If values or q contain non-numeric or complex values.
    ValueError
        If values are not one-dimensional, no non-NaN values remain, or a
        requested percentile is outside [0, 100] or interpolation is
        unsupported by NumPy.

    Notes
    -----
    Time complexity is O(n log n), dominated by sorting the sample. Space
    complexity is O(n).
    """
    array = _as_numeric_array(values)
    quantiles = np.asarray(q)
    scalar = quantiles.ndim == 0
    if not scalar and quantiles.ndim != 1:
        raise ValueError("q must be a scalar or one-dimensional sequence")
    if not np.issubdtype(quantiles.dtype, np.number) or np.issubdtype(
        quantiles.dtype, np.complexfloating
    ):
        raise TypeError("q must contain real numeric percentiles")
    if not np.all(np.isfinite(quantiles)) or np.any(
        (quantiles < 0) | (quantiles > 100)
    ):
        raise ValueError("q values must be finite and in the range [0, 100]")

    valid_values = array[~np.isnan(array)]
    if valid_values.size == 0:
        raise ValueError("values must contain at least one non-NaN value")
    result = np.percentile(valid_values, quantiles, method=interpolation)
    return float(result) if scalar else result
