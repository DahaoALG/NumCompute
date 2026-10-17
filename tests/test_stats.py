import numpy as np
import pytest

from numcompute import stats
from numcompute.utils import iter_batches


@pytest.fixture
def X():
    return np.random.default_rng(7).normal(size=(100, 3))


@pytest.fixture
def X_nan(X):
    Xn = X.copy()
    Xn[[0, 5, 9], [0, 1, 1]] = np.nan
    return Xn


# reductions

def test_mean_matches_numpy_for_each_axis(X):
    assert stats.mean(X) == pytest.approx(X.mean())
    assert isinstance(stats.mean(X), float)
    np.testing.assert_allclose(stats.mean(X, axis=0), X.mean(axis=0))
    np.testing.assert_allclose(stats.mean(X, axis=1), X.mean(axis=1))
    assert stats.mean(X, axis=0).shape == (3,)


def test_skip_nan_false_propagates_and_true_ignores(X_nan):
    prop = stats.mean(X_nan, axis=0)
    assert np.isnan(prop[0]) and np.isnan(prop[1]) and np.isfinite(prop[2])
    np.testing.assert_allclose(stats.mean(X_nan, axis=0, skip_nan=True), np.nanmean(X_nan, axis=0))
    np.testing.assert_allclose(stats.std(X_nan, axis=0, skip_nan=True), np.nanstd(X_nan, axis=0))
    np.testing.assert_allclose(stats.min(X_nan, axis=0, skip_nan=True), np.nanmin(X_nan, axis=0))


def test_median_std_min_max(X):
    np.testing.assert_allclose(stats.median(X, axis=0), np.median(X, axis=0))
    np.testing.assert_allclose(stats.std(X, axis=0, ddof=1), X.std(axis=0, ddof=1))
    np.testing.assert_allclose(stats.min(X, axis=0), X.min(axis=0))
    np.testing.assert_allclose(stats.max(X, axis=1), X.max(axis=1))


def test_all_equal_values_have_zero_spread():
    x = np.full(20, 3.5)
    assert stats.std(x) == 0.0
    assert stats.median(x) == stats.mean(x) == stats.min(x) == stats.max(x) == 3.5


def test_std_rejects_too_few_values():
    with pytest.raises(ValueError, match="ddof"):
        stats.std([1.0], ddof=1)
    with pytest.raises(ValueError, match="ddof"):
        stats.std(np.array([[1.0, np.nan], [np.nan, 2.0]]), axis=0, ddof=1, skip_nan=True)


def test_empty_and_non_numeric_inputs_rejected():
    with pytest.raises(ValueError, match="empty"):
        stats.mean(np.array([]))
    with pytest.raises(TypeError):
        stats.mean(["a", "b"])


def test_non_contiguous_input_gives_same_result(X):
    np.testing.assert_allclose(stats.mean(X.T, axis=1), X.mean(axis=0))
    np.testing.assert_allclose(stats.std(X[::3], axis=0), X[::3].std(axis=0))


# quantile

def test_quantile_shapes_and_methods(X):
    assert isinstance(stats.quantile(X, 0.5), float)
    assert stats.quantile(X, 0.5, axis=0).shape == (3,)
    assert stats.quantile(X, [0.1, 0.9], axis=0).shape == (2, 3)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert stats.quantile(x, 0.5) == 2.5
    assert stats.quantile(x, 0.5, method="lower") == 2.0
    assert stats.quantile(x, 0.5, method="higher") == 3.0


def test_quantile_nan_handling_and_bad_args(X_nan):
    assert np.isnan(stats.quantile(X_nan, 0.5, axis=0)[0])
    np.testing.assert_allclose(stats.quantile(X_nan, 0.5, axis=0, skip_nan=True), np.nanmedian(X_nan, axis=0))
    with pytest.raises(ValueError, match="q must"):
        stats.quantile(X_nan, 1.5)
    with pytest.raises(ValueError, match="method"):
        stats.quantile(X_nan, 0.5, method="cubic")


def test_quantile_extremes_match_min_max(X):
    np.testing.assert_allclose(stats.quantile(X, 0.0, axis=0), X.min(axis=0))
    np.testing.assert_allclose(stats.quantile(X, 1.0, axis=0), X.max(axis=0))


# histogram

def test_histogram_counts_and_edges(X):
    counts, edges = stats.histogram(X[:, 0], bins=7)
    assert counts.sum() == 100 and edges.shape == (8,)
    assert edges[0] == X[:, 0].min() and edges[-1] == X[:, 0].max()


def test_histogram_drops_nan_and_density_integrates_to_one(X_nan):
    col = X_nan[:, 1]
    counts, _ = stats.histogram(col, bins=5)
    assert counts.sum() == np.sum(~np.isnan(col))
    dens, edges = stats.histogram(col, bins=5, density=True)
    assert np.sum(dens * np.diff(edges)) == pytest.approx(1.0)


def test_histogram_explicit_edges_all_equal_and_all_nan():
    counts, _ = stats.histogram([1, 2, 2, 3, 10], bins=[0, 2, 4, 20])
    np.testing.assert_array_equal(counts, [1, 3, 1])
    counts, edges = stats.histogram(np.full(5, 2.0), bins=3)
    assert counts.sum() == 5 and edges[0] < 2.0 < edges[-1]
    with pytest.raises(ValueError):
        stats.histogram([np.nan, np.nan])


# RunningStats

def test_running_stats_matches_numpy(X):
    rs = stats.RunningStats()
    for batch in iter_batches(X, 16):
        rs.update(batch)
    assert rs.n == 100
    np.testing.assert_allclose(rs.mean, X.mean(axis=0))
    np.testing.assert_allclose(rs.var(), X.var(axis=0))
    np.testing.assert_allclose(rs.std(ddof=1), X.std(axis=0, ddof=1))
    np.testing.assert_allclose(rs.min, X.min(axis=0))
    np.testing.assert_allclose(rs.max, X.max(axis=0))


def test_running_stats_is_stable_for_large_offset():
    x = 1e9 + np.random.default_rng(1).normal(size=10_000)
    rs = stats.RunningStats()
    for value in x[:50]:
        rs.update(value)
    rs.update(x[50:])
    assert rs.n == 10_000
    assert rs.var() == pytest.approx(x.var(), rel=1e-6)
    naive = np.mean(x ** 2) - np.mean(x) ** 2
    assert abs(naive - x.var()) > abs(rs.var() - x.var())


def test_running_stats_skip_nan_and_errors(X_nan):
    rs = stats.RunningStats(skip_nan=True).update(X_nan)
    clean = X_nan[~np.isnan(X_nan).any(axis=1)]
    assert rs.n == clean.shape[0]
    np.testing.assert_allclose(rs.mean, clean.mean(axis=0))
    assert np.isnan(stats.RunningStats().update(X_nan).mean[0])
    with pytest.raises(ValueError, match="features"):
        stats.RunningStats().update(X_nan).update(X_nan[:, :2])
    with pytest.raises(RuntimeError):
        stats.RunningStats().var()
    with pytest.raises(ValueError, match="ddof"):
        stats.RunningStats().update([1.0]).var(ddof=1)
