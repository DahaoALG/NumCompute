import unittest
from typing import cast

import numpy as np

from numcompute.rank import (
    PercentileRankKind,
    RankMethod,
    percentile,
    percentile_rank,
    rank,
    rankdata,
)


class TestRankData(unittest.TestCase):
    def test_average_rank_for_ties(self):
        np.testing.assert_array_equal(rank([3, 1, 2, 2]), [4, 1, 2.5, 2.5])

    def test_min_max_and_dense_tie_methods(self):
        values = [30, 10, 20, 20]
        np.testing.assert_array_equal(rank(values, "min"), [4, 1, 2, 2])
        np.testing.assert_array_equal(rank(values, "max"), [4, 1, 3, 3])
        np.testing.assert_array_equal(rank(values, "dense"), [3, 1, 2, 2])

    def test_ordinal_breaks_ties_in_input_order(self):
        np.testing.assert_array_equal(
            rank([5, 1, 5, 3], "ordinal"), [3, 1, 4, 2]
        )

    def test_descending_ranks_nan_and_empty(self):
        np.testing.assert_array_equal(
            rank([2.0, np.nan, 5.0, 2.0], ascending=False),
            [2.5, np.nan, 1.0, 2.5],
        )
        self.assertEqual(rank([]).size, 0)

    def test_rejects_invalid_method_and_input(self):
        with self.assertRaises(ValueError):
            rank([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            rank([1, 2], method=cast(RankMethod, "unknown"))

    def test_rankdata_remains_available_as_compatibility_alias(self):
        np.testing.assert_array_equal(rankdata([2, 1]), [2, 1])

    def test_rank_handles_non_contiguous_input(self):
        values = np.array([3, 0, 1, 0, 2])[::2]
        np.testing.assert_array_equal(rank(values), [3, 1, 2])


class TestPercentileRank(unittest.TestCase):
    def test_scalar_score_and_mean_tie_percentile(self):
        self.assertEqual(percentile_rank([1, 2, 2, 4], 2), 50.0)

    def test_percentile_kinds(self):
        sample = [1, 2, 2, 4]
        self.assertEqual(percentile_rank(sample, 2, "strict"), 25.0)
        self.assertEqual(percentile_rank(sample, 2, "weak"), 75.0)
        self.assertEqual(percentile_rank(sample, 2, "rank"), 62.5)

    def test_multiple_scores_nan_omission_and_nan_score(self):
        result = np.asarray(
            percentile_rank([1.0, np.nan, 3.0], [0.0, 2.0, 4.0, np.nan])
        )
        np.testing.assert_allclose(result[:3], [0.0, 50.0, 100.0])
        self.assertTrue(np.isnan(result[3]))

    def test_empty_sample_and_invalid_kind(self):
        self.assertTrue(np.isnan(np.asarray(percentile_rank([np.nan], [1.0]))[0]))
        with self.assertRaises(ValueError):
            percentile_rank([1, 2], 1, kind=cast(PercentileRankKind, "invalid"))


class TestPercentile(unittest.TestCase):
    def test_scalar_percentile(self):
        self.assertEqual(percentile([0, 10, 20, 30], 50), 15.0)

    def test_multiple_percentiles_and_interpolation(self):
        np.testing.assert_array_equal(
            percentile([0, 10, 20, 30], [0, 50, 100], interpolation="nearest"),
            [0, 20, 30],
        )
        np.testing.assert_array_equal(
            percentile([0, 10, 20, 30], [25, 75], interpolation="lower"), [0, 20]
        )

    def test_omits_nans_and_rejects_all_nan_sample(self):
        self.assertEqual(percentile([0.0, np.nan, 20.0], 50), 10.0)
        with self.assertRaises(ValueError):
            percentile([np.nan], 50)

    def test_rejects_out_of_range_percentiles(self):
        with self.assertRaises(ValueError):
            percentile([1, 2], -1)
        with self.assertRaises(ValueError):
            percentile([1, 2], [50, 101])


if __name__ == "__main__":
    unittest.main()
