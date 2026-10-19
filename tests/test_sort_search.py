import unittest

import numpy as np

from numcompute.sort_search import (
    argpartition,
    binary_search,
    merge_sort,
    quickselect,
    quicksort,
    top_k,
)


class TestQuickSort(unittest.TestCase):
    def test_sorts_without_mutating_input(self):
        values = np.array([4, -1, 3, 2])
        result = quicksort(values)
        np.testing.assert_array_equal(result, [-1, 2, 3, 4])
        np.testing.assert_array_equal(values, [4, -1, 3, 2])

    def test_handles_equal_values_and_extremes(self):
        np.testing.assert_array_equal(
            quicksort([7, 7, 7, -np.inf, np.inf]), [-np.inf, 7, 7, 7, np.inf]
        )

    def test_places_nans_last(self):
        result = quicksort([np.nan, 2.0, np.nan, -1.0])
        np.testing.assert_array_equal(result[:2], [-1.0, 2.0])
        self.assertTrue(np.isnan(result[2:]).all())

    def test_validates_dimension_and_dtype(self):
        with self.assertRaises(ValueError):
            quicksort([[1, 2], [3, 4]])
        with self.assertRaises(TypeError):
            quicksort(["a", "b"])


class TestMergeSort(unittest.TestCase):
    def test_preserves_dtype_and_input(self):
        values = np.array([3, 1, 2, 1], dtype=np.int32)
        result = merge_sort(values)
        np.testing.assert_array_equal(result, [1, 1, 2, 3])
        self.assertEqual(result.dtype, values.dtype)
        np.testing.assert_array_equal(values, [3, 1, 2, 1])

    def test_handles_empty_and_single_value(self):
        np.testing.assert_array_equal(merge_sort([]), [])
        np.testing.assert_array_equal(merge_sort([5]), [5])

    def test_places_nans_last(self):
        result = merge_sort([np.nan, 2.0, np.nan, -1.0])
        np.testing.assert_array_equal(result[:2], [-1.0, 2.0])
        self.assertTrue(np.isnan(result[2:]).all())

    def test_rejects_non_numeric_input(self):
        with self.assertRaises(TypeError):
            merge_sort(["one", "two"])


class TestArgpartition(unittest.TestCase):
    def test_places_requested_rank_correctly(self):
        values = np.array([9, 1, 5, 3, 7])
        indices = argpartition(values, 2)
        self.assertEqual(values[indices[2]], 5)
        self.assertTrue(np.all(values[indices[:2]] <= 5))
        self.assertTrue(np.all(values[indices[3:]] >= 5))

    def test_handles_duplicate_values(self):
        values = np.array([4, 2, 2, 2, 8])
        indices = argpartition(values, 2)
        self.assertEqual(values[indices[2]], 2)
        self.assertTrue(np.all(values[indices[:2]] <= 2))
        self.assertTrue(np.all(values[indices[3:]] >= 2))

    def test_places_nan_ranks_last(self):
        values = np.array([np.nan, 1.0, 3.0])
        indices = argpartition(values, 2)
        self.assertTrue(np.isnan(values[indices[2]]))
        np.testing.assert_array_equal(values[indices[:2]], [1.0, 3.0])

    def test_rejects_invalid_rank(self):
        with self.assertRaises(ValueError):
            argpartition([1, 2], 2)
        with self.assertRaises(TypeError):
            argpartition([1, 2], 1.5)


class TestTopK(unittest.TestCase):
    def test_returns_largest_values_descending(self):
        np.testing.assert_array_equal(top_k([4, 1, 8, 3, 6], 3), [8, 6, 4])

    def test_returns_smallest_values_ascending(self):
        np.testing.assert_array_equal(
            top_k([4, 1, 8, 3, 6], 3, largest=False), [1, 3, 4]
        )

    def test_handles_zero_full_length_and_nan_values(self):
        np.testing.assert_array_equal(top_k([1.0, np.nan, 3.0], 0), [])
        largest = top_k([1.0, np.nan, 3.0], 3)
        np.testing.assert_array_equal(largest[:2], [3.0, 1.0])
        self.assertTrue(np.isnan(largest[2]))

    def test_rejects_out_of_range_or_non_integer_k(self):
        with self.assertRaises(ValueError):
            top_k([1, 2], 3)
        with self.assertRaises(ValueError):
            top_k([1, 2], -1)
        with self.assertRaises(TypeError):
            top_k([1, 2], 1.5)


class TestQuickselect(unittest.TestCase):
    def test_matches_sorted_rank(self):
        values = [8, 1, 4, 2, 7, 3]
        for kth, expected in enumerate(sorted(values)):
            self.assertEqual(quickselect(values, kth), expected)

    def test_handles_duplicates(self):
        self.assertEqual(quickselect([3, 1, 2, 2, 4], 2), 2)

    def test_orders_nan_after_numbers(self):
        self.assertEqual(quickselect([np.nan, 2.0, 1.0], 1), 2.0)
        self.assertTrue(np.isnan(quickselect([np.nan, 2.0, 1.0], 2)))

    def test_rejects_invalid_rank(self):
        with self.assertRaises(ValueError):
            quickselect([], 0)
        with self.assertRaises(ValueError):
            quickselect([1, 2], -1)


class TestBinarySearch(unittest.TestCase):
    def test_returns_first_matching_duplicate(self):
        self.assertEqual(binary_search([1, 2, 2, 2, 5], 2), 1)

    def test_returns_minus_one_when_absent(self):
        self.assertEqual(binary_search([1, 3, 5, 7], 4), -1)

    def test_handles_empty_and_nan_values(self):
        self.assertEqual(binary_search([], 1), -1)
        self.assertEqual(binary_search([1.0, 2.0, np.nan, np.nan], np.nan), 2)

    def test_requires_real_numeric_values(self):
        with self.assertRaises(ValueError):
            binary_search([[1, 2]], 1)
        with self.assertRaises(TypeError):
            binary_search([1, 2], "2")


if __name__ == "__main__":
    unittest.main()
