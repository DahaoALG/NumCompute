import unittest

import numpy as np

from numcompute.sort_search import (
    binary_search,
    lexsort,
    quickselect,
    quicksort,
    stable_sort,
    topk,
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


class TestStableSortAndLexsort(unittest.TestCase):
    def test_stable_sort_handles_non_contiguous_view(self):
        base = np.array([5, 99, 1, 99, 4, 99])
        values = base[::2]
        result = stable_sort(values)
        np.testing.assert_array_equal(result, [1, 4, 5])
        np.testing.assert_array_equal(values, [5, 1, 4])

    def test_stable_sort_preserves_order_of_equal_signed_zeros(self):
        values = np.array([-0.0, 0.0, 0.0, -0.0])
        result = stable_sort(values)
        np.testing.assert_array_equal(np.signbit(result), np.signbit(values))

    def test_stable_sort_places_nans_last(self):
        result = stable_sort([np.nan, 2.0, -1.0, np.nan])
        np.testing.assert_array_equal(result[:2], [-1.0, 2.0])
        self.assertTrue(np.isnan(result[2:]).all())

    def test_lexsort_uses_last_key_as_primary(self):
        primary = [1, 0, 1, 0]
        secondary = [2, 2, 1, 1]
        np.testing.assert_array_equal(lexsort((secondary, primary)), [3, 1, 2, 0])

    def test_lexsort_validates_key_lengths(self):
        with self.assertRaises(ValueError):
            lexsort(([1, 2], [1]))


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

    def test_returns_original_indices_and_handles_non_contiguous_input(self):
        base = np.array([9, 99, 1, 99, 5, 99, 3])
        values = base[::2]
        selected, indices = top_k(values, 2, return_indices=True)
        np.testing.assert_array_equal(selected, [9, 5])
        np.testing.assert_array_equal(values[indices], selected)
        np.testing.assert_array_equal(values, [9, 1, 5, 3])

    def test_spec_topk_defaults_to_returning_indices(self):
        selected, indices = topk([4, 1, 8, 3], 2)
        np.testing.assert_array_equal(selected, [8, 4])
        np.testing.assert_array_equal(np.array([4, 1, 8, 3])[indices], selected)

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
    def test_returns_first_matching_duplicate_and_found_status(self):
        self.assertEqual(binary_search([1, 2, 2, 2, 5], 2), (1, True))

    def test_returns_insertion_index_when_absent(self):
        self.assertEqual(binary_search([1, 3, 5, 7], 4), (2, False))

    def test_handles_empty_and_nan_values(self):
        self.assertEqual(binary_search([], 1), (0, False))
        self.assertEqual(
            binary_search([1.0, 2.0, np.nan, np.nan], np.nan), (2, True)
        )

    def test_requires_real_numeric_values(self):
        with self.assertRaises(ValueError):
            binary_search([[1, 2]], 1)
        with self.assertRaises(TypeError):
            binary_search([1, 2], "2")


if __name__ == "__main__":
    unittest.main()
