"""Plain-Python versions of NumCompute functions: the "loop" side of the benchmarks.

They loop over Python lists one element at a time, with no NumPy inside the loop,
so the benchmarks show what vectorisation gains. Results are turned into NumPy
arrays only at the end, so they can be checked against the NumCompute versions.
"""

import csv
import math

import numpy as np

from numcompute.io import DEFAULT_MISSING


# io / metrics
def load_csv_loop(path, delimiter=","):
    """Numeric CSV with a header row -> float array"""
    missing = set(DEFAULT_MISSING)

    rows = []
    
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip the header row (only numbers are kept)
        next(reader)

        # Read row by row and convert values accordingly
        for row in reader:
            values = []
            for cell in row:
                cell = cell.strip()
                values.append(math.nan if cell in missing else float(cell))
            rows.append(values)
    
    return np.array(rows, dtype=float)


def accuracy_loop(y_true, y_pred):
    matches = 0

    # Loop through the arrays to count the matching values
    for t, p in zip(y_true, y_pred):
        if t == p:
            matches += 1
    
    return matches / len(y_true)


def confusion_matrix_loop(y_true, y_pred):
    """True labels in rows. Predicted labels in columns. Labels sorted."""
    labels = sorted(set(y_true) | set(y_pred))
    
    index = {label: i for i, label in enumerate(labels)}
    
    counts = [[0] * len(labels) for _ in labels]
    
    for t, p in zip(y_true, y_pred):
        counts[index[t]][index[p]] += 1
    
    return np.array(counts)


def mse_loop(y_true, y_pred):
    total = 0.0

    # Sum the squared error of each prediction
    for t, p in zip(y_true, y_pred):
        total += (t - p) ** 2
    
    return total / len(y_true)


def auc_loop(y_true, y_score, pos_label=1):
    """Share of pairs (positive, negative) where the positive scores higher, a tie counting half."""

    pos = [s for t, s in zip(y_true, y_score) if t == pos_label]
    neg = [s for t, s in zip(y_true, y_score) if t != pos_label]
    
    wins = 0.0

    # Compare every positive score with every negative score
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    # Share of pairs the positive wins
    return wins / (len(pos) * len(neg))


# Preprocessing / stats
def standard_scale_loop(X):
    """Column-wise z-score with the population std (ddof=0); X is a list of rows."""
    n_rows, n_cols = len(X), len(X[0])
    out = [[0.0] * n_cols for _ in range(n_rows)]
    
    for j in range(n_cols):
        # Mean computation
        mean = 0.0
        for i in range(n_rows):
            mean += X[i][j]
        mean /= n_rows

        # Standard deviation computation
        var = 0.0
        for i in range(n_rows):
            var += (X[i][j] - mean) ** 2
        std = math.sqrt(var / n_rows) or 1.0  # or 1.0 to avoid zero division
    
        for i in range(n_rows):
            out[i][j] = (X[i][j] - mean) / std

    return np.array(out)


def histogram_loop(x, bins):
    """Counts in `bins` equal-width bins from min(x) to max(x); the last bin includes max(x)."""
    low, high = min(x), max(x)

    width = (high - low) / bins
    
    counts = [0] * bins
    
    for v in x:
        i = int((v - low) / width)
        counts[min(i, bins - 1)] += 1                 # max(x) lands on the last edge
    
    return np.array(counts)


# sort_search / rank
def average_rank_loop(x):
    """1-based ranks; tied values share the average of their ranks. O(n^2)."""
    ranks = []
    
    for v in x:
        less = 0
        equal = 0
        
        for w in x:
            if w < v:
                less += 1
            elif w == v:
                equal += 1
        
        ranks.append(less + (equal + 1) / 2)
    
    return np.array(ranks)


def topk_loop(x, k):
    """The k largest values, largest first, and their indices, by k passes over the data."""
    taken = [False] * len(x)
    values, indices = [], []
    for _ in range(k):
        best = -1
        for i in range(len(x)):
            if not taken[i] and (best == -1 or x[i] > x[best]):
                best = i
        
        taken[best] = True
        values.append(x[best])
        indices.append(best)
    
    return np.array(values), np.array(indices)


def stable_sort_loop(x):
    """Ascending sort by merge sort; equal values keep their input order. O(n log n)."""
    return np.array(_merge_sort(x))


def _merge_sort(x):
    """Sorted copy of the list x: sort each half, then merge them."""
    if len(x) <= 1:
        return list(x)
    
    middle = len(x) // 2
    left = _merge_sort(x[:middle])
    right = _merge_sort(x[middle:])
    
    merged = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if right[j] < left[i]:                    # strict: on a tie the left value goes first
            merged.append(right[j])
            j += 1
        
        else:
            merged.append(left[i])
            i += 1
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged
