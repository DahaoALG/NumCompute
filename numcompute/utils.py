"""Shared helpers. Each member owns the functions in their own block."""

import numpy as np


# ---- Dat: validation and batching ----

def check_array(X, ndim: int = None) -> np.ndarray:
    """Convert X to a float64 array and check it is non-empty with the expected ndim.

    Args: X array_like of numbers; ndim required dimensions, None accepts any.
    Returns: float64 ndarray, same shape as X.
    Raises: TypeError if X is not numeric; ValueError if X is empty or ndim is wrong.
    Complexity: O(n) time; copies only when the dtype changes.
    """
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("X must be numeric")

    if X.size == 0:
        raise ValueError("X is empty")
    if ndim is not None and X.ndim != ndim:
        raise ValueError(f"X must be {ndim}-D, got shape {X.shape}; use X.reshape(-1, 1) for one feature")
    return X


def iter_batches(X: np.ndarray, batch_size: int):
    """Yield X in consecutive row batches (views, no copies).

    Args: X array of shape (n, ...); batch_size rows per batch, the last may be shorter.
    Returns: generator of arrays with at most batch_size rows.
    Raises: ValueError if batch_size is not positive.
    Complexity: O(n / batch_size) iterations, O(1) extra space.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X = np.asarray(X)
    for start in range(0, X.shape[0], batch_size):
        yield X[start:start + batch_size]


if __name__ == "__main__":
    assert check_array([[1, 2]]).dtype == float

    sizes = [batch.shape[0] for batch in iter_batches(np.zeros((10, 1)), 4)]
    assert sizes == [4, 4, 2]

    try:
        check_array([1.0], ndim=2)
        assert False, "1-D input should be rejected"
    except ValueError:
        pass

    print("utils ok")
