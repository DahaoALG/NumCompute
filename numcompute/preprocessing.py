"""StandardScaler, MinMaxScaler, Imputer, and OneHot encoding.

Task allocation: Dat.
"""

import numpy as np


def share_array(X, n_features=None, dtype=None):
    """Convert X to an array and check its shape and optional feature count."""
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a nonempty 2-D array (samples, features)")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("X must have the same number of columns as the fitted data")
    if dtype is float and np.isinf(X).any():
        raise ValueError("X must not contain infinity")
    return X


def share_transform_input(transformer, X, dtype=None):
    """Check that fit was called, then validate X using share_array."""
    if not hasattr(transformer, "n_features_in_"):
        raise ValueError("Call fit before transform")
    return share_array(X, transformer.n_features_in_, dtype)


class StandardScaler:
    """Standardize each column using (X - mean) / standard deviation.
    """

    def fit(self, X):
        """Learn from numeric X (n, p); return self.

        Raises ValueError for invalid shapes, infinity or all-NaN columns;
        TypeError/ValueError for nonnumeric input.
        """
        X = share_array(X, dtype=float)
        if np.isnan(X).all(axis=0).any():
            raise ValueError("Cannot fit a column containing only NaN; impute it first")
        
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1, self.scale_) # Avoiding then divide ZERO!!!!!
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return scaled float X (n, p); input is unchanged. Time/space: O(n*p).

        Raises ValueError if unfitted, invalid shape/column count or infinity;
        TypeError/ValueError if values cannot be converted to float.
        """
        X = share_transform_input(self, X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit numeric X (n, p) and return scaled (n, p); same errors as fit/transform.
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale training columns to feature_range=(low, high), default (0, 1).

    NaNs are ignored during fit and preserved during transform. Constant
    training columns map to low (use a denominator of 1). New values are not
    clipped, so values outside the training range can map outside the range.
    """

    def __init__(self, feature_range=(0, 1)):
        """Set a finite (low, high) pair; raise ValueError unless low < high.
        """
        bounds = np.asarray(feature_range, dtype=float)
        if bounds.shape != (2,):
            raise ValueError("feature_range must contain two numbers")

        if not np.isfinite(bounds).all():
            raise ValueError("feature_range cannot contain NaN or infinity")
        
        low, high = bounds
        if low >= high:
            raise ValueError("low must be less than high")
        
        self.feature_range = tuple(bounds)

    def fit(self, X):
        """Learn minima/ranges from numeric X (n, p); return self.

        Raises ValueError for invalid shapes, infinity or all-NaN columns;
        TypeError/ValueError for nonnumeric input.
        """
        X = share_array(X, dtype=float)
        if np.isnan(X).all(axis=0).any():
            raise ValueError("Cannot fit a column containing only NaN; impute it first")
        
        self.min_ = np.nanmin(X, axis=0)
        self.range_ = np.nanmax(X, axis=0) - self.min_
        self.range_ = np.where(self.range_ == 0, 1, self.range_) # Avoiding then divide ZERO!!!!!
        self.n_features_in_ = X.shape[1]
        
        return self

    def transform(self, X):
        """Return rescaled float X (n, p); input unchanged. Time/space: O(n*p).

        Raises ValueError if unfitted, invalid shape/column count or infinity;
        TypeError/ValueError for nonnumeric input.
        """
        X = share_transform_input(self, X, dtype=float)
        low, high = self.feature_range
        
        X_normalized = (X - self.min_) / self.range_
        X_out = low + X_normalized * (high - low)
        
        return X_out

    def fit_transform(self, X):
        """Fit X (n, p) and return rescaled (n, p); same errors as fit/transform.
        """
        return self.fit(X).transform(X)


class Imputer:
    """Replace actual NaN values with a constant; default fill_value=0.

    Supports numeric arrays and object arrays containing strings and NaNs.
    Literal strings such as 'nan' are not treated as missing by this class.
    """

    def __init__(self, fill_value=0):
        """Store a scalar constant; raise ValueError for nonscalars. Time/space: O(1)."""
        if np.ndim(fill_value) != 0:
            raise ValueError("fill_value must be a scalar")
        self.fill_value = fill_value

    def fit(self, X):
        """Record feature count from X (n, p); return self.
        """
        X = share_array(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return filled X (n, p), without mutation. Time/space: O(n*p).
        """
        X = share_transform_input(self, X)
        return np.where(X != X, self.fill_value, X)

    def fit_transform(self, X):
        """Fit X (n, p) and return filled (n, p); same errors as fit/transform.
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode categorical columns without Python loops.

    Categories are converted to strings and sorted within each input column.
    Missing NaNs must be imputed first. Unknown categories produce all zeros
    for their input column. categories_ and feature_indices_ describe each
    """

    def fit(self, X):
        """Learn categories from X (n, p); return self.

        Raises ValueError for invalid shapes or NaNs. Numeric/string labels
        with the same string representation are treated as the same category.
        """
        X = share_array(X, dtype=object)
        if np.any(X != X):
            raise ValueError("Fill missing categories with Imputer before encoding")
        X = X.astype(str)
        
        categories, codes = np.unique(X, return_inverse=True)
        # Pair each category code with its original feature column.
        columns = np.tile(np.arange(X.shape[1]), X.shape[0])
        pairs = np.column_stack((columns, codes.reshape(-1)))
        pairs = np.unique(pairs, axis=0)
        self.feature_indices_ = pairs[:, 0]
        self.categories_ = categories[pairs[:, 1]]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return integer one-hot array (n, k), k = total learned categories.
        """
        X = share_transform_input(self, X, dtype=object)
        if np.any(X != X):
            raise ValueError("Fill missing categories with Imputer before encoding")
        X = X.astype(str)
        return (X[:, self.feature_indices_] == self.categories_).astype(int)

    def fit_transform(self, X):
        """Fit X (n, p), return encoded (n, k); same errors as fit/transform.

        Time: O(n*p*log(n*p) + n*k); space: O(n*p + n*k), bounded label length.
        """
        return self.fit(X).transform(X)
