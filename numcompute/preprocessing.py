"""Scalers, imputer and one-hot encoder with the fit / transform / fit_transform API.

X is always (n_samples, n_features). Column statistics ignore NaN; NaN passes
through transform unchanged, so impute first if that is not wanted.
"""

from typing import Optional, Union

import numpy as np

from .utils import check_array


MISSING = ("", "nan", "None")      # string cells treated as missing by SimpleImputer and OneHotEncoder


def _check_fitted(transformer, X: np.ndarray) -> None:
    """Raise if transform is called before fit or with a different number of columns."""
    if not hasattr(transformer, "n_features_in_"):
        raise RuntimeError(f"{type(transformer).__name__} is not fitted; call fit(X) first")
    if X.ndim != 2 or X.shape[1] != transformer.n_features_in_:
        raise ValueError(f"expected {transformer.n_features_in_} features, got shape {X.shape}")


class StandardScaler:
    """Standardise each column: (x - mean) / std. A constant column becomes 0.

    Attributes: mean_, scale_ of shape (n_features,).
    Raises: TypeError / ValueError on bad input; RuntimeError if not fitted.
    Complexity: O(n * d) time, O(d) extra space.
    """

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = check_array(X, ndim=2)

        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0       # constant column: do not divide by zero
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, ndim=2)
        _check_fitted(self, X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale each column linearly into feature_range. A constant column maps to the range minimum.

    Attributes: data_min_, data_range_ of shape (n_features,).
    Raises: ValueError if feature_range is not (low, high) with low < high.
    Complexity: O(n * d) time, O(d) extra space.
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        low, high = feature_range
        if low >= high:
            raise ValueError(f"feature_range must have low < high, got {feature_range}")
        self.feature_range = (low, high)

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = check_array(X, ndim=2)

        self.data_min_ = np.nanmin(X, axis=0)
        self.data_range_ = np.nanmax(X, axis=0) - self.data_min_
        self.data_range_[self.data_range_ == 0] = np.inf   # constant column: (x - min) / inf = 0
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, ndim=2)
        _check_fitted(self, X)

        low, high = self.feature_range
        unit = (X - self.data_min_) / self.data_range_     # in [0, 1]
        return low + unit * (high - low)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace missing values with the column mean, median or a constant.

    Args: strategy 'mean' | 'median' | 'constant'; omitted, it is 'constant' when fill_value is
          given and 'mean' otherwise. fill_value for 'constant' and for all-NaN columns; a string
          fill_value (e.g. "Unknown") imputes a categorical array, where "", "nan" and "None" count
          as missing and the output stays a string array.
    Attributes: statistics_ of shape (n_features,).
    Complexity: O(n * d) time (O(n * d log n) for median), O(n * d) space for the output.
    """

    def __init__(self, strategy: str = None, fill_value=None):
        if strategy is None:
            strategy = "mean" if fill_value is None else "constant"
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if isinstance(fill_value, str) and strategy != "constant":
            raise ValueError("a string fill_value requires strategy='constant'")
        self.strategy = strategy
        self.fill_value = 0.0 if fill_value is None else fill_value
        if not isinstance(self.fill_value, str):
            self.fill_value = float(self.fill_value)

    def _as_array(self, X) -> np.ndarray:
        if not isinstance(self.fill_value, str):
            return check_array(X, ndim=2)
        X = np.asarray(X).astype(str)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")
        return X

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        X = self._as_array(X)
        n_features = X.shape[1]

        if self.strategy == "constant":
            self.statistics_ = np.full(n_features, self.fill_value)
        else:
            all_nan = np.isnan(X).all(axis=0)
            X_safe = np.where(all_nan, 0.0, X)      # nanmean warns on an all-NaN column
            if self.strategy == "mean":
                self.statistics_ = np.nanmean(X_safe, axis=0)
            else:
                self.statistics_ = np.nanmedian(X_safe, axis=0)
            self.statistics_[all_nan] = self.fill_value

        self.n_features_in_ = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._as_array(X)
        _check_fitted(self, X)
        missing = np.isin(X, MISSING) if X.dtype.kind == "U" else np.isnan(X)
        return np.where(missing, self.statistics_, X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class Imputer(SimpleImputer):
    """Backward-compatible imputer that accepts a fill value positionally."""

    def __init__(self, fill_value: Optional[Union[float, int, str]] = None):
        if fill_value is None:
            super().__init__()
        else:
            super().__init__(strategy="constant", fill_value=fill_value)


class OneHotEncoder:
    """Encode categorical columns as 0/1 indicator columns, one block per input column.

    Args: handle_unknown 'error' | 'ignore' (an unknown value encodes to a row of zeros).
    Attributes: categories_ list of sorted string arrays, one per column; n_features_out_.
    Returns: transform gives (n_samples, n_features_out_); NaN or blank cells encode to zeros.
    Complexity: fit O(n * d log n), transform O(n * d * c) with c categories per column.
    """

    MISSING = MISSING

    def __init__(self, handle_unknown: str = "error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"unknown handle_unknown {handle_unknown!r}")
        self.handle_unknown = handle_unknown

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        X = np.asarray(X).astype(str)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")

        self.categories_ = []
        for column in X.T:
            present = ~np.isin(column, self.MISSING)
            self.categories_.append(np.unique(column[present]))

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = sum(len(categories) for categories in self.categories_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).astype(str)
        _check_fitted(self, X)

        out = np.zeros((X.shape[0], self.n_features_out_))
        start = 0
        for j, categories in enumerate(self.categories_):
            column = X[:, j]
            block = column[:, None] == categories          # (n, c) by broadcasting
            present = ~np.isin(column, self.MISSING)
            unknown = present & ~block.any(axis=1)
            if unknown.any() and self.handle_unknown == "error":
                raise ValueError(f"unknown categories in column {j}: {np.unique(column[unknown]).tolist()}")

            out[:, start:start + len(categories)] = block
            start += len(categories)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])

    Z = StandardScaler().fit_transform(X)
    assert np.allclose(np.nanmean(Z, axis=0), 0.0)
    assert np.isnan(Z[1, 1])

    Z = MinMaxScaler((-1, 1)).fit_transform(X)
    assert np.allclose(Z[:, 0], [-1, 0, 1])

    Z = SimpleImputer("median").fit_transform(X)
    assert not np.isnan(Z).any()

    assert np.allclose(SimpleImputer(fill_value=0).fit(X).statistics_, 0.0)
    cities = SimpleImputer(fill_value="Unknown").fit_transform([["Perth"], ["nan"], [""]])
    assert cities[:, 0].tolist() == ["Perth", "Unknown", "Unknown"]

    colours = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    Z = OneHotEncoder().fit_transform(colours)
    assert Z.shape == (3, 4) and Z.sum() == 6

    print("preprocessing ok")
