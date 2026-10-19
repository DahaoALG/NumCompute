"""Unit tests for preprocessing chains and the estimator interface."""

import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder


@pytest.fixture
def X():
    return np.random.default_rng(7).normal(size=(100, 3))


@pytest.fixture
def X_nan(X):
    result = X.copy()
    result[[0, 5, 9], [0, 1, 1]] = np.nan
    return result


@pytest.fixture
def numeric_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", StandardScaler()),
    ])


class RecordingEstimator:
    """A small test double for fit/predict; no ML model is trained."""

    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y):
        self.received_X = X.copy()
        self.received_y = y.copy()
        self.fit_calls += 1
        return self

    def predict(self, X):
        return X.sum(axis=1)


# preprocessing pipelines

def test_fit_transform_matches_numpy(X_nan, numeric_pipe):
    result = numeric_pipe.fit_transform(X_nan)
    filled = np.where(np.isnan(X_nan), 0, X_nan)
    expected = (filled - filled.mean(axis=0)) / filled.std(axis=0)
    np.testing.assert_allclose(result, expected)
    assert result.shape == X_nan.shape


def test_fit_returns_pipeline_and_transform_matches_fit_transform(X, numeric_pipe):
    assert numeric_pipe.fit(X) is numeric_pipe
    result = numeric_pipe.transform(X)
    expected = Pipeline([("scale", StandardScaler())]).fit_transform(X)
    np.testing.assert_allclose(result, expected)


def test_new_data_reuses_training_parameters():
    scaler = StandardScaler()
    pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)), ("scale", scaler)])
    pipe.fit([[0.0], [2.0]])
    result = pipe.transform([[4.0], [np.nan]])
    np.testing.assert_allclose(result, [[3.0], [-1.0]])
    np.testing.assert_array_equal(scaler.mean_, [1.0])
    np.testing.assert_array_equal(scaler.scale_, [1.0])


def test_input_is_not_modified(X_nan, numeric_pipe):
    original = X_nan.copy()
    numeric_pipe.fit_transform(X_nan)
    numeric_pipe.transform(X_nan)
    np.testing.assert_array_equal(X_nan, original)


def test_non_contiguous_input(X, numeric_pipe):
    view = X[::3]
    assert not view.flags.c_contiguous
    expected = (view - view.mean(axis=0)) / view.std(axis=0)
    np.testing.assert_allclose(numeric_pipe.fit_transform(view), expected)


def test_categorical_pipeline_and_unknown_category():
    X = np.array([["Adelaide"], [np.nan], ["Sydney"]], dtype=object)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    result = pipe.fit_transform(X)
    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    np.testing.assert_array_equal(pipe.transform([["Perth"]]), [[0, 0, 0]])


def test_single_step_pipeline():
    pipe = Pipeline([("scale", StandardScaler())])
    np.testing.assert_allclose(pipe.fit_transform([[0.0], [2.0]]), [[-1.0], [1.0]])


# estimator interface

def test_estimator_receives_transformed_data_and_labels():
    estimator = RecordingEstimator()
    pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
    X = np.array([[0.0, 10.0], [2.0, 20.0]])
    y = np.array([0, 1])
    assert pipe.fit(X, y) is pipe
    np.testing.assert_allclose(estimator.received_X, [[-1, -1], [1, 1]])
    np.testing.assert_array_equal(estimator.received_y, y)
    result = pipe.predict([[4.0, 30.0]])
    assert result.shape == (1,)
    assert result[0] == pytest.approx(6.0)
    assert estimator.fit_calls == 1


# invalid operations and inputs

def test_transform_and_predict_require_fit(numeric_pipe):
    with pytest.raises(ValueError, match="fit"):
        numeric_pipe.transform([[1, 2, 3]])
    with pytest.raises(ValueError, match="fit"):
        numeric_pipe.predict([[1, 2, 3]])


@pytest.mark.parametrize("steps, message", [
    ([], "at least one"),
    ([("", SimpleImputer())], "names"),
    ([("same", SimpleImputer()), ("same", StandardScaler())], "names"),
    ([("bad", object())], "fit"),
    ([("model", RecordingEstimator()), ("scale", StandardScaler())], "Intermediate"),
])
def test_invalid_steps_rejected(steps, message):
    with pytest.raises(ValueError, match=message):
        Pipeline(steps)


def test_transformer_pipeline_cannot_predict(numeric_pipe, X):
    numeric_pipe.fit(X)
    with pytest.raises(ValueError, match="predict"):
        numeric_pipe.predict(X)


def test_model_pipeline_cannot_transform():
    pipe = Pipeline([("model", RecordingEstimator())])
    X = np.array([[1.0], [2.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="transformer"):
        pipe.fit_transform(X, y)
    pipe.fit(X, y)
    with pytest.raises(ValueError, match="transform"):
        pipe.transform(X)


def test_failed_refit_clears_fitted_state(numeric_pipe, X):
    numeric_pipe.fit(X)
    with pytest.raises(ValueError):
        numeric_pipe.fit([])
    with pytest.raises(ValueError, match="fit"):
        numeric_pipe.transform(X)


def test_feature_count_mismatch_rejected(numeric_pipe, X):
    numeric_pipe.fit(X)
    with pytest.raises(ValueError, match="features"):
        numeric_pipe.transform(X[:, :2])


def test_categorical_pipeline_rejects_unknown_when_configured():
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="error")),
    ])
    pipe.fit(np.array([["Adelaide"], [np.nan], ["Sydney"]], dtype=object))
    with pytest.raises(ValueError, match="unknown categories"):
        pipe.transform([["Perth"]])