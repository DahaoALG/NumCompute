"""Streaming training utilities for chunked model updates."""

from __future__ import annotations

import tracemalloc
from typing import Any

import numpy as np

from .metrics import accuracy


class StreamTrainer:
    """Train and score a model or pipeline incrementally on data chunks.

    Parameters
    ----------
    model : object
        A fitted model or estimator that implements ``fit``/``predict`` and
        optionally ``partial_fit``.
    pipeline : object, optional
        Optional preprocessing pipeline to apply before the model. If supplied,
        the trainer routes training and prediction through the pipeline.
    log_metrics : bool, default True
        If ``True``, keep per-chunk log entries in ``self.history``.
    """

    def __init__(self, model: Any, pipeline: Any | None = None, log_metrics: bool = True):
        self.model = model
        self.pipeline = pipeline
        self.log_metrics = log_metrics
        self.history = []
        self.total_seen = 0
        self.total_correct = 0

    @property
    def estimator(self):
        """Return the active training target: the pipeline if present, else the model."""
        return self.pipeline if self.pipeline is not None else self.model

    def _to_numpy_2d(self, X: Any, name: str = "X") -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array; got shape {X.shape}")
        return X

    def _to_numpy_1d(self, y: Any, name: str = "y") -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 0:
            y = y.reshape(1)
        if y.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got shape {y.shape}")
        return y

    def _memory_mb(self) -> float:
        """Return the peak traced memory in megabytes."""
        _, peak = tracemalloc.get_traced_memory()
        return peak / (1024.0 * 1024.0)

    def _fit_chunk(self, X: np.ndarray, y: np.ndarray) -> None:
        estimator = self.estimator

        if hasattr(estimator, "partial_fit"):
            estimator.partial_fit(X, y)
        elif hasattr(estimator, "fit"):
            estimator.fit(X, y)
        else:
            raise AttributeError(f"{type(estimator).__name__} does not support fit or partial_fit")

    def _predict_chunk(self, X: np.ndarray) -> np.ndarray:
        estimator = self.estimator
        if hasattr(estimator, "predict"):
            y_pred = estimator.predict(X)
            return np.asarray(y_pred)
        raise AttributeError(f"{type(estimator).__name__} does not provide predict")

    def fit_chunk(self, X: Any, y: Any):
        """Fit the active estimator on a single chunk of data.

        The trainer prefers ``partial_fit`` when available; otherwise it falls
        back to a full ``fit`` call for the chunk.
        """
        X = self._to_numpy_2d(X, "X")
        y = self._to_numpy_1d(y, "y")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows; got {X.shape[0]} and {y.shape[0]}")

        tracemalloc.start()
        try:
            self._fit_chunk(X, y)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        if self.log_metrics:
            self.history.append(
                {
                    "chunk_index": len(self.history),
                    "samples": X.shape[0],
                    "memory_mb": self._memory_mb(),
                    "peak_memory_mb": peak / (1024.0 * 1024.0),
                }
            )

        return self

    def score_chunk(self, X: Any, y: Any) -> float:
        """Predict on one chunk and report its chunk accuracy.

        Returns
        -------
        float
            Accuracy for the supplied chunk.
        """
        X = self._to_numpy_2d(X, "X")
        y = self._to_numpy_1d(y, "y")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have the same number of rows; got {X.shape[0]} and {y.shape[0]}")

        y_pred = self._predict_chunk(X)
        if y_pred.shape[0] != y.shape[0]:
            raise ValueError(f"Prediction length mismatch: expected {y.shape[0]}, got {y_pred.shape[0]}")

        chunk_acc = float(accuracy(y, y_pred))
        self.total_seen += int(y.shape[0])
        self.total_correct += int(np.sum(y == y_pred))
        cumulative_accuracy = self.total_correct / self.total_seen if self.total_seen else 0.0

        if self.log_metrics:
            entry = {
                "chunk_index": len(self.history),
                "samples": y.shape[0],
                "accuracy": chunk_acc,
                "cumulative_accuracy": float(cumulative_accuracy),
                "memory_mb": self._memory_mb(),
            }
            self.history.append(entry)

        return chunk_acc

    @property
    def cumulative_accuracy(self) -> float:
        """Return the total accuracy seen so far across all scored chunks."""
        if self.total_seen == 0:
            return 0.0
        return self.total_correct / self.total_seen

    def get_history(self):
        """Return a shallow copy of the accumulated log entries."""
        return list(self.history)

    def reset(self):
        """Clear chunk log history and cumulative counters."""
        self.history.clear()
        self.total_seen = 0
        self.total_correct = 0
        return self
