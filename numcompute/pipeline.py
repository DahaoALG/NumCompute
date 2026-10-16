"""Transformer/Estimator API, Compose, and FeatureUnion.

Task allocation: Shuhao.
"""

from typing import Protocol


class Transformer(Protocol):

    def fit(self, X): ...

    def transform(self, X): ...


class Estimator(Protocol):
    """Interface for a model; this module does not implement a model."""

    def fit(self, X, y): ...

    def predict(self, X): ...


class Pipeline:
    """Chain a nonempty sequence of (name, component) pairs.
    """

    def __init__(self, steps):
        """Store named steps; raise ValueError for empty or invalid steps.

        Parameters: steps is a sequence of (unique nonempty string, component).
        Returns a Pipeline instance. Time/space: O(s), s = number of steps.
        """
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("Pipeline must contain at least one step")

        names = set()
        
        for name, step in self.steps:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Step names must be unique nonempty strings")
            if not callable(getattr(step, "fit", None)):
                raise ValueError(f"Step '{name}' must provide fit")
            names.add(name)

        for name, step in self.steps[:-1]:
            if not callable(getattr(step, "transform", None)):
                raise ValueError(f"Intermediate step '{name}' must provide transform")

        last = self.steps[-1][1]
        if not callable(getattr(last, "transform", None)) and not callable(getattr(last, "predict", None)):
            raise ValueError("The last step must provide transform or predict")
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Fit each step in order and return self.
        """
        self.is_fitted_ = False
        X_out = X
        for name, step in self.steps[:-1]:
            step.fit(X_out)
            X_out = step.transform(X_out)

        last = self.steps[-1][1]
        if y is not None and callable(getattr(last, "predict", None)):
            last.fit(X_out, y)
        else:
            last.fit(X_out)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform X (n, p) through every fitted step; return X_out (n, q).
        """
        if not self.is_fitted_:
            raise ValueError("Call fit before transform")
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("The last step does not provide transform; use predict")

        X_out = X
        for name, step in self.steps:
            X_out = step.transform(X_out)
        return X_out

    def fit_transform(self, X, y=None):
        """Fit X (n, p), then return transformed X_out (n, q).
        """
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("fit_transform requires a final transformer")
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Transform X (n, p), then return the final estimator's predictions.
        """
        if not self.is_fitted_:
            raise ValueError("Call fit before predict")
        last = self.steps[-1][1]
        if not callable(getattr(last, "predict", None)):
            raise ValueError("The last step does not provide predict; use transform")

        X_out = X
        for name, step in self.steps[:-1]:
            X_out = step.transform(X_out)
        return last.predict(X_out)
