"""A small pipeline that runs named steps in order.

Transformers provide fit(X) and transform(X). Estimators provide fit(X, y)
and predict(X). Classes only need these methods; inheritance is not required.
Loops below iterate over steps, not individual samples or array elements.
"""

from typing import Protocol


class Transformer(Protocol):
    """Interface for a preprocessor: learn parameters, then transform data."""

    def fit(self, X): ...

    def transform(self, X): ...


class Estimator(Protocol):
    """Interface for a model; this module does not implement a model."""

    def fit(self, X, y): ...

    def predict(self, X): ...


class Pipeline:
    """Chain a nonempty sequence of (name, component) pairs.

    All steps except the last must provide fit and transform. The last step
    provides fit and either transform (preprocessing) or predict (a model).
    Supplied components are fitted in place, without cloning.

    Method costs are the sum of the called components' costs, plus O(s)
    dispatch for s steps. Working space is determined by the components and
    the largest intermediate arrays; intermediate results are not cached.
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

        X has shape (n, p). Optional y has n targets and is passed only to a
        final estimator providing predict. Intermediate preprocessors receive
        X only. Each intermediate step transforms its output for the next step.
        Errors from components (e.g. invalid shapes) propagate to the caller.
        Time: sum of fits plus intermediate transforms; see class for space.
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

        Raises ValueError before fit or if the final step lacks transform.
        Component errors propagate. Time is the sum of transforms; space is
        determined by their intermediate arrays. Does not refit any step.
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

        y is optional as in fit. Raises ValueError if the last step lacks
        transform; otherwise errors/costs are those of fit plus transform.
        """
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("fit_transform requires a final transformer")
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Transform X (n, p), then return the final estimator's predictions.

        Prediction shape follows the estimator, usually (n,). Raises ValueError
        before fit or when the final step lacks predict; component errors
        propagate. Cost: intermediate transforms plus the final predict call.
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
