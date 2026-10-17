"""Finite-difference gradients/Jacobians and optional line search.

Task allocation: Shuhao.
"""

# TODO (Shuhao): implement finite-difference derivatives and line-search tools.
"""Approximate derivatives by changing one input coordinate at a time.

Forward: (f(x + h) - f(x)) / h.
Central: (f(x + h) - f(x - h)) / (2*h).
The coordinate loops are necessary for ordinary user functions that do not
accept batches of evaluation points. Input arrays are never passed directly
to user functions; each evaluation receives a copy.
"""

import numpy as np


def share_input(x, h, method):
    """Check the input vector, step size, and difference method."""
    if np.iscomplexobj(x):
        raise TypeError("x must contain real numbers")
    
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty, finite, one-dimensional array")
    if np.ndim(h) != 0 or np.iscomplexobj(h):
        raise ValueError("h must be a positive finite scalar")
    
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite scalar")
    if method not in ("central", "forward"):
        raise ValueError("method must be 'central' or 'forward'")

    # A step that rounds back to x cannot measure a derivative.
    with np.errstate(over="ignore"):
        plus = x + h
        minus = x - h if method == "central" else x
    if not np.isfinite(plus).all() or np.any(plus == x):
        raise ValueError("h is too small or too large for x; change h or rescale x")
    if method == "central" and (not np.isfinite(minus).all() or np.any(minus == x)):
        raise ValueError("h is too small or too large for x; change h or rescale x")
    
    return x, h


def share_output(function, x, expected_shape=None):
    """Evaluate a function on a copy, checking real, finite, consistent output."""
    value = function(x.copy())
    if np.iscomplexobj(value):
        raise TypeError("function output must contain real numbers")
    
    value = np.array(value, dtype=float, copy=True)
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError("function output shape must remain unchanged")
    if not np.isfinite(value).all():
        raise ValueError("function output must be finite at all evaluation points")
    
    return value


def grad(f, x, h=1e-5, method="central"):
    """Estimate the gradient of scalar function f at vector x.

    Parameters: f accepts a real array (n,) and returns a scalar; x is nonempty
    and finite; h is a positive finite scalar; method is central or forward.
    Returns a float array (n,), in input-coordinate order, without mutating x.
    Raises ValueError for invalid input/method/step or nonfinite, nonscalar,
    inconsistent output; TypeError for nonnumeric/complex input or output.
    Errors raised inside f propagate. Central uses 2*n+1 evaluations, forward
    n+1 (including a base evaluation for validation). Time O(n*T + n**2), where
    T is one function evaluation; own working space O(n), excluding f's memory.
    """
    x, h = share_input(x, h, method)
    base = share_output(f, x)
    if base.ndim != 0:
        raise ValueError("f must return a scalar; use jacobian for vector output")

    gradient = np.empty(x.size)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += h
        f_plus = share_output(f, x_plus, base.shape)

        if method == "forward":
            gradient[i] = (f_plus - base) / h
        else:
            x_minus = x.copy()
            x_minus[i] -= h
            f_minus = share_output(f, x_minus, base.shape)
            gradient[i] = (f_plus - f_minus) / (2 * h)

    return gradient


def jacobian(F, x, h=1e-5, method="central"):
    """Estimate the Jacobian of vector function F at vector x.

    Parameters: F accepts real x (n,) and returns a nonempty real vector (m,).
    x must be finite, h a positive finite scalar, method central or forward.
    Returns float array (m, n): row j, column i is dF[j]/dx[i]. No input mutation.
    Raises ValueError for invalid inputs/step/method, nonfinite output or
    changing/nonvector output shape; TypeError for nonnumeric/complex values.
    Function errors propagate. Evaluations: 2*n+1 central, n+1 forward.
    Time O(n*T + n**2 + m*n), T = one evaluation; own space O(m*n+n+m).
    """
    x, h = share_input(x, h, method)
    base = share_output(F, x)
    if base.ndim != 1 or base.size == 0:
        raise ValueError("F must return a nonempty one-dimensional vector")

    result = np.empty((base.size, x.size))
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += h
        f_plus = share_output(F, x_plus, base.shape)

        if method == "forward":
            result[:, i] = (f_plus - base) / h
        else:
            x_minus = x.copy()
            x_minus[i] -= h
            f_minus = share_output(F, x_minus, base.shape)
            result[:, i] = (f_plus - f_minus) / (2 * h)

    return result
