"""Finite-difference gradients and Jacobians.

Task allocation: Shuhao.

Approximate derivatives by changing one input coordinate at a time.

Forward: (f(x + h) - f(x)) / h.
Central: (f(x + h) - f(x - h)) / (2*h).
The coordinate loops are necessary for ordinary user functions that do not
accept batches of evaluation points. Input arrays are never passed directly
to user functions; each evaluation receives a copy.
"""

import numpy as np

__all__ = ["grad", "jacobian"]


def _check_input(x, h, method):
    """Validate a finite-difference input vector and perturbation.

    Parameters
    ----------
    x : array_like, shape (n,)
        Nonempty, finite real vector.
    h : real scalar
        Positive finite step that changes every coordinate in floating point.
    method : {'central', 'forward'}
        Difference formula; central checks both perturbation directions.

    Returns
    -------
    tuple of (ndarray, float)
        Converted floating-point x of shape (n,) and validated step h.

    Raises
    ------
    TypeError
        If x is complex or values cannot be converted to real numbers.
    ValueError
        If shape, finiteness, step or method is invalid, or conversion fails.

    Notes
    -----
    Time and space: O(n) for validation and perturbation arrays.
    """
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


def _evaluate(function, x, expected_shape=None):
    """Evaluate a callable on a copy and check its numeric output.

    Parameters
    ----------
    function : callable
        Called with a copy of x.
    x : ndarray, shape (n,)
        Evaluation point.
    expected_shape : tuple of int or None, default None
        Required output shape; None skips the shape check.

    Returns
    -------
    ndarray of float
        Copied result with the callable's shape; scalar output has shape ().

    Raises
    ------
    TypeError
        If output is complex or cannot be converted to real numbers.
    ValueError
        If conversion fails, output is nonfinite or its shape differs.
    Exception
        Exceptions raised by function propagate.

    Notes
    -----
    Time: O(T + n + m); space O(n + m) excluding callable internals,
    where T is evaluation cost and m the output element count.
    """
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
    """Estimate the gradient of a scalar function.

    Parameters
    ----------
    f : callable
        Accepts a real vector of shape (n,) and returns a real scalar.
    x : array_like, shape (n,)
        Nonempty finite real evaluation point; the input is not modified.
    h : float, default 1e-5
        Positive finite absolute step that changes each coordinate in
        floating-point arithmetic.
    method : {'central', 'forward'}, default 'central'
        Finite-difference formula.

    Returns
    -------
    ndarray of float, shape (n,)
        Approximate partial derivatives in input-coordinate order.

    Raises
    ------
    TypeError
        If inputs or function outputs are complex or cannot be converted to
        real numbers.
    ValueError
        If input shape, finiteness, step or method is invalid, numeric
        conversion fails, or output has a wrong/changing shape or is nonfinite.
    Exception
        Exceptions from the callable propagate.

    Notes
    -----
    Time: O(n*T + n**2); own working space O(n), excluding callable
    allocations, with T the cost of one evaluation.
    Central differences use 2*n+1 evaluations; forward differences use
    n+1. Both include a base evaluation to validate the output.
    """
    x, h = _check_input(x, h, method)
    base = _evaluate(f, x)
    if base.ndim != 0:
        raise ValueError("f must return a scalar; use jacobian for vector output")

    gradient = np.empty(x.size)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += h
        f_plus = _evaluate(f, x_plus, base.shape)

        if method == "forward":
            gradient[i] = (f_plus - base) / h
        else:
            x_minus = x.copy()
            x_minus[i] -= h
            f_minus = _evaluate(f, x_minus, base.shape)
            gradient[i] = (f_plus - f_minus) / (2 * h)

    return gradient


def jacobian(F, x, h=1e-5, method="central"):
    """Estimate the Jacobian of a vector-valued function.

    Parameters
    ----------
    F : callable
        Accepts a real vector of shape (n,) and returns a nonempty real vector of shape (m,).
    x : array_like, shape (n,)
        Nonempty finite real evaluation point; the input is not modified.
    h : float, default 1e-5
        Positive finite absolute step that changes each coordinate in
        floating-point arithmetic.
    method : {'central', 'forward'}, default 'central'
        Finite-difference formula.

    Returns
    -------
    ndarray of float, shape (m, n)
        Jacobian with entry [j, i] approximating dF[j]/dx[i].

    Raises
    ------
    TypeError
        If inputs or function outputs are complex or cannot be converted to
        real numbers.
    ValueError
        If input shape, finiteness, step or method is invalid, numeric
        conversion fails, or output has a wrong/changing shape or is nonfinite.
    Exception
        Exceptions from the callable propagate.

    Notes
    -----
    Time: O(n*T + n**2 + m*n); own space O(m*n + n + m), excluding
    callable allocations, with T the cost of one evaluation.
    Central differences use 2*n+1 evaluations; forward differences use
    n+1. Both include a base evaluation to validate the output.
    """
    x, h = _check_input(x, h, method)
    base = _evaluate(F, x)
    if base.ndim != 1 or base.size == 0:
        raise ValueError("F must return a nonempty one-dimensional vector")

    result = np.empty((base.size, x.size))
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += h
        f_plus = _evaluate(F, x_plus, base.shape)

        if method == "forward":
            result[:, i] = (f_plus - base) / h
        else:
            x_minus = x.copy()
            x_minus[i] -= h
            f_minus = _evaluate(F, x_minus, base.shape)
            result[:, i] = (f_plus - f_minus) / (2 * h)

    return result
