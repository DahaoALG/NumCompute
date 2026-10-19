"""Unit tests for finite-difference gradients and Jacobians."""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian


@pytest.fixture
def x():
    return np.random.default_rng(7).normal(size=3)


# gradients

@pytest.mark.parametrize("method, atol", [("central", 1e-8), ("forward", 2e-5)])
def test_grad_matches_analytical_derivative(x, method, atol):
    def f(x):
        return np.sum(x ** 2)

    result = grad(f, x, method=method)
    np.testing.assert_allclose(result, 2 * x, rtol=0, atol=atol)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


def test_grad_uses_custom_step():
    def f(x):
        return np.sin(x[0])

    result = grad(f, [0.5], h=1e-4)
    assert result[0] == pytest.approx(np.cos(0.5), abs=1e-8)


@pytest.mark.parametrize("method", ["central", "forward"])
def test_constant_functions_have_zero_derivatives(method):
    np.testing.assert_array_equal(grad(lambda x: 7.0, [0, 0], method=method), [0, 0])
    result = jacobian(lambda x: [1.0, 2.0, 3.0], [0, 0], method=method)
    np.testing.assert_array_equal(result, np.zeros((3, 2)))


# Jacobians

@pytest.mark.parametrize("method, atol", [("central", 1e-8), ("forward", 2e-5)])
def test_jacobian_matches_analytical_derivative(method, atol):
    def F(x):
        return np.array([x[0] ** 2 + x[1], x[0] * x[1]])

    result = jacobian(F, [2.0, 3.0], method=method)
    np.testing.assert_allclose(result, [[4, 1], [3, 2]], rtol=0, atol=atol)
    assert result.shape == (2, 2)


@pytest.mark.parametrize("method, atol", [("central", 1e-8), ("forward", 2e-5)])
def test_rectangular_jacobian_shape_and_orientation(method, atol):
    def F(x):
        return np.array([x[0] + x[1], x[0] ** 2, x[0] * x[1]])

    result = jacobian(F, [2.0, 3.0], method=method)
    assert result.shape == (3, 2)  # output components x input coordinates
    np.testing.assert_allclose(result, [[1, 1], [4, 0], [3, 2]], rtol=0, atol=atol)


# input preservation and errors

def test_non_contiguous_input_gives_same_result_and_is_not_modified():
    x = np.arange(6.0)[::2]
    original = x.copy()
    assert not x.flags.c_contiguous
    np.testing.assert_allclose(grad(lambda x: np.sum(x ** 2), x), 2 * x, atol=1e-8)
    np.testing.assert_allclose(jacobian(lambda x: x ** 2, x), np.diag(2 * x), atol=1e-8)
    np.testing.assert_array_equal(x, original)


def test_wrong_function_output_dimensions_rejected():
    with pytest.raises(ValueError, match="scalar"):
        grad(lambda x: [x[0]], [1.0])
    with pytest.raises(ValueError, match="vector"):
        jacobian(lambda x: x[0], [1.0])
    with pytest.raises(ValueError, match="vector"):
        jacobian(lambda x: [], [1.0])


@pytest.mark.parametrize("bad_x", [[], [[1, 2]], [np.nan], [np.inf]])
def test_invalid_input_vectors_rejected(bad_x):
    with pytest.raises(ValueError, match="one-dimensional"):
        grad(lambda x: np.sum(x), bad_x)
    with pytest.raises(ValueError, match="one-dimensional"):
        jacobian(lambda x: x, bad_x)


@pytest.mark.parametrize("h", [0, -1, np.nan, np.inf, [1e-5]])
def test_invalid_step_sizes_rejected(x, h):
    with pytest.raises(ValueError, match="h must"):
        grad(lambda x: np.sum(x), x, h=h)
    with pytest.raises(ValueError, match="h must"):
        jacobian(lambda x: x, x, h=h)


def test_unknown_method_rejected(x):
    with pytest.raises(ValueError, match="method"):
        grad(lambda x: np.sum(x), x, method="backward")
    with pytest.raises(ValueError, match="method"):
        jacobian(lambda x: x, x, method="backward")


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_function_outputs_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        grad(lambda x: value, [1.0])
    with pytest.raises(ValueError, match="finite"):
        jacobian(lambda x: [value], [1.0])


def test_changing_output_shape_rejected():
    def F(x):
        return [1, 2] if x[0] > 0 else [1]

    with pytest.raises(ValueError, match="shape"):
        jacobian(F, [0.0])


def test_complex_inputs_and_outputs_rejected():
    with pytest.raises(TypeError, match="real"):
        grad(lambda x: np.sum(x), [1j])
    with pytest.raises(TypeError, match="real"):
        jacobian(lambda x: x, [1j])
    with pytest.raises(TypeError, match="real"):
        grad(lambda x: 1j, [1.0])
    with pytest.raises(TypeError, match="real"):
        jacobian(lambda x: [1j], [1.0])


def test_step_lost_to_rounding_rejected():
    with pytest.raises(ValueError, match="too small"):
        grad(lambda x: x[0], [1e20])
    with pytest.raises(ValueError, match="too small"):
        jacobian(lambda x: x, [1e20])


def test_errors_inside_user_function_propagate():
    def invalid_function(x):
        raise RuntimeError("function failed")

    with pytest.raises(RuntimeError, match="function failed"):
        grad(invalid_function, [1.0])
    with pytest.raises(RuntimeError, match="function failed"):
        jacobian(invalid_function, [1.0])
