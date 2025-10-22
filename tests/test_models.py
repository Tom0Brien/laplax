"""Tests for process and measurement models."""

import jax.numpy as jnp
import numpy as np  # For allclose compatibility

from laplax.models import (
    AnalyticLinearization,
    GaussianMeasurementModel,
    LinearProcessModel,
    NonlinearProcessModel,
    ObjectiveFunction,
)


def test_linear_process_model() -> None:
    """Test linear process model."""
    F = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    Q = jnp.eye(2) * 0.01

    model = LinearProcessModel(F, Q)

    x_prev = jnp.array([1.0, 0.5])
    w = jnp.array([0.01, 0.02])

    x_next = model(x_prev, w)
    expected = F @ x_prev + w
    assert np.allclose(x_next, expected)

    # Test prediction
    mu_pred = model.predict_mean(x_prev)
    assert np.allclose(mu_pred, F @ x_prev)

    P_prev = jnp.eye(2) * 0.1
    P_pred = model.predict_cov(P_prev)
    expected_P = F @ P_prev @ F.T + Q
    assert np.allclose(P_pred, expected_P)


def test_nonlinear_process_model() -> None:
    """Test nonlinear process model."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] + 0.1 * x[1], x[1]])

    Q = jnp.eye(2) * 0.01
    model = NonlinearProcessModel(f, Q)

    x_prev = jnp.array([1.0, 0.5])
    w = jnp.array([0.01, 0.02])

    x_next = model(x_prev, w)
    expected = f(x_prev) + w
    assert np.allclose(x_next, expected)

    # Test Jacobian
    F = model.jacobian(x_prev)
    assert F.shape == (2, 2)

    # Numerical check: F[0,1] should be approximately 0.1
    assert np.isclose(F[0, 1], 0.1, atol=1e-5)


def test_gaussian_measurement_model() -> None:
    """Test Gaussian measurement model."""

    def h(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] ** 2, x[1]])

    R = jnp.eye(2) * 0.1
    model = GaussianMeasurementModel(h, R)

    x = jnp.array([2.0, 1.0])
    y = jnp.array([4.1, 1.05])

    # Test NLL
    nll = model.nll(y)
    nll_val = nll(x)
    assert nll_val >= 0

    # Perfect measurement should give low NLL
    y_perfect = h(x)
    nll_perfect = model.nll(y_perfect)
    assert nll_perfect(x) < 1e-10

    # Test Jacobian
    H = model.jacobian(x)
    assert H.shape == (2, 2)
    # ∂h0/∂x0 = 2*x0 = 4
    assert np.isclose(H[0, 0], 4.0, atol=1e-5)


def test_objective_function() -> None:
    """Test objective function for Laplace update."""
    mu_pred = jnp.array([0.0, 0.0])
    P_inv_pred = jnp.eye(2)

    def nll(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(x**2)

    obj = ObjectiveFunction(mu_pred, P_inv_pred, nll)

    # At mu_pred, prior term is zero
    val = obj(mu_pred)
    assert np.isclose(val, nll(mu_pred))

    # Test gradient
    x = jnp.array([1.0, 1.0])
    grad = obj.gradient(x)
    assert grad.shape == (2,)

    # Test Hessian
    hess = obj.hessian(x)
    assert hess.shape == (2, 2)
    assert np.allclose(hess, hess.T)  # Symmetric


def test_analytic_linearization() -> None:
    """Test analytic linearization for Gaussian measurements."""

    def h(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] + x[1], x[0] - x[1]])

    R = jnp.eye(2) * 0.1
    meas_model = GaussianMeasurementModel(h, R)

    mu_pred = jnp.array([0.0, 0.0])
    P_inv_pred = jnp.eye(2)
    y = jnp.array([1.0, 0.0])

    nll = meas_model.nll(y)
    obj = ObjectiveFunction(mu_pred, P_inv_pred, nll)

    lin = AnalyticLinearization(obj, meas_model, y)

    x = jnp.array([0.5, 0.5])

    # Test gradient
    grad = lin.grad(x)
    assert grad.shape == (2,)

    # Test Hessian
    hess = lin.hessian(x)
    assert hess.shape == (2, 2)
    assert np.allclose(hess, hess.T)  # Symmetric
