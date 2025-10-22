"""Tests for Laplace filter."""

import numpy as np

from laplace.filter import LaplaceFilter, SquareRootLaplaceFilter
from laplace.models import GaussianMeasurementModel, LinearProcessModel


def test_laplace_filter_linear_gaussian() -> None:
    """Test Laplace filter on linear-Gaussian problem."""
    # Setup: 1D constant velocity model
    F = np.array([[1.0, 1.0], [0.0, 1.0]])  # x_k = F x_{k-1}
    Q = np.eye(2) * 0.01  # Small process noise
    process = LinearProcessModel(F, Q)

    # Measurement: observe position only
    def h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    R = np.array([[0.1]])  # Measurement noise
    meas_model = GaussianMeasurementModel(h, R)

    # Initial state
    x_true = np.array([0.0, 1.0])  # Position 0, velocity 1
    P_init = np.eye(2) * 0.1

    # Filter
    filt = LaplaceFilter(optimizer="trust_region", max_iter=50)

    # Prediction
    mu_pred, P_pred = filt.predict(x_true, P_init, process)
    assert mu_pred.shape == (2,)
    assert P_pred.shape == (2, 2)

    # Simulate measurement
    x_next = F @ x_true
    y = h(x_next) + np.random.randn(1) * 0.01
    nll = meas_model.nll(y)

    # Update
    state = filt.update(mu_pred, P_pred, nll)
    assert state.mean.shape == (2,)
    assert state.cov.shape == (2, 2)

    # Covariance should be positive definite
    eigvals = np.linalg.eigvalsh(state.cov)
    assert np.all(eigvals > 0)


def test_laplace_filter_filter_step() -> None:
    """Test complete filter step."""
    # Simple 1D model
    F = np.array([[1.0]])
    Q = np.array([[0.01]])
    process = LinearProcessModel(F, Q)

    def h(x: np.ndarray) -> np.ndarray:
        return x

    R = np.array([[0.1]])
    meas_model = GaussianMeasurementModel(h, R)

    # Initial state
    from laplace.types import FilterState

    state = FilterState(mean=np.array([0.0]), cov=np.array([[1.0]]))

    # Measurement
    y = np.array([1.0])
    nll = meas_model.nll(y)

    # Filter step
    filt = LaplaceFilter(optimizer="bfgs")
    state_new = filt.filter_step(state, process, nll)

    assert state_new.mean.shape == (1,)
    assert state_new.cov.shape == (1, 1)
    # Mean should move toward measurement
    assert 0.0 < state_new.mean[0] < 1.0


def test_square_root_filter() -> None:
    """Test square-root form of Laplace filter."""
    # 2D linear model
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    Q = np.eye(2) * 0.01
    process = LinearProcessModel(F, Q)

    def h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])

    R = np.array([[0.1]])
    meas_model = GaussianMeasurementModel(h, R)

    # Initial state
    mu = np.array([0.0, 1.0])
    P = np.eye(2) * 0.5

    # Square-root information
    from laplace.math import cov_to_sqrt_inv

    S = cov_to_sqrt_inv(P)

    # Filter
    sqrt_filt = SquareRootLaplaceFilter()

    # Prediction
    mu_pred, S_pred, P_pred = sqrt_filt.predict_sqrt(mu, S, process)
    assert mu_pred.shape == (2,)
    assert S_pred.shape == (2, 2)
    assert P_pred.shape == (2, 2)

    # Check consistency
    from laplace.math import sqrt_inv_to_cov

    P_from_S = sqrt_inv_to_cov(S_pred)
    assert np.allclose(P_pred, P_from_S)

    # Update
    y = np.array([1.0])
    nll = meas_model.nll(y)
    state = sqrt_filt.update_sqrt(mu_pred, S_pred, nll)

    assert state.sqrt_inv is not None
    assert state.sqrt_inv.shape == (2, 2)


def test_laplace_filter_nonlinear_measurement() -> None:
    """Test Laplace filter with nonlinear measurement."""
    # Linear process
    F = np.array([[1.0]])
    Q = np.array([[0.01]])
    process = LinearProcessModel(F, Q)

    # Nonlinear measurement: y = x^2
    def h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] ** 2])

    R = np.array([[0.1]])
    meas_model = GaussianMeasurementModel(h, R)

    # Initial state
    mu = np.array([1.0])
    P = np.array([[0.1]])

    # Filter
    filt = LaplaceFilter(optimizer="trust_region", max_iter=100)

    # Prediction
    mu_pred, P_pred = filt.predict(mu, P, process)

    # Measurement: y = 4 (suggests x ≈ ±2)
    y = np.array([4.0])
    nll = meas_model.nll(y)

    # Update
    state = filt.update(mu_pred, P_pred, nll)

    # Should converge to x ≈ 2 (closer to prediction than x ≈ -2)
    assert state.mean[0] > 0
    assert 1.5 < state.mean[0] < 2.5


def test_filter_convergence() -> None:
    """Test that filter optimization converges."""

    def h(x: np.ndarray) -> np.ndarray:
        return x

    R = np.array([[0.1]])
    meas_model = GaussianMeasurementModel(h, R)

    mu_pred = np.array([0.0])
    P_pred = np.array([[1.0]])
    y = np.array([1.0])
    nll = meas_model.nll(y)

    # Try both optimizers
    for opt in ["trust_region", "bfgs"]:
        filt = LaplaceFilter(optimizer=opt, tol=1e-6)
        state = filt.update(mu_pred, P_pred, nll)

        # Should converge to reasonable value
        assert 0.0 < state.mean[0] < 1.0
        assert state.cov[0, 0] > 0
