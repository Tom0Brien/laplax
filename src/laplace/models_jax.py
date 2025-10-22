"""JAX-enabled process and measurement models with automatic differentiation."""

from typing import Callable

import jax.numpy as jnp
from jax import jacfwd, jit

from laplace.math_jax import (
    gradient_autodiff,
    hessian_autodiff,
    mahalanobis_squared,
)
from laplace.types import Array, MeasurementLogLik, NoiseSpec


class LinearProcessModel:
    """Linear process model: x_k = F x_{k-1} + w_{k-1} (JAX-accelerated)."""

    def __init__(self, F: Array, Q: Array):
        """
        Initialize linear process model.

        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        self.F = jnp.asarray(F)
        self.Q = jnp.asarray(Q)
        self.noise = NoiseSpec(mean=jnp.zeros(Q.shape[0]), cov=jnp.asarray(Q))

    def __call__(self, x_prev: Array, w: Array) -> Array:
        """Compute x_k = F x_{k-1} + w."""
        return self.F @ jnp.asarray(x_prev) + jnp.asarray(w)

    @jit
    def predict_mean(self, x_prev: Array) -> Array:
        """Predict mean (with zero noise)."""
        return self.F @ jnp.asarray(x_prev)

    @jit
    def predict_cov(self, P_prev: Array) -> Array:
        """Predict covariance: P_{k|k-1} = F P_{k-1|k-1} F^T + Q."""
        P = jnp.asarray(P_prev)
        return self.F @ P @ self.F.T + self.Q


class NonlinearProcessModel:
    """
    Nonlinear process model: x_k = f(x_{k-1}) + w_{k-1} (JAX-accelerated).

    Automatically computes Jacobian using JAX autodiff.
    """

    def __init__(self, f: Callable[[Array], Array], Q: Array):
        """
        Initialize nonlinear process model.

        Args:
            f: Nonlinear state transition function (should be JAX-traceable)
            Q: Process noise covariance
        """
        self.f = f
        self.Q = jnp.asarray(Q)
        self.noise = NoiseSpec(mean=jnp.zeros(Q.shape[0]), cov=jnp.asarray(Q))
        # Pre-compile Jacobian function
        self._jacobian = jit(jacfwd(f))

    def __call__(self, x_prev: Array, w: Array) -> Array:
        """Compute x_k = f(x_{k-1}) + w."""
        return self.f(jnp.asarray(x_prev)) + jnp.asarray(w)

    def predict_mean(self, x_prev: Array) -> Array:
        """Predict mean (with zero noise)."""
        return self.f(jnp.asarray(x_prev))

    def jacobian(self, x: Array) -> Array:
        """Compute Jacobian F = ∂f/∂x at x using JAX autodiff."""
        return self._jacobian(jnp.asarray(x))

    def predict_cov(self, x_prev: Array, P_prev: Array) -> Array:
        """Predict covariance using linearization: P_{k|k-1} ≈ F P_{k-1|k-1} F^T + Q."""
        F = self.jacobian(x_prev)
        P = jnp.asarray(P_prev)
        return F @ P @ F.T + self.Q


class GaussianMeasurementModel:
    """
    Gaussian measurement model: y_k = h(x_k) + v_k, v_k ~ N(0, R) (JAX-accelerated).

    Automatically computes Jacobian using JAX autodiff.
    """

    def __init__(self, h: Callable[[Array], Array], R: Array):
        """
        Initialize Gaussian measurement model.

        Args:
            h: Measurement function (should be JAX-traceable)
            R: Measurement noise covariance
        """
        self.h = h
        self.R = jnp.asarray(R)
        self.R_inv = jnp.linalg.inv(self.R)
        # Pre-compile Jacobian function
        self._jacobian = jit(jacfwd(h))

    def jacobian(self, x: Array) -> Array:
        """Compute Jacobian H = ∂h/∂x at x using JAX autodiff."""
        return self._jacobian(jnp.asarray(x))

    def nll(self, y: Array) -> MeasurementLogLik:
        """
        Create negative log-likelihood function for measurement y.

        Returns:
            Function that computes -log p(y | x)
        """
        y_jax = jnp.asarray(y)
        R_inv = self.R_inv

        def _nll(x: Array) -> float:
            x_jax = jnp.asarray(x)
            residual = y_jax - self.h(x_jax)
            # Convert to Python float for optimizer
            result = 0.5 * (residual.T @ R_inv @ residual)
            return float(result)

        return _nll


class ObjectiveFunction:
    """
    Combined objective V(x) for Laplace update (JAX-accelerated).

    V(x) = 0.5 * |x - μ|²_{P^{-1}} - log p(y | x)

    Automatically computes gradient and Hessian using JAX autodiff.
    """

    def __init__(self, mu_pred: Array, P_inv_pred: Array, nll: MeasurementLogLik):
        """
        Initialize objective function.

        Args:
            mu_pred: Predicted mean μ_{k|k-1}
            P_inv_pred: Predicted information matrix P_{k|k-1}^{-1}
            nll: Measurement negative log-likelihood
        """
        self.mu_pred = jnp.asarray(mu_pred)
        self.P_inv_pred = jnp.asarray(P_inv_pred)
        self.nll = nll

        # Pre-compile gradient and Hessian
        self._grad = jit(gradient_autodiff(self._objective_func))
        self._hess = jit(hessian_autodiff(self._objective_func))

    def _objective_func(self, x: Array) -> float:
        """Underlying objective function."""
        x_jax = jnp.asarray(x)
        prior_term = 0.5 * mahalanobis_squared(x_jax, self.mu_pred, self.P_inv_pred)
        likelihood_term = self.nll(x_jax)
        return prior_term + likelihood_term

    def __call__(self, x: Array) -> float:
        """Compute V(x)."""
        return self._objective_func(jnp.asarray(x))

    def gradient(self, x: Array) -> Array:
        """Compute gradient ∂V/∂x using JAX autodiff."""
        return self._grad(jnp.asarray(x))

    def hessian(self, x: Array) -> Array:
        """Compute Hessian ∂²V/∂x² using JAX autodiff."""
        return self._hess(jnp.asarray(x))


class AnalyticLinearization:
    """
    Analytic linearization for Gaussian measurement models (JAX-accelerated).

    Uses Gauss-Newton approximation for the Hessian.
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        meas_model: GaussianMeasurementModel,
        y: Array,
    ):
        """
        Initialize analytic linearization.

        Args:
            objective: Objective function
            meas_model: Gaussian measurement model
            y: Measurement
        """
        self.objective = objective
        self.meas_model = meas_model
        self.y = jnp.asarray(y)

        # Pre-compile gradient and Hessian
        self._grad_fn = jit(self._grad_impl)
        self._hess_fn = jit(self._hess_impl)

    def _grad_impl(self, x: Array) -> Array:
        """Compute gradient analytically."""
        x_jax = jnp.asarray(x)
        # Prior: P_inv (x - μ)
        prior_grad = self.objective.P_inv_pred @ (x_jax - self.objective.mu_pred)
        # Measurement: H^T R^{-1} (h(x) - y)
        H = self.meas_model.jacobian(x_jax)
        residual = self.meas_model.h(x_jax) - self.y
        meas_grad = H.T @ self.meas_model.R_inv @ residual
        return prior_grad + meas_grad

    def _hess_impl(self, x: Array) -> Array:
        """Compute Hessian (Gauss-Newton approximation)."""
        x_jax = jnp.asarray(x)
        # Prior: P_inv
        prior_hess = self.objective.P_inv_pred
        # Measurement (Gauss-Newton): H^T R^{-1} H
        H = self.meas_model.jacobian(x_jax)
        meas_hess = H.T @ self.meas_model.R_inv @ H
        return prior_hess + meas_hess

    def grad(self, x: Array) -> Array:
        """Compute gradient analytically."""
        return self._grad_fn(jnp.asarray(x))

    def hessian(self, x: Array) -> Array:
        """Compute Hessian (Gauss-Newton approximation)."""
        return self._hess_fn(jnp.asarray(x))
