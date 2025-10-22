"""Process and measurement model implementations."""

from typing import Callable

import jax
import jax.numpy as jnp

from laplax.math import gradient, hessian, mahalanobis_squared
from laplax.types import Array, MeasurementLogLik, NoiseSpec


class LinearProcessModel:
    """Linear process model: x_k = F x_{k-1} + w_{k-1}."""

    def __init__(self, F: Array, Q: Array):
        """
        Initialize linear process model.

        Args:
            F: State transition matrix
            Q: Process noise covariance
        """
        self.F = F
        self.Q = Q
        self.noise = NoiseSpec(mean=jnp.zeros(Q.shape[0]), cov=Q)

    def __call__(self, x_prev: Array, w: Array) -> Array:
        """Compute x_k = F x_{k-1} + w."""
        return self.F @ x_prev + w

    def predict_mean(self, x_prev: Array) -> Array:
        """Predict mean (with zero noise)."""
        return self.F @ x_prev

    def predict_cov(self, P_prev: Array) -> Array:
        """Predict covariance: P_{k|k-1} = F P_{k-1|k-1} F^T + Q."""
        return self.F @ P_prev @ self.F.T + self.Q


class NonlinearProcessModel:
    """Nonlinear process model: x_k = f(x_{k-1}) + w_{k-1}."""

    def __init__(
        self,
        f: Callable[[Array], Array],
        Q: Array,
        F: Callable[[Array], Array] | None = None,
    ):
        """
        Initialize nonlinear process model.

        Args:
            f: Nonlinear state transition function
            Q: Process noise covariance
            F: Optional Jacobian function; if None, computed numerically
        """
        self.f = f
        self.Q = Q
        self.F_func = F
        self.noise = NoiseSpec(mean=jnp.zeros(Q.shape[0]), cov=Q)

    def __call__(self, x_prev: Array, w: Array) -> Array:
        """Compute x_k = f(x_{k-1}) + w."""
        return self.f(x_prev) + w

    def predict_mean(self, x_prev: Array) -> Array:
        """Predict mean (with zero noise)."""
        return self.f(x_prev)

    def jacobian(self, x: Array) -> Array:
        """Compute Jacobian F = ∂f/∂x at x using JAX autodiff."""
        if self.F_func is not None:
            return self.F_func(x)
        # Use JAX automatic differentiation
        return jax.jacfwd(self.f)(x)

    def predict_cov(self, x_prev: Array, P_prev: Array) -> Array:
        """Predict covariance using linearization: P_{k|k-1} ≈ F P_{k-1|k-1} F^T + Q."""
        F = self.jacobian(x_prev)
        return F @ P_prev @ F.T + self.Q


class GaussianMeasurementModel:
    """Gaussian measurement model: y_k = h(x_k) + v_k, v_k ~ N(0, R)."""

    def __init__(
        self,
        h: Callable[[Array], Array],
        R: Array,
        H: Callable[[Array], Array] | None = None,
    ):
        """
        Initialize Gaussian measurement model.

        Args:
            h: Measurement function
            R: Measurement noise covariance
            H: Optional Jacobian function; if None, computed numerically
        """
        self.h = h
        self.R = R
        self.R_inv = jnp.linalg.inv(R)
        self.H_func = H

    def jacobian(self, x: Array) -> Array:
        """Compute Jacobian H = ∂h/∂x at x using JAX autodiff."""
        if self.H_func is not None:
            return self.H_func(x)
        # Use JAX automatic differentiation
        return jax.jacfwd(self.h)(x)

    def nll(self, y: Array) -> MeasurementLogLik:
        """
        Create negative log-likelihood function for measurement y.

        Returns:
            Function that computes -log p(y | x)
        """

        def _nll(x: Array) -> Array:
            residual = y - self.h(x)
            # Return JAX scalar (0-d array)
            return 0.5 * (residual.T @ self.R_inv @ residual)

        return _nll


class ObjectiveFunction:
    """
    Combined objective V(x) for Laplace update.

    V(x) = 0.5 * |x - μ|²_{P^{-1}} - log p(y | x)
    """

    def __init__(self, mu_pred: Array, P_inv_pred: Array, nll: MeasurementLogLik):
        """
        Initialize objective function.

        Args:
            mu_pred: Predicted mean μ_{k|k-1}
            P_inv_pred: Predicted information matrix P_{k|k-1}^{-1}
            nll: Measurement negative log-likelihood
        """
        self.mu_pred = mu_pred
        self.P_inv_pred = P_inv_pred
        self.nll = nll

    def __call__(self, x: Array) -> float:
        """Compute V(x)."""
        prior_term = 0.5 * mahalanobis_squared(x, self.mu_pred, self.P_inv_pred)
        likelihood_term = self.nll(x)
        return prior_term + likelihood_term

    def gradient(self, x: Array) -> Array:
        """Compute gradient ∂V/∂x."""
        # Prior gradient: P_inv (x - μ)
        prior_grad = self.P_inv_pred @ (x - self.mu_pred)
        # Measurement gradient: automatic differentiation
        meas_grad = gradient(self.nll, x)
        return prior_grad + meas_grad

    def hessian(self, x: Array) -> Array:
        """Compute Hessian ∂²V/∂x²."""
        # Prior Hessian: P_inv
        prior_hess = self.P_inv_pred
        # Measurement Hessian: automatic differentiation
        meas_hess = hessian(self.nll, x)
        return prior_hess + meas_hess


class AnalyticLinearization:
    """Analytic linearization for Gaussian measurement models."""

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
        self.y = y

    def grad(self, x: Array) -> Array:
        """Compute gradient analytically."""
        # Prior: P_inv (x - μ)
        prior_grad = self.objective.P_inv_pred @ (x - self.objective.mu_pred)
        # Measurement: H^T R^{-1} (h(x) - y)
        H = self.meas_model.jacobian(x)
        residual = self.meas_model.h(x) - self.y
        meas_grad = H.T @ self.meas_model.R_inv @ residual
        return prior_grad + meas_grad

    def hessian(self, x: Array) -> Array:
        """Compute Hessian (Gauss-Newton approximation)."""
        # Prior: P_inv
        prior_hess = self.objective.P_inv_pred
        # Measurement (Gauss-Newton): H^T R^{-1} H
        H = self.meas_model.jacobian(x)
        meas_hess = H.T @ self.meas_model.R_inv @ H
        return prior_hess + meas_hess
