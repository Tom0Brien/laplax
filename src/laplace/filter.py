"""Laplace filter implementations for Bayesian state estimation."""

import numpy as np
from numpy.linalg import inv

from laplace.math import cov_to_sqrt_inv, regularize_covariance, sqrt_inv_to_cov
from laplace.models import LinearProcessModel, NonlinearProcessModel, ObjectiveFunction
from laplace.optim import minimize_bfgs, minimize_trust_region
from laplace.types import (
    Array,
    FilterState,
    Linearization,
    MeasurementLogLik,
    ProcessModel,
)


class LaplaceFilter:
    """
    Laplace filter for nonlinear/non-Gaussian Bayesian state estimation.

    Uses Laplace approximation for the measurement update:
    - Find MAP estimate: μ_{k|k} = argmin V(x)
    - Local covariance: P_{k|k} = [∂²V/∂x²]^{-1} at μ_{k|k}

    where V(x) = 0.5 * |x - μ_{k|k-1}|²_{P_{k|k-1}^{-1}} - log p(y_k | x)
    """

    def __init__(
        self, optimizer: str = "trust_region", max_iter: int = 100, tol: float = 1e-6
    ):
        """
        Initialize Laplace filter.

        Args:
            optimizer: Optimization method ("trust_region" or "bfgs")
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol

    def predict(
        self,
        mean_prev: Array,
        cov_prev: Array,
        process: ProcessModel | LinearProcessModel | NonlinearProcessModel,
    ) -> tuple[Array, Array]:
        """
        Prediction step: compute μ_{k|k-1} and P_{k|k-1}.

        Args:
            mean_prev: Previous posterior mean μ_{k-1|k-1}
            cov_prev: Previous posterior covariance P_{k-1|k-1}
            process: Process model

        Returns:
            mu_pred: Predicted mean μ_{k|k-1}
            P_pred: Predicted covariance P_{k|k-1}
        """
        # Handle different process model types
        if isinstance(process, LinearProcessModel):
            mu_pred = process.predict_mean(mean_prev)
            P_pred = process.predict_cov(cov_prev)
        elif isinstance(process, NonlinearProcessModel):
            mu_pred = process.predict_mean(mean_prev)
            P_pred = process.predict_cov(mean_prev, cov_prev)
        else:
            # Generic protocol: assume zero-mean noise
            mu_pred = process(mean_prev, np.zeros_like(mean_prev))
            # For generic process, use identity (no change in covariance)
            # This is a placeholder - user should provide proper linearization
            P_pred = cov_prev

        return mu_pred, P_pred

    def update(
        self,
        mu_pred: Array,
        P_pred: Array,
        nll: MeasurementLogLik,
        lin: Linearization | None = None,
    ) -> FilterState:
        """
        Update step using Laplace approximation.

        Computes:
        - μ_{k|k} = argmin V(x)
        - P_{k|k} = [∂²V/∂x²]^{-1} at μ_{k|k}

        Args:
            mu_pred: Predicted mean μ_{k|k-1}
            P_pred: Predicted covariance P_{k|k-1}
            nll: Measurement negative log-likelihood
            lin: Optional linearization provider (gradient/Hessian)

        Returns:
            Posterior filter state
        """
        # Regularize predicted covariance
        P_pred = regularize_covariance(P_pred)
        P_inv_pred = inv(P_pred)

        # Create objective function
        objective = ObjectiveFunction(mu_pred, P_inv_pred, nll)

        # Set up optimization functions
        if lin is not None:
            # Use provided linearization
            grad_func = lin.grad
            hess_func = lin.hessian
        else:
            # Use numerical derivatives from objective
            grad_func = objective.gradient
            hess_func = objective.hessian

        # Optimize to find MAP estimate
        if self.optimizer == "trust_region":
            result = minimize_trust_region(
                f=objective,
                grad_f=grad_func,
                hess_f=hess_func,
                x0=mu_pred,
                tol_grad=self.tol,
                max_iter=self.max_iter,
            )
        elif self.optimizer == "bfgs":
            result = minimize_bfgs(
                f=objective,
                grad_f=grad_func,
                x0=mu_pred,
                tol_grad=self.tol,
                max_iter=self.max_iter,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")

        # Compute posterior covariance from Hessian at MAP
        mu_post = result.x
        hess = hess_func(mu_post)
        hess = regularize_covariance(hess)
        P_post = inv(hess)
        P_post = regularize_covariance(P_post)

        return FilterState(mean=mu_post, cov=P_post, sqrt_inv=None)

    def filter_step(
        self,
        state_prev: FilterState,
        process: ProcessModel,
        nll: MeasurementLogLik,
        lin: Linearization | None = None,
    ) -> FilterState:
        """
        Complete filter step (predict + update).

        Args:
            state_prev: Previous filter state
            process: Process model
            nll: Measurement negative log-likelihood
            lin: Optional linearization

        Returns:
            Updated filter state
        """
        mu_pred, P_pred = self.predict(state_prev.mean, state_prev.cov, process)
        return self.update(mu_pred, P_pred, nll, lin)


class SquareRootLaplaceFilter(LaplaceFilter):
    """
    Square-root form of Laplace filter for improved numerical stability.

    Maintains P^{-1} = S S^T instead of P directly.
    """

    def predict_sqrt(
        self,
        mean_prev: Array,
        S_prev: Array,
        process: LinearProcessModel | NonlinearProcessModel,
    ) -> tuple[Array, Array, Array]:
        """
        Prediction step in square-root form.

        Args:
            mean_prev: Previous posterior mean
            S_prev: Previous square-root information S_{k-1|k-1}
            process: Process model

        Returns:
            mu_pred: Predicted mean
            S_pred: Predicted square-root information
            P_pred: Predicted covariance (for convenience)
        """
        # First predict mean
        mu_pred = process.predict_mean(mean_prev)

        # Convert to covariance, predict, then convert back
        P_prev = sqrt_inv_to_cov(S_prev)
        P_pred = (
            process.predict_cov(mean_prev, P_prev)
            if isinstance(process, NonlinearProcessModel)
            else process.predict_cov(P_prev)
        )
        P_pred = regularize_covariance(P_pred)
        S_pred = cov_to_sqrt_inv(P_pred)

        return mu_pred, S_pred, P_pred

    def update_sqrt(
        self,
        mu_pred: Array,
        S_pred: Array,
        nll: MeasurementLogLik,
        lin: Linearization | None = None,
    ) -> FilterState:
        """
        Update step in square-root form.

        Args:
            mu_pred: Predicted mean
            S_pred: Predicted square-root information
            nll: Measurement negative log-likelihood
            lin: Optional linearization

        Returns:
            Posterior filter state with square-root information
        """
        # Convert to covariance for update
        P_pred = sqrt_inv_to_cov(S_pred)

        # Perform standard update
        state = self.update(mu_pred, P_pred, nll, lin)

        # Compute square-root information form
        S_post = cov_to_sqrt_inv(state.cov)
        state.sqrt_inv = S_post

        return state

    def filter_step_sqrt(
        self,
        mean_prev: Array,
        S_prev: Array,
        process: LinearProcessModel | NonlinearProcessModel,
        nll: MeasurementLogLik,
        lin: Linearization | None = None,
    ) -> FilterState:
        """
        Complete filter step in square-root form.

        Args:
            mean_prev: Previous posterior mean
            S_prev: Previous square-root information
            process: Process model
            nll: Measurement negative log-likelihood
            lin: Optional linearization

        Returns:
            Updated filter state with square-root information
        """
        mu_pred, S_pred, _ = self.predict_sqrt(mean_prev, S_prev, process)
        return self.update_sqrt(mu_pred, S_pred, nll, lin)
