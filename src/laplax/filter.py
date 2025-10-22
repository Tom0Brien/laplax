"""
Filter implementations for Bayesian state estimation.

Includes:
- LaplaceFilter: Nonlinear/non-Gaussian filtering via Laplace approximation
- SquareRootLaplaceFilter: Numerically stable square-root form
- KalmanFilter: Standard linear-Gaussian optimal filter
- ExtendedKalmanFilter: First-order linearization for nonlinear systems
- UnscentedKalmanFilter: Unscented transform for nonlinear systems
"""

import jax.numpy as jnp
from jax.numpy.linalg import inv

from laplax.math import cov_to_sqrt_inv, regularize_covariance, sqrt_inv_to_cov
from laplax.models import LinearProcessModel, NonlinearProcessModel, ObjectiveFunction
from laplax.optim import minimize_bfgs, minimize_trust_region
from laplax.types import (
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
            mu_pred = process(mean_prev, jnp.zeros_like(mean_prev))
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


class KalmanFilter:
    """
    Standard Kalman Filter for linear-Gaussian systems.

    Optimal for linear process and measurement models with Gaussian noise.
    Provides the minimum mean squared error (MMSE) estimate under these assumptions.
    """

    def __init__(self):
        """Initialize Kalman filter."""
        pass

    def predict(
        self,
        mean_prev: Array,
        cov_prev: Array,
        process: LinearProcessModel,
    ) -> tuple[Array, Array]:
        """
        Prediction step: compute μ_{k|k-1} and P_{k|k-1}.

        Args:
            mean_prev: Previous posterior mean μ_{k-1|k-1}
            cov_prev: Previous posterior covariance P_{k-1|k-1}
            process: Linear process model

        Returns:
            mu_pred: Predicted mean μ_{k|k-1}
            P_pred: Predicted covariance P_{k|k-1}
        """
        F = process.F
        Q = process.Q

        # Prediction equations
        mu_pred = F @ mean_prev
        P_pred = F @ cov_prev @ F.T + Q

        return mu_pred, P_pred

    def update(
        self,
        mu_pred: Array,
        P_pred: Array,
        y: Array,
        H: Array,
        R: Array,
    ) -> FilterState:
        """
        Update step using Kalman gain.

        Args:
            mu_pred: Predicted mean μ_{k|k-1}
            P_pred: Predicted covariance P_{k|k-1}
            y: Measurement
            H: Measurement Jacobian
            R: Measurement noise covariance

        Returns:
            Posterior filter state
        """
        # Innovation
        innovation = y - H @ mu_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T @ inv(S)

        # Update equations
        mu_post = mu_pred + K @ innovation
        P_post = (jnp.eye(len(mu_pred)) - K @ H) @ P_pred

        return FilterState(mean=mu_post, cov=P_post)

    def filter_step(
        self,
        state_prev: FilterState,
        process: LinearProcessModel,
        y: Array,
        H: Array,
        R: Array,
    ) -> FilterState:
        """
        Complete filter step (predict + update).

        Args:
            state_prev: Previous filter state
            process: Process model
            y: Measurement
            H: Measurement Jacobian
            R: Measurement noise covariance

        Returns:
            Updated filter state
        """
        mu_pred, P_pred = self.predict(state_prev.mean, state_prev.cov, process)
        return self.update(mu_pred, P_pred, y, H, R)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.

    Uses first-order Taylor series linearization around the current estimate.
    Good for mildly nonlinear systems where linearization is accurate.
    """

    def __init__(self):
        """Initialize Extended Kalman filter."""
        pass

    def predict(
        self,
        mean_prev: Array,
        cov_prev: Array,
        f: callable,
        F: Array,
        Q: Array,
    ) -> tuple[Array, Array]:
        """
        Prediction step with nonlinear process model.

        Args:
            mean_prev: Previous posterior mean
            cov_prev: Previous posterior covariance
            f: Nonlinear process function
            F: Process Jacobian at mean_prev
            Q: Process noise covariance

        Returns:
            mu_pred: Predicted mean
            P_pred: Predicted covariance
        """
        # Nonlinear prediction
        mu_pred = f(mean_prev)

        # Linearized covariance propagation
        P_pred = F @ cov_prev @ F.T + Q

        return mu_pred, P_pred

    def update(
        self,
        mu_pred: Array,
        P_pred: Array,
        y: Array,
        h: callable,
        H: Array,
        R: Array,
    ) -> FilterState:
        """
        Update step with nonlinear measurement model.

        Args:
            mu_pred: Predicted mean
            P_pred: Predicted covariance
            y: Measurement
            h: Nonlinear measurement function
            H: Measurement Jacobian at mu_pred
            R: Measurement noise covariance

        Returns:
            Posterior filter state
        """
        # Innovation with nonlinear measurement
        innovation = y - h(mu_pred)

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = P_pred @ H.T @ inv(S)

        # Update equations
        mu_post = mu_pred + K @ innovation
        P_post = (jnp.eye(len(mu_pred)) - K @ H) @ P_pred

        return FilterState(mean=mu_post, cov=P_post)


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear systems.

    Uses unscented transform to capture mean and covariance through nonlinear
    transformations more accurately than EKF (up to 3rd order for Gaussian).
    No Jacobians required - uses sigma points instead.
    """

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """
        Initialize Unscented Kalman filter.

        Args:
            alpha: Spread of sigma points (typically 1e-4 to 1)
            beta: Prior knowledge parameter (2 is optimal for Gaussian)
            kappa: Secondary scaling parameter (typically 0 or 3-n)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _compute_sigma_points(self, mean: Array, cov: Array) -> tuple[Array, Array]:
        """
        Compute sigma points and weights for unscented transform.

        Args:
            mean: Mean vector (n,)
            cov: Covariance matrix (n, n)

        Returns:
            sigma_points: Sigma points (2n+1, n)
            weights_mean: Weights for mean (2n+1,)
            weights_cov: Weights for covariance (2n+1,)
        """
        n = len(mean)

        # Scaling parameter
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        # Compute matrix square root
        L = jnp.linalg.cholesky((n + lambda_) * cov)

        # Generate sigma points
        sigma_points = jnp.zeros((2 * n + 1, n))
        sigma_points[0] = mean

        for i in range(n):
            sigma_points[i + 1] = mean + L[:, i]
            sigma_points[n + i + 1] = mean - L[:, i]

        # Compute weights
        weights_mean = jnp.zeros(2 * n + 1)
        weights_cov = jnp.zeros(2 * n + 1)

        weights_mean[0] = lambda_ / (n + lambda_)
        weights_cov[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)

        for i in range(1, 2 * n + 1):
            weights_mean[i] = 1 / (2 * (n + lambda_))
            weights_cov[i] = 1 / (2 * (n + lambda_))

        return sigma_points, weights_mean, weights_cov

    def _unscented_transform(
        self,
        sigma_points: Array,
        weights_mean: Array,
        weights_cov: Array,
        f: callable,
        noise_cov: Array = None,
    ) -> tuple[Array, Array]:
        """
        Apply unscented transform through nonlinear function.

        Args:
            sigma_points: Input sigma points (2n+1, n)
            weights_mean: Weights for mean
            weights_cov: Weights for covariance
            f: Nonlinear function
            noise_cov: Additive noise covariance (optional)

        Returns:
            mean: Transformed mean
            cov: Transformed covariance
        """
        # Transform sigma points
        n_points = sigma_points.shape[0]
        y_dim = len(f(sigma_points[0]))
        transformed = jnp.zeros((n_points, y_dim))

        for i in range(n_points):
            transformed[i] = f(sigma_points[i])

        # Compute mean
        mean = jnp.sum(weights_mean[:, jnp.newaxis] * transformed, axis=0)

        # Compute covariance
        cov = jnp.zeros((y_dim, y_dim))
        for i in range(n_points):
            diff = transformed[i] - mean
            cov += weights_cov[i] * jnp.outer(diff, diff)

        # Add noise covariance if provided
        if noise_cov is not None:
            cov += noise_cov

        return mean, cov

    def predict(
        self,
        mean_prev: Array,
        cov_prev: Array,
        f: callable,
        Q: Array,
    ) -> tuple[Array, Array]:
        """
        Prediction step using unscented transform.

        Args:
            mean_prev: Previous posterior mean
            cov_prev: Previous posterior covariance
            f: Nonlinear process function
            Q: Process noise covariance

        Returns:
            mu_pred: Predicted mean
            P_pred: Predicted covariance
        """
        # Generate sigma points
        sigma_points, weights_mean, weights_cov = self._compute_sigma_points(
            mean_prev, cov_prev
        )

        # Apply unscented transform
        mu_pred, P_pred = self._unscented_transform(
            sigma_points, weights_mean, weights_cov, f, Q
        )

        return mu_pred, P_pred

    def update(
        self,
        mu_pred: Array,
        P_pred: Array,
        y: Array,
        h: callable,
        R: Array,
    ) -> FilterState:
        """
        Update step using unscented transform.

        Args:
            mu_pred: Predicted mean
            P_pred: Predicted covariance
            y: Measurement
            h: Nonlinear measurement function
            R: Measurement noise covariance

        Returns:
            Posterior filter state
        """
        # Generate sigma points from prediction
        sigma_points, weights_mean, weights_cov = self._compute_sigma_points(
            mu_pred, P_pred
        )

        # Transform through measurement model
        n_points = sigma_points.shape[0]
        y_dim = len(h(sigma_points[0]))
        meas_sigma = jnp.zeros((n_points, y_dim))

        for i in range(n_points):
            meas_sigma[i] = h(sigma_points[i])

        # Predicted measurement mean and covariance
        y_pred = jnp.sum(weights_mean[:, jnp.newaxis] * meas_sigma, axis=0)
        S = R.copy()
        for i in range(n_points):
            diff = meas_sigma[i] - y_pred
            S += weights_cov[i] * jnp.outer(diff, diff)

        # Cross-covariance
        Pxy = jnp.zeros((len(mu_pred), y_dim))
        for i in range(n_points):
            dx = sigma_points[i] - mu_pred
            dy = meas_sigma[i] - y_pred
            Pxy += weights_cov[i] * jnp.outer(dx, dy)

        # Kalman gain
        K = Pxy @ inv(S)

        # Update
        innovation = y - y_pred
        mu_post = mu_pred + K @ innovation
        P_post = P_pred - K @ S @ K.T

        return FilterState(mean=mu_post, cov=P_post)

    def filter_step(
        self,
        state_prev: FilterState,
        f: callable,
        Q: Array,
        y: Array,
        h: callable,
        R: Array,
    ) -> FilterState:
        """
        Complete UKF filter step.

        Args:
            state_prev: Previous filter state
            f: Process function
            Q: Process noise covariance
            y: Measurement
            h: Measurement function
            R: Measurement noise covariance

        Returns:
            Updated filter state
        """
        mu_pred, P_pred = self.predict(state_prev.mean, state_prev.cov, f, Q)
        return self.update(mu_pred, P_pred, y, h, R)
