"""Kalman filter implementations for comparison with Laplace filter."""

import numpy as np
from numpy.linalg import inv

from laplace.models import LinearProcessModel
from laplace.types import Array, FilterState


class KalmanFilter:
    """
    Standard Kalman Filter for linear-Gaussian systems.

    Optimal for linear process and measurement models with Gaussian noise.
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
        P_post = (np.eye(len(mu_pred)) - K @ H) @ P_pred

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
        P_post = (np.eye(len(mu_pred)) - K @ H) @ P_pred

        return FilterState(mean=mu_post, cov=P_post)


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear systems.

    Uses unscented transform to capture mean and covariance through nonlinear
    transformations more accurately than EKF (up to 3rd order for Gaussian).
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
        L = np.linalg.cholesky((n + lambda_) * cov)

        # Generate sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean

        for i in range(n):
            sigma_points[i + 1] = mean + L[:, i]
            sigma_points[n + i + 1] = mean - L[:, i]

        # Compute weights
        weights_mean = np.zeros(2 * n + 1)
        weights_cov = np.zeros(2 * n + 1)

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
        transformed = np.zeros((n_points, y_dim))

        for i in range(n_points):
            transformed[i] = f(sigma_points[i])

        # Compute mean
        mean = np.sum(weights_mean[:, np.newaxis] * transformed, axis=0)

        # Compute covariance
        cov = np.zeros((y_dim, y_dim))
        for i in range(n_points):
            diff = transformed[i] - mean
            cov += weights_cov[i] * np.outer(diff, diff)

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
        meas_sigma = np.zeros((n_points, y_dim))

        for i in range(n_points):
            meas_sigma[i] = h(sigma_points[i])

        # Predicted measurement mean and covariance
        y_pred = np.sum(weights_mean[:, np.newaxis] * meas_sigma, axis=0)
        S = R.copy()
        for i in range(n_points):
            diff = meas_sigma[i] - y_pred
            S += weights_cov[i] * np.outer(diff, diff)

        # Cross-covariance
        Pxy = np.zeros((len(mu_pred), y_dim))
        for i in range(n_points):
            dx = sigma_points[i] - mu_pred
            dy = meas_sigma[i] - y_pred
            Pxy += weights_cov[i] * np.outer(dx, dy)

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
