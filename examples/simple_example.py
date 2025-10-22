"""
Simple example: 1D scalar tracking.

Demonstrates basic usage of the Laplace filter for a simple 1D problem.
"""

import numpy as np

from laplace.filter import LaplaceFilter
from laplace.models import GaussianMeasurementModel, LinearProcessModel


def run_simple_example() -> None:
    """Run simple 1D tracking example."""
    print("=" * 60)
    print("Simple 1D Tracking Example")
    print("=" * 60)

    # Process model: constant velocity in 1D
    # State: [position, velocity]
    F = np.array([[1.0, 1.0], [0.0, 1.0]])  # x_k = F x_{k-1} + w
    Q = np.eye(2) * 0.01  # Process noise covariance

    process = LinearProcessModel(F, Q)

    # Measurement model: observe position with Gaussian noise
    def h(x: np.ndarray) -> np.ndarray:
        return np.array([x[0]])  # Measure position only

    R = np.array([[0.1]])  # Measurement noise covariance
    meas_model = GaussianMeasurementModel(h, R)

    # Initial state estimate
    x_est = np.array([0.0, 1.0])  # Position 0, velocity 1
    P_est = np.eye(2) * 0.5

    # Create filter
    filt = LaplaceFilter(optimizer="trust_region", max_iter=50)

    # Simulate true trajectory
    x_true = np.array([0.0, 1.0])
    n_steps = 20

    print(f"\nSimulating {n_steps} time steps...")
    print(f"{'Step':>4} {'True Pos':>10} {'Est Pos':>10} {'Error':>10}")
    print("-" * 40)

    for k in range(n_steps):
        # Simulate true state evolution
        w = np.random.randn(2) * np.sqrt(0.01)
        x_true = process(x_true, w)

        # Prediction step
        x_pred, P_pred = filt.predict(x_est, P_est, process)

        # Simulate measurement
        v = np.random.randn(1) * np.sqrt(R[0, 0])
        y = h(x_true) + v

        # Update step
        nll = meas_model.nll(y)
        state = filt.update(x_pred, P_pred, nll)

        x_est = state.mean
        P_est = state.cov

        # Print results
        error = abs(x_est[0] - x_true[0])
        print(f"{k + 1:4d} {x_true[0]:10.3f} {x_est[0]:10.3f} {error:10.3f}")

    print("\n" + "=" * 60)
    print(f"Final position estimate: {x_est[0]:.3f}")
    print(f"Final velocity estimate: {x_est[1]:.3f}")
    print(f"Final position uncertainty: {np.sqrt(P_est[0, 0]):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    run_simple_example()
