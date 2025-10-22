"""
Example: 2D tracking with nonlinear measurements.

A target moves with constant velocity in 2D. We observe range and bearing
(nonlinear measurements) and use the Laplace filter for state estimation.
"""

import numpy as np
from matplotlib import pyplot as plt

from laplace.filter import LaplaceFilter
from laplace.models import (
    AnalyticLinearization,
    GaussianMeasurementModel,
    LinearProcessModel,
)


def run_tracking_example() -> None:
    """Run 2D tracking example with range-bearing measurements."""
    print("=" * 60)
    print("2D Tracking with Laplace Filter")
    print("=" * 60)

    # Simulation parameters
    dt = 0.1  # Time step
    n_steps = 50
    np.random.seed(42)

    # Process model: [x, vx, y, vy] - position and velocity in 2D
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    Q = np.eye(4) * 0.01  # Small process noise
    process = LinearProcessModel(F, Q)

    # True initial state
    x_true = np.array([0.0, 1.0, 0.0, 0.5])  # Start at origin, moving right and up

    # Measurement model: range and bearing from origin
    def h(x: np.ndarray) -> np.ndarray:
        """Compute range and bearing from state."""
        pos = x[[0, 2]]  # [x, y]
        r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)  # Range
        theta = np.arctan2(pos[1], pos[0])  # Bearing
        return np.array([r, theta])

    def H_jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian of measurement function."""
        px, py = x[0], x[2]
        r = np.sqrt(px**2 + py**2) + 1e-8
        H = np.zeros((2, 4))
        H[0, 0] = px / r  # ∂r/∂x
        H[0, 2] = py / r  # ∂r/∂y
        H[1, 0] = -py / (r**2)  # ∂θ/∂x
        H[1, 2] = px / (r**2)  # ∂θ/∂y
        return H

    R = np.diag([0.1, 0.05])  # Range and bearing noise
    meas_model = GaussianMeasurementModel(h, R, H=H_jacobian)

    # Initialize filter
    filt = LaplaceFilter(optimizer="trust_region", max_iter=50, tol=1e-6)
    x_est = x_true + np.random.randn(4) * 0.5  # Noisy initial estimate
    P_est = np.eye(4) * 1.0

    # Storage
    x_true_history = [x_true.copy()]
    x_est_history = [x_est.copy()]
    measurements = []

    print(f"\nRunning filter for {n_steps} time steps...")
    print(f"Initial error: {np.linalg.norm(x_est - x_true):.3f}")

    # Run filter
    for k in range(n_steps):
        # Simulate true state
        w = np.random.randn(4) * np.sqrt(0.01)
        x_true = process(x_true, w)
        x_true_history.append(x_true.copy())

        # Prediction
        x_pred, P_pred = filt.predict(x_est, P_est, process)

        # Simulate measurement
        v = np.random.randn(2) * np.sqrt(np.diag(R))
        y = h(x_true) + v
        measurements.append(y.copy())

        # Update with analytic linearization
        nll = meas_model.nll(y)
        obj_for_lin = None
        # Create objective for linearization
        from laplace.models import ObjectiveFunction

        P_inv_pred = np.linalg.inv(P_pred)
        obj_for_lin = ObjectiveFunction(x_pred, P_inv_pred, nll)
        lin = AnalyticLinearization(obj_for_lin, meas_model, y)

        state = filt.update(x_pred, P_pred, nll, lin=lin)
        x_est = state.mean
        P_est = state.cov
        x_est_history.append(x_est.copy())

        if (k + 1) % 10 == 0:
            error = np.linalg.norm(x_est[[0, 2]] - x_true[[0, 2]])
            print(f"Step {k + 1:3d}: Position error = {error:.3f}")

    # Final statistics
    x_true_history = np.array(x_true_history)
    x_est_history = np.array(x_est_history)
    pos_errors = np.linalg.norm(
        x_est_history[:, [0, 2]] - x_true_history[:, [0, 2]], axis=1
    )

    print("\nFinal Results:")
    print(f"  Mean position error: {np.mean(pos_errors):.3f}")
    print(f"  RMS position error:  {np.sqrt(np.mean(pos_errors**2)):.3f}")
    print(f"  Final position error: {pos_errors[-1]:.3f}")

    # Plotting
    try:
        plt.figure(figsize=(12, 5))

        # Trajectory plot
        plt.subplot(1, 2, 1)
        plt.plot(
            x_true_history[:, 0], x_true_history[:, 2], "b-", label="True", linewidth=2
        )
        plt.plot(
            x_est_history[:, 0],
            x_est_history[:, 2],
            "r--",
            label="Estimated",
            linewidth=2,
        )
        plt.plot(
            x_true_history[0, 0],
            x_true_history[0, 2],
            "go",
            markersize=10,
            label="Start",
        )
        plt.plot(
            x_true_history[-1, 0],
            x_true_history[-1, 2],
            "rs",
            markersize=10,
            label="End",
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title("2D Trajectory")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")

        # Error plot
        plt.subplot(1, 2, 2)
        plt.plot(pos_errors, "b-", linewidth=2)
        plt.xlabel("Time Step")
        plt.ylabel("Position Error")
        plt.title("Estimation Error Over Time")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("tracking_example.png", dpi=150)
        print("\nPlot saved to tracking_example.png")
        plt.show()
    except Exception as e:
        print(f"\nNote: Plotting failed ({e}), but simulation completed successfully.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_tracking_example()
