"""
Filter Comparison: KF vs EKF vs UKF vs Laplace Filter

Compares different filtering approaches on a nonlinear tracking problem:
- Kalman Filter (KF) - Optimal for linear systems
- Extended Kalman Filter (EKF) - First-order linearization
- Unscented Kalman Filter (UKF) - Unscented transform (no Jacobians!)
- Laplace Filter - MAP estimation with Hessian-based covariance

Tests on 2D tracking with nonlinear (range-bearing) measurements.
"""

import time

import numpy as np
from matplotlib import pyplot as plt

from laplace.filter import LaplaceFilter
from laplace.kalman import ExtendedKalmanFilter, KalmanFilter, UnscentedKalmanFilter
from laplace.models import (
    AnalyticLinearization,
    GaussianMeasurementModel,
    LinearProcessModel,
    ObjectiveFunction,
)
from laplace.types import FilterState


def run_comparison():
    """Run filter comparison on 2D tracking problem."""
    print("=" * 80)
    print("Filter Comparison: KF vs EKF vs UKF vs Laplace Filter")
    print("=" * 80)

    # Simulation parameters
    np.random.seed(42)
    dt = 0.1
    n_steps = 50

    # Process model: constant velocity in 2D
    # State: [x, vx, y, vy]
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    Q = np.eye(4) * 0.01
    process_model = LinearProcessModel(F, Q)

    # Nonlinear measurement model: range and bearing from origin
    def h(x):
        """Compute range and bearing."""
        pos = x[[0, 2]]
        r = np.sqrt(pos[0] ** 2 + pos[1] ** 2) + 1e-8
        theta = np.arctan2(pos[1], pos[0])
        return np.array([r, theta])

    def H_jacobian(x):
        """Measurement Jacobian."""
        px, py = x[0], x[2]
        r = np.sqrt(px**2 + py**2) + 1e-8
        H = np.zeros((2, 4))
        H[0, 0] = px / r
        H[0, 2] = py / r
        H[1, 0] = -py / (r**2)
        H[1, 2] = px / (r**2)
        return H

    R = np.diag([0.1, 0.05])  # Range and bearing noise

    # True initial state
    x_true = np.array([0.0, 1.0, 0.0, 0.5])

    # Initialize filters with same initial conditions
    x_init = x_true + np.random.randn(4) * 0.5
    P_init = np.eye(4) * 1.0

    # Create filters
    kf = KalmanFilter()
    ekf = ExtendedKalmanFilter()
    ukf = UnscentedKalmanFilter(alpha=1e-3, beta=2.0, kappa=1.0)
    laplace = LaplaceFilter(optimizer="trust_region", max_iter=30, tol=1e-6)

    # Storage
    filters = {"KF": kf, "EKF": ekf, "UKF": ukf, "Laplace": laplace}
    estimates = {name: [x_init.copy()] for name in filters}
    covariances = {name: [P_init.copy()] for name in filters}
    times = {name: [] for name in filters}
    x_true_history = [x_true.copy()]

    print(f"\nRunning {n_steps} time steps with 4 filters...")
    print("Nonlinear measurement: range-bearing from origin")

    # Run simulation
    for k in range(n_steps):
        # Simulate true state
        w = np.random.randn(4) * np.sqrt(0.01)
        x_true = F @ x_true + w
        x_true_history.append(x_true.copy())

        # Simulate measurement
        v = np.random.randn(2) * np.sqrt(np.diag(R))
        y = h(x_true) + v

        # Run each filter
        for name, filt in filters.items():
            x_est = estimates[name][-1]
            P_est = covariances[name][-1]

            start = time.time()

            if name == "KF":
                # KF: Uses linear approximation (will perform poorly)
                mu_pred, P_pred = filt.predict(x_est, P_est, process_model)
                H = H_jacobian(mu_pred)  # Linearize at prediction
                state = filt.update(mu_pred, P_pred, y, H, R)

            elif name == "EKF":
                # EKF: Linearize around current estimate
                def f_nonlin(x):
                    return F @ x

                F_jac = F
                mu_pred, P_pred = ekf.predict(x_est, P_est, f_nonlin, F_jac, Q)
                H = H_jacobian(mu_pred)
                state = ekf.update(mu_pred, P_pred, y, h, H, R)

            elif name == "UKF":
                # UKF: No Jacobians needed!
                def f_ukf(x):
                    return F @ x

                state = ukf.filter_step(FilterState(x_est, P_est), f_ukf, Q, y, h, R)

            elif name == "Laplace":
                # Laplace: MAP estimation
                mu_pred, P_pred = laplace.predict(x_est, P_est, process_model)

                # Use analytic linearization
                meas_model = GaussianMeasurementModel(h, R, H=H_jacobian)
                nll = meas_model.nll(y)
                P_inv_pred = np.linalg.inv(P_pred)
                obj = ObjectiveFunction(mu_pred, P_inv_pred, nll)
                lin = AnalyticLinearization(obj, meas_model, y)

                state = laplace.update(mu_pred, P_pred, nll, lin=lin)

            elapsed = time.time() - start
            times[name].append(elapsed)

            estimates[name].append(state.mean.copy())
            covariances[name].append(state.cov.copy())

        if (k + 1) % 10 == 0:
            print(f"Step {k + 1:3d}/{n_steps} completed")

    # Convert to arrays
    x_true_history = np.array(x_true_history)
    for name in filters:
        estimates[name] = np.array(estimates[name])

    # Compute errors
    errors = {}
    for name in filters:
        pos_errors = np.linalg.norm(
            estimates[name][:, [0, 2]] - x_true_history[:, [0, 2]], axis=1
        )
        errors[name] = pos_errors

    # Print statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Filter':<12} {'Mean Error':>12} {'RMS Error':>12} {'Final Error':>12} {'Avg Time (ms)':>15}"
    )
    print("-" * 80)

    for name in ["KF", "EKF", "UKF", "Laplace"]:
        mean_err = np.mean(errors[name])
        rms_err = np.sqrt(np.mean(errors[name] ** 2))
        final_err = errors[name][-1]
        avg_time = np.mean(times[name]) * 1000

        print(
            f"{name:<12} {mean_err:>12.3f} {rms_err:>12.3f} {final_err:>12.3f} {avg_time:>15.2f}"
        )

    print("=" * 80)

    # Plotting
    plt.figure(figsize=(15, 10))

    # Trajectory plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(
        x_true_history[:, 0],
        x_true_history[:, 2],
        "k-",
        label="True",
        linewidth=2.5,
        zorder=5,
    )
    colors = {"KF": "blue", "EKF": "green", "UKF": "orange", "Laplace": "red"}
    styles = {"KF": "--", "EKF": "--", "UKF": "-", "Laplace": "-"}

    for name in ["KF", "EKF", "UKF", "Laplace"]:
        ax1.plot(
            estimates[name][:, 0],
            estimates[name][:, 2],
            styles[name],
            color=colors[name],
            label=name,
            linewidth=2,
            alpha=0.8,
        )

    ax1.plot(x_true_history[0, 0], x_true_history[0, 2], "go", markersize=10)
    ax1.plot(x_true_history[-1, 0], x_true_history[-1, 2], "rs", markersize=10)
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("2D Trajectory Comparison")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Position errors over time
    ax2 = plt.subplot(2, 3, 2)
    time_steps = np.arange(n_steps + 1)
    for name in ["KF", "EKF", "UKF", "Laplace"]:
        ax2.plot(
            time_steps,
            errors[name],
            styles[name],
            color=colors[name],
            label=name,
            linewidth=2,
        )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title("Position Error Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Error distribution (box plot)
    ax3 = plt.subplot(2, 3, 3)
    error_data = [errors[name][1:] for name in ["KF", "EKF", "UKF", "Laplace"]]
    bp = ax3.boxplot(
        error_data, labels=["KF", "EKF", "UKF", "Laplace"], patch_artist=True
    )
    for patch, name in zip(bp["boxes"], ["KF", "EKF", "UKF", "Laplace"]):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title("Error Distribution")
    ax3.grid(True, alpha=0.3, axis="y")

    # X position tracking
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_steps, x_true_history[:, 0], "k-", label="True", linewidth=2.5)
    for name in ["EKF", "UKF", "Laplace"]:  # Skip KF for clarity
        ax4.plot(
            time_steps,
            estimates[name][:, 0],
            styles[name],
            color=colors[name],
            label=name,
            linewidth=2,
            alpha=0.8,
        )
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("X Position (m)")
    ax4.set_title("X Position Tracking (EKF/UKF/Laplace)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Y position tracking
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_steps, x_true_history[:, 2], "k-", label="True", linewidth=2.5)
    for name in ["EKF", "UKF", "Laplace"]:
        ax5.plot(
            time_steps,
            estimates[name][:, 2],
            styles[name],
            color=colors[name],
            label=name,
            linewidth=2,
            alpha=0.8,
        )
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Y Position (m)")
    ax5.set_title("Y Position Tracking (EKF/UKF/Laplace)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Computation time comparison
    ax6 = plt.subplot(2, 3, 6)
    avg_times = [
        np.mean(times[name]) * 1000 for name in ["KF", "EKF", "UKF", "Laplace"]
    ]
    ax6.bar(
        ["KF", "EKF", "UKF", "Laplace"],
        avg_times,
        color=[colors[name] for name in ["KF", "EKF", "UKF", "Laplace"]],
        alpha=0.7,
    )
    ax6.set_ylabel("Average Time (ms)")
    ax6.set_title("Computational Cost per Step")
    ax6.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("filter_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to filter_comparison.png")
    plt.show()

    # Print key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(
        "• KF performs poorly due to linearization at wrong point (ignores nonlinearity)"
    )
    print("• EKF better but still relies on first-order Taylor approximation")
    print("• UKF captures nonlinearity better (no Jacobians needed!)")
    print("• Laplace uses optimization for MAP, can handle non-Gaussian cases")
    print(f"• Best accuracy: {min(filters, key=lambda n: np.mean(errors[n]))}")
    print(f"• Fastest: {min(filters, key=lambda n: np.mean(times[n]))}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
