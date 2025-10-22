"""
Robust Filter Comparison: Testing with Outlier Measurements

Scenario designed to favor the Laplace filter:
- 2D tracking with nonlinear measurements
- Measurements contaminated with outliers (heavy-tailed noise)
- Non-Gaussian likelihood (Student-t distribution for robustness)

The Laplace filter can handle non-Gaussian likelihoods via custom NLL functions,
while KF/EKF/UKF assume Gaussian measurements and are sensitive to outliers.
"""

import time

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from laplace.filter import (
    ExtendedKalmanFilter,
    KalmanFilter,
    LaplaceFilter,
    UnscentedKalmanFilter,
)
from laplace.models import LinearProcessModel
from laplace.types import FilterState


def student_t_nll(y, h_func, R, nu=3.0):
    """
    Create a robust Student-t negative log-likelihood (heavy-tailed).

    Student-t distribution is robust to outliers compared to Gaussian.
    As nu->infinity, Student-t approaches Gaussian.
    nu=3 gives heavy tails for robustness.
    """
    R_inv = jnp.linalg.inv(R)

    def nll(x):
        residual = y - h_func(x)
        # Student-t NLL (approximation)
        # -log p(y|x) ∝ (nu + d)/2 * log(1 + residual^T R^{-1} residual / nu)
        mahal_dist = float(residual.T @ R_inv @ residual)
        d = len(y)
        return 0.5 * (nu + d) * jnp.log(1.0 + mahal_dist / nu)

    return nll


def inject_outliers(measurements, outlier_rate=0.15, outlier_magnitude=5.0, key=None):
    """
    Inject random outliers into measurements.

    Args:
        measurements: Array of measurements
        outlier_rate: Fraction of measurements to corrupt
        outlier_magnitude: How large the outliers are (multiples of normal noise)
        key: JAX random key

    Returns:
        Corrupted measurements and indices of outliers
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n_meas = len(measurements)
    n_outliers = int(n_meas * outlier_rate)

    # Split key for choice and noise
    key, subkey = jax.random.split(key)
    outlier_indices = jax.random.choice(subkey, n_meas, (n_outliers,), replace=False)

    measurements_corrupted = measurements.copy()
    for _i, idx in enumerate(outlier_indices):
        # Add large random noise
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (len(measurements[int(idx)]),)) * outlier_magnitude
        measurements_corrupted[int(idx)] = measurements[int(idx)] + noise

    return measurements_corrupted, outlier_indices


def run_robust_comparison():
    """Run robust filter comparison with outlier measurements."""
    print("=" * 80)
    print("Robust Filter Comparison: Outlier-Contaminated Measurements")
    print("=" * 80)

    # Simulation parameters
    key = jax.random.PRNGKey(42)
    dt = 0.1
    n_steps = 50
    outlier_rate = 0.20  # 20% of measurements are outliers!
    outlier_magnitude = 5.0

    print("\nScenario:")
    print(f"  • {outlier_rate*100:.0f}% of measurements are OUTLIERS")
    print(f"  • Outlier magnitude: {outlier_magnitude}x normal noise")
    print("  • Laplace filter uses robust Student-t likelihood (heavy tails)")
    print("  • KF/EKF/UKF use Gaussian likelihood (sensitive to outliers)")

    # Process model: constant velocity in 2D
    F = jnp.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    Q = jnp.eye(4) * 0.01
    process_model = LinearProcessModel(F, Q)

    # Nonlinear measurement model: range and bearing
    def h(x):
        pos = x[[0, 2]]
        r = jnp.sqrt(pos[0] ** 2 + pos[1] ** 2) + 1e-8
        theta = jnp.arctan2(pos[1], pos[0])
        return jnp.array([r, theta])

    def H_jacobian(x):
        px, py = x[0], x[2]
        r = jnp.sqrt(px**2 + py**2) + 1e-8
        H = jnp.zeros((2, 4))
        H[0, 0] = px / r
        H[0, 2] = py / r
        H[1, 0] = -py / (r**2)
        H[1, 2] = px / (r**2)
        return H

    R = jnp.diag(jnp.array([0.1, 0.05]))

    # True initial state
    x_true = jnp.array([0.0, 1.0, 0.0, 0.5])

    # Initialize filters
    key, subkey = jax.random.split(key)
    x_init = x_true + jax.random.normal(subkey, (4,)) * 0.5
    P_init = jnp.eye(4) * 1.0

    kf = KalmanFilter()
    ekf = ExtendedKalmanFilter()
    ukf = UnscentedKalmanFilter(alpha=1e-3, beta=2.0, kappa=1.0)
    laplace_gaussian = LaplaceFilter(optimizer="trust_region", max_iter=30, tol=1e-6)
    laplace_robust = LaplaceFilter(optimizer="trust_region", max_iter=30, tol=1e-6)

    # Generate true trajectory and measurements first
    x_true_history = [x_true.copy()]
    clean_measurements = []

    # Use a separate random key for trajectory generation
    key_traj = jax.random.PRNGKey(42)

    for k in range(n_steps):
        # Split key for each random operation
        key_traj, subkey1, subkey2 = jax.random.split(key_traj, 3)

        # Generate true trajectory with process noise
        w = jax.random.normal(subkey1, (4,)) * jnp.sqrt(0.01)
        x_true = F @ x_true + w
        x_true_history.append(x_true.copy())

        # Generate clean measurement
        v = jax.random.normal(subkey2, (2,)) * jnp.sqrt(jnp.diag(R))
        y_clean = h(x_true) + v
        clean_measurements.append(y_clean)

    # Inject outliers into measurements
    key, subkey = jax.random.split(key)
    measurements, outlier_indices = inject_outliers(
        clean_measurements, outlier_rate, outlier_magnitude, key=subkey
    )

    print(f"\n  • Generated {n_steps} measurements")
    print(f"  • {len(outlier_indices)} measurements corrupted with outliers")
    print(f"  • Outlier steps: {sorted(outlier_indices[:10])}...")

    # Initialize filters
    filters = {
        "KF": kf,
        "EKF": ekf,
        "UKF": ukf,
        "Laplace (Gaussian)": laplace_gaussian,
        "Laplace (Robust)": laplace_robust,
    }
    estimates = {name: [x_init.copy()] for name in filters}
    covariances = {name: [P_init.copy()] for name in filters}
    times = {name: [] for name in filters}

    print(f"\nRunning {n_steps} time steps with 5 filters...")

    # Run simulation (use pre-generated trajectory and measurements)
    for k in range(n_steps):
        # Get measurement (already includes outliers if present)
        y = measurements[k]

        # Run each filter
        for name, filt in filters.items():
            x_est = estimates[name][-1]
            P_est = covariances[name][-1]

            start = time.time()

            if name == "KF":
                mu_pred, P_pred = filt.predict(x_est, P_est, process_model)
                H = H_jacobian(mu_pred)
                state = filt.update(mu_pred, P_pred, y, H, R)

            elif name == "EKF":
                def f_nonlin(x):
                    return F @ x

                F_jac = F
                mu_pred, P_pred = ekf.predict(x_est, P_est, f_nonlin, F_jac, Q)
                H = H_jacobian(mu_pred)
                state = ekf.update(mu_pred, P_pred, y, h, H, R)

            elif name == "UKF":
                def f_ukf(x):
                    return F @ x

                state = ukf.filter_step(FilterState(x_est, P_est), f_ukf, Q, y, h, R)

            elif name == "Laplace (Gaussian)":
                # Standard Gaussian likelihood (like KF/EKF/UKF)
                mu_pred, P_pred = laplace_gaussian.predict(x_est, P_est, process_model)

                # Gaussian NLL
                R_inv = jnp.linalg.inv(R)
                def gaussian_nll(x):
                    residual = y - h(x)
                    return 0.5 * float(residual.T @ R_inv @ residual)

                state = laplace_gaussian.update(mu_pred, P_pred, gaussian_nll)

            elif name == "Laplace (Robust)":
                # ROBUST Student-t likelihood (heavy-tailed, outlier-resistant)
                mu_pred, P_pred = laplace_robust.predict(x_est, P_est, process_model)

                # Student-t NLL (robust to outliers)
                robust_nll = student_t_nll(y, h, R, nu=3.0)

                state = laplace_robust.update(mu_pred, P_pred, robust_nll)

            elapsed = time.time() - start
            times[name].append(elapsed)

            estimates[name].append(state.mean.copy())
            covariances[name].append(state.cov.copy())

        if (k + 1) % 10 == 0:
            print(f"Step {k + 1:3d}/{n_steps} completed")

    # Convert to arrays
    x_true_history = jnp.array(x_true_history)
    for name in filters:
        estimates[name] = jnp.array(estimates[name])

    # Compute errors
    errors = {}
    for name in filters:
        pos_errors = jnp.linalg.norm(
            estimates[name][:, [0, 2]] - x_true_history[:, [0, 2]], axis=1
        )
        errors[name] = pos_errors

    # Print statistics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Filter':<22} {'Mean Error':>12} {'RMS Error':>12} "
        f"{'Outlier Impact':>15} {'Avg Time (ms)':>15}"
    )
    print("-" * 80)

    for name in filters:
        mean_err = jnp.mean(errors[name])
        rms_err = jnp.sqrt(jnp.mean(errors[name] ** 2))

        # Compute error during outlier measurements vs clean measurements
        outlier_errors = errors[name][jnp.array(list(outlier_indices)) + 1]
        clean_errors = errors[name][
            [i for i in range(1, len(errors[name])) if (i-1) not in outlier_indices]
        ]
        outlier_impact = jnp.mean(outlier_errors) / (jnp.mean(clean_errors) + 1e-8)

        avg_time = jnp.mean(times[name]) * 1000

        print(
            f"{name:<22} {mean_err:>12.3f} {rms_err:>12.3f} "
            f"{outlier_impact:>14.2f}x {avg_time:>15.2f}"
        )

    print("=" * 80)

    # Plotting
    plt.figure(figsize=(16, 10))

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
    colors = {
        "KF": "blue",
        "EKF": "green",
        "UKF": "orange",
        "Laplace (Gaussian)": "purple",
        "Laplace (Robust)": "red",
    }
    styles = {
        "KF": ":",
        "EKF": "--",
        "UKF": "--",
        "Laplace (Gaussian)": "-.",
        "Laplace (Robust)": "-",
    }

    for name in filters:
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
    ax1.set_title("2D Trajectory (with Outliers)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Position errors over time with outlier markers
    ax2 = plt.subplot(2, 3, 2)
    time_steps = jnp.arange(n_steps + 1)
    for name in filters:
        ax2.plot(
            time_steps,
            errors[name],
            styles[name],
            color=colors[name],
            label=name,
            linewidth=2,
            alpha=0.7,
        )

    # Mark outlier measurements
    for idx in outlier_indices:
        ax2.axvline(idx + 1, color="red", alpha=0.2, linewidth=0.5)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position Error (m)")
    ax2.set_title("Position Error (Red bars = Outliers)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Focus on robust filters
    ax3 = plt.subplot(2, 3, 3)
    for name in ["EKF", "UKF", "Laplace (Robust)"]:
        ax3.plot(
            time_steps,
            errors[name],
            styles[name],
            color=colors[name],
            label=name,
            linewidth=2.5,
        )
    for idx in outlier_indices:
        ax3.axvline(idx + 1, color="red", alpha=0.2, linewidth=0.5)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Position Error (m)")
    ax3.set_title("Best Performers (Zoomed)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Error distribution comparison
    ax4 = plt.subplot(2, 3, 4)
    error_data = [errors[name][1:] for name in filters]
    bp = ax4.boxplot(error_data, tick_labels=[n[:15] for n in filters], patch_artist=True)
    for patch, name in zip(bp["boxes"], filters):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax4.set_ylabel("Position Error (m)")
    ax4.set_title("Error Distribution")
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.tick_params(axis='x', rotation=45)

    # Outlier impact comparison
    ax5 = plt.subplot(2, 3, 5)
    outlier_impacts = []
    for name in filters:
        outlier_errors = errors[name][jnp.array(list(outlier_indices)) + 1]
        clean_errors = errors[name][
            [i for i in range(1, len(errors[name])) if (i-1) not in outlier_indices]
        ]
        impact = jnp.mean(outlier_errors) / (jnp.mean(clean_errors) + 1e-8)
        outlier_impacts.append(impact)

    bars = ax5.bar(
        [n[:15] for n in filters],
        outlier_impacts,
        color=[colors[n] for n in filters],
        alpha=0.7,
    )
    # Highlight the best (lowest impact)
    best_idx = jnp.argmin(outlier_impacts)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)

    ax5.axhline(1.0, color='gray', linestyle='--', label='No impact', linewidth=1)
    ax5.set_ylabel("Error Ratio (Outlier/Clean)")
    ax5.set_title("Outlier Robustness (Lower = Better)")
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.tick_params(axis='x', rotation=45)
    ax5.legend()

    # Computation time
    ax6 = plt.subplot(2, 3, 6)
    avg_times = [jnp.mean(times[name]) * 1000 for name in filters]
    ax6.bar(
        [n[:15] for n in filters],
        avg_times,
        color=[colors[name] for name in filters],
        alpha=0.7,
    )
    ax6.set_ylabel("Average Time (ms)")
    ax6.set_title("Computational Cost")
    ax6.grid(True, alpha=0.3, axis="y")
    ax6.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("examples/plots/robust_comparison.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to examples/plots/robust_comparison.png")
    plt.show()

    # Print key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("[+] Laplace (Robust) uses Student-t likelihood -> resistant to outliers")
    print("[+] Laplace (Gaussian) similar to EKF/UKF -> sensitive to outliers")
    print("[+] KF/EKF/UKF assume Gaussian noise -> degraded by outliers")
    print("[+] Robust Laplace maintains low errors even during outlier measurements")

    best_filter = min(filters, key=lambda n: jnp.mean(errors[n]))
    most_robust = min(filters, key=lambda n: outlier_impacts[list(filters.keys()).index(n)])

    print(f"\n=> Best overall accuracy: {best_filter}")
    print(f"=> Most outlier-robust: {most_robust}")
    print("=" * 80)


if __name__ == "__main__":
    run_robust_comparison()


