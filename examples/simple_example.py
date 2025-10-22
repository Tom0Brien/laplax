"""
Simple example: 1D scalar tracking.

Demonstrates basic usage of the Laplace filter for a simple 1D problem.
"""

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

from laplace.filter import LaplaceFilter
from laplace.models import GaussianMeasurementModel, LinearProcessModel


def run_simple_example() -> None:
    """Run simple 1D tracking example."""
    print("=" * 60)
    print("Simple 1D Tracking Example")
    print("=" * 60)

    # JAX random key
    key = jax.random.PRNGKey(42)

    # Process model: constant velocity in 1D
    # State: [position, velocity]
    F = jnp.array([[1.0, 1.0], [0.0, 1.0]])  # x_k = F x_{k-1} + w
    Q = jnp.eye(2) * 0.01  # Process noise covariance

    process = LinearProcessModel(F, Q)

    # Measurement model: observe position with Gaussian noise
    def h(x):
        return jnp.array([x[0]])  # Measure position only

    R = jnp.array([[0.1]])  # Measurement noise covariance
    meas_model = GaussianMeasurementModel(h, R)

    # Initial state estimate
    x_est = jnp.array([0.0, 1.0])  # Position 0, velocity 1
    P_est = jnp.eye(2) * 0.5

    # Create filter
    filt = LaplaceFilter(optimizer="trust_region", max_iter=50)

    # Simulate true trajectory
    x_true = jnp.array([0.0, 1.0])
    n_steps = 20

    # Storage for plotting
    x_true_history = [x_true.copy()]
    x_est_history = [x_est.copy()]
    measurements = []
    uncertainties = [jnp.sqrt(P_est[0, 0])]

    print(f"\nSimulating {n_steps} time steps...")
    print(f"{'Step':>4} {'True Pos':>10} {'Est Pos':>10} {'Error':>10}")
    print("-" * 40)

    for k in range(n_steps):
        # Split key for each random operation
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Simulate true state evolution
        w = jax.random.normal(subkey1, (2,)) * jnp.sqrt(0.01)
        x_true = process(x_true, w)

        # Prediction step
        x_pred, P_pred = filt.predict(x_est, P_est, process)

        # Simulate measurement
        v = jax.random.normal(subkey2, (1,)) * jnp.sqrt(R[0, 0])
        y = h(x_true) + v
        measurements.append(y[0])

        # Update step
        nll = meas_model.nll(y)
        state = filt.update(x_pred, P_pred, nll)

        x_est = state.mean
        P_est = state.cov

        # Store for plotting
        x_true_history.append(x_true.copy())
        x_est_history.append(x_est.copy())
        uncertainties.append(jnp.sqrt(P_est[0, 0]))

        # Print results
        error = abs(x_est[0] - x_true[0])
        print(f"{k + 1:4d} {x_true[0]:10.3f} {x_est[0]:10.3f} {error:10.3f}")

    # Convert to arrays for plotting
    x_true_history = jnp.array(x_true_history)
    x_est_history = jnp.array(x_est_history)
    measurements = jnp.array(measurements)
    uncertainties = jnp.array(uncertainties)
    time_steps = jnp.arange(len(x_true_history))

    # Compute statistics
    pos_errors = jnp.abs(x_est_history[:, 0] - x_true_history[:, 0])

    print("\n" + "=" * 60)
    print(f"Final position estimate: {x_est[0]:.3f}")
    print(f"Final velocity estimate: {x_est[1]:.3f}")
    print(f"Final position uncertainty: {jnp.sqrt(P_est[0, 0]):.3f}")
    print(f"\nMean position error: {jnp.mean(pos_errors):.3f}")
    print(f"RMS position error: {jnp.sqrt(jnp.mean(pos_errors**2)):.3f}")
    print("=" * 60)

    # Plotting
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Position tracking
        ax = axes[0, 0]
        ax.plot(time_steps, x_true_history[:, 0], "b-", label="True", linewidth=2)
        ax.plot(time_steps, x_est_history[:, 0], "r--", label="Estimated", linewidth=2)
        ax.scatter(
            time_steps[1:],
            measurements,
            c="gray",
            s=20,
            alpha=0.5,
            label="Measurements",
        )
        ax.fill_between(
            time_steps,
            x_est_history[:, 0] - 2 * uncertainties,
            x_est_history[:, 0] + 2 * uncertainties,
            alpha=0.2,
            color="red",
            label="95% Confidence",
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Position")
        ax.set_title("Position Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity tracking
        ax = axes[0, 1]
        ax.plot(time_steps, x_true_history[:, 1], "b-", label="True", linewidth=2)
        ax.plot(time_steps, x_est_history[:, 1], "r--", label="Estimated", linewidth=2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Velocity")
        ax.set_title("Velocity Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Position error
        ax = axes[1, 0]
        ax.plot(time_steps, pos_errors, "b-", linewidth=2)
        ax.axhline(
            jnp.mean(pos_errors), color="r", linestyle="--", label="Mean", linewidth=1.5
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Position Error")
        ax.set_title("Position Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Uncertainty evolution
        ax = axes[1, 1]
        ax.plot(time_steps, uncertainties, "g-", linewidth=2)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Position Uncertainty (σ)")
        ax.set_title("Estimation Uncertainty")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("examples/plots/simple_example.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to examples/plots/simple_example.png")
        plt.show()
    except Exception as e:
        print(f"\nNote: Plotting failed ({e}), but simulation completed successfully.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_simple_example()

