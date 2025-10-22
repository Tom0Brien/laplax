"""
Example: 1D tracking with JAX-accelerated Laplace filter.

Demonstrates automatic differentiation and JIT compilation benefits.
"""

import time

import jax.numpy as jnp
from jax import random

# Use the original filter with JAX-enabled models
from laplace.filter import LaplaceFilter
from laplace.models_jax import (
    AnalyticLinearization,
    GaussianMeasurementModel,
    LinearProcessModel,
    ObjectiveFunction,
)


def run_jax_example() -> None:
    """Run 1D tracking example with JAX."""
    print("=" * 70)
    print("JAX-Accelerated Laplace Filter Example")
    print("=" * 70)

    # Process model: constant velocity in 1D
    # State: [position, velocity]
    F = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    Q = jnp.eye(2) * 0.01
    process = LinearProcessModel(F, Q)

    # Measurement model: nonlinear (position squared)
    def h(x: jnp.ndarray) -> jnp.ndarray:
        """Nonlinear measurement: observe position squared."""
        return jnp.array([x[0] ** 2])  # y = x^2

    R = jnp.array([[0.1]])
    meas_model = GaussianMeasurementModel(h, R)

    print("\nMeasurement model: y = x² + noise")
    print(f"Process noise covariance: {Q[0, 0]:.3f}")
    print(f"Measurement noise variance: {R[0, 0]:.3f}")

    # Initial state estimate
    x_est = jnp.array([0.0, 1.0])
    P_est = jnp.eye(2) * 0.5

    # Create filter
    filt = LaplaceFilter(optimizer="trust_region", max_iter=50, tol=1e-6)

    # Simulate true trajectory
    key = random.PRNGKey(42)
    x_true = jnp.array([0.0, 1.0])
    n_steps = 30

    print(f"\nRunning filter for {n_steps} time steps...")
    print(f"{'Step':>4} {'True Pos':>10} {'Meas':>10} {'Est Pos':>10} {'Error':>10}")
    print("-" * 60)

    # Timing
    start_time = time.time()

    for k in range(n_steps):
        # Simulate true state evolution
        key, subkey = random.split(key)
        w = random.normal(subkey, (2,)) * jnp.sqrt(0.01)
        x_true = process(x_true, w)

        # Prediction step
        x_pred, P_pred = filt.predict(x_est, P_est, process)

        # Simulate measurement
        key, subkey = random.split(key)
        v = random.normal(subkey, (1,)) * jnp.sqrt(R[0, 0])
        y = h(x_true) + v

        # Update step with analytic linearization (uses JAX autodiff internally!)
        nll = meas_model.nll(y)

        # Create objective for analytic linearization
        P_inv_pred = jnp.linalg.inv(P_pred)
        obj = ObjectiveFunction(x_pred, P_inv_pred, nll)
        lin = AnalyticLinearization(obj, meas_model, y)

        state = filt.update(x_pred, P_pred, nll, lin=lin)
        x_est = state.mean
        P_est = state.cov

        # Print results
        error = abs(float(x_est[0]) - float(x_true[0]))
        if (k + 1) % 5 == 0 or k < 5:
            print(
                f"{k + 1:4d} {float(x_true[0]):10.3f} {float(y[0]):10.3f} "
                f"{float(x_est[0]):10.3f} {error:10.3f}"
            )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"Final position estimate: {float(x_est[0]):.3f}")
    print(f"Final velocity estimate: {float(x_est[1]):.3f}")
    print(f"Final uncertainty: {float(jnp.sqrt(P_est[0, 0])):.3f}")
    print(f"\nTotal time: {elapsed:.3f}s ({elapsed / n_steps * 1000:.1f}ms per step)")
    print("=" * 70)

    # Demonstrate JAX's automatic differentiation
    print("\n" + "=" * 70)
    print("JAX Automatic Differentiation Demo")
    print("=" * 70)

    # Show Jacobian computation
    x_test = jnp.array([2.0])
    print("\nFor measurement function h(x) = x²")
    print(f"At x = {float(x_test[0]):.1f}:")

    # Compute Jacobian using JAX
    from jax import jacfwd

    def h_scalar(x):
        return x[0] ** 2

    dh_dx = jacfwd(lambda x: jnp.array([h_scalar(x)]))(x_test)
    print(f"  JAX computed Jacobian: {float(dh_dx[0, 0]):.3f}")
    print(f"  Analytical derivative: {2 * float(x_test[0]):.3f}")
    print(f"  Match: {jnp.allclose(dh_dx[0, 0], 2 * x_test[0])}")

    print("\nKey benefits of JAX:")
    print("  [+] Automatic differentiation (no manual Jacobians needed)")
    print("  [+] JIT compilation for faster execution")
    print("  [+] GPU/TPU support (if available)")
    print("  [+] Vectorization and parallelization")
    print("=" * 70)


if __name__ == "__main__":
    run_jax_example()
