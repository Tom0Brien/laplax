"""
Laplax Showcase: Beautiful Laplace Filter Demonstration

Creates a clean, visually appealing demonstration of the Laplace filter
tracking a target through a smooth circular trajectory with nonlinear measurements.
"""

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from laplax.filter import LaplaceFilter
from laplax.models import GaussianMeasurementModel, LinearProcessModel


def add_confidence_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Add a confidence ellipse to the plot."""
    P_2d = jnp.array([[cov[0, 0], cov[0, 2]], [cov[2, 0], cov[2, 2]]])
    
    eigvals, eigvecs = jnp.linalg.eigh(P_2d)
    angle = jnp.arctan2(eigvecs[1, 1], eigvecs[0, 1]) * 180 / jnp.pi
    width, height = 2 * n_std * jnp.sqrt(eigvals)
    
    ellipse = Ellipse(
        (mean[0], mean[2]), width, height, angle=float(angle), **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


def create_showcase():
    """Create the showcase visualization."""
    print("\n" + "="*70)
    print("LAPLAX SHOWCASE")
    print("="*70)
    
    # Parameters
    key = jax.random.PRNGKey(42)
    dt = 0.1
    n_steps = 100
    
    # Process model: simple constant velocity
    F = jnp.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    Q = jnp.eye(4) * 0.01
    process_model = LinearProcessModel(F, Q)
    
    # Nonlinear measurement: range and bearing from origin
    def h(x):
        r = jnp.sqrt(x[0]**2 + x[2]**2)
        theta = jnp.arctan2(x[2], x[0])
        return jnp.array([r, theta])
    
    R = jnp.diag(jnp.array([0.15, 0.05]))
    meas_model = GaussianMeasurementModel(h, R)
    
    # Generate smooth circular trajectory
    print("\nGenerating trajectory...")
    radius = 10.0
    omega = 0.15  # Angular velocity
    
    x_true_history = []
    t = 0
    for i in range(n_steps + 1):
        angle = omega * t
        x = radius * jnp.cos(angle)
        y = radius * jnp.sin(angle)
        vx = -radius * omega * jnp.sin(angle)
        vy = radius * omega * jnp.cos(angle)
        x_true_history.append(jnp.array([x, vx, y, vy]))
        t += dt
    
    # Generate measurements
    key_meas = jax.random.PRNGKey(456)
    measurements = []
    for i in range(1, n_steps + 1):
        key_meas, subkey = jax.random.split(key_meas)
        v = jax.random.normal(subkey, (2,)) * jnp.sqrt(jnp.diag(R))
        y = h(x_true_history[i]) + v
        measurements.append(y)
    
    # Run filter
    print("Running Laplace filter...")
    filt = LaplaceFilter(optimizer="trust_region", max_iter=50, tol=1e-6)
    
    key, subkey = jax.random.split(key)
    x_init = x_true_history[0] + jax.random.normal(subkey, (4,)) * 0.5
    P_init = jnp.eye(4) * 1.0
    
    estimates = [x_init]
    covariances = [P_init]
    
    for y in measurements:
        mu_pred, P_pred = filt.predict(estimates[-1], covariances[-1], process_model)
        nll = meas_model.nll(y)
        state = filt.update(mu_pred, P_pred, nll)
        estimates.append(state.mean)
        covariances.append(state.cov)
    
    # Calculate errors
    true_array = jnp.array(x_true_history[1:])
    est_array = jnp.array(estimates[1:])
    position_errors = jnp.sqrt(
        (est_array[:, 0] - true_array[:, 0])**2 + 
        (est_array[:, 2] - true_array[:, 2])**2
    )
    
    # Create figure
    print("Creating visualization...")
    fig = plt.figure(figsize=(16, 6))
    
    # Color scheme
    color_true = '#2E4053'
    color_est = '#E74C3C'
    color_sensor = '#F39C12'
    
    # ============ Main trajectory plot ============
    ax1 = plt.subplot(1, 3, 1)
    
    # Plot true trajectory
    ax1.plot(true_array[:, 0], true_array[:, 2], '-', 
            color=color_true, linewidth=3, label='True', zorder=5)
    
    # Plot estimate
    ax1.plot(est_array[:, 0], est_array[:, 2], '--', 
            color=color_est, linewidth=2.5, alpha=0.9, label='Estimate', zorder=4)
    
    # Add uncertainty ellipses
    for i in range(0, len(estimates)-1, 10):
        add_confidence_ellipse(
            ax1, estimates[i+1], covariances[i+1], n_std=2.0,
            fill=False, edgecolor=color_est, linewidth=1.5, alpha=0.5, zorder=3
        )
    
    # Highlight key positions
    for i in [0, 25, 50, 75, 99]:
        add_confidence_ellipse(
            ax1, estimates[i+1], covariances[i+1], n_std=2.0,
            fill=True, facecolor=color_est, alpha=0.1, edgecolor=color_est, 
            linewidth=2, zorder=2
        )
    
    # Mark start/end
    ax1.plot(true_array[0, 0], true_array[0, 2], 'o', 
            color='green', markersize=12, label='Start', zorder=10)
    ax1.plot(true_array[-1, 0], true_array[-1, 2], 's', 
            color='red', markersize=12, label='End', zorder=10)
    
    # Mark sensor at origin
    ax1.plot(0, 0, '^', color=color_sensor, markersize=18, 
            markeredgewidth=2, markeredgecolor='black', label='Sensor', zorder=10)
    
    ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Laplace Filter Tracking\nwith Uncertainty Ellipses (95% confidence)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_facecolor('#FAFAFA')
    
    # ============ Error plot ============
    ax2 = plt.subplot(1, 3, 2)
    
    time_steps = jnp.arange(len(position_errors))
    ax2.plot(time_steps, position_errors, '-', 
            color=color_est, linewidth=2.5, label='Position Error')
    ax2.fill_between(time_steps, 0, position_errors, alpha=0.3, color=color_est)
    
    mean_error = float(jnp.mean(position_errors))
    ax2.axhline(mean_error, color='black', linestyle='--', 
               linewidth=2, alpha=0.7, label=f'Mean: {mean_error:.3f}m')
    
    # Add uncertainty
    uncertainties = []
    for P in covariances[1:]:
        P_2d = jnp.array([[P[0, 0], P[0, 2]], [P[2, 0], P[2, 2]]])
        unc = jnp.sqrt(jnp.trace(P_2d))
        uncertainties.append(float(unc))
    
    ax2.plot(time_steps, uncertainties, '--', 
            color='#3498DB', linewidth=2, alpha=0.8, label='Uncertainty (σ)')
    
    ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Error / Uncertainty (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Tracking Performance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#FAFAFA')
    
    # ============ Measurement space (polar) ============
    ax3 = plt.subplot(1, 3, 3, projection='polar')
    
    meas_array = jnp.array(measurements)
    scatter = ax3.scatter(meas_array[:, 1], meas_array[:, 0], 
                         c=time_steps, cmap='plasma', s=40, alpha=0.7, 
                         edgecolors='white', linewidths=0.5, zorder=3)
    
    # True values
    true_meas = jnp.array([h(x) for x in x_true_history[1:]])
    ax3.plot(true_meas[:, 1], true_meas[:, 0], 
            '-', color=color_true, linewidth=2.5, alpha=0.8, label='True', zorder=4)
    
    ax3.set_title('Measurement Space\n(Range-Bearing from Origin)', 
                  fontsize=13, fontweight='bold', pad=20)
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.1, fraction=0.046)
    cbar.set_label('Time Step', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = "examples/plots/laplax_showcase.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[SUCCESS] Saved to: {output_path}")
    
    # Stats
    rmse = float(jnp.sqrt(jnp.mean(position_errors**2)))
    max_error = float(jnp.max(position_errors))
    mean_unc = float(jnp.mean(jnp.array(uncertainties)))
    
    print("\n" + "="*70)
    print("Performance Summary")
    print("="*70)
    print(f"Mean Position Error:  {mean_error:.4f} m")
    print(f"RMSE:                 {rmse:.4f} m")
    print(f"Max Error:            {max_error:.4f} m")
    print(f"Mean Uncertainty:     {mean_unc:.4f} m")
    print("\nFeatures:")
    print("  [+] JAX-powered automatic differentiation")
    print("  [+] Nonlinear range-bearing measurements")
    print("  [+] Full uncertainty quantification")
    print("  [+] Trust-region optimization")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    create_showcase()
