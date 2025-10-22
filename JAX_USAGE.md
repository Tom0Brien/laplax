# JAX Integration Guide

The Laplace filter library now includes **JAX-accelerated implementations** that provide automatic differentiation, JIT compilation, and GPU support.

## Quick Start

```python
# Use JAX-accelerated models for automatic Jacobian computation
from laplace.models_jax import (
    LinearProcessModel,
    NonlinearProcessModel, 
    GaussianMeasurementModel,
    AnalyticLinearization,
    ObjectiveFunction
)
import jax.numpy as jnp

# Define your model using JAX-traceable functions
def h(x):
    return jnp.array([x[0]**2, jnp.sin(x[1])])  # Any JAX-traceable function!

# JAX will automatically compute Jacobians - no manual derivatives needed!
meas_model = GaussianMeasurementModel(h, R)
```

## Key Benefits

### 1. **Automatic Differentiation**
No need to manually derive Jacobians or Hessians. JAX computes them automatically:

```python
# Your nonlinear function
def f(x):
    return jnp.array([x[0] + x[1]**2, jnp.exp(x[0])])

# Create process model - Jacobian computed automatically!
process = NonlinearProcessModel(f, Q)
F = process.jacobian(x)  # Automatic!
```

### 2. **JIT Compilation**
JAX JIT-compiles functions for near-C performance:

```python
from jax import jit
from laplace.math_jax import regularize_covariance

# Functions are JIT-compiled for speed
P_reg = regularize_covariance(P)  # Fast!
```

### 3. **GPU/TPU Support**
Computations automatically run on GPU/TPU if available:

```python
import jax
print(f"Using device: {jax.devices()}")  # Shows GPU if available
```

## Available Modules

### `laplace.math_jax`
JAX-accelerated linear algebra operations:
- `ensure_symmetric` - Symmetrize matrices
- `cov_to_sqrt_inv` / `sqrt_inv_to_cov` - Information form conversions
- `mahalanobis_squared` - Mahalanobis distance
- `gradient_autodiff`, `hessian_autodiff`, `jacobian_autodiff` - Automatic differentiation
- `regularize_covariance` - Covariance regularization
- `solve_psd`, `inv_psd` - Efficient PSD matrix operations

### `laplace.models_jax`
JAX-enabled models with automatic differentiation:
- `LinearProcessModel` - JIT-compiled linear process model
- `NonlinearProcessModel` - Auto-differentiable nonlinear process model
- `GaussianMeasurementModel` - Auto-differentiable measurement model
- `ObjectiveFunction` - Objective with automatic gradient/Hessian
- `AnalyticLinearization` - Gauss-Newton approximation

## Example: Nonlinear Tracking

```python
import jax.numpy as jnp
from jax import random
from laplace.filter import LaplaceFilter
from laplace.models_jax import (
    LinearProcessModel,
    GaussianMeasurementModel,
    AnalyticLinearization,
    ObjectiveFunction
)

# Process model
F = jnp.array([[1.0, 1.0], [0.0, 1.0]])
Q = jnp.eye(2) * 0.01
process = LinearProcessModel(F, Q)

# Nonlinear measurement (no manual Jacobian needed!)
def h(x):
    return jnp.array([x[0]**2])  # Measure position squared

R = jnp.array([[0.1]])
meas_model = GaussianMeasurementModel(h, R)

# Initialize filter
filt = LaplaceFilter(optimizer="trust_region")
x_est = jnp.array([0.0, 1.0])
P_est = jnp.eye(2) * 0.5

# Run filter loop
for k in range(n_steps):
    # Prediction
    x_pred, P_pred = filt.predict(x_est, P_est, process)
    
    # Get measurement
    y = get_measurement()
    nll = meas_model.nll(y)
    
    # Update with automatic linearization
    P_inv_pred = jnp.linalg.inv(P_pred)
    obj = ObjectiveFunction(x_pred, P_inv_pred, nll)
    lin = AnalyticLinearization(obj, meas_model, y)
    
    state = filt.update(x_pred, P_pred, nll, lin=lin)
    x_est, P_est = state.mean, state.cov
```

## Performance Tips

1. **Use JAX arrays**: Convert NumPy arrays with `jnp.asarray()`
2. **JIT compile your functions**: Wrap with `@jit` for speed
3. **Batch operations**: JAX excels at vectorized operations
4. **GPU usage**: Set `CUDA_VISIBLE_DEVICES` to use specific GPUs

## Compatibility

- The original NumPy-based implementations (`laplace.models`, `laplace.math`) remain available
- JAX arrays and NumPy arrays can be mixed (with automatic conversion)
- Check JAX availability: `from laplace import JAX_AVAILABLE`

## Running the Example

```bash
uv run python examples/jax_example.py
```

This demonstrates:
- Automatic Jacobian computation
- JAX-accelerated filtering
- Performance comparison
- Automatic differentiation capabilities

## Requirements

- Python >= 3.9
- JAX >= 0.4.0
- jaxlib >= 0.4.0

Install with:
```bash
uv sync  # Installs JAX automatically
```

## Further Reading

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [JAX GPU Tutorial](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html)

