# Laplax

A Python library for **Bayesian state estimation** using the Laplace filter and classical Kalman filtering variants (KF, EKF, UKF).

**Built on JAX** for automatic differentiation, JIT compilation, and GPU/TPU support.

## Getting Started

### Prerequisites

* Python 3.11+
* UV installed
* **JAX** (automatically installed with dependencies)

### Install

```bash
pip install uv
uv sync                 # install runtime deps (including JAX)
uv sync --extra dev     # install dev deps (ruff, mypy, pytest, etc.)
```

### Development

```bash
# Run tests
uv run pytest

# Lint & format
uv run ruff check src tests
uv run ruff check --fix src tests
uv run ruff format src tests

# Type check
uv run mypy src

# One-liner
uv run ruff check --fix src tests && uv run ruff format src tests && uv run mypy src
```

## Project Structure

```
laplax/
├── src/
│   └── laplace/
│       ├── filter.py        # All filters: Laplace, KF, EKF, UKF
│       ├── models.py        # Process & measurement models (JAX-based)
│       ├── math.py          # Linear algebra helpers (JAX autodiff)
│       ├── optim.py         # Trust-region / BFGS optimizers
│       └── types.py         # Protocols / dataclasses
├── tests/                   # Unit tests
├── examples/                # Usage examples
│   ├── simple_example.py    # Basic 1D tracking
│   ├── tracking_example.py  # 2D nonlinear tracking
│   ├── filter_comparison.py # Compare KF/EKF/UKF/Laplace
│   └── robust_comparison.py # Outlier robustness test
├── pyproject.toml
├── .python-version
└── README.md
```

---

## How It Works

### Laplace Filter Overview

We estimate state (x_k) with process model (x_k = f_{k-1}(x_{k-1}, w_{k-1})) and measurement model (y_k \sim p(y_k \mid x_k)) (not necessarily Gaussian).

Maintain a **Gaussian prediction** (x_k \sim \mathcal N(\mu_{k|k-1}, P_{k|k-1})).
Form the negative log-posterior
[
V(x_k) ;=; \tfrac{1}{2}|x_k-\mu_{k|k-1}|^2_{P_{k|k-1}^{-1}} ;-; \log p(y_k\mid x_k).
]
The **Laplace update** computes the MAP and local curvature:
[
\mu_{k|k} ;=; \arg\min_{x_k} V(x_k), \qquad
P_{k|k} ;=; \left[\nabla^2 V(x_k)\right]^{-1}\Big|*{x_k=\mu*{k|k}}.
]

**Square-root form (numerically stable):** maintain (S) with (P^{-1}=SS^\top) and update ((\mu, S)) using a trust-region/BFGS step.

### One-Step Filtering Recipe (time k)

1. **Predict:** From ((\mu_{k-1|k-1}, P_{k-1|k-1})) and process model (f), produce ((\mu_{k|k-1}, P_{k|k-1})).
2. **Update (Laplace):** Minimize (V) to get (\mu_{k|k}); invert Hessian at the mode to get (P_{k|k}).

   * Square-root variant updates (S_{k|k-1}!\to!S_{k|k}) directly.

### Interfaces Expected by `laplax.filter.LaplaceFilter`

```python
# src/laplace/types.py
from typing import Protocol
from jax import Array  # JAX arrays for autodiff

class ProcessModel(Protocol):
    def __call__(self, x_prev: Array, w: Array) -> Array: ...

class MeasurementLogLik(Protocol):
    def __call__(self, x: Array) -> Array:
        """Return -log p(y_k | x) (JAX scalar)."""

class Linearization(Protocol):
    def hessian(self, x: Array) -> Array: ...
    def grad(self, x: Array) -> Array: ...
```

```python
# src/laplace/filter.py
from dataclasses import dataclass
from jax import Array

@dataclass
class FilterState:
    mean: Array            # μ_{k|k}
    cov: Array             # P_{k|k}
    sqrt_inv: Array | None # S s.t. P^{-1} = S Sᵀ (optional)

class LaplaceFilter:
    def predict(self, mean_prev: Array, cov_prev: Array, process: ProcessModel
               ) -> tuple[Array, Array]:
        """Return (μ_{k|k-1}, P_{k|k-1})."""

    def update(self, mu_pred: Array, P_pred: Array,
               nll: MeasurementLogLik, lin: Linearization | None = None
               ) -> FilterState:
        """Compute μ_{k|k} = argmin_x V(x), P_{k|k} = (∂²V/∂x²)^{-1} at μ_{k|k}."""

class SquareRootLaplaceFilter(LaplaceFilter):
    def update_sqrt(self, mu_pred: Array, S_pred: Array,
                    nll: MeasurementLogLik
                    ) -> FilterState:
        """Trust-region BFGS in S-space; returns (μ_{k|k}, S_{k|k}, P_{k|k})."""
```

### Optimization Notes (Update Step)

* Objective: (V(x)=\tfrac12|x-\mu_{k|k-1}|^2_{P_{k|k-1}^{-1}} + \underbrace{\big(-\log p(y_k|x)\big)}_{\text{measurement NLL}}).
* Gradient/Hessian combine the prior term ((P_{k|k-1}^{-1}(x-\mu_{k|k-1}))) and measurement derivatives.
* At convergence, set (P_{k|k} = \big[\nabla^2 V(\mu_{k|k})\big]^{-1}).
* Square-root variant updates (S) with a trust region for robustness.

### Likelihood Suggestions

* **Euclidean measurements:** Gaussian (p(y|x)=\mathcal N(h(x), R)).
* **Directional/spherical measurements:** von Mises–Fisher on (S^n).
* Start with well-calibrated (R) (or concentration κ for vMF). Ensure (V) is locally convex near the solution.

---

## Quick Example

```python
import jax.numpy as jnp
from laplax.filter import LaplaceFilter
from laplax.models import LinearProcessModel, GaussianMeasurementModel

# Process model: x_k = F x_{k-1} + w
F = jnp.array([[1.0, 1.0], [0.0, 1.0]])  # Constant velocity
Q = jnp.eye(2) * 0.01
process = LinearProcessModel(F, Q)

# Measurement model: y = h(x) + v
def h(x):
    return jnp.array([x[0]])  # Observe position only

R = jnp.array([[0.1]])
meas_model = GaussianMeasurementModel(h, R)

# Filter
filt = LaplaceFilter()
mu_pred, P_pred = filt.predict(mu_prev, P_prev, process)

# Update with measurement y
y = jnp.array([1.0])
nll = meas_model.nll(y)
state = filt.update(mu_pred, P_pred, nll)

print("Estimate:", state.mean)
print("Covariance:", state.cov)
```

Run examples:
```bash
uv run python examples/simple_example.py
uv run python examples/filter_comparison.py
```

---

## Key Features

* **JAX-Powered**: Automatic differentiation for Jacobians and Hessians—no manual derivatives!
* **Multiple Filters**: Laplace, KF, EKF, UKF all in one library
* **Robust Estimation**: Support for non-Gaussian likelihoods (e.g., Student-t for outliers)
* **Numerically Stable**: Square-root information forms available
* **GPU/TPU Ready**: JAX enables hardware acceleration out of the box

## Tips for Contributors / Agents

* All array operations use `jax.numpy` for consistency and autodiff compatibility
* Keep `MeasurementLogLik` functions pure (no side effects) for JIT compilation
* Prefer square-root updates for high condition numbers or ill-scaled problems
* Add unit tests for: gradient correctness, Hessian positive-definiteness, and filter consistency

---
