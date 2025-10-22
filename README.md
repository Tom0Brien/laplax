Here you go—clean, concise, and ready to paste into `README.md`.

---

# Laplace

A Python project (managed with [UV](https://github.com/astral-sh/uv)) for experimenting with the **Laplace (Laplace-approximation) filter** for Bayesian state estimation.

## Getting Started

### Prerequisites

* Python 3.8+
* UV installed

### Install

```bash
pip install uv
uv sync                 # install runtime deps
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
laplace/
├── src/
│   └── laplace/
│       ├── filter.py        # LaplaceFilter + SquareRootLaplaceFilter
│       ├── models.py        # f_k(·), h_k(·) interfaces + noise types
│       ├── math.py          # linear algebra helpers, Hessian ops
│       ├── optim.py         # trust-region / BFGS routines
│       └── types.py         # Protocols / dataclasses for state & covariances
├── tests/                   # Unit tests
├── pyproject.toml
├── .python-version
└── README.md
```

---

## Technical Reference (agent-friendly)

### Problem Setup

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

### Interfaces Expected by `laplace.filter.LaplaceFilter`

```python
# src/laplace/types.py
from typing import Protocol
import numpy as np

Array = np.ndarray

class ProcessModel(Protocol):
    def __call__(self, x_prev: Array, w: Array) -> Array: ...

class MeasurementLogLik(Protocol):
    def __call__(self, x: Array) -> float:
        """Return -log p(y_k | x)."""

class Linearization(Protocol):
    def hessian(self, x: Array) -> Array: ...
    def grad(self, x: Array) -> Array: ...
```

```python
# src/laplace/filter.py
from dataclasses import dataclass
import numpy as np
Array = np.ndarray

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
import numpy as np
from laplace.filter import LaplaceFilter

def nll_gaussian(y, h, R_inv):
    def _call(x):
        r = y - h(x)
        return 0.5 * r.T @ R_inv @ r  # -log p(y|x) up to a constant
    return _call

# Given: mu_prev, P_prev, process_model, measurement function h, y, and R_inv
filt = LaplaceFilter()
mu_pred, P_pred = filt.predict(mu_prev, P_prev, process_model)
state = filt.update(mu_pred, P_pred, nll_gaussian(y, h, R_inv))

print(state.mean, state.cov)
```

---

## Tips for Contributors / Agents

* Keep `MeasurementLogLik` pure (no side effects); enable JIT/autodiff later if needed.
* Prefer square-root updates for high condition numbers or ill-scaled problems.
* Add unit tests for: gradient correctness, Hessian positive-definiteness at the mode, and filter consistency on synthetic data.

---
