"""Laplace filter for Bayesian state estimation with JAX acceleration."""

from laplace.filter import (
    ExtendedKalmanFilter,
    KalmanFilter,
    LaplaceFilter,
    SquareRootLaplaceFilter,
    UnscentedKalmanFilter,
)
from laplace.models import (
    AnalyticLinearization,
    GaussianMeasurementModel,
    LinearProcessModel,
    NonlinearProcessModel,
    ObjectiveFunction,
)
from laplace.types import FilterState, NoiseSpec

# JAX-accelerated modules
try:
    from laplace import math_jax, models_jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    math_jax = None
    models_jax = None

__version__ = "0.1.0"

__all__ = [
    # Laplace filters
    "LaplaceFilter",
    "SquareRootLaplaceFilter",
    # Kalman filters
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    # Types
    "FilterState",
    "NoiseSpec",
    # Models
    "LinearProcessModel",
    "NonlinearProcessModel",
    "GaussianMeasurementModel",
    "ObjectiveFunction",
    "AnalyticLinearization",
    # JAX modules (if available)
    "math_jax",
    "models_jax",
    "JAX_AVAILABLE",
]
