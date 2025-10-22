"""Laplace filter for Bayesian state estimation with JAX acceleration."""

from laplace.filter import LaplaceFilter, SquareRootLaplaceFilter
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
    "LaplaceFilter",
    "SquareRootLaplaceFilter",
    "FilterState",
    "NoiseSpec",
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
