"""Laplace: Bayesian state estimation with JAX."""

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

__version__ = "0.1.0"

__all__ = [
    # Filters
    "LaplaceFilter",
    "SquareRootLaplaceFilter",
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
]
