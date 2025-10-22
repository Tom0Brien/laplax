"""Type definitions and protocols for Laplace filtering."""

from dataclasses import dataclass
from typing import Protocol, Union

import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray

# Accept both numpy and JAX arrays
Array = Union[np.ndarray, JaxArray]


class ProcessModel(Protocol):
    """Protocol for process model: x_k = f_{k-1}(x_{k-1}, w_{k-1})."""

    def __call__(self, x_prev: Array, w: Array) -> Array:
        """
        Compute next state from previous state and process noise.

        Args:
            x_prev: Previous state vector
            w: Process noise vector

        Returns:
            Next state vector
        """
        ...


class MeasurementLogLik(Protocol):
    """Protocol for measurement negative log-likelihood."""

    def __call__(self, x: Array) -> float:
        """
        Return -log p(y_k | x).

        Args:
            x: State vector

        Returns:
            Negative log-likelihood value
        """
        ...


class Linearization(Protocol):
    """Protocol for providing derivatives of the objective function."""

    def hessian(self, x: Array) -> Array:
        """
        Compute Hessian matrix ∂²V/∂x² at x.

        Args:
            x: State vector

        Returns:
            Hessian matrix (n x n)
        """
        ...

    def grad(self, x: Array) -> Array:
        """
        Compute gradient ∂V/∂x at x.

        Args:
            x: State vector

        Returns:
            Gradient vector (n,)
        """
        ...


@dataclass
class FilterState:
    """State of the filter after prediction or update."""

    mean: Array  # μ_{k|k} - posterior mean
    cov: Array  # P_{k|k} - posterior covariance
    sqrt_inv: Array | None = (
        None  # S s.t. P^{-1} = S Sᵀ (optional, for square-root form)
    )

    def __post_init__(self) -> None:
        """Validate dimensions."""
        assert self.mean.ndim == 1, "mean must be 1D vector"
        assert self.cov.ndim == 2, "cov must be 2D matrix"
        assert self.cov.shape[0] == self.cov.shape[1], "cov must be square"
        assert self.mean.shape[0] == self.cov.shape[0], (
            "mean and cov dimensions must match"
        )
        if self.sqrt_inv is not None:
            assert self.sqrt_inv.ndim == 2, "sqrt_inv must be 2D matrix"
            assert self.sqrt_inv.shape[0] == self.mean.shape[0], (
                "sqrt_inv dimension must match state dimension"
            )


@dataclass
class NoiseSpec:
    """Specification for noise in process or measurement models."""

    mean: Array  # Noise mean (typically zero)
    cov: Array  # Noise covariance

    def __post_init__(self) -> None:
        """Validate dimensions."""
        assert self.mean.ndim == 1, "mean must be 1D vector"
        assert self.cov.ndim == 2, "cov must be 2D matrix"
        assert self.cov.shape[0] == self.cov.shape[1], "cov must be square"
        assert self.mean.shape[0] == self.cov.shape[0], (
            "mean and cov dimensions must match"
        )
