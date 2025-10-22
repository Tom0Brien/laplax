"""Tests for math utilities."""

import jax.numpy as jnp
import numpy as np  # For allclose, isclose compatibility

from laplace.math import (
    cov_to_sqrt_inv,
    ensure_symmetric,
    is_positive_definite,
    mahalanobis_squared,
    numerical_gradient,
    numerical_hessian,
    regularize_covariance,
    sqrt_inv_to_cov,
)


def test_ensure_symmetric() -> None:
    """Test symmetrization of matrices."""
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    A_sym = ensure_symmetric(A)
    assert np.allclose(A_sym, A_sym.T)
    assert np.allclose(A_sym, jnp.array([[1.0, 2.5], [2.5, 4.0]]))


def test_sqrt_inv_roundtrip() -> None:
    """Test conversion between covariance and square-root information."""
    P = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    S = cov_to_sqrt_inv(P)
    P_recovered = sqrt_inv_to_cov(S)
    assert np.allclose(P, P_recovered)

    # Verify P^{-1} = S^T S (our convention)
    P_inv = jnp.linalg.inv(P)
    assert np.allclose(P_inv, S @ S.T)


def test_mahalanobis_squared() -> None:
    """Test Mahalanobis distance computation."""
    x = jnp.array([1.0, 2.0])
    mean = jnp.array([0.0, 0.0])
    P = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    P_inv = jnp.linalg.inv(P)

    dist = mahalanobis_squared(x, mean, P_inv)
    expected = 1.0**2 + 2.0**2
    assert np.isclose(dist, expected)


def test_numerical_gradient() -> None:
    """Test numerical gradient computation."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return x[0] ** 2 + 2 * x[1] ** 2 + x[0] * x[1]

    x = jnp.array([1.0, 2.0])
    grad = numerical_gradient(f, x)
    # ∂f/∂x0 = 2*x0 + x1 = 2*1 + 2 = 4
    # ∂f/∂x1 = 4*x1 + x0 = 4*2 + 1 = 9
    assert np.allclose(grad, jnp.array([4.0, 9.0]), atol=1e-5)


def test_numerical_hessian() -> None:
    """Test numerical Hessian computation."""

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return x[0] ** 2 + 2 * x[1] ** 2 + x[0] * x[1]

    x = jnp.array([1.0, 2.0])
    hess = numerical_hessian(f, x)
    # ∂²f/∂x0² = 2, ∂²f/∂x1² = 4, ∂²f/∂x0∂x1 = 1
    expected = jnp.array([[2.0, 1.0], [1.0, 4.0]])
    assert np.allclose(hess, expected, atol=1e-4)


def test_is_positive_definite() -> None:
    """Test positive definiteness check."""
    # Positive definite
    P_pd = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    assert is_positive_definite(P_pd)

    # Not positive definite (negative eigenvalue)
    P_neg = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    assert not is_positive_definite(P_neg)

    # Singular (zero eigenvalue)
    P_sing = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    assert not is_positive_definite(P_sing)


def test_regularize_covariance() -> None:
    """Test covariance regularization."""
    # Singular matrix
    P = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    P_reg = regularize_covariance(P, min_eig=0.1)

    assert is_positive_definite(P_reg)
    eigvals = jnp.linalg.eigvalsh(P_reg)
    # Allow for float32 precision (JAX defaults to float32)
    assert np.all(eigvals >= 0.1 - 1e-6)
