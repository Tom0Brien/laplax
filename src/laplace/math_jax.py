"""JAX-accelerated linear algebra utilities with automatic differentiation."""

import jax.numpy as jnp
from jax import grad, hessian, jit
from jax.scipy.linalg import cho_factor, cho_solve, solve

from laplace.types import Array


@jit
def ensure_symmetric(A: Array) -> Array:
    """
    Ensure matrix is symmetric by averaging with its transpose.

    Args:
        A: Square matrix

    Returns:
        Symmetric matrix (A + A.T) / 2
    """
    return 0.5 * (A + A.T)


@jit
def sqrt_inv_to_cov(S: Array) -> Array:
    """
    Convert square-root information form S (where P^{-1} = S S^T) to covariance P.

    Args:
        S: Square-root information matrix

    Returns:
        Covariance matrix P
    """
    n = S.shape[0]
    identity = jnp.eye(n)
    S_inv = solve(S, identity, assume_a="pos")
    P = S_inv.T @ S_inv
    return ensure_symmetric(P)


@jit
def cov_to_sqrt_inv(P: Array) -> Array:
    """
    Convert covariance P to square-root information form S (where P^{-1} = S S^T).

    Args:
        P: Covariance matrix

    Returns:
        Square-root information matrix S
    """
    # Use Cholesky: P = L L^T, so P^{-1} = (L^T)^{-1} L^{-1} = (L^{-1})^T L^{-1}
    c, lower = cho_factor(P, lower=True)
    n = P.shape[0]
    L_inv = solve(c, jnp.eye(n), assume_a="pos")
    S = L_inv.T  # S = L^{-T}
    return S


@jit
def mahalanobis_squared(x: Array, mean: Array, P_inv: Array) -> Array:
    """
    Compute squared Mahalanobis distance: (x - mean)^T P_inv (x - mean).

    Args:
        x: State vector
        mean: Mean vector
        P_inv: Inverse covariance (information matrix)

    Returns:
        Squared Mahalanobis distance (JAX scalar)
    """
    diff = x - mean
    return diff.T @ P_inv @ diff


def jacobian_autodiff(f, x: Array) -> Array:
    """
    Compute Jacobian using JAX automatic differentiation.

    Args:
        f: Vector-valued function
        x: Point to evaluate Jacobian

    Returns:
        Jacobian matrix
    """
    from jax import jacfwd

    return jacfwd(f)(x)


def hessian_autodiff(f):
    """
    Create Hessian function using JAX automatic differentiation.

    Args:
        f: Scalar-valued function

    Returns:
        Hessian function
    """
    return hessian(f)


def gradient_autodiff(f):
    """
    Create gradient function using JAX automatic differentiation.

    Args:
        f: Scalar-valued function

    Returns:
        Gradient function
    """
    return grad(f)


@jit
def is_positive_definite(A: Array, tol: float = 1e-10) -> bool:
    """
    Check if matrix is positive definite.

    Args:
        A: Square matrix
        tol: Tolerance for eigenvalue positivity

    Returns:
        True if all eigenvalues > tol
    """
    try:
        # Try Cholesky decomposition - fastest for PD matrices
        cho_factor(A, lower=True)
        return True
    except Exception:
        # Fall back to eigenvalue check
        eigvals = jnp.linalg.eigvalsh(ensure_symmetric(A))
        return bool(jnp.all(eigvals > tol))


@jit
def regularize_covariance(P: Array, min_eig: float = 1e-8) -> Array:
    """
    Regularize covariance matrix to ensure positive definiteness.

    Args:
        P: Covariance matrix
        min_eig: Minimum eigenvalue to enforce

    Returns:
        Regularized covariance matrix
    """
    P_sym = ensure_symmetric(P)
    eigvals, eigvecs = jnp.linalg.eigh(P_sym)
    eigvals = jnp.maximum(eigvals, min_eig)
    return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


@jit
def solve_psd(A: Array, b: Array) -> Array:
    """
    Solve Ax = b for positive semi-definite A using Cholesky.

    Args:
        A: Positive semi-definite matrix
        b: Right-hand side

    Returns:
        Solution x
    """
    c, lower = cho_factor(A, lower=True)
    return cho_solve((c, lower), b)


@jit
def inv_psd(A: Array) -> Array:
    """
    Invert positive semi-definite matrix using Cholesky.

    Args:
        A: Positive semi-definite matrix

    Returns:
        Inverse of A
    """
    n = A.shape[0]
    return solve_psd(A, jnp.eye(n))


@jit
def matrix_sqrt(A: Array) -> Array:
    """
    Compute matrix square root via eigendecomposition.

    Args:
        A: Symmetric positive definite matrix

    Returns:
        Matrix S such that S @ S = A
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 0.0)
    return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T
