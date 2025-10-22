"""Linear algebra helpers and numerical utilities for Laplace filtering."""

import jax.numpy as jnp
from jax.numpy.linalg import cholesky, inv, solve

from laplace.types import Array


def _ensure_jax(x: Array) -> Array:
    """Convert to JAX array if needed."""
    if hasattr(x, '__array__'):  # numpy array
        return jnp.array(x)
    return x


def ensure_symmetric(A: Array) -> Array:
    """
    Ensure matrix is symmetric by averaging with its transpose.

    Args:
        A: Square matrix

    Returns:
        Symmetric matrix (A + A.T) / 2
    """
    return 0.5 * (A + A.T)


def sqrt_inv_to_cov(S: Array) -> Array:
    """
    Convert square-root information form S (where P^{-1} = S S^T) to covariance P.

    Args:
        S: Square-root information matrix

    Returns:
        Covariance matrix P
    """
    # P^{-1} = S S^T, so P = (S^T)^{-1} S^{-1}
    # More stable: solve P S^T = S^{-T}, i.e., P = solve(S^T, solve(S, I))
    n = S.shape[0]
    identity = jnp.eye(n)
    S_inv = solve(S, identity)
    P = S_inv.T @ S_inv
    return ensure_symmetric(P)


def cov_to_sqrt_inv(P: Array) -> Array:
    """
    Convert covariance P to square-root information form S (where P^{-1} = S S^T).

    Args:
        P: Covariance matrix

    Returns:
        Square-root information matrix S
    """
    # Use Cholesky: P = L L^T, so P^{-1} = (L^T)^{-1} L^{-1} = (L^{-1})^T L^{-1}
    # Thus S = L^{-T} (upper triangular)
    L = cholesky(P)  # P = L L^T
    n = L.shape[0]
    L_inv = solve(L, jnp.eye(n))
    S = L_inv.T  # S = L^{-T}
    return S


def mahalanobis_squared(x: Array, mean: Array, P_inv: Array) -> float:
    """
    Compute squared Mahalanobis distance: (x - mean)^T P_inv (x - mean).

    Args:
        x: State vector
        mean: Mean vector
        P_inv: Inverse covariance (information matrix)

    Returns:
        Squared Mahalanobis distance
    """
    diff = x - mean
    return float(diff.T @ P_inv @ diff)


def woodbury_identity(A_inv: Array, U: Array, C: Array, V: Array) -> Array:
    """
    Woodbury matrix identity for efficient inversion updates.

    (A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}

    Args:
        A_inv: Inverse of A
        U: Update matrix
        C: Perturbation matrix
        V: Update matrix

    Returns:
        Inverse of (A + U C V)
    """
    C_inv = inv(C)
    inner = C_inv + V @ A_inv @ U
    inner_inv = inv(inner)
    return A_inv - A_inv @ U @ inner_inv @ V @ A_inv


def numerical_gradient(f: callable, x: Array, eps: float = 1e-7) -> Array:
    """
    Compute numerical gradient using central differences.

    Args:
        f: Scalar-valued function
        x: Point to evaluate gradient
        eps: Finite difference step size

    Returns:
        Gradient vector
    """
    x = _ensure_jax(x)
    n = len(x)
    grad = jnp.zeros(n)
    for i in range(n):
        x_plus = x.at[i].add(eps)
        x_minus = x.at[i].add(-eps)
        grad_i = (f(x_plus) - f(x_minus)) / (2 * eps)
        grad = grad.at[i].set(grad_i)
    return grad


def numerical_hessian(f: callable, x: Array, eps: float = 1e-5) -> Array:
    """
    Compute numerical Hessian using finite differences.

    Args:
        f: Scalar-valued function
        x: Point to evaluate Hessian
        eps: Finite difference step size

    Returns:
        Hessian matrix
    """
    x = _ensure_jax(x)
    n = len(x)
    H = jnp.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x_pp = x.at[i].add(eps).at[j].add(eps)
            x_pm = x.at[i].add(eps).at[j].add(-eps)
            x_mp = x.at[i].add(-eps).at[j].add(eps)
            x_mm = x.at[i].add(-eps).at[j].add(-eps)

            H_ij = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps)
            H = H.at[i, j].set(H_ij)
            if i != j:
                H = H.at[j, i].set(H_ij)

    return ensure_symmetric(H)


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
        cholesky(A)
        return True
    except (ValueError, Exception):
        # Fall back to eigenvalue check (JAX may raise different exceptions)
        eigvals = jnp.linalg.eigvalsh(ensure_symmetric(A))
        return bool(jnp.all(eigvals > tol))


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
