"""Linear algebra helpers and numerical utilities for Laplace filtering."""

import numpy as np
from numpy.linalg import cholesky, inv, solve

from laplace.types import Array


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
    identity = np.eye(n)
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
    L_inv = solve(L, np.eye(n))
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
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
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
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps * eps)
            H[j, i] = H[i, j]  # Symmetric

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
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue check
        eigvals = np.linalg.eigvalsh(ensure_symmetric(A))
        return bool(np.all(eigvals > tol))


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
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
