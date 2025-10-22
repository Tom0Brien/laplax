"""Optimization routines for Laplace filter update step."""

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax.numpy.linalg import norm, solve

from laplace.math import is_positive_definite, regularize_covariance
from laplace.types import Array


@dataclass
class OptimResult:
    """Result of optimization."""

    x: Array  # Optimal point
    f_val: float  # Function value at optimum
    success: bool  # Whether optimization succeeded
    n_iter: int  # Number of iterations
    message: str  # Status message


def line_search_backtracking(
    f: Callable[[Array], float],
    x: Array,
    direction: Array,
    grad: Array,
    alpha_init: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20,
) -> float:
    """
    Backtracking line search with Armijo condition.

    Args:
        f: Objective function
        x: Current point
        direction: Search direction
        grad: Gradient at current point
        alpha_init: Initial step size
        c: Armijo condition parameter
        rho: Backtracking factor
        max_iter: Maximum iterations

    Returns:
        Step size alpha
    """
    alpha = alpha_init
    f_x = f(x)
    directional_deriv = grad.T @ direction

    for _ in range(max_iter):
        x_new = x + alpha * direction
        f_new = f(x_new)

        # Armijo condition
        if f_new <= f_x + c * alpha * directional_deriv:
            return alpha

        alpha *= rho

    return alpha


def trust_region_step(grad: Array, hess: Array, delta: float) -> tuple[Array, bool]:
    """
    Compute trust region step using dogleg method.

    Args:
        grad: Gradient vector
        hess: Hessian matrix
        delta: Trust region radius

    Returns:
        step: Computed step
        on_boundary: Whether step is on trust region boundary
    """
    # Try full Newton step
    try:
        if is_positive_definite(hess):
            p_newton = -solve(hess, grad)
            if norm(p_newton) <= delta:
                return p_newton, False
        else:
            # Hessian not PD, fall back to gradient descent direction
            raise ValueError("Hessian not positive definite")
    except (ValueError, Exception):
        # Use steepest descent direction
        p_newton = None

    # Cauchy point (steepest descent)
    g_norm_sq = grad.T @ grad
    if g_norm_sq < 1e-14:
        return jnp.zeros_like(grad), False

    try:
        curvature = grad.T @ hess @ grad
        if curvature > 0:
            tau = min(1.0, g_norm_sq / curvature)
        else:
            tau = 1.0
    except Exception:
        tau = 1.0

    p_cauchy = -tau * (g_norm_sq / norm(grad)) * grad / norm(grad)

    if norm(p_cauchy) >= delta:
        # Scale to boundary
        return -delta * grad / norm(grad), True

    # Dogleg: interpolate between Cauchy and Newton
    if p_newton is not None:
        p_diff = p_newton - p_cauchy
        a = p_diff.T @ p_diff
        b = 2 * p_cauchy.T @ p_diff
        c = p_cauchy.T @ p_cauchy - delta * delta

        discriminant = b * b - 4 * a * c
        if discriminant >= 0 and a > 0:
            tau_dl = (-b + jnp.sqrt(discriminant)) / (2 * a)
            tau_dl = jnp.clip(tau_dl, 0, 1)
            return p_cauchy + tau_dl * p_diff, True

    return p_cauchy, True


def minimize_trust_region(
    f: Callable[[Array], float],
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array], Array],
    x0: Array,
    delta_init: float = 1.0,
    delta_max: float = 10.0,
    eta: float = 0.15,
    tol_grad: float = 1e-6,
    tol_step: float = 1e-8,
    max_iter: int = 100,
) -> OptimResult:
    """
    Trust region optimization with dogleg step.

    Args:
        f: Objective function
        grad_f: Gradient function
        hess_f: Hessian function
        x0: Initial point
        delta_init: Initial trust region radius
        delta_max: Maximum trust region radius
        eta: Acceptance threshold for step
        tol_grad: Gradient norm tolerance
        tol_step: Step size tolerance
        max_iter: Maximum iterations

    Returns:
        Optimization result
    """
    x = x0.copy()
    delta = delta_init
    f_val = f(x)

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = norm(grad)

        # Check gradient convergence
        if grad_norm < tol_grad:
            return OptimResult(
                x=x, f_val=f_val, success=True, n_iter=k, message="Gradient converged"
            )

        # Compute Hessian and regularize if needed
        hess = hess_f(x)
        if not is_positive_definite(hess):
            hess = regularize_covariance(hess, min_eig=1e-6)

        # Compute trust region step
        step, on_boundary = trust_region_step(grad, hess, delta)
        step_norm = norm(step)

        # Check step size convergence
        if step_norm < tol_step:
            return OptimResult(
                x=x, f_val=f_val, success=True, n_iter=k, message="Step size converged"
            )

        # Evaluate quality of step
        x_new = x + step
        f_new = f(x_new)

        # Predicted reduction (quadratic model)
        pred_reduction = -(grad.T @ step + 0.5 * step.T @ hess @ step)
        actual_reduction = f_val - f_new

        # Compute ratio
        if abs(pred_reduction) < 1e-14:
            rho = 0.0
        else:
            rho = actual_reduction / pred_reduction

        # Update trust region radius
        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and on_boundary:
            delta = min(2 * delta, delta_max)

        # Accept or reject step
        if rho > eta:
            x = x_new
            f_val = f_new
        else:
            # Reject step, decrease trust region
            delta *= 0.5

        # Check for stagnation
        if delta < 1e-12:
            return OptimResult(
                x=x,
                f_val=f_val,
                success=True,
                n_iter=k,
                message="Trust region too small",
            )

    return OptimResult(
        x=x,
        f_val=f_val,
        success=False,
        n_iter=max_iter,
        message="Max iterations reached",
    )


def minimize_bfgs(
    f: Callable[[Array], float],
    grad_f: Callable[[Array], Array],
    x0: Array,
    tol_grad: float = 1e-6,
    tol_step: float = 1e-8,
    max_iter: int = 100,
) -> OptimResult:
    """
    BFGS quasi-Newton optimization.

    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        tol_grad: Gradient norm tolerance
        tol_step: Step size tolerance
        max_iter: Maximum iterations

    Returns:
        Optimization result
    """
    n = len(x0)
    x = x0.copy()
    H = jnp.eye(n)  # Inverse Hessian approximation
    f_val = f(x)
    grad = grad_f(x)

    for k in range(max_iter):
        grad_norm = norm(grad)

        # Check gradient convergence
        if grad_norm < tol_grad:
            return OptimResult(
                x=x, f_val=f_val, success=True, n_iter=k, message="Gradient converged"
            )

        # Compute search direction
        direction = -H @ grad

        # Ensure descent direction
        if grad.T @ direction >= 0:
            direction = -grad
            H = jnp.eye(n)

        # Line search
        alpha = line_search_backtracking(f, x, direction, grad)
        step = alpha * direction
        step_norm = norm(step)

        # Check step size convergence
        if step_norm < tol_step:
            return OptimResult(
                x=x, f_val=f_val, success=True, n_iter=k, message="Step size converged"
            )

        # Update point
        x_new = x + step
        f_val = f(x_new)
        grad_new = grad_f(x_new)

        # BFGS update
        y = grad_new - grad
        s = step
        rho = 1.0 / (y.T @ s + 1e-14)

        if rho > 0:  # Curvature condition satisfied
            identity = jnp.eye(n)
            H = (identity - rho * jnp.outer(s, y)) @ H @ (
                identity - rho * jnp.outer(y, s)
            ) + rho * jnp.outer(s, s)

        x = x_new
        grad = grad_new

    return OptimResult(
        x=x,
        f_val=f_val,
        success=False,
        n_iter=max_iter,
        message="Max iterations reached",
    )
