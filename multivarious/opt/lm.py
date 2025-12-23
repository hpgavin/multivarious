"""
Levenberg-Marquardt Nonlinear Least Squares Optimization

This module provides a transparent implementation of the Levenberg-Marquardt 
algorithm for nonlinear least squares curve fitting. The algorithm structure 
is kept explicit for educational purposes, showing students the adaptive 
damping strategy and multiple update methods.

The Levenberg-Marquardt algorithm interpolates between Gauss-Newton (good near
the minimum) and gradient descent (good far from the minimum) by adaptively
adjusting a damping parameter λ.

Author: Translated from MATLAB by Claude
Original: Henri Gavin, Duke University
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Any, Literal
from dataclasses import dataclass


@dataclass
class LMResult:
    """
    Results from Levenberg-Marquardt optimization.
    
    Attributes
    ----------
    coefficients : ndarray
        Optimized coefficient values, shape (n,)
    reduced_chi_sq : float
        Reduced chi-squared statistic (should be near 1 for good fit)
    sigma_coefficients : ndarray
        Standard errors of coefficients, shape (n,)
    sigma_fit : ndarray
        Standard errors of fitted curve, shape (m,)
    correlation : ndarray
        Correlation matrix of coefficients, shape (n, n)
    r_squared : float
        Coefficient of determination (0 to 1)
    convergence_history : ndarray
        History of optimization: [iteration, coeffs..., chi_sq, lambda]
    func_calls : int
        Total number of function evaluations
    message : str
        Convergence message
    aic : float
        Akaike Information Criterion
    bic : float
        Bayes Information Criterion
    """
    coefficients: np.ndarray
    reduced_chi_sq: float
    sigma_coefficients: np.ndarray
    sigma_fit: np.ndarray
    correlation: np.ndarray
    r_squared: float
    convergence_history: np.ndarray
    func_calls: int
    message: str
    aic: float 
    bic: float


def levenberg_marquardt(
    func: Callable,
    coeffs_init: np.ndarray,
    t: np.ndarray,
    y_data: np.ndarray,
    weight: Optional[np.ndarray] = None,
    delta_coeffs: Optional[np.ndarray] = None,
    coeffs_lb: Optional[np.ndarray] = None,
    coeffs_ub: Optional[np.ndarray] = None,
    func_args: Tuple = (),
    print_level: int = 0,
    max_iter: Optional[int] = None,
    tol_gradient: float = 1e-3,
    tol_coeffs: float = 1e-3,
    tol_chi_sq: float = 1e-1,
    tol_improvement: float = 1e-1,
    lambda_init: float = 1e-2,
    lambda_up_factor: float = 11.0,
    lambda_dn_factor: float = 9.0,
    update_type: Literal[1, 2, 3] = 1,
    plot_iterations: bool = False
) -> LMResult:
    """
    Levenberg-Marquardt curve fitting: minimize sum of weighted squared residuals.
    
    Solves the nonlinear least squares problem:
        min  Σ weight[i] * (y_data[i] - func(t[i], coeffs))²
    
    using an adaptive damping parameter that interpolates between Gauss-Newton
    and gradient descent methods.
    
    Parameters
    ----------
    func : callable
        Model function: y_hat = func(t, coeffs, *func_args)
        Must return array of shape (m,)
    coeffs_init : ndarray
        Initial guess for coefficient values, shape (n,)
    t : ndarray
        Independent variable(s), shape (m,) or (m, k)
    y_data : ndarray
        Data to be fit, shape (m,)
    weight : ndarray, optional
        Weights for least squares (inverse of measurement variance)
        Shape (m,) or scalar. Default: uniform weights
    delta_coeffs : ndarray, optional
        Fractional increments for numerical derivatives, shape (n,)
        - delta_coeffs[j] > 0: central differences
        - delta_coeffs[j] < 0: one-sided backward differences  
        - delta_coeffs[j] = 0: hold coefficient fixed
        Default: 0.001 for all coefficients
    coeffs_lb : ndarray, optional
        Lower bounds for coefficients, shape (n,)
        Default: -100 * |coeffs_init|
    coeffs_ub : ndarray, optional
        Upper bounds for coefficients, shape (n,)
        Default: +100 * |coeffs_init|
    func_args : tuple, optional
        Additional arguments passed to func
    print_level : int, optional
        Verbosity: 0=silent, 1=final, 2=iteration, 3=detailed
        Default: 0
    max_iter : int, optional
        Maximum function evaluations. Default: 10 * n²
    tol_gradient : float, optional
        Convergence tolerance for gradient. Default: 1e-3
    tol_coeffs : float, optional
        Convergence tolerance for coefficient changes. Default: 1e-3
    tol_chi_sq : float, optional
        Convergence tolerance for chi-squared changes. Default: 1e-1
    tol_improvement : float, optional
        Acceptance threshold for trial step. Default: 1e-1
    lambda_init : float, optional
        Initial damping parameter. Default: 1e-2
    lambda_up_factor : float, optional
        Factor for increasing lambda (reject step). Default: 11
    lambda_dn_factor : float, optional
        Factor for decreasing lambda (accept step). Default: 9
    update_type : {1, 2, 3}, optional
        Lambda update strategy:
        1: Levenberg-Marquardt (default)
        2: Quadratic 
        3: Nielsen
    plot_iterations : bool, optional
        If True, plot data and current fit at each iteration (like MATLAB prnt>2)
        Default: False
    
    Returns
    -------
    result : LMResult
        Optimization results including coefficients, statistics, and history
    
    Notes
    -----
    Convergence criteria (any one triggers termination):
    1. ||gradient|| < tol_gradient
    2. ||Δcoeffs|| < tol_coeffs * ||coeffs||
    3. |Δχ²| < tol_chi_sq * χ²
    4. func_calls >= max_iter
    
    The damping parameter λ is adjusted based on the improvement ratio:
        ρ = (actual reduction) / (predicted reduction)
    - If ρ > tol_improvement: accept step, decrease λ
    - If ρ ≤ tol_improvement: reject step, increase λ
    
    Three update strategies for λ:
    1. Marquardt: λ affects only diagonal of JᵀWJ
    2. Quadratic/Nielsen: λ added to all diagonal elements
    3. Nielsen: includes adaptive ν parameter
    
    Examples
    --------
    >>> import numpy as np
    >>> # Define model function
    >>> def exponential(t, coeffs, *args):
    ...     return coeffs[0] * np.exp(-t / coeffs[1])
    >>> 
    >>> # Generate noisy data
    >>> t = np.linspace(0, 5, 100)
    >>> y_true = 10 * np.exp(-t / 2)
    >>> y_data = y_true + 0.5 * np.random.randn(len(t))
    >>> 
    >>> # Fit the model
    >>> result = levenberg_marquardt(
    ...     exponential, 
    ...     coeffs_init=np.array([5.0, 1.0]),
    ...     t=t, 
    ...     y_data=y_data,
    ...     print_level=1
    ... )
    >>> print(f"Fitted: {result.coefficients}")
    >>> print(f"Errors: {result.sigma_coefficients}")
    
    References
    ----------
    .. [1] Levenberg, K. (1944) "A method for the solution of certain non-linear 
           problems in least squares", Quarterly of Applied Mathematics 2: 164-168.
    .. [2] Marquardt, D. (1963) "An algorithm for least-squares estimation of 
           nonlinear parameters", SIAM Journal on Applied Mathematics 11(2): 431-441.
    .. [3] Press, W.H., et al. (1992) "Numerical Recipes", Cambridge Univ. Press, Ch. 15.
    .. [4] Madsen, K., Nielsen, H.B., Tingleff, O. (2004) "Methods for Non-Linear 
           Least Squares Problems", IMM, Technical University of Denmark.
    """
    # ========================================================================
    # Initialize parameters
    # ========================================================================
    
    # Convert to numpy arrays and ensure proper shapes
    coeffs = np.atleast_1d(coeffs_init).astype(float).flatten()
    t = np.atleast_1d(t)
    y_data = np.atleast_1d(y_data).astype(float).flatten()
    
    n_coeffs = len(coeffs)
    n_points = len(y_data)
    dof = n_points - n_coeffs  # degrees of freedom
    
    # Validate dimensions
    if t.ndim == 1 and len(t) != n_points:
        raise ValueError(f"Length of t ({len(t)}) must match length of y_data ({n_points})")
    
    # Set default parameters
    if weight is None:
        weight = 1.0 / (y_data.T @ y_data) if n_points > 0 else 1.0
    
    if delta_coeffs is None:
        delta_coeffs = 0.001 * np.ones(n_coeffs)
    else:
        delta_coeffs = np.atleast_1d(delta_coeffs).flatten()
        if len(delta_coeffs) == 1:
            delta_coeffs = delta_coeffs[0] * np.ones(n_coeffs)
    
    if coeffs_lb is None:
        coeffs_lb = -100 * np.abs(coeffs)
    else:
        coeffs_lb = np.atleast_1d(coeffs_lb).flatten()
    
    if coeffs_ub is None:
        coeffs_ub = 100 * np.abs(coeffs)
    else:
        coeffs_ub = np.atleast_1d(coeffs_ub).flatten()
    
    if max_iter is None:
        max_iter = 10 * n_coeffs ** 2
    
    # Handle weights
    weight = np.atleast_1d(weight).flatten()
    if len(weight) == 1:
        weight = np.abs(weight[0]) * np.ones(n_points)
        if print_level >= 1:
            print("Using uniform weights for error analysis")
    else:
        weight = np.abs(weight)
    
    # Identify coefficients to fit (where delta_coeffs != 0)
    fit_idx = delta_coeffs != 0
    n_fit = np.sum(fit_idx)
    
    # ========================================================================
    # Initialize variables
    # ========================================================================
    iteration = 0
    func_calls = 0
    stop = False
    
    coeffs_old = np.zeros(n_coeffs)
    y_old = np.zeros(n_points)
    chi_sq = 1e-3 / np.finfo(float).eps  # Very large initial value
    chi_sq_old = chi_sq
    J = np.zeros((n_points, n_coeffs))  # Jacobian matrix
    
    # Convergence history: [iteration, coeffs..., chi_sq, lambda]
    cvg_history = np.zeros((max_iter, n_coeffs + 3))
    
    # Setup for iteration plotting (like MATLAB prnt > 2)
    if plot_iterations or print_level > 2:
        plt.ion()
        fig_iter, ax_iter = plt.subplots(figsize=(10, 6))
        y_init = func(t, coeffs, *func_args)  # Initial model for plotting
    
    # Initialize Jacobian and matrices
    JtWJ, JtWdy, chi_sq, y_hat, J, calls_used = _compute_matrices(
        func, t, coeffs_old, y_old, 1, J, coeffs, y_data, weight, delta_coeffs, func_args, 0
    )
    func_calls += calls_used
    
    # Check initial gradient
    if np.max(np.abs(JtWdy)) < tol_gradient:
        message = (f"Initial guess meets gradient convergence criterion "
                  f"(||gradient|| = {np.max(np.abs(JtWdy)):.3e} < {tol_gradient:.3e})")
        if print_level >= 1:
            print(f" *** {message}")
            print(f" *** To converge further, reduce tol_gradient and restart")
        stop = True
    
    # Initialize lambda (damping parameter)
    if update_type == 1:  # Marquardt
        lambda_param = lambda_init
    else:  # Quadratic and Nielsen
        lambda_param = lambda_init * np.max(np.diag(JtWJ))
        nu = 2.0
    
    # Initialize alpha for Quadratic update (will be recomputed each iteration)
    alpha = 1.0
    
    chi_sq_old = chi_sq
    
    # ========================================================================
    # Main optimization loop
    # ========================================================================
    if print_level >= 2:
        print("\n" + "="*80)
        print(f"{'Iter':<6} {'Chi-sq':>12} {'Lambda':>12} {'||grad||':>12} {'||Δcoeffs||':>12}")
        print("="*80)
    
    while not stop and func_calls < max_iter:
        iteration += 1
        
        # ====================================================================
        # Compute update step
        # ====================================================================
        if update_type == 1:  # Marquardt
            # Augment diagonal elements: (JᵀWJ + λ*diag(JᵀWJ))
            X = JtWJ + lambda_param * np.diag(np.diag(JtWJ))
        else:  # Quadratic and Nielsen
            # Augment with identity: (JᵀWJ + λ*I)
            X = JtWJ + lambda_param * np.eye(n_coeffs)
        
        # Ensure matrix is well-conditioned
        while np.linalg.cond(X) > 1e15:
            X = X + 1e-6 * np.sum(np.diag(X)) / n_coeffs * np.eye(n_coeffs)
        
        # Solve for update step: X * h = JᵀWdy
        h = np.linalg.solve(X, JtWdy)
        
        # Scale down if matrix is ill-conditioned
        if np.linalg.cond(X) > 1e14:
            h = 0.1 * h
        
        # ====================================================================
        # Trial step
        # ====================================================================
        coeffs_try = coeffs.copy()
        coeffs_try[fit_idx] = coeffs[fit_idx] + h[fit_idx]
        
        # Apply bounds
        coeffs_try = np.clip(coeffs_try, coeffs_lb, coeffs_ub)
        
        # Evaluate model with trial coefficients
        delta_y = y_data - func(t, coeffs_try, *func_args)
        func_calls += 1
        
        # Check for numerical errors
        if not np.all(np.isfinite(delta_y)):
            message = "Floating point error in function evaluation"
            if print_level >= 1:
                print(f" *** {message}")
            stop = True
            break
        
        # Compute chi-squared for trial step
        chi_sq_try = delta_y.T @ (delta_y * weight)
        
        # ====================================================================
        # Quadratic line search (Update_Type == 2 only)
        # ====================================================================
        if update_type == 2:
            # One step of quadratic line update in h direction for minimum chi-sq
            alpha = JtWdy.T @ h / ((chi_sq_try - chi_sq) / 2 + 2 * JtWdy.T @ h)
            h = alpha * h
            
            # Re-evaluate with scaled step
            coeffs_try = coeffs.copy()
            coeffs_try[fit_idx] = coeffs[fit_idx] + h[fit_idx]
            coeffs_try = np.clip(coeffs_try, coeffs_lb, coeffs_ub)
            
            delta_y = y_data - func(t, coeffs_try, *func_args)
            func_calls += 1
            chi_sq_try = delta_y.T @ (delta_y * weight)
        
        # ====================================================================
        # Evaluate step quality (improvement ratio rho)
        # ====================================================================
        if update_type == 1:  # Levenberg-Marquardt
            rho = (chi_sq - chi_sq_try) / abs(h.T @ (lambda_param * np.diag(np.diag(JtWJ)) @ h + JtWdy))
        else:  # Quadratic and Nielsen
            rho = (chi_sq - chi_sq_try) / abs(h.T @ (lambda_param * h + JtWdy))
        
        # ====================================================================
        # Accept or reject step based on improvement
        # ====================================================================
        if rho > tol_improvement:  # ACCEPT STEP - significantly better
            dchi_sq = chi_sq - chi_sq_old
            chi_sq_old = chi_sq
            coeffs_old = coeffs.copy()
            y_old = y_hat.copy()
            coeffs = coeffs_try.copy()
            
            # Recompute matrices with new coefficients
            JtWJ, JtWdy, chi_sq, y_hat, J, calls_used = _compute_matrices(
                func, t, coeffs_old, y_old, dchi_sq, J, coeffs, y_data, 
                weight, delta_coeffs, func_args, iteration
            )
            func_calls += calls_used
            
            # Decrease lambda (move toward Gauss-Newton)
            if update_type == 1:  # Levenberg-Marquardt
                lambda_param = max(lambda_param / lambda_dn_factor, 1e-7)
            elif update_type == 2:  # Quadratic
                lambda_param = max(lambda_param / (1 + alpha), 1e-7)
            elif update_type == 3:  # Nielsen
                lambda_param = lambda_param * max(1/3, 1 - (2*rho - 1)**3)
                nu = 2.0
            
            # Plot current fit if requested
            if plot_iterations or print_level > 2:
                ax_iter.clear()
                ax_iter.plot(t if t.ndim == 1 else t[:, 0], y_init, '-k', 
                           linewidth=1, label='Initial')
                ax_iter.plot(t if t.ndim == 1 else t[:, 0], y_hat, '-b', 
                           linewidth=2, label='Current fit')
                ax_iter.plot(t if t.ndim == 1 else t[:, 0], y_data, 'o', 
                           color=[0, 0.6, 0], markersize=4, label='Data')
                ax_iter.set_title(f'Iteration {iteration}: $\\chi^2_\\nu$ = {chi_sq/dof:.6f}')
                ax_iter.legend()
                ax_iter.grid(True, alpha=0.3)
                plt.pause(0.01)
        
        else:  # REJECT STEP - not better
            chi_sq = chi_sq_old  # Restore old chi-squared
            
            # If we're at a multiple of 2*n_coeffs iterations, recompute Jacobian
            if iteration % (2 * n_coeffs) == 0:
                JtWJ, JtWdy, _, y_hat, J, calls_used = _compute_matrices(
                    func, t, coeffs_old, y_old, -1, J, coeffs, y_data,
                    weight, delta_coeffs, func_args, iteration
                )
                func_calls += calls_used
            
            # Increase lambda (move toward gradient descent)
            if update_type == 1:  # Levenberg-Marquardt
                lambda_param = min(lambda_param * lambda_up_factor, 1e7)
            elif update_type == 2:  # Quadratic
                lambda_param = lambda_param + abs((chi_sq_try - chi_sq) / 2 / alpha)
            elif update_type == 3:  # Nielsen
                lambda_param = lambda_param * nu
                nu = 2 * nu
        
        # ====================================================================
        # Print iteration details
        # ====================================================================
        if print_level > 1:
            print(f">{iteration:3d}:{func_calls:3d} | chi_sq={chi_sq/dof:10.3e} | lambda={lambda_param:8.1e}")
            print(f"      a  : ", end='')
            for coeff in coeffs:
                print(f" {coeff:10.3e}", end='')
            print()
            print(f"    da/a : ", end='')
            for i, coeff in enumerate(coeffs):
                rel_change = h[i] / coeff if abs(coeff) > 1e-12 else h[i]
                print(f" {rel_change:10.3e}", end='')
            print()
        
        # ====================================================================
        # Record convergence history
        # ====================================================================
        cvg_history[iteration-1, 0] = func_calls
        cvg_history[iteration-1, 1:n_coeffs+1] = coeffs
        cvg_history[iteration-1, n_coeffs+1] = chi_sq / dof
        cvg_history[iteration-1, n_coeffs+2] = lambda_param
        
        # ====================================================================
        # Check convergence criteria (MATLAB-style)
        # ====================================================================
        
        # Gradient convergence
        if np.max(np.abs(JtWdy)) < tol_gradient and iteration > 2:
            if print_level >= 1:
                print(' **** Convergence in r.h.s. ("JtWdy")  ****')
                print(f' **** tol_gradient = {tol_gradient:e}')
            message = f"Gradient convergence"
            stop = True
        
        # Coefficient convergence (element-wise relative change - MATLAB style)
        if np.max(np.abs(h) / (np.abs(coeffs) + 1e-12)) < tol_coeffs and iteration > 2:
            if print_level >= 1:
                print(' **** Convergence in Parameters ****')
                print(f' **** tol_coeffs = {tol_coeffs:e}')
            message = f"Parameter convergence"
            stop = True
        
        # Chi-squared convergence (absolute criterion on reduced chi-sq - MATLAB style)
        if chi_sq / dof < tol_chi_sq and iteration > 2:
            if print_level >= 1:
                print(' **** Convergence in reduced Chi-square  ****')
                print(f' **** tol_chi_sq = {tol_chi_sq:e}')
            message = f"Chi-squared convergence"
            stop = True
        
        # Maximum iterations
        if func_calls >= max_iter:
            if print_level >= 1:
                print(' !! Maximum Number of Function Calls Reached Without Convergence !!')
            message = f"Maximum iterations reached ({max_iter})"
            stop = True
    
    # ========================================================================
    # Convergence achieved - final computations
    # ========================================================================
    
    # Recompute weights if they were uniform (MATLAB lines 299-301)
    if np.var(weight) == 0:
        delta_y_final = y_data - y_hat
        weight = dof / (delta_y_final.T @ delta_y_final) * np.ones(n_points)
    
    # Final matrix computation
    JtWJ, JtWdy, _, y_hat, J, calls_used = _compute_matrices(
        func, t, coeffs_old, y_old, -1, J, coeffs, y_data, weight, delta_coeffs, func_args, iteration
    )
    func_calls += calls_used
    # Note: The chi_sq from this call is always = dof, so we ignore it
    
    # Reduced chi-squared
    reduced_chi_sq = chi_sq / dof if dof > 0 else chi_sq
    log_likelihood = -0.5*chi_sq

    aic = 2*n_coeffs - 2*log_likelihood
    # Small sample correction
    if n_points / n_coeffs < 40:
        aic += (2 * n_coeffs * (n_coeffs + 1)) / (n_points - n_coeffs - 1)

    bic = n_coeffs * np.log(n_points) - 2*log_likelihood
    
    # Covariance matrix
    if np.linalg.cond(JtWJ) > 1e15:
        covar = np.linalg.inv(JtWJ + 1e-6 * np.sum(np.diag(JtWJ)) / n_coeffs * np.eye(n_coeffs))
    else:
        covar = np.linalg.inv(JtWJ)
    
    # Standard errors of coefficients
    sigma_coeffs = np.sqrt(np.diag(covar))
    
    # Standard errors of fit
    sigma_fit = np.zeros(n_points)
    for i in range(n_points):
        sigma_fit[i] = np.sqrt(J[i, :] @ covar @ J[i, :].T)
    
    # Coefficient correlation matrix
    correlation = covar / np.outer(sigma_coeffs, sigma_coeffs)
    
    # R-squared (coefficient of determination)
    r_squared = np.corrcoef(y_data, y_hat)[0, 1] ** 2
    
    # Trim convergence history
    cvg_history = cvg_history[:iteration, :]
    
    # ========================================================================
    # Print final results
    # ========================================================================
    if print_level >= 1:
        print("="*80)
        print(f"Convergence: {message}")
        print(f"Function calls: {func_calls}")
        print(f"Reduced χ²: {reduced_chi_sq:.6f}")
        print(f"R²: {r_squared:.6f}")
        print(f"AIC: {aic:.6f}")
        print(f"BIC: {bic:.6f}")
        if print_level >= 2:
            print("\nFinal coefficients:")
            print(f"{'Index':<8} {'Value':>15} {'Std Error':>15} {'Rel Error %':>15}")
            print("-"*60)
            for i in range(n_coeffs):
                rel_err = 100 * sigma_coeffs[i] / abs(coeffs[i]) if coeffs[i] != 0 else np.inf
                print(f"{i:<8d} {coeffs[i]:15.6e} {sigma_coeffs[i]:15.6e} {rel_err:15.2f}")
        print("="*80 + "\n")
    
    return LMResult(
        coefficients=coeffs,
        reduced_chi_sq=reduced_chi_sq,
        sigma_coefficients=sigma_coeffs,
        sigma_fit=sigma_fit,
        correlation=correlation,
        r_squared=r_squared,
        convergence_history=cvg_history,
        func_calls=func_calls,
        message=message,
        aic=aic,
        bic=bic
    )


def _compute_jacobian_fd(
    func: Callable,
    t: np.ndarray,
    coeffs: np.ndarray,
    y: np.ndarray,
    delta_coeffs: np.ndarray,
    func_args: Tuple
) -> Tuple[np.ndarray, int]:
    """
    Compute Jacobian matrix using finite differences.
    
    Parameters
    ----------
    func : callable
        Model function
    t : ndarray
        Independent variables
    coeffs : ndarray
        Current coefficient values
    y : ndarray
        Model evaluated at current coefficients
    delta_coeffs : ndarray
        Fractional increments for derivatives
    func_args : tuple
        Additional arguments for func
    
    Returns
    -------
    J : ndarray
        Jacobian matrix, shape (m, n)
    func_calls : int
        Number of function evaluations used
    """
    m = len(y)
    n = len(coeffs)
    
    J = np.zeros((m, n))
    coeffs_save = coeffs.copy()
    func_calls = 0
    
    for j in range(n):
        delta = delta_coeffs[j] * (1 + abs(coeffs[j]))
        
        if delta != 0:
            # Forward evaluation
            coeffs[j] = coeffs_save[j] + delta
            y1 = func(t, coeffs, *func_args)
            func_calls += 1
            
            if delta_coeffs[j] < 0:  # Backward difference
                J[:, j] = (y1 - y) / delta
            else:  # Central difference
                coeffs[j] = coeffs_save[j] - delta
                y2 = func(t, coeffs, *func_args)
                func_calls += 1
                J[:, j] = (y1 - y2) / (2 * delta)
        
        coeffs[j] = coeffs_save[j]
    
    return J, func_calls


def _compute_jacobian_broyden(
    coeffs_old: np.ndarray,
    y_old: np.ndarray,
    J: np.ndarray,
    coeffs: np.ndarray,
    y: np.ndarray
) -> np.ndarray:
    """
    Rank-1 update to Jacobian using Broyden's method.
    
    This saves function evaluations by updating the Jacobian incrementally
    rather than recomputing from scratch.
    
    Parameters
    ----------
    coeffs_old : ndarray
        Previous coefficient values
    y_old : ndarray
        Model at previous coefficients
    J : ndarray
        Current Jacobian
    coeffs : ndarray
        Current coefficient values
    y : ndarray
        Model at current coefficients
    
    Returns
    -------
    J : ndarray
        Updated Jacobian matrix
    """
    h = coeffs - coeffs_old
    J = J + np.outer(y - y_old - J @ h, h) / (h.T @ h)
    return J


def _compute_matrices(
    func: Callable,
    t: np.ndarray,
    coeffs_old: np.ndarray,
    y_old: np.ndarray,
    dchi_sq: float,
    J: np.ndarray,
    coeffs: np.ndarray,
    y_data: np.ndarray,
    weight: np.ndarray,
    delta_coeffs: np.ndarray,
    func_args: Tuple,
    iteration: int
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, int]:
    """
    Compute linearized Hessian (JᵀWJ) and gradient (JᵀWdy).
    
    Uses either finite differences or Broyden update for Jacobian.
    
    Parameters
    ----------
    iteration : int
        Current iteration number (needed for Jacobian update strategy)
    
    Returns
    -------
    JtWJ : ndarray
        Approximate Hessian matrix, shape (n, n)
    JtWdy : ndarray
        Gradient vector, shape (n,)
    chi_sq : float
        Chi-squared error
    y_hat : ndarray
        Model prediction, shape (m,)
    J : ndarray
        Jacobian matrix, shape (m, n)
    func_calls_used : int
        Number of function evaluations used
    """
    n_coeffs = len(coeffs)
    
    # Evaluate model (costs 1 function call)
    y_hat = func(t, coeffs, *func_args)
    func_calls_used = 1
    
    # Update Jacobian (finite differences every 2n iterations or if chi-sq increased)
    if iteration % (2 * n_coeffs) == 0 or dchi_sq > 0:
        J, jac_calls = _compute_jacobian_fd(func, t, coeffs, y_hat, delta_coeffs, func_args)
        func_calls_used += jac_calls
    else:
        J = _compute_jacobian_broyden(coeffs_old, y_old, J, coeffs, y_hat)
    
    # Residuals
    delta_y = y_data - y_hat
    
    # Chi-squared
    chi_sq = delta_y.T @ (delta_y * weight)
    
    # Weighted matrices
    W_diag = weight
    JtWJ = J.T @ (J * W_diag[:, np.newaxis])
    JtWdy = J.T @ (W_diag * delta_y)
    
    return JtWJ, JtWdy, chi_sq, y_hat, J, func_calls_used


# Convenience function with simpler interface
def lm(func: Callable,
       coeffs_init: np.ndarray,
       t: np.ndarray,
       y_data: np.ndarray,
       weight: Optional[np.ndarray] = None,
       delta_coeffs: Optional[np.ndarray] = None,
       coeffs_lb: Optional[np.ndarray] = None,
       coeffs_ub: Optional[np.ndarray] = None,
       func_args: Tuple = (),
       opts: Optional[np.ndarray] = None) -> Tuple:
    """
    Simplified interface matching MATLAB lm.m function signature.
    
    Parameters
    ----------
    opts : ndarray, optional
        Algorithm options [prnt, MaxEvals, eps1, eps2, eps3, eps4, 
                          lam0, lamUP, lamDN, UpdateType]
    
    Returns
    -------
    coeffs : ndarray
        Optimized coefficients
    reduced_chi_sq : float
        Reduced chi-squared
    sigma_coeffs : ndarray
        Standard errors of coefficients
    sigma_fit : ndarray
        Standard errors of fit
    correlation : ndarray
        Correlation matrix
    r_squared : float
        R-squared
    cvg_history : ndarray
        Convergence history
    """
    # Parse options array (MATLAB-style interface)
    if opts is None:
        n_coeffs = len(coeffs_init)
        opts = np.array([0, 10*n_coeffs**2, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2, 11, 9, 1])
    
    prnt = int(opts[0])
    
    result = levenberg_marquardt(
        func, coeffs_init, t, y_data, weight, delta_coeffs, 
        coeffs_lb, coeffs_ub, func_args,
        print_level=prnt,
        max_iter=int(opts[1]),
        tol_gradient=opts[2],
        tol_coeffs=opts[3],
        tol_chi_sq=opts[4],
        tol_improvement=opts[5],
        lambda_init=opts[6],
        lambda_up_factor=opts[7],
        lambda_dn_factor=opts[8],
        update_type=int(opts[9]),
        plot_iterations=(prnt > 2)  # Plot at each iteration like MATLAB
    )
    
    return (result.coefficients, result.reduced_chi_sq, result.sigma_coefficients,
            result.sigma_fit, result.correlation, result.r_squared, 
            result.convergence_history, result.message, result.aic, result.bic )


# ============================================================================
# Module-level documentation
# ============================================================================
if __name__ == "__main__":
    print(__doc__)
    print("\nThis module provides the Levenberg-Marquardt algorithm for nonlinear")
    print("least squares curve-fitting. See examples in lm_examples.py")
