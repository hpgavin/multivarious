"""
Example Functions and Demonstrations for Levenberg-Marquardt

This module provides four example functions of increasing difficulty for
testing the Levenberg-Marquardt curve fitting algorithm, along with 
demonstration scripts that match the MATLAB examples.

Example Functions:
1. Polynomial (medium difficulty) - has local minima
2. Exponential decay (easy) - poor initial guess acceptable
3. Mixed exponential-sinusoidal (difficult) - needs good initial guess
4. Two-dimensional fitting (medium difficulty) - multiple independent variables
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting in Example 4
from typing import Tuple, List
from multivarious.fit.lm import levenberg_marquardt, lm


def lm_func(t: np.ndarray, coeffs: np.ndarray, example_number: int = 1) -> np.ndarray:
    """
    Example functions for nonlinear least squares curve fitting.
    
    Parameters
    ----------
    t : ndarray
        Independent variable (time), shape (m,)
    coeffs : ndarray
        Coefficient values [a0, a1, a2, a3], shape (4,)
    example_number : int
        Which example function to use (1, 2, or 3)
    
    Returns
    -------
    y_hat : ndarray
        Model prediction, shape (m,)
    
    Examples
    --------
    Example 1: Polynomial (medium difficulty)
        y = a₀(t/T) + a₁(t/T)² + a₂(t/T)³ + a₃(t/T)⁴
        Has local minima, moderately difficult for LM
    
    Example 2: Exponential decay (easy)
        y = a₀exp(-t/a₁) + a₂t·exp(-t/a₃)
        Easy for LM, poor initial guess is acceptable
    
    Example 3: Mixed exponential-sinusoidal (difficult)
        y = a₀exp(-t/a₁) + a₂sin(t/a₃)
        Difficult for LM, needs good initial guess for a₃
    """
    if example_number == 1:
        # Polynomial: medium difficulty
        T = np.max(t)
        tau = t / T
        y_hat = (coeffs[0] * tau + 
                coeffs[1] * tau**2 + 
                coeffs[2] * tau**3 + 
                coeffs[3] * tau**4)
    
    elif example_number == 2:
        # Exponential decay: easy
        y_hat = (coeffs[0] * np.exp(-t / coeffs[1]) + 
                coeffs[2] * t * np.exp(-t / coeffs[3]))
    
    elif example_number == 3:
        # Exponential + sinusoidal: difficult
        y_hat = (coeffs[0] * np.exp(-t / coeffs[1]) + 
                coeffs[2] * np.sin(t / coeffs[3]))
    
    else:
        raise ValueError(f"Unknown example_number: {example_number}. Use 1, 2, or 3")
    
    return y_hat


def lm_func2d(t: np.ndarray, coeffs: np.ndarray, const: float = 1.0) -> np.ndarray:
    """
    Two-dimensional example function for nonlinear least squares curve fitting.
    
    Demonstrates fitting with multiple independent variables (x and y).
    
    Parameters
    ----------
    t : ndarray
        Independent variables, shape (m, 2) where:
        - t[:, 0] = x values
        - t[:, 1] = y values
    coeffs : ndarray
        Coefficient values [w, q], shape (2,)
        - w: weight parameter (0 < w < 1)
        - q: power parameter  
    const : float, optional
        Optional constant (not used in this example)
    
    Returns
    -------
    z_hat : ndarray
        Model prediction, shape (m,)
        z = (w*x^q + (1-w)*y^q)^(1/q)
    
    Notes
    -----
    This is a generalized mean function:
    - q → 0: geometric mean
    - q = 1: arithmetic mean  
    - q = 2: quadratic mean (RMS)
    - q → ∞: maximum
    
    Example
    -------
    >>> t = np.column_stack([x, y])  # Stack x and y as columns
    >>> coeffs = [0.7, -2.1]
    >>> z = lm_func2d(t, coeffs)
    """
    x = t[:, 0]
    y = t[:, 1]
    w = coeffs[0]
    q = coeffs[1]
    
    z_hat = (w * x**q + (1 - w) * y**q) ** (1 / q)
    
    return z_hat


def lm_plots(t: np.ndarray,
             y_data: np.ndarray,
             y_fit: np.ndarray,
             sigma_y: np.ndarray,
             cvg_history: np.ndarray,
             title_prefix: str = "LM_fit") -> None:
    """
    Plot convergence history and fit results for Levenberg-Marquardt.
    
    Creates three figures:
    1. Convergence history (coefficients and chi-squared vs iterations)
    2. Data, fit, and confidence intervals
    3. Histogram of residuals
    
    Parameters
    ----------
    t : ndarray
        Independent variable
    y_data : ndarray
        Measured data
    y_fit : ndarray
        Fitted model
    sigma_y : ndarray
        Standard errors of fit
    cvg_history : ndarray
        Convergence history from LM algorithm
    title_prefix : str
        Prefix for plot titles and filenames
    """
    plt.ion()
    
    max_iter, n_cols = cvg_history.shape
    n_coeffs = n_cols - 3
    
    # ========================================================================
    # Figure 1: Convergence history
    # ========================================================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot coefficient evolution
    for i in range(n_coeffs):
        ax1.plot(cvg_history[:, 0], cvg_history[:, i+1], '-o', 
                linewidth=2, markersize=4, label=f'$a_{i+1}$')
        # Label final values
        ax1.text(cvg_history[-1, 0] * 1.02, cvg_history[-1, i+1], 
                f'{i+1}', fontsize=10)
    
    ax1.set_ylabel('Coefficient Values', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{title_prefix}: Convergence History', fontsize=14, fontweight='bold')
    
    # Plot chi-squared and lambda
    ax2.semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+1], 
                '-o', linewidth=2, markersize=4, label='$\\chi^2_\\nu$')
    ax2.semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+2], 
                '-s', linewidth=2, markersize=4, label='$\\lambda$')
    
    # Label start and end points
    ax2.text(cvg_history[0, 0], cvg_history[0, n_coeffs+1], 
            '$\\chi^2_\\nu$', fontsize=12, ha='right')
    ax2.text(cvg_history[0, 0], cvg_history[0, n_coeffs+2], 
            '$\\lambda$', fontsize=12, ha='right')
    ax2.text(cvg_history[-1, 0], cvg_history[-1, n_coeffs+1], 
            '$\\chi^2_\\nu$', fontsize=12)
    ax2.text(cvg_history[-1, 0], cvg_history[-1, n_coeffs+2], 
            '$\\lambda$', fontsize=12)
    
    ax2.set_xlabel('Number of Function Evaluations', fontsize=12)
    ax2.set_ylabel('$\\chi^2_\\nu$ and $\\lambda$', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ========================================================================
    # Figure 2: Data, fit, and confidence intervals
    # ========================================================================
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Confidence interval patches
    color_95 = [0.95, 0.95, 0.1]
    color_99 = [0.2, 0.95, 0.2]
    
    # 95% confidence interval
    y_upper_95 = y_fit + 1.96 * sigma_y
    y_lower_95 = y_fit - 1.96 * sigma_y
    
    # 99% confidence interval
    y_upper_99 = y_fit + 2.58 * sigma_y
    y_lower_99 = y_fit - 2.58 * sigma_y
    
    # Plot confidence intervals as filled regions
    ax.fill_between(t, y_lower_99, y_upper_99, 
                    color=color_99, alpha=0.6, label='99% C.I.')
    ax.fill_between(t, y_lower_95, y_upper_95, 
                    color=color_95, alpha=0.8, label='95% C.I.')
    
    # Plot data and fit
    ax.plot(t, y_data, 'ob', markersize=4, label='$y_{data}$')
    ax.plot(t, y_fit, '-k', linewidth=2, label='$y_{fit}$')
    
    ax.set_xlabel('$t$', fontsize=12)
    ax.set_ylabel('$y(t)$', fontsize=12)
    ax.set_title(f'{title_prefix}: Data and Fit with Confidence Intervals', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ========================================================================
    # Figure 3: Histogram of residuals
    # ========================================================================
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    residuals = y_data - y_fit
    ax.hist(residuals, bins=20, color='blue', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('$y_{data} - y_{fit}$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{title_prefix}: Histogram of Residuals', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics to plot
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.axvline(mean_res, color='r', linestyle='--', linewidth=2, 
              label=f'Mean = {mean_res:.3f}')
    ax.axvline(mean_res + std_res, color='orange', linestyle=':', linewidth=2)
    ax.axvline(mean_res - std_res, color='orange', linestyle=':', linewidth=2,
              label=f'Std = {std_res:.3f}')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def lm_plots2d(t: np.ndarray,
               z_data: np.ndarray,
               z_fit: np.ndarray,
               sigma_z: np.ndarray,
               cvg_history: np.ndarray,
               title_prefix: str = "LM_fit_2D") -> None:
    """
    Plot convergence history and 3D fit results for 2D Levenberg-Marquardt.
    
    Creates three figures for two-dimensional (x,y) → z fitting:
    1. Convergence history (parameters, χ², λ)
    2. 3D scatter plot of data, fit, and confidence bounds
    3. Histogram of residuals
    
    Parameters
    ----------
    t : ndarray, shape (m, 2)
        Independent variables [x, y]
    z_data : ndarray, shape (m,)
        Measured data points
    z_fit : ndarray, shape (m,)
        Fitted model values
    sigma_z : ndarray, shape (m,)
        Standard error of fit at each point
    cvg_history : ndarray
        Convergence history from LM algorithm
    title_prefix : str, optional
        Prefix for plot titles
    """
    x = t[:, 0]
    y = t[:, 1]
    n_coeffs = cvg_history.shape[1] - 3
    
    # ========================================================================
    # Figure 1: Convergence history
    # ========================================================================
    fig1 = plt.figure(figsize=(12, 10))
    
    # Plot 1: Parameter evolution
    ax1 = fig1.add_subplot(2, 1, 1)
    for i in range(n_coeffs):
        ax1.plot(cvg_history[:, 0], cvg_history[:, i+1], 
                linewidth=3, label=f'$a_{i}$')
    
    ax1.set_xlabel('Number of Function Evaluations', fontsize=12)
    ax1.set_ylabel('Parameter Values', fontsize=12)
    ax1.set_title(f'{title_prefix}: Parameter Convergence', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Chi-squared and lambda
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+1], 
                'b-', linewidth=3, label='$\\chi^2_\\nu$')
    ax2.semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+2], 
                'r-', linewidth=3, label='$\\lambda$')
    
    ax2.set_xlabel('Number of Function Evaluations', fontsize=12)
    ax2.set_ylabel('$\\chi^2_\\nu$ and $\\lambda$', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # ========================================================================
    # Figure 2: 3D scatter plot of data and fit
    # ========================================================================
    fig2 = plt.figure(figsize=(14, 10))
    ax = fig2.add_subplot(111, projection='3d')
    
    # Plot data
    ax.scatter(x, y, z_data, c='black', marker='o', s=30, 
              label='$z_{data}$', alpha=0.6)
    
    # Plot fit
    ax.scatter(x, y, z_fit, c='green', marker='*', s=50, 
              label='$z_{fit}$', alpha=0.8)
    
    # Plot 95% confidence interval bounds
    ax.scatter(x, y, z_fit + 1.96*sigma_z, c='red', marker='+', s=20,
              label='$z_{fit} \\pm 1.96\\sigma_z$', alpha=0.5)
    ax.scatter(x, y, z_fit - 1.96*sigma_z, c='red', marker='+', s=20, alpha=0.5)
    
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_zlabel('$z(x,y)$', fontsize=12)
    ax.set_title(f'{title_prefix}: Data and Fit with 95% Confidence Intervals',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ========================================================================
    # Figure 3: Histogram of residuals
    # ========================================================================
    fig3, ax = plt.subplots(figsize=(10, 6))
    
    residuals = z_data - z_fit
    ax.hist(residuals, bins=20, color='blue', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('$z_{data} - z_{fit}$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{title_prefix}: Histogram of Residuals',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def run_example(example_number: int = 1, 
               print_level: int = 2) -> Tuple:
    """
    Run a complete Levenberg-Marquardt curve fitting example.
    
    Generates synthetic data with noise, performs curve fitting, and
    creates visualization plots.
    
    Parameters
    ----------
    example_number : int
        Which example to run (1, 2, 3, or 4)
        1: Polynomial (1D)
        2: Exponential decay (1D)
        3: Exponential + sinusoidal (1D)
        4: Two-dimensional fitting (2D)
    print_level : int
        Verbosity level (0=silent, 1=final, 2=iteration, 3=detailed)
    
    Returns
    -------
    result : tuple
        (coeffs_fit, chi_sq, sigma_coeffs, sigma_y, corr, R_sq, cvg_history, func_calls, message, aic, bic )
    """
    np.random.seed(42)  # Reproducible results
    
    # ========================================================================
    # Generate synthetic data
    # ========================================================================
    
    # Example 4 is different - it has 2D independent variables
    if example_number == 4:
        # Two-dimensional example
        n_points = 200
        x = 2 * np.random.rand(n_points)
        y = 3 * np.random.rand(n_points)
        t = np.column_stack([x, y])  # Shape: (200, 2)
        
        coeffs_true = np.array([0.7, -2.1])
        coeffs_init = np.array([0.5, 1.0])
        
        # Generate noisy data
        msmnt_err = 0.01
        z_dat = lm_func2d(t, coeffs_true)
        z_dat = z_dat + msmnt_err * np.random.randn(n_points)
        
        # Weights
        weight = n_points / np.sqrt(z_dat.T @ z_dat)
        
        # Bounds
        coeffs_lb = np.array([0.01, -5.0])
        coeffs_ub = np.array([0.99, 5.0])
        
    else:
        # Examples 1-3 are 1D
        n_points = 100
        t = np.arange(1, n_points + 1, dtype=float)
        
        # True coefficient values
        if example_number == 1:
            coeffs_true = np.array([20.0, -24.0, 30.0, -40.0])
            coeffs_init = np.array([4.0, -5.0, 6.0, 10.0])
        elif example_number == 2:
            coeffs_true = np.array([20.0, 10.0, 1.0, 50.0])
            coeffs_init = np.array([5.0, 2.0, 0.2, 10.0])
        elif example_number == 3:
            coeffs_true = np.array([6.0, 20.0, 1.0, 5.0])
            coeffs_init = np.array([10.0, 50.0, 5.0, 5.7])
        else:
            raise ValueError(f"example_number must be 1, 2, 3, or 4, got {example_number}")
        
        # Generate noisy data
        msmnt_err = 0.5
        y_dat = lm_func(t, coeffs_true, example_number)
        y_dat = y_dat + msmnt_err * np.random.randn(n_points)
        
        # Weights (inverse of measurement variance)
        weight = 1.0 / msmnt_err**2
        
        # Bounds
        coeffs_lb = -10 * np.abs(coeffs_init)
        coeffs_ub = 10 * np.abs(coeffs_init)
    
    # ========================================================================
    # Perform curve fitting
    # ========================================================================
    print("\n" + "="*80)
    print(f"LEVENBERG-MARQUARDT EXAMPLE {example_number}")
    print("="*80)
    
    # Algorithm options
    # [prnt, MaxEvals, eps1, eps2, eps3, eps4, lam0, lamUP, lamDN, UpdateType]
    opts = np.array([print_level, 100, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2, 11, 9, 1])
    
    # Run optimization
    if example_number == 4:
        # 2D fitting
        result = lm(
            lm_func2d,
            coeffs_init, t, z_dat, weight, 0.01,
            coeffs_lb, coeffs_ub, (), opts
        )
    else:
        # 1D fitting
        result = lm(
            lambda t_in, c: lm_func(t_in, c, example_number),
            coeffs_init, t, y_dat, weight, -0.01,
            coeffs_lb, coeffs_ub, (), opts
        )
    
    coeffs_fit, chi_sq, sigma_coeffs, sigma_y, corr, R_sq, cvg_history, func_calls, message, aic, bic  = result
    
    # ========================================================================
    # Print results
    # ========================================================================
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'':>8} {'Initial':>12} {'True':>12} {'Fitted':>12} {'Std Err':>12} {'% Error':>10}")
    print("-"*80)
    for i in range(len(coeffs_true)):
        pct_err = 100 * abs(sigma_coeffs[i] / coeffs_fit[i]) if coeffs_fit[i] != 0 else np.inf
        print(f"a[{i}]  {coeffs_init[i]:12.4f} {coeffs_true[i]:12.4f} "
              f"{coeffs_fit[i]:12.4f} {sigma_coeffs[i]:12.4f} {pct_err:10.2f}")
    
    print("\n" + "-"*80)
    print(f"Reduced χ²: {chi_sq:.6f}")
    print(f"R²:         {R_sq:.6f}")
    print("\nCorrelation matrix:")
    print(corr)
    print("="*80 + "\n")
    
    # ========================================================================
    # Create plots
    # ========================================================================
    if example_number == 4:
        z_fit = lm_func2d(t, coeffs_fit)
        lm_plots2d(t, z_dat, z_fit, sigma_y, cvg_history,
                  f"Example_{example_number}")
    else:
        y_fit = lm_func(t, coeffs_fit, example_number)
        lm_plots(t, y_dat, y_fit, sigma_y, cvg_history,
                f"Example_{example_number}")
    
    return result


def sensitivity_to_initial_guess(example_number: int = 3,
                                 n_trials: int = 100) -> None:
    """
    Demonstrate sensitivity to initial guess by running many random starts.
    
    This recreates the MATLAB lm_examp_init.m demonstration showing how
    different initial guesses lead to different local minima (or convergence
    to the global minimum).
    
    Parameters
    ----------
    example_number : int
        Which example to run (1, 2, or 3)
    n_trials : int
        Number of random initial guesses to try
    """
    np.random.seed(42)
    
    # ========================================================================
    # Setup
    # ========================================================================
    n_points = 100
    t = np.arange(1, n_points + 1, dtype=float)
    
    # True coefficients
    if example_number == 1:
        coeffs_true = np.array([20.0, -24.0, 30.0, -40.0])
    elif example_number == 2:
        coeffs_true = np.array([20.0, 10.0, 1.0, 50.0])
    elif example_number == 3:
        coeffs_true = np.array([6.0, 20.0, 1.0, 5.0])
    
    # Generate data
    msmnt_err = 0.5
    y_dat = lm_func(t, coeffs_true, example_number)
    y_dat = y_dat + msmnt_err * np.random.randn(n_points)
    weight = 1.0 / msmnt_err**2
    
    # ========================================================================
    # Run multiple trials with random initial guesses
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS: {n_trials} random initial guesses")
    print(f"Example {example_number}")
    print(f"{'='*80}\n")
    
    coeffs_init_all = np.zeros((4, n_trials))
    coeffs_fit_all = np.zeros((4, n_trials))
    chi_sq_all = np.zeros(n_trials)
    
    # Generate random initial guesses
    for i in range(n_trials):
        coeffs_init_all[:, i] = (0.1 * coeffs_true + 
                                1.9 * coeffs_true * np.random.rand(4))
    
    # Bounds
    coeffs_lb = -2 * np.abs(coeffs_true)
    coeffs_ub = 2 * np.abs(coeffs_true)
    
    # Options (silent mode)
    opts = np.array([0, 800, 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 11, 9, 1])
    
    # Run all trials
    for i in range(n_trials):
        if (i+1) % 10 == 0:
            print(f"  Trial {i+1}/{n_trials}...", end='\r')
        
        result = lm(
            lambda t_in, c: lm_func(t_in, c, example_number),
            coeffs_init_all[:, i], t, y_dat, weight, -1e-5,
            coeffs_lb, coeffs_ub, (), opts
        )
        
        coeffs_fit_all[:, i] = result[0]
        chi_sq_all[i] = result[1]
    
    print(f"  Completed {n_trials} trials" + " "*20)
    
    # Sort by chi-squared
    idx = np.argsort(chi_sq_all)
    coeffs_init_all = coeffs_init_all[:, idx]
    coeffs_fit_all = coeffs_fit_all[:, idx]
    chi_sq_all = chi_sq_all[idx]
    
    print(f"\nBest  χ² = {chi_sq_all[0]:.6f}")
    print(f"Worst χ² = {chi_sq_all[-1]:.6f}")
    print(f"Median χ² = {np.median(chi_sq_all):.6f}")
    
    # ========================================================================
    # Plot results
    # ========================================================================
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f'Sensitivity to Initial Guess - Example {example_number}', 
                fontsize=16, fontweight='bold')
    
    for ii in range(4):
        for jj in range(4):
            ax = axes[ii, jj]
            
            # Plot all trials
            ax.plot(coeffs_init_all[jj, :], coeffs_fit_all[ii, :], 
                   'ok', markersize=3, alpha=0.5)
            
            # Highlight best 15 fits
            ax.plot(coeffs_init_all[jj, :15], coeffs_fit_all[ii, :15], 
                   'og', markersize=8, markeredgewidth=2, markerfacecolor='none')
            
            # Highlight worst fit
            ax.plot(coeffs_init_all[jj, -1], coeffs_fit_all[ii, -1], 
                   'or', markersize=12, markeredgewidth=3, markerfacecolor='none')
            
            # Highlight best fit
            ax.plot(coeffs_init_all[jj, 0], coeffs_fit_all[ii, 0], 
                   'om', markersize=12, markeredgewidth=3, markerfacecolor='none')
            
            # True value
            ax.plot(coeffs_true[jj], coeffs_true[ii], 
                   'ob', markersize=12, markeredgewidth=3, markerfacecolor='none')
            
            # Set axis limits
            x_range = [min(coeffs_true[jj] * 0.1, coeffs_true[jj] * 2),
                      max(coeffs_true[jj] * 0.1, coeffs_true[jj] * 2)]
            y_range = [min(coeffs_true[ii] * 0.1, coeffs_true[ii] * 2),
                      max(coeffs_true[ii] * 0.1, coeffs_true[ii] * 2)]
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Labels
            if ii == 3:
                ax.set_xlabel(f'$a_{{INIT}}[{jj}]$', fontsize=10)
            if jj == 0:
                ax.set_ylabel(f'$a_{{FIT}}[{ii}]$', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nLegend:")
    print("  Black circles: All trials")
    print("  Green circles: Best 15 fits")
    print("  Red circle:    Worst fit")
    print("  Magenta circle: Best fit")
    print("  Blue circle:    True value")
    print("="*80 + "\n")


# ============================================================================
# Main demonstration
# ============================================================================
if __name__ == "__main__":
    """
    Run all three examples and sensitivity analysis.
    """
    plt.ion()
    
    print("\n" + "="*80)
    print("LEVENBERG-MARQUARDT CURVE FITTING EXAMPLES")
    print("="*80)
    print("\nFour examples of increasing difficulty:")
    print("  1. Polynomial - medium difficulty, has local minima")
    print("  2. Exponential decay - easy, poor initial guess acceptable")
    print("  3. Exponential + sinusoidal - difficult, needs good initial guess")
    print("  4. Two-dimensional fitting - multiple independent variables (x,y)")
    print("\n" + "="*80 + "\n")
    
    # Run each example
    for ex_num in [1, 2, 3, 4]:
        input(f"Press Enter to run Example {ex_num}...")
        run_example(ex_num, print_level=2)
    
    # Sensitivity analysis
    input("\nPress Enter to run sensitivity analysis (Example 3)...")
    sensitivity_to_initial_guess(example_number=3, n_trials=100)
    
    input("\nPress Enter to exit...")
    plt.close('all')
