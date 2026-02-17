#! /usr/bin/env -S python3 -i
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
from multivarious.utl.plot_lm import plot_lm


def lm_func(t: np.ndarray, coeffs: np.ndarray, example_number: float = 1.0) -> np.ndarray:
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

    elif example_number == 4:
        x1 = t[:, 0]
        x2 = t[:, 1]
        w = coeffs[0]
        q = coeffs[1]
    
        y_hat = (w * x1**q + (1 - w) * x2**q) ** (1 / q)
   
    else:
        raise ValueError(f"Unknown example_number: {example_number}. Use 1, 2, or 3")
    
    return y_hat


def run_example(example_number: int = 1, print_level: int = 3): #  -> Tuple:
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
    
    n_points = 100
    t = np.arange(1, n_points + 1, dtype=float)
        
    # True coefficient values
    if example_number == 1:
        coeffs_true = np.array([20.0, -24.0, 30.0, -40.0])
        coeffs_init = np.array([4.0, -5.0, 6.0, 10.0])
        msmnt_err = 0.5
    elif example_number == 2:
        coeffs_true = np.array([20.0, 10.0, 1.0, 50.0])
        coeffs_init = np.array([5.0, 2.0, 0.2, 10.0])
        msmnt_err = 0.5
    elif example_number == 3:
        coeffs_true = np.array([6.0, 20.0, 1.0, 5.0])
        coeffs_init = np.array([10.0, 50.0, 5.0, 5.7])
        msmnt_err = 0.5
    elif example_number == 4: 
        x1 = 2 * np.random.rand(n_points)
        x2 = 3 * np.random.rand(n_points)
        t = np.column_stack([x1, x2])  # Shape: (200, 2)
        
        coeffs_true = np.array([0.7, -2.1])
        coeffs_init = np.array([0.5, 1.0])
        msmnt_err = 0.10
 
    else:
        raise ValueError(f"example_number must be 1, 2, 3, or 4, got {example_number}")
        
    # Generate noisy data
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

    # ========================================================================
    # Print true and initial guess of coefficient values 
    # ========================================================================
    print("\n" + "="*80)
    print(f"Example {example_number}  Initial  and  True    Coefficient Values")
    print("-"*80)
    for i in range(len(coeffs_true)):
        print(f"a[{i}]  {coeffs_init[i]:12.4f} {coeffs_true[i]:12.4f} ")
    
    print("\n" + "-"*80)
     
    # Run optimization
    result = lm( lm_func,  
            coeffs_init, t, y_dat, weight, -0.01,
            coeffs_lb, coeffs_ub, (example_number,), opts  # Pass example_number here
    )

    coeffs_fit, chi_sq, sigma_coeffs, sigma_y, corr, R_sq, cvg_history, func_calls, message, aic, bic  = result

    # ========================================================================
    # Create plots
    # ========================================================================
    y_fit = lm_func(t, coeffs_fit, example_number)
    plot_lm(t, y_dat, y_fit, sigma_y, chi_sq, aic, bic, cvg_history, f"Example_{example_number}")
    
    return result


def sensitivity_to_initial_guess(example_number: int = 3, n_trials: int = 100) -> None:
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
        run_example(ex_num, print_level=3)
    
    # Sensitivity analysis
    input("\nPress Enter to run sensitivity analysis (Example 3)...")
    sensitivity_to_initial_guess(example_number=3, n_trials=100)
    
    input("\nPress Enter to exit...")
    plt.close('all')
