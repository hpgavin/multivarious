#! /usr/bin/env -S python3 -i
"""
L1_fit_example.py - Test script for L1 regularization

Demonstrates L1_fit on synthetic polynomial data with various true functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from multivarious.fit import L1_fit
from multivarious.utl import L1_plots, format_plot


def test_L1_basic():
    """
    Basic test of L1_fit with polynomial basis and nonlinear data.
    """
    
    print("\n" + "="*70)
    print("L1_fit Test - Basic Example")
    print("="*70)
    
    # Independent variables
    x = np.arange(-1.2, 1.21, 0.05)
    m = len(x)
    
    print(f"\nData points: {m}")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    
    # Power-polynomial fit basis functions (the design matrix)
    B = np.column_stack([x**i for i in range(8)])
    # Alternative sparse basis (commented out in original):
    # B = np.column_stack([x**1, x**2, x**3, x**5, x**7])
    
    print(f"Basis dimension: {B.shape}")
    print(f"Number of basis functions: {B.shape[1]}")
    
    # Generate noise
    np.random.seed(42)  # For reproducibility
    noise = 0.15 * np.random.randn(m)
    
    # Test different true functions (uncomment one)
    
    # Option 1: Quadratic + tanh
    # y = x**2 + np.tanh(5*x) + noise
    
    # Option 2: Quadratic + sine (default)
    y = 1 - x**2 + np.sin(np.pi * x) + noise
    print("\nTrue function: y = 1 - x² + sin(πx) + noise")
    
    # Option 3: Linear + Gaussian
    # y = 1 - x + np.exp(-(2*x)**2) + noise
    
    # L1 regularization parameters
    w = 1.0      # 0: without weighting, >0: with weighting
    alpha = 0.1  # L1 regularization parameter
    
    print(f"\nL1 parameters:")
    print(f"  Initial α: {alpha}")
    print(f"  Weighting w: {w}")
    
    # Fit model using L1 regularization
    print("\nFitting model with L1 regularization...")
    c, mu, nu, cvg_hst = L1_fit(B, y, alpha, w)
    
    # Extract final values
    n_iter = cvg_hst.shape[1]
    alpha_final = cvg_hst[-2, -1]
    error_final = cvg_hst[-1, -1]
    
    print(f"\nResults:")
    print(f"  Iterations: {n_iter}")
    print(f"  Final α: {alpha_final:.6f}")
    print(f"  Final error: {error_final:.6f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(c) > 1e-6)}/{len(c)}")
    
    # Create plots
    format_plot(font_size=15, line_width=3, marker_size=9)
    L1_plots(B, c, y, cvg_hst, alpha_final, w, fig_no=1, save_plots=False)
    
    return c, cvg_hst


def test_L1_comparison():
    """
    Compare L1 results with different regularization strengths.
    """
    
    print("\n" + "="*70)
    print("L1_fit Test - Parameter Comparison")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    x = np.arange(-1.2, 1.21, 0.05)
    m = len(x)
    B = np.column_stack([x**i for i in range(8)])
    noise = 0.15 * np.random.randn(m)
    y = 1 - x**2 + np.sin(np.pi * x) + noise
    
    # Test different alpha values
    alphas = [0.01, 0.05, 0.1, 0.5]
    w = 1.0
    
    plt.figure(figsize=(14, 10))
    
    for idx, alpha in enumerate(alphas):
        print(f"\nTesting α = {alpha}")
        
        # Fit model
        c, mu, nu, cvg_hst = L1_fit(B, y, alpha, w)
        alpha_final = cvg_hst[-2, -1]
        
        n_nonzero = np.sum(np.abs(c) > 1e-6)
        print(f"  Non-zero coefficients: {n_nonzero}/{len(c)}")
        print(f"  Final α: {alpha_final:.6f}")
        
        # Plot coefficients
        plt.subplot(2, 2, idx + 1)
        n = len(c)
        plt.plot(range(n), c, 'o-', color=[0, 0.8, 0], markersize=8, linewidth=2)
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Coefficient index')
        plt.ylabel('Coefficient value')
        plt.title(f'α = {alpha} → {n_nonzero} non-zero terms')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('L1_alpha_comparison.png', dpi=150)
    print("\nPlot saved: L1_alpha_comparison.png")


def test_L1_weighting():
    """
    Compare weighted vs unweighted L1.
    """
    
    print("\n" + "="*70)
    print("L1_fit Test - Weighting Comparison")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    x = np.arange(-1.2, 1.21, 0.05)
    m = len(x)
    B = np.column_stack([x**i for i in range(8)])
    noise = 0.15 * np.random.randn(m)
    y = 1 - x**2 + np.sin(np.pi * x) + noise
    
    alpha = 0.1
    
    # Test with and without weighting
    weights = [0.0, 0.5, 1.0, 2.0]
    
    plt.figure(figsize=(14, 10))
    
    for idx, w in enumerate(weights):
        print(f"\nTesting w = {w}")
        
        c, mu, nu, cvg_hst = L1_fit(B, y, alpha, w)
        
        n_nonzero = np.sum(np.abs(c) > 1e-6)
        print(f"  Non-zero coefficients: {n_nonzero}/{len(c)}")
        
        # Plot coefficients
        plt.subplot(2, 2, idx + 1)
        n = len(c)
        plt.plot(range(n), c, 'o-', color=[0.8, 0, 0.8], markersize=8, linewidth=2)
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Coefficient index')
        plt.ylabel('Coefficient value')
        title_str = f'w = {w} ({"unweighted" if w == 0 else "weighted"})'
        plt.title(f'{title_str} → {n_nonzero} terms')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('L1_weighting_comparison.png', dpi=150)
    print("\nPlot saved: L1_weighting_comparison.png")


def main():
    """
    Run all L1_fit tests.
    """
    
    print("\n" + "#"*70)
    print("# L1_fit Testing Suite")
    print("#"*70)
    
    # Test 1: Basic functionality
    c, cvg_hst = test_L1_basic()
    
    # Test 2: Compare different alphas
    test_L1_comparison()
    
    # Test 3: Compare weighting options
    test_L1_weighting()
    
    print("\n" + "#"*70)
    print("# All tests completed successfully!")
    print("#"*70)
    
    # Show all plots
    plt.show()

if __name__ == '__main__':
    main()
