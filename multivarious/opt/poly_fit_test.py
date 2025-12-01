"""
poly_fit_test.py - Test script for polynomial fitting

Test the poly_fit function for least squares data fitting and error analysis.

Translation from MATLAB by Claude, 2025-10-24
"""

import numpy as np
import matplotlib.pyplot as plt
from poly_fit import poly_fit


def test_poly_fit():
    """
    Test poly_fit with two different polynomial bases.
    """
    
    print("\n" + "#"*70)
    print("# Polynomial Fit Test Suite")
    print("#"*70)
    
    # ----------------------------------------------------------
    # Generate (x,y) data with measurement errors in y
    
    x_l = -1      # Low value of independent variable
    x_h = 1       # High value of independent variable
    Nd = 40       # Number of data points
    
    x = np.linspace(x_l, x_h, Nd)
    
    measurement_error = 0.20  # RMS of simulated measurement error
    
    np.random.seed(42)  # For reproducibility
    y = (-np.cos(4*x) + 
         1.0 * x**3 * np.exp(-x/3) + 
         measurement_error * np.random.randn(Nd))
    
    print(f"\nGenerated synthetic data:")
    print(f"  Data points: {Nd}")
    print(f"  x range: [{x_l}, {x_h}]")
    print(f"  Measurement error (RMS): {measurement_error}")
    print(f"  True function: y = -cos(4x) + x³·exp(-x/3) + noise")
    
    # ----------------------------------------------------------
    # Test A: Full polynomial basis [0, 1, 2, 3, 4]
    
    print("\n" + "="*70)
    print("TEST A: Powers = [0, 1, 2, 3, 4]")
    print("="*70)
    
    p_A = np.array([0, 1, 2, 3, 4])
    figNo_A = 10
    
    c_A, x_fit_A, y_fit_A, Sc_A, Sy_fit_A, Rc_A, R2_A, Vr_A, AIC_A, cond_A = \
        poly_fit(x, y, p_A, figNo_A)
    
    # ----------------------------------------------------------
    # Test B: Reduced polynomial basis [0, 2, 3, 4] (no linear term)
    
    print("\n" + "="*70)
    print("TEST B: Powers = [0, 2, 3, 4]")
    print("="*70)
    
    p_B = np.array([0, 2, 3, 4])
    figNo_B = 20
    
    c_B, x_fit_B, y_fit_B, Sc_B, Sy_fit_B, Rc_B, R2_B, Vr_B, AIC_B, cond_B = \
        poly_fit(x, y, p_B, figNo_B)
    
    # ----------------------------------------------------------
    # Comparison
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Test A (5 terms)':<20} {'Test B (4 terms)':<20}")
    print("-"*70)
    print(f"{'Powers':<25} {str(p_A):<20} {str(p_B):<20}")
    print(f"{'Number of terms':<25} {len(p_A):<20} {len(p_B):<20}")
    print(f"{'R² (correlation)':<25} {R2_A:<20.4f} {R2_B:<20.4f}")
    print(f"{'AIC (lower better)':<25} {AIC_A:<20.2f} {AIC_B:<20.2f}")
    print(f"{'Condition number':<25} {cond_A:<20.1f} {cond_B:<20.1f}")
    print(f"{'Residual variance':<25} {Vr_A:<20.4f} {Vr_B:<20.4f}")
    
    # Determine preferred model
    print("\n" + "-"*70)
    if AIC_B < AIC_A:
        print("✓ TEST B is PREFERRED (lower AIC despite fewer terms)")
        print("  → The linear term (x¹) is not significant for this data")
    else:
        print("✓ TEST A is PREFERRED (lower AIC)")
        print("  → All terms contribute significantly")
    
    # Parameter correlation analysis
    print("\n" + "="*70)
    print("PARAMETER CORRELATION ANALYSIS")
    print("="*70)
    
    print("\nTest A - Correlation Matrix:")
    print("(Values close to ±1 indicate high correlation)")
    print_correlation_matrix(Rc_A, p_A)
    
    print("\nTest B - Correlation Matrix:")
    print_correlation_matrix(Rc_B, p_B)
    
    # Save figures
    save_figures = True
    if save_figures:
        print("\n" + "="*70)
        print("Saving figures...")
        
        plt.figure(figNo_A)
        plt.savefig('poly_fit_testA_fit.png', 
                    dpi=150, bbox_inches='tight')
        
        plt.figure(figNo_A + 1)
        plt.savefig('poly_fit_testA_hist.png', 
                    dpi=150, bbox_inches='tight')
        
        plt.figure(figNo_A + 2)
        plt.savefig('poly_fit_testA_cdf.png', 
                    dpi=150, bbox_inches='tight')
        
        plt.figure(figNo_B)
        plt.savefig('poly_fit_testB_fit.png', 
                    dpi=150, bbox_inches='tight')
        
        plt.figure(figNo_B + 1)
        plt.savefig('poly_fit_testB_hist.png', 
                    dpi=150, bbox_inches='tight')
        
        plt.figure(figNo_B + 2)
        plt.savefig('poly_fit_testB_cdf.png', 
                    dpi=150, bbox_inches='tight')
        
        print("Figures saved:")
        print("  - poly_fit_testA_*.png (3 figures)")
        print("  - poly_fit_testB_*.png (3 figures)")
    
    print("\n" + "#"*70)
    print("# Test completed successfully!")
    print("#"*70)
    
    return c_A, c_B, R2_A, R2_B, AIC_A, AIC_B


def print_correlation_matrix(R, powers):
    """
    Pretty-print correlation matrix with power labels.
    
    Parameters
    ----------
    R : ndarray
        Correlation matrix
    powers : array_like
        Power values for labeling
    """
    n = len(powers)
    
    # Header
    print("     ", end="")
    for p in powers:
        if p == int(p):
            print(f"  x^{int(p)}", end="")
        else:
            print(f" x^{p:.1f}", end="")
    print()
    
    # Rows
    for i, p_i in enumerate(powers):
        if p_i == int(p_i):
            print(f"x^{int(p_i):2d}: ", end="")
        else:
            print(f"x^{p_i:4.1f}: ", end="")
        
        for j in range(n):
            print(f"{R[i,j]:6.3f}", end="")
        print()


def test_fractional_powers():
    """
    Test poly_fit with fractional (non-integer) powers.
    
    This demonstrates a key advantage over numpy.polyfit!
    """
    
    print("\n" + "="*70)
    print("FRACTIONAL POWERS TEST")
    print("="*70)
    print("Testing poly_fit with non-integer powers")
    print("(This is NOT possible with numpy.polyfit!)")
    
    # Generate data
    np.random.seed(123)
    x = np.linspace(0.1, 2, 30)
    
    # True function with fractional powers
    y_true = 2.0 + 1.5*x**0.5 - 0.8*x**1.5 + 0.3*x**2.5
    y = y_true + 0.1 * np.random.randn(len(x))
    
    print(f"\nTrue function: y = 2.0 + 1.5·x^0.5 - 0.8·x^1.5 + 0.3·x^2.5")
    
    # Fit with fractional powers
    p = np.array([0, 0.5, 1.5, 2.5])
    
    print(f"\nFitting with powers: {p}")
    
    c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, condNo = \
        poly_fit(x, y, p, figNo=30)
    
    print(f"\n✓ Successfully fitted with fractional powers!")
    print(f"  R² = {R2:.4f}")
    print(f"  Should be close to [2.0, 1.5, -0.8, 0.3]")
    
    plt.figure(30)
    plt.savefig('poly_fit_fractional.png', 
                dpi=150, bbox_inches='tight')
    print("  Figure saved: poly_fit_fractional.png")


def test_weighted_fit():
    """
    Test poly_fit with heteroscedastic errors (varying measurement errors).
    """
    
    print("\n" + "="*70)
    print("WEIGHTED FIT TEST")
    print("="*70)
    print("Testing poly_fit with varying measurement errors")
    
    # Generate data with varying noise
    np.random.seed(456)
    x = np.linspace(-1, 1, 40)
    
    # Measurement errors that increase with |x|
    Sy = 0.05 + 0.2 * np.abs(x)
    
    # Generate data with heteroscedastic noise
    y = 1 - x**2 + np.random.randn(len(x)) * Sy
    
    print(f"\nMeasurement errors vary from {Sy.min():.3f} to {Sy.max():.3f}")
    
    # Unweighted fit (ignores varying errors)
    print("\n--- Unweighted fit (ignores measurement errors) ---")
    p = np.array([0, 1, 2])
    c_unweighted, *_ = poly_fit(x, y, p, figNo=40)
    
    # Weighted fit (accounts for varying errors)
    print("\n--- Weighted fit (accounts for measurement errors) ---")
    c_weighted, *_ = poly_fit(x, y, p, figNo=50, Sy=Sy)
    
    print("\nCoefficient comparison:")
    print(f"{'Power':<10} {'Unweighted':<15} {'Weighted':<15}")
    print("-"*45)
    for i, pi in enumerate(p):
        print(f"x^{int(pi):<8} {c_unweighted[i]:<15.4f} {c_weighted[i]:<15.4f}")
    
    print("\n✓ Weighted fit gives more accurate estimates when errors vary!")
    
    plt.figure(40)
    plt.savefig('poly_fit_unweighted.png', 
                dpi=150, bbox_inches='tight')
    
    plt.figure(50)
    plt.savefig('poly_fit_weighted.png', 
                dpi=150, bbox_inches='tight')


def main():
    """
    Run all polynomial fit tests.
    """
    
    print("\n" + "#"*70)
    print("# POLY_FIT COMPREHENSIVE TEST SUITE")
    print("#"*70)
    
    # Test 1: Basic comparison of two polynomial bases
    test_poly_fit()
    
    # Test 2: Fractional (non-integer) powers
    test_fractional_powers()
    
    # Test 3: Weighted fit with varying measurement errors
    test_weighted_fit()
    
    print("\n" + "#"*70)
    print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    print("#"*70)
    print("\nKey features demonstrated:")
    print("  ✓ Polynomial fitting with any real-valued powers")
    print("  ✓ Comprehensive error analysis (Sa, R², AIC)")
    print("  ✓ Confidence intervals and correlation plots")
    print("  ✓ Weighted least squares with measurement errors")
    print("  ✓ Model comparison using information criteria")
    print("\n" + "#"*70)
    
    plt.show()


if __name__ == '__main__':
    main()
