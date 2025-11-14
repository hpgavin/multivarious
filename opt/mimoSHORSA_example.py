#!/usr/bin/env python3
''' ... from console 
import importlib , example_usage
importlib.reload(example_usage)     # to reload edits
'''
"""
Example usage of mimoSHORSA for polynomial response surface fitting

This script demonstrates:
1. Creating synthetic nonlinear data
2. Fitting a high-order polynomial model
3. Evaluating model performance
4. Visualizing results
"""
'''
    scaling     scale the data before fitting                                 1
                scaling = 0 : no scaling
                scaling = 1 : subtract mean and divide by std.dev
                scaling = 2 : subtract mean and decorrelate
                scaling = 3 : log-transform, subtract mean and divide by std.dev
                scaling = 4 : log-transform, subtract mean and decorrelate
'''

import numpy as np
import matplotlib.pyplot as plt
from mimoSHORSA import mimoSHORSA

def example_1_simple_polynomial():
    """
    Example 1: Fit a simple 2D polynomial function
    """
    print("\n" + "="*70)
    print("Example 1: Simple 2D Polynomial")
    print("="*70)
    
    np.random.seed(42)
    nInp = 2     # 2  input variables
    nOut = 1     # 1 output variable
    mData = 500  # number of data points
    
    # Generate input data
    dataX = 2 * np.random.randn(nInp, mData)
    
    # Generate output: y = 1 + 2*x1 + 0.5*x2^2 + 0.3*x1*x2 + noise
    dataY = np.zeros((nOut, mData))
    dataY[0, :] = (1.0 + 
                   2.0 * dataX[0, :] + 
                   0.5 * dataX[1, :] ** 2 + 
                   0.3 * dataX[0, :] * dataX[1, :] + 
                   0.4 * np.random.randn(mData))
    
    # Fit model
    print("\nFitting model...")
    order, coeff, meanX, meanY, trfrmX, trfrmY, testX, testY, testModelY = \
        mimoSHORSA(dataX, dataY, 
                   maxOrder=2,     # Maximum polynomial order
                   pTrain=70,      # 70% training, 30% testing
                   pCull=0,        # Will be set to 0 if L1_pnlty > 0
                   cov_tol=0.10,   # Maximum coefficient of variation
                   scaling=1,      # Decorrelation scaling
                   L1_pnlty=50,    # L1 regularization penalty
                   basis_fctn='L') # 'H'=Hermite, 'L'=Legendre, 'P'=Power
   
    return order, coeff, testModelY, testX, testY


def example_2_multi_output():
    """
    Example 2: Multi-output system with coupling
    """
    print("\n" + "="*70)
    print("Example 2: Multi-Output System")
    print("="*70)
    
    np.random.seed(123)
    nInp = 3   # 3 input variables
    nOut = 2   # 2 output variables
    mData = 200  # 200 data points
    
    # Generate input data
    dataX = np.random.randn(nInp, mData)
    
    # Generate coupled outputs
    dataY = np.zeros((nOut, mData))
    # Output 1: y1 = 1 + x1 + 0.5*x2^2 + 0.2*x1*x3
    dataY[0, :] = (1.0 + 
                   1.0 * dataX[0, :] + 
                   0.5 * dataX[1, :] ** 2 + 
                   0.2 * dataX[0, :] * dataX[2, :] + 
                   0.05 * np.random.randn(mData))
    
    # Output 2: y2 = 0.5 + 1.5*x2 - 0.8*x3^2 + 0.3*x1*x2
    dataY[1, :] = (0.5 + 
                   1.5 * dataX[1, :] - 
                   0.8 * dataX[2, :] ** 2 + 
                   0.3 * dataX[0, :] * dataX[1, :] + 
                   0.15 * np.random.randn(mData))
    
    # Fit model
    print("\nFitting multi-output model...")
    order, coeff, meanX, meanY, trfrmX, trfrmY, testX, testY, testModelY = \
        mimoSHORSA(dataX, dataY,
                   maxOrder=3,
                   pTrain=75, 
                   pCull=0,      # Ignored if L1_pnlty > 0
                   cov_tol=0.0, 
                   scaling=2,
                   L1_pnlty=10,   # L1 regularization
                   basis_fctn='L') # Hermite basis
    
    return order, coeff, testModelY, testX, testY


def example_3_multi_input():
    """
    Example 3: Higher dimensional problem
    """
    print("\n" + "="*70)
    print("Example 3: High-Dimensional Problem")
    print("="*70)
    
    np.random.seed(456)
    nInp = 5   # 5 input variables
    nOut = 1   # 1 output variable
    mData = 300  # 300 data points
    
    # Generate input data
    dataX = np.random.randn(nInp, mData)
    
    # Generate output: complex nonlinear function
    dataY = np.zeros((nOut, mData))
    dataY[0, :] = (2.0 + 
                   1.5 * dataX[0, :] + 
                   0.8 * dataX[1, :] ** 2 - 
                   0.6 * dataX[2, :] ** 2 + 
                   0.4 * dataX[3, :] +
                   0.3 * dataX[0, :] * dataX[1, :] +
                   0.2 * dataX[2, :] * dataX[4, :] +
                   0.25 * np.random.randn(mData))
    
    # Fit model with lower maximum order due to curse of dimensionality
    print("\nFitting high-dimensional model...")
    order, coeff, meanX, meanY, trfrmX, trfrmY, testX, testY, testModelY = \
        mimoSHORSA(dataX, dataY,
                   maxOrder=2,
                   pTrain=70,
                   pCull=0,
                   cov_tol=0.1,
                   scaling=2,
                   L1_pnlty= 89,    # Light L1 regularization
                   basis_fctn='L') # Power polynomial basis

   
    return order, coeff, testModelY, testX, testY


def example_4_with_scaling():
    """
    Example 4: Demonstrate different scaling options
    """
    print("\n" + "="*70)
    print("Example 4: Scaling Options Comparison")
    print("="*70)
    
    np.random.seed(789)
    nInp = 3
    nOut = 1
    mData = 100
    
    # Generate data with different scales
    dataX = np.zeros((nInp, mData))
    dataX[0, :] = 100  * np.random.randn(mData)   # Large scale
    dataX[1, :] = 1.00 * np.random.randn(mData)   # Small scale
    dataX[2, :] = dataX[0,:] + dataX[1,:]         # Correlated inputs
    
    # Generate output
    dataY = np.zeros((nOut, mData))
    dataY[0, :] =    0.01 * dataX[0, :] + \
                   100.0  * dataX[1, :]**2 + \
                     0.5  * dataX[1,:]*dataX[2,:] + \
                     0.1  * np.random.randn(mData)
    
    scaling_names = { 0: "no scaling", 1: "standardization (mean=0, std=1)", 2: "decorrelation" }
    
    scaling_option = 2   #  use 0 , 1, 2 
    print(f"\n--- Scaling Option {scaling_option}: {scaling_names[scaling_option]} ---")
        
    order, coeff, *_ = mimoSHORSA(
        dataX, dataY,
        maxOrder=2,
        pTrain=70,
        pCull=0,
        cov_tol=0.1,
        scaling=scaling_option,
        L1_pnlty=10,      # Use L1 for all scaling tests
        basis_fctn='L'    # Legendre basis
    )
#   except Exception as e:
#       print(f"Error: {e}")

    return


def example_5_basis_comparison():
    """
    Example 5: Compare different basis functions.

    Demonstrates:
    - Hermite basis (H)
    - Legendre basis (L)
    - Power polynomial basis (P)
    """

    print("\n" + "="*70)
    print("Example 5: Basis Function Comparison")
    print("="*70)

    np.random.seed(999)

    nInp = 2
    nOut = 1
    mData = 200

    # Generate input data
    dataX = 2 * np.random.randn(nInp, mData)

    # Generate output: quadratic function
    dataY = np.zeros((nOut, mData))
    dataY[0, :] = (1.0 +
                   2.0 * dataX[0, :] +
                   0.5 * dataX[1, :] ** 2 +
                   0.3 * dataX[0, :] * dataX[1, :] +
                   0.2 * np.random.randn(mData))

    basis_types = ['H', 'L', 'P']
    basis_names = {
        'H': 'Hermite functions',
        'L': 'Legendre polynomials',
        'P': 'Power polynomials'
    }

    results = {}

    for basis in basis_types:
        print(f"\n--- Testing {basis_names[basis]} ---")

        order, coeff, meanX, meanY, trfrmX, trfrmY, testX, testY, testModelY = \
            mimoSHORSA(dataX, dataY,
                       maxOrder=3,
                       pTrain=70,
                       pCull=0,        # Not used with L1
                       cov_tol=0.2,
                       scaling=2,      # Decorrelation
                       L1_pnlty=20,    # L1 regularization
                       basis_fctn=basis)

        # Compute test correlation
        corr = np.corrcoef(testY[0, :], testModelY[0, :])[0, 1]
        n_terms = np.sum(np.abs(coeff[0]) > 1e-6)  # Count non-zero terms

        results[basis] = {
            'correlation': corr,
            'n_terms': n_terms,
            'order': order,
            'coeff': coeff
        }

        print(f"  Number of terms: {n_terms}")
        print(f"  Test correlation: {corr:.4f}")

    # Summary comparison
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"{'Basis':<20} {'Terms':<10} {'Correlation':<15}")
    print("-"*70)
    for basis in basis_types:
        print(f"{basis_names[basis]:<20} "
              f"{results[basis]['n_terms']:<10} "
              f"{results[basis]['correlation']:<15.4f}")

    return results


def example_6_L1_vs_COV():
    """
    Example 6: Compare L1 regularization vs COV culling.

    Demonstrates the difference between:
    - Traditional COV-based iterative culling (L1_pnlty = 0)
    - L1 regularization (L1_pnlty > 0)
    """

    print("\n" + "="*70)
    print("Example 6: L1 Regularization vs COV Culling")
    print("="*70)

    np.random.seed(123)

    nInp = 3
    nOut = 1
    mData = 250

    # Generate data
    dataX = np.random.randn(nInp, mData)

    # True model: sparse (only 4 terms)
    dataY = np.zeros((nOut, mData))
    dataY[0, :] = (1.0 +
                   1.5 * dataX[0, :] +
                   0.8 * dataX[1, :] ** 2 +
                   0.4 * dataX[0, :] * dataX[2, :] +
                   0.15 * np.random.randn(mData))

    # Test 1: COV Culling (traditional method)
    print("\n--- Method 1: COV-based Culling ---")
    order1, coeff1, *_, testModelY1, testX1, testY1 = \
        mimoSHORSA(dataX, dataY,
                   maxOrder=3,
                   pTrain=70,
                   pCull=0,       # Enable culling
                   cov_tol=0.2,
                   scaling=2,
                   L1_pnlty=0,     # Disable L1 (use COV)
                   basis_fctn='L')

    corr1 = np.corrcoef(testY1[0, :], testModelY1[0, :])[0, 1]
    n_terms1 = len(coeff1[0])

    print(f"  Terms after culling: {n_terms1}")
    print(f"  Test correlation: {corr1:.4f}")

    # Test 2: L1 Regularization
    print("\n--- Method 2: L1 Regularization ---")
    order2, coeff2, *_, testModelY2, testX2, testY2 = \
        mimoSHORSA(dataX, dataY,
                   maxOrder=3,
                   pTrain=70,
                   pCull=0,        # No culling (automatic with L1)
                   cov_tol=0.2,
                   scaling=2,
                   L1_pnlty=10,    # Enable L1 regularization
                   basis_fctn='L')

    corr2 = np.corrcoef(testY2[0, :], testModelY2[0, :])[0, 1]
    n_terms2 = np.sum(np.abs(coeff2[0]) > 1e-6)  # Count non-zero

    print(f"  Non-zero terms: {n_terms2}")
    print(f"  Test correlation: {corr2:.4f}")

    # Comparison
    print("\n" + "="*70)
    print("Method Comparison")
    print("="*70)
    print(f"{'Method':<30} {'Terms':<10} {'Test �~A':<10}")
    print("-"*70)
    print(f"{'COV Culling':<30} {n_terms1:<10} {corr1:<10.4f}")
    print(f"{'L1 Regularization':<30} {n_terms2:<10} {corr2:<10.4f}")
    print(f"{'True model':<30} {4:<10} {'N/A':<10}")

    print("\nKey observations:")
    print("  - L1 tends to find sparser models")
    print("  - L1 is faster (no iterative culling)")
    print("  - COV culling decisions are irreversible")
    print("  - L1 provides global optimization")


def main():
    """
    Run all examples
    """
    print("\n" + "#"*70)
    print("# mimoSHORSA Example Usage")
    print("#"*70)
    
    # Run examples
    print("\n\nRunning examples (this may take a few minutes)...\n")

    '''
    # Example 1 - simple polynomial
    order1, coeff1, testModelY1, testX1, testY1 = example_1_simple_polynomial()
    
    # Example 2: multi-output
    order2, coeff2, testModelY2, testX2, testY2 = example_2_multi_output()
    
    # Example 3: multi-input
    order3, coeff3, testModelY3, testX3, testY3 = example_3_multi_input()
    
    '''
    # Example 4: scaling and decorrelation
    example_4_with_scaling()

    '''
    # Example 5: Basis comparison
    example_5_basis_comparison()

    # Example 6: L1 vs COV comparison
    example_6_L1_vs_COV()
    '''
   
    '''
    print("\n" + "#"*70)
    print("# All examples completed successfully!")
    print("#"*70)
    print("\nKey Takeaways:")
    print("  1. mimoSHORSA automatically identifies important polynomial terms")
    print("  2. Model reduction removes uncertain coefficients iteratively")
    print("  3. Scaling is important for numerical stability")
    print("  4. The method works for single and multiple outputs")
    print("  5. Higher dimensions require careful choice of maxOrder")
    '''
    

if __name__ == '__main__':
    main()
