#! /usr/bin/env -S python3 -i
"""
correlated_rvs.py

generate a sample of (n,N) correlated standard normal variables Y
and associated standard uniform variables U 

The correlation matrix R is corrected using the 
Shrinking by the "shrink Newton" method by Nick Higham

Based on Nick Higham's MATLAB implementation:
https://github.com/higham/shrinking

Reference: N. J. Higham, "Computing the nearest correlation matrix—a problem
from finance," IMA Journal of Numerical Analysis, 22(3), 329-343, 2002.
"""

import numpy as np

def shrink_newton(M0, M1, tolN=1e-4, tolB=0):
    """
    Shrinking by Newton's method.
    
    Uses Newton's method to compute the smallest alpha in [0,1] such that 
    S(t) = alpha*M1 + (1-alpha)*M0 is positive semidefinite,
    where M0 is symmetric indefinite and M1 is symmetric positive definite.
    
    Parameters
    ----------
    M0 : ndarray, shape (n, n)
        Symmetric indefinite matrix to be fixed
    M1 : ndarray, shape (n, n)
        Symmetric positive definite target matrix (usually Identity)
    tolN : float, optional
        Convergence tolerance for Newton's method (default: 1e-4)
    tolB : float, optional
        Retained for compatibility with MATLAB version (not used here)
    
    Returns
    -------
    alpha : float
        The shrinking parameter in [0, 1] such that 
        S(alpha) = alpha*M1 + (1-alpha)*M0 is positive semidefinite
    
    Notes
    -----
    The purpose of shrinking is to replace the indefinite symmetric 
    matrix M0 by the positive semidefinite matrix S(alpha).
    
    M0 and M1 are not checked for validity.
    
    Examples
    --------
    >>> C = np.array([[1.0, 0.9, 0.8],
    ...               [0.9, 1.0, 0.9],
    ...               [0.8, 0.9, 1.0]])  # Not PSD
    >>> I = np.eye(3)
    >>> alpha = shrink_newton(C, I)
    >>> C_fixed = alpha * I + (1 - alpha) * C

    -----
    This is a simplified version of Higham's MATLAB code that used
    tridiagonalization (dsytrd) + bisection (dstebz) + inverse iteration 
    (dstein). Here we simply use numpy's eigh which returns eigenvalues 
    in ascending order.
    """

    max_iter = 500

    # Initialize the weight parameter alpha
    alpha_0 = 0.0 
    
    # If M0 is a symmetric matrix with unit diagonal and M1 is identity, 
    # then M is the off-diagonal part of M0
    M = M0 - M1
    
    for n in range(max_iter):
        # Weighted average of M1 (symm) and M0 (I_n)
        S = alpha_0 * M1 + (1 - alpha_0) * M0

        # Compute all eigenvalues and eigenvectors of S
        eigval, eigvec = np.linalg.eigh(S)
    
        # Eigenvector for the smallest eigenvalue of S
        # "eigh" returns eigenvalues in ascending order
        v0 = eigvec[:, 0]
        
        alpha_1 = ( v0 @ M0 @ v0 ) / ( v0 @ M @ v0 )
        
        # convergence criteria
        if abs(alpha_0 - alpha_1) <= tolN:
            return alpha_1
        
        # update and continue
        alpha_0 = alpha_1
    
    raise RuntimeError(f' shrink_newton: not converged in {max_iter} iterations')

def nearcorr_shrink(C, tolN=1e-4):
    """
    Compute nearest correlation matrix using shrinking method.
    
    Convenience wrapper for the common case of fixing a correlation matrix
    by shrinking toward the identity matrix.
    
    Parameters
    ----------
    C : ndarray, shape (n, n)
        Symmetric correlation-like matrix (unit diagonal, values in [-1,1])
        that may not be positive definite
    tolN : float, optional
        Convergence tolerance for Newton's method (default: 1e-4)
    
    Returns
    -------
    C_fixed : ndarray, shape (n, n)
        Nearest positive semidefinite correlation matrix
    alpha : float
        The shrinking parameter used
    
    Examples
    --------
    >>> C = np.array([[1.0, 0.9, 0.8],
    ...               [0.9, 1.0, 0.9],
    ...               [0.8, 0.9, 1.0]])
    >>> C_fixed, alpha = nearcorr_shrink(C)
    >>> print(f"Shrinkage parameter: {alpha:.4f}")
    >>> print(f"Minimum eigenvalue: {np.linalg.eigh(C_fixed)[0][0]:.2e}")
    """

    # Convert C to array and validate its properties
    C = np.asarray(C)
    n = C.shape[0]
    if C.shape != (n, n):
        raise ValueError(f": Correlation matrix must be square {n}×{n}, not {C.shape}")
    if not np.allclose(np.diag(C), 1.0): # diagonals must be 1s
        raise ValueError(": Correlation matrix diagonal must be 1s")
    if np.any(np.abs(C) > 1):
        raise ValueError(": Correlation matrix values must be in [-1,1]")

    In = np.eye(n)
    
    # Find optimal shrinking parameter
    alpha = shrink_newton(C, In, tolN)
    
    # Compute the fixed correlation matrix
    C_fixed =  (1 - alpha) * C + alpha * In
    
    return C_fixed, alpha

def correlated_rvs(R,n,N) 
    """
    Fix a potentialy erroneous correlation matrix, 
    generate correlated standard normal random variables Y (n,N) 
    and associated standard uniform random variables U (n,N) 
    """

    # If no correlation matrix provided, default to identity matrix
    # Identity matrix R = I means all variables are independent (correl'n = 0)
    if R is None:
        R = np.eye(n) # In
        eVal = np.ones(n)
        eVec = np.eye(n)
    else:
        R, alpha = nearcorr_shrink(R, tolN = 1e-4)
        print(f" Correlation matrix shrinkage: {alpha2:.6f}")
        # Eigenvalue decomposition of correlation matrix: R = V @ Λ @ V^T
        #   eVec (V): matrix of eigenvectors (n×n)
        #   eVal (Λ): array of eigenvalues (length n)
        eVal, eVec = np.linalg.eigh(R)
        
        if np.any(eVal < 0):
            raise ValueError(" fix_R_Y_U: R must be positive definite")
        
    # Generate independent standard normal samples: Z ~ N(0, I)
    Z = np.random.randn(n, N)
    
    # Apply correlation structure
    Y = eVec @ np.diag(np.sqrt(eVal)) @ Z

    # Transform to uniform [0,1] via standard normal CDF, preserving correlation
    U = norm.cdf(Y)

    return R, Y, U

# Example usage and testing
if __name__ == "__main__":
    print("Testing shrink_newton on a non-PD correlation matrix\n")
    
    # Example 1: Small matrix from Higham's papers
    C = np.array([[1.00, 0.90, 0.70],
                  [0.90, 1.00, 0.90],
                  [0.70, 0.90, 1.00]])
    
    print("Original matrix C:")
    print(C)
    eigval_orig = np.linalg.eigh(C)[0]
    print(f"\nEigenvalues: {eigval_orig}")
    print(f"Minimum eigenvalue: {eigval_orig[0]:.6f} (negative = not PSD)")
    
    # Fix using shrinking
    C_fixed, alpha = nearcorr_shrink(C)
    
    print(f"\nShrinkage parameter alpha: {alpha:.6f}")
    print(f"\nFixed matrix C_fixed = {alpha:.4f}*I + {1-alpha:.4f}*C:")
    print(C_fixed)
    
    eigval_fixed = np.linalg.eigh(C_fixed)[0]
    print(f"\nEigenvalues: {eigval_fixed}")
    print(f"Minimum eigenvalue: {eigval_fixed[0]:.2e} (should be ≈ 0)")
    
    print(f"\nFrobenius norm of change: {np.linalg.norm(C - C_fixed, 'fro'):.6f}")
    
    # Example 2: Larger random matrix
    print("\n" + "="*60)
    
    n = 25
    print(f"Testing on a {n}x{n} matrix\n")
    np.random.seed(42)
    C2 = np.random.randn(n, n)
    C2 = C2 @ C2.T # symmetric pos.def
    C2 = C2 + 0.10*np.random.randn(n,n)
    C2 = ( C2 + C2.T ) /2 # make symmetric
    C2 = C2 / np.max(C2) # bounded to [-1,1]
    C2 = C2 / np.outer(np.sqrt(np.diag(C2)), np.sqrt(np.diag(C2)))  # Unit diag
    
    eigval_orig2 = np.linalg.eigh(C2)[0]
    print(f"Original minimum eigenvalue: {eigval_orig2[0]:.6f}")
    
    C2_fixed, alpha2 = nearcorr_shrink(C2)
    eigval_fixed2 = np.linalg.eigh(C2_fixed)[0]
    print(C2_fixed)
    
    print(f"Shrinkage parameter: {alpha2:.6f}")
    print(f"Fixed minimum eigenvalue: {eigval_fixed2[0]:.2e}")
    print(f"Frobenius norm of change: {np.linalg.norm(C2 - C2_fixed, 'fro'):.6f}")
