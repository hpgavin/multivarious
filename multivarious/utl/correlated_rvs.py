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

#from multivarious.rvs import normal
#from scipy.stats import norm as scipy_norm

from scipy.special import erf
from math import sqrt

def shrink_newton(M0, M1, tolrnc=1e-4 ):
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
    tolrnc : float, optional
        Convergence tolerance for Newton's method (default: 1e-4)
    
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
    >>> C_nnd = alpha * I + (1 - alpha) * C

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
    
    for iter in range(max_iter):
        # Weighted average of M1 (symm) and M0 (I_n)
        S = alpha_0 * M1 + (1 - alpha_0) * M0

        # Compute all eigenvalues and eigenvectors of S
        eigval, eigvec = np.linalg.eigh(S)
    
        # Eigenvector for the smallest eigenvalue of S
        # "eigh" returns eigenvalues in ascending order
        val0 = eigval[0] 
        vec0 = eigvec[:, 0]
        
        alpha_1 = ( vec0 @ M0 @ vec0 ) / ( vec0 @ M @ vec0 )
        
        # convergence criteria
        if val0 > -tolrnc or abs(alpha_0 - alpha_1) <= tolrnc:
            return alpha_0, iter, val0
        
        # update and continue
        alpha_0 = alpha_1
    
    raise RuntimeError(f' shrink_newton: not converged in {max_iter} iterations')


def nearcorr_shrink(C, tolrnc=1e-4):
    """
    Compute nearest correlation matrix using shrinking method.
    
    Convenience wrapper for the common case of fixing a correlation matrix
    by shrinking toward the identity matrix.
    
    Parameters
    ----------
    C : ndarray, shape (n, n)
        Symmetric correlation-like matrix (unit diagonal, values in [-1,1])
        that may not be positive definite
    tolrnc : float, optional
        Convergence tolerance for Newton's method (default: 1e-4)
    
    Returns
    -------
    C_nnd : ndarray, shape (n, n)
        Nearest positive semidefinite correlation matrix
    alpha : float
        The shrinking parameter used
    
    Examples
    --------
    >>> C = np.array([[1.0, 0.9, 0.8],
    ...               [0.9, 1.0, 0.9],
    ...               [0.8, 0.9, 1.0]])
    >>> C_nnd, alpha = nearcorr_shrink(C)
    >>> print(f"\nShrinkage parameter alpha: {alpha:.6f}  eigenvalue {eval0} in  {iter} iterations")
    >>> print(f"Minimum eigenvalue: {np.linalg.eigh(C_nnd)[0][0]:.2e}")
    """
 
    # Convert C to array and validate its properties
    C = np.asarray(C)
    n = C.shape[0]
    if C.shape != (n, n):
        raise ValueError(f": Correlation matrix must be square ({n},{n}), not {C.shape}")
    if not np.allclose(np.diag(C), 1.0): # diagonals must be 1s
        raise ValueError(": Correlation matrix diagonal must be 1s")
    if np.any(np.abs(C) > 1):
        raise ValueError(": Correlation matrix values must be in [-1,1]")

    In = np.eye(n)
    
    # Find optimal shrinking parameter
    alpha, iter, eval0 = shrink_newton(C, In, tolrnc)
    
    # Compute the fixed correlation matrix
    C_nnd =  (1 - alpha) * C + alpha * In
    
    return C_nnd, alpha, iter, eval0


def correlated_rvs(R, n, N=1, seed=None):
    """
    Fix a potentialy erroneous correlation matrix, 
    generate correlated standard normal random variables Y (n,N) 
    and associated standard uniform random variables U (n,N) 
    """
    rng = np.random.default_rng(seed)

    tolrnc = 0.05  # eigenvalue tolerance
    # If no correlation matrix provided, default to identity matrix
    # Identity matrix R = I means all variables are independent (correl'n = 0)
    if R is None:
        R = np.eye(n) # In
        eigval = np.ones(n)
        eigvec = np.eye(n)
    elif n > 1:
        R, alpha, iter, eval0 = nearcorr_shrink(R, tolrnc)
        print(f" correlated_rvs: Correlation matrix shrinkage ")
        print(f"          alpha: {alpha:.6f}, iter: {iter}, eval[0]: {eval0:.6f}")
        # Eigenvalue decomposition of correlation matrix: R = V @ Λ @ V^T
        #   eigvec (V): matrix of eigenvectors (n×n)
        #   eigval (Λ): array of eigenvalues (length n)
        eigval, eigvec = np.linalg.eigh(R)
        
        if np.any(eigval < -2*tolrnc):
            raise ValueError(f" correlated_rvs: R must be positive definite, eigval0 = {eigval[0]:10.2e}, iter = {iter}")
        
    # Generate independent standard normal samples: Z ~ N(0, I)
    Z = rng.standard_normal((n, N))
    
    # Apply correlation structure
    if n > 1: 
        Y = eigvec @ np.diag(np.sqrt(eigval)) @ Z
    else:
        Y = Z

    # Transform to uniform [0,1] via standard normal CDF, preserving correlation
    # Standard normal CDF of Y are correlated uniformly distributed rv's in [0 1]
    U = (1.0 + erf(Y / sqrt(2.0))) / 2.0  

    return R, Y, U


# Example usage and testing
if __name__ == "__main__":
    print("Testing shrink_newton on a non-PD correlation matrix\n")
    
    # seed = 42
    rng = np.random.default_rng()

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
    C_nnd, alpha, iter, eval0 = nearcorr_shrink(C)
    
    print(f"\nShrinkage parameter alpha: {alpha:.6f}  eigenvalue {eval0} in  {iter} iterations")
    print(f"\nFixed matrix C_nnd = {alpha:.4f}*I + {1-alpha:.4f}*C:")
    print(C_nnd)
    
    eigval_nnd = np.linalg.eigh(C_nnd)[0]
    print(f"\nEigenvalues: {eigval_nnd}")
    print(f"Minimum eigenvalue: {eigval_nnd[0]:.2e} (should be ≈ 0)")
    
    print(f"\nFrobenius norm of change: {np.linalg.norm(C - C_nnd, 'fro'):.6f}")
    
    # Example 2: Larger random matrix
    print("\n" + "="*60)
    
    n = 10
    c = 1*n  # Columns of Z ... c > n : more positive definite correlation matx
    q =  2   # larger q, less positive definite correlation matx
    print(f"Testing on a {n}x{n} matrix\n")
    Z = rng.standard_normal((n, c))

    C2 = Z @ Z.T # symmetric pos.def
    C2 = C2 + q*rng.standard_normal((n,n))  
    C2 = ( C2 + C2.T ) / 2.0 # make symmetric, not necc pos.def
    C2 = C2 / np.max(np.abs(C2)) # bounded to [-1,1]
    C2 = C2 / np.outer(np.sqrt(np.diag(C2)), np.sqrt(np.diag(C2)))  # Unit diag
    C2 = C2 / np.max(np.abs(C2)) # re-check bounded to [-1,1]
    
    eigval_orig2 = np.linalg.eigh(C2)[0]
    print(f"Original minimum eigenvalue: {eigval_orig2[0]:.6f}")
    
    C2_nnd, alpha, iter, eval0 = nearcorr_shrink(C2)
    eigval_nnd2 = np.linalg.eigh(C2_nnd)[0]
    print(C2_nnd)
    
    print(f"\nShrinkage parameter alpha: {alpha:.6f}  eigenvalue {eval0} in  {iter} iterations")
    print(f"Fixed minimum eigenvalue: {eigval_nnd2[0]:.2e}")
    print(f"Frobenius norm of change: {np.linalg.norm(C2 - C2_nnd, 'fro'):.6f}")
