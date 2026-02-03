# beta distribution
# github.com/hpgavin/multivarious ... rvs/beta

import numpy as np
from scipy.special import beta as beta_func
from scipy.special import betainc
from scipy.special import betaincinv
from scipy.stats import norm


def pdf(x, a, b, q, p):
    '''
    beta.pdf

    Computes the Probability Density Function (PDF) of the Beta distribution
    with lower bound a, upper bound b, and shape parameters q and p.

    INPUTS:
        x : array_like
            Evaluation points
        a : float
            Minimum of the distribution
        b : float
            Maximum of the distribution (must be > a)
        q : float
            First shape parameter
        p : float
            Second shape parameter
 
    OUTPUT:
        f : ndarray
            PDF evaluated at x
    '''
    x = np.asarray(x, dtype=float)
    
    # Check parameter validity
    if b <= a:
        raise ValueError(f"beta_pdf: a = {a}, b = {b} — a must be less than b")
    
    # Initialize PDF output as zeros
    f = np.zeros_like(x)

    # Only compute for values within [a, b]
    valid = (x >= a) & (x <= b)
    
    # Beta PDF formula (with change of variable from [0,1] to [a,b])
    numerator = (x[valid] - a) ** (q - 1) * (b - x[valid]) ** (p - 1)
    denominator = beta_func(q, p) * (b - a) ** (q + p - 1)
    f[valid] = numerator / denominator

    return f


def cdf(x, a, b, q, p):
    '''
    beta.cdf

    Computes the Cumulative Distribution Function (CDF) of the beta distribution
    with lower bound a, upper bound b, and shape parameters q and p.

    INPUTS:
        x = array_like 
            Evaluation points
        a = float 
            Minimum of the distribution
        b = float
            Maximum of the distribution (must be > a)
        q = float
            First shape parameter
        p = float 
            Second shape parameter

    OUTPUT:
        F = ndarray (CDF evaluated at x)
        
    Formula: 
        F(x) = I_{(x - a) / (b - a)} (q, p)
    where I is the regularized incomplete beta function.
    '''
    x = np.asarray(x, dtype=float)

    # Check parameter validity
    if b <= a:
        raise ValueError(f"beta_cdf: a = {a}, b = {b} — a must be less than b")

    # Compute z = (x - a) / (b - a), clipped to [0, 1]
    z = (x - a) / (b - a)
    z = np.clip(z, 0, 1)

    # Evaluate the regularized incomplete beta function
    F = betainc(q, p, z)

    return F


def inv(F, a, b, q, p):
    '''
    beta.inv

    Compute the inverse CDF (quantile function) of the beta distribution.

    INPUTS:
        F : array_like
            Non-exceedance probability values (between 0 and 1)
        a : float
            Lower bound of the distribution
        b : float
            Upper bound of the distribution (must be > a)
        q : float
            First shape parameter
        p : float
            Second shape parameter

    OUTPUT:
    x : ndarray
        Quantile values corresponding to input probabilities F
    '''
    F = np.asarray(F, dtype=float)
    
    # Check that a < b (valid interval)
    if b <= a:
        raise ValueError(f'beta_inv: a = {a}, b = {b} → a must be less than b')
    
    # Check that F values are valid probabilities
    if np.any((F < 0) | (F > 1)):
        raise ValueError('beta_inv: F must be between 0 and 1')

    # Compute inverse of regularized incomplete beta function
    z = betaincinv(q, p, F)

    # Rescale from [0, 1] to [a, b]
    x = a + z * (b - a)

    return x


def rnd(a, b, q, p, N, R=None):
    '''
    beta.rnd
    Generate N observations of correlated (or uncorrelated) beta random variables.

    INPUT:
        a : float or array_like
            Lower bound(s) of the distribution. If array, shape (n,) for n variables.
        b : float or array_like
            Upper bound(s) of the distribution. If array, shape (n,) for n variables.
        q : float or array_like
            First shape parameter(s). If array, shape (n,) for n variables.
        p : float or array_like
            Second shape parameter(s). If array, shape (n,) for n variables.
        N : int
            Number of observations (samples) to generate.
        R : ndarray, optional
            n×n correlation matrix of standard normal deviates.
            If None, defaults to identity matrix (uncorrelated samples).

    OUTPUT:
        X : ndarray
            Shape (n, N) array of correlated beta random samples.
            Each row corresponds to one random variable.
            Each column corresponds to one observation.

    Method:
        1. Perform eigenvalue decomposition of correlation matrix R
        2. Generate uncorrelated standard normal samples
        3. Apply correlation structure: Y = eVec @ sqrt(eVal) @ Z
        4. Transform to uniform via standard normal CDF
        5. Transform to beta via inverse CDF for each variable

    Example (Usage):
        # Single variable, uncorrelated samples
            x = rnd(0, 1, 2, 5, N=1000)
        
        # Multiple correlated variables
            a = np.array([0, 1])
            b = np.array([1, 3])
            q = np.array([2, 3])
            p = np.array([5, 4])
            R = np.array([[1.0, 0.7], [0.7, 1.0]])
            x = rnd(a, b, q, p, N=1000, R=R)
    '''
    
    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    a = np.atleast_1d(a).astype(float) 
    b = np.atleast_1d(b).astype(float)
    q = np.atleast_1d(q).astype(float)
    p = np.atleast_1d(p).astype(float)
    
    # Determine number of random variables
    n = len(a)
    

    # -------------------------------------- Input Validations:
    # Validate that all parameter arrays have the same length
    if not (len(b) == n and len(q) == n and len(p) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}, q:{len(q)}, p:{len(p)}")
    
    if np.any(b <= a):
        raise ValueError("beta_rnd: all b values must be greater than corresponding a values")
    if np.any(q <= 0):
        raise ValueError("beta_rnd: q must be positive")
    if np.any(p <= 0):
        raise ValueError("beta_rnd: p must be positive")
    
    # If no correlation matrix provided, default to identity matrix
    # Identity matrix R = I means all variables are independent (correlation = 0)
    if R is None:
        R = np.eye(n) # In
    
    # Convert R to array and validate its properties
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError(f"beta.rnd: Correlation matrix R must b square {n}×{n}, got {R.shape}")
    
    if not np.allclose(np.diag(R), 1.0): # diagonals must be 1s
        raise ValueError("beta.rnd: diagonal of R must equal 1")
    
    if np.any(np.abs(R) > 1): # all elements must be [-1, 1] i.e valid correlations
        raise ValueError("beta.rnd: R values must be between -1 and 1")
    # # -------------------------------------- End Input Validations
    
    # Eigenvalue decomposition of correlation matrix: R = V @ Λ @ V^T
    #   eVec (V): matrix of eigenvectors (n×n)
    #   eVal (Λ): array of eigenvalues (length n)
    eVal, eVec = np.linalg.eig(R)
    
    if np.any(eVal < 0):
        raise ValueError("beta.rnd: R must be positive definite")
    
    # Generate independent standard normal samples: Z ~ N(0, I)
    Z = np.random.randn(n, N) 
    
    # Apply correlation structure
    # Y = V @ sqrt(Λ) @ Z, so Y ~ N(0, R)
    #   = eVec @ sqrt(eVal) @ Z
    Y = eVec @ np.diag(np.sqrt(eVal)) @ Z
    
    # Transform to uniform [0,1] via standard normal CDF, preserving correlation
    U = norm.cdf(Y)
    
    # Transform each variable to its beta distribution via inverse CDF
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i, :], a[i], b[i], q[i], p[i])
        
    if n == 1:
        X = X.flatten()

    return X
