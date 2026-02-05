# exponential distribution
# github.com/hpgavin/multivarious ... rvs/exponential

import numpy as np
from scipy.stats import norm

def pdf(x, muX):
    '''
    exponential.pdf

    Computes the Probability Density Function (PDF) of the exponential distribution.

    INPUTS:
      x    = evaluation points (must be x >= 0)
      muX  = mean of the exponential distribution
 
    OUTPUT:
      f    = PDF evaluated at x
 
    FORMULA:
      f(x) = (1/muX) * exp(-x / muX), for x >= 0
    '''
    x = np.asarray(x, dtype=float)

    # Prevent negative or zero values (log not defined)
    x = np.where(x < 0, 0.01, x)

    f = np.exp(-x / muX) / muX
    
    return f


def cdf(x, muX):
    '''
    exponential.cdf

    Computes the Cumulative Distribution Function (CDF) of the exponential.

    INPUTS:
      x    = values at which to evaluate the CDF (x >= 0)
      muX  = mean of the exponential distribution
 
    OUTPUT:
      F    = CDF values at each x
 
    FORMULA:
      F(x) = 1 - exp(-x / muX), for x >= 0
    '''
    x = np.asarray(x, dtype=float)

    # Prevent issues with x <= 0
    x = np.where(x <= 0, 0.01, x)

    F = 1.0 - np.exp(-x / muX)
    return F


def inv(P, muX):
    '''
    exponential.inv

    Computes the inverse CDF (quantile function) of the exponential distribution.

    INPUTS:
      P    = probability values (0 <= P <= 1)
      muX  = mean of the exponential distribution

    OUTPUT:
      X    = quantiles corresponding to P

    FORMULA:
      X = -muX * log(1 - P)
    '''
    P = np.asarray(P, dtype=float)

    # Clip invalid P values just like in MATLAB
    P = np.where(P < 0, 0.0, P)
    P = np.where(P > 1, 1.0, P)

    X = -muX * np.log(1 - P)
    return X


def rnd(muX, N, R=None):
    '''
    exponential.rnd

    Generate random samples from an exponential distribution with mean muX.

    INPUTS:
        muX  = mean (n,) one component for each r.v.
        N = number of values for each variable in the sample (columns)
        R = correlation matrix (n x n) - not yet implemented
 
    OUTPUT:
        X    = random samples shaped (n, N)

    METHOD:
        Use inverse CDF method: x = -muX * log(U), where U ~ Uniform(0,1)
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    muX = np.atleast_1d(muX).astype(int)

    # Check parameter validity
    if np.any(muX <= 0) or np.any(np.isinf(muX)):
        raise ValueError(f"exp.rnd: muX must be > 0 and finite")

    n = len(muX)

    if R is None:
        R = np.eye(n) # In
    
    # Generate correlated standard normal ~N(0,1)
    Z = np.random.randn(n, N)

    # Eigenvalue decomposition of correlation matrix: R = V @ Λ @ V^T
    #   eVec (V): matrix of eigenvectors (n×n)
    #   eVal (Λ): array of eigenvalues (length n)
    eVal, eVec = np.linalg.eigh(R)

    if np.any(eVal < 0):
        raise ValueError("beta.rnd: R must be positive definite")

    # Apply correlation structure
    Y = eVec @ np.diag(np.sqrt(eVal)) @ Z

    # Generate correlated standard uniform ~U[0,1]
    U = norm.cdf(Y)
    
    # Inverse transform: x = -muX * log(U)
    X = -muX * np.log(U)

    if n == 1:
        X = X.flatten()

    return X
