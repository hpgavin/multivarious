# extreme_value_II distribution (Fréchet)
# github.com/hpgavin/multivarious ... rvs/extreme_value_II

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def pdf(x, m, s, k):
    '''
    extreme_value_II.pdf
    
    Computes the PDF of the Extreme Value Type II (Fréchet) distribution.
    
    Parameters:
        x : array_like
            Evaluation points
        m : float
            Location parameter (lower bound)
        s : float
            Scale parameter (must be > 0)
        k : float
            Shape parameter
    
    Returns:
        f : ndarray
            PDF values at each point in x
    '''
    x = np.asarray(x, dtype=float)
    
    # Check parameter validity
    if s <= 0:
        raise ValueError(f"extII_pdf: s = {s}, must be > 0")
    
    # Initialize PDF as zeros
    f = np.zeros_like(x)
    
    # Only compute for x > m
    valid = x > m
    z = (x[valid] - m) / s
    f[valid] = (k / s) * z**(-1 - k) * np.exp(-z**(-k))
    
    return f


def cdf(x, params ):
    '''
    extreme_value_II.cdf
    
    Computes the CDF of the Extreme Value Type II (Fréchet) distribution.
    
    Parameters:
        x : array_like
            Evaluation points
        params : array_like  [ m , s , k ] 
        m : float
            Location parameter (lower bound)
        s : float
            Scale parameter (must be > 0)
        k : float
            Shape parameter
    
    Returns:
        F : ndarray
            CDF values at each point in x
    '''
    x = np.asarray(x, dtype=float)
    
    m, s, k = params

    # Check parameter validity
    if s <= 0:
        raise ValueError(f"extII_cdf: s = {s}, must be > 0")
    
    # Initialize CDF as zeros
    F = np.zeros_like(x)
    
    # Only compute for x > m
    valid = x > m
    z = (x[valid] - m) / s
    F[valid] = np.exp(-z**(-k))
    
    return F


def inv(P, m, s, k):
    '''
    extreme_value_II.inv
    
    Computes the inverse CDF (quantile function) of the Extreme Value Type II distribution.
    
    Parameters:
        P : array_like
            Probability values (must be in (0, 1))
        m : float
            Location parameter
        s : float
            Scale parameter (must be > 0)
        k : float
            Shape parameter
    
    Returns:
        x : ndarray
            Quantile values corresponding to probabilities P
    '''
    P = np.asarray(P, dtype=float)
    
    # Check parameter validity
    if s <= 0:
        raise ValueError(f"extII_inv: s = {s}, must be > 0")
    
    # Clip probabilities to avoid log(0) or log(1)
    eps = np.finfo(float).eps
    P = np.clip(P, eps, 1 - eps)
    
    # Inverse CDF formula
    x = m + s * (-np.log(P))**(-1 / k)
    
    return x


def rnd(m, s, k, N, R=None, seed=None):
    '''
    extreme_value_II.rnd
    
    Generate random samples from Extreme Value Type II (Fréchet) distribution.
    
    Parameters:
        m : float (n,)
            Location parameter
        s : float (n,)
            Scale parameter (must be > 0)
        k : float (n,)
            Shape parameter
        N : int
            Number of observations of each variable
        R  : float (n,n) optional
             correlation matrix
    
    Returns:
        X : ndarray
            Shape (n, N) array of random samples
    '''
    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    m = np.atleast_1d(m).astype(float)
    s = np.atleast_1d(s).astype(float)
    k = np.atleast_1d(k).astype(float)

    n = len(m)

    if not (len(m) == n and len(s) == n and len(k) == n):  
       raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got m:{len(m)}, s:{len(s)}, k:{len(k)}")

    # Check parameter validity
    if np.any(s) <= 0:
        raise ValueError(f" extreme_value_II.rnd: s = {s}, must be > 0")

    _, _, U = correlated_rvs(R, n, N, seed)

    # Inverse transform: x = m + s * (-log(u))^(-1/k)
    # Transform each variable to its extreme type II  distribution 
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i,:], m[i], s[i], k[i])
    
    if n == 1:
        X = X.flatten()

    return X
