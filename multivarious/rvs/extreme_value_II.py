# extreme_value_II distribution (Fréchet)
# github.com/hpgavin/multivarious ... rvs/extreme_value_II

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# generic pre processing of parameters (ppp) 

def _ppp_(x, m, s, k):
    '''
    Validate and preprocess input parameters for consistency and correctness.

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
    ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    m = np.atleast_1d(m).reshape(-1,1).astype(float)
    s = np.atleast_1d(s).reshape(-1,1).astype(float)
    k = np.atleast_1d(k).astype(float)
    n = len(m)   
    N = len(x)   
        
    # Validate parameter dimensions 
    if not (len(m) == n and len(s) == n and len(k) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got m:{len(m)}, s:{len(s)}, k:{len(k)}")

   # Validate parameter values 
    if np.any(m <= 0):
        raise ValueError("extreme_value_II: m must be > 0")
    if np.any(s <= 0):
        raise ValueError("extreme_value_II: s must be > 0")
    if np.any(k <= 0):
        raise ValueError("extreme_value_II: k must be > 0")

    return x, m, s, k, n, N


def pdf(x, m, s, k):
    '''
    extreme_value_II.pdf
    
    Computes the PDF of the Extreme Value Type II (Fréchet) distribution.
    
    INPUTS:
        x : array_like
            Evaluation points
        m : float
            Location parameter (lower bound)
        s : float
            Scale parameter (must be > 0)
        k : float
            Shape parameter
    
    OUTPUTS:
        f : ndarray
            PDF values at each point in x
    '''

    x, m, s, k, n, N = _ppp_(x, m, s, k)
    
    f = np.zeros((n,N))  # Initialize PDF as zeros
    
    for i in range(n): 
        mask = x > m[i]  # Compute only for x > m
        z = (x[mask] - m[i]) / s[i]
        f[i,mask] = (k[i] / s[i]) * z**(-1 - k[i]) * np.exp(-z**(-k[i]))
    
    return f


def cdf(x, params ):
    '''
    extreme_value_II.cdf
    
    Computes the CDF of the Extreme Value Type II (Fréchet) distribution.
    
    INPUTS:
        x : array_like
            Evaluation points
        params : array_like  [ m , s , k ] 
        m : float
            Location parameter (lower bound)
        s : float
            Scale parameter (must be > 0)
        k : float
            Shape parameter
    
    OUTPUTS:
        F : ndarray
            CDF values at each point in x
    '''
    m, s, k = params

    x, m, s, k, n, N = _ppp_(x, m, s, k)

    F = np.zeros((n,N))  # Initialize PDF as zeros
    
    # Only compute for x > m
    for i in range(n): 
        mask = x > m[i]  # Compute only for x > m
        z = (x[mask] - m) / s
        F[i,mask] = np.exp(-z**(-k[i]))
    
    return F


def inv(P, m, s, k):
    '''
    extreme_value_II.inv
    
    Computes the inverse CDF (quantile function) of the Extreme Value Type II distribution.
    
    INPUTS:
        P : array_like
            Probability values (must be in (0, 1))
        m : float
            Location parameter
        s : float
            Scale parameter (must be > 0)
        k : float
            Shape parameter
    
    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities P
    '''

    _, m, s, k, n, _ = _ppp_(0, m, s, k)

    P = np.asarray(P, dtype=float)
    
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
    
    INPUTS:
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
    
    OUTPUTS:
        X : ndarray
            Shape (n, N) array of random samples
    '''
    _, m, s, k, n, _ = _ppp_(0, m, s, k)

    _, _, U = correlated_rvs(R, n, N, seed)

    # Inverse transform: x = m + s * (-log(u))^(-1/k)
    # Transform each variable to its extreme type II  distribution 
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i,:], m[i], s[i], k[i])
    
    if n == 1:
        X = X.flatten()

    return X
