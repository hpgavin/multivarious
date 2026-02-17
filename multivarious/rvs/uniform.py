## uniform distribution
# github.com/hpgavin/multivarious ... rvs/uniform

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, a, b):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        a : float or array_like
            Minimum of the distribution
        b : float or array_like
            Maximum of the distribution (must be > a)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        a : ndarray
            Lower bounds as column array
        b : ndarray
            Upper bounds as column array
        n : int
            Number of random variables
        N : int
            Number of evaluation points
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    a = np.atleast_1d(a).reshape(-1,1).astype(float)
    b = np.atleast_1d(b).reshape(-1,1).astype(float)
    n = len(a)   
    N = len(x)
        
    # Validate parameter dimensions 
    if not (len(a) == n and len(b) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}")

    # Validate parameter values 
    if np.any(b <= a):
        raise ValueError("uniform: all b values must be greater than corresponding a values")

    return x, a, b, n, N


def pdf(x, a, b):
    """
    uniform.pdf
    
    Computes the PDF of the uniform distribution on [a, b].
    
    INPUTS:
        x : array_like
            Evaluation points
        a : float or array_like
            Lower bound(s)
        b : float or array_like
            Upper bound(s) (must be > a)
    
    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables
    
    Notes
    -----
    The uniform distribution has constant probability density 1/(b-a) 
    over the interval [a, b] and zero elsewhere.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """

    x, a, b, n, N = _ppp_(x, a, b)
    
    f = np.zeros((n,N))

    for i in range(n): 
        mask = (x >= a[i]) & (x <= b[i])
        f[i,mask] = 1.0 / (b[i] - a[i])
    
    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f


def cdf(x, params):
    """
    uniform.cdf

    Computes the CDF of the uniform distribution on [a, b].
    
    INPUTS:
        x : array_like
            Evaluation points
        a : float or array_like
            Lower bound(s)
        b : float or array_like
            Upper bound(s) (must be > a)
    
    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables
    
    Notes
    -----
    F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, and 1 for x > b.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """

    a, b = params
    
    x, a, b, n, N = _ppp_(x, a, b)
    
    F = np.zeros((n,N))

    for i in range(n): 
        mask_in = (x >= a[i]) & (x <= b[i])
        mask_above = x > b[i]
        F[i,mask_in] = (x[mask_in] - a[i]) / (b[i] - a[i])
        F[i,mask_above] = 1.0
    
    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, a, b):
    """
    uniform.inv
    
    Computes the inverse CDF (quantile function) of the uniform distribution.
    
    INPUTS:
        F : array_like
            Probability values (must be in [0, 1])
        a : float or array_like
            Lower bound(s)
        b : float or array_like
            Upper bound(s) (must be > a)
    
    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities F
    
    Notes
    -----
    x = a + F(b - a) for F in [0, 1].
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """
    _, a, b, n, _ = _ppp_(0, a, b)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    x = a + F * (b - a)
    
    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

    return x

def rnd(a, b, N, R=None, seed=None):
    """
    uniform.rnd
    
    Generate random samples from uniform distribution on [a, b].
    
    INPUTS:
        a : float or array_like, shape (n,)
            Lower bound(s)
        b : float or array_like, shape (n,)
            Upper bound(s) (must be > a)
        N : int
            Number of samples per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility
    
    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Array of uniform random samples.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.
    
    Notes
    -----
    Uses inverse transform method with correlated uniform variates.
    
    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """
    # Python does not implicitly handle scalars as arrays. 
    # Convert inputs to 2D (column) arrays

    # Determine number of random variables
    # Validate that all parameter arrays have the same length

    _, a, b, n, _ = _ppp_(0, a, b)

    # Generate correlated uniform [0,1]
    _, _, U = correlated_rvs(R, n, N, seed)

    # Transform to [a, b]: x = a + U * (b - a)
    X = inv(U, a, b )

    return X
