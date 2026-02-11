## uniform distribution
# github.com/hpgavin/multivarious ... rvs/uniform

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, a, b):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        a : float or array_like
            Minimum of the distribution
        b : float or array_like
    ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    a = np.atleast_1d(a).reshape(-1,1).astype(float)
    b = np.atleast_1d(b).reshape(-1,1).astype(float)
    n = len(a)   
    N = len(x)
        
    # Validate parameter dimensions 
    if not (len(a) == n and len(b)):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}")

    # Validate parameter values 
    if np.any(b <= a):
        raise ValueError("uniform: all b values must be greater than corresponding a values")

    return x, a, b, n, N


def pdf(x, a, b):
    '''
    uniform.pdf
    
    Computes the PDF of the uniform distribution on [a, b].
    
    INPUT:
        x   = array_like of evaluation points
        a   = float for Lower bound
        b   = float for Upper bound (must be > a)
    
    OUTPUT:
        f   = ndarray
              PDF values at each point in x
    '''

    x, a, b, n, N = _ppp_(x, a, b)
    
    f = np.zeros((n,N))

    for i in range(n): 
        mask = (x >= a[i]) & (x <= b[i])
        f[i,mask] = 1.0 / (b[i] - a[i])
    
    return f


def cdf(x, params ):
    '''
    uniform.cdf

    Computes the CDF of the uniform distribution on [a, b].
    
    INPUT:
        x   : array_like of evaluation points
        params: array_like  [ a , b ]
        a   : float Lower bound
        b   : float Upper bound (must be > a)
    
    OUTPUT:
        F   = ndarray
              CDF values at each point in x
    '''
    a, b = params
    
    x, a, b, n, N = _ppp_(x, a, b)
    
    F = np.zeros((n,N))

    for i in range(n): 
        mask = (x >= a[i]) & (x <= b[i])
        F[i,mask] = (x[mask] - a[i]) / (b[i] - a[i])
    
    return F


def inv(F, a, b):
    '''
    uniform.inv
    
    Computes the inverse CDF (quantile function) of the uniform distribution.
    INPUT:
        F = array_like of Probability values (must be in [0, 1])
        a = float of Lower bound
        b = float of Upper bound (must be > a)
    
    OUTPUT:
        x = ndarray
            Quantile values corresponding to probabilities F
    '''
    _, a, b, n, _ = _ppp_(0, a, b)

    F = np.asarray(F, dtype=float)
    
    if np.any((F < 0) | (F > 1)):
        raise ValueError('uniform_inv: F must be between 0 and 1')
    
    x = a + F * (b - a)
    
    return x


def rnd(a, b, N, R=None, seed=None ):
    '''
    uniform.rnd
    
    Generate random samples from uniform distribution on [a, b].
    
    INPUTS:
        a = float Lower bound
        b = float Upper bound (must be > a)
        n = int Number of random variables (rows)
        N = int Number of samples (columns)
    
    OUTPUT:
        X : ndarray
            Shape (r, c) array of uniform random samples
    '''
    # Python does not implicitly handle scalars as arrays. 
    # Convert inputs to 2D (column) arrays

    # Determine number of random variables
    # Validate that all parameter arrays have the same length

    _, a, b, n, _ = _ppp_(0, a, b)

    # Generate correlated [0,1]
    _, _, U = correlated_rvs( R, n, N, seed )

    # Transform to [a, b]: x = a + U * (b - a)
    X = a + U * (b - a)

    if n == 1:
       X = X.flatten()
    
    return X
