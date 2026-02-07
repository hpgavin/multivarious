## uniform distribution
# github.com/hpgavin/multivarious ... rvs/uniform

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

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
    x = np.asarray(x, dtype=float)
    
    if b <= a:
        raise ValueError(f"uniform_pdf: a = {a}, b = {b} — a must be less than b")
    
    f = np.zeros_like(x)
    valid = (x >= a) & (x <= b)
    f[valid] = 1.0 / (b - a)
    
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
    x = np.asarray(x, dtype=float)

    a, b = params
    
    if b <= a:
        raise ValueError(f"uniform_cdf: a = {a}, b = {b} — a must be less than b")
    
    F = np.clip((x - a) / (b - a), 0, 1)
    
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
    F = np.asarray(F, dtype=float)
    
    if b <= a:
        raise ValueError(f'uniform_inv: a = {a}, b = {b} → a must be less than b')
    
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
    a = np.atleast_2d(a).reshape(-1,1).astype(float)
    b = np.atleast_2d(b).reshape(-1,1).astype(float)

    # Determine number of random variables
    n = len(a)

    # Validate that all parameter arrays have the same length
    if not (len(a) == n and len(b) == n ):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}")

    if np.any(b <= a):
        raise ValueError(f" uniform.rnd: a = {a}, b = {b} : a must be less than b")
    
    # Generate correlated [0,1]
    _, _, U = correlated_rvs( R, n, N, seed )

    # Transform to [a, b]: x = a + U * (b - a)
    X = a + U * (b - a)

    if n == 1:
       X = X.flatten()
    
    return X
