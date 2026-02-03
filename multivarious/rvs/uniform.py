## uniform distribution
# github.com/hpgavin/multivarious ... rvs/uniform

import numpy as np
from scipy.stats import uniform as scipy_uniform

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


def cdf(x, a, b):
    '''
    uniform.cdf

    Computes the CDF of the uniform distribution on [a, b].
    
    INPUT:
        x   = array_like of evaluation points
        a   = float Lower bound
        b   = float Upper bound (must be > a)
    
    OUTPUT:
        F   = ndarray
              CDF values at each point in x
    '''
    x = np.asarray(x, dtype=float)
    
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


def rnd(a, b, n, N):
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
    if b <= a:
        raise ValueError(f"uniform_rnd: a = {a}, b = {b} — a must be less than b")
    
    # Generate standard uniform [0,1]
    U = np.random.rand(n, N)
    
    # Transform to [a, b]: x = a + u * (b - a)
    X = a + U * (b - a)

    if n == 1:
       X = X.flatten()
    
    return X
