# generalized extreme value distrubution
# github.com/hpgavin/multivarious ... rvs/gev.py

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

def pdf(x, m, s, k):
    '''
    gev.pdf

    Compute the PDF of the generalized extreme value distribution.
    Parameters:
        x     : scalar or array-like
        param : list or array-like of [m, s, k]
    Returns:
        f     : same shape as x, PDF values
    '''
    z = (x - m) / s
    kzp1 = k * z + 1
    f = (1 / s) * np.exp(-kzp1**(-1 / k)) * kzp1**(-1 - 1 / k)
    f = np.where(kzp1 < 0, np.finfo(float).eps, f)

    return np.real(f)


def cdf(x, params):
    '''
    gev.cdf 

    Compute the CDF of the generalized extreme value distribution.
    Parameters:
        x      : scalar or array-like
        params : array-like [m, s, k]
    Returns:
        F      : same shape as x, CDF values
    '''
    m, s, k = params
    z = (x - m) / s
    kzp1 = k * z + 1
    F = np.exp(-kzp1**(-1 / k))
    F = np.where(kzp1 < 0, np.finfo(float).eps, F)

    return np.real(F)


def inv(p, m, s, k):
    '''
    gev.inv

    Compute the inverse CDF (quantile function) of the GEV distribution.
    Parameters:
        p     : scalar or array-like in (0,1)
        param : list or array-like of [m, s, k]
    Returns:
        x     : same shape as p, quantiles
    '''
    x = m + (s / k) * ((-np.log(p))**(-k) - 1)

    return x


def rnd(m, s, k, N, R=None, seed=None):
    '''
    gev.rnd

    Generate random samples from the GEV distribution.
    
    Parameters:
        m     : float (n,)
        s     : float (n,)
        k     : float (n,)
        N     : int 
                Number of observations of the gev distribution
        R     : float (n,n) correlation matrix
    Returns:
        X : ndarray of GEV samples
    '''
    # Python does not implicitly handle scalars as arrays. 
    # Convert inputs to arrays
    m = np.atleast_1d(m).astype(float)
    s = np.atleast_1d(s).astype(float)
    k = np.atleast_1d(k).astype(float)

    # Determine number of random variables
    n = len(m)

    # Validate that all parameter arrays have the same length
    if not (len(m) == n and len(s) == n and len(k) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got m:{len(m)}, s:{len(s)}, k:{len(k)}")
    
    _, _, U = correlated_rvs( R, n, N, seed )

    # Apply transformation --- this is wrong ---
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i,:], m[i], s[i], k[i])

    if n == 1:
        X = X.flatten()

    return X
