# generalized extreme value distrubution
# github.com/hpgavin/multivarious ... rvs/gev.py

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


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
    k = np.atleast_1d(k).reshape(-1,1).astype(float)
    n = len(m)   
        
    # Validate parameter dimensions 
    if not (len(m) == n and len(s) == n and len(k) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got m:{len(m)}, s:{len(s)}, k:{len(k)}")

    return x, m, s, k, n


def pdf(x, m, s, k):
    '''
    gev.pdf

    Compute the PDF of the generalized extreme value distribution.
    INPUTS:
        x     : scalar or array-like
        param : list or array-like of [m, s, k]
    OUTPUTS:
        f     : same shape as x, PDF values
    '''

    x, m, s, k, n = _ppp_(x, m, s, k) 

    z = (x - m) / s
    kzp1 = k * z + 1
    f = (1 / s) * np.exp(-kzp1**(-1 / k)) * kzp1**(-1 - 1 / k)

    f = np.where(kzp1 < 0, np.finfo(float).eps, f)

    return np.real(f)


def cdf(x, params):
    '''
    gev.cdf 

    Compute the CDF of the generalized extreme value distribution.
    INPUTS:
        x      : scalar or array-like
        params : array-like [m, s, k]
    OUTPUTS:
        F      : same shape as x, CDF values
    '''
    m, s, k = params

    x, m, s, k, n = _ppp_(x, m, s, k) 

    z = (x - m) / s
    kzp1 = k * z + 1
    F = np.exp(-kzp1**(-1 / k))
    F = np.where(kzp1 < 0, np.finfo(float).eps, F)

    return np.real(F)


def inv(p, m, s, k):
    '''
    gev.inv

    Compute the inverse CDF (quantile function) of the GEV distribution.
    INPUTS:
        p     : scalar or array-like in (0,1)
        param : list or array-like of [m, s, k]
    OUTPUTS:
        x     : same shape as p, quantiles
    '''

    _, m, s, k, n = _ppp_(0, m, s, k) 

    x = m + (s / k) * ((-np.log(p))**(-k) - 1)

    return x


def rnd(m, s, k, N, R=None, seed=None):
    '''
    gev.rnd

    Generate random samples from the GEV distribution.
    
    INPUTS:
        m     : float (n,)
        s     : float (n,)
        k     : float (n,)
        N     : int 
                Number of observations of the gev distribution
        R     : float (n,n) correlation matrix
    OUTPUTS:
        X : ndarray of GEV samples
    '''
    _, m, s, k, n = _ppp_(0, m, s, k) 
   
    _, _, U = correlated_rvs( R, n, N, seed )

    # Apply transformation --- this is wrong ---
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i,:], m[i], s[i], k[i])

    if n == 1:
        X = X.flatten()

    return X
