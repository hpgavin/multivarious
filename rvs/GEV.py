# multivarious/distributions/gev.py

import numpy as np

def pdf(x, param):
    '''
    Compute the PDF of the generalized extreme value distribution.
    Parameters:
        x     : scalar or array-like
        param : list or array-like of [m, s, k]
    Returns:
        f     : same shape as x, PDF values
    '''
    m, s, k = param
    z = (x - m) / s
    arg = 1 + k * z
    f = (1 / s) * np.exp(-arg**(-1 / k)) * arg**(-1 - 1 / k)
    f = np.where(arg < 0, np.finfo(float).eps, f)
    return np.real(f)


def cdf(x, param):
    '''
    Compute the CDF of the generalized extreme value distribution.
    Parameters:
        x     : scalar or array-like
        param : list or array-like of [m, s, k]
    Returns:
        F     : same shape as x, CDF values
    '''
    m, s, k = param
    z = (x - m) / s
    arg = 1 + k * z
    F = np.exp(-arg**(-1 / k))
    F = np.where(arg < 0, np.finfo(float).eps, F)
    return np.real(F)


def inv(p, param):
    '''
    Compute the inverse CDF (quantile function) of the GEV distribution.
    Parameters:
        p     : scalar or array-like in (0,1)
        param : list or array-like of [m, s, k]
    Returns:
        x     : same shape as p, quantiles
    '''
    m, s, k = param
    x = m + (s / k) * ((-np.log(p))**(-k) - 1)
    return x


def rnd(param, r, c=None):
    '''
    Generate random samples from the GEV distribution.
    Parameters:
        param : [m, s, k]
        r     : rows OR uniform sample array
        c     : cols (optional)
    Returns:
        x     : random matrix (r x c) or shape of r if r is array-like
    '''
    m, s, k = param

    if c is None:
        # Case: r is a matrix of uniform samples (from norm_cdf or others)
        u = np.asarray(r)
    else:
        # Case: generate uniform [0,1] samples
        u = np.random.rand(r, c)

    x = m + (s / k) * ((-np.log(u))**(-k) - 1)
    return x
