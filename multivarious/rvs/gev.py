# generalized extreme value distrubution
# github.com/hpgavin/multivarious ... rvs/gev.py

import numpy as np

def pdf(x, param):
    '''
    gev.pdf

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
    gev.cdf 

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
    gev.inv

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
    gev.rnd

    Generate random samples from the GEV distribution.
    
    Parameters:
        param : list [m, s, k]
        r     : int or ndarray
                If c is None: treat r as pre-generated samples
                If c is provided: r is number of rows
        c     : int or None
                Number of columns (optional)
    Returns:
        x : ndarray of GEV samples
    '''
    m, s, k = param

    if c is None:
        # r is a pre-generated sample matrix
        u = np.asarray(r)
    else:
        # Generate r×c uniform samples
        u = np.random.rand(r, c)

    X = m + (s / k) * ((-np.log(u))**(-k) - 1)

    if r == 1:
        X = X.flatten()

    return X
