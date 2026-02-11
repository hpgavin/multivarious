## Student's t distribution
# github.com/hpgavin/multivarious ... rvs/students_t

import numpy as np
import math
from scipy.special import gamma
from scipy.special import betainc, betaincinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(t, k):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        t : array_like
            Evaluation points
        k : int or float or array_like
            Degrees of freedom (must be > 0)

    OUTPUTS:
        t : ndarray
            Evaluation points as array
        k : ndarray
            Degrees of freedom as column array (integer type)
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    t = np.atleast_1d(t).astype(float)
    k = np.atleast_1d(k).reshape(-1,1).astype(int)
    n = len(k)   
    N = len(t)
        
    # Validate parameter values 
    if np.any(k <= 0):
        raise ValueError("students_t: k must be > 0")

    return t, k, n, N


def pdf(t, k):
    """
    students_t.pdf

    Computes the PDF of the Student's t-distribution with k degrees of freedom.

    INPUTS:
        t : array_like
            Evaluation points
        k : int or float or array_like, shape (n,)
            Degrees of freedom (must be > 0)
    
    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in t for each of n random variables

    Notes
    -----
    f(t) = Γ((k+1)/2) / (√(πk) Γ(k/2)) * (1 + t²/k)^(-(k+1)/2)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    
    t, k, n, _ = _ppp_(t, k)

    numerator = gamma((k + 1) / 2)
    denominator = np.sqrt(k * np.pi) * gamma(k / 2)
    power = -(k + 1) / 2
    f = (numerator / denominator) * (1 + (t**2) / k) ** power

    return f

#   # Compute the PDF using the known closed-form
#   f = (np.exp(-(k + 1) * np.log(1 + (t ** 2) / k) / 2)) / (np.sqrt(k) * beta_func(k / 2, 0.5))

    return f


def cdf(t, k):
    """
    students_t.cdf

    Computes the CDF of the Student's t-distribution with k degrees of freedom.
    Handles k = 1 and k = 2 analytically, and uses recurrence relations
    for integer k > 2.

    INPUTS:
        t : array_like
            Evaluation points
        k : int or float or array_like, shape (n,)
            Degrees of freedom (must be > 0)

    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in t for each of n random variables

    Notes
    -----
    For k=1: Cauchy distribution, F(t) = 1/2 + arctan(t)/π
    For k=2: F(t) = 1/2 + t/(2√(2+t²))
    For k>2: Uses recurrence relation

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    
    t, k, n, N = _ppp_(t, k)

    F = np.zeros((n,N))

    for i in range(n): 

        a = k[i] / 2.0
        x = k[i] / (k[i] + t**2)

        F[i,t == 0] = 0.5

        mask = t > 0
        F[i,mask] = 1 - 0.5 * betainc(a, 0.5, x[mask])

        mask = t < 0
        F[i,mask] =     0.5 * betainc(a, 0.5, x[mask])

    return F


def inv(p, k):
    """
    students_t.inv

    Computes the inverse CDF (quantile function) of the Student's t-distribution
    with k degrees of freedom, using the inverse incomplete beta function.

    INPUTS:
        p : array_like
            Probability values (must be in [0, 1])
        k : int or float or array_like, shape (n,)
            Degrees of freedom (must be > 0)

    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities p

    Notes
    -----
    Uses relationship with incomplete beta function.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    
    _, k, _, _ = _ppp_(0, k)
    
    p = np.asarray(p)

    # Compute the inverse CDF using the relationship with the incomplete beta function
    # betaincinv(a, b, y) finds x such that betainc(a, b, x) = y
    z = betaincinv(k / 2.0, 0.5, 2 * np.minimum(p, 1 - p))
    
    # Convert from beta quantile to t quantile
    x = np.sign(p - 0.5) * np.sqrt(k * (1 / z - 1))

    return x


def rnd(k, N, R=None, seed=None): 
    """
    students_t.rnd

    Generate random samples from the Student's t-distribution with k degrees of freedom.

    INPUTS:
        k : int or float or array_like, shape (n,)
            Degrees of freedom (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility

    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples from Student's t-distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses inverse transform method with correlated uniform variates.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """

    _, k, n, _ = _ppp_(0, k)

    _, _, U = correlated_rvs(R, n, N, seed)

    X = np.zeros((n,N))
    for i in range(n):
        X[i, :] = inv(U[i, :], k[i]) 

    if n == 1:
        X = X.flatten()

    return X
