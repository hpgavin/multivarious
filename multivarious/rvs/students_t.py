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

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f

#   # Compute the PDF using the known closed-form
#   f = (np.exp(-(k + 1) * np.log(1 + (t ** 2) / k) / 2)) / (np.sqrt(k) * beta_func(k / 2, 0.5))

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

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F

def inv(F, k):
    """
    students_t.inv

    Computes the inverse CDF (quantile function) of the Student's t-distribution
    with k degrees of freedom, using the inverse incomplete beta function.

    INPUTS:
        F : array_like
            Probability values (must be in [0, 1])
        k : int or float or array_like, shape (n,)
            Degrees of freedom (must be > 0)

    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities F

    Notes
    -----
    Uses relationship with incomplete beta function.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    
    _, k, n, _ = _ppp_(0, k)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    print('F shape : ', F.shape )

    # Compute the inverse CDF using the relationship with the incomplete beta function
    # betaincinv(a, b, y) finds x such that betainc(a, b, x) = y
    z = betaincinv(k / 2.0, 0.5, 2 * np.minimum(F, 1 - F))
    
    # Convert from beta quantile to t quantile
    x = np.sign(F - 0.5) * np.sqrt(k * (1 / z - 1))

    print('x shape : ', x.shape )
    print('n : ', n)

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

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

    X = inv(U, k) 

    return X
