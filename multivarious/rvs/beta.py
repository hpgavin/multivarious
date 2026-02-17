# beta distribution
# github.com/hpgavin/multivarious ... rvs/beta

import numpy as np
from scipy.special import beta as beta_func
from scipy.special import betainc
from scipy.special import betaincinv

from multivarious.utl.correlated_rvs import correlated_rvs

def _ppp_(x, a, b, q, p ):
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
    a = np.atleast_1d(a).astype(float) 
    b = np.atleast_1d(b).astype(float)
    q = np.atleast_1d(q).astype(float)
    p = np.atleast_1d(p).astype(float)
    n = len(a)   
    N = len(x)

    # Validate parameter dimensions 
    if not (len(b) == n and len(q) == n and len(p) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}, q:{len(q)}, p:{len(p)}")
    
    # Validate parameter values 
    if np.any(b <= a):
        raise ValueError("beta.rnd: all b values must be greater than corresponding a values")
    if np.any(q <= 0):
        raise ValueError("beta.rnd: q must be positive")
    if np.any(p <= 0):
        raise ValueError("beta.rnd: p must be positive")

    return x, a, b, q, p, n, N


def pdf(x, a, b, q, p):
    '''
    beta.pdf

    Computes the Probability Density Function (PDF) of the Beta distribution
    with lower bound a, upper bound b, and shape parameters q and p.

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
 
    OUTPUT:
        f : ndarray
            PDF evaluated at x
    '''

    x, a, b, q, p, n, N = _ppp_(x, a, b, q, p)

    # Initialize PDF output as zeros
    f = np.zeros((n,N))

    # Only compute for values within [a, b]
    
    # Beta PDF formula (with change of variable from [0,1] to [a,b])

    for i in range(n): 
        mask = (x >= a[i]) & (x <= b[i])
        numerator = (x[mask] - a[i])**(q[i] - 1) * (b[i] - x[mask])**(p[i]-1)
        denominator = beta_func(q[i], p[i]) * (b[i] - a[i])**(q[i] + p[i] - 1)
        f[i,mask] = numerator / denominator

    if n == 1 and x.shape[0] == 1:
        f = f.flatten()

    return f


def cdf(x, params ):
    '''
    beta.cdf

    Computes the Cumulative Distribution Function (CDF) of the beta distribution
    with lower bound a, upper bound b, and shape parameters q and p.

    INPUTS:
        x = array_like 
            Evaluation points
        params = array_like [ a , b , q, p  ]
        a = params[0] float 
            Minimum of the distribution
        b = params[1] float
            Maximum of the distribution (must be > a)
        q = params[2] float
            First shape parameter
        p = params[3] float 
            Second shape parameter

    OUTPUT:
        F = ndarray (CDF evaluated at x)
        
    Formula: 
        F(x) = I_{(x - a) / (b - a)} (q, p)
    where I is the regularized incomplete beta function.
    '''

    a, b, q, p = params

    x, a, b, q, p, n, N = _ppp_(x, a, b, q, p)

    # Compute z = (x - a) / (b - a), clipped to [0, 1]
    z = (x - a) / (b - a)
    z = np.clip(z, 0, 1)

    # Evaluate the regularized incomplete beta function
    F = np.zeros((n,N))
    for i in range(n):
       F[i,:] = betainc(q[i], p[i], z[i,:])

    if n == 1 and x.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, a, b, q, p):
    '''
    beta.inv

    Compute the inverse CDF (quantile function) of the beta distribution.

    INPUTS:
        F : array_like
            Non-exceedance probability values (between 0 and 1)
        a : float
            Lower bound of the distribution
        b : float
            Upper bound of the distribution (must be > a)
        q : float
            First shape parameter
        p : float
            Second shape parameter

    OUTPUT:
    x : ndarray
        Quantile values corresponding to input probabilities F
    '''

    _, a, b, q, p, n, _ = _ppp_(0, a, b, q, p)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    x = np.zeros((n,N)) 

    for i in range(n): 
        # Compute inverse of regularized incomplete beta function
        z = betaincinv(q[i], p[i], F[i,:])

        # Rescale from [0, 1] to [a, b]
        x[i,:] = a[i] + z * (b[i] - a[i])

    if n == 1 and F.shape[0] == 1:
        x = x.flatten()

    return x


def rnd(a, b, q, p, N, R=None, seed=None):
    '''
    beta.rnd
    Generate N observations of n correlated (or uncorrelated) beta random var's

    INPUT:
        a : float or array_like (n,)
            Lower bound(s) of the distribution. shape (n,) for n random variables
        b : float or array_like (n,)
            Upper bound(s) of the distribution. shape (n,) for n random variables
        q : float or array_like
            First shape parameter(s). shape (n,) for n random variables
        p : float or array_like
            Second shape parameter(s). shape (n,) for n random variables
        N : int
            Number of observations (samples) to generate.
        R : ndarray, optional
            n×n correlation matrix of standard normal deviates.
            If None, defaults to identity matrix (uncorrelated samples).

    OUTPUT:
        X : ndarray
            Shape (n, N) array of correlated beta random samples.
            Each row corresponds to one random variable.
            Each column corresponds to one observation.

    Method:
        1. Perform eigenvalue decomposition of correlation matrix R
        2. Generate uncorrelated standard normal samples
        3. Apply correlation structure: Y = eVec @ sqrt(eVal) @ Z
        4. Transform to uniform via standard normal CDF
        5. Transform to beta via inverse CDF for each variable

    Example (Usage):
        # Single variable, uncorrelated samples
            x = rnd(0, 1, 2, 5, N=1000)
        
        # Multiple correlated variables
            a = np.array([0, 1])
            b = np.array([1, 3])
            q = np.array([2, 3])
            p = np.array([5, 4])
            R = np.array([[1.0, 0.7], [0.7, 1.0]])
            x = rnd(a, b, q, p, N=1000, R=R)
    '''

    _, a, b, q, p, n, _ = _ppp_(0, a, b, q, p)
   
    _, _, U = correlated_rvs(R,n,N,seed)

    X = inv( U, a, b, q, p )
       
    return X
