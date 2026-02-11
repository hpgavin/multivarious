import numpy as np
from scipy.special import erf as scipy_erf
from scipy.special import erfinv as scipy_erfinv

from multivarious.utl import correlated_rvs


def _ppp_(x, meanX, sdvnX):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        meanX : float
            Minimum of the distribution
        sdvnX : float
            Maximum of the distribution (must be > a)
     ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    meanX = np.atleast_2d(meanX).reshape(-1,1).astype(float)
    sdvnX = np.atleast_2d(sdvnX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(sdvnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, sdvnX:{len(sdvnX)}") 

    # Validate parameter values 
    if np.any(sdvnX <= 0):
        raise ValueError("normal: sdvnX must be > 0")

    return x, meanX, sdvnX, n


def pdf(x, meanX=0.0, sdvnX=1.0):
    '''
    normal.pdf

    Computes the PDF of the normal distribution N(meanX, sdvnX²).

    Parameters:
        x : array_like or float
            Evaluation points
        meanX : float
            Mean of the distribution
        sdvnX : float
            Standard deviation of the distribution (meanXst be > 0)

    Output:
        f : ndarray or float
            PDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''
 
    x, meanX, covnX = _ppp_(x, meanX, covnX)

    z = (x - meanX) / sdvnX

    f = 1.0 / np.sqrt(2 * np.pi*sdvnX**2) * np.exp(-(z**2.0) / 2.0)

    return f 


def cdf(x, params ):
    '''
    normal.cdf

    Computes the CDF of the normal distribution N(meanX, sdvnX²).

    Parameters:
        x : array_like or float
            Evaluation points
        params : array_like [ meanX , sdvnX ]
        meanX : float
            Mean of the distribution
        sdvnX : float
            Standard deviation of the distribution (meanXst be > 0)

    Output:
        F : ndarray or float
            CDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''

    meanX, sdvnX = params 

    _, meanX, covnX = _ppp_(0, meanX, covnX)

    z = (x - meanX) / sdvnX

    F = (1.0 + scipy_erf(z / np.sqrt(2.0))) / 2.0

    return F


def inv(p, meanX=0.0, sdvnX=1.0):
    '''
    normal.inv

    Computes the inverse CDF (quantile function) of the normal distribution N(meanX, sdvnX²).

    Parameters:
        p : array_like or float
            Probability values (meanXst be in [0, 1])
        meanX : float
            Mean of the distribution
        sdvnX : float
            Standard deviation of the distribution (sdvnX > 0)

    Output:
        x : ndarray or float
            Quantile values corresponding to probabilities p

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''

    x, meanX, covnX = _ppp_(x, meanX, covnX)

    # Clip probabilities to avoid erfinv(±1) = ±∞
    my_eps = 1e-12     # small, not zero
    P = np.clip(P, my_eps, 1.0 - my_eps)  # restrict P to (my_eps, 1-my_eps)
    
    # Compute lognormal quantile using inverse CDF formula
    z = np.sqrt(2) * scipy_erfinv(2 * P - 1) 
    x = meanX + stdvX * z

    return x


def rnd(meanX=0.0, sdvnX=1.0, N=1, R=None, seed=None):
    '''
    normal.rnd
    Generates correlated random samples from the normal distribution N(meanX, sdvnX²).

    Parameters:
        meanX : float (n,)
            Mean of the distribution
        sdvnX : float (n,)
            Standard deviation of the distribution (meanXst be > 0)
        N : int
            number of observations of each of the n random variables 
        R : float (n,n) - optional
            correlation matrix
        seed : int ( seed >= 0 )
            seed for numpy.random.default_rng

    Output:
        X : ndarray
            Array of normal random samples with shape `size`

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''

    _, meanX, sdvnX, n = _ppp_(0, meanX, sdvnX) # Correlated standard normal variables (n,N)

    _, Y, _ = correlated_rvs( R, n, N, seed )

    X = meanX + sdvnX*Y

    if n == 1:
        X = X.flatten()

    return X
