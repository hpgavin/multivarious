import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

def _ppp_(x, meanX, sdvnX ):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
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

    meanX = np.atleast_1d(meanX).astype(float)
    sdvnX = np.atleast_1d(sdvnX).astype(float)
    n = len(meanX)   
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(sdvnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, meanX:{len(meanX)}")

    # Validate parameter values 
    if np.any(sdvnX <= 0):
        raise ValueError("laplace: all sdvnX values must be greater than zero")

    return x, meanX, sdvnX, n


def pdf(x, meanX, sdvnX):
    '''
    laplace.pdf

    Computes the PDF of the Laplace distribution with mean (location)
    parameter meanX and standard deviation (scale) parameter sdvnX.

    Parameters:
        x : array_like or float
            Evaluation points
        meanX : float
            Mean (location) parameter
        sdvnX : float
            Standard deviation (scale) parameter (must be > 0)

    Output:
        f : ndarray or float
            PDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''

    x, meanX, sdvnX, n = _ppp_(x, meanX, sdvnX)

    sr2 = np.sqrt(2)
    f = (1 / (sr2 * sdvnX)) * np.exp(-sr2 * np.abs(x - meanX) / sdvnX)
    return f


def cdf(x, params):
    '''
    laplace.cdf

    Computes the CDF of the Laplace distribution with parameters meanX and sdvnX.

    Parameters:
        x : array_like or float
            Evaluation points
        params : array_like 
            [meanX, sdvnX] where meanX is the mean (location) parameter and
            sdvnX is the standard deviation (scale) parameter (must be > 0)

    Output:
        F : ndarray or float
            CDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    meanX, sdvnX = params

    x, meanX, sdvnX, n = _ppp_(x, meanX, sdvnX)

    sr2 = np.sqrt(2)

    F = np.zeros_like(x)
    F[x <= meanX] = 0.5 * np.exp(-sr2 * np.abs(x[x <= meanX] - meanX) / sdvnX)
    F[x > meanX] = 1 - 0.5 * np.exp(-sr2 * np.abs(x[x > meanX] - meanX) / sdvnX)
    return F


def inv(P, meanX, sdvnX):
    '''
    laplace.inv

    Computes the inverse CDF (quantile function) of the Laplace distribution
    with mean (location) meanX and standard deviation (scale) sdvnX.

    Parameters:
        P : array_like or float
            Non-exceedance probabilities (must be in [0, 1])
        meanX : float
            Mean (location) parameter
        sdvnX : float
            Standard deviation (scale) parameter (must be > 0)

    Output:
        X : ndarray or float
            Quantile values corresponding to probabilities P

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    _, meanX, sdvnX, n = _ppp_(0, meanX, sdvnX)

    sr2 = np.sqrt(2)

    x = np.zeros(P.shape)
  
    idx = P <= 0.5 
    x[idx] = meanX + sdvnX / sr2 * np.log(2 * P[idx]) 
    idx = ~idx
    x[idx] = meanX - sdvnX / sr2 * np.log(2 * (1 - P[idx]))

    return x if x.size > 1 else x[0]

def rnd(meanX, sdvnX, N, R=None, seed=None):
    '''
    laplace.rnd

    Generates random samples from the Laplace distribution using the
    inverse transform sampling method.

    Parameters:
        meanX : float
            Mean (location) parameter
        sdvnX : float
            Standard deviation (scale) parameter (must be > 0)
        r : int
            Number of rows in the output
        c : int
            Number of columns in the output
        z : ndarray, optional
            Matrix of uniform(0,1) random numbers to use directly instead of
            generating new random samples

    Output:
        x : ndarray of shape (r, c)
            Random samples drawn from the Laplace distribution

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    
    _, meanX, sdvnX, n = _ppp_(0, meanX, sdvnX)

    sr2 = np.sqrt(2)

    _, _, U = correlated_rvs( R, n, N, seed )

    X = np.empty((n, N))
    for i in range(n):
        X[i,:] = inv(U[i,:], meanX[i], sdvnX[i])

    if n == 1:
        X = X.flatten()

    return X
