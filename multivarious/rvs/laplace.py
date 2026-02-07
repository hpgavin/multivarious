import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

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
    x = np.asarray(x, dtype=float)
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
    x = np.asarray(x, dtype=float)

    meanX, sdvnX = params

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
    P = np.atleast_1d(np.asarray(P, dtype=float))
    sr2 = np.sqrt(2)
    X = meanX + sdvnX / sr2 * np.log(2 * P)  
    idx = X >= meanX
    X[idx] = meanX - sdvnX / sr2 * np.log(2 - 2 * P[idx]) 
    return X if X.size > 1 else X[0]

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
    
    # Python does not implicitly handle scalars as arrays ... so ...
    # Convert array_like inputs to numpy column vectors (2D arrays) of floats 
    meanX = np.atleast_2d(meanX).reshape(-1, 1).astype(float)
    sdvnX = np.atleast_2d(sdvnX).reshape(-1, 1).astype(float)

    # Determine number of random variables
    n = len(meanX)

    # Validate that all parameter arrays have the same length
    if not (len(meanX) == n and len(sdvnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, sdvnX:{len(sdvnX)}")

    if np.any(sdvnX <= 0) or np.any(np.isinf(sdvnX)):
        raise ValueError(" laplace_rnd: sdvnX must be > 0 and finite")

    sr2 = np.sqrt(2)

    _, _, U = correlated_rvs( R, n, N, seed )

    # Broadcast parameters (meaning, make meanX the same dimesion as X)
    meanX = meanX @ np.ones((1,N))
    sdvnX = sdvnX @ np.ones((1,N))

    X = np.empty((n, N))
    idx = U <= 0.5
    X[idx] = meanX[idx] + sdvnX[idx] / sr2 * np.log(2 * U[idx])
    idx = ~idx
    X[idx] = meanX[idx] - sdvnX[idx] / sr2 * np.log(2 * (1 - U[idx]))

    if n == 1:
        X = X.flatten()

    return X
