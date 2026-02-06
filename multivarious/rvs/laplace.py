import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

def pdf(x, meanX, stdvX):
    '''
    laplace.pdf

    Computes the PDF of the Laplace distribution with mean (location)
    parameter meanX and standard deviation (scale) parameter stdvX.

    Parameters:
        x : array_like or float
            Evaluation points
        meanX : float
            Mean (location) parameter
        stdvX : float
            Standard deviation (scale) parameter (must be > 0)

    Output:
        f : ndarray or float
            PDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    x = np.asarray(x, dtype=float)
    sr2 = np.sqrt(2)
    f = (1 / (sr2 * stdvX)) * np.exp(-sr2 * np.abs(x - meanX) / stdvX)
    return f


def cdf(x, params):
    '''
    laplace.cdf

    Computes the CDF of the Laplace distribution with parameters meanX and stdvX.

    Parameters:
        x : array_like or float
            Evaluation points
        params : array_like 
            [meanX, stdvX] where meanX is the mean (location) parameter and
            stdvX is the standard deviation (scale) parameter (must be > 0)

    Output:
        F : ndarray or float
            CDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    x = np.asarray(x, dtype=float)

    meanX, stdvX = params

    sr2 = np.sqrt(2)

    F = np.zeros_like(x)
    F[x <= meanX] = 0.5 * np.exp(-sr2 * np.abs(x[x <= meanX] - meanX) / stdvX)
    F[x > meanX] = 1 - 0.5 * np.exp(-sr2 * np.abs(x[x > meanX] - meanX) / stdvX)
    return F


def inv(P, meanX, stdvX):
    '''
    laplace.inv

    Computes the inverse CDF (quantile function) of the Laplace distribution
    with mean (location) meanX and standard deviation (scale) stdvX.

    Parameters:
        P : array_like or float
            Non-exceedance probabilities (must be in [0, 1])
        meanX : float
            Mean (location) parameter
        stdvX : float
            Standard deviation (scale) parameter (must be > 0)

    Output:
        X : ndarray or float
            Quantile values corresponding to probabilities P

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    P = np.atleast_1d(np.asarray(P, dtype=float))
    sr2 = np.sqrt(2)
    X = meanX + stdvX / sr2 * np.log(2 * P)  
    idx = X >= meanX
    X[idx] = meanX - stdvX / sr2 * np.log(2 - 2 * P[idx]) 
    return X if X.size > 1 else X[0]

def rnd(meanX, stdvX, N, R=None, seed=None):
    '''
    laplace.rnd

    Generates random samples from the Laplace distribution using the
    inverse transform sampling method.

    Parameters:
        meanX : float
            Mean (location) parameter
        stdvX : float
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
    
    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    meanX = np.atleast_1d(meanX).astype(float)
    stdvX = np.atleast_1d(stdvX).astype(float)

    # Determine number of random variables
    n = len(meanX)

    # Validate that all parameter arrays have the same length
    if not (len(meanX) == n and len(stdvX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, stdvX:{len(stdvX)}")

    if np.any(stdvX <= 0) or np.any(np.isinf(stdvX)):
        raise ValueError(" laplace_rnd: stdvX must be > 0 and finite")

    sr2 = np.sqrt(2)

    _, _, U = correlated_rvs( R, n, N, seed )

    # Broadcast parameters
#   meanX = np.full((n, N), meanX) if np.isscalar(meanX) else np.asarray(meanX)
#   stdvX = np.full((n, N), stdvX) if np.isscalar(stdvX) else np.asarray(stdvX)

    X = np.empty((n, N))
    in_mask = U <= 0.5
    X[in_mask] = meanX[in_mask] + stdvX[in_mask] / sr2 * np.log(2 * U[in_mask])
    ip_mask = ~in_mask

    X = np.zeros(n,N)
    for i in range(n):
        X[i,ip_mask] = meanX[i] - stdvX[i] / sr2 * np.log(2 * (1 - U[i,ip_mask]))
        
    if n == 1:
        X = X.flatten()

    return X
