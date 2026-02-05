import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

def pdf(x, muX, sigmaX):
    '''
    laplace.pdf

    Computes the PDF of the Laplace distribution with mean (location)
    parameter muX and standard deviation (scale) parameter sigmaX.

    Parameters:
        x : array_like or float
            Evaluation points
        muX : float
            Mean (location) parameter
        sigmaX : float
            Standard deviation (scale) parameter (must be > 0)

    Output:
        f : ndarray or float
            PDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    x = np.asarray(x, dtype=float)
    sr2 = np.sqrt(2)
    f = (1 / (sr2 * sigmaX)) * np.exp(-sr2 * np.abs(x - muX) / sigmaX)
    return f


def cdf(x, params):
    '''
    laplace.cdf

    Computes the CDF of the Laplace distribution with parameters muX and sigmaX.

    Parameters:
        x : array_like or float
            Evaluation points
        params : sequence of floats
            [muX, sigmaX] where muX is the mean (location) parameter and
            sigmaX is the standard deviation (scale) parameter (must be > 0)

    Output:
        F : ndarray or float
            CDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    x = np.asarray(x, dtype=float)
    muX = params[0]
    sigmaX = params[1]
    sr2 = np.sqrt(2)

    F = np.zeros_like(x)
    F[x <= muX] = 0.5 * np.exp(-sr2 * np.abs(x[x <= muX] - muX) / sigmaX)
    F[x > muX] = 1 - 0.5 * np.exp(-sr2 * np.abs(x[x > muX] - muX) / sigmaX)
    return F


def inv(P, muX, sigmaX):
    '''
    laplace.inv

    Computes the inverse CDF (quantile function) of the Laplace distribution
    with mean (location) muX and standard deviation (scale) sigmaX.

    Parameters:
        P : array_like or float
            Non-exceedance probabilities (must be in [0, 1])
        muX : float
            Mean (location) parameter
        sigmaX : float
            Standard deviation (scale) parameter (must be > 0)

    Output:
        X : ndarray or float
            Quantile values corresponding to probabilities P

    Reference:
    https://en.wikipedia.org/wiki/Laplace_distribution
    '''
    P = np.atleast_1d(np.asarray(P, dtype=float))
    sr2 = np.sqrt(2)
    X = muX + sigmaX / sr2 * np.log(2 * P)  
    idx = X >= muX
    X[idx] = muX - sigmaX / sr2 * np.log(2 - 2 * P[idx]) 
    return X if X.size > 1 else X[0]

def rnd(mX, sX, N, R=None):
    '''
    laplace.rnd

    Generates random samples from the Laplace distribution using the
    inverse transform sampling method.

    Parameters:
        mX : float
            Mean (location) parameter
        sX : float
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
    mX = np.atleast_1d(mX).astype(float)
    sX = np.atleast_1d(sX).astype(float)

    # Determine number of random variables
    n = len(mX)

    # Validate that all parameter arrays have the same length
    if not (len(mX) == n and len(sX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got mX:{len(mX)}, sX:{len(sX)}")

    if np.any(sX <= 0) or np.any(np.isinf(sX)):
        raise ValueError(" laplace_rnd: sX must be > 0 and finite")

    sr2 = np.sqrt(2)

    _, _, U = correlated_rvs(R,n,N)

    # Broadcast parameters
    mX = np.full((n, N), mX) if np.isscalar(mX) else np.asarray(mX)
    sX = np.full((n, N), sX) if np.isscalar(sX) else np.asarray(sX)

    X = np.empty((n, N))
    in_mask = u <= 0.5
    X[in_mask] = mX[in_mask] + sX[in_mask] / sr2 * np.log(2 * U[in_mask])
    ip_mask = ~in_mask
    X[ip_mask] = mX[ip_mask] - sX[ip_mask] / sr2 * np.log(2 * (1 - U[ip_mask]))
        
    if n == 1:
        X = X.flatten()

    return X
