import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

def pdf(X, muX):
    '''
    rayleigh.pdf

    Computes the PDF of the Rayleigh distribution using the mean parameter muX.

    Input:
        X : array_like
            Evaluation points
        muX : float
            Mean of the Rayleigh distribution (must be > 0)

    Output:
        f : ndarray
            PDF values at each point in X

    Reference:
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    X = np.asarray(X, dtype=float)

    # Convert mean to mode: modeX = muX * sqrt(2 / pi)
    modeX = muX * np.sqrt(2 / np.pi)

    # Replace non-positive values to prevent invalid evaluation
    X = np.where(X <= 0, 0.01, X)

    # Apply the Rayleigh PDF formula
    f = (X / modeX**2) * np.exp(-0.5 * (X / modeX)**2)

    return f


def cdf(X, muX):
    '''
    rayleigh.cdf
    Computes the CDF of the Rayleigh distribution using the mean parameter muX.

    Input:
        X : array_like
            Evaluation points
        muX : float
            Mean of the Rayleigh distribution (must be > 0)

    Output:
        F : ndarray
            CDF values at each point in X

    Reference:
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    X = np.asarray(X, dtype=float).copy()
    
    # Replace X <= 0 with small positive number (to match MATLAB behavior)
    X[X <= 0] = 0.01

    # Convert mean muX to modeX using Rayleigh identity
    modeX = muX * np.sqrt(2 / np.pi)

    # Apply the Rayleigh CDF formula
    F = 1.0 - np.exp(-0.5 * (X / modeX)**2)

    return F


def inv(P, muX):
    '''
    rayleigh.inv

    Computes the inverse CDF (quantile function) of the Rayleigh distribution
    using the mean parameter muX.

    INPUT:
        P : array_like
            Non-exceedance probabilities (0 ≤ P ≤ 1)
        muX : float
            Mean of the Rayleigh distribution (must be > 0)

    OUTPUT:
        x : ndarray
            Quantile values corresponding to probabilities P

    Reference:
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''
    P = np.asarray(P, dtype=float).copy()

    # Clamp values: ensure P stays in [0, 1] just like MATLAB does
    P[P <= 0] = 0.0
    P[P >= 1] = 1.0

    # Convert mean to mode using mu = mode * sqrt(pi / 2)
    modeX = muX * np.sqrt(2 / np.pi)

    # Compute the inverse CDF formula
    x = modeX * np.sqrt(-2.0 * np.log(1 - P))

    return x


def rnd(muX, N, R=None)
    '''
    rayleigh.rnd

    Generates random samples from the Rayleigh distribution using the mean
    parameter muX. Samples can be generated either by passing a matrix of
    uniform random numbers or by specifying the output dimensions.

    Input:
        muX : float (n,1)
        N : int, optional
            Number of observations of n rayleigh random variables
        R : float (n,n) , optional
            correlation matrix

    Output:
        X : ndarray
            Random samples drawn from the Rayleigh distribution

    Reference
    ----------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    muX = np.atleast_1d(muX).astype(float)

    n = len(muX) # number of rows

    if np.any(muX <= 0) or np.any(np.isinf(muX)):
        raise ValueError(" rayleigh.rnd(muX,N): muX must be greater than zero")
    if N == None or N < 1:
        raise ValueError(" rayleigh.rnd(muX,N): N must be greater than zero")

    # Convert mean to mode
    modeX = muX * np.sqrt(2 / np.pi)

    # Broadcast modeX if needed
    if np.isscalar(modeX):
        modeX = modeX * np.ones((r_rows, c_cols))
        
    _, _, U = correlated_rvs(R,n,N)

    # Inverse transform sampling
    X = modeX * np.sqrt(-2.0 * np.log(U))

    if n == 1:
        X = X.flatten()

    return X
