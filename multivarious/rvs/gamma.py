import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc

#from multivarious.utl.correlated_rvs import correlated_rvs


def pdf(x, meanX, covnX):
    '''
    gamma.pdf

    Computes the PDF of the Gamma distribution using its mean (m)
    and coefficient of variation (covnX).

    Parameters
    ----------
    x : array_like or float
        Evaluation points
    meanX : float
        Mean of the distribution
    covnX : float
        Coefficient of variation (sdv/mean), must be ≥ 0.5

    Output
    ------
    f : ndarray or float
        PDF values evaluated at each point in x

    Reference
    ----------
    https://en.wikipedia.org/wiki/Gamma_distribution
    '''
    x = np.asarray(x, dtype=float)  # ensure x is a NumPy array

    k = 1.0 / c**2                  # shape parameter (alpha)
    theta = covnX**2 * meanX                # scale parameter (beta)
    f = theta**(-k) * x**(k - 1) * np.exp(-x / theta) / gamma_func(k)  # Gamma PDF formula
    f[x < 0] = 1e-12                # assign small value for negative x (domain correction)
    return f


def cdf(x, params):
    '''
    gamma.cdf

    Computes the CDF of the Gamma distribution in terms of its mean (m)
    and coefficient of variation (covnX).

    Parameters
    ----------
    x : array_like or float
        Evaluation points
    params : array_like [ meanX, covnX ]
        meanX : (float) the mean of X
        covnX : (float) the coefficient of variation of X
        coefficient of variation (sdv/mean)

    Output
    ------
    F : ndarray or float
        CDF values evaluated at each point in x

    Reference
    ----------
    https://en.wikipedia.org/wiki/Gamma_distribution
    '''
    x = np.asarray(x, dtype=float)

    meanX, covnX = params

    if covnX < 0.5:
        print(f'gamma_cdf: covnX = {c:.4f}, lower limit of coefficient of variation is 0.5')

    k = 1.0 / covnX**2              # shape parameter
    theta = covnX**2 * meanX        # scale parameter
    xp = np.copy(x)                 # copy x to avoid modifying input
    xp[x < 0] = 1e-5                # replace negative x with small value for stability
    F = gammainc(k, xp / theta)     # regularized lower incomplete gamma function
    F[x <= 0] = 0.0                 # set CDF = 0 for x ≤ 0
    return F


def inv(P, meanX, covnX):
    '''
    gamma.inv

    Computes the inverse CDF (quantile function) of the Gamma distribution
    defined by mean meanX and coefficient of variation covnX, for given probabilities P.

    Parameters
    ----------
    P : array_like or float
        Non-exceedance probabilities (values between 0 and 1)
    meanX : float
        Mean of the Gamma distribution
    covnX : float
        Coefficient of variation (must be ≥ 0.5)

    Output
    ------
    x : ndarray or float
        Quantile values such that Prob[X ≤ x] = P

    Reference
    ----------
    https://en.wikipedia.org/wiki/Gamma_distribution
    '''
    P = np.asarray(P, dtype=float)  # ensure array type

    if np.any(covnX < 0.5):
        print(f'gamma_inv: covnX = {c:.4f}, lower limit of coefficient of variation is 0.5')

    myeps = 1e-12                   # numerical tolerance
    max_iter = 100                   # max Newton-Raphson iterations
    x_old = np.sqrt(myeps) * np.ones_like(P)  # initialize guesses for x

    for iter in range(max_iter):
        F_x = cdf(x_old, [meanX, covnX])    # evaluate current CDF
        f_x = pdf(x_old,  meanX, covnX )    # evaluate current PDF
        h = (F_x - P) / f_x                 # Newton-Raphson step
        x_new = x_old - h                   # update estimate
        x_new[x_new <= myeps] = x_old[x_new <= myeps] / 10 # prevent values <= 0
        h = x_old - x_new                   # change in x
        if np.max(np.abs(h)) < np.sqrt(myeps):  # convergence check
            break
        x_old = x_new                       # update for next iteration

    return x_new                            # return final quantiles


def rnd(meanX, covnX, N, R=None, seed=None):
    '''
    gamma.rnd

    Generates random samples from the Gamma distribution with mean meanX and
    coefficient of variation c.

    Parameters
    ----------
    meanX : float
        Mean of the distribution
    covnX : float
        Coefficient of variation (sdv/mean)
    N : int
        Number of values of each random variable (columns)
    R : correlation matrix (n,n) --- not implemented 

    Output
    ------
    X : ndarray of shape (n, N)
        Random samples drawn from the Gamma distribution

    Reference
    ----------
    https://en.wikipedia.org/wiki/Gamma_distribution
    '''

    rng = np.random.default_rng(seed)

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    meanX = np.atleast_1d(meanX).astype(float)
    covnX = np.atleast_1d(covnX).astype(float)

    # Determine number of random variables
    n = len(meanX)

    # Validate that all parameter arrays have the same length
    if not len(meanX) == n and len(covnX) == n:
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got m:{len(meanX)}, c:{len(covnX)}")

    if np.any(meanX <= 0):
        raise ValueError("gamma.rnd: all meanX values must be greater than 0") 

    k = 1.0 / covnX**2                  # shape parameter
    theta = covnX**2 * meanX            # scale parameter

    # draw samples using NumPy’s gamma RNG
    X = rng.gamma(shape=k, scale=theta, size=(n, N)) 

    if n == 1:
        X = X.flatten()

    return X
