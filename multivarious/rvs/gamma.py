import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc

from multivarious.utl.correlated_rvs import correlated_rvs


def pdf(x, m, c):
    '''
    gamma.pdf

    Computes the PDF of the Gamma distribution using its mean (m)
    and coefficient of variation (c).

    Parameters
    ----------
    x : array_like or float
        Evaluation points
    m : float
        Mean of the distribution
    c : float
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
    theta = c**2 * m                # scale parameter (beta)
    f = theta**(-k) * x**(k - 1) * np.exp(-x / theta) / gamma_func(k)  # Gamma PDF formula
    f[x < 0] = 1e-12                # assign small value for negative x (domain correction)
    return f


def cdf(x, params):
    '''
    gamma.cdf

    Computes the CDF of the Gamma distribution in terms of its mean (m)
    and coefficient of variation (c).

    Parameters
    ----------
    x : array_like or float
        Evaluation points
    params : sequence of floats
        [m, c] where m is the mean of the distribution and c is the
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
    m, c = params  # unpack mean and coefficient of variation

    if c < 0.5:
        print(f'gamma_cdf: c = {c:.4f}, lower limit of coefficient of variation is 0.5')

    k = 1.0 / c**2                  # shape parameter
    theta = c**2 * m                # scale parameter
    xp = np.copy(x)                 # copy x to avoid modifying input
    xp[x < 0] = 1e-5                # replace negative x with small value for stability
    F = gammainc(k, xp / theta)     # regularized lower incomplete gamma function
    F[x <= 0] = 0.0                 # set CDF = 0 for x ≤ 0
    return F


def inv(P, m, c):
    '''
    gamma.inv

    Computes the inverse CDF (quantile function) of the Gamma distribution
    defined by mean m and coefficient of variation c, for given probabilities P.

    Parameters
    ----------
    P : array_like or float
        Non-exceedance probabilities (values between 0 and 1)
    m : float
        Mean of the Gamma distribution
    c : float
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

    if np.any(c < 0.5):
        print(f'gamma_inv: c = {c:.4f}, lower limit of coefficient of variation is 0.5')

    myeps = 1e-12                   # numerical tolerance
    MaxIter = 100                   # max Newton-Raphson iterations
    x_old = np.sqrt(myeps) * np.ones_like(P)  # initialize guesses for x

    for _ in range(MaxIter):
        F_x = cdf(x_old, [m, c])    # evaluate current CDF
        f_x = pdf(x_old, m, c)      # evaluate current PDF
        h = (F_x - P) / f_x         # Newton-Raphson step
        x_new = x_old - h           # update estimate
        x_new[x_new <= myeps] = x_old[x_new <= myeps] / 10  # prevent nonpositive values
        h = x_old - x_new           # change in x
        if np.max(np.abs(h)) < np.sqrt(myeps):  # convergence check
            break
        x_old = x_new               # update for next iteration

    return x_new                    # return final quantiles


def rnd(m, c, N, R, R):
    '''
    gamma.rnd

    Generates random samples from the Gamma distribution with mean m and
    coefficient of variation c.

    Parameters
    ----------
    m : float
        Mean of the distribution
    c : float
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

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    m = np.atleast_1d(m).astype(float)
    c = np.atleast_1d(c).astype(float)

    # Determine number of random variables
    n = len(m)

    # Validate that all parameter arrays have the same length
    if not (len(m) == n and len(c) == n:
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got m:{len(m)}, c:{len(c)}")

    if np.any(m <= 0):
        raise ValueError("gamma.rnd: all m values must be greater than 0") 

    k = 1.0 / c**2                  # shape parameter
    theta = c**2 * m                # scale parameter

    # draw samples using NumPy’s gamma RNG
    X = np.random.gamma(shape=k, scale=theta, size=(n, N)) 

    if n == 1:
        X = X.flatten()

    return X
