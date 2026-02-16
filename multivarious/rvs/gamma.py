## gamma distribution
# github.com/hpgavin/multivarious ... rvs/gamma

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, meanX, covnX):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like
            Mean(s) of the distribution (must be > 0)
        covnX : float or array_like
            Coefficient(s) of variation (must be ≥ 0.5)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        meanX : ndarray
            Means as column array
        covnX : ndarray
            Coefficients of variation as column array
        k : ndarray
            Shape parameters (alpha)
        theta : ndarray
            Scale parameters (beta)
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    meanX = np.atleast_1d(meanX).reshape(-1,1).astype(float)
    covnX = np.atleast_1d(covnX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(covnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, covnX:{len(covnX)}")

    # Validate parameter values 
    if np.any(covnX < 0.5):
        print(f'gamma: lower limit of coefficient of variation is 0.5')
        print(f'gamma: covnX = {covnX.flatten()}')
        covnX[covnX < 0.5] = 0.5

    k = 1.0 / covnX**2              # shape parameter (alpha)
    theta = covnX**2 * meanX        # scale parameter (beta)

    return x, meanX, covnX, k, theta, n


def pdf(x, meanX, covnX):
    """
    gamma.pdf

    Computes the PDF of the Gamma distribution using its mean and
    coefficient of variation.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation (stddev/mean), must be ≥ 0.5

    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values evaluated at each point in x for each of n random variables

    Notes
    -----
    f(x) = (1/Γ(k)θ^k) x^(k-1) exp(-x/θ)
    where k = 1/covnX² and θ = covnX²·meanX

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    x, meanX, covnX, k, theta, n = _ppp_(x, meanX, covnX)

    f = theta**(-k) * x**(k - 1) * np.exp(-x / theta) / gamma_func(k)

    f = np.where(x <= 0, 1e-12, f)  # replace negative x with small value for stability

    if n == 1:
        f = f.flatten()

    return f


def cdf(x, params):
    """
    gamma.cdf

    Computes the CDF of the Gamma distribution in terms of its mean and
    coefficient of variation.

    INPUTS:
        x : array_like
            Evaluation points
        params : array_like [meanX, covnX]
            meanX : float or array_like
                Mean(s) of X
            covnX : float or array_like
                Coefficient(s) of variation (stddev/mean)

    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values evaluated at each point in x for each of n random variables

    Notes
    -----
    F(x) = γ(k, x/θ) / Γ(k) where γ is the lower incomplete gamma function

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    meanX, covnX = params

    x, meanX, covnX, k, theta, n = _ppp_(x, meanX, covnX)

    xp = np.copy(x)                   # copy x to avoid modifying input
    xp = np.where(x <= 0, 1e-12, xp)  # replace negative x with small value for stability
    F = gammainc(k, xp / theta)       # regularized lower incomplete gamma function
    F = np.where(x <= 0, 0.0, F)      # replace negative x with small value for stability
    
    if n == 1:
        F = F.flatten()

    return F


def inv(P, meanX, covnX):
    """
    gamma.inv

    Computes the inverse CDF (quantile function) of the Gamma distribution
    defined by mean meanX and coefficient of variation covnX, for given probabilities P.

    INPUTS:
        P : array_like
            Non-exceedance probabilities (values between 0 and 1)
        meanX : float or array_like, shape (n,)
            Mean(s) of the Gamma distribution
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation (must be ≥ 0.5)

    OUTPUTS:
        x : ndarray
            Quantile values such that Prob[X ≤ x] = P

    Notes
    -----
    Uses Newton-Raphson iteration to solve F(x) = P

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    _, meanX, covnX, _, _, _ = _ppp_(0, meanX, covnX)

    P = np.asarray(P, dtype=float)  # ensure array type

    my_eps = 1e-12                   # numerical tolerance
    max_iter = 100                   # max Newton-Raphson iterations
    x_old = np.sqrt(my_eps) * np.ones_like(P)  # initialize guesses for x

    for iter in range(max_iter):
        F_x = cdf(x_old, [meanX, covnX])    # evaluate current CDF
        f_x = pdf(x_old, meanX, covnX)      # evaluate current PDF
        h = (F_x - P) / f_x                 # Newton-Raphson step
        x_new = x_old - h                   # update estimate
        idx = np.where(x_new <= my_eps)
        if np.any(idx):
            x_new[idx] = x_old[idx] / 10.0  # prevent values <= 0
        h = x_old - x_new                   # change in x
        if np.max(np.abs(h)) < np.sqrt(my_eps):  # convergence check
            break
        x_old = x_new                       # update for next iteration

    x = x_new                               # return final quantiles

    if n == 1:
        x = x.flatten() 

    return x  


def rnd(meanX, covnX, N, R=None, seed=None):
    """
    gamma.rnd

    Generates random samples from the Gamma distribution with mean meanX and
    coefficient of variation covnX.

    INPUTS:
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation (stddev/mean)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility

    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples drawn from the Gamma distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses inverse transform method with correlated uniform variates.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """

    _, meanX, covnX, _, _, n = _ppp_(0, meanX, covnX)

    # Correlated uniform random variables 
    _, _, U = correlated_rvs(R, n, N, seed)

    # Transform each variable to its gamma distribution 
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i,:], meanX[i], covnX[i])
    
    if n == 1:
        X = X.flatten()

    return X
