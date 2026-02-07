import numpy as np
from scipy.special import erf as scipy_erf
from scipy.special import erfinv as scipy_erfinv


from multivarious.utl.correlated_rvs import correlated_rvs


def pdf(x, mednX, covnX):
    '''
    lognormal.pdf
 
    Computes the Probability Density Function (PDF) of the lognormal distribution.
 
    INPUTS:
      x     = (array_like) 
              Evaluation points. Must be > 0 since lognormal is defined for x > 0.
      mednX  = float
              Median of the lognormal distribution (X = exp(Y), where Y is normal)
      covnX  = float
              Coefficient of variation of X: (covnX = std(X) / mean(X))
 
    OUTPUT:
      f     = ndarray
              values of the PDF at each point in x
 
    FORMULAS:
      Variance log(x): V_lnX = log(1 + covnX²), then:

    If X ~ Lognormal(mednX, covnX), then log(X) ~ Normal with:
        mean = log(mednX)
        variance = log(1 + covnX²)
        
    Reference: https://en.wikipedia.org/wiki/Log-normal_distribution
    '''
    x = np.asarray(x, dtype=float)

    # Store size (as in MATLAB)
    # r_x, c_x = x.shape if x.ndim == 2 else (x.size, 1) # Not used!

    # Replace invalid x <= 0 with 0.01 to avoid log(0)
    # This matches MATLAB behavior: X(idx) = 0.01
    x = np.where(x <= 0, 0.01, x)
    
    # Compute variance of log(X)
    VlnX = np.log(1 + covnX**2)
    
    # Compute using Lognormal PDF formula
    f = (1 / (x * np.sqrt(2 * np.pi * VlnX))) * \
        np.exp(-0.5 * (np.log(x / mednX))**2 / VlnX)
    
    return f


def cdf(x, params, return_ll=False):
    ''' 
    lognormal.cdf
 
    Computes the CDF of a lognormal distribution.
 
    Parameters:
         x     = array_like of Evaluation points
        params = array_like [ mednX, covnX ]
        mednX   = float of Median of X
        covnX   = float of Coefficient of variation of X
     return_ll = bool, optional
                 If True, also return log-likelihood
    Output: 
         F     = ndarray
                 CDF values at each point in x
         ll    = float, optional
                 Log-likelihood (only if return_ll=True)
    Notes:
        Lognormal CDF: F(x) = (1 + erf((log(x) - log(mednX)) / sqrt(2V))) / 2
        where V = log(1 + covnX²)
    ''' 
    x = np.asarray(x, dtype=float)

    mednX, covnX = params
    
    # Check parameter validity
    if mednX <= 0:
        raise ValueError(f"lognormal_cdf: mednX = {mednX}, must be > 0")
    if covnX <= 0:
        raise ValueError(f"lognormal_cdf: covnX = {covnX}, must be > 0")
    
    # Replace invalid x <= 0 with 0.01 to avoid log(0)
    x = np.where(x <= 0, 0.01, x)
    
    # Compute variance of log(X)
    VlnX = np.log(1 + covnX**2)
    
    # Lognormal CDF formula
    F = 0.5 * (1 + scipy_erf((np.log(x) - np.log(mednX)) / np.sqrt(2 * VlnX)))
    
    if return_ll:
        ll = np.sum(np.log(pdf(x, mednX, covnX)))
        return F, ll
    
    return F


def inv(P, mednX, covnX):
    '''
    lognormal.inv
 
    Computes the inverse CDF (quantile function) for lognormal distribution.
 
    INPUTS:
      p     = array_like of Probability values (scalar or array), must be in (0,1)
      mednX  = float (Median of the lognormal distribution)
      covnX  = float (Coefficient of variation)
 
    OUTPUT:
      x     = ndarray
              Quantile values such that P(X <= x) = P
   
    FORMULA:
        x = exp( log(mednX) + sqrt(2V_lnX) * erfinv(2p - 1) )
            where V_lnX = log(1 + covnX²)
    NOTES:
        Instead of calculating P(X <= x) = p, this function finds x such that
        P(X <= x) = p. In other words, what number x has a cumulative probability p?
    '''
    P = np.asarray(P, dtype=float)
    
    # Clip probabilities to avoid erfinv(±1) = ±∞
    eps = np.finfo(float).eps       # smallest positive float
    P = np.clip(P, eps, 1.0 - eps)  # restrict P to (0, 1)

    # Compute lognormal quantile using inverse CDF formula
    VlnX = np.log(1 + covnX**2)  # Variance of log(X)
    x = np.exp(np.log(mednX) + np.sqrt(2 * VlnX) * scipy_erfinv(2 * P - 1)) 

    return x


def rnd(mednX, covnX, N, R=None, seed=None):
    '''
    lognormal.rnd
 
    Generate N observations of correlated (or uncorrelated) lognormal random variables.
 
    Parameters:
        mednX : float or array_like
               Median(s) of the lognormal distribution. If array, shape (n,) for n variables.
        covnX : float or array_like
               Coefficient(s) of variation. If array, shape (n,) for n variables.
          N  : int
               Number of observations of n lognormal random variables
          R  : ndarray, optional
               If None, defaults to identity matrix (uncorrelated samples).
 
    Output:
          X  : ndarray
               Shape (n, N) array of correlated lognormal random samples.
               Each row corresponds to one random variable.
               Each column corresponds to one observation.
  
    Method: (Gaussian Copula)
        1. Perform eigenvalue decomposition of correlation matrix R = V @ Lambda @ V^T
        2. Generate uncorrelated standard normal samples Z ~ N(0, I)
        3. Apply correlation structure: Y = V @ sqrt(Lambda) @ Z, so Y ~ N(0, R)
        4. Transform to lognormal: X = exp(log(mednX) + Y * sqrt(V))
        where V = log(1 + covnX²)
    
    If X is a lognormal random variable, then log(X) is normally 
    distributed with mean log(mednX) and variance log(1+covnX²)
    
    Examples
    --------
        # Single variable, uncorrelated samples
            x = rnd(1.0, 0.5, N=1000)
        
        # Multiple correlated variables
            mednX = np.array([1.0, 2.0])
            covnX = np.array([0.5, 0.3])
            R = np.array([[1.0, 0.7], [0.7, 1.0]])
            x = rnd(mednX, covnX, N=1000, R=R)
    '''
    
    # Python does not implicitly handle scalars as arrays ... so ...
    # Convert array_like inputs to numpy column vectors (2D arrays) of floats
    mednX = np.atleast_2d(mednX).reshape(-1, 1).astype(float)
    covnX = np.atleast_2d(covnX).reshape(-1, 1).astype(float)

    n = len(mednX)

    # Validate that all parameter arrays have the same length
    if not (len(mednX) == n and len(covnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got mednX:{len(mednX)}, covnX:{len(covnX)}")
    
    if np.any(mednX <= 0):
        raise ValueError(" lognormal.rnd: mednX must be positive")
    if np.any(covnX <= 0):
        raise ValueError(" lognormal.rnd: covnX must be positive")

    # Compute variance of log(X) for each variable
    VlnX = np.log(1 + covnX**2)
    
    _, Y, _ = correlated_rvs(R, n, N, seed)

    # Transform to lognormal: x = exp(log(mednX) + Y * sqrt(VlnX))
    # Broadcasting: mednX and VlnX are (n,),
    # need to reshape for broadcasting with (n, N) ?
    X = np.exp( np.log(mednX) + Y * np.sqrt(VlnX) )
    
    if n == 1:
        X = X.flatten()

    return X
