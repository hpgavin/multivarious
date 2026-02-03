import numpy as np
from scipy.special import erf, erfinv


def pdf(x, medX, covX):
    '''
    lognormal.pdf
 
    Computes the Probability Density Function (PDF) of the lognormal distribution.
 
    INPUTS:
      x     = (array_like) 
              Evaluation points. Must be > 0 since lognormal is defined for x > 0.
      medX  = float
              Median of the lognormal distribution (X = exp(Y), where Y is normal)
      covX  = float
              Coefficient of variation of X: (covX = std(X) / mean(X))
 
    OUTPUT:
      f     = ndarray
              values of the PDF at each point in x
 
    FORMULAS:
      Variance log(x): V_lnX = log(1 + covX²), then:

    If X ~ Lognormal(medX, covX), then log(X) ~ Normal with:
        mean = log(medX)
        variance = log(1 + covX²)
        
    Reference: https://en.wikipedia.org/wiki/Log-normal_distribution
    '''
    x = np.asarray(x, dtype=float)

    # Store size (as in MATLAB)
    # r_x, c_x = x.shape if x.ndim == 2 else (x.size, 1) # Not used!

    # Replace invalid x <= 0 with 0.01 to avoid log(0)
    # This matches MATLAB behavior: X(idx) = 0.01
    x = np.where(x <= 0, 0.01, x)
    
    # Compute variance of log(X)
    VlnX = np.log(1 + covX**2)
    
    # Compute using Lognormal PDF formula
    f = (1 / (x * np.sqrt(2 * np.pi * VlnX))) * \
        np.exp(-0.5 * (np.log(x / medX))**2 / VlnX)
    
    return f


def cdf(x, medX, covX, return_ll=False):
    ''' 
    lognormal.cdf
 
    Computes the CDF of a lognormal distribution.
 
    Parameters:
         x     = array_like of Evaluation points
        medX   = float of Median of X
        covX   = float of Coefficient of variation of X
     return_ll = bool, optional
                 If True, also return log-likelihood
    Output: 
         F     = ndarray
                 CDF values at each point in x
         ll    = float, optional
                 Log-likelihood (only if return_ll=True)
    Notes:
        Lognormal CDF: F(x) = (1 + erf((log(x) - log(medX)) / sqrt(2V))) / 2
        where V = log(1 + covX²)
    ''' 
    x = np.asarray(x, dtype=float)
    
    # Check parameter validity
    if medX <= 0:
        raise ValueError(f"lognormal_cdf: medX = {medX}, must be > 0")
    if covX <= 0:
        raise ValueError(f"lognormal_cdf: covX = {covX}, must be > 0")
    
    # Replace invalid x <= 0 with 0.01 to avoid log(0)
    x = np.where(x <= 0, 0.01, x)
    
    # Compute variance of log(X)
    VlnX = np.log(1 + covX**2)
    
    # Lognormal CDF formula
    F = 0.5 * (1 + erf((np.log(x) - np.log(medX)) / np.sqrt(2 * VlnX)))
    
    if return_ll:
        ll = np.sum(np.log(pdf(x, medX, covX)))
        return F, ll
    
    return F


def inv(P, medX, covX):
    '''
    lognormal.inv
 
    Computes the inverse CDF (quantile function) for lognormal distribution.
 
    INPUTS:
      p     = array_like of Probability values (scalar or array), must be in (0,1)
      medX  = float (Median of the lognormal distribution)
      covX  = float (Coefficient of variation)
 
    OUTPUT:
      x     = ndarray
              Quantile values such that P(X <= x) = P
   
    FORMULA:
        x = exp( log(medX) + sqrt(2V_lnX) * erfinv(2p - 1) )
            where V_lnX = log(1 + covX²)
    NOTES:
        Instead of calculating P(X <= x) = p, this function finds x such that
        P(X <= x) = p. In other words, what number x has a cumulative probability p?
    '''
    P = np.asarray(P, dtype=float)
    
    # Clip probabilities to avoid erfinv(±1) = ±∞
    # Matches MATLAB: P(find(P<=0)) = eps; P(find(P>=1)) = 1-eps;
    eps = np.finfo(float).eps     # smallest positive float
    P = np.clip(P, eps, 1 - eps)  # restrict P to (0, 1)

    # Compute lognormal quantile using inverse CDF formula
    VlnX = np.log(1 + covX**2)  # Variance of log(X)
    x = np.exp(np.log(medX) + np.sqrt(2 * VlnX) * erfinv(2 * P - 1)) 

    return x


def rnd(medX, covX, N, R=None):
    '''
    lognormal.rnd
 
    Generate N observations of correlated (or uncorrelated) lognormal random variables.
 
    Parameters:
        medX : float or array_like
               Median(s) of the lognormal distribution. If array, shape (n,) for n variables.
        covX : float or array_like
               Coefficient(s) of variation. If array, shape (n,) for n variables.
          N  : int
               Number of observations (samples) to generate.
          R  : ndarray, optional
               If None, defaults to identity matrix (uncorrelated samples).
 
    Output:
          x  : ndarray
               Shape (n, N) array of correlated lognormal random samples.
               Each row corresponds to one random variable.
               Each column corresponds to one observation.
  
    Method: (Gaussian Copula)
        1. Perform eigenvalue decomposition of correlation matrix R = V @ Lambda @ V^T
        2. Generate uncorrelated standard normal samples Z ~ N(0, I)
        3. Apply correlation structure: Y = V @ sqrt(Lambda) @ Z, so Y ~ N(0, R)
        4. Transform to lognormal: X = exp(log(medX) + Y * sqrt(V))
        where V = log(1 + covX²)
    
    If X is a lognormal random variable, then log(X) is normally 
    distributed with mean log(medX) and variance log(1+covX²)
    
    Examples
    --------
        # Single variable, uncorrelated samples
            x = rnd(1.0, 0.5, N=1000)
        
        # Multiple correlated variables
            medX = np.array([1.0, 2.0])
            covX = np.array([0.5, 0.3])
            R = np.array([[1.0, 0.7], [0.7, 1.0]])
            x = rnd(medX, covX, N=1000, R=R)
    '''
    
    # Convert inputs to arrays # Python needs this to handle both scalars and arrays!
    medX = np.atleast_1d(medX).astype(float)
    covX = np.atleast_1d(covX).astype(float)
    
    # Determine number of random variables
    n = len(medX)
    
    # Validate all parameters --------------------------------------
    if len(covX) != n:
        raise ValueError(f"medX and covX must have the same length. "
                        f"Got medX:{len(medX)}, covX:{len(covX)}")
    
    # Check parameter validity
    if np.any(medX <= 0):
        raise ValueError("lognormal_rnd: medX must be greater than zero")
    if np.any(covX <= 0):
        raise ValueError("lognormal_rnd: covX must be greater than zero")
    
    # Default to identity matrix (uncorrelated samples) if no correlation specified
    if R is None:
        R = np.eye(n) # In
    
    # Validate correlation matrix
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError(f"Correlation matrix R must be square {n}x{n}, got {R.shape}")
    
    if not np.allclose(np.diag(R), 1.0):
        raise ValueError("corr_logn_rnd: diagonal of R must equal 1")
    
    if np.any(np.abs(R) > 1):
        raise ValueError("corr_logn_rnd: R values must be between -1 and 1")
    # ---------------------------------------------------------------------
    
    # Decompose correlation matrix: R = V @ Λ @ V^T
    eVal, eVec = np.linalg.eig(R)
    
    if np.any(eVal < 0):
        raise ValueError("corr_logn_rnd: R must be positive definite")
    
    # Generate independent standard normal samples: Z ~ N(0, I)
    Z = np.random.randn(n, N)
    
    # Apply correlation structure: Y = V @ sqrt(Λ) @ Z, so Y ~ N(0, R)
    Y = eVec @ np.diag(np.sqrt(eVal)) @ Z
    
    # Compute variance of log(X) for each variable
    VlnX = np.log(1 + covX**2)
    
    # Transform to lognormal: x = exp(log(medX) + Y * sqrt(VlnX))
    # Broadcasting: medX and VlnX are (n,), need to reshape for broadcasting with (n, N)
    x = np.exp(np.log(medX[:, np.newaxis]) + Y * np.sqrt(VlnX[:, np.newaxis]))
    
    if n == 1:
        x = x.flatten()

    ''' 
    # current output shape is (n, N). Add this if we want to transpose output:
    if n == 1:
    return x.T  # Return (N, 1) instead of (1, N) for single variable
    return x
    '''
    return x
