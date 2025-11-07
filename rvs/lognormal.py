import numpy as np
from scipy.special import erf, erfinv


def pdf(x, medX, covX):
    '''
    lognormal.pdf
 
    Computes the Probability Density Function (PDF) of the lognormal distribution.
 
    INPUTS:
      x     = evaluation points. Must be > 0 since lognormal is defined for x > 0.
      medX  = median of the lognormal distribution (X = exp(Y), where Y is normal)
      covX  = coefficient of variation of X: (covX = std(X) / mean(X))
 
    OUTPUT:
      f     = values of the PDF at each point in x
 
    FORMULAS:
      Variance log(x): V_lnX = log(1 + covX²), then:
 
    If X ~ Lognormal(medX, covX), then log(X) ~ Normal(mean = log(medX), var = log(1 + covX²)).
 
    Reference: https://en.wikipedia.org/wiki/Log-normal_distribution
    '''
    x = np.asarray(x, dtype=float)

    # Store size (as in MATLAB)
    r_x, c_x = x.shape if x.ndim == 2 else (x.size, 1)

    # Replace invalid x <= 0 with small positive constant to avoid log(0)
    x = np.where(x <= 0, 0.01, x)

    # Compute variance of log(X)
    VlnX = np.log(1 + covX**2)

    # Compute the PDF using the lognormal formula
    f = (1 / np.sqrt(2 * np.pi * VlnX)) * (1 / x) * np.exp(-0.5 * (np.log(x / medX))**2 / VlnX)

    return f


def cdf(x, params, return_ll=False):  # return_ll if True returns log-likelihood
    ''' 
    lognormal.cdf
 
    Returns the CDF of a lognormal distribution with:
      - median = medX
      - coefficient of variation = covX
 
    INPUTS:
      x      = points at which to evaluate the CDF
      params = [medX, covX], a vector where:
                 medX  = median of X
                 covX  = coefficient of variation of X
 
    OUTPUT:
      F             = values of the CDF at each point in x
      ll (optional) = log-likelihood of x under lognormal(medX, covX)
 
    FORMULA:
      lognormal CDF F(x) = (1 + erf((log(x) - log(medX)) / sqrt(2V))) / 2
    ''' 
    x = np.asarray(x, dtype=float)
    medX = params[0]
    covX = params[1]

    # Store shape (as in MATLAB)
    r_x, c_x = x.shape if x.ndim == 2 else (x.size, 1)

    # Avoid log(0) or log of negatives
    x = np.where(x <= 0, 0.01, x)

    VlnX = np.log(1 + covX**2)
    F = 0.5 * (1 + erf((np.log(x) - np.log(medX)) / np.sqrt(2 * VlnX)))

    if return_ll:
        ll = np.sum(np.log(pdf(x, medX, covX)))
        return F, ll

    return F


def inv(P, medX, covX):
    '''
    lognormal.inv
 
    Computes the inverse CDF (also called quantile function) for lognormal.
 
    INPUTS:
      p     = probability values (scalar or array), must be in (0,1)
      medX  = median of the lognormal distribution
      covX  = coefficient of variation
 
    OUTPUT:
      x     = quantile values, such that P(X <= x) = p
   
    FORMULA:
      If V_lnX = log(1 + covX²), then:
        x = exp( log(medX) + sqrt(2V_lnX) * erfinv(2p - 1) )
    '''
    '''
    Instead of calculating P(X <= x) = p, this function finds x such that
    P(X <= x) = p
    In other words, what number x has a cumulative probability p?
    '''
    P = np.asarray(P, dtype=float)
    
    # Avoid invalid probabilities: clip to machine epsilon
    eps = np.finfo(float).eps     # smallest positive float
    P = np.clip(P, eps, 1 - eps)  # restrict P to (0, 1)

    # Compute lognormal quantile using inverse CDF formula
    VlnX = np.log(1 + covX**2)  # Variance of log(X)
    x = np.exp(np.log(medX) + np.sqrt(2 * VlnX) * erfinv(2 * P - 1)) 

    return x


def rnd(medX, covX, r=1, c=1, z=None, seed=None):
    '''
    lognormal.rnd
 
    Returns random samples from a lognormal distribution defined by:
      - median = medX
      - coefficient of variation = covX
 
    INPUTS:
      medX  = median of the lognormal distribution (scalar or array)
      covX  = coefficient of variation of the lognormal distribution
      r     = number of rows in output (default 1)
      c     = number of columns in output (default 1)
      z     = optional standard normal samples (r × c), possibly correlated
      seed  = optional random seed or Generator (used if z is None)
   
    OUTPUT:
      x     = (r × c) array of lognormal random samples
   
      If X is a lognormal random variable, then the log of X is normally 
      distributed with a mean of  log(medX) and a variance of log(1+covX^2)
    FORMULA:
      V_lnX = log(1 + covX²)
      x = exp( log(medX) + z * sqrt(V) )
    '''
    # Input checks
    if np.any(np.asarray(medX) <= 0):
        raise ValueError("medX must be greater than zero")
    if np.any(np.asarray(covX) <= 0):
        raise ValueError("covX must be greater than zero")

    # Case 1: z is provided (e.g., for correlated samples)
    if z is not None:
        z = np.asarray(z)
        r, c = z.shape
    else:
        # Case 2: generate z ~ N(0, 1)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((r, c))

    # Expand medX and covX to (r, c) if they are scalars
    medX = np.broadcast_to(medX, (r, c))
    covX = np.broadcast_to(covX, (r, c))

    # Compute variance of log(X)
    VlnX = np.log(1 + covX**2)

    # Apply lognormal transformation
    x = np.exp(np.log(medX) + z * np.sqrt(VlnX))
    return x


if __name__ == "__main__":
    x = 1.5
    medX = 1.0
    covX = 0.5

    f = pdf(x, medX, covX)
    print("PDF at x = 1.5:", f)
