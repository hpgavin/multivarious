import numpy as np

# -------------------------------------------------------------------------
# PDF: laplace_pdf
#
# Computes the Probability Density Function (PDF) of the Laplace distribution.
#
# INPUTS:
#   x       = evaluation points (scalar or array)
#   muX     = mean (location parameter) of the Laplace distribution
#   sigmaX  = standard deviation (scale parameter)
#
# OUTPUT:
#   f       = values of the PDF at each point in x
#
# FORMULA:
#   f(x) = (1 / (sqrt(2) * sigmaX)) * exp(-sqrt(2) * |x - muX| / sigmaX)
# -------------------------------------------------------------------------
def laplace_pdf(x, muX, sigmaX):
    x = np.asarray(x, dtype=float)
    sr2 = np.sqrt(2)

    # Laplace PDF formula (symmetric around muX)
    f = (1 / (sr2 * sigmaX)) * np.exp(-sr2 * np.abs(x - muX) / sigmaX)
    return f


# -------------------------------------------------------------------------
# CDF: laplace_cdf
#
# Computes the Cumulative Distribution Function (CDF) of the Laplace distribution.
#
# INPUTS:
#   x       = evaluation points (scalar or array)
#   params  = [muX, sigmaX] (mean and std. dev.)
#
# OUTPUT:
#   F       = values of the CDF at each point in x
#
# FORMULA:
#   For x <= mu: F(x) = 0.5 * exp( (x - mu) / (sigma / sqrt(2)) )
#   For x >  mu: F(x) = 1 - 0.5 * exp( -(x - mu) / (sigma / sqrt(2)) )
# -------------------------------------------------------------------------
def laplace_cdf(x, params):
    x = np.asarray(x, dtype=float)
    muX = params[0]
    sigmaX = params[1]
    sr2 = np.sqrt(2)

    # Allocate output array
    F = np.zeros_like(x)

    # Apply formula piecewise
    F[x <= muX] = 0.5 * np.exp(-sr2 * np.abs(x[x <= muX] - muX) / sigmaX)
    F[x >  muX] = 1 - 0.5 * np.exp(-sr2 * np.abs(x[x > muX] - muX) / sigmaX)

    return F


import numpy as np

# -------------------------------------------------------------------------
# INV: laplace_inv
#
# Computes the inverse CDF (quantile function) of the Laplace distribution.
#
# INPUTS:
#   P      = array of non-exceedance probabilities (must be in [0, 1])
#   muX    = mean of the Laplace distribution
#   sigmaX = standard deviation of the Laplace distribution
#
# OUTPUT:
#   X      = values x such that P = CDF(x)
#
# FORMULA:
#   If U ~ Uniform(0, 1), the Laplace inverse CDF is:
#     X = μ + (σ/√2) * log(2P)      for P ≤ 0.5
#     X = μ - (σ/√2) * log(2 - 2P)  for P > 0.5
#
# Reference: https://en.wikipedia.org/wiki/Laplace_distribution
# -------------------------------------------------------------------------
def laplace_inv(P, muX, sigmaX):
    P = np.asarray(P, dtype=float)
    sr2 = np.sqrt(2)

    # Start by assuming all P values on left side
    X = muX + sigmaX / sr2 * np.log(2 * P)

    # Find those where X >= muX (should've used right-side formula)
    idx = X >= muX
    X[idx] = muX - sigmaX / sr2 * np.log(2 - 2 * P[idx])

    return X


# -------------------------------------------------------------------------
# RANDOM SAMPLING: laplace_rnd
#
# Generates a matrix of random samples from the Laplace distribution.
#
# INPUTS:
#   muX     = mean of the distribution
#   sigmaX  = standard deviation
#   r       = number of rows
#   c       = number of columns
#
# OUTPUT:
#   x       = r x c matrix of Laplace samples
#
# METHOD:
#   Uses inverse transform sampling:
#     1. Draw u ~ Uniform(0, 1)
#     2. Apply inverse CDF of Laplace to get x
# -------------------------------------------------------------------------
import numpy as np

def laplace_rnd(mX, sX, r, c, z=None):
    """
    Generate Laplace-distributed random samples.

    If `z` is provided, it is used directly (like passing in a random normal matrix).
    Otherwise, we generate uniform randoms ourselves.
    """
    if np.any(sX <= 0) or np.any(np.isinf(sX)):
        raise ValueError("laplace_rnd: sX must be > 0 and finite")

    sr2 = np.sqrt(2)

    # Case: user provides a matrix instead of r and c (like nargin==3 in MATLAB)
    if z is not None:
        u = z  # Use provided matrix directly
        r, c = u.shape
    else:
        u = np.random.rand(r, c)

    # Expand scalar params to matrix if needed
    mX = np.full((r, c), mX) if np.isscalar(mX) else mX
    sX = np.full((r, c), sX) if np.isscalar(sX) else sX

    # Preallocate
    x = np.empty((r, c))

    # Left side (u ≤ 0.5)
    in_mask = u <= 0.5
    x[in_mask] = mX[in_mask] + sX[in_mask] / sr2 * np.log(2 * u[in_mask])

    # Right side (u > 0.5)
    ip_mask = ~in_mask
    x[ip_mask] = mX[ip_mask] - sX[ip_mask] / sr2 * np.log(2 * (1 - u[ip_mask]))

    '''
    this function in this module must be revised- 
    implementation to be reviewed.
    '''
    return x
