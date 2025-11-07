import numpy as np

# -------------------------------------------------------------------------
# PDF: rayleigh_pdf
#
# Computes the Probability Density Function (PDF) of the Rayleigh distribution
# using the mean parameter muX.
#
# INPUT:
#   X    = Evaluation points (scalar or array-like)
#   muX  = Mean of the Rayleigh distribution (must be > 0)
#
# OUTPUT:
#   f    = PDF values evaluated at X
# -------------------------------------------------------------------------
def pdf(X, muX):
    X = np.asarray(X, dtype=float)

    # Convert mean to mode: modeX = muX * sqrt(2 / pi)
    modeX = muX * np.sqrt(2 / np.pi)

    # Replace non-positive values to prevent invalid evaluation
    X = np.where(X <= 0, 0.01, X)

    # Apply the Rayleigh PDF formula
    f = (X / modeX**2) * np.exp(-0.5 * (X / modeX)**2)

    return f


# -------------------------------------------------------------------------
# CDF: rayleigh_cdf
#
# Computes the Cumulative Distribution Function (CDF) of the Rayleigh distribution
# with mean muX, evaluated at values in X.
#
# INPUTS:
#   X    = Values to evaluate the CDF at (array-like)
#   muX  = Mean of the Rayleigh distribution
#
# OUTPUT:
#   F    = CDF values at each point in X
# -------------------------------------------------------------------------
def cdf(X, muX):
    X = np.asarray(X, dtype=float)
    
    # Replace X <= 0 with small positive number (to match MATLAB behavior)
    X[X <= 0] = 0.01

    # Convert mean muX to modeX using Rayleigh identity
    modeX = muX * np.sqrt(2 / np.pi)

    # Apply the Rayleigh CDF formula
    F = 1.0 - np.exp(-0.5 * (X / modeX)**2)

    return F


import numpy as np

# -------------------------------------------------------------------------
# INV: rayleigh_inv
#
# Computes the inverse CDF (quantile function) of the Rayleigh distribution.
# Mirrors the MATLAB version using the mean muX as input instead of scale.
#
# INPUTS:
#   P     = non-exceedance probabilities (0 ≤ P ≤ 1)
#   muX   = mean of the Rayleigh distribution
#
# OUTPUT:
#   x     = inverse CDF values (same shape as P)
# -------------------------------------------------------------------------
def inv(P, muX):
    P = np.asarray(P, dtype=float)

    # Clamp values: ensure P stays in [0, 1] just like MATLAB does
    P[P <= 0] = 0.0
    P[P >= 1] = 1.0

    # Convert mean to mode using mu = mode * sqrt(pi / 2)
    modeX = muX * np.sqrt(2 / np.pi)

    # Compute the inverse CDF formula
    x = modeX * np.sqrt(-2.0 * np.log(1 - P))

    return x


def rnd(muX, r, c=None):
    """
    Generate samples from the Rayleigh distribution using either:
    (1) muX, and a custom matrix of uniform samples (r = matrix, c=None)
    (2) muX, and integers r, c → will generate random samples via np.random.rand(r, c)
    """

    if np.any(muX <= 0) or np.any(np.isinf(muX)):
        raise ValueError("rayleigh_rnd: muX must be greater than zero")

    # Convert mean to mode
    modeX = muX * np.sqrt(2 / np.pi)

    # Case (1): r is already a matrix of uniform random numbers
    if c is None and isinstance(r, np.ndarray):
        u = r
        r, c = u.shape

    # Case (2): Generate uniform samples with shape (r, c)
    elif c is not None:
        u = np.random.rand(r, c)

    else:
        raise ValueError("rayleigh_rnd: Either provide a matrix (r) or integers (r, c)")

    # Broadcast muX if needed
    if np.prod(np.shape(modeX)) == 1:
        modeX = modeX * np.ones((r, c))

    # Inverse transform sampling
    x = modeX * np.sqrt(-2.0 * np.log(u))

    return x


