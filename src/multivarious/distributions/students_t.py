import numpy as np
from scipy.special import beta as beta_func, betaincinv, betainc

# -------------------------------------------------------------------------
# PDF: t_pdf
#
# Computes the PDF of the Student's t-distribution.
#
# INPUTS:
#   t : evaluation points
#   k : degrees of freedom (must be > 0)
#
# OUTPUT:
#   f : PDF evaluated at t
# -------------------------------------------------------------------------
def pdf(t, k):
    t = np.asarray(t, dtype=float)

    # Compute the PDF using the known closed-form
    f = (np.exp(-(k + 1) * np.log(1 + (t ** 2) / k) / 2)) / (np.sqrt(k) * beta_func(k / 2, 0.5))

    return f


# -------------------------------------------------------------------------
# CDF: t_cdf
#
# Computes the CDF of the Student's t-distribution.
#
# INPUTS:
#   t : evaluation points
#   k : degrees of freedom
#
# OUTPUT:
#   F : cumulative probability values
# -------------------------------------------------------------------------
import numpy as np
from scipy.stats import norm

# -------------------------------------------------------------------------
# CDF: t_cdf (Student's t-distribution)
# Mirrors the MATLAB implementation using the Wilson-Hilferty approximation
# and piecewise handling for degrees of freedom k = 1, 2, and k > 2.
# -------------------------------------------------------------------------
def cdf(t, k):
    t = np.asarray(t, dtype=float)

    if k == 1:
        return 0.5 + np.arctan(t) / np.pi

    elif k == 2:
        return 0.5 + t / (2 * np.sqrt(2 + t**2))

    else:
        ts = t / np.sqrt(k)
        ttf = 1 / (1 + ts**2)

        u = np.ones_like(ts)
        s = np.ones_like(ts)

        if k % 2 == 1:  # odd degrees of freedom
            m = (k - 1) // 2
            for ii in range(2, m + 1):
                u *= (1 - 1 / (2 * ii - 1)) * ttf
                s += u
            return 0.5 + (ts * ttf * s + np.arctan(ts)) / np.pi

        else:  # even degrees of freedom
            m = k // 2
            for ii in range(1, m):
                u *= (1 - 1 / (2 * ii)) * ttf
                s += u
            return 0.5 + (ts * np.sqrt(ttf) * s) / 2.0


# -------------------------------------------------------------------------
# INV: t_inv (inverse CDF / quantile function)
# Mirrors the MATLAB implementation using the Wilson-Hilferty approximation
# -------------------------------------------------------------------------
from scipy.special import betaincinv
def inv(p, k):
    p = np.asarray(p)
    z = betaincinv(k / 2.0, 0.5, 2 * np.minimum(p, 1 - p))
    x = np.sign(p - 0.5) * np.sqrt(k * (1 / z - 1))
    '''
    must review this implementation.
    '''
    return x

#def t_inv(p, k):
#    p = np.asarray(p, dtype=float)
#
#    # Wilson-Hilferty transformation parameters
#    m = 1 - 2 / (9 * k)
#    s = np.sqrt(2 / (9 * k))
#
#   # Inverse CDF (quantile) of normal distribution
#    z = norm.ppf(p, loc=m, scale=s)
#
#    # Apply inverse transformation
#    t = k * z**3
#
#    return t