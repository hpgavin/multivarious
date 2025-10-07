import numpy as np

from scipy.special import gamma as gamma_func  # gamma function from scipy
# -------------------------------------------------------------------------
# PDF: gamma_pdf
#
# Computes the Probability Density Function (PDF) of the Gamma distribution
# using its mean (m) and coefficient of variation (c)
#
# INPUTS:
#   x = evaluation points (can be scalar or array-like)
#   m = mean of the distribution (scalar)
#   c = coefficient of variation (std/mean), must be >= 0.5
#
# OUTPUT:
#   f = PDF values evaluated at x
# -------------------------------------------------------------------------
def gamma_pdf(x, m, c):
    x = np.asarray(x, dtype=float)

    k = 1.0 / c**2            # shape parameter
    theta = c**2 * m          # scale parameter

    # Direct translation of MATLAB line
    f = theta**(-k) * x**(k - 1) * np.exp(-x / theta) / gamma_func(k)

    # Apply small value to negative x values
    f[x < 0] = 1e-12

    return f



from scipy.special import gammainc  # Regularized lower incomplete gamma function
# -------------------------------------------------------------------------
# CDF: gamma_cdf
#
# Computes the Cumulative Distribution Function (CDF) of the Gamma distribution
# in terms of mean (m) and coefficient of variation (c).
#
# INPUTS:
#   x      = evaluation points (can be scalar, vector, matrix)
#   params = [m, c], where:
#              m = mean of the distribution
#              c = coefficient of variation (std/mean)
#
# OUTPUT:
#   F = cumulative probability evaluated at x
# -------------------------------------------------------------------------
def gamma_cdf(x, params):
    x = np.asarray(x, dtype=float)

    m, c = params

    if c < 0.5:
        print(f'gamma_cdf: c = {c:.4f}, lower limit of coefficient of variation is 0.5')

    # Compute shape (k) and scale (theta) from mean and coeff. of variation
    k = 1.0 / c**2
    theta = c**2 * m

    # Create a safe version of x (avoid div-by-zero in gammainc)
    xp = np.copy(x)
    xp[x < 0] = 1e-5

    # Regularized lower incomplete gamma function
    F = gammainc(k, xp / theta)

    # Set CDF = 0 for x ≤ 0
    F[x <= 0] = 0.0

    return F


# -------------------------------------------------------------------------
# INV: gamma_inv
#
# Computes the inverse CDF (quantile function) of the Gamma distribution
# defined by mean `m` and coefficient of variation `c`, for given probability P.
#
# INPUTS:
#   P : array-like
#       Non-exceedance probabilities (values between 0 and 1)
#   m : float
#       Mean of the Gamma distribution
#   c : float
#       Coefficient of variation (must be ≥ 0.5)
#
# OUTPUT:
#   x : array-like
#       Quantiles such that Prob[X ≤ x] = P
# -------------------------------------------------------------------------
def gamma_inv(P, m, c):
    P = np.asarray(P, dtype=float)

    if np.any(c < 0.5):
        print(f'gamma_inv: c = {c:.4f}, lower limit of coefficient of variation is 0.5')

    # Compute shape and scale
    k = 1.0 / c**2
    theta = c**2 * m

    # Initialize parameters for Newton-Raphson
    myeps = 1e-12
    MaxIter = 100
    x_old = np.sqrt(myeps) * np.ones_like(P)  # ones_like makes x_old same shape as P

    for _ in range(MaxIter):
        # Evaluate function value and derivative
        F_x = gamma_cdf(x_old, [m, c])
        f_x = gamma_pdf(x_old, m, c)
        h = (F_x - P) / f_x

        x_new = x_old - h

        # Avoid values too close to zero
        x_new[x_new <= myeps] = x_old[x_new <= myeps] / 10
        h = x_old - x_new

        if np.max(np.abs(h)) < np.sqrt(myeps):
            break

        x_old = x_new

    # Final output
    x = x_new
    '''
    this output must be compared with MATLAB implementation––
    to be revised.
    '''
    return x


# -------------------------------------------------------------------------
# RND: gamma_rnd
#
# Generates random samples from a Gamma distribution given mean and CV.
#
# INPUTS:
#   m = mean of the distribution
#   c = coefficient of variation (std/mean)
#   R = number of rows in output
#   C = number of columns in output
#
# OUTPUT:
#   x = random samples from Gamma(m, c), shape (R, C)
# -------------------------------------------------------------------------
def gamma_rnd(m, c, R, C):
    
    k = 1.0 / c**2          # shape parameter
    theta = c**2 * m        # scale parameter

    # Generate gamma-distributed samples with shape k and scale theta
    x = np.random.gamma(shape=k, scale=theta, size=(R, C))
    
    return x
