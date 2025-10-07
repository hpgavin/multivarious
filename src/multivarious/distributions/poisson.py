import numpy as np
from math import exp, factorial

# -------------------------------------------------------------------------
# PMF: poisson_pmf
#
# Computes the Probability Mass Function (PMF) of the Poisson distribution.
#
# INPUTS:
#   n : array-like or scalar, number of events (should be integer ≥ 0)
#   L : event rate (λ = t / T) or expected number of events
#
# OUTPUT:
#   p : probability of observing n events
#
# FORMULA:
#   P(N = n) = (L^n / n!) * exp(-L)
# -------------------------------------------------------------------------
def poisson_pmf(n, L):
    '''
    The Poisson distribution models the number of times an event occurs in a 
    fixed interval of time or space.
        * Events occur independently
        * The average rate of occurrence is constant (λ events per interval)
        * Two events can't happen at the exact same instant (events are discrete)
    λ (or sometimes L) = expected number of events in the interval.
    '''
    n = np.asarray(n, dtype=int)
    n = np.where(n < 0, 0, n)  # clip negative values to 0

    p = (L ** n) / np.vectorize(np.math.factorial)(n) * np.exp(-L)
    return p


# -------------------------------------------------------------------------
# CDF: poisson_cdf
#
# Computes the Cumulative Distribution Function (CDF) of the Poisson distribution.
#
# INPUTS:
#   n = scalar or array of integer values (number of observed events)
#   L = expected number of events (λ = t / T, from MATLAB context)
#
# OUTPUT:
#   F = array or scalar of probabilities P(N ≤ n)
#
# FORMULA:
#   F(n; L) = sum_{k=0}^{n} (L^k / k!) * exp(-L)
# -------------------------------------------------------------------------
def poisson_cdf(n, L):

    n = np.round(n).astype(int)            # Ensure n is integer-valued
    n = np.atleast_1d(n)                   # Support scalar and array inputs
    F = np.empty_like(n, dtype=float)      # Initialize result array

    for i, ni in enumerate(n):
        ks = np.arange(0, ni + 1)          # k = 0 to n
        F[i] = np.sum((L**ks) / np.vectorize(factorial)(ks)) * exp(-L)

    return F if F.size > 1 else F[0]        # Return scalar if input was scalar


# -------------------------------------------------------------------------
# RND: poisson_rnd
#
# Generates random samples from the Poisson distribution.
#
# INPUTS:
#   T : return period (float or array of shape r × c)
#   r : number of rows
#   c : number of columns
#
# OUTPUT:
#   x : r × c matrix of random Poisson samples
#
# METHOD:
#   Uses the waiting time method (multiplicative) similar to Knuth's algorithm.
#   See: https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
# -------------------------------------------------------------------------
def poisson_rnd(T, r, c):

    # Infer r, c if not explicitly passed
    if r is None or c is None:
        if np.isscalar(T):
            r, c = 1, 1
        else:
            r, c = T.shape

    # Validate T is scalar or matches shape
    T = np.asarray(T)
    if T.size != 1 and T.shape != (r, c):
        raise ValueError("poisson_rnd: T must be scalar or of shape (r, c)")

    # Compute L = exp(-1/T)
    L = np.exp(-1.0 / T)

    # Initialize output matrix
    p = np.ones((r, c))        # running product
    x = np.zeros((r, c), dtype=int)  # counter

    # Run multiplicative loop
    active = p >= L
    while np.any(active):
        p[active] *= np.random.rand(*p[active].shape)
        x[active] += 1
        active = p >= L

    return x - 1
