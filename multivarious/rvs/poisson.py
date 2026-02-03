import numpy as np
from scipy.special import factorial


def pmf(n, L):
    '''
    poisson.pmf

    Computes the Probability Mass Function (PMF) of the Poisson distribution.

    Parameters:
        n : array_like or scalar
            Number of observed events (integer ≥ 0)
        L : float
            Expected number of events (rate parameter λ)

    Output:
        p : ndarray
            Probability of observing n events

    Reference:
    https://en.wikipedia.org/wiki/Poisson_distribution

    Notes:
    The Poisson distribution models the number of times an event occurs in a 
    fixed interval of time or space.
        * Events occur independently
        * The average rate of occurrence is constant (λ events per interval)
        * Two events can't happen at the exact same instant (events are discrete)
        * λ (or sometimes L) = expected number of events in the interval.
    '''
    n = np.asarray(n, dtype=int)
    n = np.where(n < 0, 0, n)  # clip negative values to 0

    p = (L ** n) / factorial(n) * np.exp(-L)
    return p


def cdf(n, L):
    '''
    poisson.cdf

    Computes the Cumulative Distribution Function (CDF) of the Poisson distribution.

    Parameters:
        n : array_like or scalar
            Number of observed events (integer ≥ 0)
        L : float
            Expected number of events (rate parameter λ)

    Output:
        F : ndarray or float
            Cumulative probability P(N ≤ n)

    Reference:
    https://en.wikipedia.org/wiki/Poisson_distribution
    '''
    n = np.round(n).astype(int)            # Ensure n is integer-valued
    n = np.atleast_1d(n)                   # Support scalar and array inputs
    F = np.empty_like(n, dtype=float)      # Initialize result array

    for i, ni in enumerate(n):
        ks = np.arange(0, ni + 1)          # k = 0 to n
        F[i] = np.sum((L**ks) / factorial(ks)) * np.exp(-L)

    return F if F.size > 1 else F[0]        # Return scalar if input was scalar


def rnd(T, r=None, c=None):
    '''
    poisson.rnd

    Generates random samples from the Poisson distribution using the
    multiplicative (Knuth) algorithm.

    Parameters:
        T : float or ndarray
            Return period (used to compute λ = 1/T)
        r : int
            Number of rows in the output
        c : int
            Number of columns in the output

    Output:
        X : ndarray of shape (r, c)
            Random samples drawn from the Poisson distribution

    Notes:
    Uses the waiting time method (multiplicative) similar to Knuth's algorithm.
    Reference:
    https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    '''
    # Infer r, c if not explicitly passed
    if r is None or c is None:
        if np.isscalar(T):
            r, c = 1, 1
        else:
            r, c = np.asarray(T).shape

    # Validate T is scalar or matches shape
    T = np.asarray(T)
    if T.size != 1 and T.shape != (r, c):
        raise ValueError("poisson_rnd: T must be scalar or of shape (r, c)")

    # Compute L = exp(-1/T)
    L = np.exp(-1.0 / T)

    # Initialize output matrix
    p = np.ones((r, c))                    # running product
    X = np.zeros((r, c), dtype=int)        # counter

    # Run multiplicative loop
    active = p >= L
    while np.any(active):
        p[active] *= np.random.rand(np.sum(active))
        X[active] += 1
        active = p >= L

    X = X - 1

    if r == 1:
        X = X.flatten() 

    return X 
