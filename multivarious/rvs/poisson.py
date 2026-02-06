import numpy as np
from scipy.special import factorial


#from multivarious.utl.correlated_rvs import correlated_rvs

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


def rnd(T, N, R=None, seed=None):
    '''
    poisson.rnd

    Generates random samples from the Poisson distribution using the
    multiplicative (Knuth) algorithm.

    Parameters:
        T : float or ndarray (n,)
            Return period (used to compute λ = 1/T)
        N : int
            Number of values of each of the n Poisson random variables
        R : float (n,n)
            Correlation matrix among the standardized Poisson random varibles
            --- not implemented 

    Output:
        X : ndarray of shape (r, c)
            Random samples drawn from the Poisson distribution

    Notes:
    Uses the waiting time method (multiplicative) similar to Knuth's algorithm.
    Reference:
    https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
    '''

    rng = np.random.default_rng(seed)

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    T = np.atleast_1d(T).astype(float)

    # Determine number of random variables
    n = len(T)

    # Initialize output matrix
    p = np.ones((n, N))                    # running product
    X = np.zeros((n, N), dtype=int)        # counter

    L = np.exp(-1.0 / T)

    # Run multiplicative loop
    active = p >= L
    while np.any(active):
        p[active] *= rng.random(np.sum(active))
        X[active] += 1
        active = p >= L

    X = X - 1

    if r == 1:
        X = X.flatten() 

    return X 
