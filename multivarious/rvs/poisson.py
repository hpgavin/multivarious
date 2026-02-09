import numpy as np
from scipy.special import factorial

from multivarious.utl.correlated_rvs import correlated_rvs

def pmf(n, t, T):
    '''
    poisson.pmf

    Computes the Probability Mass Function (PMF) of the Poisson distribution.

    Parameters:
        n : array_like or scalar
            Number of observed events (integer ≥ 0)
        t : scalar float positive valued
            The duration of time for the occurance of events
        T : scalar float positive valued
            mean return period of events 

    Output:
        p : ndarray
            Probability of observing n events

    Reference:
    https://en.wikipedia.org/wiki/Poisson_distribution

    Notes:
    The Poisson distribution models the number of times an event occurs in a 
    fixed interval of time or space.
        * Events occur independently
        * The mean return period of occurances is constant 
        * Two events can't happen at the exact same instant (events are discrete)
        * T = expected return period of events
    '''
    n = np.atleast_1d(np.round(n)).astype(int)  # Ensure n is integer-valued
    n = np.where(n < 0, 0, n)  # Clip negative values to 0

    p = ((t / T) ** n) / factorial(n) * np.exp(-t / T)

    return p


def cdf(n, t, T):
    '''
    poisson.cdf

    Computes the Cumulative Distribution Function (CDF) of the Poisson distribution.

    Parameters:
        n : array_like or scalar
            Number of observed events (integer ≥ 0)
        t : scalar float positive valued
            The duration of time for the occurance of events
        T : scalar float positive valued
            mean return period of events 

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
        F[i] = np.sum(((t/T)**ks) / factorial(ks)) * np.exp(-t/T)

    return F if F.size > 1 else F[0]        # Return scalar if input was scalar


def rnd(t, T, N, K, R, seed=None):
    '''
    Generate N samples of n correlated Poisson counts using K iterations of the multiplicative algorithm.
    Correlation is applied across the n processes at each iteration and shared for all N samples.
    
    Parameters:
        t : array-like shape (n,)
            Duration of observation for each process
        T : array-like shape (n,)
            Return period for each process (lambda = 1/T)
        N : int
            Number of samples per process
        K : int
            Maximum number of iterations (max count + buffer)
        R : ndarray shape (n, n)
            Correlation matrix among the n processes
        seed : int or None
            Random seed
    
    Returns:
        X : ndarray shape (n, N)
            Poisson counts for each process and sample
    '''

    t = np.atleast_1d(t).reshape(-1, 1).astype(float)
    T = np.atleast_1d(T).reshape(-1, 1).astype(float)
    n = len(T)
    
    exp_tT = np.exp(-t / T).flatten()  # shape (n,)
    
    X = np.zeros((n, N), dtype=int)
    
    rng = np.random.default_rng(seed)

    for i in range(N):
        p = np.ones(n)
        x = np.zeros(n, dtype=int)
        
        # Generate correlated uniforms for this sample
        _, _, U = correlated_rvs(n, K, R)

        active = p >= exp_tT
        iter_idx = 0
        
        while np.any(active) and iter_idx < K:
            p[active] *= U[active, iter_idx]
            x[active] += 1
            active = p >= exp_tT
            iter_idx += 1
        
        X[:, i] = x - 1  # Adjust per Knuth's algorithm
    
    return X
