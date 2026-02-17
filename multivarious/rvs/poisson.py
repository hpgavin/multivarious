import numpy as np
from scipy.special import gammaln, gammaincc

from multivarious.utl.correlated_rvs import correlated_rvs

def _ppp_(k, t, T): 
    '''
    Validate and preprocess input parameters for consistency and correctness."

    INPUTS:
        k : array_like or scalar
            Number of observed events (integer ≥ 0)
        t : scalar float positive valued
            The duration of time for the occurance of events
        T : scalar float positive valued
            mean return period of events 
    '''

    k = np.atleast_1d(np.round(k)).reshape(1, -1).astype(int)
    k = np.where(k < 0, 0, k)  # Clip negative values to 0
    t = np.atleast_1d(t).reshape(-1, 1).astype(float)
    T = np.atleast_1d(T).reshape(-1, 1).astype(float)
    n = len(T)  

    if not ( (len(t) == n or len(t) == 1) and (len(T) == n or len(T) == 1)):
        raise ValueError(f"T and t arrays must have the same length. "
                         f"Got t:{len(t)}, T:{len(T)}")

    return k, t, T, n


def pmf(k, t, T):
    '''
    poisson.pmf

    Computes the Probability Mass Function (PMF) of the Poisson distribution.

    INPUTS:
        k : array_like or scalar
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
    https://en.wikipedia.org/wiki/Poisson_distribution#Evaluating_the_Poisson_distribution
    '''

    k, t, T, n = _ppp_(k, t, T)

    r = t/T  # rates of each Poisson process "nu" or "lambda"

#   p = (r ** n) / factorial(n) * np.exp(-r) # more round-off error

    p = np.exp( k * np.log(r) - r - gammaln(k+1) ) # less round-off error

    if n == 1 and p.shape[0] == 1:
        p = p.flatten()

    return p


def cdf(k, params ):
    '''
    poisson.cdf

    Computes the Cumulative Distribution Function (CDF) of the Poisson distribution.

    INPUTS:
        k : array_like or scalar
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
    https://en.wikipedia.org/wiki/Incomplete_gamma_function#Regularized_gamma_functions_and_Poisson_random_variables
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html
    '''

    t, T = params

    k, t, T, n = _ppp_(k, t, T)
   
    '''
    # more round-off error
    F = np.zeros(n, len(k) )
    for i, ki in enumerate(k):
        k = np.arange(0, ki + 1)          # k = 0 to n
        F[i] = np.sum( pmf( k, t, T ) ) 
    '''
    
    # Regularized upper incomplete gamma function. less round-off error
    F = gammaincc(k+1, (t/T))

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()     # Return 1D array for a single rv

    if len(k) == 1:
        F = F[0]            # Return scalar for a singe rv and a single k
 
    return F


def rnd(t, T, N, R=None, seed=None):
    '''
    Generate N samples of n correlated Poisson counts 
    Correlation is applied across the n processes at each iteration and shared for all N samples.
    
    INPUTS:
        t : scalar or array-like shape (n,)
            Duration of observation for each process
        T : scalar or array-like shape (n,)
            Return period for each process (lambda = 1/T)
        N : int
            Number of samples per process
        R : ndarray shape (n, n) or None
            Correlation matrix among the n processes
        seed : int or None
            Random number generator seed
    
    OUTPUTS:
        X : ndarray shape (n, N)
            Poisson counts for each of n processes and each of N samples
  

    https://en.wikipedia.org/wiki/Poisson_distribution#Computational_methods

    Poisson generator based upon the inversion by sequential search:

    Devroye, Luc (1986). "Discrete Univariate Distributions".
    Non-Uniform Random Variate Generation. New York, NY: Springer-Verlag.
    pp. 485–553.
    doi:10.1007/978-1-4613-8643-8_10. ISBN 978-1-4613-8645-2.

    init:
        Let x ← 0, p ← e−λ, s ← p.
        Generate uniform random number u in [0,1].
    while u > s do:
        x ← x + 1.
        p ← p × λ / x.
        s ← s + p.
    return x.
    '''
    _, t, T, n = _ppp_(0, t, T)
    
    L = (t/T).flatten()

    exp_tT = np.exp(-L)  # shape (n,)
    
    X = np.zeros((n, N), dtype=int)

    # N observations of n correlated standard uniform variables 
    _, _, U = correlated_rvs(R, n, N, seed)

    for j in range(N):
        x = np.zeros(n, dtype=int)
        p = exp_tT.copy() # !!
        s = p.copy()      # !!
        
        active_idx = U[:,j] > s
        iteration = 0

        while np.any(active_idx) and  np.max(x) < 100:
            x[active_idx] += 1
            p[active_idx] *= L[active_idx] / x[active_idx]
            s[active_idx] += p[active_idx]
            active_idx = U[:, j] > s
            iteration += 1

        X[:, j] = x  

        if iteration >= 100:
            print(f"Warning: large iteration count at sample {j}")

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()
    
    return X

