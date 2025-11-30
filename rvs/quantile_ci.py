

import numpy as np
from scipy.stats import binom

"""
quantile_ci.py

Exact distribution-free quantile CI via order-statistics.

Function:
    quantile_ci(sample, p, alpha=0.05)

Returns a distribution-free (1-alpha) confidence interval for the population
p-quantile q_p = F^{-1}(p) using order statistics. The CI is [X_(r), X_(s)]
where r and s are chosen from the Binomial(N,p) distribution so that
P(r <= K <= s-1) >= 1-alpha for K ~ Binomial(N,p).

Notes:
- The interval is exact and distribution-free for continuous parent distributions.
- If r == 1, the lower endpoint is conceptually -infinity (returned as None).
- If s == N+1, the upper endpoint is conceptually +infinity (returned as None).
- This module has no external dependencies besides numpy and scipy (for binomial CDF).

example 

...
from quantile_ci import quantile_ci
import numpy as np

alpha = 0.05
p = 0.9

lower, upper, r, s = quantile_ci(sample, p, alpha=alpha)
print("Selected ranks:", r, s)
print("Quantile CI:", lower, upper)
...

"""

import numpy as np
from scipy.stats import binom

def quantile_ci(sample, p, alpha=0.05):
    """
    Compute an exact distribution-free (1-alpha) CI for the population p-quantile q_p = F^{-1}(p)
    using order statistics. The CI is [X_(r), X_(s)] where r and s are chosen from the Binomial(N,p).
    
    Parameters
    ----------
    sample : array-like
        Observed sample (will be sorted internally).
    p : float
        Target quantile level (0 < p < 1), e.g. 0.5 for median.
    alpha : float
        Desired total error (two-sided). Default 0.05 for 95% CI.
    
    Returns
    -------
    (lower, upper, r, s)
        lower  - lower endpoint (float) or None if unbounded below (r==1 implies conceptual -inf)
        upper  - upper endpoint (float) or None if unbounded above (s==N+1 implies +inf)
        r, s   - integer ranks such that P( X_(r) <= q_p <= X_(s) ) >= 1-alpha.
                 Note X_(r) corresponds to sorted_sample[r-1].
    """
    sample = np.asarray(sample)
    if not (0 < p < 1):
        raise ValueError("p must be in (0,1)")
    N = len(sample)
    if N == 0:
        raise ValueError("sample must be non-empty")
    # Binomial CDF function for K ~ Binom(N,p): CDF(k) = P(K <= k)
    cdf = lambda k: binom.cdf(k, N, p)
    
    # Find r: smallest integer r in {1,..,N+1} such that P(K <= r-1) >= alpha/2
    r_minus_1 = 0
    while r_minus_1 <= N and cdf(r_minus_1) < alpha/2:
        r_minus_1 += 1
    r = r_minus_1 + 1  # because r_minus_1 = r-1
    
    # Find s-1: largest integer k with cdf(k) <= 1 - alpha/2
    s_minus_1 = N
    while s_minus_1 >= 0 and cdf(s_minus_1) > 1 - alpha/2:
        s_minus_1 -= 1
    s = s_minus_1 + 1  # because s = (s-1) + 1
    
    # Ensure r and s are within [1, N+1]
    r = max(1, min(r, N+1))
    s = max(1, min(s, N+1))
    
    # If r > s then no central interval available; fall back to smallest covering interval by expanding
    if r > s:
        best = None
        for r0 in range(1, N+2):
            for s0 in range(r0, N+2):
                prob = cdf(s0-1) - cdf(r0-1)
                if prob >= 1-alpha:
                    width = s0 - r0
                    if best is None or width < best[0]:
                        best = (width, r0, s0)
        if best is None:
            r, s = 1, N+1
        else:
            _, r, s = best
    
    sorted_sample = np.sort(sample)
    lower = None if r == 1 else sorted_sample[r-2]  # X_(r) with r=1 => conceptual -inf
    upper = None if s == N+1 else sorted_sample[s-1]  # X_(s) with s=N+1 => conceptual +inf
    return lower, upper, r, s

