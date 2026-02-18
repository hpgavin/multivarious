# avg_cov_func.py
# -----------------------------------------------------------------------------
# Translated from MATLAB's avg_cov_func.m
# Computes the risk-adjusted (penalized) average cost and coefficient of variation.
# -----------------------------------------------------------------------------

import numpy as np

def avg_cov_func(func, u, s0, s1, options, consts=None, BOX=1):
    """
    Compute the average and coefficient of variation of a penalized cost function.

    Parameters
    ----------
    func : callable
        Function to optimize: f, g = func(v, consts)
    u : np.ndarray
        Scaled design variables (column-like) ( -1 < v < +1 )
    s0, s1 : np.ndarray or float
        Linear scaling factors mapping [v_lb, v_ub] -> [-1, +1]
    options : np.ndarray
        Optimization settings vector (see opt_options)
    consts : np.ndarray, optional
        Additional constants (non-design variables)
    BOX : int, optional
        1 to bound u within [-1, 1], 0 to allow unbounded (default=1)

    Returns
    -------
    F_risk : float
        Risk-adjusted average cost (84th percentile of mean)
    avg_g : np.ndarray
        Average constraint vector
    u : np.ndarray
        Possibly bounded u (if BOX=1)
    cov_F : float
        Coefficient of variation of F
    m : int
        Number of evaluations used
    """

    tol_g   = options[3]
    penalty = options[5]
    q       = options[6]
    m_max   = int(options[7])
    err_F   = options[8]
    Za2     = 1.645  # 90% confidence level

    avg_F = 0.0    # mean of F
    ssq_F = 0.0  # sum square values for F 
    cov_F = 0.0    # coefficient of variation for F
    max_F = 0.0  # maximum value of F
    avg_g = 0.0
    m = 0

    u = np.asarray(u, dtype=float).flatten()
    if BOX:
        u = np.clip(u, -1.0, 1.0)

    for m in range(1, m_max + 1):
        f, g = func(s0+s1*u, consts)             # objective, constraints
        g = np.asarray(g, dtype=float).flatten()       # constraints as a vector
        F_A = f + penalty * np.sum(g * (g > tol_g))**q # augmented objective

        # Welford's recursive update of a mean and a standard deviation 
        dF = F_A - avg_F
        avg_F += dF / m                 # update the mean of F
        ssq_F += dF * (F_A - avg_F)     # update the sum of squares of F
        max_F = max(max_F, F_A)
        avg_g = avg_g + (g - avg_g) / m if m>1 else g      # update average constraint

        if m > 1:
            cov_F = np.sqrt(ssq_F / (m - 1)) / np.abs(avg_F)  # update the c.o.v. of F
            if m > 2 and m > (Za2 * cov_F / err_F)**2:
                break

    F_risk = avg_F
    if m > 1:
#       CHOOSE ONE OF THE FOLLOWING RISK-BASED PERFORMANCE MEASURES ...
#       F_risk = avg_F                          # average-of-N values
#       F_risk = avg_F * ( 1 + cov_F/np.sqrt(m) ) # 84th percentile of the avg. of F
        F_risk = avg_F * ( 1 + cov_F )            # 84th percentile of F
#       F_risk = max_F;                       # largest-of-N values

    return F_risk, avg_g, u, cov_F, m

"""
Welford, B. P. (1962).
"Note on a method for calculating corrected sums of squares and products".
Technometrics. 4 (3): 419–420. doi:10.2307/1266577. JSTOR 1266577.

Donald E. Knuth (1998).
The Art of Computer Programming, volume 2: Seminumerical Algorithms, 3rd edn.,
p. 232. Boston: Addison-Wesley.
See Section 4.2.2, Exercise 12, and the accompanying discussion on updating variance and mean recursively.
"""
