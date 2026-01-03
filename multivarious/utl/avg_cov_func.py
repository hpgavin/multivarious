# avg_cov_func.py
# -----------------------------------------------------------------------------
# Translated from MATLAB's avg_cov_func.m
# Computes the risk-adjusted (penalized) average cost and coefficient of variation.
# -----------------------------------------------------------------------------

import numpy as np

def avg_cov_func(func, v, s0, s1, options, consts=None, BOX=1):
    """
    Compute the average and coefficient of variation of a penalized cost function.

    Parameters
    ----------
    func : callable
        Function to optimize: f, g = func(v, consts)
    v : np.ndarray
        Scaled design variables (column-like) ( -1 < v < +1 )
    s0, s1 : np.ndarray or float
        Linear scaling factors mapping [v_lb, v_ub] -> [-1, +1]
    options : np.ndarray
        Optimization settings vector (see opt_options)
    consts : np.ndarray, optional
        Additional constants (non-design variables)
    BOX : int, optional
        1 to bound v within [-1, 1], 0 to allow unbounded (default=1)

    Returns
    -------
    F_risk : float
        Risk-adjusted average cost (84th percentile of mean)
    avg_g : np.ndarray
        Average constraint vector
    v : np.ndarray
        Possibly bounded v (if BOX=1)
    C_F : float
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

    M_F = 0.0
    ssq_F = 0.0
    C_F = 0.0
    max_F = 0.0
    avg_g = 0.0
    m = 0

    v = np.asarray(v, dtype=float).flatten()
    if BOX:
        v = np.clip(v, -1.0, 1.0)

    for m in range(1, m_max + 1):
        f, g = func((v - s0) / s1, consts)             # objective, constraints
        g = np.asarray(g, dtype=float).flatten()       # constraints as a vector
        F_A = f + penalty * np.sum(g * (g > tol_g))**q # augmented objective

        # Knuth's recursive update of a mean and a standard deviation 
        dF = F_A - M_F
        M_F += dF / m                 # update the mean of F
        ssq_F += dF * (F_A - M_F)     # update the sum of squares of F
        max_F = max(max_F, F_A)
        avg_g = avg_g + (g - avg_g) / m if m>1 else g      # update average constraint

        if m > 1:
            C_F = np.sqrt(ssq_F / (m - 1)) / np.abs(M_F)  # update the c.o.v. of F
            if m > 2 and m > (Za2 * C_F / err_F)**2:
                break

    F_risk = M_F
    if m > 1:
#       CHOOSE ONE OF THE FOLLOWING RISK-BASED PERFORMANCE MEASURES ...
#       F_risk = M_F                          # average-of-N values
#       F_risk = M_F * ( 1 + C_F/np.sqrt(m) ) # 84th percentile of the avg. of F
        F_risk = M_F * ( 1 + C_F )            # 84th percentile of F
#       F_risk = max_F;                       # largest-of-N values

    return F_risk, avg_g, v, C_F, m
