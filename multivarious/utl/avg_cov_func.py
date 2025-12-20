# avg_cov_func.py
# -----------------------------------------------------------------------------
# Translated from MATLAB's avg_cov_func.m
# Computes the risk-adjusted (penalized) average cost and coefficient of variation.
# -----------------------------------------------------------------------------

import numpy as np

def avg_cov_func(func, x, s0, s1, options, consts=None, BOX=1):
    """
    Compute the average and coefficient of variation of a penalized cost function.

    Parameters
    ----------
    func : callable
        Function to optimize: f, g = func(x, consts)
    x : np.ndarray
        Parameter vector (column-like)
    s0, s1 : np.ndarray or float
        Linear scaling factors mapping [x_lb, x_ub] -> [-1, +1]
    options : np.ndarray
        Optimization settings vector (see opt_options)
    consts : np.ndarray, optional
        Additional constants (non-design variables)
    BOX : int, optional
        1 to bound x within [-1, 1], 0 to allow unbounded (default=1)

    Returns
    -------
    F_risk : float
        Risk-adjusted average cost (84th percentile of mean)
    g_avg : np.ndarray
        Average constraint vector
    x : np.ndarray
        Possibly bounded x (if BOX=1)
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

    avg_F = 0.0
    ssq_F = 0.0
    cov_F = 0.0
    max_F = 0.0
    g_avg = 0.0
    m = 0

    x = np.asarray(x, dtype=float).flatten()
    if BOX:
        x = np.clip(x, -1.0, 1.0)

    for m in range(1, m_max + 1):
        f, g = func((x - s0) / s1, consts)
        g = np.asarray(g, dtype=float).flatten()
        F_aug = f + penalty * np.sum(g * (g > tol_g))**q

        dF = F_aug - avg_F
        avg_F += dF / m
        ssq_F += dF * (F_aug - avg_F)
        g_avg = g_avg + (g - g_avg) / m if m > 1 else g
        max_F = max(max_F, F_aug)

        if m > 1:
            cov_F = np.sqrt(ssq_F / (m - 1)) / avg_F
            if m > 2 and m > (Za2 * cov_F / err_F)**2:
                break

    F_risk = avg_F
    if m > 1:
        F_risk = avg_F * (1 + cov_F / np.sqrt(m))

    return F_risk, g_avg, x, cov_F, m
