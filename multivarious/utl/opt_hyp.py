# opt_hyp.py
# -----------------------------------------------------------------------------
# default optimization hperparameters for Multivarious optimization routines 
# -----------------------------------------------------------------------------

import numpy as np

def opt_hyp(hyp_in=None):
    """
    Return the default optimization parameters, optionally updating with user input.

    Parameters
    ----------
    hyp_in : list or np.ndarray, optional
        Custom user-defined hyp. Missing or zero values will be replaced by defaults.

    Returns
    -------
    hyp : array-like
        Array of optimization hyperparameter settings.
    """

    # Default hyperparameter settings 
    default_hyp = np.array([
        1,       # [0]  msg message level flag
        1e-3,    # [1]  tol_v    tolerance on design variables 
        1e-3,    # [2]  tol_f    tolerance on design objective
        0e-4,    # [3]  tol_g    tolerance on constraints
        1000,    # [4]  maxEval  max number of function evaluations
        10,      # [5]  penalty  on constraint violations
        1,       # [6]  exponent on constraint violations
        1,       # [7]  mAvg     number of function evaluations in average
        0.1,     # [8]  desired coefficient of variation on mean estimate
        0,       # [9]  stop when feasible
        0,       # [10] index for plotting surface
        1,       # [11] index for plotting surface
        25,      # [12] # of first-index values for plotting
        35,      # [13] # of second-index values for plotting
        1e-6,    # [14] finite diff minimum change to design variables 
        2,       # [15] penalty type
        1e-6,    # [16] min change to design variables for finite diff gradients
        1e-1,    # [17] max change to design variables for finite diff gradients
        0        # [18] number of equality constraints
    ], dtype=float)

    # Initialize
    if hyp_in is None:
        return default_hyp.copy()

    hyp_in = np.abs(np.array(hyp_in, dtype=float))
    n = len(hyp_in)
    hyp = default_hyp.copy()
    hyp[:n] = hyp_in[:n]

    # Sanity checks and constraints
    if n > 5 and hyp_in[5] == 0:
        hyp[5] = 0
    hyp[0] = abs(round(hyp[0]))
    hyp[6] = np.clip(hyp[6], 0.001, 5)
    hyp[7] = max(hyp[7], 1)
    hyp[8] = abs(hyp[8])
    hyp[9] = abs(hyp[9])

    return hyp
