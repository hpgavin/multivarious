# opt_options.py
# -----------------------------------------------------------------------------
# default optimization parameters for Multivarious optimization routines 
# -----------------------------------------------------------------------------

import numpy as np

def opt_options(options_in=None):
    """
    Return the default optimization parameters, optionally updating with user input.

    Parameters
    ----------
    options_in : list or np.ndarray, optional
        Custom user-defined options. Missing or zero values will be replaced by defaults.

    Returns
    -------
    options : np.ndarray
        Vector of optimization settings.
    """

    # Default parameters (same order and values as MATLAB)
    default_options = np.array([
        1,       # [0]  message level flag
        1e-3,    # [1]  tolerance on parameters
        1e-3,    # [2]  tolerance on cost
        0e-4,    # [3]  tolerance on constraints
        1000,    # [4]  max number of function evaluations
        10,      # [5]  penalty on constraint violations
        1,       # [6]  exponent on constraint violations
        1,       # [7]  number of function evaluations in average
        0.1,     # [8]  desired coefficient of variation on mean estimate
        0,       # [9]  stop when feasible
        0,       # [10] index for plotting surface
        1,       # [11] index for plotting surface
        25,      # [12] # of first-index values for plotting
        35,      # [13] # of second-index values for plotting
        1e-6,    # [14] finite diff minimum param. change
        2,       # [15] penalty type
        1e-6,    # [16] min param. change for finite diff gradients
        1e-1,    # [17] max param. change for finite diff gradients
        0        # [18] number of equality constraints
    ], dtype=float)

    # Initialize
    if options_in is None:
        return default_options.copy()

    options_in = np.abs(np.array(options_in, dtype=float))
    n = len(options_in)
    options = default_options.copy()
    options[:n] = options_in[:n]

    # Sanity checks and constraints
    if n > 5 and options_in[5] == 0:
        options[5] = 0
    options[0] = abs(round(options[0]))
    options[6] = np.clip(options[6], 0.001, 5)
    options[7] = max(options[7], 1)
    options[8] = abs(options[8])
    options[9] = abs(options[9])

    return options
