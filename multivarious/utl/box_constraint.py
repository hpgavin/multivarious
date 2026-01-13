# box_constraint.py
# -----------------------------------------------------------------------------
# Translated from MATLAB's box_constraint.m
# Determines box-scaling factors (aa, bb) for feasible perturbations.
# -----------------------------------------------------------------------------

import numpy as np

def box_constraint(x, r):
    '''
    Determine box constraint scaling factors (aa, bb) such that:
        max(x + aa*r) < +1 and min(x + aa*r) > -1
        max(x - bb*r) < +1 and min(x - bb*r) > -1

    Parameters
    ----------
    x : np.ndarray
        Current point (n,).
    r : np.ndarray
        Random perturbation vector (n,).

    Returns
    -------
    aa : float
        Maximum feasible positive step size.
    bb : float
        Maximum feasible negative step size.
    '''
    x = np.asarray(x).flatten()
    r = np.asarray(r).flatten()
    n = len(x)

    a_vals =  np.ones(n)
    b_vals = -np.ones(n)
    I = np.eye(n)

    R = 1e-6 * np.eye(n)  # regularization

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        Ii = I.copy()
        Ii[:, i] = -r
        vab = np.linalg.solve((Ii+R), np.column_stack(((x - ei), (x + ei))))
        aa_i, bb_i = vab[i, 0], vab[i, 1]
        if aa_i > 0:
            a_vals[i], b_vals[i] = aa_i, bb_i
        else:
            a_vals[i], b_vals[i] = bb_i, aa_i

    aa =  np.min([np.min(a_vals), +1.0])
    bb = -np.max([np.max(b_vals), -1.0])

    return aa, bb
