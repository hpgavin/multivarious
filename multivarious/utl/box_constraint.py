# box_constraint.py
# -----------------------------------------------------------------------------
# Translated from MATLAB's box_constraint.m
# Determines box-scaling factors (aa, bb) for feasible perturbations.
# H.P.Gavin: 2015-03-14  (pi day 3.14.15)
# -----------------------------------------------------------------------------

import numpy as np

def box_constraint(u, r):
    '''
    Determine box constraint scaling factors (aa, bb) such that:
        max(u + aa*r) < +1 and min(u + aa*r) > -1
        max(u - bb*r) < +1 and min(u - bb*r) > -1
        aa>0 and bb>0

    Parameters
    ----------
    u : np.ndarray
        Current point (n,).  in scaled variables ... -1 < u < +1
    r : np.ndarray
        Random perturbation vector (n,).

    Returns
    -------
    aa : float
        Maximum feasible positive step size.  ( 0 < aa < +1)
    bb : float
        Maximum feasible negative step size.  (-1 < bb <  0)
    '''
    u = np.asarray(u).flatten()
    r = np.asarray(r).flatten()
    n = len(u)

    a_vals =  np.ones(n)
    b_vals = -np.ones(n)
    I = np.eye(n)

    R = 1e-6 * np.eye(n)  # a little regularization

    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        Ii = I.copy()
        Ii[:, i] = -r
        vab = np.linalg.solve((Ii+R), np.column_stack(((u - ei), (u + ei))))
        aa_i, bb_i = vab[i, 0], vab[i, 1]
        if aa_i > 0:
            a_vals[i], b_vals[i] = aa_i, bb_i
        else:
            a_vals[i], b_vals[i] = bb_i, aa_i

    aa =  np.min([np.min(a_vals), +1.0])
    bb = -np.max([np.max(b_vals), -1.0])

    return aa, bb
