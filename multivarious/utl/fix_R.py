import numpy as np
import matplotlib.pyplot as plt
from multivarious.opt import ors
from multivarious.opt import nms
from multivarious.opt import sqp
from multivarious.utl.plot_cvg_hst import plot_cvg_hst
from multivarious.utl.format_plot import format_plot
from types import SimpleNamespace

def fit_R(v, C):

    # extract constants from the SimpleNamespace C
    n = C.n
    upper_tri_idx = C.idx
    Ro = C.Ro
    eVal_lb = C.eVal_lb

    R = np.eye( n, n ) # Initialize an n x n identity matrix 

    # Place the values v into the upper triangle
    R[upper_tri_idx] = v

    # Mirror the upper triangle to the lower triangle to make symmetric
    R[(upper_tri_idx[1], upper_tri_idx[0])] = v

    eVal, eVec = np.linalg.eigh(R) 

    f = np.trace( (R - Ro) @ (R - Ro) ) # Frobeneous norm squared. 

    g = np.array([ eVal_lb - eVal[0] ]) 

    return f, g

def fix_R(Ro, eVal_lb):
    '''
    fix_R

    min || R(v) - Ro ||_F^2

    where v is the upper triangle (and lower triangle) of (symm) R

    s.t. -1 < v < 1  and  0 < eVal_lb < min(eVal(R(v)))

    INPUTS:
      Ro = initial guess for the correlation matrix (2D array)
      eVal_lb = lower bound on eigenvalue of R (int)
 
    OUTPUT:
      R = the fixed correlation matrix (2D array)

    # Transformation matrix
    T = eVec @ np.diag(np.sqrt(eVal))
    '''

    n = Ro.shape[0]

    plt.ion()
   
    # indices of the upper triangle (excluding the diagonal)
    upper_tri_idx = np.triu_indices(n, k=1)

    # Constants used within the optimization analysis 
    C = SimpleNamespace()
    C.n = n
    C.Ro = Ro
    C.idx = upper_tri_idx
    C.eVal_lb = eVal_lb

    # upper triangle elements into a 1D array
    v_init = Ro[upper_tri_idx]
    v_ub =  np.ones( int(n*(n-1)/2) )
    v_lb = -np.ones( int(n*(n-1)/2) )

    # optimization options ...
    #        0     1       2       3        4         5     6     7    8
    #       msg   tol_v   tol_f   tol_g  max_evals  pnlty expn  m_max cov_F
    opts = [ 1,   1e-3,   1e-3,   1e-3,    50*n**3,  10,   0.5,   1,  0.05 ]

    # Solve the optimization problem using one of ... ors , nms , sqp 
    v_opt, f_opt, g_opt, cvg_hst, _,_ = nms(fit_R, v_init, v_lb, v_ub, opts, C)

    # Build the correlation matrix from v_opt
    R = np.eye( n, n ) # Initialize an n x n identity matrix 

    # Place the values into the upper triangle
    R[upper_tri_idx] = v_opt

    # Mirror the upper triangle to the lower triangle to make symmetric
    R[(upper_tri_idx[1], upper_tri_idx[0])] = v_opt

    # plot the convergence history
    format_plot(font_size=15, line_width=3, marker_size=7)
    plot_cvg_hst( cvg_hst , v_opt, opts, save_plots=True )

    return R
