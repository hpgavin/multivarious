import numpy as np
from multivarious.opt import sqp
from multivarious.utl import format_plot
from multivarious.utl import plot_cvg_hst

def fit_R(v, C):

    n = C.n
    upper_tri_indices = C.idx
    Ro = C.Ro
    eVal_min = C.eVal

    R = np.eye((n, n)) # Initialize an n x n identity matrix 

    # Place the values into the upper triangle
    R[upper_tri_indices] = v

    # Mirror the upper triangle to the lower triangle to make symmetric
    R[(upper_tri_indices[1], upper_tri_indices[0])] = v

    eVal_0 = np.min ( np.eigh(R) )

    f = np.trace( (R - Ro)@(R-Ro) ) # Frobeneous norm trace ??

    g = np.array([ eVal_min - eVal_0 ]) # ??

    return f, g

def fix_R(Ro, eVal_min):
    '''
    fix_R

    min || Ro - R(v) ||_F^2

    v is the upper triangle (and lower triangle) of R

    s.t. -1 < v < 1  and  0 < eVal_min < min(eVal(R(v)))

    INPUTS:
      Ro = initial guess for the correlation matrix
      eVal_min ... lowest acceptable eigenvalue of R
 
    OUTPUT:
      R = the fixed correlation matrix

    # Transformation matrix
    T = eVec @ np.diag(np.sqrt(eVal))
    '''

    C = SimpleNamespace() # Constants used within the optimization analysis 

    n = Ro.shape[0]
   
    # indices of the upper triangle (excluding the diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)

    # upper triangle elements into a 1D array
    v_init = Ro[upper_tri_indices]
    v_ub =  np.ones( n*(n-1)/2 )
    v_lb = -np.ones( n*(n-1)/2 )

    C.n = n
    C.Ro = Ro
    C.idx = uppoer_triangle_indices
    C.eVal = eVal_min

    # optimization options ...
    #        0     1       2       3        4         5     6     7    8
    #       msg   tol_v   tol_f   tol_g  max_evals  pnlty expn  m_max cov_F
    opts = [ 1,   0.01,   0.01,   1e-3,    50*n**3,  0.7,  0.5,   1,  0.05 ]

    # Solve the optimization problem using one of ... ors , nms , sqp 
    v_opt, f_opt, g_opt, cvg_hst, _,_ = sqp(fit_R, v_init, v_lb, v_ub, opts, C)

    # Build the correlation matrix from v_opt
    R = np.eye((n, n)) # Initialize an n x n identity matrix 

    # Place the values into the upper triangle
    R[upper_tri_indices] = v_opt

    # Mirror the upper triangle to the lower triangle to make symmetric
    R[(upper_tri_indices[1], upper_tri_indices[0])] = v_opt

    # plot the convergence history
    format_plot(font_size=15, line_width=3, marker_size=7)
    plot_cvg_hst( cvg_hst , v_opt, opts, save_plots=True )

    return R
