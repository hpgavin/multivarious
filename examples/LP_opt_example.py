#! /usr/bin/env -S python3 -i

import numpy as np
from types import SimpleNamespace

from multivarious.opt import ors
from multivarious.opt import nms
from multivarious.opt import sqp
from multivarious.utl import plot_cvg_hst

# Define the optimization problem. ===============================
def LP_analysis( v , C ):
    '''
    [ f , g ] = LP_analysis( v , constants )
    analyze a trial solution v to any linear programming problem,
    minimize f = c' * v such that g = A * v - b <= 0
    constants A, b, and c are in a named tuple C

    Parameters
    ----------
    v : array-like (n,)
        design variables 
    C : any
        a SimpleNamespace containing [ A, b, c ]

    Returns
    -------
    f : float
        the design objective 
    g : array-like (m,)
        the design constraints 
    '''

    A = C.A             # constraint coefficient matrix (dimension m by n)
    b = C.b             # constraint vector (dimension m by 1)
    c = C.c             # cost coefficient vector (dimension n by 1)

    f = c @ v           # the cost function

    g = A @ v - b       # the constraint inequalities, compared to zero

    return f, g

# Set-up and Solve the optimization problem. ===============================

# Constants used within the optimization analysis ... 

C = SimpleNamespace()

C.A = np.array([ [ 2, 1, 1, 3 ] , 
                 [ 1, 3, 2, 1 ] , 
                 [ 1, 1, 4, 2 ] ]) 

C.b =  np.array([ 10, 15, 20 ]) 

C.c = -np.array([ 3, 5, 2, 4 ])       # minimize f(v) = c@v

(m,n) = C.A.shape

v_init = np.ones(n) 
v_lb   = np.zeros(n)
v_ub   = 10*np.ones(n)

# optimization options ...
#        0      1        2        3         4          5      6      7     8
#       msg    tol_v    tol_f    tol_g    max_evals  pnlty  expn  m_max  cov_F
opts = [ 1 ,   1e-2 ,   1e-2 ,   1e-3 ,    50*n**3 ,  0.5 ,  0.5 ,   1 ,  0.05 ]

# Solve the optimization problem using one of ... ors , nms , sqp 
v_opt, f_opt, g_opt, cvg_hst, _,_ = sqp(LP_analysis, v_init, v_lb, v_ub, opts, C)

# plot the convergence history
plot_cvg_hst( cvg_hst , v_opt , opts )
