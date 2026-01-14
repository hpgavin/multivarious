#! /usr/bin/python3 -i 

import numpy as np
from types import SimpleNamespace

from multivarious.opt import ors
from multivarious.opt import nms
from multivarious.opt import sqp
from multivarious.utl import plot_cvg_hst

# Define the optimization problem. =========================================
def opt_example_analysis( v, C ):
    """
    Relate the design objective, f, and the design constraints, g, to 
    the design variables, v, and constants, C. 

    Parameters
    ----------
    v : array-like (n,)
        design varialbes (in original units, not scaled) 
    C : any
        collection of constants needed for the analysis 

    Returns
    -------
    f : float
        design objective value 
    g : arry-like (m,)
        design constraints
    """

    v1 = v[0]                   # description of design variable "v1", units
    v2 = v[1]                   # description of design variable "v2", units

    a = C.a                     # description of constant a, units
    b = C.b                     # description of constant b, units
    c = C.c                     # description of constant c, units

    # the design objective 
    f = ( v1 - c[0] )**2 + ( v2 - c[1] )**2 +  c[2]*np.random.randn(1)

    # the array of design constraints 
    g = np.array([
        a[0] * ( (v1 - a[1])**2 + (v2 - a[2])**2 )/a[3] - 1,  # "g1"
        1 - b[0]*( (v1 - b[1])**2 + (v2 - b[2])**2 )/b[3]     # "g2"
    ])

    return f[0], g              # end of opt_example_analysis()

# Set-up and Solve the optimization problem. ===============================

# Constants used within the optimization analysis ... 
C = SimpleNamespace()
C.a = np.array([  1.0,  0.2,  0.5,  0.3 ])
C.b = np.array([ -2.0, -0.5,  0.5, -1.5 ])
C.c = np.array([  0.8,  0.2,  0.00 ])

v_lb = np.array([ 0.0,  0.0])       # lower bound on the design variables 
v_ub = np.array([ 1.0,  1.0])       # upper bound on the design variables 

n = len(v_lb)                       # the number of design variables 

#v_init = v_lb + np.random.rand(n)*(v_ub - v_lb)   # a  random   initial guess
v_init = np.array([ 0.8 , 0.8 ])                  # a specified initial guess 

# optimization options ...
#        0      1        2        3         4          5      6      7     8
#       msg    tol_v    tol_f    tol_g    max_evals  pnlty  expn  m_max  cov_F
opts = [ 3 ,   2e-2 ,   2e-2 ,   1e-3 ,    50*n**3 ,  0.5 ,  0.5 ,   1 ,  0.05 ]

# Solve the optimization problem using one of ... ors , nms , sqp 
v_opt, f_opt, g_opt, cvg_hst, _,_  = ors(opt_example_analysis, v_init, v_lb, v_ub, opts, C)

# plot the convergence history
plot_cvg_hst( cvg_hst , v_opt, opts, pdf_plots=True )

