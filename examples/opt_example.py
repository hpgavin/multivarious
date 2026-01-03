#! /usr/bin/python3 -i 

import numpy as np

from collections import namedtuple

from multivarious.opt import ors
from multivarious.opt import nms
from multivarious.opt import sqp
from multivarious.utl import plot_cvg_hst

# Define the optimization problem. =========================================
def opt_example_analysis(v,C):
    '''
    A simple 2D constrained optimization test problem.
    '''
    v1 = v[0]                   # description of design variable "v1", units
    v2 = v[1]                   # description of design variable "v2", units

    a = C.a                     # description of constant a, units
    b = C.b                     # description of constant b, units
    c = C.c                     # description of constant c, units

    # the objective function 
    f = ( v1 - c[0] )**2 + ( v2 - c[1] )**2 +  c[2]*np.random.randn(1)

    # the array of constraints 
    g = np.array([
        a[0] * ( (v1 - a[1])**2 + (v2 - a[2])**2 )/a[3] - 1,  # "g1"
        1 - b[0]*( (v1 - b[1])**2 + (v2 - b[2])**2 )/b[3]     # "g2"
    ])

    f = f[0]                    # the design objective   - make f a scalar  
    g = np.array([g]).T         # the design constraints - make g a column vector

    return f, g

# Solve the optimization problem. ==========================================

# various constants used within the optimization analysis ... in a named tuple
a = np.array([  1.0 ,  0.2 ,  0.5 ,  0.3 ])
b = np.array([ -2.0 , -0.5 ,  0.5 , -1.5 ])
c = np.array([  0.8 ,  0.2 ,  0.0 ])

Constant = namedtuple('Constant', [ 'a', 'b', 'c' ])
C = Constant( a , b , c )

v_lb = np.array([ 0.0,  0.0])       # lower bound on the design variables 
v_ub = np.array([ 1.0,  1.0])       # upper bound on the design variables 

n = len(v_lb)                       # the number of design variables 

#v_init = v_lb + np.random.randn(n)*(v_ub - v_lb) # a  random   initial guess
v_init = np.array([ 0.8 , 0.8 ])                  # a specified initial guess 

# adjust the optimization options ...
# with ORS, convergence tolerances should be kind of large
# with NMA, convergence tolerances should be smaller than ORS

#  msg_level   tol_v    tol_f    tol_g    max_evals  pnlty  expn
opts = [ 3 ,   1e-2 ,   1e-2 ,   1e-3 ,   50*n**3 ,   0.5  , 0.5 ]

# solve the optimization problem using one of ... ors , nms , sqp 
v_opt, f_opt, g_opt, cvg_hst, _,_  = nms(opt_example_analysis, v_init, v_lb, v_ub, opts, C)

plot_cvg_hst( cvg_hst , v_opt )   # plot the convergence history
