#! /usr/bin/python3 -i 

import numpy as np
from opt_options import opt_options
from ors import ors
from nms import nms
from sqp import sqp
from plot_cvg_hst import plot_cvg_hst
from collections import namedtuple

def opt_example_analysis(v,c):
    '''
    A simple 2D constrained optimization test problem.
    ''' 
    v1 = v[0]                   # description of design variable 0, units
    v2 = v[1]                   # description of design variable 1, units

    a = c.a                     # description of constant a, units
    b = c.b                     # description of constant b, units
    c = c.c                     # description of constant c, units

    # the objective function 
    f = ( v1 - c[0] )**2 + ( v2 - c[1] )**2 +  c[2]*np.random.randn(1)
    
    # the array of constraints 
    g = np.array([
        a[0] * ( (v1 - a[1])**2 + (v2 - a[2])**2 )/a[3] - 1,  # g1
        1 - b[0]*( (v1 - b[1])**2 + (v2 - b[2])**2 )/b[3]     # g2
    ])

    f = f[0]                    # make f a scalar
    g = np.array([g]).T         # make g a column vector
    
    return f, g

# -------------------------------------------------------------------------

# various constants used in the analysis ... in a named tuple
Constant = namedtuple('Constant', [ 'a', 'b', 'c' ])

a = np.array([  1.0 ,  0.2 ,  0.5 ,  0.3 ])
b = np.array([ -2.0 , -0.5 ,  0.5 , -1.5 ])
c = np.array([  0.8 ,  0.2 ,  0.0 ])

c = Constant( a , b , c )

v_lb = np.array([ 0.0,  0.0])       # lower bound
v_ub = np.array([ 1.0,  1.0])       # upper bound

n = len(v_lb)

# random initial guess or a specified initial guess 
#v_init = v_lb + np.random.randn(n)*(v_ub - v_lb) 
v_init = np.array([ 0.8 , 0.8 ])                

# adjust the optimization options ...
# with ORS, convergence tolerances should be kind of large
# with NMA, convergence tolerances should be smaller than ORS
#              msg_level tol_v  tol_f  tol_g  max_evals  pnlty  expn
opts = opt_options([ 3 , 1e-2 , 1e-2 , 1e-3 , 100*n**3 , 0.5  , 0.5 ])  # for ORS

# solve the optimization problem using one of ... ors , nms , sqp 
v_opt, f_opt, g_opt, cvg_hst, _,_  = ors(opt_example_analysis, v_init, v_lb, v_ub, opts, c)

# plot the convergence history
plot_cvg_hst( cvg_hst , v_opt )  

