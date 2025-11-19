import numpy as np
from opt_options import opt_options
from box_constraint import box_constraint
from avg_cov_func import avg_cov_func
from ors import ors
from plot_cvg_hst import plot_cvg_hst
from plot_opt_surface import plot_opt_surface
from collections import namedtuple

def opt_example_analysis(v,c):

    v1 = v[0]                   # description of design variable 0, units
    v2 = v[1]                   # description of design variable 1, units

    a = c.a                     # description of constant a, units
    b = c.b                     # description of constant b, units
    c = c.c                     # description of constant c, units

    # the objective function 
    f = ( v1 - c[0] )**2 + ( v2 - c[1] )**2 +  c[2]*np.random.randn(1)
    
    # the array of constraints 
    g = np.array([     a[0]*( (v1 - a[1])**2 + (v2 - a[2])**2 )/a[3] - 1 ,  # g1
                   1 - b[0]*( (v1 - b[1])**2 + (v2 - b[2])**2 )/b[3]     ]) # g2

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

v_lb = np.array([ 0.0,  0.0])                    # lower bound
v_ub = np.array([ 1.0,  1.0])                    # upper bound

n = len(v_lb)

v_init = v_lb + np.random.randn(n)*(v_ub - v_lb) # random initial guess, or 
v_init = np.array([ 0.8 , 0.8 ])                 # a specified initial guess

# adjust the optimization options ...
# with ORS, convergence tolerances should be kind of loose
#              msg_level tol_v  tol_f  tol_g  max_evals
opts = opt_options([ 3 , 1e-1 , 1e-1 , 1e-1 , 200 ])  # for ORS

# solve the optimization problem
v_opt, f_opt, g_opt, cvg_hst = ors(opt_example_analysis, v_init, v_lb, v_ub, opts, c)

plot_cvg_hst( cvg_hst , v_opt )                  # plot the convergence history
