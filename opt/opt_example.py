import numpy as np
from optim_options import optim_options
from box_constraint import box_constraint
from avg_cov_func import avg_cov_func
from ors_opt import ors_opt
from collections import namedtuple

def optim_example_analysis(v,c):

    v0 = v[0]           # description of design variable 0, units
    v1 = v[1]           # description of design variable 1, units

    a = c.a             # description of constant a, units
    b = c.b             # description of constant b, units
    c = c.c             # description of constant c, units

    f = ( v0 - c[0] )**2 + ( v1 - c[1] )**2 +  c[2]*np.random.randn(1) # the objective function
    
    g = np.array([     a[0]*( (v0 - a[1])**2 + (v1 - a[2])**2 )/a[3] - 1 ,  # constraint g1
                   1 - b[0]*( (v0 - b[1])**2 + (v1 - b[2])**2 )/b[3]     ]) # constraint g2

    g = [g]

    return f[0], g

# -------------------------------------------------------------------------

# a set of constants used in the analysis
a = np.array([  1.0 ,  0.2 ,  0.5 ,  0.3 ])
b = np.array([ -2.0 , -0.5 ,  0.5 , -1.5 ])
c = np.array([  0.8 ,  0.2 ,  0.0 ])

Constant = namedtuple('Constant', [ 'a', 'b', 'c' ])
c = Constant( a , b , c )

v_lb = np.array([ 0.0,  0.0])
v_ub = np.array([ 1.0,  1.0])

n = len(v_lb)

v_init = v_lb + np.random.randn(n)*(v_ub - v_lb) # random initial guess, or 
v_init = np.array([ 0.8 , 0.8 ])                 # a specified initial guess

print(v_init)

#                  msgLvel tolX   tolF   tolG   maxEvals
opts = optim_options([ 1,  1e-4 , 1e-4 , 1e-4 , 1e3 ])  

v_opt, f_opt, g_opt, hist = ors_opt(optim_example_analysis, v_init, v_lb, v_ub, opts, c)

