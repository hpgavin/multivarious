#! /usr/bin/env -S /usr/bin/python3 -i 

import numpy as np
from multivarious.opt import ors, nms, sqp
from multivarious.utl import StableNamespace, plot_cvg_hst, format_plot 
from rich.traceback import install; install()

# Define the optimization problem. =========================================
def opt_example_analysis( v, cts ):
    """
    Relate the design objective, f, and the design constraints, g, to 
    the design variables, v, and constants, cts. 

    Parameters
    ----------
    v : array-like (n,)
        design varialbes (in original units, not scaled) 
    cts : any
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

    a = cts.a                   # description of constant a, units
    b = cts.b                   # description of constant b, units
    c = cts.c                   # description of constant c, units

    # the design objective 
    f = ( v1 - c[1] )**2 + ( v2 - c[2] )**2 + c[0]*np.random.randn(1)

    # the array of design constraints 
    g = np.array([
       a[0] + a[3]*( v1 - a[1] )**2 + a[4]*( v2 - a[2] )**2 ,  # "g1"
       b[0] + b[3]*( v1 - b[1] )**2 + b[4]*( v2 - b[2] )**2    # "g2"
    ])

    return f, g       

def opt_example_analysis_HW01P10( v, cts ):
    """
    HW 1 , P 10
    """

    v1 = v[0]                   # description of design variable "v1", units
    v2 = v[1]                   # description of design variable "v2", units

    c = cts.c                   # description of constant c, units

    f = 2 + v1/c[0] + v2/c[1] + cos(v1*v2/c[2])  # HW 1, P 10

    g = np.array([ -1 ]);                        # HW 1, P 10 , no constraint

    return f, g       

# Set-up and Solve the optimization problem. ===============================

# Three examples to choose from:
#    example A: open constraints
#    example B: gauntlet constraints
#    example C: multiple minima

example = 'A' 

# example A is the default example 
cts = StableNamespace(
    a = [ -0.4,  0.2,  0.5,  1.4,  1.4 ], # open constraints
    b = [  1.0, -0.5,  0.5, -1.4, -1.4 ],
    c = [  0.0,  0.8,  0.2 ] )

fctn = opt_example_analysis

v_lb = np.array([ 0.0,  0.0])    # lower bound on the design variables 
v_ub = np.array([ 1.0,  1.0])    # upper bound on the design variables 

v_init = np.array([ 0.8 , 0.8 ]) # a specific initial guess 

if example == 'B': # gauntlet constraints
    fctn = opt_example_analysis
    cts = StableNamespace(
        a = [  0.7,  1.25,  0.5, -1.0, -1.0 ], # gauntlet constraints
        b = [  1.0, -0.5,   0.5, -1.4, -1.4 ],
        c = [  0.0,  0.8,   0.2 ] )

    v_lb = np.array([ 0.0,  0.0])    # lower bound on the design variables 
    v_ub = np.array([ 1.0,  1.0])    # upper bound on the design variables 

    v_init = np.array([ 0.3 , 0.9 ]) # a specific initial guess 

if example == 'C': # multiple minima
    fctn = opt_example_analysis_HW01P10
    cts = StableNamespace(
        c = [ 40, 30, 20 ] )
    v_lb = np.array([ -10.0, -10.0]) # lower bound on the design variables HW 1, P10
    v_ub = np.array([  10.0,  10.0]) # upper bound on the design variables HW 1, P10

    v_init = np.array([ 0.8, 0.4 ]) # a specific initial guess 

n = len(v_lb) # the number of design variables 

#v_init = v_lb + np.random.rand(n)*(v_ub - v_lb) # a random initial guess
v_init = v_init + 0.10*np.random.rand(n)*(v_ub - v_lb) # a random initial guess

# optimization hyperparameters ...
#       0     1       2      3       4         5      6     7      8
#      msg   tol_v   tol_f  tol_g  max_evals  pnlty  expn  m_max  cov_F
hyp = [ 3,   2e-2,   2e-2,  1e-3,   50*n**3,  0.7,   0.5,   1,    0.05 ]

# Solve the optimization problem using one of ... ors , nms , sqp 
# in opt_example_analysis select the "open constraint" or "gauntlet constraint"
v_opt, f_opt, g_opt, cvg_hst, _,_ = nms( fctn, v_init, v_lb, v_ub, hyp, cts )

# plot the convergence history
format_plot(font_size=15, line_width=3, marker_size=7)
plot_cvg_hst( cvg_hst , v_opt, hyp, save_plots = True ) 

