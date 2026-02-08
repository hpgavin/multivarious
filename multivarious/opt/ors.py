"""
ors.py 
-----------------------------------------------------------------------------
Optimized Step Size Randomized Search Algorithm for Nonlinear Optimization
Depends on: opt_options(), avg_cov_func(), plot_opt_surface()
-----------------------------------------------------------------------------

Nonlinear optimization with inequality constraints using Random Search
with optimized step sizes based on quadratic approximations.

Minimizes f(v) such that g(v) < 0 and v_lb <= v_opt <= v_ub.
- f is a scalar objective function
- v is a vector of design variables
- g is a vector of inequality constraints

Reference:
Sheela Belur V, "An Optimized Step Size Random Search",
Computer Methods in Applied Mechanics & Eng'g, Vol 19, 99-106, 1979

S.Rao, Optimization Theory and Applications, 2nd ed, John Wiley, 1984

H.P. Gavin, Civil & Environmental Eng'g, Duke Univ.
Translation from MATLAB to Python, 2025-11-24

updated 2011-04-13, 2014-01-12, 2015-03-14, (pi day 03.14.15), 2015-03-26, 
2016-04-06, 2019-02-23, 2020-01-17, 2024-04-03, 2025-11-24
"""


import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from numpy.linalg import norm
from numpy.linalg import solve

from multivarious.utl.avg_cov_func import avg_cov_func
from multivarious.utl.box_constraint import box_constraint
from multivarious.utl.plot_opt_surface import plot_opt_surface  
from multivarious.utl.opt_options import opt_options
from multivarious.utl.opt_report import opt_report 

def ors(func, v_init, v_lb=None, v_ub=None, options=None, consts=None):
    """
    Optimized Random Search with inequality constraints.
    
    Parameters
    ----------
    func : callable
        Function to optimize with signature: f, g = func(v, consts)
        Returns objective value (float) and constraint values (array)
    v_init : ndarray, shape (n,)
        Initial design variable values
    v_lb : ndarray, shape (n,), optional
        Lower bounds on design variables (default: -100*|v_init|)
    v_ub : ndarray, shape (n,), optional
        Upper bounds on design variables (default: +100*|v_init|)
    options : ndarray or list, optional
        Optimization settings (see opt_options.py for details):
        options[0] = message level (0=quiet, 1+=verbose)
        options[1] = tol_v - tolerance on design variables
        options[2] = tol_f - tolerance on objective function
        options[3] = tol_g - tolerance on constraints
        options[4] = max_evals - max function evaluations
        options[5] = penalty - penalty on constraint violations
        options[6] = q - exponent on constraint violations
        options[7] = m_max - num. function evals in mean estimate
        options[8] = err_F - desired coefficient of variation on mean
        options[9] = find_feas - stop when solution is feasible (1) or not (0)
        options[10:13] = surface plotting parameters
    consts : optional
        Additional constants passed to func(v, consts)
    
    Returns
    -------
    v_opt : ndarray
        Optimal design variables
    f_opt : float
        Optimal objective function value
    g_opt : ndarray
        Constraint values at optimum
    cvg_hst : ndarray
        Convergence history: [v; f; max(g); function_evals; cvg_v; cvg_f]
    iteration : int
        Number of iterations completed
    function_evals : int
        Total number of function evaluations
    """
    
    # algorithm hyper-parameters
    BOX = 1             # use box constraints
    step_stdev = 0.200  # standard deviation of random step
    nu = 2.5            # exponent for reducing step_stdev
    regularization = 1e-6 * np.eye(3) # regularization for matrix inversion
    
    # handle missing arguments
    v_init = np.asarray(v_init, dtype=float).flatten()
    n = len(v_init)
    
    if v_lb is None:
        v_lb = -1.0e2 * np.abs(v_init)
    if v_ub is None:
        v_ub = 1.0e2 * np.abs(v_init)
    
    v_lb = np.asarray(v_lb, dtype=float).flatten()
    v_ub = np.asarray(v_ub, dtype=float).flatten()
    
    if options is None:
        options = opt_options()
    else:
        options = opt_options(options)
    
    if consts is None:
        consts = 1.0
    
    # extract options
    msg   = int(options[0])
    tol_v = options[1]
    tol_f = options[2]
    tol_g = options[3]
    max_evals = int(options[4])
    penalty = options[5]
    q = options[6]
    find_feas = int(options[9])
    
    # check that uounds are valid
    if np.any(v_ub < v_lb):
        print('Error: v_ub cannot be less than v_lb for any variable')
        return v_init, (np.sqrt(5)-1)/2, np.array([1.0]), None, 0, 0
    
    # initialize
    function_evals = iteration = 0
    cvg_f = 1.0
    cvg_hst = np.full((n + 5, max_evals), np.nan)
    f = np.zeros(4)
    
    feasible = converged = stalled = False # convergence criteria

    # scale v from bounds [v_lb, v_ub] to u in bounds [-1, +1]
    s0 = (v_lb + v_ub) / 2.0  # average of v_lb and v_ub
    s1 = (v_ub - v_lb) / 2.0  # half-distance from v_lb to v_ub
    u_init = (v_init - s0) / s1
    u_init = np.clip(u_init, -1.0, 1.0) # keep u_init in bounds, just in case
    
    # analyze the initial guess
    u0 = u_init.copy() # use .copy() to keep changes in u0 from changing u_init
    f0, g0, u0, c0, nAvg = avg_cov_func(func, u0, s0, s1, options, consts, BOX)
    function_evals += nAvg
    
    # check dimensions
    if np.prod(np.shape(f0)) != 1:
        raise ValueError('Objective must be a scalar')
    if g0.ndim == 2 and g0.shape[0] == 1:
        raise ValueError('Constraints must be a column vector')
    
    f[0] = f0 
    f[3] = f0
    u3 = u0.copy() # use .copy() to keep changes in u3 from changing u0
    g3 = g0.copy() # use .copy() to keep changes in g3 from changing g0
    
    m = len(g0)  # number of constraints
   
    if msg:
        start_time = time.time()

    # plot objective surface
    if msg > 2:
        f_min, f_max, ax = plot_opt_surface(
            func, v_init, v_lb, v_ub, options, consts, 1003)
     
    # initialize optimal values
    f_opt = f[0]
    u_opt = u0.copy() # use .copy() to keep changes in u0 from changing u_opt
    g_opt = g0.copy() # use .copy() to keep changes in g0 from changing g_opt
    v_opt = s0 + s1*u_opt # scale (u) back to original units (v)
    
    # save the initial guess to the convergence history
    cvg_hst[:, iteration] = np.concatenate([
        s0 + s1*u_opt, [f_opt], [np.max(g_opt)], 
        [function_evals], [step_stdev], [1.0]
    ])
    
    if msg:
        print()  # clear screen effect
    
    last_update = function_evals
    
    # ========== main optimization loop ==========
    while function_evals < max_evals:
        
        # a random search perturbation with standard deviation 'step_stdev'
        r = step_stdev * np.random.randn(n)
        r1 = r / norm(r) # unit vector along r
        
        # 1st perturbation: +1*r : "random single step"
        aa, _ = box_constraint(u0, r) # keep u1 within bounds
        u1 = u0 + aa * r
        d1 = norm(u1 - u0) 
        
        f[1], g1, u1, c1, nAvg = avg_cov_func(func, u1, s0, s1, options, consts, BOX)
        function_evals += nAvg
        
        # is f[1] downhill from f[0]?
        downhill = np.sign(f[0] - f[1]) # +1: yes, -1: no
        
        # 2nd perturbation: 2*downhill*r : "downhill double-step"
        aa, bb = box_constraint(u0, 2*d1*r1) # keep u2 within bounds
        if downhill > 0:
            u2 = u0 + aa * 2*d1*r1
        else:
            u2 = u0 + bb * 2*d1*r1 * downhill
        d2 = norm(u2 - u0) * downhill 

        f[2], g2, u2, c2, nAvg = avg_cov_func(func, u2, s0, s1, options, consts, BOX)
        function_evals += nAvg
        
        # 3rd perturbation : try quadratic update if curvature is positive
        # signed distances from (u0 to u1) and from (u0 to u2) for quadratic fit
        
        # fit quadratic: f(d) = c[0] + c[1]*d + c[2]*d^2 
        D = np.array([
            [1,  0,     0 ],
            [1, d1, d1**2 ],
            [1, d2, d2**2 ]
        ]) + regularization
        
        c = solve(D, f[0:3])
        
        quad_update = False
        if c[2] > 0:            # positive curvature ... so look for a minimum
            d3 = -c[1]/(2*c[2]) # d3 = distance from u0 to the zero-slope point
            aa, bb = box_constraint(u0, d3*r1) # keep u3 within bounds
            if d3 > 0:
                u3 = u0 + aa * d3*r1
            else:
                u3 = u0 + bb * d3*r1 * downhill
            
            f[3], g3, u3, c3, nAvg = avg_cov_func(func, u3, s0, s1, options, consts, BOX)
            function_evals += nAvg

        # save function values and variable values in their original units 
        f0 , v0 = f[0] , s0 + s1*u0 
        f1 , v1 = f[1] , s0 + s1*u1 
        f2 , v2 = f[2] , s0 + s1*u2 
        f3 , v3 = f[3] , s0 + s1*u3 

        # find the best (min(f)) objective value of the 4 evaluations in f
        i_min = np.argmin(f)

        # adaptive update to the step size standard deviation 
        if i_min > 0: # one or more of u1, u2, u3 is a better point
            step_txt = '          none'
            FA = (f2-f0)/(f1-f0)
            FB = (f0-f2)/(f1-f0)
            FC = (f2-f1)/(f0-f1)
            FD = (f1-f2)/(f0-f1)
            if f1 > f0:                 # step 1 is uphill
                if FA > 1:              # (f2-f0) > (f1-f0) reduce step
                    step_stdev /= FA
                    step_txt = '        uphill reduction'
                if FB > 1:              # (f0-f2) > (f1-f0) extend step
                    step_stdev *= FB
                    step_txt = '        uphill extension'
            if f1 < f0:                 # step 1 is downhill
                if FC > 1:              # (f2-f1) > (f0-f1) reduce step
                    step_stdev /= FC
                    step_txt = '      downhill reduction'
                if FD > 1:              # (f1-f2) > (f0-f1) extend step
                    step_stdev *= FD
                    step_txt = '      downhill extension'
        else: # u1, u2, and u3 are all worse points
            step_stdev *= 0.8
            step_txt = '             contraction'

        '''
        # scripted update to the step size (alternative to adaptive update)
        # if the solution improved, reduce the scope of the search 
        if i_min > 0:
            step_stdev = step_stdev * (1 - function_evals / max_evals) ** nu
        '''
        step_stdev = min(max(step_stdev,2*tol_v),0.2) # bound the step size

        # update the best point out of the four trials
        if i_min == 1:
            u0, g0, c0 = u1, g1, c1
        elif i_min == 2:
            u0, g0, c0 = u2, g2, c2
        elif i_min == 3:
            u0, g0, c0 = u3, g3, c3
            quad_update = True
        
        u0 = np.clip(u0, -1.0, +1.0)  # keep u0 within bounds, just to be sure
        f[0] = f[i_min]
        
        # update optimal solution if improved
        if f[0] < f_opt:
            # use .copy() to keep changes to u0 from changing u_opt!
            u_opt = u0.copy()
            f_opt = f[0].copy() # .copy() not needed since f_opt is immutable
            g_opt = g0.copy()
            v_opt = s0 + s1*u_opt # scale (u) back to original units (v)
            
            # Convergence metrics
            cvg_v, cvg_f, max_g = cvg_metrics(cvg_hst, v_opt, f_opt, g_opt, iteration)

            iteration += 1
            cvg_hst[:, iteration] = np.concatenate([
                v_opt, [f_opt], [max_g], [function_evals], [cvg_v], [cvg_f]
            ])
            last_update = function_evals
            
            # Display progress for this iteration
            if msg:
                elapsed = time.time() - start_time
                secs_left = int((max_evals - function_evals) * elapsed / function_evals)
                eta = (datetime.now() + timedelta(seconds=secs_left)).strftime('%H:%M:%S')
                
                idx_max_g = np.argmax(g_opt) # the index of the largest constraint
                
                #print('\033[H\033[J', end='')  # clear screen
                print('\n +-+-+-+-+-+-+-+-+-+-+- ORS -+-+-+-+-+-+-+-+-+-+-+-+-+')
                print(f' iteration               = {iteration:5d}', end='')
                if max_g > tol_g:
                    print('     !!! infeasible !!!')
                else:
                    print('     ***  feasible  ***')
                print(f' function evaluations    = {function_evals:5d}  of  {max_evals:5d}  '
                      f'({100*function_evals/max_evals:4.1f}%)')
                print(f' e.t.a.                  = {eta}')
                if n < 15:
                    print(f' variables               = ', end='')
                    for val in v_opt:
                        print(f'{val:11.3e}', end='')
                print()
                print(f' objective               = {f_opt:11.3e}')
                print(f" constraint              = {np.max(g_opt):11.4e}    tol_g = {tol_g:8.6f}")
                print(f' variable  convergence   = {cvg_v:11.4e}    tol_v = {tol_v:8.6f}')
                print(f' objective convergence   = {cvg_f:11.4e}    tol_f = {tol_f:8.6f}')
                print(f' c.o.v. of F_A           = {c0:11.3e}')
                print(f' step std.dev            = {step_stdev:7.3f}{step_txt}')
                print(' +-+-+-+-+-+-+-+-+-+-+- ORS -+-+-+-+-+-+-+-+-+-+-+-+-+')
                if quad_update:
                    print(' successful quadratic update')
        
            # plot on surface for this iteration
            if msg > 2:
           
                plt.figure(1003)
                ii = int(options[10])
                jj = int(options[11])
            
                if downhill > 0:
                    plt.plot([ v0[ii], v1[ii], v2[ii] ], 
                             [ v0[jj], v1[jj], v2[jj] ],
                             [ f0,     f1,     f2 ], '-or', markersize=4) 
                else:
                    plt.plot([v1[ii], v0[ii], v2[ii] ], 
                             [v1[jj], v0[jj], v2[jj] ],
                             [f1,     f0,     f2 ], '-or', markersize=4) 
                if quad_update:
                    plt.plot([v3[ii]], [v3[jj]], f[3], 
                            'ob', markersize=9, linewidth=3)
                
                plt.draw()
                plt.pause(0.10)

        # ----- Termination checks -----
        # check for feasibility of constraints 
        if np.max(g_opt) < tol_g and find_feas:                   # :)
            feasible = True
        # check for convergence in variables and objective 
        if iteration > n*n and (cvg_v < tol_v and cvg_f < tol_f): # :)
            converged = True 
        # check for stalled computations
        if function_evals - last_update > 0.20*max_evals:         # :(
            stalled = True   

        if stalled or (step_stdev <= 2*tol_v and (feasible or converged)):
            break 

    # ========== main optimization loop ==========

    # trim the convergence history
    cvg_hst = cvg_hst[:, 0:iteration+1]

    # plot the converged point
    if msg > 2:
        plt.figure(1003)
        ii = int(options[10])
        jj = int(options[11])
        plt.plot( v_opt[ii], v_opt[jj], f_opt, '-or', markersize=14 )
        plt.draw()
        plt.pause(0.10)
 
    # final report
    if msg:
        lambda_qp = None
        opt_report(v_init, v_opt, f_opt, g_opt, v_lb, v_ub,
                   tol_v, tol_f, tol_g,
                   lambda_qp, start_time, function_evals, max_evals, 
                   find_feas, feasible, converged, stalled )

    return v_opt, f_opt, g_opt, cvg_hst, iteration, function_evals

def cvg_metrics(cvg_hst, v, f, g, iteration ):
    '''
    Compute convergence metrics for ors which are defined as: 
    the ratio of (the difference between the current variables and the most recent iteration variables)
    to (the average of the current variables and the most recent iteration variables)
    in terms of the variables and the objective function, f. 
    
    Parameters
    ----------    
    cvg_hst ndarray
       convergence history 
    v array
       design variables
    f float
       objective function
    g array
       constraints 
    iteration int
       current iteration number 

    Returns
    -------
      cvg_v float
        convergence metric for design variables
      cvg_f float
        convergence metric for design objective
      max_g float
        the maximum of the design constraints
    '''

    from numpy.linalg import norm
    from numpy import max

    n = len(v) # number of design variabls 

    cvg_v = 2 * norm(cvg_hst[0:n, iteration] - v) / (norm(cvg_hst[0:n, iteration] + v)+1e-9)
    cvg_f = 2 * norm(cvg_hst[  n, iteration] - f) / (norm(cvg_hst[  n, iteration] + f)+1e-9)
    max_g = max(g)     

    return cvg_v, cvg_f, max_g

