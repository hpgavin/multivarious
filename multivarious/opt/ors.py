"""
ors.py - Optimized Step Size Randomized Search
===============================================

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

Translation from MATLAB to Python, 2025-11-24
Original by H.P. Gavin, Civil & Environmental Eng'g, Duke Univ.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from numpy.linalg import norm
from numpy.linalg import solve

from multivarious.utl.avg_cov_func import avg_cov_func
from multivarious.utl.box_constraint import box_constraint
from multivarious.utl.opt_options import opt_options
from multivarious.utl.plot_opt_surface import plot_opt_surface  # ??

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
        Convergence history: [v; f; max(g); func_count; cvg_v; cvg_f]
    iteration : int
        Number of iterations completed
    function_count : int
        Total number of function evaluations
    """
    
    # algorithm hyper-parameters
    BOX = 1             # use box constraints
    step_stdev = 0.200  # standard deviation of random step, S
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
    
    # check that bounds are valid
    if np.any(v_ub < v_lb):
        print('Error: v_ub cannot be less than v_lb for any variable')
        return v_init, (np.sqrt(5)-1)/2, np.array([1.0]), None, 0, 0
    
    # initialize
    function_count = iteration = 0
    cvg_f = 1.0
    cvg_hst = np.full((n + 5, max_evals), np.nan)
    fa = np.zeros(4)
    
    # scale v from bounds [v_lb, v_ub] to x in bounds [-1, +1]
    s0 = (v_lb + v_ub) / (v_lb - v_ub)
    s1 = 2.0 / (v_ub - v_lb)
    x_init = s0 + s1 * v_init
    x_init = np.clip(x_init, -1.0, 1.0) # keep x_init in bounds, just in case
    
    # analyze the initial guess
    x0 = x_init.copy() # use .copy() to keep changes in x0 from changing x_init
    f0, g0, x0, c0, nAvg = avg_cov_func(func, x0, s0, s1, options, consts, BOX)
    function_count += nAvg
    
    # check dimensions
    if np.prod(np.shape(f0)) != 1:
        raise ValueError('Objective must be a scalar')
    if g0.ndim == 2 and g0.shape[0] == 1:
        raise ValueError('Constraints must be a column vector')
    
    fa[0] = f0 
    fa[3] = f0
    x3 = x0.copy() # use .copy() to keep changes in x3 from changing x0
    g3 = g0.copy() # use .copy() to keep changes in g3 from changing g0
    
    m = len(g0)  # number of constraints
   
    if msg:
        start_time = time.time()

    # plot objective surface
    if msg > 2:
        f_min, f_max, ax = plot_opt_surface(
            func, v_init, v_lb, v_ub, options, consts, 1003)
     
    # initialize optimal values
    f_opt = fa[0]
    x_opt = x0.copy() # use .copy() to keep changes in x0 from changing x_opt
    g_opt = g0.copy() # use .copy() to keep changes in g0 from changing g_opt
    
    # save the initial guess to the convergence history
    cvg_hst[:, iteration] = np.concatenate([
        (x_opt - s0) / s1, [f_opt], [np.max(g_opt)], 
        [function_count], [step_stdev], [1.0]
    ])
    
    if msg:
        print()  # clear screen effect
    
    last_update = function_count
    
    # ========== main optimization loop ==========
    while function_count < max_evals:
        
        # a random search perturbation with standard deviation 'step_stdev'
        sr = step_stdev * np.random.randn(n)
        
        # 1st perturbation: +1*r : "random single step"
        aa, _ = box_constraint(x0, sr) # keep x1 within bounds
        x1 = x0 + aa * sr
        
        fa[1], g1, x1, c0, nAvg = avg_cov_func(func, x1, s0, s1, options, consts, BOX)
        function_count += nAvg
        
        # is fa[1] downhill from fa[0]?
        downhill = np.sign(fa[0] - fa[1]) # +1: yes, -1: no
        
        # 2nd perturbation: 2*downhill*s*r : "downhill double step"
        aa, bb = box_constraint(x0, 2*downhill*sr) # keep x2 within bounds
        if downhill > 0:
            x2 = x0 + aa * 2*downhill * sr
        else:
            x2 = x0 + bb * 2*downhill * sr
        
        fa[2], g2, x2, c2, nAvg = avg_cov_func(func, x2, s0, s1, options, consts, BOX)
        function_count += nAvg
        
        # distances from (x0 to x1) and from (x0 to x2) for quadratic fit
        dx1 = norm(x1 - x0) / norm(sr)
        dx2 = norm(x2 - x0) / norm(sr)
        
        # fit quadratic: f(d) = 0.5*a*d^2 + b*d + c
        A = np.array([
            [0,           0,    1],
            [0.5*dx1**2, dx1,   1],
            [0.5*dx2**2, dx2,   1]
        ]) + regularization
        
        abc = solve(A, fa[0:3])
        a, b, c = abc[0], abc[1], abc[2]
        
        # 3rd perturbation : try quadratic update if curvature is positive
        quad_update = False
        if a > 0:       # positive curvature ... so look for a minimum
            d = -b / a  # d*r is the distance from x0 to the zero-slope point
            aa, bb = box_constraint(x0, d * sr) # keep x3 within bounds
            if d > 0:
                x3 = x0 + aa * d * sr
            else:
                x3 = x0 + bb * d * sr
            
            fa[3], g3, x3, c3, nAvg = avg_cov_func(func, x3, s0, s1, options, consts, BOX)
            function_count += nAvg

        # save values of variables and functions for plotting (msg > 2)
        v0 , f0 = (x0 - s0) / s1 , fa[0]
        v1 , f1 = (x1 - s0) / s1 , fa[1]
        v2 , f2 = (x2 - s0) / s1 , fa[2]
        v3 , f3 = (x3 - s0) / s1 , fa[3]

        i_min = np.argmin(fa)

        # adaptive step size
        if i_min > 0:
            step_txt = '          none'
            FA = (f2-f0)/(f1-f0)
            FB = (f0-f2)/(f1-f0)
            FC = (f2-f1)/(f0-f1)
            FD = (f1-f2)/(f0-f1)
            if f1 > f0:                 # step 1 is uphill
                if FA > 1:              # (f2-f0) > (f1-f0) reduce step
                   step_stdev /= FA
                   step_txt = '       uphill reduction'
                if FB > 1:              # (f0-f2) > (f1-f0) extend step
                   step_stdev *= FB
                   step_txt = '       uphill extension'
            if f1 < f0:                 # step 1 is downhill
                if FC > 1:              # (f2-f1) > (f0-f1) reduce step
                   step_stdev /= FC
                   step_txt = '     downhill reduction'
                if FD > 1:              # (f1-f2) > (f0-f1) extend step
                   step_stdev *= FD
                   step_txt = '     downhill extension'

        step_stdev = max(step_stdev,5*tol_v)
            
        # find the best (min(fa)) objective value of the 4 evaluations
        fa[0] = fa[i_min]

        if i_min == 1:
            x0, g0, c0 = x1, g1, c0
        elif i_min == 2:
            x0, g0, c0 = x2, g2, c2
        elif i_min == 3:
            x0, g0, c0 = x3, g3, c3
            quad_update = True
        
        # if the solution improved, reduce the scope of the search 
            # step_stdev = step_stdev * (1 - function_count / max_evals) ** nu
       
        x0 = np.clip(x0, -1.0, 1.0)  # keep x0 within bounds, just to be sure
        
        # update optimal solution if improved
        if fa[0] < f_opt:
            # use .copy() to keep changes to x0 from changing x_opt!
            x_opt = x0.copy()
            f_opt = fa[0].copy() # .copy() not needed since f_opt is immutable
            g_opt = g0.copy()
            v_opt = (x_opt - s0) / s1 # scale (x) back to original units (v)
            
            # Convergence criteria
            cvg_v = norm(cvg_hst[0:n, iteration] - v_opt) / (norm(cvg_hst[0:n, iteration] + v_opt)+1e-9)
            cvg_f = norm(cvg_hst[  n, iteration] - f_opt) / (norm(cvg_hst[  n, iteration] + f_opt)+1e-9)
            
            iteration += 1
            cvg_hst[:, iteration] = np.concatenate([
                v_opt, [f_opt], [np.max(g_opt)],
                [function_count], [cvg_v], [cvg_f]
            ])
            last_update = function_count
            
            # Display progress
            if msg:
                elapsed = time.time() - start_time
                secs_left = int((max_evals - function_count) * elapsed / function_count)
                eta = (datetime.now() + timedelta(seconds=secs_left)).strftime('%H:%M:%S')
                
                max_g = np.max(g_opt)
                idx_ub_g = np.argmax(g_opt)
                
                #print('\033[H\033[J', end='')  # clear screen
                print(' +-+-+-+-+-+-+-+-+-+-+- ORS -+-+-+-+-+-+-+-+-+-+-+-+-+')
                print(f' iteration               = {iteration:5d}', end='')
                if max_g > tol_g:
                    print('     !!! infeasible !!!')
                else:
                    print('     ***  feasible  ***')
                print(f' function evaluations    = {function_count:5d}  of  {max_evals:5d}  '
                      f'({100*function_count/max_evals:4.1f}%)')
                print(f' e.t.a.                  = {eta}')
                print(f' objective               = {f_opt:11.3e}')
                print(f' variables               = ', end='')
                for val in v_opt:
                    print(f'{val:11.3e}', end='')
                print()
                print(f' max constraint          = {max_g:11.3e} ({idx_ub_g})')
                print(f' objective convergence   = {cvg_f:11.4e}    tol_f = {tol_f:8.6f}')
                print(f' variable  convergence   = {cvg_v:11.4e}    tol_v = {tol_v:8.6f}')
                print(f' c.o.v. of F_A           = {c0:11.3e}')
                print(f' step std.dev            = {step_stdev:7.3f}{step_txt}')
                print(' +-+-+-+-+-+-+-+-+-+-+- ORS -+-+-+-+-+-+-+-+-+-+-+-+-+')
                if quad_update:
                    print(' line quadratic update successful')
        
            # plot on surface
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
                    plt.plot([v3[ii]], [v3[jj]], fa[3], 
                            'ob', markersize=9, linewidth=3)
                
                plt.draw()
                plt.pause(0.01)
        
        # check for feasible solution
        if np.max(g_opt) < tol_g and find_feas:
            print('\n * Woo Hoo! Feasible solution found!', end='')
            print(' *          ... and that is all we are asking for.')
            break
        
        # check convergence
        if iteration > n*n and (cvg_v < tol_v or cvg_f < tol_f or 
                                (function_count - last_update) > 0.2*max_evals):
            print('\n * Woo-Hoo! Converged solution found!')
            if cvg_v < tol_v:
                print(' *           convergence in design variables')
            if cvg_f < tol_f:
                print(' *           convergence in design objective')
            
            if np.max(g_opt) < tol_g:
                print(' * Woo-Hoo! Converged solution is feasible!')
            else:
                print(' * Boo-Hoo! Converged solution is NOT feasible!')
                if np.max(g_opt) > tol_g:
                    print(' *   ... Increase options[5] (penalty) and try, try again ...')
                else:
                    print(' *   ... Decrease options[5] (penalty) and try, try again ...')
            break
    
    # ========== end main loop ==========
    
    # check if maximum evaluations exceeded
    if function_count >= max_evals:
        if msg:
            print(f'\n * Enough!! Maximum number of function evaluations ({max_evals}) '
                  'has been exceeded')
            print(' *   ... Increase tol_v (options[1]) or max_evals (options[4]) '
                  'and try try again.')
    
    # scale back to original units
#   v_init = (x_init - s0) / s1
#   v_opt = (x_opt - s0) / s1
    
    # final report
    if msg:
        dur = time.time() - start_time
        print(f" * objective = {f_opt:11.3e}   evals = {function_count}   " f"time = {dur:.2f}s")
        print(' * --------------------------------------------------------------')
        print(' *             v_init         v_lb     <     v_opt    <     v_ub')
        print(' * --------------------------------------------------------------')
        
        for i in range(n):
            eqlb = ' '
            equb = ' '
            if v_opt[i] < v_lb[i] + tol_g + 10*np.finfo(float).eps:
                eqlb = '='
            elif v_opt[i] > v_ub[i] - tol_g - 10*np.finfo(float).eps:
                equb = '='
            
            print(f' * v[{i:3d}] {v_init[i]:11.4f}    {v_lb[i]:11.4f} {eqlb} '
                  f' {v_opt[i]:12.5f} {equb} {v_ub[i]:11.4f}')
        
        print(' * Constraints :')
        for j in range(m):
            binding = ' '
            if g_opt[j] > -tol_g:
                binding = ' ** binding ** '
            if g_opt[j] > tol_g:
                binding = ' ** not ok  ** '
            print(f' *     g({j:3d}) = {g_opt[j]:12.5f}  {binding}')
        print(' *\n * --------------------------------------------------------------')

    if msg > 2:
        plt.figure(1003)
        ii = int(options[10])
        jj = int(options[11])
        plt.plot( v_opt[ii], v_opt[jj], f_opt, '-or', markersize=14 )
        plt.draw()
        plt.pause(0.10)
    
    # save the final iteration
    cvg_hst[:, iteration] = np.concatenate([
        v_opt, [f_opt], [np.max(g_opt)], [function_count], [cvg_v], [cvg_f]
    ])
    cvg_hst[n+4, 0:2] = cvg_hst[n+4, 2]
    cvg_hst = cvg_hst[:, 0:iteration+1]

    elapsed = time.time() - start_time
    completion_time = datetime.now().strftime('%H:%M:%S')
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f' * Completion  : {completion_time} ({elapsed_str})')
   
    return v_opt, f_opt, g_opt, cvg_hst, iteration, function_count

# ======================================================================
# updated 2011-04-13, 2014-01-12, 2015-03-14, (pi day 03.14.15), 2015-03-26, 
# 2016-04-06, 2019-02-23, 2020-01-17, 2024-04-03, 2025-11-24

