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
    
    BOX = 1  # use box constraints
    
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
    msglev = int(options[0])
    tol_v = options[1]
    tol_f = options[2]
    tol_g = options[3]
    max_evals = int(options[4])
    penalty = options[5]
    q = options[6]
    find_feas = int(options[9])
    
    # check bounds are valid
    if np.any(v_ub < v_lb):
        print('Error: v_ub cannot be less than v_lb for any variable')
        return v_init, (np.sqrt(5)-1)/2, np.array([1.0]), None, 0, 0
    
    # initialize
    function_count = 0
    iteration = 1
    cvg_f = 1.0
    cvg_hst = np.full((n + 5, max_evals), np.nan)
    fa = np.zeros(4)
    
    # scale v from [v_lb, v_ub] to x in [-1, +1]
    s0 = (v_lb + v_ub) / (v_lb - v_ub)
    s1 = 2.0 / (v_ub - v_lb)
    x_init = s0 + s1 * v_init
    x_init = np.clip(x_init, -1.0, 1.0)
    
    i3 = 1e-9 * np.eye(3)  # Regularization for matrix inversion
    
    # evaluate initial guess
    fv, gv, v_init_scaled, cJ, nAvg = avg_cov_func(
        func, x_init, s0, s1, options, consts, BOX
    )
    function_count += nAvg
    
    m = len(gv)  # Number of constraints
    
    # check dimensions
    if np.prod(np.shape(fv)) != 1:
        raise ValueError('Objective must be a scalar')
    if gv.ndim == 2 and gv.shape[0] == 1:
        raise ValueError('Constraints must be a column vector')
    
    # optional: plot objective surface
    if msglev > 2:
        f_min, f_max, ax = plot_opt_surface(
            func, (x_init - s0) / s1, v_lb, v_ub, options, consts, 103)

    
    # algorithm hyper-parameters
    sigma = 0.200  # standard deviation of random perturbations
    nu = 2.5       # exponent for reducing sigma
    
    if msglev:
        start_time = time.time()
    
    # analyze initial guess
    x1 = x_init.copy()
    fa[0], g1, x1, c1, nAvg = avg_cov_func(func, x1, s0, s1, options, consts, BOX)
    function_count += nAvg
    
    # initialize optimal values
    f_opt = fa[0]
    x_opt = x1.copy()
    g_opt = g1.copy()
    
    # save initial guess to convergence history
    cvg_hst[:, iteration-1] = np.concatenate([
        (x_opt - s0) / s1, [f_opt], [np.max(g_opt)], 
        [function_count], [sigma], [1.0]
    ])
    
    if msglev:
        print()  # clear screen effect
    
    x4 = x1.copy()
    g4 = g1.copy()
    fa[3] = fa[0]
    
    last_update = function_count
    
    # ========== main optimization loop ==========
    while function_count < max_evals:
        
        # random search perturbation
        r = sigma * np.random.randn(n)
        
        # 1st perturbation: +1*r
        aa, _ = box_constraint(x1, r)
        x2 = x1 + aa * r
        
        fa[1], g2, x2, c2, nAvg = avg_cov_func(func, x2, s0, s1, options, consts, BOX)
        function_count += nAvg
        
        # determine direction for second perturbation
        if fa[1] < fa[0]:
            step = +2
        else:
            step = -1
        
        # 2nd perturbation: -1*r or +2*r
        aa, bb = box_constraint(x1, step * r)
        if step > 0:
            x3 = x1 + aa * step * r
        else:
            x3 = x1 + bb * r
        
        fa[2], g3, x3, c3, nAvg = avg_cov_func(func, x3, s0, s1, options, consts, BOX)
        function_count += nAvg
        
        # distances for quadratic fit
        dx2 = np.linalg.norm(x2 - x1) / np.linalg.norm(r)
        dx3 = np.linalg.norm(x3 - x1) / np.linalg.norm(r)
        
        # fit quadratic: f(d) = 0.5*a*d^2 + b*d + c
        A = np.array([
            [0,           0,    1],
            [0.5*dx2**2, dx2,   1],
            [0.5*dx3**2, dx3,   1]
        ]) + i3
        
        abc = np.linalg.solve(A, fa[0:3])
        a, b, c = abc[0], abc[1], abc[2]
        
        # try quadratic update if curvature is positive
        quad_update = False
        if a > 0:
            d = -b / a  # zero-slope point
            aa, bb = box_constraint(x1, d * r)
            if d > 0:
                x4 = x1 + aa * d * r
            else:
                x4 = x1 + bb * d * r
            
            fa[3], g4, x4, c4, nAvg = avg_cov_func(func, x4, s0, s1, options, consts, BOX)
            function_count += nAvg
        
        # find best of the 4 evaluations
        i_min = np.argmin(fa)
        fa[0] = fa[i_min]
        
        if i_min == 1:
            x1, g1, c1 = x2, g2, c2
        elif i_min == 2:
            x1, g1, c1 = x3, g3, c3
        elif i_min == 3:
            x1, g1, c1 = x4, g4, c4
            quad_update = True
        
        # update search scope if solution improved
        if i_min > 0:
            sigma = sigma * (1 - function_count / max_evals) ** nu
        
        x1 = np.clip(x1, -1.0, 1.0)  # Keep within bounds
        
        # update optimal solution if improved
        if fa[0] < f_opt:
            x_opt = x1.copy()
            f_opt = fa[0]
            g_opt = g1.copy()
            
            # Convergence criteria
            cvg_v = np.linalg.norm(cvg_hst[0:n, iteration-1] - (x_opt - s0) / s1) / \
                    np.linalg.norm((x_opt - s0) / s1)
            cvg_f = np.linalg.norm(cvg_hst[n, iteration-1] - f_opt) / np.abs(f_opt)
            
            last_update = function_count
            cvg_hst[:, iteration] = np.concatenate([
                (x_opt - s0) / s1, [f_opt], [np.max(g_opt)],
                [function_count], [cvg_v], [cvg_f]
            ])
            iteration += 1
            
            # Display progress
            if msglev:
                elapsed = time.time() - start_time
                secs_left = int((max_evals - function_count) * elapsed / function_count)
                eta = (datetime.now() + timedelta(seconds=secs_left)).strftime('%H:%M:%S')
                
                max_g = np.max(g_opt)
                idx_ub_g = np.argmax(g_opt)
                
                print('\033[H\033[J', end='')  # clear screen
                print(' -+-+-+-+-+-+-+-+-+-+- ORS -+-+-+-+-+-+-+-+-+-+-+-+-+')
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
                for val in (x_opt - s0) / s1:
                    print(f'{val:11.3e}', end='')
                print()
                print(f' max constraint          = {max_g:11.3e} ({idx_ub_g})')
                print(f' Convergence Criterion F = {cvg_f:11.4e}    tolF = {tol_f:8.6f}')
                print(f' Convergence Criterion X = {cvg_v:11.4e}    tolX = {tol_v:8.6f}')
                print(f' c.o.v. of f_a           = {c1:11.3e}')
                print(f' step std.dev (sigma)    = {sigma:5.3f}')
                print(' -+-+-+-+-+-+-+-+-+-+- ORS -+-+-+-+-+-+-+-+-+-+-+-+-+')
                if quad_update:
                    print(' line quadratic update successful')
        
        # optional: plot on surface
        if msglev > 2:
            try:
                v1 = (x1 - s0) / s1
                v2 = (x2 - s0) / s1
                v3 = (x3 - s0) / s1
                v4 = (x4 - s0) / s1
                f_offset = (f_max - f_min) / 100
                
                plt.figure(103)
                ii = int(options[10])
                jj = int(options[11])
                
                if step == -1:
                    plt.plot([v2[ii], v1[ii], v3[ii]], 
                            [v2[jj], v1[jj], v3[jj]],
                            fa[[1, 0, 2]] + f_offset, '-or', markersize=4)
                else:
                    plt.plot([v1[ii], v2[ii], v3[ii]], 
                            [v1[jj], v2[jj], v3[jj]],
                            fa[[0, 1, 2]] + f_offset, '-or', markersize=4)
                
                if quad_update:
                    plt.plot([v4[ii]], [v4[jj]], fa[3] + f_offset, 
                            'ob', markersize=9, linewidth=3)
                
                plt.draw()
                plt.pause(0.01)
            except:
                pass
        
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
        if msglev:
            print(f'\n * Enough!! Maximum number of function evaluations ({max_evals}) '
                  'has been exceeded')
            print(' *   ... Increase tol_v (options[1]) or max_evals (options[4]) '
                  'and try try again.')
    
    # scale back to original units
    v_init = (x_init - s0) / s1
    v_opt = (x_opt - s0) / s1
    
    # final report
    if msglev:
        elapsed = time.time() - start_time
        completion_time = datetime.now().strftime('%H:%M:%S')
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f' * \n * Completion  : {completion_time} ({elapsed_str})')
        print(f' * Objective   : {f_opt:11.3e}')
        print(' * Variables   :')
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
        print(' *\n * --------------------------------------------------------------\n')
    
    # save final iteration
    cvg_hst[:, iteration] = np.concatenate([
        v_opt, [f_opt], [np.max(g_opt)], [function_count], [cvg_v], [cvg_f]
    ])
    cvg_hst[n+4, 0:2] = cvg_hst[n+4, 2]
    cvg_hst = cvg_hst[:, 0:iteration+1]
    
    return v_opt, f_opt, g_opt, cvg_hst, iteration, function_count

# ======================================================================
# updated 2011-04-13, 2014-01-12, 2015-03-14, (pi day 03.14.15), 2015-03-26, 
# 2016-04-06, 2019-02-23, 2020-01-17, 2024-04-03, 2025-11-24

