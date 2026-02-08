"""
nms.py
-----------------------------------------------------------------------------
Nelder-Mead Algorithm for Nonlinear Optimization
Depends on: opt_options(), avg_cov_func(), plot_opt_surface()
-----------------------------------------------------------------------------

Nonlinear optimization with inequality constraints via the Nelder-Mead Simplex

Minimizes f(v) such that g(v) < 0 and v_lb <= v_opt <= v_ub.
- f is a scalar objective function
- v is a vector of design variables
- g is a vector of inequality constraints

Reference:

Nelder, J.A., and Mead, R., "A simplex method for function minimization,: 
Computer Journal, 7(4) (1965): 308-313.

William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, 
Numerical Recipes in C 
Cambridge University Press, (1992)

H.P. Gavin, Civil & Environmental Eng'g, Duke Univ.
Translation from MATLAB to Python, 2025-11-24

updated ...
2005-1-22, 2006-1-26, 2011-1-31, 2011-4-13, 2016-03-24, 2016-04-06,
2019-02-23, 2019-03-21, 2019-11-22, 2020-01-17, 2021-01-19, 2024-04-03,
2025-11-24
"""

import numpy as np
from numpy.linalg import norm 
from matplotlib import pyplot as plt
import time
from datetime import datetime, timedelta

from multivarious.utl.avg_cov_func import avg_cov_func
from multivarious.utl.plot_opt_surface import plot_opt_surface
from multivarious.utl.opt_options import opt_options
from multivarious.utl.opt_report import opt_report

def nms(func, v_init, v_lb=None, v_ub=None, options_in=None, consts=1.0):
    """
    Nelder-Mead Algorithm for nonlinear optimization with inequality constraints

    minimizes f(v) such that g(v) < 0 and v_lb <= v_opt <= v_ub.
    f is a scalar objective function, v is a vector of design variables, and
    g is a vector of constraints.

    The algorithm uses a simplex search method. For optimization with respect 
    to N variables, a simplex is a combination of N+1 parameter sets represented
    by an N by N+1 matrix. Each column represents a set of design variables with
    an associated objective value. Simplex columns are sorted by increasing 
    objective function value.

    Parameters
    ----------
    func : callable
        Signature: f, g = func(v, consts). 
                   f is scalar, g is (m,) constraints (g<0 feasible).
                   v is in *original* units (not scaled).
    v_init : array-like (n,)
        Initial guess.
    v_lb, v_ub : array-like (n,), optional
        Lower/upper bounds on v. If omitted, wide bounds are used (1e2*|v_init|).
    options_in : array-like, optional
        See opt_options() for the 19 parameters 
    consts : any
        Passed through to `func`.

    Returns
    -------
    v_opt : np.ndarray (n,)
    f_opt : float
    g_opt : np.ndarray (m,)
    cvg_hst : np.ndarray (n+5, k)
        Columns store [v; f; max(g); func_count; cvg_v; cvg_f] per iteration.
    iteration : int
        Number of iterations completed
    function_evals : int
        Total number of function evaluations

    References
    ----------
    J.A. Nelder and R. Mead, "A simplex method for function minimization,"
    Computer Journal, 7(4)(1965):308-313.

    S. Rao, Optimization Theory and Applications, 2nd ed, John Wiley, 1984
    S. Rao, Computer Methods in Applied Mechanics & Engg, 19(1979):99-106.

    J.E. Dennis, Jr. and D.J. Woods,
    New Computing Environments: Microcomputers in Large-Scale Computing,
    edited by A. Wouk, SIAM, (1987):116-122.

    F. Gao and L. Han, "Implementing the Nelder-Mead simplex algorithm with 
    adaptive parameters," Comput Optim Appl (2012) 51:259-277.
    """

    v_init = np.asarray(v_init, dtype=float).flatten()
    n = v_init.size

    # ----- Nelder-Mead algorithm hyperparameters -----
    # Gao + Han, Comput Optim Appl (2012) 51:259-277
    a_reflect  = 1.0
    a_extend   = 1.0 + 2.0 / n      # Standard: 2.0
    a_contract = 0.75 - 0.5 / n     # Standard: 0.5
    a_shrink   = 1.0 - 1.0 / n      # Standard: 0.5
    a_expand   = 1.3

    BOX = 1 # enforce bounds inside avg_cov_func

    # ----- options & inputs -----

    if v_lb is None or v_ub is None:
        v_lb = -1.0e2 * np.abs(v_init)
        v_ub = +1.0e2 * np.abs(v_init)
    v_lb = np.asarray(v_lb, dtype=float).flatten()
    v_ub = np.asarray(v_ub, dtype=float).flatten()

    # Bounds should not be zero
    v_lb[v_lb == 0] =  1e-4
    v_ub[v_ub == 0] = -1e-4

    # Check for valid bounds
    if np.any(v_ub <= v_lb):
        raise ValueError("v_ub must be greater than v_lb for all parameters")

    options   = opt_options(options_in)
    msg       = int(options[0])   # display level
    tol_v     = float(options[1]) # design var convergence tol
    tol_f     = float(options[2]) # objective convergence tol
    tol_g     = float(options[3]) # constraint tol
    max_evals = int(options[4])   # budget
    find_feas = bool(options[9])  # stop once feasible

    optimize_contraction = False  # Option to optimize contraction step

    feasible = converged = stalled = False # convergence criteria

    # ----- scale variables linearly to [-1, +1]  -----
    s0 = (v_lb + v_ub) / 2.0  # average of v_lb and v_ub 
    s1 = (v_ub - v_lb) / 2.0  # half distance from v_lb to v_ub
    u0 = (v_init-s0)/s1
    u0 = np.clip(u0, -0.8, 0.8)  # Not too close to edges

    # book-keeping
    function_evals = iteration = 0
    cvg_hst = np.full((n + 5, max(1, max_evals)), np.nan)

    # ----- analyze the initial guess -----
    f0, g0, u0, cJ, nAvg = avg_cov_func(func, u0, s0, s1, options, consts, BOX)
    function_evals += nAvg
    g0 = np.atleast_1d(g0).astype(float).flatten()
    m = g0.size  # number of constraints

    if msg > 2:
        f_min, f_max, ax = plot_opt_surface(func, v_init, v_lb, v_ub, 
                                            options, consts, 1003)

    start_time = time.time()

    # ----- Set up initial simplex -----
    # Use equilateral simplex (Stanford AA-222 course notes)
    simplex = np.full((n, n + 1), np.nan)
    f_all = np.full(n + 1, np.nan)
    g_all = np.full((m, n + 1), np.nan)
    g_max = np.full(n + 1, np.nan)
    cJ_all = np.full(n + 1, np.nan)

    # Include initial guess as first vertex
    simplex[:, 0] = u0
    f_all[0] = f0
    g_all[:, 0] = g0
    g_max[0] = np.max(g0)
    cJ_all[0] = cJ

    # Create equilateral simplex
    cc = 0.3
    bb = cc / (np.sqrt(2) * n) * (np.sqrt(n + 1) - 1)
    aa = bb + cc / np.sqrt(2)

    for i in range(n):
        delta_u = bb * np.ones(n)
        delta_u[i] = aa
        u = u0 + delta_u

        fz, gz, u, cu, nAvg = avg_cov_func(func, u, s0, s1, options, consts, BOX)
        j = i + 1
        simplex[:, j] = u
        f_all[j] = fz
        g_all[:, j] = gz
        g_max[j] = np.max(gz)
        cJ_all[j] = cu
        function_evals += nAvg

    # SORT vertices by increasing objective value
    idx = np.argsort(f_all)
    simplex = simplex[:, idx]
    f_all = f_all[idx]
    g_all = g_all[:, idx]
    g_max = g_max[idx]
    cJ_all = cJ_all[idx]

    # Initialize best solution
    u_opt = simplex[:, 0].copy()
    f_opt = f_all[0]
    g_opt = g_all[:, 0].copy()
    f_old = f_opt
    last_update = function_evals

    # Convergence metrics for initial simplex
    cvg_v = norm(simplex[:, 1] - simplex[:, n])/norm(simplex[:, 1] + simplex[:, n]+1e-9)
    cvg_f = 1.0; 
    max_g = max(g_opt); 

    cvg_hst[:, iteration] = np.concatenate([ s0+s1*simplex[:, 0],
                      [f_all[0], max_g, function_evals, cvg_v, cvg_f ]])

    xtx = " simplex :    vertex 1 "
    for i in range(n):
        xtx += f"   vertex {(i+2):1d} "

    # Print the initial simplex, objective and constraints
    if msg > 1:
#       print('\033[H\033[J', end='')  # Clear screen
        print(" ======================= NMS ============================")
        print(f" iteration                = {iteration:5d}   "
              f"{'*** feasible ***' if np.max(g_opt) <= tol_g else '!!! infeasible !!!'}")
        print(f" function evaluations     = {function_evals:5d} of {max_evals:5d}")
        if n < 15:
            vv = s0[:, np.newaxis] + s1[:, np.newaxis] * simplex
            print(xtx)
            for j in range(n):
                vstr = "           " + " ".join(f"{vp:11.3e}" for vp in vv[j,:])
                print(vstr)
            print(" f_A    = " + " ".join(f"{f:11.3e}" for f in f_all))
            print(" max(g) = " + " ".join(f"{g:11.3e}" for g in g_max))
        print(f" objective                = {f_opt:12.4e}")
        print(" ======================= NMS ============================\n")

    # ============================ main loop ============================
    while function_evals < max_evals:
        accept_point = False
        move_type = ""

        # Centroid of best n vertices (excluding worst)
        uo = np.mean(simplex[:, :n], axis=1)

        # ----- REFLECT ----u
        ur = uo + a_reflect * (uo - simplex[:, n])
        fr, gr, ur, cj, nAvg = avg_cov_func(func, ur, s0, s1, options, consts, BOX)
        function_evals += nAvg

        if f_all[0] <= fr < f_all[n - 1]:  # fr between best and second-worst
            uw, fw, gw, cw = ur, fr, gr, cj
            move_type = 'reflect'
            accept_point = True

        # ----- EXTEND -----
        if not accept_point and fr < f_all[0]:  # fr better than best
            ue = uo + a_extend * (ur - uo)
            fe, ge, ue, cj, nAvg = avg_cov_func(func, ue, s0, s1, options, consts, BOX)
            function_evals += nAvg

            if fe < fr:
                uw, fw, gw, cw = ue, fe, ge, cj
                move_type = 'extend'
            else:
                uw, fw, gw, cw = ur, fr, gr, cj
                move_type = 'reflect'
            accept_point = True

        # ----- CONTRACT -----
        if not accept_point and fr > f_all[n - 1]:  # fr worse than second-worst
            uci = uo - a_contract * (ur - uo)  # inside contraction
            uco = uo + a_contract * (ur - uo)  # outside contraction

            fci, gci, uci, ci, nAvg = avg_cov_func(func, uci, s0, s1, options, consts, BOX)
            function_evals += nAvg

            fco, gco, uco, co, nAvg = avg_cov_func(func, uco, s0, s1, options, consts, BOX)
            function_evals += nAvg

            # Optional: optimize contraction step
            if optimize_contraction:
                d = np.array([
                    -norm(uo - simplex[:, n]),
                    -norm(uo - uci),
                     norm(uo - uco),
                     norm(uo - ur)
                ])
                A = np.column_stack([np.ones(4), d, 0.5 * d**2])
                a_coef = np.linalg.solve(A, np.array([f_all[n], fci, fco, fr]))
                du = -a_coef[1] / a_coef[2]

                if abs(du) < d[3] and a_coef[2] > 0:
                    uc_opt = uo + du * (ur - uo) / d[3]
                    fc_opt, gc_opt, uc_opt, cj, nAvg = avg_cov_func(func, uc_opt, s0, s1,
                                                                      options, consts, BOX)
                    function_evals += nAvg

                    if fc_opt < min(fci, fco) and fc_opt < f_all[n - 1]:
                        uw, fw, gw, cw = uc_opt, fc_opt, gc_opt, cj
                        move_type = 'contract opt'
                        accept_point = True

            if not accept_point and fci < fco and fci < f_all[n - 1]:
                uw, fw, gw, cw = uci, fci, gci, ci
                move_type = 'contract in'
                accept_point = True

            if not accept_point and fco < fci and fco < f_all[n - 1]:
                uw, fw, gw, cw = uco, fco, gco, co
                move_type = 'contract out'
                accept_point = True

        # ----- ACCEPT or SHRINK -----
        if accept_point:
            # Replace worst point with new point
            simplex[:, n] = uw
            f_all[n] = fw
            g_all[:, n] = gw
            g_max[n] = np.max(gw)
            cJ_all[n] = cw
        else:
            # SHRINK all points toward best point
            u0 = simplex[:, 0]
            for i in range(1, n + 1):
                uk = u0 + a_shrink * (simplex[:, i] - u0)
                fk, gk, uk, cj, nAvg = avg_cov_func(func, uk, s0, s1, options, consts, BOX)
                simplex[:, i] = uk
                f_all[i] = fk
                g_all[:, i] = gk
                g_max[i] = np.max(gk)
                cJ_all[i] = cj
                function_evals += nAvg
            move_type = 'shrink'

        # ----- EXPAND elongated (degenerate) simplex -----
        vo = np.full((n, n + 1), np.nan)
        lo = np.full(n + 1, np.nan)

        for i in range(n + 1):
            idx = np.arange(n + 1) != i
            vo[:, i] = np.mean(simplex[:, idx], axis=1)  # centroid of opposite vertices
            lo[i] = norm(simplex[:, i] - vo[:, i])

        elongated_idx = np.where(lo / np.max(lo) < (a_expand - 1))[0]
        for j in elongated_idx:
            uz = vo[:, j] + a_expand * (simplex[:, j] - vo[:, j])
            fz, gz, uz, cj, nAvg = avg_cov_func(func, uz, s0, s1, options, consts, BOX)
            simplex[:, j] = uz
            f_all[j] = fz
            g_all[:, j] = gz
            g_max[j] = np.max(gz)
            cJ_all[j] = cj
            function_evals += nAvg

        # ----- SORT vertices by increasing objective -----
        idx = np.argsort(f_all)
        simplex = simplex[:, idx]
        f_all = f_all[idx]
        g_all = g_all[:, idx]
        g_max = g_max[idx]
        cJ_all = cJ_all[idx]

        # Update best solution
        u_opt = simplex[:, 0].copy()
        f_opt = f_all[0]
        g_opt = g_all[:, 0].copy()
        v_opt = s0 + s1 * u_opt

        if f_opt < f_old:
            last_update = function_evals
            f_old = f_opt

        # ----- Convergence metrics -----
        cvg_v, cvg_f, max_g = cvg_metrics( simplex, f_all, g_opt)

        iteration += 1
        cvg_hst[:, iteration] = np.concatenate([ v_opt,
                      [ f_opt, max_g, function_evals, cvg_v, cvg_f ] ])

        # ----- Display progress -----
        if msg > 1:
            elapsed = time.time() - start_time
            secs_left = int((max_evals - function_evals) * elapsed / function_evals)
            eta = (datetime.now() + timedelta(seconds=secs_left)).strftime('%H:%M:%S')

            # print('\033[H\033[J', end='')  # clear screen
            print(" ======================= NMS ============================")
            print(f" iteration                = {iteration:5d}   "
                  f"{'*** feasible ***' if np.max(g_opt) <= tol_g else '!!! infeasible !!!'}")
            print(f"                                             {move_type}")
            print(f" function evaluations     = {function_evals:5d} of {max_evals:5d}"
                  f" ({100.0*function_evals/max_evals:4.1f}%)")
            print(f" e.t.a.                   = {eta} ")
            if n < 15:
                vv = (simplex - s0[:, np.newaxis]) / s1[:, np.newaxis]
                print(xtx)
                for j in range(n):
                    vstr = "           " + " ".join(f"{vp:11.3e}" for vp in vv[j,:])
                    print(vstr)
                print(" f_A       " + " ".join(f"{f:11.3e}" for f in f_all))
                print(" max(g)   =" + " ".join(f"{g:11.3e}" for g in g_max))
                print(" cov(F_A) =" + " ".join(f"{c:11.3e}" for c in cJ_all))
            else:
                vv = s0 + s1 * u_opt
                print(" variables             = " + " ".join(f"{v:11.3e}" for v in vv))
                print(f" max constraint       = {np.max(g_opt):11.3e}")

            print(f" objective                = {f_opt:11.4e}")
            print(f" constraint               = {np.max(g_opt):11.4e}   tol_g = {tol_g:8.6f}")
            print(f" variable  convergence    = {cvg_v:11.4e}   tol_v = {tol_v:8.6f}")
            print(f" objective convergence    = {cvg_f:11.4e}   tol_f = {tol_f:8.6f}")
            print(f" c.o.v. of F_A            = {cJ_all[0]:11.4e}")
            print(" ======================= NMS ============================\n")

        # ----- Plot simplex on surface -----
        if msg > 2:
            ii = int(options[10])
            jj = int(options[11])
            simplex_plot = s0[:, np.newaxis] + s1[:, np.newaxis] * simplex

            # Plot simplex as connected triangles (for n=2, plots all 3 vertices)
            if n == 2:
                # Close the triangle by appending first vertex
                x_coords = np.append(simplex_plot[ii, :3], simplex_plot[ii, 0])
                y_coords = np.append(simplex_plot[jj, :3], simplex_plot[jj, 0])
                f_vals = np.append(f_all[:3], f_all[0])
                ax.plot(x_coords, y_coords, f_vals ,
                       '-or', alpha=1.0, markersize=6, linewidth=2,
                       markerfacecolor='red', markeredgecolor='darkred')
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
        if function_evals - last_update > 0.2*max_evals:          # :(
            stalled = True   
        if feasible or converged or stalled:
            break 

    # ========== end main loop ==========

    # trim the convergence history
    cvg_hst = cvg_hst[:, :iteration+1]

    # plot the converged point
    if msg > 2:
        plt.figure(1003)
        ii = int(options[10])
        jj = int(options[11])
        plt.plot( v_opt[ii], v_opt[jj], f_opt, '-or', markersize=14 )

    # final report
    if msg:
        lambda_qp = None
        opt_report(v_init, v_opt, f_opt, g_opt, v_lb, v_ub, tol_v, tol_f, tol_g,
                   lambda_qp, start_time, function_evals, max_evals,
                   find_feas, feasible, converged, stalled )

    return v_opt, f_opt, g_opt, cvg_hst, function_evals, iteration


def cvg_metrics(simplex, fv, g0):
    '''
    Compute convergence metrics for nms, defined as:  
    the ratio of (the difference between the best and worst vertices)
    to (the average of the best and worst vertices) 
    in terms of the vertices in the simplex and the objective function, f. 
    
    Parameters
    ----------    
    simplex ndarray (n,n+1)
       the nelder mead simplex of design variables 
    fv array
       the design objective at each simplex point 
    g0 array
       constraints at the best simplex point

    Returns
    -------
      cvg_v float
        convergence metric for design variables
      cvg_f float
        convergence metric for design objective
      max_g float
        the maximum of the design constraints
    '''

    n = np.shape(simplex)[0] # number of design variabls 

    u0 = simplex[:,0]
    un = simplex[:,n]
    f0 = fv[0]
    fn = fv[n]

    cvg_v = 2 * norm(un - u0) / (norm(un + u0)+1e-9)
    cvg_f = 2 * norm(fn - f0) / (norm(fn + f0)+1e-9)
    max_g = max(g0)

    return cvg_v, cvg_f, max_g
