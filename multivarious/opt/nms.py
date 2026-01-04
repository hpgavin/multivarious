# nms.py
# -----------------------------------------------------------------------------
# Nelder-Mead Algorithm for Nonlinear Optimization
# Depends on: opt_options(), avg_cov_func(), plot_opt_surface()
# -----------------------------------------------------------------------------
# 
# updated ...
# 2005-1-22, 2006-1-26, 2011-1-31, 2011-4-13, 2016-03-24, 2016-04-06,
# 2019-02-23, 2019-03-21, 2019-11-22, 2020-01-17, 2021-01-19, 2024-04-03,
# 2025-11-24


import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime, timedelta

from multivarious.utl.avg_cov_func import avg_cov_func
from multivarious.utl.plot_opt_surface import plot_opt_surface
from multivarious.utl.opt_options import opt_options


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

    # ----- options & inputs -----
    v_init = np.asarray(v_init, dtype=float).flatten()
    n = v_init.size

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

    # ----- scale variables linearly to [-1, +1]  -----
    s0 = (v_lb + v_ub) / (v_lb - v_ub)
    s1 = 2.0 / (v_ub - v_lb)
    x1 = s0 + s1 * v_init
    x1 = np.clip(x1, -0.8, 0.8)  # Not too close to edges

    # book-keeping
    function_count = 0
    iteration = 1
    cvg_hst = np.full((n + 5, max(1, max_evals)), np.nan)
    BX = 1  # enforce bounds inside avg_cov_func

    # ----- analyze initial guess -----
    fx, gx, x1, cJ, nAvg = avg_cov_func(func, x1, s0, s1, options, consts, BX)
    function_count += nAvg
    if not np.isscalar(fx):
        raise ValueError("Objective returned by func(v,consts) must be scalar.")
    gx = np.atleast_1d(gx).astype(float).flatten()
    m = gx.size  # number of constraints

    # Save initial guess
    cvg_hst[:, iteration - 1] = np.concatenate([(x1 - s0) / s1,
                                [fx, np.max(gx), function_count, 1.0, 1.0]])

    if msg > 2:
        f_min, f_max, ax = plot_opt_surface(func, (x1-s0)/s1, v_lb, v_ub, 
                                            options, consts, 103)

    start_time = time.time()

    # ----- Nelder-Mead algorithm coefficients -----
    # Gao + Han, Comput Optim Appl (2012) 51:259-277
    a_reflect  = 1.0
    a_extend   = 1.0 + 2.0 / n      # Standard: 2.0
    a_contract = 0.75 - 0.5 / n     # Standard: 0.5
    a_shrink   = 1.0 - 1.0 / n      # Standard: 0.5
    a_expand   = 1.3

    # ----- Set up initial simplex -----
    # Use equilateral simplex (Stanford AA-222 course notes)
    simplex = np.full((n, n + 1), np.nan)
    fx_all = np.full(n + 1, np.nan)
    gx_all = np.full((m, n + 1), np.nan)
    g_max = np.full(n + 1, np.nan)
    cJ_all = np.full(n + 1, np.nan)

    # Include initial guess as first vertex
    simplex[:, 0] = x1
    fx_all[0] = fx
    gx_all[:, 0] = gx
    g_max[0] = np.max(gx)
    cJ_all[0] = cJ

    # Create equilateral simplex
    cc = 0.3
    bb = cc / (np.sqrt(2) * n) * (np.sqrt(n + 1) - 1)
    aa = bb + cc / np.sqrt(2)

    for i in range(n):
        delta_x = bb * np.ones(n)
        delta_x[i] = aa
        x = x1 + delta_x

        fz, gz, x, cx, nAvg = avg_cov_func(func, x, s0, s1, options, consts, BX)
        j = i + 1
        simplex[:, j] = x
        fx_all[j] = fz
        gx_all[:, j] = gz
        g_max[j] = np.max(gz)
        cJ_all[j] = cx
        function_count += nAvg

    # SORT vertices by increasing objective value
    idx = np.argsort(fx_all)
    simplex = simplex[:, idx]
    fx_all = fx_all[idx]
    gx_all = gx_all[:, idx]
    g_max = g_max[idx]
    cJ_all = cJ_all[idx]

    # Initialize best solution
    x_opt = simplex[:, 0].copy()
    f_opt = fx_all[0]
    g_opt = gx_all[:, 0].copy()
    f_old = f_opt
    last_update = function_count

    # Convergence criteria for initial simplex
    cvg_v = 2 * np.max(np.abs((simplex[:, :n] - simplex[:, 1:n+1]) /
                              (simplex[:, :n] + simplex[:, 1:n+1] + 1e-9)))
    cvg_f = abs((fx_all[n] - fx_all[0]) / (fx_all[0] + 1e-9))

    xtx = " simplex :    vertex 1 "
    for i in range(n):
        xtx += f"   vertex {(i+2):1d} "

    iteration += 1
    cvg_hst[:, iteration - 1] = np.concatenate([(simplex[:, 0] - s0) / s1,
                      [fx_all[0], g_max[0], function_count, cvg_v, cvg_f]])

    if msg:
        print('\033[H\033[J', end='')  # Clear screen
        print(" ======================= NMS ============================")
        print(f" iteration                = {iteration:5d}   "
              f"{'*** feasible ***' if np.max(g_opt) <= tol_g else '!!! infeasible !!!'}")
        print(f" function evaluations     = {function_count:5d} of {max_evals:5d}")
        print(f" objective                = {f_opt:11.3e}")
        if n < 10:
            xx = (simplex - s0[:, np.newaxis]) / s1[:, np.newaxis]
            print(xtx)
            for j in range(n):
                xstr = "           " + " ".join(f"{xp:11.3e}" for xp in xx[j,:])
                print(xstr)
            print(" f_A    = " + " ".join(f"{f:11.3e}" for f in fx_all))
            print(" max(g) = " + " ".join(f"{g:11.3e}" for g in g_max))
        print(" ======================= NMS ============================\n")

    # ============================ main loop ============================
    while function_count < max_evals:
        accept_point = False
        move_type = ""

        # Centroid of best n vertices (excluding worst)
        xo = np.mean(simplex[:, :n], axis=1)

        # ----- REFLECT -----
        xr = xo + a_reflect * (xo - simplex[:, n])
        fr, gr, xr, cj, nAvg = avg_cov_func(func, xr, s0, s1, options, consts, BX)
        function_count += nAvg

        if fx_all[0] <= fr < fx_all[n - 1]:  # fr between best and second-worst
            xw, fw, gw, cw = xr, fr, gr, cj
            move_type = 'reflect'
            accept_point = True

        # ----- EXTEND -----
        if not accept_point and fr < fx_all[0]:  # fr better than best
            xe = xo + a_extend * (xr - xo)
            fe, ge, xe, cj, nAvg = avg_cov_func(func, xe, s0, s1, options, consts, BX)
            function_count += nAvg

            if fe < fr:
                xw, fw, gw, cw = xe, fe, ge, cj
                move_type = 'extend'
            else:
                xw, fw, gw, cw = xr, fr, gr, cj
                move_type = 'reflect'
            accept_point = True

        # ----- CONTRACT -----
        if not accept_point and fr > fx_all[n - 1]:  # fr worse than second-worst
            xci = xo - a_contract * (xr - xo)  # inside contraction
            xco = xo + a_contract * (xr - xo)  # outside contraction

            fci, gci, xci, ci, nAvg = avg_cov_func(func, xci, s0, s1, options, consts, BX)
            function_count += nAvg

            fco, gco, xco, co, nAvg = avg_cov_func(func, xco, s0, s1, options, consts, BX)
            function_count += nAvg

            # Optional: optimize contraction step
            if optimize_contraction:
                d = np.array([
                    -np.linalg.norm(xo - simplex[:, n]),
                    -np.linalg.norm(xo - xci),
                    np.linalg.norm(xo - xco),
                    np.linalg.norm(xo - xr)
                ])
                A = np.column_stack([np.ones(4), d, 0.5 * d**2])
                a_coef = np.linalg.solve(A, np.array([fx_all[n], fci, fco, fr]))
                dx = -a_coef[1] / a_coef[2]

                if abs(dx) < d[3] and a_coef[2] > 0:
                    xc_opt = xo + dx * (xr - xo) / d[3]
                    fc_opt, gc_opt, xc_opt, cj, nAvg = avg_cov_func(func, xc_opt, s0, s1,
                                                                      options, consts, BX)
                    function_count += nAvg

                    if fc_opt < min(fci, fco) and fc_opt < fx_all[n - 1]:
                        xw, fw, gw, cw = xc_opt, fc_opt, gc_opt, cj
                        move_type = 'contract opt'
                        accept_point = True

            if not accept_point and fci < fco and fci < fx_all[n - 1]:
                xw, fw, gw, cw = xci, fci, gci, ci
                move_type = 'contract in'
                accept_point = True

            if not accept_point and fco < fci and fco < fx_all[n - 1]:
                xw, fw, gw, cw = xco, fco, gco, co
                move_type = 'contract out'
                accept_point = True

        # ----- ACCEPT or SHRINK -----
        if accept_point:
            # Replace worst point with new point
            simplex[:, n] = xw
            fx_all[n] = fw
            gx_all[:, n] = gw
            g_max[n] = np.max(gw)
            cJ_all[n] = cw
        else:
            # SHRINK all points toward best point
            x1 = simplex[:, 0]
            for i in range(1, n + 1):
                xk = x1 + a_shrink * (simplex[:, i] - x1)
                fk, gk, xk, cj, nAvg = avg_cov_func(func, xk, s0, s1, options, consts, BX)
                simplex[:, i] = xk
                fx_all[i] = fk
                gx_all[:, i] = gk
                g_max[i] = np.max(gk)
                cJ_all[i] = cj
                function_count += nAvg
            move_type = 'shrink'

        # ----- EXPAND elongated (degenerate) simplex -----
        vo = np.full((n, n + 1), np.nan)
        lo = np.full(n + 1, np.nan)

        for i in range(n + 1):
            idx = np.arange(n + 1) != i
            vo[:, i] = np.mean(simplex[:, idx], axis=1)  # centroid of opposite vertices
            lo[i] = np.linalg.norm(simplex[:, i] - vo[:, i])

        elongated_idx = np.where(lo / np.max(lo) < (a_expand - 1))[0]
        for j in elongated_idx:
            xz = vo[:, j] + a_expand * (simplex[:, j] - vo[:, j])
            fz, gz, xz, cj, nAvg = avg_cov_func(func, xz, s0, s1, options, consts, BX)
            simplex[:, j] = xz
            fx_all[j] = fz
            gx_all[:, j] = gz
            g_max[j] = np.max(gz)
            cJ_all[j] = cj
            function_count += nAvg

        # ----- SORT vertices by increasing objective -----
        idx = np.argsort(fx_all)
        simplex = simplex[:, idx]
        fx_all = fx_all[idx]
        gx_all = gx_all[:, idx]
        g_max = g_max[idx]
        cJ_all = cJ_all[idx]

        # Update best solution
        x_opt = simplex[:, 0].copy()
        f_opt = fx_all[0]
        g_opt = gx_all[:, 0].copy()

        if f_opt < f_old:
            last_update = function_count
            f_old = f_opt

        # ----- Convergence criteria -----
        cvg_v = 2 * np.max(np.abs((simplex[:, :n] - simplex[:, 1:n+1]) /
                                  (simplex[:, :n] + simplex[:, 1:n+1] + 1e-9)))
        cvg_f = abs((fx_all[n] - fx_all[0]) / (fx_all[0] + 1e-9))

        iteration += 1
        cvg_hst[:, iteration - 1] = np.concatenate([(x_opt - s0) / s1,
                                                    [f_opt, np.max(g_opt), function_count, cvg_v, cvg_f]])

        # ----- Display progress -----
        if msg:
            elapsed = time.time() - start_time
            secs_left = int((max_evals - function_count) * elapsed / function_count)
            eta = (datetime.now() + timedelta(seconds=secs_left)).strftime('%H:%M:%S')

            print('\033[H\033[J', end='')  # clear screen
            print(" ======================= NMS ============================")
            print(f" iteration                = {iteration:5d}   "
                  f"{'*** feasible ***' if np.max(g_opt) <= tol_g else '!!! infeasible !!!'}")
            print(f"                                             {move_type}")
            print(f" function evaluations     = {function_count:5d} of {max_evals:5d}"
                  f" ({100.0*function_count/max_evals:4.1f}%)")
            print(f" e.t.a.                   = {eta} ")
            print(f" objective                = {f_opt:11.3e}")

            if n < 10:
                xx = (simplex - s0[:, np.newaxis]) / s1[:, np.newaxis]
                print(xtx)
                for j in range(n):
                    xstr = "           " + " ".join(f"{xp:11.3e}" for xp in xx[j,:])
                    print(xstr)
                print(" f_A       " + " ".join(f"{f:11.3e}" for f in fx_all))
                print(" max(g)   =" + " ".join(f"{g:11.3e}" for g in g_max))
                print(" cov(F_A) =" + " ".join(f"{c:11.3e}" for c in cJ_all))
            else:
                xx = (x_opt - s0) / s1
                print(" variables             = " + " ".join(f"{v:11.3e}" for v in xx))
                print(f" max constraint       = {np.max(g_opt):11.3e}")

            print(f" objective convergence    = {cvg_f:11.4e}   tol_f = {tol_f:8.6f}")
            print(f" variable  convergence    = {cvg_v:11.4e}   tol_v = {tol_v:8.6f}")
            print(f" c.o.v. of F_A            = {cJ_all[0]:11.4e}")
            print(" ======================= NMS ============================\n")

        # ----- Plot simplex on surface -----
        if msg > 2:
            ii = int(options[10])
            jj = int(options[11])
            simplex_plot = (simplex - s0[:, np.newaxis]) / s1[:, np.newaxis]

            # Plot simplex as connected triangles (for n=2, plots all 3 vertices)
            if n == 2:
                # Close the triangle by appending first vertex
                x_coords = np.append(simplex_plot[ii, :3], simplex_plot[ii, 0])
                y_coords = np.append(simplex_plot[jj, :3], simplex_plot[jj, 0])
                f_vals = np.append(fx_all[:3], fx_all[0])
                ax.plot(x_coords, y_coords, f_vals ,
                       '-or', alpha=1.0, markersize=6, linewidth=2,
                       markerfacecolor='red', markeredgecolor='darkred')
            plt.draw()
            plt.pause(0.01)

        # ----- Termination checks -----
        if np.max(g_opt) < tol_g and find_feas:
            if msg:
                print(" * Woo Hoo!  Feasible solution found!")
                print(" *           ... and that is all we are asking for.")
            break

        if cvg_v < tol_v or cvg_f < tol_f:
            if msg:
                print(" * Woo Hoo!  Converged solution found!")
                if cvg_v < tol_v:
                    print(" *           convergence in design variables")
                if cvg_f < tol_f:
                    print(" *           convergence in design objective")
                if np.max(g_opt) < tol_g:
                    print(" * Woo Hoo!  Converged solution is feasible")
                else:
                    print(" * Boo Hoo!  Converged solution is NOT feasible!")
                    print(" *            ... Increase options[5] and try, try again ...")
            break

        if function_count - last_update > 0.20 * max_evals:
            if msg:
                print(f" * Hmmm ... Best solution not improved in last "
                      f"{function_count - last_update} evaluations")
                print(" * Increase tol_v (options[1]) or tol_f (options[2]) "
                      "or max_evals (options[4]) and try again")
                if np.max(g_opt) < tol_g:
                    print(" * Woo Hoo!  Best solution is feasible!")
                else:
                    print(" * Boo Hoo!  Best solution is NOT feasible!")
                    print(" *            ... Increase options[5] and try, try again ...")
            break

    # ----- Time-out message -----
    if function_count >= max_evals and msg:
        print(f" * Enough! max evaluations ({max_evals}) exceeded.")
        print(" * Increase tol_v (options[1]) or tol_f (options[2]) "
              "or max_evals (options[4]) and try again")

    # Scale back to original units
    v_opt = (x_opt - s0) / s1

    # ----- Summary -----
    if msg:
        dur = time.time() - start_time
        print(f" *          objective = {f_opt:11.3e}   evals = {function_count}   " f"time = {dur:.2f}s")
        print(" * ----------------------------------------------------------------------------")
        print(" *                v_init      v_lb     <    v_opt     <    v_ub      ")
        print(" * ----------------------------------------------------------------------------")
        for i in range(n):
            eqlb = '=' if v_opt[i] < v_lb[i] + tol_g + 10 * np.finfo(float).eps else ' '
            equb = '=' if v_opt[i] > v_ub[i] - tol_g - 10 * np.finfo(float).eps else ' '
            binding = ' ** binding **' if eqlb == '=' or equb == '=' else ''
            print(f" *  v[{i+1:3d}]  {v_init[i]:11.4f} "
                  f"{v_lb[i]:11.4f} {eqlb} {v_opt[i]:12.5f} {equb} {v_ub[i]:11.4f} {binding}")
        print(" * ---------------------------------------------------------------------------")
        print(" * Constraints:")
        for j, gj in enumerate(np.atleast_1d(g_opt).flatten(), 1):
            tag = " ** binding **" if gj > -tol_g else ""
            if gj > tol_g:
                tag = " ** not ok **"
            print(f" *  g[{j:3d}] = {gj:12.5f}  {tag}")

        print(" *\n * ---------------------------------------------------------------------------")

    if msg > 2:
        plt.figure(103)
        ii = int(options[10])
        jj = int(options[11])
        plt.plot( v_opt[ii], v_opt[jj], f_opt, '-or', markersize=14 )

    # Trim history
    cvg_hist = cvg_hst[:, :iteration].copy()

    elapsed = time.time() - start_time
    completion_time = datetime.now().strftime('%H:%M:%S')
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f' * Completion  : {completion_time} ({elapsed_str})')

    return v_opt, f_opt, g_opt, cvg_hist, 0, 0
