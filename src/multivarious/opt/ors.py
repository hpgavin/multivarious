# ors.py
# -----------------------------------------------------------------------------
# Optimized Step Size Random Search (ORS)
# Translation of Henri P. Gavin's ORSopt.m (Duke CEE).
# Depends on: opt_options(), box_constraint(), avg_cov_func()
# -----------------------------------------------------------------------------

from __future__ import annotations
import time
import numpy as np

from multivarious.utils.opt_options import opt_options
from multivarious.utils.box_constraint import box_constraint
from multivarious.utils.avg_cov_func import avg_cov_func
from multivarious.utils.plot_opt_surface import plot_opt_surface
from matplotlib import pyplot as plt



'''  Commented out for testing, replaced with version above.
from __future__ import annotations
import time
import numpy as np

from opt_options import opt_options
from box_constraint import box_constraint
from avg_cov_func import avg_cov_func
from matplotlib import pyplot as plt
from plot_opt_surface import plot_opt_surface
'''

def ors(func, v_init, v_lb=None, v_ub=None, options_in=None, consts=1.0):
    """
    Optimized Step Size Random Search (inequality constraints via penalties).

    Parameters
    ----------
    func : callable
        Signature: f, g = func(v, consts).  f is scalar, g is (m,) constraints (g<0 feasible).
        v is in *original* units (not scaled).
    v_init : array-like (n,)
        Initial guess.
    v_lb, v_ub : array-like (n,), optional
        Lower/upper bounds on v. If omitted, wide bounds are used (±1e2*|v_init|).
    options_in : array-like, optional
        See opt_options() for the 19 parameters (same positions as MATLAB).
    consts : any
        Passed through to `func`.

    Returns
    -------
    v_opt : np.ndarray (n,)
    f_opt : float
    g_opt : np.ndarray (m,)
    cvg_hst : np.ndarray (n+5, k)
        Columns store [v; f; max(g); func_count; cvg_v; cvg_f] per iteration.
    """

    # ----- options & inputs -----
    v_init = np.asarray(v_init, dtype=float).flatten()
    n = v_init.size

    if v_lb is None or v_ub is None:
        v_lb = -1.0e2 * np.abs(v_init)
        v_ub = +1.0e2 * np.abs(v_init)
    v_lb = np.asarray(v_lb, dtype=float).flatten()
    v_ub = np.asarray(v_ub, dtype=float).flatten()

    options = opt_options(options_in)
    msglev    = int(options[0])   # display level
    tol_v     = float(options[1]) # design var convergence tol
    tol_f     = float(options[2]) # objective convergence tol
    tol_g     = float(options[3]) # constraint tol
    max_evals = int(options[4])   # budget
    # options[5], [6] handled inside avg_cov_func
    find_feas = bool(options[9])  # stop once feasible

    # ----- scale to [-1, +1] (as in MATLAB) -----
    s0 = (v_lb + v_ub) / (v_lb - v_ub)
    s1 = 2.0 / (v_ub - v_lb)
    v1 = s0 + s1 * v_init
    v1 = np.clip(v1, -1.0, 1.0)

    # book-keeping
    function_count = 0
    iteration = 1
    cvg_hst = np.full((n + 5, max(1, max_evals)), np.nan)
    fa = np.zeros(4)  # augmented costs for up to 4 evaluations
    BX = 1           # enforce bounds inside avg_cov_func

    # ----- analyze initial guess -----
    fv, gv, v1, cJ, nAvg = avg_cov_func(func, v1, s0, s1, options, consts, BX)
    function_count += nAvg
    if not np.isscalar(fv):
        raise ValueError("Objective returned by func(v,consts) must be a scalar.")
    gv = np.atleast_1d(gv).astype(float).flatten()

    # initial records
    f_opt = float(fv)
    v_opt = v1.copy()
    g_opt = gv.copy()

    cvg_v = 1.0
    cvg_f = 1.0
    cvg_hst[:, iteration - 1] = np.concatenate([(v_opt - s0) / s1,
                         [f_opt, np.max(g_opt), function_count, cvg_v, cvg_f]])

    if msglev > 2:
        f_min, f_max, ax = plot_opt_surface(func,(v_init-s0)/s1,v_lb,v_ub,options,consts,103)
        
    # search parameters
    sigma = 0.300  # step scale
    nu = 1.0       # exponent in sigma schedule
    t0 = time.time()

    # initialize four points (v1 already done)
    fa[0] = f_opt
    v2 = v1.copy(); g2 = g_opt.copy()
    v3 = v1.copy(); g3 = g_opt.copy()
    v4 = v1.copy(); g4 = g_opt.copy(); fa[3] = fa[0]

    last_update = function_count

    # ============================ main loop ============================
    while function_count < max_evals:
        # random direction
        r = sigma * np.random.randn(n)

        # +1 step
        a2, _ = box_constraint(v1, r)
        v2 = v1 + a2 * r
        fa2, g2, v2, c2, nAvg = avg_cov_func(func, v2, s0, s1, options, consts, BX)
        function_count += nAvg
        fa[1] = fa2

        # decide direction for second probe (+2 or -1)
        step = +2.0 if fa[1] < fa[0] else -1.0
        a3, _ = box_constraint(v1, step * r)
        v3 = v1 + a3 * step * r
        fa3, g3, v3, c3, nAvg = avg_cov_func(func, v3, s0, s1, options, consts, BX)
        function_count += nAvg
        fa[2] = fa3

        # fit local quadratic along r using (0, dv2, dv3)
        dv2 = np.linalg.norm(v2 - v1) / (np.linalg.norm(r) + 1e-16)
        dv3 = np.linalg.norm(v3 - v1) / (np.linalg.norm(r) + 1e-16)
        # regularization 
        i3 = 1e-9 * np.eye(3)
        A = np.array([[0.0,         0.0, 1.0],
                      [0.5*dv2**2,  dv2, 1.0],
                      [0.5*dv3**2,  dv3, 1.0]], dtype=float) + i3
        a, b, c = np.linalg.solve(A, fa[:3])

        quad_update = False
        if a > 0.0:                 # curvature is positive!
            d = -b / a              # try to go to the zero-slope point
            a4, _ = box_constraint(v1, d * r)
            v4 = v1 + a4 * d * r
            fa4, g4, v4, c4, nAvg = avg_cov_func(func, v4, s0, s1, options, consts, BX)
            function_count += nAvg
            fa[3] = fa4
            quad_update = True

        if msglev > 2:              # plot values on the surface 
            p1 = (v1-s0)/s1
            p2 = (v2-s0)/s1
            p3 = (v3-s0)/s1
            p4 = (v4-s0)/s1
            ii = int(options[10])
            jj = int(options[11])
            ax.plot( [p2[ii],p3[ii]], [p2[jj],p3[jj]], [fa[1],fa[2]], 
                '-o', alpha=1.0, linewidth=3, markersize=6, color='red',
                 markerfacecolor='red', markeredgecolor='darkred')
            if quad_update:
                ax.plot( [p2[ii],p4[ii]], [p2[jj],p4[jj]], [fa[1],fa[3]],
                '-o', alpha=1.0, linewidth=3, markersize=6, color='red',
                 markerfacecolor='red', markeredgecolor='darkred')
            #plt.draw()

        # choose best of the four 
        i_min = int(np.argmin(fa))
        if i_min == 0:
            pass
        elif i_min == 1:
            v1, g1, c1 = v2, g2, c2
        elif i_min == 2:
            v1, g1, c1 = v3, g3, c3
        else:
            v1, g1, c1 = v4, g4, c4

        if i_min > 0:
            # shrink scope as evaluations proceed
            sigma = sigma * (1.0 - function_count / max_evals)**nu
        v1 = np.clip(v1, -1.0, 1.0)

        # update incumbent if improved
        if fa[i_min] < f_opt:
            v_opt = v1.copy()
            f_opt = float(fa[i_min])
            g_opt = np.atleast_1d(g1).astype(float).flatten()

            # convergence metrics vs last recorded iteration
            prev = cvg_hst[:n, iteration - 1]
            prev_f = cvg_hst[n, iteration - 1]
            vv = (v_opt - s0) / s1
            cvg_v = (np.linalg.norm(prev - vv) /
                     (np.linalg.norm(vv) + 1e-16)) if iteration >= 1 and np.all(np.isfinite(prev)) else 1.0
            cvg_f = (abs(prev_f - f_opt) / (abs(f_opt) + 1e-16)) if iteration >= 1 and np.isfinite(prev_f) else 1.0

            last_update = function_count
            iteration += 1
            cvg_hst[:, iteration - 1] = np.concatenate([vv,
                                                        [f_opt, np.max(g_opt), function_count, cvg_v, cvg_f]])

            if msglev:
                elapsed = time.time() - t0
                rate = function_count / max(elapsed, 1e-9)
                remaining = max_evals - function_count
                eta_sec = int(remaining / max(rate, 1e-9))
                print(" -+-+-+-+-+-+-+-+-+-+-+- ORS +--+-+-+-+-+-+-+-+-+-+-+-+-+")
                print(f" iteration                = {iteration:5d}   "
                      f"{'*** feasible ***' if np.max(g_opt) <= tol_g else '!!! infeasible !!!'}")
                print(f" function evaluations     = {function_count:5d} of {max_evals:5d}"
                      f" ({100.0*function_count/max_evals:4.1f}%)")
                print(f" e.t.a.                   = ~{eta_sec//60}m{eta_sec%60:02d}s")
                print(f" objective                = {f_opt:11.3e}")
                print(" variables                = " + " ".join(f"{v:11.3e}" for v in vv))
                print(f" max constraint           = {np.max(g_opt):11.3e}")
                print(f" Convergence F            = {cvg_f:11.4e}   tolF = {tol_f:8.6f}")
                print(f" Convergence X            = {cvg_v:11.4e}   tolX = {tol_v:8.6f}")
                if quad_update:
                    print(f" *** quadratic update ***    a = {a:9.2e} sigma = {sigma:5.3f}")
                print("\n")

        # termination checks
        if np.max(g_opt) < tol_g and find_feas:
            if msglev:
                print(" * Woo Hoo!  Feasible solution found! ")
                print(" *           ... and that is all we are asking for.")
            break                  # ... and that's all we want

        if iteration > 1 and (cvg_v < tol_v or cvg_f < tol_f) and sigma < 0.1:
            if msglev:
                print(" * Woo Hoo!  Converged solution found!")
            if cvg_v < tol_v:
                print(" *           convergence in design variables") 
            if cvg_f < tol_f:
                print(" *           convergence in design objective") 
            if np.max(g_opt) < tol_g:
                print(" * Woo Hoo!  Converged solution is feasible") 
            else:
                print(" * Boo Hoo!  Converged solution is NOT feasible!") 
                print(" *            ... Increase options[6] and try, try again ...")
            break

    # time-out message
    if function_count >= max_evals and msglev:
        print(f" * Enough! max evaluations ({max_evals}) exceeded. \n"
              " * Increase tol_v (options[1]) or max_evals (options[4]) and try again.")

    # scale back to original units
    v_init_out = (s0 + s1 * v_init - s0) / s1  # = v_init (kept for parity)
    v_opt_out = (v_opt - s0) / s1

    # print summary (compact)
    if msglev:
        dur = time.time() - t0
        print(f" *          objective = {f_opt:11.3e}   evals = {function_count}   "
              f"time = {dur:.2f}s")
        print(" *  ----------------------------------------------------------------")
        print(" *                v_init        v_lb     <    v_opt    <    v_ub")
        print(" *  ----------------------------------------------------------------")
        for i in range(n):
            print(f" *  v[{i+1:3d}]  {v_init_out[i]:12.5f}  {v_lb[i]:12.5f}  {v_opt_out[i]:12.5f}  {v_ub[i]:12.5f}")
        print(" *  ----------------------------------------------------------------")
        print(" * Constraints:")
        for j, gj in enumerate(np.atleast_1d(g_opt).flatten(), 1):
            tag = " ** binding ** " if gj > -tol_g else ""
            if gj > tol_g:
                tag = " ** not ok ** "
            print(f" *  g[{j:3d}] = {gj:12.5f}{tag}")

    # close history by adding a final column mirroring the last iteration
    k = max(1, iteration)
    cvg_hist = cvg_hst[:, :k].copy()
    if not np.isfinite(cvg_hist[-1, -1]):
        cvg_hist[-1, -1] = cvg_hist[-1, max(0, k-2)]

    return v_opt_out, f_opt, g_opt, cvg_hist

