"""
sqp.py 
-----------------------------------------------------------------------------
Sequential Quadratic Programming for Nonlinear Optimization
Depends on: opt_options(), plot_opt_surface()
Use SciPy optimize.minimize package for QP subproblems
In the future: use quadprog package for QP subproblems
-----------------------------------------------------------------------------

Nonlinear optimization with inequality constraints using S.Q.P.

Minimizes f(v) such that g(v) < 0 and v_lb <= v_opt <= v_ub.
- f is a scalar objective function
- v is a vector of design variables
- g is a vector of inequality constraints

References:

S.Rao, Optimization Theory and Applications, 2nd ed, John Wiley, 1984

William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, 
Numerical Recipes in C 
Cambridge University Press, (1992)

H.P. Gavin, Civil & Environmental Eng'g, Duke Univ.
Translation from MATLAB to Python, 2025-11-24

updated ... 
2010 - 2023, 2024-02-03, 2025-01-26, 2025-11-24
"""

import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime, timedelta

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
#from quadprog import solve_qp # ... in the future

from multivarious.utl.plot_opt_surface import plot_opt_surface
from multivarious.utl.opt_options import opt_options
from multivarious.utl.opt_report import opt_report
 

def sqp(func, v_init, v_lb=None, v_ub=None, options_in=None, consts=1.0):
    """
    Sequential Quadratic Programming for nonlinear optimization with inequality constraints.

    Minimizes f(v) such that g(v) < 0 and v_lb <= v_opt <= v_ub.
    Uses finite differences for gradients and BFGS Hessian updates.

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
        Optimal design variables
    f_opt : float
        Optimal objective value
    g_opt : np.ndarray (m,)
        Optimal constraint values
    cvg_hst : np.ndarray (n+5, k)
        Convergence history: [v; f; max(g); func_count; cvg_v; cvg_f] per iteration
    lambda_opt : np.ndarray
        Lagrange multipliers at active constraints
    hess : np.ndarray (n, n)
        Final Hessian matrix approximation
    """

    # ----- options & inputs -----
    v_init = np.asarray(v_init, dtype=float).flatten()
    n = v_init.size

    if v_lb is None or v_ub is None:
        v_lb = -1.0e2 * np.abs(v_init)
        v_ub = +1.0e2 * np.abs(v_init)
    v_lb = np.asarray(v_lb, dtype=float).flatten()
    v_ub = np.asarray(v_ub, dtype=float).flatten()

    # Check for valid bounds
    if np.any(v_ub <= v_lb):
        raise ValueError("v_ub must be greater than v_lb for all parameters")

    # Put initial guess within bounds
    v0 = np.clip(v_init, 0.9 * v_lb, 0.9 * v_ub)

    options   = opt_options(options_in)
    msg       = int(options[0])    # display level
    tol_v     = float(options[1])  # design var convergence tol
    tol_f     = float(options[2])  # objective convergence tol
    tol_g     = float(options[3])  # constraint tol
    max_evals = int(options[4])    # budget
    find_feas = int(options[9])    # stop once a feasible solution is found
    del_min   = float(options[16]) # min parameter change for finite diff
    del_max   = float(options[17]) # max parameter change for finite diff
    options[5] = -1                # no penalty factor involved in SQP

    # ----- scale to [-1, +1] -----
    s0 = (v_lb + v_ub) / 2.0
    s1 = (v_ub - v_lb) / 2.0
    u0 = (v0 - s0) / s1
    u_lb = -1.0 * np.ones(n)
    u_ub = +1.0 * np.ones(n)

    # ----- initialize -----
    function_evals = iteration = 0
    cvg_hst = np.full((n + 5, max(1, max_evals)), np.nan)

    start_time = time.time()

    # First function evaluation
    f0, g0 = func(v0, consts)
    function_evals += 1

#   if not np.isscalar(f0):
#       raise ValueError("Objective returned by func(v,consts) must be a scalar.")
    g0 = np.atleast_1d(g0).astype(float).flatten()
    m = g0.size  # number of constraints

    if msg > 2:
        f_min, f_max, ax = plot_opt_surface(func, v_init, v_lb, v_ub, options, consts, 1003)

    # Initialize gradient and Hessian storage
    OLDU = u0.copy()
    OLDG = g0.copy()
    gradf = np.zeros(n)
    OLDgradf = np.zeros(n)
    gradg = np.zeros((m, n))
    OLDgradg = np.zeros((m, n))
    LAMBDA = np.zeros(m)  # Lagrange multipliers
    HESS = np.eye(n)      # Hessian approximation
    PENALTY = np.ones(m)  # Penalty factors

    # Finite difference step sizes
    CHG = 1e-7 * (1.0 + np.abs(u0))
    GNEW = 1e8 * CHG

    StepLength = 1.0
    end_iterations = False

    # Initialize best solution
    f_opt = float(f0)
    u_opt = u0.copy()
    v_opt = v0.copy()
    g_opt = g0.copy()

    # Initialize current values
    f = float(f0)
    u = u0.copy()
    v = v0.copy()
    g = g0.copy()

    # Save initial state
    cvg_hst[:, iteration] = np.concatenate([ v, 
              [f, np.max(g), function_evals, 1.0, 1.0] ])


    if msg > 1:
        print(" *********************** SQP ****************************")
        print(f" iteration                = {iteration:5d}   "
              f"{'*** feasible ***' if np.max(g0) <= tol_g else '!!! infeasible !!!'}")
        print(f" function evaluations     = {function_evals:5d} of {max_evals:5d}")
        if n < 15:
            print(" variables                 = " + " ".join(f"{v:11.3e}" for v in v0))
        print(f" objective                = {f:11.3e}")
        print(f" max constraint           = {np.max(g0):11.3e}")
        print(" *********************** SQP ****************************")

    # ============================ main loop ============================
    while not end_iterations:

        # ----- Compute gradients via finite differences -----
        oldf = f
        oldg = g.copy()

        # Adaptive step sizing
        CHG = -1.0e-8 / (GNEW + np.finfo(float).eps)
        CHG = np.sign(CHG + np.finfo(float).eps) * np.clip(np.abs(CHG), del_min, del_max)

        for i in range(n):
            temp = u[i]
            u[i] = temp + CHG[i]
            f_fd, g_fd = func(s0+s1*u, consts)
            g_fd = np.atleast_1d(g_fd).astype(float).flatten()  # Ensure proper shape
            function_evals += 1

            # Update best solution if improved
            if np.max(g_fd) < tol_g and f_fd < f_opt:
                if msg > 1:
                    print(' update optimum point')
                f_opt = f_fd
                g_opt = g_fd.copy()
                u_opt = u.copy()

            gradf[i] = (f_fd - oldf) / CHG[i]
            gradg[:, i] = (g_fd - oldg) / CHG[i]
            u[i] = temp

        f = oldf
        g = oldg.copy()

        # Initialize penalty on first iteration
        if iteration == 0:
            PENALTY = (np.finfo(float).eps + np.dot(gradf, gradf)) * np.ones(m) / \
                     (np.sum(gradg**2, axis=1) + np.finfo(float).eps)

        # Compute gradient of augmented Lagrangian
        GOLD = OLDgradf + np.dot(LAMBDA, OLDgradg)
        GNEW = gradf + np.dot(LAMBDA, gradg)
        q = GNEW - GOLD  # change in augmented gradient
        p = u - OLDU     # change in design variables

        # ----- Update Hessian (BFGS) -----
        how = 'regular'
        qp = np.dot(q, p)

        # Ensure Hessian positive definiteness
        if qp < StepLength**2 * 1e-3:
            how = 'modify gradients to ensure Hessian > 0'
            # Modify q to ensure qp > 0
            while qp < -1e-5:
                qp_components = q * p
                idx_min = np.argmin(qp_components)
                q[idx_min] = q[idx_min] / 2
                qp = np.dot(q, p)

            if qp < np.finfo(float).eps * np.linalg.norm(HESS, 'fro'):
                FACTOR = np.dot(gradg.T, g) - np.dot(OLDgradg.T, OLDG)
                FACTOR = FACTOR * (p * FACTOR > 0) * (q * p <= np.finfo(float).eps)
                WT = 1e-2
                if np.max(np.abs(FACTOR)) == 0:
                    FACTOR = 1e-5 * np.sign(p)
                    how = 'small gradients'
                while qp < np.finfo(float).eps * np.linalg.norm(HESS, 'fro') and WT < 1 / np.finfo(float).eps:
                    q = q + WT * FACTOR
                    qp = np.dot(q, p)
                    WT = WT * 2

        # Perform BFGS update if qp > 0
        if qp > np.finfo(float).eps:
            HESS = HESS + np.outer(q, q) / qp - \
                   np.dot(HESS, np.outer(p, p)).dot(HESS) / np.dot(p, HESS).dot(p)
            how = how + ', Hessian update'
        else:
            how = how + ', no Hessian update'

        # ----- Save old values -----
        OLDU = u.copy()
        OLDF = f
        OLDG = g.copy()
        OLDgradf = gradf.copy()
        OLDgradg = gradg.copy()
        OLDLAMBDA = LAMBDA.copy()

        # ----- Solve QP subproblem for search direction -----
        # Append box constraints to nonlinear constraints
        GT = np.concatenate([g, -u + u_lb, u - u_ub])
        gradg_augmented = np.vstack([gradg, -np.eye(n), np.eye(n)])

        # Solve QP: min 0.5*SD'*HESS*SD + gradf'*SD  s.t.  gradg_augmented*SD <= -GT
        SD, lambda_qp, howqp = solve_qp_subproblem(HESS, gradf, gradg_augmented, -GT)

        # Extract Lagrange multipliers for nonlinear constraints only
        LAMBDA = lambda_qp[:m]

        # Update penalty factors (don't change too quickly)
        PENALTY = np.maximum(LAMBDA, 0.5 * (LAMBDA + PENALTY))

        g_max = np.max(g)

        # ----- Line search -----
        infeas = (howqp == 'infeasible')

        # Goal functions for merit-based line search
        GOAL_1 = f + np.sum(PENALTY * (g > tol_g) * g) + 1e-30
        if g_max > tol_g:
            GOAL_2 = g_max
        elif f >= 0:
            GOAL_2 = -1 / (f + 1)
        else:
            GOAL_2 = 0
        if not infeas and f < 0:
            GOAL_2 = GOAL_2 + f - 1

        COST_1 = GOAL_1 + 1.0
        COST_2 = GOAL_2 + 1.0

        if msg > 1:
            print('   alpha      max{g}          COST_1           COST_2')

        StepLength = 2.0

        while (COST_1 > GOAL_1) and (COST_2 > GOAL_2) and (function_evals < max_evals):
            StepLength = StepLength / 2
            if StepLength < 1e-4:
                StepLength = -StepLength  # change direction

            u = OLDU + StepLength * SD
            f, g = func(s0+s1*u, consts)
            g = np.atleast_1d(g).astype(float).flatten()  # Ensure proper shape
            function_evals += 1

            # Update best solution
            if np.max(g) < tol_g and f < f_opt:
                if msg > 1:
                    print(' update optimum')
                f_opt = f
                g_opt = g.copy()
                u_opt = u.copy()

            g_max = np.max(g)
            COST_1 = f + np.sum(PENALTY * (g > 0) * g)
            if g_max > tol_g:
                COST_2 = g_max
            elif f >= 0:
                COST_2 = -1 / (f + 1)
            else:
                COST_2 = 0
            if not infeas and f < 0:
                COST_2 = COST_2 + f - 1

#           print(type(f), np.shape(f))
#           print(type(np.sum(PENALTY * (g > 0) * g)), np.shape(np.sum(PENALTY * (g > 0) * g)))
#           print(type(COST_1), np.shape(COST_1))

            if msg > 1:
                print(f'  {StepLength:9.2e}   {g_max:9.2e}  {COST_1:9.2e} '
                      f'{COST_1/GOAL_1:6.3f} {COST_2:9.2e} {COST_2/GOAL_2:6.3f}')

        # ----- Update Lagrange multipliers -----
        absSL = abs(StepLength)
        LAMBDA = absSL * LAMBDA + (1 - absSL) * OLDLAMBDA
        g_ok = g < -tol_g
        LAMBDA[g_ok] = 0
        # Note: lambda_qp includes box constraints, LAMBDA is only nonlinear constraints

        iteration += 1

        g_max, idx_max_g = np.max(g), np.argmax(g)
        cvg_f = abs(absSL * np.dot(gradf, SD) / (f + 1e-9)) 
        cvg_v = np.max(np.abs(absSL * SD / (u + 1e-9)))

        # ----- Display progress -----
        if msg > 1:
            elapsed = time.time() - start_time
            secs_left = int((max_evals - function_evals) * elapsed / function_evals)
            eta = (datetime.now() + timedelta(seconds=secs_left)).strftime('%H:%M:%S')
            # print('\033[H\033[J', end='')  # clear screen
            print("\n *********************** SQP ****************************")
            print(f" iteration                = {iteration:5d}   "
                  f"{'*** feasible ***' if g_max <= tol_g and np.all(u >= -1) and np.all(u <= 1) else '!!! infeasible !!!'}")
            print(f" function evaluations     = {function_evals:5d} of {max_evals:5d}"
                  f" ({100.0*function_evals/max_evals:4.1f}%)")
            print(f" e.t.a.                   = {eta} ")
            if n < 15:
                print(f" variables                = " + " ".join(f"{v:11.3e}" for v in (u-s0)/s1))
            print(f" objective                = {f:11.3e}")
            print(f" max constraint           = {g_max:11.3e}  ({idx_max_g+1})")
            print(f" Step Size                = {StepLength:11.3e}")
            print(f" BFGS method              : {how}")
            print(f" QP method                : {howqp}")
            print(f" objective convergence    = {cvg_f:11.4e}   tol_f = {tol_f:8.6f}")
            print(f" variable  convergence    = {cvg_v:11.4e}   tol_v = {tol_v:8.6f}")
            print(" *********************** SQP ****************************\n")

        # Save convergence history
        cvg_hst[:, iteration] = np.concatenate([ s0+s1*u,
            [f, g_max, function_evals, cvg_v, cvg_f] ])

        # Plot current point
        if msg > 2:
            ii = int(options[10])
            jj = int(options[11])
            v_plot = s0 + s1*u
            ax.plot([v_plot[ii]], [v_plot[jj]], f ,
                   'ro', alpha=1.0, markersize=8, linewidth=4,
                   markerfacecolor='red', markeredgecolor='darkred')
            plt.draw()
            plt.pause(0.10)

        # ----- Check convergence -----
        cvg_f = abs(absSL * np.dot(gradf, SD) / (f + 1e-9)) 
        cvg_v = np.max(np.abs(absSL * SD / ( s0+s1+u + 1e-9 )))

        feasible = converged = stalled = False # convergence criteria

        if (g_max < tol_g and howqp != 'infeasible'):
            feasible = True

        if feasible and find_feas: 
            converged = True

        if (cvg_v < tol_v and cvg_f < tol_f):
            converged = True

        if converged:
            end_iterations = True

        """
        if feasible and converged: 
            print(f' * Woo Hoo!  Converged solution found in {function_evals} function evaluations!')
            if cvg_v < tol_v:
                print(' *           convergence in design variables')
            if cvg_f < tol_f:
                print(' *           convergence in design objective')
            if g_max < tol_g:
                print(' * Woo Hoo!  Converged solution is feasible')
        else:
            print(' * Boo Hoo!  Converged solution is NOT feasible!')
        else:
            if g_max > tol_g:
                print(' * Boo Hoo Hoo!  No feasible solution found.')

        elif function_evals >= max_evals:
            u_opt = OLDU
            f_opt = OLDF
            g_opt = OLDG
            print(f' * Enough! Maximum number of function evaluations ({max_evals}) exceeded')
            print(' * Increase tol_v (options[1]), tol_f (options[2]), or max_evals (options[4])')
            print(' * and try, try, try again!')
            end_iterations = True
        """

    # ============================ end main loop ============================

    # If better feasible solution was found during search, use it
    if f_opt < f and np.max(g_opt) < tol_g:
        u = u_opt
        f = f_opt
        g = g_opt

    # Scale back to original units
    v_opt = s0 + s1*u
    f_opt = f
    g_opt = g

    iteration += 1
    # Save convergence history
    cvg_hst[:, iteration] = np.concatenate([ v_opt,
            [f_opt, np.max(g_opt), function_evals, cvg_v, cvg_f] ])

    # ----- Summary -----
    dur = time.time() - start_time
    if msg:
        opt_report(v_init, v_opt, f_opt, g_opt, v_lb, v_ub, tol_v, tol_f, tol_g,
                   lambda_qp, start_time, function_evals, max_evals,
                   find_feas, feasible, converged, stalled )

    """
        print(f" * objective = {f_opt:11.3e} ")
        print(" * ----------------------------------------------------------------------------")
        print(" *                v_init      v_lb     <    v_opt     <    v_ub      lambda")
        print(" * ----------------------------------------------------------------------------")
        for i in range(n):
            eqlb = '=' if v_opt[i] < v_lb[i] + tol_g + 1e-6 else ' '
            equb = '=' if v_opt[i] > v_ub[i] - tol_g - 1e-6 else ' '
            lulb = ''
            if eqlb == '=' and lambda_qp != None:
                lulb = f'{lambda_qp[m + i]:12.5f}'
            elif equb == '=' and lambda_qp != None:
                lulb = f'{lambda_qp[m + n + i]:12.5f}'
            print(f" *  v[{i+1:3d}]  {v_init[i]:11.4f} "
                  f"{v_lb[i]:11.4f} {eqlb} {v_opt[i]:12.5f} {equb} {v_ub[i]:11.4f} {lulb}")
        print(" * ----------------------------------------------------------------------------")
        print(" * Constraints:")
        for j in range(m):
            binding = ''
            if lambda_qp[j] > 0:
                binding = '    ** binding **'
            if g_opt[j] >= tol_g:
                binding = '    ** not ok  **'
            print(f" *  g[{j+1:3d}] = {g_opt[j]:12.5f}      "
                  f"lambda[{j+1:3d}] = {lambda_qp[j]:12.5f}   {binding}")

        active_cstr = np.where(LAMBDA > 0)[0]
        if len(active_cstr) > 0:
            print(" * Active Constraints: " + "  ".join(f"{i+1:2d}" for i in active_cstr))
        print(" * ----------------------------------------------------------------------------")
    """

    if msg > 2:
        plt.figure(1003)
        ii = int(options[10])
        jj = int(options[11])
        plt.plot( v_opt[ii], v_opt[jj], f_opt, '-or', markersize=14 )

    # Trim history
    cvg_hst = cvg_hst[:, :iteration]

    return v_opt, f_opt, g_opt, cvg_hst, lambda_qp, HESS


def solve_qp_subproblem(H, f, A, b):
    '''
    Solve QP subproblem using scipy.optimize.minimize

    Solves: min 0.5*x'*H*x + f'*x  subject to  A*x <= b

    Parameters
    ----------
    H : np.ndarray (n, n)
        Hessian matrix
    f : np.ndarray (n,)
        Linear term gradient
    A : np.ndarray (m, n)
        Constraint matrix (inequality)
    b : np.ndarray (m,)
        Constraint RHS

    Returns
    -------
    x : np.ndarray (n,)
        Solution (search direction)
    lambda_vals : np.ndarray (m,)
        Lagrange multipliers (approximated from active constraints)
    how : str
        Status message
    '''
    n = H.shape[0]
    m = len(b)
    
    # Objective function: 0.5*x'*H*x + f'*x
    def objective(x):
        return 0.5 * np.dot(x, H.dot(x)) + np.dot(f, x)
    
    # Gradient of objective
    def grad_objective(x):
        return H.dot(x) + f
    
    # Inequality constraints: A*x - b <= 0
    constraints = []
    for i in range(m):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: b[i] - np.dot(A[i, :], x),
            'jac': lambda x, i=i: -A[i, :]
        })
    
    # Initial guess
    x0 = np.zeros(n)
    
    try:
        # Make H positive definite by adding small regularization if needed
        try:
            # Test if H is positive definite
            cho_factor(H + 1e-10 * np.eye(n))
            H_reg = H + 1e-10 * np.eye(n)
        except np.linalg.LinAlgError:
            # H is not positive definite, add more regularization
            H_reg = H + 1e-6 * np.eye(n)
        
        # Update objective with regularized H
        def objective_reg(x):
            return 0.5 * np.dot(x, H_reg.dot(x)) + np.dot(f, x)
        
        def grad_objective_reg(x):
            return H_reg.dot(x) + f
        
        # Solve with SLSQP
        result = minimize(
            objective_reg,
            x0,
            method='SLSQP',
            jac=grad_objective_reg,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
        )
        
        if result.success:
            x = result.x
            how = 'ok'
            
            # Estimate Lagrange multipliers from active constraints
            lambda_vals = np.zeros(m)
            tol_active = 1e-6
            
            for i in range(m):
                constraint_val = b[i] - np.dot(A[i, :], x)
                if abs(constraint_val) < tol_active:  # Active constraint
                    # Approximate multiplier (should be from KKT conditions)
                    # For now, use a simple approximation
                    lambda_vals[i] = max(0, -np.dot(grad_objective_reg(x), A[i, :]))
            
        else:
            # Optimization failed
            x = result.x if hasattr(result, 'x') else x0
            lambda_vals = np.zeros(m)
            
            # Check what kind of failure
            if 'infeasible' in result.message.lower():
                how = 'infeasible'
            elif 'unbounded' in result.message.lower():
                how = 'unbounded'  
            else:
                how = 'ill-posed'
        
        return x, lambda_vals, how
        
    except Exception as e:
        # Fallback: return zero solution
        x = np.zeros(n)
        lambda_vals = np.zeros(m)
        how = 'error: ' + str(e)[:20]
        return x, lambda_vals, how


"""
def solve_qp_subproblem(H, f, A, b):
    '''
    Solve QP using quadprog: min 0.5*x'*H*x + f'*x  s.t.  A*x <= b
    
    quadprog solves: min 0.5*x'*G*x - a'*x  s.t.  C'*x >= b
    
    Conversion:
    G = H
    a = -f
    C' = -A' ? C = -A.T
    b_qp = -b

    sudo pipx install quadprog --include-deps
    '''
    try:
        G = H
        a = -f
        C = -A.T
        b_qp = -b
        
        # solve_qp returns (solution, f_value, xu, iterations, lagrangian, iact)
        result = solve_qp(G, a, C, b_qp, meq=0)
        
        x = result[0]           # solution
        lagrangian = result[4]  # Lagrange multipliers
        
        how = 'ok'
        return x, lagrangian, how
        
    except ValueError as e:
        n = H.shape[0]
        x = np.zeros(n)
        lambda_vals = np.zeros(len(b))
        
        if 'infeasible' in str(e).lower():
            how = 'infeasible'
        elif 'unbounded' in str(e).lower():
            how = 'unbounded'
        else:
            how = 'ill-posed'
        
        return x, lambda_vals, how
"""

