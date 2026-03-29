"""
sqp.py - Sequential Quadratic Programming 
==================================================================

Nonlinear optimization with inequality constraints using Sequential
Quadratic Programming with a robust active-set QP solver.

Minimizes f(v) such that g(v) < 0 and v_lb <= v_opt <= v_ub.
- f is a scalar objective function
- v is a vector of design variables
- g is a vector of inequality constraints


The embedded QP solver uses active-set methods with QR factorization
for numerical stability - much more robust than scipy.optimize for
ill-conditioned problems.

References:

S.Rao, Optimization Theory and Applications, 2nd ed, John Wiley, 1984

William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery, 
Numerical Recipes in C 
Cambridge University Press, (1992)

Translation from MATLAB to Python, 2025-11-24
Original by Andy Grace (MathWorks) and H.P. Gavin (Duke University)

updated ... 
2010 - 2023, 2024-02-03, 2025-01-26, 2025-11-24, 2026-03-29
"""

import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime, timedelta

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.linalg import qr

from multivarious.utl.plot_opt_surface import plot_opt_surface
from multivarious.utl.opt_options import opt_options
from multivarious.utl.opt_report import opt_report


def sqp(func, v_init, v_lb=None, v_ub=None, options_in=None, consts=1.0):
    """
    Sequential Quadratic Programming
    for nonlinear optimization with inequality constraints.
    
    Minimizes f(v) such that g(v) < 0 and v_lb <= v <= v_ub.
    Uses finite differences for gradients and BFGS Hessian updates.
    
    Parameters
    ----------
    func : callable
        Objective function: f, g = func(v, consts)
        Returns scalar f and constraint vector g (g < 0 is feasible)
        v is in *original* units (not scaled).
    v_init : ndarray, shape (n,)
        Initial guess for design variable values
    v_lb, v_ub : ndarray, shape (n,), optional
        Lower/upper bounds (default: ±100*|v_init|)
    options_in : array-like, optional
        Optimization settings (see opt_options)
    consts : any
        Constants passed to 'func`.
    
    Returns
    -------
    v_opt : ndarray
        Optimal design variables
    f_opt : float
        Optimal value of objective function
    g_opt : ndarray
        Constraint values at optimum
    cvg_hst : np.ndarray (n+5, k)
        Convergence history: [v; f; max(g); func_count; cvg_v; cvg_f] per iteration
    lambda_opt : ndarray
        Lagrange multipliers
    hess : ndarray (n, n)
        Final Hessian matrix approximation
    """
    
    # Options and inputs 
    v_init = np.asarray(v_init, dtype=float).flatten()
    n = len(v_init)
    
    if v_lb is None or v_ub is None:
        v_lb = -1.0e2 * np.abs(v_init)
        v_ub = +1.0e2 * np.abs(v_init)
    v_lb = np.asarray(v_lb, dtype=float).flatten()
    v_ub = np.asarray(v_ub, dtype=float).flatten()

    # Check for valid bounds 
    if np.any(v_ub <= v_lb):
        raise ValueError("v_ub must be greater than v_lb for all parameters")
    
    # Clip initial guess to bounds
    v0 = np.clip(v_init, 0.9*v_lb, 0.9*v_ub)
    
    # Options
    options = opt_options(options_in)
    options    = opt_options(options_in)
    msg        = int(options[0])    # display level
    tol_v      = float(options[1])  # design var convergence tol
    tol_f      = float(options[2])  # objective convergence tol
    tol_g      = float(options[3])  # constraint tol
    max_evals  = int(options[4])    # budget
    options[5] = -1                 # no penalty factor involved in SQP
    find_feas  = int(options[9])    # stop once a feasible solution is found
    del_min    = float(options[16]) # min parameter change for finite diff
    del_max    = float(options[17]) # max parameter change for finite diff

    # Scale design variables from v_lb < v < v_ub to -1 < u < +1
    s0 = (v_lb + v_ub) / 2.0
    s1 = (v_ub - v_lb) / 2.0
    u0 = (v0 - s0) / s1
    u_lb = -1.0 * np.ones(n)
    u_ub = +1.0 * np.ones(n)
   
    # Initialize
    function_evals = iteration = 0
    start_time = time.time()
    
    # First function evaluation
    f0, g0 = func(v0, consts)
    function_evals += 1

    if not np.isscalar(f0):
        raise ValueError("Objective returned by func(v,consts) must be a scalar.")
    g0 = np.atleast_1d(g0).astype(float).flatten()
    m = len(g0)  # number of constraints

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

    cvg_hst = np.full((n + 5, max_evals), np.nan)

    # Finite difference step sizes
    CHG = 1e-7 * (np.ones(n) + np.abs(u0))
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
    cvg_hst[:, 0] = np.concatenate([ v,
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
        
        # Compute gradients via finite differences
        oldf = f
        oldg = g.copy()
        
        # Adaptive step sizing
        CHG = -1.0e-8 / (GNEW + np.finfo(float).eps)
        CHG = np.sign(CHG + np.finfo(float).eps) * np.clip(np.abs(CHG), del_min, del_max)
        
        # Finite difference gradients
        for gidx in range(n):
            temp = u[gidx]
            u[gidx] = temp + CHG[gidx]
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
            
            gradf[gidx] = (f_fd - oldf) / CHG[gidx]
            gradg[:, gidx] = (g_fd - oldg) / CHG[gidx]
            u[gidx] = temp

        f = oldf
        g = oldg.copy()
        
        # Initialize penalty on first iteration
        if iteration == 1:
            PENALTY = (np.finfo(float).eps + np.dot(gradf, gradf)) * np.ones(m) / \
                     (np.sum(gradg**2, axis=1) + np.finfo(float).eps)
        
        # Compute gradient of augmented Lagrangian
        GOLD = OLDgradf + LAMBDA @ OLDgradg
        GNEW = gradf + LAMBDA @ gradg
        q = GNEW - GOLD  # change in augmented gradient
        p = u - OLDU     # change in design variables

        # Update Hessian (BFGS)
        how = 'regular'
        qp = np.dot(q, p)
        
        # Ensure Hessian positive definiteness
        if qp < StepLength**2 * 1e-3:
            how = 'modify gradients to ensure Hessian > 0,'
            # Modify q to ensure qp > 0
            while qp < -1e-5:
                qp_components = q * p
                qp_idx_min = np.argmin(qp_components)
                q[qp_idx_min] = q[qp_idx_min] / 2
                qp = np.dot(q, p)
            
            if qp < np.finfo(float).eps * np.linalg.norm(HESS, 'fro'):
                FACTOR = gradg.T @ g - OLDgradg.T @ OLDG
                FACTOR = FACTOR * (p * FACTOR > 0) * (q * p <= np.finfo(float).eps)
                WT = 1e-2
                if np.max(np.abs(FACTOR)) == 0:
                    FACTOR = 1e-5 * np.sign(p)
                    how = 'small gradients,'
                while qp < np.finfo(float).eps * np.linalg.norm(HESS, 'fro') and WT < 1/np.finfo(float).eps:
                    q = q + WT * FACTOR
                    qp = np.dot(q, p)
                    WT = WT * 2
        
        # Perform BFGS update if qp > 0
        if qp > np.finfo(float).eps:
            HESS =  HESS + np.outer(q, q) / qp - \
                   (HESS @ np.outer(p, p) @ HESS.T) / (p @ HESS @ p)
            how = how + ' Hessian update.'
        else:
            how = how + ' no Hessian update.'
        
        # Save old values 
        OLDU = u.copy()
        OLDF = f
        OLDG = g.copy()
        OLDgradf = gradf.copy()
        OLDgradg = gradg.copy()
        OLDLAMBDA = LAMBDA.copy()

        # Set up QP subproblem
        # Append box constraints to nonlinear constraints
        SDi = np.zeros(n)
        GT = np.concatenate([g, -u + u_lb, u - u_ub])
        gradg_augmented = np.vstack([gradg, -np.eye(n), np.eye(n)])
        
        # Solve QP subproblem for search direction 
        # min 0.5*SD'*HESS*SD + gradf'*SD  s.t.  gradg_augmented*SD <= -GT
        SD, lambda_qp, howqp = mwQP(HESS, gradf, gradg_augmented, -GT, None, None, SDi) 
#       SD, lambda_qp, howqp = scipy_qp(HESS, gradf, gradg_augmented, -GT)
        
        # Extract Lagrange multipliers for nonlinear constraints only
        LAMBDA = lambda_qp[:m]

        # Update penalty factors (don't change too quickly)
        PENALTY = np.maximum(LAMBDA, 0.5 * (LAMBDA + PENALTY))
        
        g_max = np.max(g)

        # Line search
        infeas = (howqp[0] == 'infeasible')
        
        # Goal functions for merit-based line search
        GOAL_1 = f + np.sum(PENALTY * (g > tol_g) * g) + 1e-30
        
        if g_max > tol_g:
            GOAL_2 = g_max
        elif f >= 0:
            GOAL_2 = -1.0 / (f + 1.0)
        else:
            GOAL_2 = 0
        if not infeas and f < 0:
            GOAL_2 = GOAL_2 + f - 1.0
        
        COST_1 = GOAL_1 + 1.0
        COST_2 = GOAL_2 + 1.0
        
        if msg > 1:
            print('   alpha      max{g}          COST_1           COST_2')
        
        StepLength = 2.0
        
        while (COST_1 > GOAL_1) and (COST_2 > GOAL_2) and (function_evals < max_evals):
            StepLength = StepLength / 2.0
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
                COST_2 = -1.0 / (f + 1.0)
            else:
                COST_2 = f
            if not infeas and f < 0:
                COST_2 = COST_2 + f - 1.0
            
            if msg > 1:
                print(f'  {StepLength:9.2e}   {g_max:9.2e}  {COST_1:9.2e} '
                      f'{COST_1/GOAL_1:6.3f} {COST_2:9.2e} {COST_2/GOAL_2:6.3f}')
       
        u = OLDU + StepLength * SD  # Update solution (again??)
        
        # Update Lagrange multipliers 
        absSL = abs(StepLength)
        LAMBDA = absSL * LAMBDA + (1 - absSL) * OLDLAMBDA
        g_ok = g < -tol_g
        LAMBDA[g_ok] = 0
        # Note: lambda_qp includes box constraints, LAMBDA is only nonlinear constraints

        iteration += 1

        g_max, idx_max_g = np.max(g), np.argmax(g)
        cvg_v = np.max(np.abs(absSL * SD / (u + 1e-9)))
        cvg_f = abs(absSL * np.dot(gradf, SD) / (f + 1e-9))

        # Save convergence history
        cvg_hst[:, iteration] = np.concatenate([ s0+s1*u,
            [f, g_max, function_evals, cvg_v, cvg_f] ])

        # Display progress 
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

        # Check convergence
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
    
    # Final report
    if msg:
        opt_report(v_init, v_opt, f_opt, g_opt, v_lb, v_ub, tol_v, tol_f, tol_g,
                   lambda_qp, start_time, function_evals, max_evals,
                   find_feas, feasible, converged, stalled )

    if msg > 2:
        plt.figure(1003)
        ii = int(options[10])
        jj = int(options[11])
        plt.plot( v_opt[ii], v_opt[jj], f_opt, '-or', markersize=14 )
   
    # Trim history
    cvg_hst = cvg_hst[:, :iteration]
    
    return v_opt, f_opt, g_opt, cvg_hst, LAMBDA, HESS


def mwQP(H, f, A, b, vlb=None, vub=None, x0=None, neqcstr=0, verbosity=-1):
    """
    Embedded Quadratic Programming solver using active-set method.
    
    Solves: min 0.5*x'*H*x + f'*x  subject to  A*x <= b
    
    This is MathWorks' QP solver - much more robust than scipy for
    ill-conditioned problems.
    
    Parameters
    ----------
    H : ndarray, shape (n, n)
        Hessian matrix
    f : ndarray, shape (n,)
        Linear term
    A : ndarray, shape (m, n)
        Inequality constraint matrix
    b : ndarray, shape (m,)
        Inequality constraint RHS
    vlb, vub : ndarray, optional
        Variable bounds (not used in this interface)
    x0 : ndarray, shape (n,), optional
        Initial guess (default: zeros)
    neqcstr : int, optional
        Number of equality constraints (first neqcstr rows of A)
    verbosity : int, optional
        Verbosity level
    
    Returns
    -------
    X : ndarray
        Solution vector
    lambda_qp : ndarray
        Lagrange multipliers
    how : str
        Status message
    """
    
    # Get dimensions
    n = len(f)
    f = np.asarray(f).flatten()
    A = np.asarray(A)
    b = np.asarray(b).flatten()
    m = len(b)
    
    if x0 is None:
        X = np.zeros(n)
    else:
        X = np.asarray(x0).flatten()
    
    # Normalize constraints
    normA = np.sqrt(np.sum(A**2, axis=1))
    normA[normA == 0] = 1.0
    A = A / normA[:, np.newaxis]
    b = b / normA
    
    normf = 1.0 + abs(f @ X)
    
    # Check if H is positive definite
    errnorm = np.finfo(float).eps * 10
    is_qp = True
    
    try:
        np.linalg.cholesky(H + errnorm * np.eye(n))
    except np.linalg.LinAlgError:
        is_qp = False
    
    # Initialize active set
    lambda_qp = np.zeros(m)
    aix = np.zeros(m, dtype=bool)  # Active constraint indices
    
    # Equality constraints are always active
    eqix = np.arange(neqcstr)
    if neqcstr > 0:
        aix[eqix] = True
    
    ACTSET = A[aix, :]
    ACTIND = np.where(aix)[0]
    ACTCNT = len(ACTIND)
    
    # QR factorization of active set
    if ACTCNT > 0:
        Q, R = qr(ACTSET.T)
    else:
        Q = np.eye(n)
        R = np.zeros((0, 0))
    
    CIND = ACTCNT
    simplex_iter = (ACTCNT == n - 1)
    
    # Main iteration loop
    max_iter = 100 * n
    for iter_count in range(max_iter):
        
        # Compute gradient
        gf = H @ X + f
        
        # Compute null space
        Z = Q[:, ACTCNT:]
        
        # Compute search direction
        if is_qp and Z.shape[1] > 0:
            Zgf = Z.T @ gf
            if np.linalg.norm(Zgf) < 1e-15:
                SD = np.zeros(n)
            else:
                try:
                    SD = -Z @ np.linalg.solve(Z.T @ H @ Z, Zgf)
                except np.linalg.LinAlgError:
                    SD = -Z @ (Z.T @ gf)
        else:
            if not simplex_iter and Z.shape[1] > 0:
                SD = -Z @ (Z.T @ gf)
                gradsd = np.linalg.norm(SD)
            else:
                gradsd = Z.T @ gf if Z.shape[1] > 0 else 0
                if gradsd > 0:
                    SD = -Z.flatten() if Z.ndim > 1 else -Z
                else:
                    SD = Z.flatten() if Z.ndim > 1 else Z
                gradsd = abs(gradsd)
        
        # Check for convergence
        if np.linalg.norm(SD) < errnorm:
            # Compute Lagrange multipliers
            if ACTCNT > 0:
                try:
                    rlambda = -np.linalg.solve(R[:ACTCNT, :ACTCNT].T, Q[:, :ACTCNT].T @ gf)
                except np.linalg.LinAlgError:
                    rlambda = np.zeros(ACTCNT)
                
                actlambda = rlambda.copy()
                actlambda[eqix] = np.abs(actlambda[eqix])
                
                indlam = np.where(actlambda < errnorm)[0]
                
                if len(indlam) == 0:
                    lambda_qp[ACTIND] = normf * (rlambda / normA[ACTIND])
                    how = 'ok'
                    return X, lambda_qp, how
                
                # Remove constraint with most negative multiplier
                CIND = indlam[np.argmin(actlambda[indlam])]
                Q, R = qrdelete(Q, R, CIND)
                oldind = ACTIND[CIND]
                ACTIND = np.delete(ACTIND, CIND)
                ACTCNT -= 1
                aix[oldind] = False
                ACTSET = A[aix, :]
                simplex_iter = False
                continue
            else:
                how = 'ok'
                return X, lambda_qp, how
        
        # Find step length
        cstr = A @ X - b
        indix = np.where(~aix)[0]
        
        if len(indix) > 0:
            dist = (cstr[indix] / (A[indix, :] @ SD + errnorm))
            dist[dist < 0] = np.inf
            
            if len(dist) > 0:
                STEPMIN = np.min(dist)
                ind_step = indix[np.argmin(dist)]
            else:
                STEPMIN = np.inf
                ind_step = None
        else:
            STEPMIN = np.inf
            ind_step = None
        
        # Take step
        if STEPMIN < 1:
            X = X + STEPMIN * SD
            # Add blocking constraint to active set
            if ind_step is not None:
                aix[ind_step] = True
                ACTIND = np.append(ACTIND, ind_step)
                ACTCNT += 1
                Q, R = qr_insert(Q, R, ACTCNT, A[ind_step, :])
                ACTSET = A[aix, :]
                if ACTCNT == n - 1:
                    simplex_iter = True
        else:
            X = X + SD
        
        # Check for unbounded
        if not np.isfinite(STEPMIN) and np.linalg.norm(SD) > errnorm:
            how = 'unbounded'
            if verbosity > -1:
                print('Warning: Unbounded solution')
            return X, lambda_qp, how
    
    how = 'maxiter'
    return X, lambda_qp, how


def qr_insert(Q, R, j, x):
    """
    Insert a column in the QR factorization.
    
    If [Q,R] = qr(A) is the original QR factorization, this function
    updates Q and R to be the factorization after inserting column x
    before A(:,j).
    """
    m, n = R.shape if R.size > 0 else (Q.shape[0], 0)
    
    if n == 0:
        return qr(x.reshape(-1, 1))
    
    # Make room and insert
    R_new = np.zeros((m, n + 1))
    R_new[:, :j] = R[:, :j]
    R_new[:, j+1:] = R[:, j:]
    R_new[:, j] = Q.T @ x
    R = R_new
    n = n + 1
    
    # Zero out subdiagonal elements using Givens rotations
    for k in range(min(m-1, j-1), j-1, -1):
        # Givens rotation
        a = R[k, j]
        b = R[k+1, j]
        r = np.sqrt(a**2 + b**2)
        
        if r < np.finfo(float).eps:
            continue
        
        c = a / r
        s = -b / r
        
        R[k, j] = r
        R[k+1, j] = 0
        
        if k < n - 1:
            temp = R[k, k+1:n].copy()
            R[k, k+1:n] = c * temp - s * R[k+1, k+1:n]
            R[k+1, k+1:n] = s * temp + c * R[k+1, k+1:n]
        
        temp = Q[:, k].copy()
        Q[:, k] = c * temp - s * Q[:, k+1]
        Q[:, k+1] = s * temp + c * Q[:, k+1]
    
    return Q, R


def qrdelete(Q, R, j):
    """
    Delete a column from the QR factorization.
    
    If [Q,R] = qr(A) is the original QR factorization, this function
    updates Q and R to be the factorization after removing A(:,j).
    """
    # Remove j-th column
    R = np.delete(R, j, axis=1)
    m, n = R.shape if R.size > 0 else (Q.shape[0], 0)
    
    if n == 0:
        return Q, R
    
    # Use Givens rotations to restore upper triangular form
    for k in range(j, min(n, m-1)):
        # Givens rotation
        a = R[k, k]
        b = R[k+1, k]
        r = np.sqrt(a**2 + b**2)
        
        if r < np.finfo(float).eps:
            continue
        
        c = a / r
        s = -b / r
        
        R[k, k] = r
        R[k+1, k] = 0
        
        if k < n - 1:
            temp = R[k, k+1:n].copy()
            R[k, k+1:n] = c * temp - s * R[k+1, k+1:n]
            R[k+1, k+1:n] = s * temp + c * R[k+1, k+1:n]
        
        temp = Q[:, k].copy()
        Q[:, k] = c * temp - s * Q[:, k+1]
        Q[:, k+1] = s * temp + c * Q[:, k+1]
    
    return Q, R


def scipy_qp(H, f, A, b):
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
def quadprog_qp(H, f, A, b):
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
