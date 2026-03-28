#!/usr/bin/env python3
"""
sqp.py - Sequential Quadratic Programming with Embedded QP Solver
==================================================================

Nonlinear optimization with inequality constraints using Sequential
Quadratic Programming with a robust active-set QP solver.

The embedded QP solver uses active-set methods with QR factorization
for numerical stability - much more robust than scipy.optimize for
ill-conditioned problems.

Translation from MATLAB to Python, 2025-11-24
Original by Andy Grace (MathWorks) and H.P. Gavin (Duke University)
"""

import numpy as np
import time
from datetime import datetime, timedelta
from scipy.linalg import qr
from opt_options import opt_options


def sqp(func, v_init, v_lb=None, v_ub=None, options_in=None, consts=1.0):
    """
    Sequential Quadratic Programming for nonlinear optimization.
    
    Minimizes f(v) such that g(v) < 0 and v_lb <= v <= v_ub.
    
    Parameters
    ----------
    func : callable
        Objective function: f, g = func(v, consts)
        Returns scalar f and constraint vector g (g < 0 is feasible)
    v_init : ndarray, shape (n,)
        Initial design variable values
    v_lb, v_ub : ndarray, shape (n,), optional
        Lower/upper bounds (default: ±100*|v_init|)
    options_in : array-like, optional
        Optimization settings (see opt_options)
    consts : any
        Constants passed to func
    
    Returns
    -------
    v_opt : ndarray
        Optimal design variables
    f_opt : float
        Optimal objective value
    g_opt : ndarray
        Constraint values at optimum
    cvg_hst : ndarray
        Convergence history
    lambda_opt : ndarray
        Lagrange multipliers
    hess : ndarray
        Final Hessian approximation
    """
    
    # Handle inputs
    v_init = np.asarray(v_init, dtype=float).flatten()
    n = len(v_init)
    
    if v_lb is None:
        v_lb = -1.0e2 * np.abs(v_init)
    if v_ub is None:
        v_ub = 1.0e2 * np.abs(v_init)
    
    v_lb = np.asarray(v_lb, dtype=float).flatten()
    v_ub = np.asarray(v_ub, dtype=float).flatten()
    
    if np.any(v_ub <= v_lb):
        raise ValueError("v_ub must be > v_lb for all variables")
    
    # Clip initial guess to bounds
    v_init = np.clip(v_init, 0.9*v_lb, 0.9*v_ub)
    
    # Options
    options = opt_options(options_in)
    msglev = int(options[0])
    tol_v = options[1]
    tol_f = options[2]
    tol_g = options[3]
    max_evals = int(options[4])
    neqcstr = int(options[19])  # number of equality constraints
    del_min = options[16]
    del_max = options[17]
    
    # Scale to [-1, +1]
    s0 = (v_lb + v_ub) / (v_lb - v_ub)
    s1 = 2.0 / (v_ub - v_lb)
    x = s0 + s1 * v_init
    x_lb = -np.ones(n)
    x_ub = np.ones(n)
    
    # Initialize
    start_time = time.time()
    function_count = 0
    iteration = 1
    
    # First evaluation
    f, g = func((x - s0) / s1, consts)
    function_count += 1
    
    if not np.isscalar(f):
        raise ValueError("Objective must be scalar")
    g = np.atleast_1d(g).astype(float).flatten()
    m = len(g)
    
    # Storage
    OLDX = x.copy()
    OLDG = g.copy()
    gradf = np.zeros(n)
    OLDgradf = np.zeros(n)
    gradg = np.zeros((m, n))
    OLDgradg = np.zeros((m, n))
    LAMBDA = np.zeros(m)
    HESS = np.eye(n)
    SD = np.zeros(n)
    
    cvg_hst = np.full((n + 5, max_evals), np.nan)
    CHG = 1e-7 * (np.ones(n) + np.abs(x))
    GNEW = 1e8 * CHG
    
    StepLength = 1.0
    end_iterations = False
    
    # Best solution
    f_opt = np.inf
    x_opt = x.copy()
    g_opt = g.copy()
    
    # Save initial
    cvg_hst[:, 0] = np.concatenate([(x - s0) / s1, [f, np.max(g), function_count, 1.0, 1.0]])
    
    if msglev:
        print()
        print(' -+-+-+-+-+-+-+-+-+-+- SQP -+-+-+-+-+-+-+-+-+-+-+-+-+')
        print(f' iteration               = {iteration:5d}', end='')
        if np.max(g) > tol_g:
            print('     !!! infeasible !!!')
        else:
            print('     ***  feasible  ***')
        print(f' function evaluations    = {function_count:5d}  of  {max_evals:5d}')
        print(f' objective               = {f:11.3e}')
        if n < 10:
            print(' variables               = ', end='')
            for val in (x - s0) / s1:
                print(f'{val:11.3e}', end='')
            print()
        print(f' max constraint          = {np.max(g):11.3e}')
        print(' -+-+-+-+-+-+-+-+-+-+- SQP -+-+-+-+-+-+-+-+-+-+-+-+-+')
    
    # ========== MAIN LOOP ==========
    while not end_iterations:
        
        # Compute gradients via finite differences
        oldf = f
        oldg = g.copy()
        
        CHG = -1.0e-8 / (GNEW + np.finfo(float).eps)
        CHG = np.sign(CHG + np.finfo(float).eps) * np.clip(np.abs(CHG), del_min, del_max)
        
        for gcnt in range(n):
            temp = x[gcnt]
            x[gcnt] = temp + CHG[gcnt]
            f, g = func((x - s0) / s1, consts)
            function_count += 1
            
            if np.max(g) < tol_g and f < f_opt:
                if msglev > 1:
                    print(' update optimum point')
                f_opt = f
                g_opt = g.copy()
                x_opt = x.copy()
            
            gradf[gcnt] = (f - oldf) / CHG[gcnt]
            gradg[:, gcnt] = (g - oldg) / CHG[gcnt]
            x[gcnt] = temp
        
        f = oldf
        g = oldg.copy()
        
        # Initialize penalty on first iteration
        if iteration == 1:
            PENALTY = (np.finfo(float).eps + np.dot(gradf, gradf)) * np.ones(m) / \
                     (np.sum(gradg**2, axis=1) + np.finfo(float).eps)
        
        # Augmented Lagrangian gradients
        GOLD = OLDgradf + LAMBDA @ OLDgradg
        GNEW = gradf + LAMBDA @ gradg
        q = GNEW - GOLD
        p = x - OLDX
        
        # Update Hessian (BFGS)
        qp = np.dot(q, p)
        how = 'regular'
        
        if qp < StepLength**2 * 1e-3:
            how = 'modify gradients to ensure Hessian > 0,'
            while qp < -1e-5:
                qp_vals = q * p
                qp_idx = np.argmin(qp_vals)
                q[qp_idx] = q[qp_idx] / 2
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
                    WT = WT * 2
                    qp = np.dot(q, p)
        
        # BFGS Hessian update
        if qp > np.finfo(float).eps:
            HESS = HESS + np.outer(q, q) / qp - (HESS @ np.outer(p, p) @ HESS.T) / (p @ HESS @ p)
            how = how + ' Hessian update.'
        else:
            how = how + ' no Hessian update.'
        
        # Set up QP subproblem
        OLDX = x.copy()
        OLDF = f
        OLDG = g.copy()
        OLDgradf = gradf.copy()
        OLDgradg = gradg.copy()
        OLDLAMBDA = LAMBDA.copy()
        SDi = np.zeros(n)
        
        # Append box constraints to nonlinear constraints
        GT = np.concatenate([g, -x + x_lb, x - x_ub])
        gradg_augmented = np.vstack([gradg, -np.eye(n), np.eye(n)])
        
        # Solve QP subproblem
        SD, lambda_all, howqp = mwQP(HESS, gradf, gradg_augmented, -GT, None, None, SDi, neqcstr)
        
        LAMBDA = lambda_all[:m]
        PENALTY = np.maximum(LAMBDA, 0.5 * (LAMBDA + PENALTY))
        
        # Line search
        infeas = howqp[0] == 'i'
        g_max = np.max(g)
        
        GOAL_1 = f + np.sum(PENALTY * (g > tol_g) * g) + 1e-30
        
        if g_max > tol_g:
            GOAL_2 = g_max
        elif f >= 0:
            GOAL_2 = -1 / (f + 1)
        else:
            GOAL_2 = 0
        
        if not infeas and f < 0:
            GOAL_2 = GOAL_2 + f - 1
        
        COST_1 = GOAL_1 + 1
        COST_2 = GOAL_2 + 1
        
        if msglev > 1:
            print('   alpha      max{g}          COST_1           COST_2')
        
        StepLength = 2
        
        while COST_1 > GOAL_1 and COST_2 > GOAL_2 and function_count < max_evals:
            StepLength = StepLength / 2
            if StepLength < 1e-4:
                StepLength = -StepLength
            
            x = OLDX + StepLength * SD
            f, g = func((x - s0) / s1, consts)
            function_count += 1
            
            if np.max(g) < tol_g and f < f_opt:
                if msglev > 1:
                    print(' update')
                f_opt = f
                g_opt = g.copy()
                x_opt = x.copy()
            
            g_max = np.max(g)
            COST_1 = f + np.sum(PENALTY * (g > 0) * g)
            
            if g_max > tol_g:
                COST_2 = g_max
            elif f >= 0:
                COST_2 = -1 / (f + 1)
            else:
                COST_2 = f
            
            if msglev > 1:
                print(f'   {StepLength:5.3f}    {g_max:11.4e}   {COST_1:14.7e}   {COST_2:14.7e}')
            
            if abs(StepLength) < 1e-10:
                end_iterations = True
                if msglev:
                    print(' * Search stalled')
                break
        
        # Update solution
        x = OLDX + StepLength * SD
        
        # Convergence check
        cvg_v = np.linalg.norm((x - OLDX)) / (np.linalg.norm(x) + np.finfo(float).eps)
        cvg_f = abs(f - OLDF) / (abs(f) + np.finfo(float).eps)
        
        iteration += 1
        cvg_hst[:, iteration-1] = np.concatenate([(x - s0) / s1, [f, g_max, function_count, cvg_v, cvg_f]])
        
        if msglev:
            print()
            print(' -+-+-+-+-+-+-+-+-+-+- SQP -+-+-+-+-+-+-+-+-+-+-+-+-+')
            print(f' iteration               = {iteration:5d}', end='')
            if np.max(g) > tol_g:
                print('     !!! infeasible !!!')
            else:
                print('     ***  feasible  ***')
            print(f' function evaluations    = {function_count:5d}  of  {max_evals:5d}')
            print(f' objective               = {f:11.3e}')
            if n < 10:
                print(' variables               = ', end='')
                for val in (x - s0) / s1:
                    print(f'{val:11.3e}', end='')
                print()
            print(f' max constraint          = {g_max:11.3e}')
            print(f' Convergence Criterion F = {cvg_f:11.4e}    tolF = {tol_f:8.6f}')
            print(f' Convergence Criterion X = {cvg_v:11.4e}    tolX = {tol_v:8.6f}')
            print(f' how: {how}')
            print(f' howqp: {howqp}')
            print(' -+-+-+-+-+-+-+-+-+-+- SQP -+-+-+-+-+-+-+-+-+-+-+-+-+')
        
        # Check convergence
        if iteration > n and (cvg_v < tol_v or cvg_f < tol_f):
            if msglev:
                print(' *** Woo-Hoo! Converged solution found!')
                if cvg_v < tol_v:
                    print(' ***           convergence in design variables')
                if cvg_f < tol_f:
                    print(' ***           convergence in objective')
                
                if np.max(g_opt) < tol_g:
                    print(' *** Woo-Hoo! Converged solution is feasible!')
                else:
                    print(' *** Boo-Hoo! Converged solution is NOT feasible!')
            end_iterations = True
        
        if function_count >= max_evals:
            if msglev:
                print(f' Enough!! Maximum number of function evaluations ({max_evals}) exceeded')
            end_iterations = True
    
    # Final report
    v_opt = (x_opt - s0) / s1
    
    if msglev:
        elapsed = time.time() - start_time
        completion_time = datetime.now().strftime('%H:%M:%S')
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f'\n *** Completion  : {completion_time} ({elapsed_str})')
        print(f' *** Objective   : {f_opt:11.3e}')
        print(' *** Variables   :')
        print('               v_init         v_lb     <     v_opt    <     v_ub')
        print('------------------------------------------------------------------')
        
        v_init_orig = (x[:] - s0) / s1
        for i in range(n):
            eqlb = '=' if v_opt[i] < v_lb[i] + tol_g + 10*np.finfo(float).eps else ' '
            equb = '=' if v_opt[i] > v_ub[i] - tol_g - 10*np.finfo(float).eps else ' '
            print(f'v({i:3d})  {v_init_orig[i]:12.5f}   {v_lb[i]:12.5f} {eqlb} '
                  f'{v_opt[i]:12.5f} {equb}  {v_ub[i]:12.5f}')
        
        print(' *** Constraints :')
        for j in range(m):
            binding = ''
            if LAMBDA[j] > tol_g:
                binding = ' ** binding ** '
            if g_opt[j] > tol_g:
                binding = ' ** not ok  ** '
            print(f'       g({j:3d}) = {g_opt[j]:12.5f}  {binding}')
        print('--------------------------------------------------------------\n')
    
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


# =============================================================================
# Updated: 2025-11-24
