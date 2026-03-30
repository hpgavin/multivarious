# Quadratic Programming via an active set formulation

import numpy as np
import scipy.linalg

def plane_rot(x):
    """
    Returns a 2x2 Givens rotation matrix G and the updated first element y
    such that G * x = [y, 0]'.
    """
    x = np.asarray(x)
    if x[1] == 0:
        G = np.eye(2)
        y = x[0]
    else:
        r = np.hypot(x[0], x[1])
        G = np.array([[  x[0]/r , x[1]/r ] ,
                      [ -x[1]/r , x[0]/r ] ])
        y = r
    return G, y


def qr_insert(Q, R, j, x):
    """
    [Q,R] = qr_insert(Q,R,j,x)
    Insert a column in the QR factorization.

    If (Q,R) is the original QR factorization of A, then 
    Q,R  = qr_insert(Q,R,j,x) changes Q and R to be the factorization
    of the matrix obtained by inserting an extra column, x, before A(:,j).  
    If A has n columns and j=n+1, then x is inserted after the last column of A.
    """
    m, n = R.shape
    if n == 0:
        Q, R = scipy.linalg.qr(x.reshape(-1, 1), mode='full')
        return Q, R
    
    # Make room and insert x before j-th column.
    # R(:,j+1:n+1) = R(:,j:n); R(:,j) = Q'*x;
    R_new = np.zeros((m, n + 1))
    R_new[:, :j] = R[:, :j]
    R_new[:, j+1:] = R[:, j:]
    R_new[:, j] = (Q.T @ x).flatten()
    R = R_new
    n = n + 1

    # Now R has nonzeros below the diagonal in the j-th column,
    # and "extra" zeros on the diagonal in later columns.
    #    R = [x x x x x
    #         0 x x x x
    #         0 0 + x x
    #         0 0 + 0 x
    #         0 0 + 0 0]
    # Use Givens rotations to zero the +'s, one at a time, from bottom to top.
    for k in range(m - 1, j, -1):  # ... for k = m-1:-1:j
        p = [k - 1, k]
        G, r_val = plane_rot(R[p, j])
        R[p, j] = [r_val, 0]
        if j < n - 1:
            R[p, j+1:n] = G @ R[p, j+1:n]
        Q[:, p] = Q[:, p] @ G.T
        
    return Q, R


def qr_delete(Q, R, j):
    """
    Removes the j-th column from the QR factorization.
    """
    m, n = R.shape
    # Remove j-th column
    R = np.delete(R, j, axis=1)
    n = n - 1
    
    # Use Givens rotations to restore upper triangular form
    for k in range(j, min(m - 1, n)):
        p = [k, k + 1]
        G, r_val = plane_rot(R[p, k])
        R[p, k] = [r_val, 0]
        if k < n - 1:
            R[p, k+1:n] = G @ R[p, k+1:n]
        Q[:, p] = Q[:, p] @ G.T
        
    return Q, R


def qp_solve(H, f, A, B, x_lb=None, x_ub=None, X=None, neqcstr=None, verbosity=None, negdef=0, normalize=1):
    """
    X, lambda, how = qp_solve(H,f,A,B,x_lb,x_ub,X,neqcstr,verbosity,negdef,normalize)

     solves the quadratic programming problem:

          min 0.5 x' H x + f' x   subject to:  Ax <= b 
           x    

     lambda is the set of Lagrangian multipliers at the optimal point

     x_lb and x_ub define a set of lower and upper bounds on the
     design variables, X, so that the solution  
     is always in the range x_lb < X < x_ub.

     X is the initial starting point

     neqcstr indicates that the first neqcstr constraints are equality 
     constraints defined by A and b are equality constraints.

     verbosity indicates the level of warning messages displayed during
     the solution.  A value of -1 results in no warning messages. 
    """

    # Handle missing arguments 
    if normalize is None: normalize = 1
    if negdef is None: negdef = 0
    if verbosity is None: verbosity = 0
    if neqcstr is None: neqcstr = 0
    if X is None: X = np.array([])
    if x_ub is None: x_ub = np.array([])
    if x_lb is None: x_lb = np.array([])

    if A is None or A.size == 0:
        ncstr = 0
        nvars = len(f)
    else:
        ncstr, nvars = A.shape

    nvars = len(f) # In case A is empty

    if X.size == 0:
        X = np.zeros(nvars)

    # Work on local copies so the caller's arrays are never mutated.
    # In MATLAB all arguments are passed by value; numpy arrays are by reference.
    # Row-normalisation (below) modifies A in-place, which would corrupt the
    # caller's constraint matrix across repeated calls (e.g. from an SQP loop).
    f = f.flatten()                                 # flatten already copies
    B = B.flatten() if B is not None else B         # flatten already copies
    if A is not None and A.size > 0:
        A = np.array(A, dtype=float)                # explicit copy of A

    simplex_iter = 0
    if H is None or H.size == 0 or np.linalg.norm(H, np.inf) == 0:
        H = np.zeros((nvars, nvars))
        is_qp = 0
    else:
        is_qp = not negdef

    how = 'ok' 

    normf = 1.0
    if normalize > 0:
        if not is_qp:
            normf = np.linalg.norm(f)
            if normf > 0:
                f = f / normf
            else:
                normf = 1.0

    # Handle the parameter bounds as linear constraints
    lenXmin = len(x_lb) if x_lb is not None else 0
    if lenXmin > 0:
        A = np.vstack([A, -np.eye(lenXmin, nvars)]) if A.size > 0 else -np.eye(lenXmin, nvars)
        B = np.concatenate([B, -x_lb.flatten()])
    
    lenXmax = len(x_ub) if x_ub is not None else 0
    if lenXmax > 0:
        A = np.vstack([A, np.eye(lenXmax, nvars)]) if A.size > 0 else np.eye(lenXmax, nvars)
        B = np.concatenate([B, x_ub.flatten()])
    
    ncstr = ncstr + lenXmin + lenXmax
    eps = np.finfo(float).eps

    errcstr = 100 * np.sqrt(eps) * (np.linalg.norm(A) if A.size > 0 else 0) 
    # Used for determining threshold for whether a direction will violate
    # a constraint.
    normA = np.ones(ncstr)
    if normalize > 0 and A.size > 0:
        for i in range(ncstr):
            n = np.linalg.norm(A[i, :])
            if n != 0:
                A[i, :] = A[i, :] / n
                B[i] = B[i] / n
                normA[i] = n
    else:
        normA = np.ones(ncstr)

    errnorm = 0.01 * np.sqrt(eps) 

    lambda_vec = np.zeros(ncstr)
    aix = np.zeros(ncstr, dtype=bool)
    ACTCNT = 0                       # number of active constraints
    ACTSET = np.empty((0, nvars))    # the set of active constraints
    ACTIND = []                      # the indices of the active constraints
    CIND = 0                         # a constraint index (0-based)
    eqix = np.arange(neqcstr)

    # ------------EQUALITY CONSTRAINTS---------------------------
    Q = np.eye(nvars)
    R = np.empty((nvars, 0))
    if neqcstr > 0:
        aix[eqix] = True
        ACTSET = A[eqix, :]
        ACTIND = list(eqix)
        ACTCNT = neqcstr
        if ACTCNT >= nvars - 1:
            simplex_iter = 1
        CIND = neqcstr
        Q, R = scipy.linalg.qr(ACTSET.T, mode='full')
        
        if np.max(np.abs(A[eqix, :] @ X - B[eqix])) > 1e-10:
            X = np.linalg.lstsq(ACTSET, B[eqix], rcond=None)[0]
        
        m, n = ACTSET.shape
        Z = Q[:, m:nvars]
        err = 0 
        if neqcstr > nvars:
            err = np.max(np.abs(A[eqix, :] @ X - B[eqix]))
            if err > 1e-8:
                how = 'infeasible quadratic program' 
                if verbosity > -1:
                    print('qp_solve() warning:') 
                    print(' ... The equality constraints are overly stringent;')
                    print('     there is no feasible solution.') 
            
            # actlambda = -R\(Q'*(H*X+f))
            # Solve R[0:neqcstr, 0:neqcstr] * x = -(Q.T @ (H@X + f))[0:neqcstr]
            rhs = -(Q.T @ (H @ X + f))[:neqcstr]
            actlambda = scipy.linalg.solve_triangular(R[:neqcstr, :neqcstr], rhs, lower=False)
            lambda_vec[eqix] = normf * (actlambda / normA[eqix])
            return X, lambda_vec, how
        
        if Z.shape[1] == 0:
            rhs = -(Q.T @ (H @ X + f))[:neqcstr]
            actlambda = scipy.linalg.solve_triangular(R[:neqcstr, :neqcstr], rhs, lower=False)
            lambda_vec[eqix] = normf * (actlambda / normA[eqix])
            if np.max(A @ X - B) > 1e-8:
                how = 'infeasible quadratic program'
                if verbosity > -1:
                    print('qp_solve() warning:')
                    print(' ... The constraints or bounds are overly stringent;')
                    print('     there is no feasible solution.') 
                    print('     Equality constraints have been met.')
            return X, lambda_vec, how
        
        # Check whether in Phase 1 of feasibility point finding. 
        if verbosity == -2:
            cstr = A @ X - B
            if ncstr > neqcstr:
                mc = np.max(cstr[neqcstr:ncstr])
                if mc > 0:
                    X[nvars-1] = mc + 1
    else:
        Z = np.eye(nvars)

    # === Find Initial Feasible Solution ====
    cstr = A @ X - B
    mc = -1.0
    if ncstr > neqcstr:
        mc = np.max(cstr[neqcstr:ncstr])
        
    if mc > eps:
        # A2=[[A;zeros(1,nvars)],[zeros(neqcstr,1);-ones(ncstr+1-neqcstr,1)]]
        A_ext = np.vstack([A, np.zeros((1, nvars))])
        col_slack = np.zeros((ncstr + 1, 1))
        col_slack[neqcstr:, 0] = -1.0
        A2 = np.hstack([A_ext, col_slack])
        
        B_ext = np.concatenate([B, [1e-5]])
        X_start = np.concatenate([X, [mc + 1]])
        
        XS, lambdas, _ = qp_solve(None, np.concatenate([np.zeros(nvars), [1.0]]), A2, B_ext, None, None, X_start, neqcstr, -2, 0, -1)

        X = XS[:nvars]
        cstr = A @ X - B
        if XS[nvars] > eps: 
            if XS[nvars] > 1e-8: 
                how = 'infeasible quadratic program'
                if verbosity > -1:
                    print('qp_solve{) warning:')
                    print(' ... The constraints are overly stringent;')
                    print('     there is no feasible solution.')
            else:
                how = 'overly constrained quadratic program'
            lambda_vec = normf * (lambdas[:ncstr] / normA)
            return X, lambda_vec, how

    if is_qp and H.size > 0:
        gf = H @ X + f
        # Check for -ve definite problems:
        # SD=-Z*((Z'*H*Z)\(Z'*gf));
        try:
            lhs = Z.T @ H @ Z
            rhs = Z.T @ gf
            SD = -Z @ np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # Fallback if singular
            SD = -Z @ np.linalg.lstsq(Z.T @ H @ Z, Z.T @ gf, rcond=None)[0]
    else:
        gf = f
        SD = -Z @ (Z.T @ gf)
        if np.linalg.norm(SD) < 1e-10 and neqcstr > 0:
            # This happens when equality constraint
            # is perpendicular to objective function f(x).
            rhs = -(Q.T @ (H @ X + f))[:neqcstr]
            actlambda = scipy.linalg.solve_triangular(R[:neqcstr, :neqcstr], rhs, lower=False)
            lambda_vec[eqix] = normf * (actlambda / normA[eqix])
            return X, lambda_vec, how

    # Sometimes the search direction goes to zero in negative definite problems
    # when the initial feasible point rests on the top of the quadratic function.
    # In this case we can move in any direction to get an improvement in the
    # function so try a random direction.
    if negdef:
        if np.linalg.norm(SD) < np.sqrt(eps):
            SD = -Z @ (Z.T @ (np.random.rand(nvars) - 0.5))

    oldind = 0 

    # The maximum number of iterations for a simplex type method is:
    # maxiters = prod(1:ncstr)/(prod(1:nvars)*prod(1:max(1,ncstr-nvars)))
    
    # --------- START MAIN QUADRATIC PROGRAMMING ROUTINE ----------
    while True:
        # Find distance we can move in search direction SD before a 
        # constraint is violated.

        # Gradient with respect to search direction.
        GSD = A @ SD

        # Blocking constraints
        # Note: we consider only constraints whose gradients are greater
        # than some threshold. If we considered all gradients greater than 
        # zero then it might be possible to add a constraint which would lead to
        # a singular (rank deficient) working set. The gradient (GSD) of such
        # a constraint in the direction of search would be very close to zero.
        indf = np.where((GSD > errnorm * np.linalg.norm(SD)) & (~aix))[0]

        if len(indf) == 0:
            STEPMIN = 1e16
            ind = 0
        else:
            dist = np.abs(cstr[indf] / GSD[indf])
            STEPMIN = np.min(dist)
            ind2 = np.where(dist == STEPMIN)[0]
            # Bland's rule for anti-cycling:
            # If there is more than one blocking constraint
            # then add the one with the smallest index.
            ind = indf[np.min(ind2)] + 1 # Convert to 1-based for consistent logic 

        # ----------------- QP ... QUADRATIC PROGRAMMING ------------
        if is_qp and H.size > 0:
            # if STEPMIN is 1 then this is the exact distance to the solution.
            if STEPMIN >= 1:
                X = X + SD
                if ACTCNT > 0:  # if number of active constraints > 0


                    if ACTCNT >= nvars - 1:
                        if ACTSET.shape[0] > CIND:
                            ACTSET = np.delete(ACTSET, CIND, axis=0)
                        if len(ACTIND) > CIND:
                            ACTIND.pop(CIND)
                    
                    # R @ rlambda = - Q.T @ (H @ X + f) 
                    rhs = -(Q.T @ (H @ X + f))[:ACTCNT]
                    rlambda = scipy.linalg.solve_triangular(R[:ACTCNT, :ACTCNT], rhs, lower=False)
                    actlambda = rlambda.copy()
                    if neqcstr > 0:
                        actlambda[:neqcstr] = np.abs(rlambda[:neqcstr])
                    
                    indlam = np.where(actlambda < 0)[0]
                    if len(indlam) == 0:
                        lambda_vec[np.array(ACTIND) - 1] = normf * (rlambda / normA[np.array(ACTIND) - 1])
                        return X, lambda_vec, how
                    
                    # Remove constraint
                    # lind = find(ACTIND == min(ACTIND(indlam)))
                    target_idx = np.min(np.array(ACTIND)[indlam])
                    lind = ACTIND.index(target_idx)
                    
                    ACTSET = np.delete(ACTSET, lind, axis=0)
                    aix[ACTIND[lind] - 1] = False
                    Q, R = qr_delete(Q, R, lind)
                    ACTIND.pop(lind)
                    ACTCNT = ACTCNT - 2 # As per original logic
                    simplex_iter = 0
                    ind = 0
                else:
                    return X, lambda_vec, how
            else:
                X = X + STEPMIN * SD
            # Calculate gradient w.r.t objective at this point
            gf = H @ X + f
        else:
            # Unbounded Solution
            if len(indf) == 0 or not np.isfinite(STEPMIN):
                if np.linalg.norm(SD) > errnorm:
                    if normalize < 0:
                        STEPMIN = np.abs((X[nvars-1] + 1e-5) / (SD[nvars-1] + eps))
                    else: 
                        STEPMIN = 1e16
                    X = X + STEPMIN * SD
                    how = 'unbounded quadratic program' 
                else:
                    how = 'ill posed quadratic program'
                
                if verbosity > -1:
                    if np.linalg.norm(SD) > errnorm:
                        print('qp_solve() warning:')
                        print(' ... The solution is unbounded and at infinity;')
                        print('     the constraints are not restrictive enough.') 
                    else:
                        print('qp_solve() warning:')
                        print(' ... The search direction is close to zero; the problem is ill posed.')
                        print('     The gradient of the objective function may be zero')
                        print('     or the problem may be badly conditioned.')
                return X, lambda_vec, how
            else: 
                X = X + STEPMIN * SD

        # Update X and calculate constraints
        cstr = A @ X - B
        if neqcstr > 0:
            cstr[eqix] = np.abs(cstr[eqix])
            
        # Check no constraint is violated
        if normalize < 0:
            if X[nvars-1] < eps:
                return X, lambda_vec, how
            
        if np.max(cstr) > 1e5 * errnorm:
            if np.max(cstr) > np.linalg.norm(X) * errnorm:
                if verbosity > -1:
                    print('qp_solve() warning:')
                    printf(' ... The problem is badly conditioned;')
                    print('      the solution is not reliable') 
                    verbosity = -1
                how = 'unreliable quadratic program' 
                X = X - STEPMIN * SD
                return X, lambda_vec, how

        # Sometimes the search direction goes to zero in negative
        # definite problems when the current point rests on
        # the top of the quadratic function. In this case we can move in
        # any direction to get an improvement in the function so 
        # foil search direction by giving a random gradient.
        if negdef:
            if np.linalg.norm(gf) < np.sqrt(eps):
                gf = np.random.randn(nvars)
        
        if ind > 0:
            aix[ind-1] = True
            # Update ACTSET for shape monitoring (0-based CIND logic)
            if CIND >= ACTSET.shape[0]:
                ACTSET = np.vstack([ACTSET, A[ind-1, :]])
            else:
                ACTSET[CIND, :] = A[ind-1, :]
            
            if len(ACTIND) <= CIND:
                ACTIND.append(ind)
            else:
                ACTIND[CIND] = ind
                
            Q, R = qr_insert(Q, R, CIND, A[ind-1, :])
            
        if oldind > 0:
            aix[oldind-1] = False
            
        if not simplex_iter:
            m, n_active = ACTSET.shape[0], nvars
            Z = Q[:, m:nvars]
            ACTCNT = ACTCNT + 1
            if ACTCNT == nvars - 1:
                simplex_iter = 1
            CIND = ACTCNT
            oldind = 0 
        else:
            # R @ rlambda = -(Q.T @ gf)
            # Use R.shape[1] (not ACTCNT) because qr_insert may have grown R
            # beyond ACTCNT when a blocking constraint was just added in simplex mode.
            # MATLAB avoids this by using the full R in R\(Q'*gf) with no slicing.
            ncols = R.shape[1]
            rhs = -(Q.T @ gf)[:ncols]
            rlambda = scipy.linalg.solve_triangular(R[:ncols, :ncols], rhs, lower=False)
            
            if np.any(np.isinf(rlambda)):
                if verbosity > -1:
                    print('         Working set is singular; results may still be reliable.')
                m_act, n_act = ACTSET.shape
                rlambda = -np.linalg.lstsq((ACTSET + np.sqrt(eps) * np.random.randn(m_act, n_act)).T, gf, rcond=None)[0]
                
            actlambda = rlambda.copy()
            if neqcstr > 0:
                actlambda[:neqcstr] = np.abs(actlambda[:neqcstr])
                
            indlam = np.where(actlambda < 0)[0]
            if len(indlam) > 0:
                if STEPMIN > errnorm:
                    # If there is no chance of cycling then pick 
                    # the constraint which causes # the biggest reduction
                    # in the cost function. i.e the constraint with the most
                    # negative Lagrangian multiplier. Since the constraints
                    # are normalized this may result in fewer iterations.
                    minl = np.min(actlambda)
                    CIND = np.where(actlambda == minl)[0][0]
                else:
                    # Bland's rule for anti-cycling:
                    # If there is more than one negative Lagrangian multiplier
                    # then delete the constraint with the smallest index in
                    # the active set.
                    target_idx = np.min(np.array(ACTIND)[indlam])
                    CIND = ACTIND.index(target_idx)

                Q, R = qr_delete(Q, R, CIND)
                Z = Q[:, nvars-1:nvars]
                oldind = ACTIND[CIND]
            else:
                lambda_vec[np.array(ACTIND) - 1] = normf * (rlambda / normA[np.array(ACTIND) - 1])
                return X, lambda_vec, how

        if is_qp and H.size > 0:
            Zgf = Z.T @ gf
            if np.linalg.norm(Zgf) < 1e-15:
                SD = np.zeros(nvars)
            elif Zgf.size == 0:
                if verbosity > -1:
                    # Only happens in -ve semi-definite problems
                    print('qp_solve() warning:')
                    print(' ... QP problem is -ve semi-definite.')
                SD = np.zeros(nvars)
            else:
                try:
                    SD = -Z @ np.linalg.solve(Z.T @ H @ Z, Zgf)
                except np.linalg.LinAlgError:
                    SD = -Z @ np.linalg.lstsq(Z.T @ H @ Z, Zgf, rcond=None)[0]
        else:
            if not simplex_iter:
                SD = -Z @ (Z.T @ gf)
                gradsd = np.linalg.norm(SD)
            else:
                gradsd = (Z.T @ gf)[0]
                if gradsd > 0:
                    SD = -Z.flatten()
                else:
                    SD = Z.flatten()
            
            if np.abs(gradsd) < 1e-10:  # Search direction is null 
                # Check whether any constraints can be deleted from active set.
                # rlambda = -ACTSET'\gf;
                if not oldind:
                    ncols = R.shape[1]
                    rhs = -(Q.T @ gf)[:ncols]
                    rlambda = scipy.linalg.solve_triangular(R[:ncols, :ncols], rhs, lower=False)
                
                actlambda = rlambda.copy()
                if neqcstr > 0:
                    actlambda[:neqcstr] = np.abs(actlambda[:neqcstr])
                
                indlam = np.where(actlambda < errnorm)[0]
                lambda_vec[np.array(ACTIND) - 1] = normf * (rlambda / normA[np.array(ACTIND) - 1])
                
                if len(indlam) == 0:
                    return X, lambda_vec, how
                
                cindmax = len(indlam)
                cindcnt = 0
                newactcnt = 0
                while (np.abs(gradsd) < 1e-10) and (cindcnt < cindmax):
                    cindcnt += 1
                    if oldind:
                        # Put back constraint which we deleted
                        Q, R = qr_insert(Q, R, CIND, A[oldind-1, :])
                    else:
                        simplex_iter = 0
                        if not newactcnt:
                            newactcnt = ACTCNT - 1
                    
                    CIND = indlam[cindcnt-1]
                    oldind = ACTIND[CIND]
                    Q, R = qr_delete(Q, R, CIND)
                    m_act = ACTSET.shape[0]
                    Z = Q[:, m_act-1:nvars]
                    
                    if m_act != nvars:
                        SD = -Z @ (Z.T @ gf)
                        gradsd = np.linalg.norm(SD)
                    else:
                        gradsd = (Z.T @ gf)[0]
                        if gradsd > 0:
                            SD = -Z.flatten()
                        else:
                            SD = Z.flatten()
                
                if np.abs(gradsd) < 1e-10:  # Search direction still null
                    return X, lambda_vec, how
                
                lambda_vec = np.zeros(ncstr)
                if newactcnt:
                    ACTCNT = newactcnt

        if simplex_iter and oldind:
            ACTIND.pop(CIND)
            ACTSET = np.delete(ACTSET, CIND, axis=0)
            CIND = nvars - 1

    return X, lambda_vec, how
