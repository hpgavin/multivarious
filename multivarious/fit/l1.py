"""
l1.py - L1 Regularization with a fixed penalty factor (split variables + active set)

Fit model coefficients c in the model  y_hat = B c  to data y by minimizing

    J(c) = || y - B c ||_2^2  +  alfa * sum_i  omega_i |c_i|          (omega_i = 1 if w = 0)

using the split-variable formulation  c = p - q,  p >= 0,  q >= 0 :

    J(p,q) = || y - B(p-q) ||_2^2  +  alfa * sum_i omega_i (p_i + q_i)   s.t.  p >= 0, q >= 0

This is a convex quadratic program.  It is solved with a primal active-set method:

  1. Start at p = q = 0, with every bound p_i >= 0 and q_i >= 0 active.
  2. Solve the equality-constrained KKT system on the current working face
     for the steps u (for p) and v (for q) and the Lagrange multipliers mu, nu.
  3. RATIO TEST.  Shorten the step by ONE common factor
         delta = min( 1, min_{u_i<0, p_i inactive} -p_i/u_i , min_{v_i<0, q_i inactive} -q_i/v_i )
     so that p and q stay non-negative.  If delta < 1, the blocking bound becomes active.
  4. If a full step (delta = 1) was taken, check the multipliers.  With the Lagrangian
         L = J + mu'(p+u)_A + nu'(q+v)_A
     mu_i = -dJ/dp_i and nu_i = -dJ/dq_i at an active bound, so optimality requires
     mu <= 0 and nu <= 0.  If all are, STOP.  Otherwise RELEASE the active bound with the
     largest positive multiplier and go to 2.

The solution is the exact minimizer of J for the given alfa (it satisfies the LASSO optimality
conditions  |2 B'(y-Bc)_i| <= alfa*omega_i  for c_i = 0  and  = alfa*omega_i*sign(c_i)  otherwise;
see l1_kkt_violation).

Differences from the earlier (adaptive-alfa) l1.py:
  * alfa is FIXED: the returned c minimizes J for the alfa that was passed in.
  * The step length is a ratio test with one common delta for u and v (not the index of min(p+u)).
  * Bounds are explicitly released using the signs of mu and nu (previously mu, nu were unused).
  * The stopping test is the multiplier sign test, not a relative step size.

Weighting (w): w = 0 is the ordinary L1 problem.  For w > 0 the penalty is weighted as in the
"adaptive LASSO" (Zou, 2006):  omega_i = 1 / |c_ols_i|^w , so coefficients that are small in the
unregularized fit are penalized more heavily.  This is solved exactly by scaling the columns of B
(c~_i = omega_i c_i), so the same solver applies.  NOTE: this is not the weighting used in the
earlier l1.py, which did not weight the penalty.

Henri P. Gavin, Duke University (original MATLAB, 2013);
"""

import warnings
import numpy as np


# ----------------------------------------------------------------------------------------------
def _adaptive_weights(B, y, w):
    """Penalty weights omega_i = 1/|c_ols_i|^w  (all ones for w = 0)."""

    n = B.shape[1]

    w = abs(float(w))

    if w == 0.0:
        return np.ones(n)

    c_ols = np.linalg.lstsq(B, y, rcond=None)[0]

    floor = 1e-8 * max(np.max(np.abs(c_ols)), np.finfo(float).tiny)

    return (np.abs(c_ols) + floor) ** (-w)


# ----------------------------------------------------------------------------------------------
def l1(B, y, alfa, w=0.0, tol=1e-10, max_iter=None):
    """
    Fit model coefficients with L1 regularization (fixed alfa) using split variables.

    Minimizes:  J = ||y - B*c||_2^2 + alfa * sum_i omega_i |c_i|

    Parameters
    ----------
    B : ndarray, shape (m, n)
        Basis of the model (the design matrix)
    y : ndarray, shape (m,) or (m, 1)
        Vector of data to be fit to the model
    alfa : float
        L1 regularization factor (>= 0).  alfa = 0 returns the least-squares fit.
        For alfa >= 2*max|B'y| (w = 0) all coefficients are zero.
    w : float, optional
        Weighting exponent.  0 (default): ordinary L1.  > 0: adaptive-LASSO weights
        omega_i = 1/|c_ols_i|^w  (see module docstring).
    tol : float, optional
        Relative tolerance on the multiplier sign test (default 1e-10).
    max_iter : int, optional
        Maximum number of active-set iterations (default 20*n + 100).

    Returns
    -------
    c : ndarray, shape (n,)
        The model coefficients
    mu : ndarray, shape (n,)
        Lagrange multipliers for p >= 0  (mu_i = -dJ/dp_i at an active bound; <= 0 at the solution)
    nu : ndarray, shape (n,)
        Lagrange multipliers for q >= 0  (nu_i = -dJ/dq_i at an active bound; <= 0 at the solution)
    cvg_hst : ndarray, shape (5*n+2, n_iter)
        Convergence history: rows  0:n c,  n:2n p,  2n:3n q,  3n:4n mu,  4n:5n nu,
        row 5n alfa,  row 5n+1  err_norm = ||B c - y|| / (m - n).
        Column 0 is the starting point (p = q = 0); there is one column per active-set iteration.
    """

    # ---- inputs ------------------------------------------------------------------------------
    B = np.asarray(B, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    m, n = B.shape
    if y.size != m:
        raise ValueError("B and y must have the same number of rows")
    alfa = float(alfa)
    if alfa < 0.0:
        raise ValueError("alfa must be non-negative")
    if max_iter is None:
        max_iter = 20 * n + 100

    # ---- weights: solve the weighted problem as an unweighted one in scaled coefficients ------
    omega = _adaptive_weights(B, y, w)       # c~ = omega * c ,  B~ = B / omega
    Bs = B / omega
    G = 2.0 * Bs.T @ Bs                      # Hessian of ||y - Bs c~||^2 with respect to c~
    h = 2.0 * Bs.T @ y
    sK = max(float(np.max(np.abs(np.diag(G)))) if n else 1.0, np.finfo(float).tiny)   # scaling of constraint rows
    scale = max(alfa, np.max(np.abs(h)) if n else 0.0, np.finfo(float).tiny)
    tol_kkt = tol * scale
    dof = max(m - n, 1)

    # ---- start: p = q = 0, every bound active ------------------------------------------------
    p = np.zeros(n)
    q = np.zeros(n)
    Au = np.ones(n, dtype=bool)              # active bounds  p_i = 0
    Av = np.ones(n, dtype=bool)              # active bounds  q_i = 0
    mu = np.zeros(n)
    nu = np.zeros(n)
    hist = []

    def record():
        c_o = (p - q) / omega
        err = np.linalg.norm(B @ c_o - y) / dof
        hist.append(np.concatenate([c_o, p / omega, q / omega, omega * mu, omega * nu, [alfa, err]]))

    record()
    converged = False
    singular_warned = False

    for it in range(1, max_iter + 1):

        # ---- KKT system on the current working face ------------------------------------------
        #  [  G  -G  Ip'  0  ] [u ]   [ h - G p + G q - alfa ]
        #  [ -G   G  0   Iq' ] [v ] = [-h + G p - G q - alfa ]
        #  [  Ip  0  0   0   ] [mu]   [ -p(Au)               ]
        #  [  0  Iq  0   0   ] [nu]   [ -q(Av)               ]
        #  (the constraint rows and columns are scaled by sK, and the multipliers by 1/sK,
        #   so that the matrix is well scaled when G is large; this does not change u, v, mu, nu)
        iu = np.flatnonzero(Au)
        iv = np.flatnonzero(Av)
        lp, lq = iu.size, iv.size
        N = 2 * n + lp + lq
        K = np.zeros((N, N))
        K[:n, :n] = G
        K[:n, n:2 * n] = -G
        K[n:2 * n, :n] = -G
        K[n:2 * n, n:2 * n] = G
        K[iu, 2 * n + np.arange(lp)] = sK
        K[2 * n + np.arange(lp), iu] = sK
        K[n + iv, 2 * n + lp + np.arange(lq)] = sK
        K[2 * n + lp + np.arange(lq), n + iv] = sK
        rhs = np.concatenate([h - G @ p + G @ q - alfa,
                              -h + G @ p - G @ q - alfa,
                              -sK * p[iu],
                              -sK * q[iv]])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            if not singular_warned:
                warnings.warn("l1: singular KKT matrix (B rank deficient?); using least squares")
                singular_warned = True
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]

        u = sol[:n]
        v = sol[n:2 * n]
        mu = np.zeros(n)
        nu = np.zeros(n)
        mu[iu] = sK * sol[2 * n:2 * n + lp]
        nu[iv] = sK * sol[2 * n + lp:]

        # ---- ratio test: ONE common step length for u and v ------------------------------------
        d = 1.0
        blocking = None
        cand = np.flatnonzero((~Au) & (u < 0.0))
        if cand.size:
            r = -p[cand] / u[cand]
            k = int(np.argmin(r))
            if r[k] < d:
                d, blocking = float(r[k]), ('p', int(cand[k]))
        cand = np.flatnonzero((~Av) & (v < 0.0))
        if cand.size:
            r = -q[cand] / v[cand]
            k = int(np.argmin(r))
            if r[k] < d:
                d, blocking = float(r[k]), ('q', int(cand[k]))

        p = np.maximum(p + d * u, 0.0)
        q = np.maximum(q + d * v, 0.0)

        if blocking is not None:             # blocked: the blocking bound becomes active
            kind, i = blocking
            if kind == 'p':
                Au[i] = True
            else:
                Av[i] = True
        p[Au] = 0.0                          # a variable at an active bound is exactly zero
        q[Av] = 0.0
        if blocking is not None:
            record()
            continue

        # ---- full step taken: multiplier sign test (mu <= 0 and nu <= 0 at an optimum) ----------
        # a bound may be released only if the partner bound of the same index stays active,
        # which keeps the KKT matrix nonsingular
        viol_p = np.where(Au & Av & (mu > tol_kkt), mu, 0.0)
        viol_q = np.where(Au & Av & (nu > tol_kkt), nu, 0.0)
        ip, iq = int(np.argmax(viol_p)), int(np.argmax(viol_q))
        record()
        if viol_p[ip] <= 0.0 and viol_q[iq] <= 0.0:
            converged = True
            break
        if viol_p[ip] >= viol_q[iq]:
            Au[ip] = False                   # release p_ip : it may now become positive
        else:
            Av[iq] = False                   # release q_iq

    if not converged:
        warnings.warn("l1: active-set iteration did not converge in %d iterations" % max_iter)

    c = (p - q) / omega
    mu_out = omega * mu
    nu_out = omega * nu
    c[np.abs(c) == 0.0] = 0.0                # no negative zeros
    cvg_hst = np.array(hist).T
    return c, mu_out, nu_out, cvg_hst


# ----------------------------------------------------------------------------------------------
def l1_kkt_violation(B, y, c, alfa, w=0.0, zero_tol=1e-12):
    """
    Largest violation of the optimality (subgradient) conditions of
        J = ||y - B c||^2 + alfa * sum_i omega_i |c_i|
    at the coefficients c.  g = 2 B'(y - B c) must satisfy
        g_i = alfa*omega_i*sign(c_i)   for c_i != 0
        |g_i| <= alfa*omega_i           for c_i == 0
    Coefficients with |c_i| <= zero_tol*max|c| are treated as zero.
    Returns the maximum violation (0 at an exact solution).
    """

    B = np.asarray(B, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)

    omega = _adaptive_weights(B, y, w)

    g = 2.0 * B.T @ (y - B @ c)

    lim = alfa * omega

    nz = np.abs(c) > zero_tol * max(np.max(np.abs(c)) if c.size else 0.0, np.finfo(float).tiny)

    viol = np.zeros_like(c)
    viol[nz] = np.abs(g[nz] - lim[nz] * np.sign(c[nz]))
    viol[~nz] = np.maximum(np.abs(g[~nz]) - lim[~nz], 0.0)

    return float(viol.max()) if viol.size else 0.0


# ----------------------------------------------------------------------------------------------
# Test function
if __name__ == '__main__':
    """
    Simple test of l1  (same test case as the earlier l1.py)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Testing l1.py")
    print("=" * 70)

    # Generate test data
    np.random.seed(42)
    x = np.linspace(-1.2, 1.2, 49)
    m = len(x)

    # Power polynomial basis (design matrix)
    B = np.column_stack([x**i for i in range(8)])

    # Generate noisy data (not purely polynomial)
    noise = 0.15 * np.random.randn(m)
    y = 1 - x**2 + np.sin(np.pi * x) + noise

    # L1 regularization parameters
    w = 1.0      # Weighting exponent (0 = ordinary L1, >0 = adaptive-LASSO weights)
    alfa = 0.1   # L1 regularization parameter (fixed)

    # Fit model
    print(f"\nFitting model with alfa={alfa}, w={w}")
    c, mu, nu, cvg_hst = l1(B, y, alfa, w)

    # Display results
    print("\nModel coefficients:")
    print(c)

    print(f"\nNumber of iterations: {cvg_hst.shape[1] - 1}")
    print(f"alfa (fixed): {cvg_hst[-2, -1]:.6f}")
    print(f"Final error: {cvg_hst[-1, -1]:.6f}")
    print(f"Optimality (KKT) violation: {l1_kkt_violation(B, y, c, alfa, w):.2e}")
    print(f"multipliers: max(mu) = {mu.max():.2e}, max(nu) = {nu.max():.2e}   (both must be <= 0)")

    # Ordinary (unweighted) L1 for comparison
    c0, _, _, h0 = l1(B, y, alfa, 0.0)
    print("\nOrdinary L1 (w = 0), same alfa:")
    print(c0)
    print(f"Optimality (KKT) violation: {l1_kkt_violation(B, y, c0, alfa, 0.0):.2e}")

    # Plot results
    n = len(c)

    # OLS for comparison
    c_ols = np.linalg.lstsq(B, y, rcond=None)[0]

    plt.figure(figsize=(12, 4))

    # Subplot 1: Coefficients comparison
    plt.subplot(1, 3, 1)
    plt.plot(c_ols, '+r', markersize=15, linewidth=3, label='OLS (α=0)')
    plt.plot(c, 'o', color=[0, 0.8, 0], markersize=9, linewidth=4,
             label=f'L1 (α={alfa:.3f}, w={w:.1f})')
    plt.plot(c0, 'x', color=[0, 0, 0.8], markersize=9, label=f'L1 (α={alfa:.3f}, w=0)')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Coefficient index, i')
    plt.ylabel('Coefficients, c_i')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    # Subplot 2: Data fit
    plt.subplot(1, 3, 2)
    plt.plot(x, y, 'ok', label='Data', markersize=6)
    plt.plot(x, B @ c_ols, 'or', label='OLS', markersize=4)
    plt.plot(x, B @ c, 'o', color=[0, 0.8, 0], label='L1', markersize=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Convergence history (coefficients)
    plt.subplot(1, 3, 3)
    plt.plot(cvg_hst[0:n, :].T)
    plt.xlabel('Iteration')
    plt.ylabel('Coefficients')
    plt.title('Convergence History')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('l1_test_python.png', dpi=150)
    print("\nPlot saved to: l1_test_python.png")

    print("\n" + "=" * 70)
    print("l1 test completed successfully!")
    print("=" * 70)
