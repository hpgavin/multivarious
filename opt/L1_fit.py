"""
L1_fit.py - L1 Regularization with Adaptive Penalty and Optional Weighting

Fit model coefficients c in the model ŷ = B*c to data y with L1 regularization
of the coefficients using a split variable formulation and active set method.

Translation from MATLAB by Claude, 2025-10-24
Original by H.P. Gavin, 2013-10-04
"""

import numpy as np


def L1_fit(B, y, alfa, w):
    """
    Fit model coefficients with L1 regularization using split variables.
    
    Minimizes: J = ||y - B*c||₂² + alfa * ||c||₁
    
    Reformulated as:
        J = ||y - B(p-q)||₂² + alfa * sum(p+q)  such that p>=0, q>=0
    where c = p-q and |c| = p+q
    
    The main idea behind casting L1 as a QP is that the coefficient vector {c} 
    is replaced by the difference of two vectors {c}={p}-{q} that are 
    constrained to be non-negative: p_i >= 0 for all i and q_i >= 0 for all i.
    
    If c_i > 0, then p_i = c_i  and q_i = 0
    If c_i < 0, then p_i = 0    and q_i = -c_i
    
    With this constrained re-parameterization, |c_i| = p_i + q_i.
    Note that the dimension of the parameter space doubles, but the KKT 
    equations for the QP are simple and have analytical Hessians and gradients.
    
    Parameters
    ----------
    B : ndarray, shape (m, n)
        Basis of the model (the design matrix)
    y : ndarray, shape (m,) or (m, 1)
        Vector of data to be fit to the model
    alfa : float
        L1 regularization factor for sum(abs(c))
    w : float
        Weighting parameter:
        0: without weighting
        >0: with weighting (adaptive discrimination)
    
    Returns
    -------
    c : ndarray, shape (n,)
        The model coefficients
    mu : ndarray, shape (n,)
        Lagrange multipliers for p
    nu : ndarray, shape (n,)
        Lagrange multipliers for q
    cvg_hst : ndarray, shape (5*n+2, n_iter)
        Convergence history for c, p, q, mu, nu, alfa, err
    
    Notes
    -----
    This implementation uses:
    - Split variable formulation (c = p - q)
    - Active set method with KKT conditions
    - Adaptive penalty factor (alfa) inspired by Levenberg-Marquardt
    - Optional adaptive weighting for enhanced discrimination
    - Line search to ensure p, q remain non-negative
    """
    
    # Ensure y is column vector
    y = np.asarray(y).reshape(-1, 1)
    m, n = B.shape
    
    # Ensure weight is a positive scalar
    w = abs(float(w))
    
    # Precompute matrices
    BtB = 2 * B.T @ B
    Bty = 2 * B.T @ y
    
    # Good initial guess for p and q from non-regularized linear least squares
    c = np.linalg.lstsq(BtB, Bty, rcond=None)[0].flatten()
    
    # Initialize p and q from OLS solution
    p = np.zeros(n)
    q = np.zeros(n)
    p[c > 2*np.finfo(float).eps] = c[c > 2*np.finfo(float).eps]
    q[c < -2*np.finfo(float).eps] = -c[c < -2*np.finfo(float).eps]
    
    # Initialize error norm
    err_norm = np.linalg.norm(B @ c.reshape(-1, 1) - y) / (m - n)
    err_norm_old = 100.0
    
    # Convergence history
    max_iter = 5*n*n
    cvg_hst = np.zeros((5*n + 2, max_iter))
    cvg_hst[:, 0] = np.concatenate([
        c.flatten(),
        p.flatten(),
        q.flatten(),
        np.zeros(2*n),
        [alfa, err_norm]
    ])
    
    # Main iteration loop
    for iter_idx in range(1, max_iter):
        
        # Active sets: indices where p or q are near zero
        Au = np.where(p <= 2*np.finfo(float).eps)[0]  # Active set for p
        Av = np.where(q <= 2*np.finfo(float).eps)[0]  # Active set for q
        lp = len(Au)
        lq = len(Av)
        
        # Constraint gradient matrices
        Ip = np.zeros((lp, n))
        for i, idx in enumerate(Au):
            Ip[i, idx] = 1.0
        
        Iq = np.zeros((lq, n))
        for i, idx in enumerate(Av):
            Iq[i, idx] = 1.0
        
        # Initialize Lagrange multipliers
        mu = np.zeros(n)
        nu = np.zeros(n)
        
        # Assemble KKT system matrix
        # [  BtB      -BtB     Ip'           0     ] [ u  ]   [ RHS_u  ]
        # [ -BtB       BtB      0           Iq'    ] [ v  ] = [ RHS_v  ]
        # [  Ip        0        0            0     ] [ mu ]   [ RHS_mu ]
        # [  0         Iq       0            0     ] [ nu ]   [ RHS_nu ]
        
        BTB = np.block([
            [BtB,              -BtB,             Ip.T,                np.zeros((n, lq))],
            [-BtB,             BtB,              np.zeros((n, lp)),   Iq.T],
            [Ip,               np.zeros((lp, n)), np.zeros((lp, lp)), np.zeros((lp, lq))],
            [np.zeros((lq, n)), Iq,              np.zeros((lq, lp)), np.zeros((lq, lq))]
        ])
        
        # Compute adaptive weights for discrimination
        weight_p = np.abs(c[Au])**w + w*1e-5
        weight_q = np.abs(c[Av])**w + w*1e-5
        
        # Right-hand-side vector for KKT system
        BTY = np.concatenate([
            (Bty - BtB @ p.reshape(-1, 1) + BtB @ q.reshape(-1, 1) - alfa).flatten(),  # dL/du = 0
            (-Bty + BtB @ p.reshape(-1, 1) - BtB @ q.reshape(-1, 1) - alfa).flatten(), # dL/dv = 0
            (-p[Au] / weight_p).flatten(),                                              # dL/dmu = 0
            (-q[Av] / weight_q).flatten()                                               # dL/dnu = 0
        ])
        
        # Solve the KKT system
        try:
            u_v_mu_nu = np.linalg.solve(BTB, BTY)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            u_v_mu_nu = np.linalg.lstsq(BTB, BTY, rcond=None)[0]
        
        # Extract updates and Lagrange multipliers
        u = u_v_mu_nu[0:n]                           # Update for p
        v = u_v_mu_nu[n:2*n]                         # Update for q
        mu[Au] = u_v_mu_nu[2*n:2*n+lp]              # Lagrange multipliers for p
        nu[Av] = u_v_mu_nu[2*n+lp:2*n+lp+lq]        # Lagrange multipliers for q
        
        # Line search: ensure p + du*u >= 0
        du = 1.0
        p_plus_u = p + u
        if np.any(p_plus_u < 0):
            j = np.argmin(p_plus_u)
            if u[j] != 0:
                du = -p[j] / u[j]
        
        # Line search: ensure q + dv*v >= 0
        dv = 1.0
        q_plus_v = q + v
        if np.any(q_plus_v < 0):
            j = np.argmin(q_plus_v)
            if v[j] != 0:
                dv = -q[j] / v[j]
        
        # Compute error with proposed update
        c_new = p + du*u - q - dv*v
        err_norm = np.linalg.norm(B @ c_new.reshape(-1, 1) - y) / (m - n)
        
        # Adaptive penalty update (Levenberg-Marquardt inspired)
        if err_norm < err_norm_old:
            # Accept step and increase penalty (become more aggressive)
            err_norm_old = err_norm
            p = p + du * u
            q = q + dv * v
            c = p - q
            alfa = alfa * 1.2
        else:
            # Reject step and decrease penalty (become more conservative)
            alfa = alfa / 1.1
        
        # Prevent alfa from becoming too small
        if alfa < 1e-4:
            break
        
        # Zero out very small coefficients
        c[np.abs(c) < 1e-6] = 0.0
        
        # Store convergence history
        cvg_hst[:, iter_idx] = np.concatenate([
            c.flatten(),
            p.flatten(),
            q.flatten(),
            mu.flatten(),
            nu.flatten(),
            [alfa, err_norm]
        ])
        
        # Convergence check
        # Coefficients change by less than 1 percent, p>=0 and q>=0
        if (np.linalg.norm(u) <= np.linalg.norm(p) / 1e2 and np.min(p) > -1e-4 and
            np.linalg.norm(v) <= np.linalg.norm(q) / 1e2 and np.min(q) > -1e-4):
            break
    
    # Trim convergence history to actual iterations
    cvg_hst = cvg_hst[:, :iter_idx+1]
    
    return c, mu, nu, cvg_hst


# Test function
if __name__ == '__main__':
    """
    Simple test of L1_fit
    """
    import matplotlib.pyplot as plt
    
    print("Testing L1_fit.py")
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
    w = 1.0      # Weighting parameter (0 = no weighting, >0 = weighted)
    alfa = 0.1   # Initial L1 regularization parameter
    
    # Fit model
    print(f"\nFitting model with alfa={alfa}, w={w}")
    c, mu, nu, cvg_hst = L1_fit(B, y, alfa, w)
    
    # Display results
    print("\nModel coefficients:")
    print(c)
    
    print(f"\nNumber of iterations: {cvg_hst.shape[1]}")
    print(f"Final alfa: {cvg_hst[-2, -1]:.6f}")
    print(f"Final error: {cvg_hst[-1, -1]:.6f}")
    
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
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Coefficient index, i')
    plt.ylabel('Coefficients, c_i')
    plt.legend()
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
    plt.savefig('L1_fit_test_python.png', dpi=150)
    print("\nPlot saved to: L1_fit_test_python.png")
    
    print("\n" + "=" * 70)
    print("L1_fit test completed successfully!")
    print("=" * 70)
