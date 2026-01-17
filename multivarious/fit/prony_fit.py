"""
prony_fit.py - Prony Series Fitting with L1 Regularization
============================================================

Fit a Prony series to frequency domain complex modulus data using
L1 regularization and an active set method for the inequality constraints.

The Prony series model is:
    G(ω) = k₀ + Σ (iω τₖ kₖ) / (iω τₖ + 1)

Translation from MATLAB to Python, 2025-11-24
Original by H.P. Gavin, 2013-10-04
"""

import numpy as np
import matplotlib.pyplot as plt


def prony_fit(G_dat, f_dat, tau, alpha):
    """
    Fit a Prony series to frequency domain complex modulus data.
    
    Uses L1 regularization and an active set method to enforce
    non-negativity constraints on the Prony coefficients.
    
    Parameters
    ----------
    G_dat : ndarray, shape (M,), complex
        Complex modulus data (storage + loss modulus)
    f_dat : ndarray, shape (M,)
        Frequencies where data is evaluated (Hz), must be f > 0
    tau : ndarray, shape (N,)
        Specified set of relaxation times
    alpha : float
        L1 regularization factor for Prony series coefficients
    
    Returns
    -------
    ko : float
        Static stiffness (elastic modulus at ω=0)
    k : ndarray, shape (N,)
        Prony series coefficients
    cvg_hst : ndarray, shape (2*n, num_iterations)
        Convergence history of coefficients and Lagrange multipliers
    
    Notes
    -----
    The optimization solves:
        min ||G_dat - G_model||² + α||k||₁
        subject to: k ≥ 0
    
    Uses KKT conditions with active set method for inequality constraints.
    """
    
    # Convert to numpy arrays
    G_dat = np.asarray(G_dat, dtype=complex).flatten()
    f_dat = np.asarray(f_dat, dtype=float).flatten()
    tau = np.asarray(tau, dtype=float).flatten()
    
    m = len(f_dat)  # Number of data points
    w = 2 * np.pi * f_dat  # Angular frequency (rad/s)
    
    # Build design matrix: T = [1, iω*τ/(iω*τ + 1), ...]
    # Each column is a basis function
    iw = 1j * w
    T = np.column_stack([
        np.ones(m),
        (iw[:, np.newaxis] * tau) / (iw[:, np.newaxis] * tau + 1.0)
    ])
    
    # Plot the basis functions (real and imaginary parts)
    plt.ion()
    plt.figure(101, figsize=(10, 6))
    plt.clf()
    plt.semilogx(f_dat, np.real(T), '-r', label='Re(T)')
    plt.semilogx(f_dat, np.imag(T), '-b', label='Im(T)')
    plt.ylabel(r'basis functions   $T_k(\omega) = i \omega \tau_k / (i \omega \tau_k + 1)$')
    plt.xlabel('frequency, f, Hz')
    plt.axis([0.5*np.min(f_dat), 1.5*np.max(f_dat), -0.05, 1.05])
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    m, n = T.shape  # m = data points, n = parameters (including k0)
    
    # Normal equations (taking real part is like adding complex conjugate)
    TtT = 2 * np.real(T.conj().T @ T)
    TtG = 2 * np.real(T.conj().T @ G_dat)
    
    # OLS fit as initial guess (even though k might be negative)
    k = np.linalg.solve(TtT, TtG)
    
    MaxIter = 20  # Usually enough
    cvg_hst = np.zeros((2*n, MaxIter))
    
    # ========== Active Set Method Main Loop ==========
    for iteration in range(MaxIter):
        
        # Active set: indices where k < 2*eps (essentially zero or negative)
        A = np.where(k < 2 * np.finfo(float).eps)[0]
        l = len(A)
        
        # Constraint gradient matrix (Ia is identity for active constraints)
        Ia = np.zeros((l, n))
        for i in range(l):
            Ia[i, A[i]] = 1.0
        
        # KKT system of equations
        lambda_vec = np.zeros(n)  # Lagrange multipliers
        
        # Build KKT matrix: [TtT, Ia'; Ia, 0]
        if l > 0:
            XTX = np.block([
                [TtT,              Ia.T],
                [Ia,   np.zeros((l, l))]
            ])
        else:
            XTX = TtT.copy()
        
        # Build KKT right-hand side
        grad_L = -TtT @ k + TtG - alpha  # Gradient of Lagrangian w.r.t. k
        
        if l > 0:
            XTY = np.concatenate([grad_L, -k[A]])
        else:
            XTY = grad_L
        
        # Solve KKT system
        h_lambda = np.linalg.solve(XTX, XTY)
        
        # Extract step and multipliers
        h = h_lambda[:n]  # Step direction
        if l > 0:
            lambda_vec[A] = h_lambda[n:]  # Non-zero multipliers
        
        # Line search: if k+h becomes negative, reduce step length
        dh = 1.0
        k_test = k + h
        if np.min(k_test) < 0:
            idx = np.argmin(k_test)
            dh = -k[idx] / h[idx]
        
        # Update Prony coefficients
        k = k + dh * h
        
        # Plot measurement points and fit
        plt.figure(102, figsize=(10, 8))
        G_hat = T @ k
        plt.clf()
        
        plt.subplot(211)
        plt.semilogx(w, np.real(G_dat), 'or', label="G'(ω) meas.", markersize=4)
        plt.semilogx(w, np.real(G_hat), '-k', label="G'(ω) fit", linewidth=2)
        plt.ylabel("storage modulus,  G'(ω)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(212)
        plt.semilogx(w, np.imag(G_dat), 'ob', label='G"(ω) meas.', markersize=4)
        plt.semilogx(w, np.imag(G_hat), '-k', label='G"(ω) fit', linewidth=2)
        plt.xlabel('ω, rad/s')
        plt.ylabel('loss modulus, G"(ω)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.draw()
        # plt.pause(0.01)
        
        # Store convergence history
        cvg_hst[:, iteration] = np.concatenate([k, lambda_vec])
        
        # Check convergence
        if np.linalg.norm(h) < np.linalg.norm(k) / 1e3 or np.min(k) > -100*np.finfo(float).eps:
            break
    
    # ========== End Active Set Loop ==========
    
    # Trim convergence history to actual iterations
    cvg_hst = cvg_hst[:, :iteration+1]
    
    # Asymptotic standard errors (commented out as in original)
    # residual = G_dat - T @ k
    # sigma_sq = np.linalg.norm(residual)**2 / (m - n + 1)
    # k_std_err = np.sqrt(sigma_sq * np.diag(np.real(np.linalg.inv(TtT))))
    
    # Separate static stiffness from dynamic coefficients
    ko = k[0]
    k = k[1:]
    
    return ko, k, cvg_hst


# ============================================================================
# Updated: 2025-11-24
