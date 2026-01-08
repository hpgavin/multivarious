import numpy as np
from scipy.linalg import schur

def dliap(A, X):
    """
    Solve the discrete-time Lyapunov equation.
    
    Solves: A'*P*A - P + X = 0 for P
    
    By transforming the A & X matrices to complex Schur form, computes the 
    solution of the resulting triangular system, and transforms this solution back.
    A and X must be square matrices.
    
    Parameters:
        A : square matrix (n x n)
        X : square matrix (n x n)
    
    Returns:
        P : solution to A'*P*A - P + X = 0 (n x n)
    
    Reference:
        http://www.mathworks.com/matlabcentral/newsreader/view_thread/16018
        From: daly32569@my-deja.com
        Date: 12 Apr, 2000 23:02:27
        Downloaded: 2015-08-04
    
    Note: For production use, consider scipy.linalg.solve_discrete_lyapunov
    """
    
    A = np.asarray(A, dtype=complex)
    X = np.asarray(X, dtype=complex)
    
    dim = A.shape[0]
    
    # Transform the matrix A to complex Schur form
    # A = U * T * U' ... T is upper-triangular, U*U' = I
    T, U = schur(A, output='complex')  # force complex Schur form
    
    # Now ... P - (U*T'*U')*P*(U*T*U') = X  ... which means ...
    # U'*P*U - (T'*U')*P*(U*T) = U'*X*U
    # Let Q = U'*P*U yields, Q - T'*Q*T = U'*X*U = Y
    
    # Solve for Q = U'*P*U by transforming X to Y = U'*X*U
    # Therefore, solve: Q - T*Q*T' = Y ...  for Q
    # Save memory by using "P" for Q.
    
    Y = U.conj().T @ X @ U
    T1 = T
    T2 = T.conj().T
    P = Y.copy()  # Initialize P ... that is, initialize Q
    
    for col in range(dim-1, -1, -1):
        for row in range(dim-1, -1, -1):
            if row < dim - 1 and col < dim - 1:
                P[row, col] = P[row, col] + T1[row, row+1:dim] @ (P[row+1:dim, col+1:dim] @ T2[col+1:dim, col])
            if col < dim - 1:
                P[row, col] = P[row, col] + T1[row, row] * (P[row, col+1:dim] @ T2[col+1:dim, col])
            if row < dim - 1:
                P[row, col] = P[row, col] + T2[col, col] * (T1[row, row+1:dim] @ P[row+1:dim, col])
            P[row, col] = P[row, col] / (1 - T1[row, row] * T2[col, col])
    
    # Convert Q to P by P = U*Q*U'
    P = U @ P @ U.conj().T
    
    # Check: A'*P*A - P + X should be approximately zero
    
    return P


def dliap_scipy(A, X):
    """
    Wrapper using SciPy's solve_discrete_lyapunov function.
    
    Solves: A*X*A' - X + Q = 0 for X
    
    Note: SciPy's convention is different from our dliap:
        SciPy: A*X*A' - X + Q = 0
        dliap: A'*P*A - P + X = 0
    
    To convert: use A.T and solve for the same X.
    """
    from scipy.linalg import solve_discrete_lyapunov
    
    # SciPy solves: A*P*A' - P + Q = 0
    # We want: A'*P*A - P + X = 0
    # So we solve with A.T
    P = solve_discrete_lyapunov(A.T, X)
    
    return P


# Example usage and test
if __name__ == "__main__":
    print("Testing dliap: Discrete-Time Lyapunov Equation Solver\n")
    print("="*60)
    
    # Example 1: Stable discrete-time system
    print("\nExample 1: Stable system")
    print("-"*60)
    
    # Discrete-time system (stable eigenvalues inside unit circle)
    A = np.array([[0.5, 0.2], [-0.1, 0.6]])
    X = np.eye(2)
    
    print(f"A =\n{A}\n")
    print(f"X =\n{X}\n")
    print(f"Eigenvalues of A: {np.linalg.eigvals(A)}")
    print(f"Max |eigenvalue|: {np.max(np.abs(np.linalg.eigvals(A))):.4f}")
    
    if np.max(np.abs(np.linalg.eigvals(A))) < 1:
        print("✓ System is stable (eigenvalues inside unit circle)\n")
    else:
        print("✗ System is unstable\n")
    
    # Solve using dliap
    P = dliap(A, X)
    
    print(f"Solution P =\n{np.real(P)}\n")
    
    # Verify the solution: A'*P*A - P + X should be ≈ 0
    residual = A.conj().T @ P @ A - P + X
    print(f"Residual (A'*P*A - P + X) =\n{np.real(residual)}\n")
    print(f"Max absolute residual: {np.max(np.abs(residual)):.2e}\n")
    
    # Compare with SciPy
    print("="*60)
    print("Comparison with SciPy's solve_discrete_lyapunov")
    print("="*60 + "\n")
    
    try:
        P_scipy = dliap_scipy(A, X)
        print(f"SciPy solution P =\n{np.real(P_scipy)}\n")
        
        residual_scipy = A.conj().T @ P_scipy @ A - P_scipy + X
        print(f"SciPy residual =\n{np.real(residual_scipy)}\n")
        print(f"Max absolute residual: {np.max(np.abs(residual_scipy)):.2e}\n")
        
        print(f"Difference between solutions:")
        print(f"  max|P - P_scipy| = {np.max(np.abs(P - P_scipy)):.2e}\n")
    except ImportError:
        print("SciPy's solve_discrete_lyapunov not available\n")
    
    # Example 2: Controllability Gramian
    print("="*60)
    print("\nExample 2: Controllability Gramian")
    print("-"*60)
    
    A2 = np.array([[0.8, 0.1], [0, 0.9]])
    B2 = np.array([[1], [0.5]])
    
    # Controllability Gramian satisfies: W_c = A*W_c*A' + B*B'
    # Or equivalently: A'*W_c*A - W_c + B*B' = 0 (after rearranging)
    # This is in the form we want, but with negative sign
    # So we solve: A*W_c*A' - W_c + B*B' = 0
    
    BB = B2 @ B2.T
    
    # Using dliap directly (needs transpose of A)
    W_c = dliap(A2.T, BB)
    
    print(f"A =\n{A2}\n")
    print(f"B =\n{B2}\n")
    print(f"Controllability Gramian W_c =\n{np.real(W_c)}\n")
    
    # Verify
    residual_wc = A2 @ W_c @ A2.T - W_c + BB
    print(f"Residual (A*W_c*A' - W_c + B*B') =\n{np.real(residual_wc)}\n")
    print(f"Max absolute residual: {np.max(np.abs(residual_wc)):.2e}\n")
    
    # Check controllability via Gramian
    eig_wc = np.linalg.eigvals(W_c)
    print(f"Eigenvalues of W_c: {eig_wc}")
    if np.all(eig_wc > 1e-10):
        print("✓ System is controllable (W_c is positive definite)")
    else:
        print("✗ System may not be controllable")
