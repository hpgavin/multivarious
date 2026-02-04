#! /usr/bin/env -S python3 -i

import numpy as np

def near_correlation(A, tol=1e-8, max_iter=100):
    """
    Higham's algorithm to find the nearest correlation matrix to A.
    
    Parameters:
        A (np.ndarray): Symmetric input matrix with 1's on the diagonal.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        
    Returns:
        np.ndarray: Nearest correlation matrix to A.
    """
    n = A.shape[0]
    # Initialize
    Y = A.copy()
    delta_S = np.zeros_like(A)
    W = np.eye(n)  # weight matrix, identity means equal weighting
    
    for k in range(max_iter):
        # Projection onto PSD cone
        R = Y - delta_S
        eigval, eigvec = np.linalg.eigh(R)
        eigval_clipped = np.clip(eigval, 0, None)  # set negative eigenvalues to zero
        X = eigvec @ np.diag(eigval_clipped) @ eigvec.T
        
        # Projection onto unit diagonal
        delta_S = X - R
        Y = X.copy()
        np.fill_diagonal(Y, 1)
        
        # Check convergence (Frobenius norm of difference)
        diff_norm = np.linalg.norm(Y - A, 'fro')
        if diff_norm < tol:
            break
    
    return Y

# Example usage:
if __name__ == "__main__":
    # Example symmetric matrix with 1's on diagonal, but not necessarily PSD
    A = np.array([
        [1.0, 0.9, 0.7],
        [0.9, 1.0, 1.2],  # 1.2 is invalid for correlation
        [0.7, 1.2, 1.0]
    ])
    
    nearest_corr = near_correlation(A)
    print("Nearest correlation matrix:")
    print(nearest_corr)
    
    # Verify PSD
    eigs = np.linalg.eigvalsh(nearest_corr)
    print("Eigenvalues:", eigs)
    print("Is PSD:", np.all(eigs >= -1e-10))

