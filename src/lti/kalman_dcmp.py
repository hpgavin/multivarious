import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, null_space

from multivarious.lti import ctrb, obsv
from multivarious.lti import abcd_dim

def kalman_dcmp(A, B, C, D, tol=1e-9):
    """
    Transform a state-space realization (A,B,C,D) into Kalman Canonical Form.
    
    Decomposes states into four subspaces: [Xco, Xc_o, X_co, X_c_o], where:
        Xco   : controllable and observable
        Xc_o  : controllable and unobservable
        X_co  : uncontrollable and observable
        X_c_o : uncontrollable and unobservable
    
    Parameters:
        A   : state matrix (n x n)
        B   : input matrix (n x r)
        C   : output matrix (m x n)
        D   : feedthrough matrix (m x r)
        tol : tolerance on singular value ratio (sigma_n/sigma_1) to 
              determine null space from SVD (default: 1e-9)
    
    Returns:
        Ak : transformed state matrix (n x n)
        Bk : transformed input matrix (n x r)
        Ck : transformed output matrix (m x n)
        Dk : transformed feedthrough matrix (m x r)
        T  : coordinate transformation matrix x = T*xk (n x n)
    
    The transformed system has block structure revealing controllability
    and observability properties.
    
    Author: H.P. Gavin, 2017-11-05
    """
    
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    
    n, r, m = abcd_dim(A, B, C, D)
    
    # Singular value decomposition of observability and controllability matrices
    Up, Sp, Vph = svd(obsv(A, C, n), full_matrices=True)
    Uq, Sq, Vqh = svd(ctrb(A, B, n), full_matrices=True)
    
    Vp = Vph.T.conj()
    Vq = Vqh.T.conj()
    
    # Plot singular values
    plt.figure(101)
    plt.clf()
    plt.semilogy(np.arange(1, n+1), np.diag(Sq) / Sq[0, 0], 'x', 
                label='controllability', markersize=10)
    plt.semilogy(np.arange(1, n+1), np.diag(Sp) / Sp[0, 0], 'o', 
                label='observability', markersize=8)
    plt.ylabel(r'$\sigma_i / \sigma_1$')
    plt.xlabel('Index i')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Singular Values of Controllability and Observability Matrices')
    
    # Determine dimensions of controllable and observable subspaces
    no = np.sum(np.diag(Sp) / Sp[0, 0] > tol)  # dimension of observable subspace
    nc = np.sum(np.diag(Sq) / Sq[0, 0] > tol)  # dimension of controllable subspace
    
    print(f"Dimension of observable subspace: {no}")
    print(f"Dimension of controllable subspace: {nc}")
    
    # If fully controllable and observable, return original system
    if no == n and nc == n:
        print("System is fully controllable and observable")
        return A, B, C, D, np.eye(n)
    
    # Bases for subspaces
    Bo = Vp[:, :no]                # basis for observable subspace
    B_o = Vp[:, no:]               # basis for unobservable subspace
    
    Bc = Uq[:, :nc]                # basis for controllable subspace
    # B_c = Uq[:, nc:]             # basis for uncontrollable subspace (not needed)
    
    # Orthonormal bases for the Kalman decomposition
    #                                             controllable    observable
    Bc_o = Bc @ null_space(Bo.T @ Bc, rcond=tol)     #    Y             N
    Bco = Bc @ null_space(Bc_o.T @ Bc, rcond=tol)    #    Y             Y
    B_c_o = B_o @ null_space(Bc.T @ B_o, rcond=tol)  #    N             N
    B_co = null_space(np.hstack([Bco, Bc_o, B_c_o]).T, rcond=tol)  #  N    Y
    
    # Coordinate transformation x = T * xk
    T = np.hstack([Bco, Bc_o, B_co, B_c_o])
    
    # Display subspace dimensions
    nco = Bco.shape[1]
    nc_o = Bc_o.shape[1]
    n_co = B_co.shape[1]
    n_c_o = B_c_o.shape[1]
    
    print(f"\nSubspace dimensions:")
    print(f"  Controllable & Observable:     {nco}")
    print(f"  Controllable & Unobservable:   {nc_o}")
    print(f"  Uncontrollable & Observable:   {n_co}")
    print(f"  Uncontrollable & Unobservable: {n_c_o}")
    print(f"  Total: {nco + nc_o + n_co + n_c_o} (should equal {n})")
    
    # Transform the system
    Ak = np.linalg.solve(T, A @ T)
    Bk = np.linalg.solve(T, B)
    Ck = C @ T
    Dk = D.copy()
    
    return Ak, Bk, Ck, Dk, T


# =========================================================================
# H.P. Gavin, 2017-11-05

# Example usage and test
if __name__ == "__main__":
    print("Testing KalmanDcmp: Kalman Canonical Decomposition\n")
    print("="*60)
    
    # Example 1: Fully controllable and observable system
    print("\nExample 1: Fully controllable and observable system")
    print("-"*60)
    
    A1 = np.array([[0, 1], [-2, -3]])
    B1 = np.array([[0], [1]])
    C1 = np.array([[1, 0]])
    D1 = np.array([[0]])
    
    print(f"Original system:")
    print(f"A =\n{A1}\n")
    
    Ak1, Bk1, Ck1, Dk1, T1 = kalman_dcmp(A1, B1, C1, D1)
    
    print(f"\nTransformed system should be identical (fully c&o):")
    print(f"Ak =\n{Ak1}\n")
    
    # Example 2: Uncontrollable system
    print("\n" + "="*60)
    print("\nExample 2: Uncontrollable system")
    print("-"*60)
    
    # System with uncontrollable mode
    A2 = np.array([[1, 0, 0], 
                   [0, 2, 1], 
                   [0, 0, 2]])
    B2 = np.array([[0], [1], [1]])
    C2 = np.array([[1, 1, 1]])
    D2 = np.array([[0]])
    
    print(f"Original system:")
    print(f"A =\n{A2}\n")
    print(f"B =\n{B2}\n")
    print(f"C =\n{C2}\n")
    
    # Check controllability
    Q2 = ctrb(A2, B2)
    print(f"Controllability matrix rank: {np.linalg.matrix_rank(Q2)} (system order: {A2.shape[0]})")
    
    Ak2, Bk2, Ck2, Dk2, T2 = KalmanDcmp(A2, B2, C2, D2)
    
    print(f"\nTransformed Ak (Kalman form):")
    print(f"{Ak2}\n")
    print(f"Note the block structure revealing controllable/uncontrollable parts")
    
    # Example 3: Unobservable system
    print("\n" + "="*60)
    print("\nExample 3: Unobservable system")
    print("-"*60)
    
    # System with unobservable mode
    A3 = np.array([[1, 0, 0],
                   [0, 2, 0],
                   [0, 0, 3]])
    B3 = np.array([[1], [1], [1]])
    C3 = np.array([[1, 1, 0]])  # Can't observe third state
    D3 = np.array([[0]])
    
    print(f"Original system:")
    print(f"A =\n{A3}\n")
    print(f"C =\n{C3}\n")
    
    # Check observability
    P3 = obsv(A3, C3)
    print(f"Observability matrix rank: {np.linalg.matrix_rank(P3)} (system order: {A3.shape[0]})")
    
    Ak3, Bk3, Ck3, Dk3, T3 = kalman_dcmp(A3, B3, C3, D3)
    
    print(f"\nTransformed Ak (Kalman form):")
    print(f"{Ak3}\n")
    print(f"Transformed Ck:")
    print(f"{Ck3}\n")
    
    # Example 4: Neither fully controllable nor observable
    print("\n" + "="*60)
    print("\nExample 4: Partially controllable and observable")
    print("-"*60)
    
    # 4-state system with mixed properties
    A4 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [0, 0, 0, -1]])
    B4 = np.array([[0], [0], [1], [0]])
    C4 = np.array([[1, 0, 0, 0]])
    D4 = np.array([[0]])
    
    print(f"4-state system:")
    print(f"A shape: {A4.shape}")
    
    Q4 = ctrb(A4, B4)
    P4 = obsv(A4, C4)
    
    print(f"Controllability matrix rank: {np.linalg.matrix_rank(Q4)}")
    print(f"Observability matrix rank: {np.linalg.matrix_rank(P4)}")
    
    Ak4, Bk4, Ck4, Dk4, T4 = kalman_dcmp(A4, B4, C4, D4, tol=1e-9)
    
    print(f"\nTransformed Ak (Kalman canonical form):")
    print(f"{Ak4}\n")
    
    print(f"Block structure:")
    print(f"  Upper-left: Controllable & Observable")
    print(f"  Other blocks: Various combinations")
    
    # Verify transformation preserves transfer function
    print("\n" + "="*60)
    print("\nVerification: Transfer function preservation")
    print("-"*60)
    
    # Check that eigenvalues are preserved
    eig_original = np.linalg.eigvals(A4)
    eig_transformed = np.linalg.eigvals(Ak4)
    
    print(f"Original eigenvalues: {np.sort(eig_original)}")
    print(f"Transformed eigenvalues: {np.sort(eig_transformed)}")
    print(f"Max difference: {np.max(np.abs(np.sort(eig_original) - np.sort(eig_transformed))):.2e}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Kalman decomposition reveals the structure of the system,")
    print("separating controllable/uncontrollable and observable/unobservable parts.")
    print("="*60)
