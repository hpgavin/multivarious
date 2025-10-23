import numpy as np
from scipy.linalg import eig
from abcd_dim import abcd_dim

def sys_zero(A, B, C, D, tol=1e-6):
    """
    Invariant and decoupling zeros of a continuous-time LTI system.
    
    The condition for an invariant zero is that the pencil [zI-A, -B; C, D] is
    rank deficient. For zeros that are not poles (i.e., for minimal realizations)
    invariant zeros, z, make H(z) rank-deficient.
    
    Method: Use QZ decomposition to find the generalized eigenvalues of the
    Rosenbrock system matrix, padded with zeros or randn to make it square.
    
    Parameters:
        A   : state matrix (n x n)
        B   : input matrix (n x r)
        C   : output matrix (m x n)
        D   : feedthrough matrix (m x r)
        tol : tolerance value for rank determination (default: 1e-6)
    
    Returns:
        zz : array of system zeros
    
    References:
        Hodel, Computation of Zeros with Balancing, 1992 Lin. Alg. Appl.
        Emami-Naeini and Van Dooren, Automatica, 1982.
        A.J.G. MacFarlane and N. Karcanias, Int. J. Control, 24(1):33-74, 1976
    
    Original Author: A.S. Hodel, August 1993
    Modified by: R. Bruce Tenison, July 4, 1994
    Modified by: H.P. Gavin, 2017-11-09, 2022-12-28
    """
    
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    
    nn, rr, mm = abcd_dim(A, B, C, D)
    
    # Make the system square by padding
    re = mm - rr  # excess outputs
    me = rr - mm  # excess inputs
    rm = max(mm, rr)
    
    zivi = []  # zeros and associated eigenvectors
    
    for iter in range(4):
        
        Bx = B.copy()
        Cx = C.copy()
        Dx = D.copy()
        
        if iter == 0:  # zero padding for system zeros
            if mm > rr:
                Bx = np.hstack([B, np.zeros((nn, re))])
                Dx = np.hstack([D, np.zeros((mm, re))])
            if mm < rr:
                Cx = np.vstack([C, np.zeros((me, nn))])
                Dx = np.vstack([D, np.zeros((me, rr))])
        else:  # random padding for invariant zeros
            if mm > rr:
                Bx = np.hstack([B, np.random.randn(nn, re)])
                Dx = np.hstack([D, np.random.randn(mm, re)])
            if mm < rr:
                Cx = np.vstack([C, np.random.randn(me, nn)])
                Dx = np.vstack([D, np.random.randn(me, rr)])
        
        # Rosenbrock System Matrix
        abcd = np.block([[-A, -Bx], [Cx, Dx]])
        I0 = np.block([[np.eye(nn), np.zeros((nn, rm))],
                      [np.zeros((rm, nn + rm))]])
        
        # Generalized eigenvalue problem
        eigvals, eigvecs = eig(abcd, I0)
        zz_iter = -eigvals
        
        idx = np.isfinite(zz_iter)
        
        if iter == 0:
            z1v1 = np.vstack([zz_iter[idx].reshape(1, -1),
                             eigvecs[:nn+rr, idx]])
        else:
            if np.sum(idx) > 0:
                zivi_iter = np.vstack([zz_iter[idx].reshape(1, -1),
                                      eigvecs[:nn+rr, idx]])
                if len(zivi) == 0:
                    zivi = zivi_iter
                else:
                    zivi = np.hstack([zivi, zivi_iter])
    
    # Combine and find unique zeros
    if len(zivi) == 0:
        zv = z1v1
    else:
        zv = np.hstack([z1v1, zivi])
    
    zv = unique_cols(zv, tol)
    
    zz = zv[0, :]
    nz = len(zz)
    
    # Verify that zeros are invariant zeros
    nrcABCD = min(np.block([[A, B], [C, D]]).shape)
    min_mr = min(D.shape)
    pp = np.linalg.eigvals(A)
    
    good_zero_index = []
    
    print('                   rank  rank  rank  rank ')
    print('     zero         zABCD     G   zAB   zAC sys inv trans inp outp')
    print('  ______________________________________________________________')
    
    for jj in range(nz):
        z = zz[jj]
        
        rank_G = min_mr
        rank_zAB = np.linalg.matrix_rank(
            np.hstack([z * np.eye(nn) - A, -B]), tol=tol
        )  # input decoupling zero
        rank_zAC = np.linalg.matrix_rank(
            np.vstack([z * np.eye(nn) - A, C]), tol=tol
        )  # output decoupling zero
        rank_zABCD = np.linalg.matrix_rank(
            np.block([[z * np.eye(nn) - A, -B], [C, D]]), tol=tol
        )  # invariant zero
        
        if not np.any(np.abs(pp - z) < tol):
            try:
                rank_G = np.linalg.matrix_rank(
                    C @ np.linalg.inv(z * np.eye(nn) - A) @ B + D, tol=tol
                )
            except np.linalg.LinAlgError:
                rank_G = 0
        
        if (rank_zAB < nn) or (rank_zAC < nn) or (rank_zABCD < nrcABCD) or (rank_G < min_mr):
            good_zero_index.append(jj)
            
            print(f' {z.real:15.6f}', end='')
            print(f' {rank_zABCD:6d} {rank_G:5d} {rank_zAB:5d} {rank_zAC:5d}', end='')
            print('   *', end='')
            
            if rank_zABCD < nrcABCD:
                print('   *', end='')
            else:
                print('    ', end='')
            
            if rank_G < min_mr:
                print('     *', end='')
            else:
                print('      ', end='')
            
            if rank_zAB < nn:
                print('   *', end='')
            else:
                print('    ', end='')
            
            if rank_zAC < nn:
                print('    *', end='')
            else:
                print('     ', end='')
            
            print()
    
    zz = zz[good_zero_index]  # return only the invariant zeros
    
    return zz


def unique_cols(X, tol):
    """
    Eliminate repeated columns of X.
    
    Columns are classified as being equal to each other within a tolerance, tol,
    if the absolute value of their differences is less than tol for all elements.
    
    Parameters:
        X   : matrix (m x n)
        tol : tolerance for equality
    
    Returns:
        X : matrix with unique columns
    """
    
    m, n = X.shape
    
    idx = np.ones(n, dtype=bool)
    
    for jj in range(n):
        for kk in range(jj + 1, n):
            if np.all(np.abs(X[:, jj] - X[:, kk]) < tol):
                idx[kk] = False
    
    X = X[:, idx]
    
    return X


def sys_zero_scipy(A, B, C, D):
    """
    Alternative implementation using scipy.signal.ss2zpk.
    
    This is generally more robust and efficient than the padding method.
    
    Parameters:
        A, B, C, D : state-space matrices
    
    Returns:
        zeros : system zeros
    """
    from scipy.signal import ss2zpk
    
    # For MIMO systems, ss2zpk may not work directly
    # Need to extract zeros from transfer function matrix
    
    # This is a simplified version for SISO or small MIMO systems
    try:
        z, p, k = ss2zpk(A, B, C, D)
        return z
    except:
        print("Warning: scipy.signal.ss2zpk failed for this MIMO system")
        return None


# H.P. Gavin, 2017-11-09, 2022-12-28
# ----------------------------------------------------------------------------

# Example usage and test
if __name__ == "__main__":
    print("Testing sys_zero: System Zeros Computation\n")
    print("="*60)
    
    # Example 1: Simple SISO system
    print("\nExample 1: SISO system with known zeros")
    print("-"*60)
    
    # System: G(s) = (s+2) / ((s+1)(s+3))
    # Zero at s = -2, Poles at s = -1, -3
    
    # State-space realization (controllable canonical form)
    A1 = np.array([[0, 1], [-3, -4]])
    B1 = np.array([[0], [1]])
    C1 = np.array([[2, 1]])
    D1 = np.array([[0]])
    
    print(f"A =\n{A1}\n")
    print(f"B =\n{B1}\n")
    print(f"C =\n{C1}\n")
    print(f"D =\n{D1}\n")
    
    poles = np.linalg.eigvals(A1)
    print(f"Poles: {poles}")
    print(f"Expected zero: -2\n")
    
    zeros1 = sys_zero(A1, B1, C1, D1)
    print(f"\nComputed zeros: {zeros1}\n")
    
    # Example 2: MIMO system
    print("\n" + "="*60)
    print("\nExample 2: 2x2 MIMO system")
    print("-"*60)
    
    A2 = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]])
    B2 = np.array([[0, 0], [0, 1], [1, 0]])
    C2 = np.array([[1, 0, 0], [0, 1, 0]])
    D2 = np.zeros((2, 2))
    
    print(f"A shape: {A2.shape}")
    print(f"B shape: {B2.shape}")
    print(f"C shape: {C2.shape}")
    
    poles2 = np.linalg.eigvals(A2)
    print(f"Poles: {poles2}\n")
    
    zeros2 = sys_zero(A2, B2, C2, D2)
    print(f"\nComputed zeros: {zeros2}\n")
    
    # Try scipy version for comparison (SISO only)
    print("\n" + "="*60)
    print("Comparison with scipy.signal.ss2zpk (SISO only)")
    print("="*60 + "\n")
    
    zeros_scipy = sys_zero_scipy(A1, B1, C1, D1)
    if zeros_scipy is not None:
        print(f"SciPy zeros: {zeros_scipy}")
        print(f"Our zeros: {zeros1}")
