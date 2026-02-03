import numpy as np

def fix_R(n, R):
    '''
    fix_R

    fix a correlation matrix R to be square, pos.def, with 1s on diag

    INPUTS:
      n = dimension of R
      R = the correlation matrix
 
    OUTPUT:
      R = the fixed correlation matrix
    '''

    # If no correlation matrix provided, default to identity matrix
    # Identity matrix R = I means all variables are independent (correlation = 0)
    if R is None:
        R = np.eye(n) # In
        T = np.eye(n) # In
        return R, T
    
    # Convert R to square array and validate its properties
    R = np.asarray(R)
    if R.shape != (n, n):
        R = np.eye(n)
        T = np.eye(n) # In
        print(f"fix_R: Made the correlation matrix R I_{n}×{n}, got {R.shape}")
        return R, T

    # ensure that R is symmetric
    R = 0.5 * ( R + R.T ) 

    R_ok = False
    while R_ok == False:
 
        if not np.allclose(np.diag(R), 1.0): # diagonals must be 1s
            D = np.diag( 1/np.sqrt(np.diag(R)) )
            R = D @ R @ D  # make diagonal equal to 1s 
            print("fix_R: made diagonal of R equal to ones")

        if np.any(np.abs(R) > 1): # all elements must be [-1, 1] 
            R( R < -1 ) = -1
            R( R > +1 ) = +1
            print("fix_R: made values of R between -1 and 1")

        # Eigenvalue decomposition of correlation matrix: R = V @ Λ @ V^T
        #   eVec (V): matrix of eigenvectors (n×n)
        #   eVal (Λ): array of eigenvalues (length n)
        eVal, eVec = np.linalg.eigh(R)

        if np.all(eVal > 1e-6):
            R_ok = True
        else: 
            eVal( eVal < 1e-6 ) = 1e-6  
            R = eVec @ np.diag(eVal) @ eVec.T
            print("fix_R: Made R sufficiently positive definite")

    # Transformation matrix
    T = eVec @ np.diag(np.sqrt(eVal))

    return R, T

