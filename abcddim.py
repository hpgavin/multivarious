import numpy as np

def abcddim(A, B, C, D):
    """
    Check for compatibility of the dimensions of the matrices defining
    the linear system (A, B, C, D).
    
    State-space system:
        dx/dt = Ax + Bu
        y     = Cx + Du
    
    Parameters:
        A : dynamics matrix (n x n)
        B : input matrix (n x r)
        C : output matrix (m x n)
        D : feedthrough matrix (m x r)
    
    Returns:
        n : number of system states
        r : number of system inputs
        m : number of system outputs
        
    Raises:
        ValueError if matrices are not compatible
    
    Author: A.S. Hodel <scotte@eng.auburn.edu>
    """
    
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    
    # Ensure all inputs are 2D
    if A.ndim != 2:
        raise ValueError('abcddim: A must be a 2D array')
    if B.ndim != 2:
        raise ValueError('abcddim: B must be a 2D array')
    if C.ndim != 2:
        raise ValueError('abcddim: C must be a 2D array')
    if D.ndim != 2:
        raise ValueError('abcddim: D must be a 2D array')
    
    an, am = A.shape
    if an != am:
        raise ValueError('abcddim: A is not square')
    
    bn, br = B.shape
    if bn != an:
        raise ValueError(
            f'abcddim: A and B are not compatible, A:({am}x{an}) B:({bn}x{br})'
        )
    
    cm, cn = C.shape
    if cn != an:
        raise ValueError(
            f'abcddim: A and C are not compatible, A:({am}x{an}) C:({cm}x{cn})'
        )
    
    dm, dr = D.shape
    if cm != dm:
        raise ValueError(
            f'abcddim: C and D are not compatible, C:({cm}x{cn}) D:({dm}x{dr})'
        )
    if br != dr:
        raise ValueError(
            f'abcddim: B and D are not compatible, B:({bn}x{br}) D:({dm}x{dr})'
        )
    
    n = an
    r = br
    m = cm
    
    return n, r, m


# -------------------------------------------------------------- abcddim.py
