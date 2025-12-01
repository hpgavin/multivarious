import numpy as np
def LP_analysis( v , C ):
    '''
    [ f , g ] = LP_analysis( v , constants )
    analyze a trial solution v to any linear programming problem,
    minimize f = c' * v such that g = A * v - b <= 0
    constants A, b, and c are in a named tuple C
    
    from collections import namedtuple
    Constant = namedtuple('Constant', [ 'A' , 'b' , 'c' ])
    C = Constant( A , b, c )
    '''

    A = C.A             # constraint coefficient matrix (dimension m by n)
    b = C.b             # constraint vector (dimension m by 1)
    c = C.c             # cost coefficient vector (dimension n by 1)

    f = np.dot(c.T, v)  # the cost function

    g = A @ v - b       # the constraint inequalities, compared to zero

    return f, g
