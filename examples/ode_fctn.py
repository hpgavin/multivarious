import numpy as np

def ode_test_fctn(t, x, u, params):
    """
    Test ODEs from Cash and Karp paper
    
    Parameters:
        t      : scalar time
        x      : state vector
        u      : input vector
        params : list/tuple of parameters
        
    Returns:
        dxdt : state derivative
        y    : output (last element of dxdt)
    """
    
    A = params[0]
    example = params[1]
    
    if example == 1:  # .........................................
        dxdt = np.array([
            x[1],
            x[1]**2 - 0.1 / (A + x[0]**2)
        ])
    
    elif example == 2:  # .........................................
        dxdt = np.array([
            x[1],
            ((-1 + np.pi**2 * A) * np.cos(np.pi * t) - 
             np.pi * t * np.sin(np.pi * t) - 
             t * x[1] + x[0]) / A
        ])
    
    elif example == 3:  # .........................................
        if t < 0:
            dxdt = np.array([0.0])
        else:
            dxdt = np.array([t**A])
    
    elif example == 4:  # .........................................
        t_floor = np.floor(t)
        if np.mod(t_floor, 2) < 1:
            dxdt = np.array([50 + A * t_floor])
        else:
            dxdt = np.array([50 - A * t_floor])
    
    elif example == 5:  # .........................................
        dxdt = np.array([-1 / (x[0] - A)])
    
    elif example == 6:  # .........................................
        nr = params[2]  # number of rows in the state "matrix"
        nc = params[3]  # number of cols in the state "matrix"
        
        A = params[4]   # A is nr by nr
        B = params[5]   # B is nr by nc
        
        x = x.reshape(nr, nc)  # reshape column vector x to nr x nc matrix
        
        dxdt = A @ x + B * u   # operate on x as if it were a matrix
        
        dxdt = dxdt.flatten()  # reshape matrix dxdt to a column vector
    
    elif example > 6:  # .........................................
        A = params[0]
        B = params[1]
        
        dxdt = A @ x + B * u
    
    else:
        dxdt = np.zeros_like(x)
    
    y = dxdt.flatten()[-1]  # last element of dxdt
    
    return dxdt, y
