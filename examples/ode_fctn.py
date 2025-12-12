"""
ode_fctn.py - various ordinary differential equations 
"""

import numpy as np

def ode_example_fctn(t, x, u, constant):
    """
    Test ODEs from Cash and Karp paper
    
    Parameters:
        t         : scalar time
        x         : state vector
        u         : input vector
        constant  : a list or named tuple of constant
        
    Returns:
        dxdt : state derivative
        y    : output (last element of dxdt)
    """
    
    c = constant[0]
    example = constant[1]
    
    if example == 1:  # .........................................
        dxdt = np.array([
            x[1],
            x[1]**2 - 0.1 / (c + x[0]**2)
        ])
    
    elif example == 2:  # .........................................
        dxdt = np.array([
            x[1],
            ((-1 + np.pi**2 * c) * np.cos(np.pi * t) - 
             np.pi * t * np.sin(np.pi * t) - 
             t * x[1] + x[0]) / c
        ])
    
    elif example == 3:  # .........................................
        if t < 0:
            dxdt = np.array([0.0])
        else:
            dxdt = np.array([t**c])
    
    elif example == 4:  # .........................................
        t_floor = np.floor(t)
        if np.mod(t_floor, 2) < 1:
            dxdt = np.array([50 + c * t_floor])
        else:
            dxdt = np.array([50 - c * t_floor])
    
    elif example == 5:  # .........................................
        dxdt = np.array([ -1 / ( x[0] - c )])
    
    elif example == 6:  # .........................................
        nr = constant[2]  # number of rows in the state "matrix"
        nc = constant[3]  # number of cols in the state "matrix"
        
        A = constant[4]   # A is nr by nr
        B = constant[5]   # B is nr by nc
        
        x = x.reshape(nr, nc)  # reshape column vector x to nr x nc matrix
        
        dxdt = A @ x + B * u   # operate on x as if it were a matrix
        
        dxdt = dxdt.flatten()  # reshape matrix dxdt to a column vector
    
    elif example > 6:  # .........................................
        A = constant[4]
        B = constant[5]
        
        dxdt = A @ x + B @ u
    
    else:
        dxdt = np.zeros_like(x)
    
    y = dxdt.flatten()[-1]  # last element of dxdt
    
    return dxdt, y
