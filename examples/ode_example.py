#! /usr/bin/python3 -i

import numpy as np
import matplotlib.pyplot as plt
from multivarious.ode import ode4u
from multivarious.ode import ode45u
from ode_fctn import ode_example_fctn

def ode_example(number, tolerance, constant):
    """
    Tests ODE solvers with a number of examples, some from:
    J. R. CASH and ALAN H. KARP,
    "A Variable Order Runge-Kutta Method for Initial Value Problems with 
    Rapidly Varying Right-Hand Sides,"
    ACM Transactions on Mathematical Software, 16(3) 1990: 201-222.
    
    Parameters:
        number    : test problem number (1-9)
                    1-5: problems from Cash & Karp article
                    6: matrix-valued states
                    7: structural dynamics - free response
                    8: structural dynamics - step function input
                    9: structural dynamics - random input
        tolerance : desired fractional tolerance
        constant  : parameter value for test problems 1 and 2
    """
    
    # Set up plotting style (adjust as needed for your preferences)
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    
    if number == 1:  # .....................................
        
        t = np.arange(0, 10.2, 0.2)
        x0 = np.array([1.0, 0.0])
        u = np.zeros((1, len(t)))
        params = [constant, number]
        
        _, x4, _, _ = ode4u(ode_example_fctn, t, x0, u, params)
        _, x5, _, _ = ode45u(ode_example_fctn, t, x0, u, params, tolerance, 2)
        
        plt.figure(number)
        plt.clf()
        
        plt.subplot(211)
        plt.plot(t, x4[0, :], label='$x_1(t)$ : ode4u')
        plt.plot(t, x5[0, :], '--', label='$x_1(t)$ : ode45u')
        plt.axis([0, 10, np.min(x5[0, :]), 1])
        plt.legend()
        plt.grid(True)
        
        plt.subplot(212)
        plt.plot(t, x4[1, :], label='$x_2(t)$ : ode4u')
        plt.plot(t, x5[1, :], '--', label='$x_2(t)$ : ode45u')
        plt.axis([0, 10, np.min(x5[1, :]), 0])
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
    
    elif number == 2:  # .....................................
        
        t = np.arange(-1, 1.1, 0.1)
        x0 = np.array([-1.0, -0.01])
        u = np.zeros((1, len(t)))
        params = [constant, number]
        
        _, x4, _, _ = ode4u(ode_example_fctn, t, x0, u, params)
        _, x5, _, _ = ode45u(ode_example_fctn, t, x0, u, params, tolerance, 2)
        
        plt.figure(number)
        plt.clf()
        plt.plot(t, x4.T, label='ode4u')
        plt.plot(t, x5.T, '--', label='ode45u')
        plt.axis([-1, 1, 1.1*np.min(x5), 1.1*np.max(x5)])
        plt.legend()
        plt.grid(True)
    
    elif number == 3:  # .....................................
        
        t = np.arange(-1, 1.1, 0.1)
        x0 = np.array([-1.0])
        u = np.zeros((1, len(t)))
        params = [constant, number]
        
        _, x4, _, _ = ode4u(ode_example_fctn, t, x0, u, params)
        _, x5, _, _ = ode45u(ode_example_fctn, t, x0, u, params, tolerance, 2)
        
        plt.figure(number)
        plt.clf()
        plt.plot(t, x4.T, label='ode4u')
        plt.plot(t, x5.T, '--', label='ode45u')
        plt.axis([-1, 1, -1.1, 0.0])
        plt.legend()
        plt.grid(True)
    
    elif number == 4:  # .....................................
        
        t = np.arange(0, 10.1, 0.1)
        x0 = np.array([100.0])
        u = np.zeros((1, len(t)))
        params = [constant, number]
        
        _, x4, _, _ = ode4u(ode_example_fctn, t, x0, u, params)
        _, x5, _, _ = ode45u(ode_example_fctn, t, x0, u, params, tolerance, 2)
        
        plt.figure(number)
        plt.clf()
        plt.plot(t, x4.T - 100 - 50*t, label='ode4u')
        plt.plot(t, x5.T - 100 - 50*t, '--', label='ode45u')
        plt.axis([0, 10, -5*constant, 5*constant])
        plt.legend()
        plt.grid(True)
    
    elif number == 5:  # .....................................
        
        t = np.arange(0, 10.1, 0.1)
        x0 = np.array([2.0])
        u = np.zeros((1, len(t)))
        params = [constant, number]
        
        _, x4, _, _ = ode4u(ode_example_fctn, t, x0, u, params)
        _, x5, _, _ = ode45u(ode_example_fctn, t, x0, u, params, tolerance, 2)
        
        plt.figure(number)
        plt.clf()
        plt.plot(t, x4.T, label='ode4u')
        plt.plot(t, x5.T, '-o', label='ode45u')
        plt.legend()
        plt.grid(True)
    
    elif number == 6:  # .....................................
        
        T = 2
        dt = 0.01
        N = int(np.floor(T / dt))
        t = np.arange(N) * dt
        n = 4  # even!
        
        # Create Toeplitz-like initial condition
        x0 = 0.5 * np.array([[min(i+1, j+1) for j in range(n)] for i in range(n)])
        x0 = x0[:, :n-1]  # (non-square) matrix-valued initial state
        
        eVec = 10 * np.random.randn(n, n)
        eVal = -10 * np.diag(np.arange(1, n+1))
        A = eVal @ eVec @ np.linalg.inv(eVec)
        B = 100 * np.random.randn(n, n-1)
        
        # Generate input signal (simplified - you may need lsym function)
        u = np.random.randn(1, N) / np.sqrt(dt)
        
        params = [constant, number, x0.shape[0], x0.shape[1], A, B]
        
        x0_vec = x0.flatten()  # make the state a column vector
        
        _, x4, _, _ = ode4u(ode_example_fctn, t, x0_vec, u, params)
        _, x5, _, _ = ode45u(ode_example_fctn, t, x0_vec, u, params, tolerance, 2)
        
        plt.figure(number + 1)
        plt.clf()
        plt.plot(t, u.T)
        plt.title('Input signal')
        plt.grid(True)
        
        plt.figure(number)
        plt.clf()
        plt.plot(t, x5.T, 'ok', markersize=4, linewidth=0.01, label='ode45u')
        plt.plot(t, x4.T, label='ode4u')
        plt.legend()
        plt.grid(True)
        
        x4_reshaped = x4.reshape(n, n-1, N, order='F')
        x5_reshaped = x5.reshape(n, n-1, N, order='F')
        print(f'Size of reshaped state matrix sequence (x4): {x4_reshaped.shape}')
        print(f'Size of reshaped state matrix sequence (x5): {x5_reshaped.shape}')
    
    elif number > 6:  # -------------- numbers 7, 8, 9
        
        # Mass and stiffness matrices correspond to a series of springs
        dof = 10
        k = 1e4  # representative stiffness value
        m = 1e2
        alpha = 1e-4
        beta = 1e-2
        
        # Tri-diagonal stiffness matrix
        Ks = k * (np.triu(np.tril(-np.ones((dof, dof)), 1), -1) + 3*np.eye(dof))
        # Diagonal mass matrix
        Ms = m * (np.eye(dof) + 0.1*np.diag(np.random.rand(dof)))
        # Proportional damping
        Cs = alpha * Ks + beta * Ms
        
        # Dynamics matrix
        A = np.block([
            [np.zeros((dof, dof)), np.eye(dof)],
            [-np.linalg.solve(Ms, Ks), -np.linalg.solve(Ms, Cs)]
        ])
        
        # Input matrix
        B = np.vstack([
            np.zeros((dof, 1)),
            -np.linalg.solve(Ms, np.ones((dof, 1)))
        ])
        
        # Eigenvalue analysis (simplified - you may need damp function)
        eigvals = np.linalg.eigvals(A)
        print(f'Eigenvalues: {eigvals[:5]}...')  # print first 5
        
        dt = 0.010
        points = 1000
        t = np.arange(1, points+1) * dt
        
        u = np.zeros((1, points))
        x0 = np.zeros(20)
        
        if number == 7:  # .....................................
            x0 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 
                          1, 1, 1, 1, 1, 1, 1, -1, 1, -1])
        
        if number == 8:  # .....................................
            u = np.concatenate([
                np.zeros((1, points//10)),
                np.ones((1, points//10))
            ], axis=1)
            # Pad if necessary
            if u.shape[1] < points:
                u = np.pad(u, ((0, 0), (0, points - u.shape[1])), mode='constant')
        
        if number == 9:  # .....................................
            u = np.random.randn(1, points//2) / np.sqrt(dt)
            # Pad if necessary
            if u.shape[1] < points:
                u = np.pad(u, ((0, 0), (0, points - u.shape[1])), mode='constant')
        
        _, x4, xdot4, _ = ode4u(ode_example_fctn, t, x0, u, [A, B])
        _, x5, xdot5, _ = ode45u(ode_example_fctn, t, x0, u, [A, B], tolerance, 2)
        
        # Rows and columns of output data to plot
        r1 = 11
        r2 = 15
        c1 = 100
        c2 = 995
        
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.plot(t[c1:c2], x4[r1:r2, c1:c2].T)
        plt.axis([t[c1], t[c2], 
                 np.min(x5[r1:r2, :]), 
                 np.max(x5[r1:r2, :])])
        plt.title('ode4u')
        plt.grid(True)
        
        plt.figure(2)
        plt.clf()
        plt.plot(t[c1:c2], x5[r1:r2, c1:c2].T)
        plt.axis([t[c1], t[c2], 
                 np.min(x5[r1:r2, :]), 
                 np.max(x5[r1:r2, :])])
        plt.title('ode45u')
        plt.grid(True)
        
        plt.figure(3)
        plt.clf()
        plt.plot(t[c1:c2], x4[r1:r2, c1:c2].T, label='ode4u')
        plt.plot(t[c1:c2], x5[r1:r2, c1:c2].T, '--', label='ode45u')
        plt.axis([t[c1], t[c2], 
                 np.min(x5[r1:r2, :]), 
                 np.max(x5[r1:r2, :])])
        plt.legend()
        plt.title('Comparison')
        plt.grid(True)
    
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Test problem 1 with tolerance 1e-3 and constant 0.1
    ode_example(1, 1e-3, 0.1)
