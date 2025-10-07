import numpy as np

def ode4u(odefun, time, x0, u=None, c=None):
    """
    Solve a system of nonhomogeneous ODEs using the 4th-order Runge-Kutta method.
    (it depends on not just time and state but also external inputs (u) and constanstant (c))

    Parameters:
        odefun : function (t, x, u, c) -> (dxdt, y)
            Function that returns state derivative and output.
        time : array-like, shape (p,)
            Time values at which the solution is computed.
        x0 : array-like, shape (n,)
            Initial state at time[0].
        u : array-like, shape (m, p), optional
            System forcing input sampled at each time step.
        c : any
            Optional constants passed to odefun.

    Returns:
        time  : ndarray, shape (1, p)
        x_sol : ndarray, shape (n, p)
        x_drv : ndarray, shape (n, p)
        y_sol : ndarray, shape (m, p)
    """

    time = np.asarray(time)
    x0 = np.asarray(x0).flatten()
    points = len(time) # the total number of time steps

# create defaults if not provided
    if c is None:
        c = 0
    if u is None:
        u = np.zeros((1, points))
    else:
        u = np.asarray(u)

# verify inputs recieved are in 2D array shape
    if u.ndim == 1:
        u = u[np.newaxis, :]

    if u.shape[1] < points:
        pad_width = points - u.shape[1]
        u = np.pad(u, ((0, 0), (0, pad_width)), mode='constant')

    # Initial output
    # odefun is the user defined derivitive
    # give us np.array(dxdt), np.array([fi]) from the function above
    # dxdt is the first deriviative and y is an array containing other outputs 
    dxdt1, y1 = odefun(time[0], x0, u[:, 0], c)

# set up arrays
    n = x0.size
    m = np.asarray(y1).size

    x_sol = np.full((n, points), np.nan)
    x_drv = np.full((n, points), np.nan)
    y_sol = np.full((m, points), np.nan)

    x_sol[:, 0] = x0    # states
    x_drv[:, 0] = dxdt1 # state derivitivs
    y_sol[:, 0] = y1    # outputs

# time steping loop and main integration
    for p in range(points - 1):
# get pieces needed for rk4
        t = time[p]
        dt = time[p + 1] - t
        dt2 = dt / 2.0

        u_mid = (u[:, p] + u[:, p + 1]) / 2.0

# compute rk4 intermediate dervitives
        dxdt2, _ = odefun(t + dt2, x0 + dxdt1 * dt2, u_mid, c)
        dxdt3, _ = odefun(t + dt2, x0 + dxdt2 * dt2, u_mid, c)
        dxdt4, _ = odefun(t + dt,  x0 + dxdt3 * dt, u[:, p + 1], c)

# update the state using the intermediate derivities to get a more accurate result than just one derv
        x0 = x0 + (dxdt1 + 2 * (dxdt2 + dxdt3) + dxdt4) * dt / 6.0
# next level
# this gives us the next output from the provided differencial equation
        dxdt1, y1 = odefun(time[p + 1], x0, u[:, p + 1], c)

# then we put it into the array
        x_sol[:, p+1] = x0    # state
        x_drv[:, p+1] = dxdt1 # state derivitives 
        y_sol[:, p+1] = y1    # output
# safety - incase NaN or Inf
        if not np.all(np.isfinite(x0)):
            break

    return time, x_sol, x_drv, y_sol
