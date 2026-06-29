import numpy as np

def ode45u(odefun, time, x0, u=None, params=None, tolerance=1e-3, display=0):
    """
    Solve a system of nonhomogeneous ODEs using the embedded Runge-Kutta 
    formulas of J.R. Cash and A.H. Karp, with linear interpolation of 
    external forcing within time-steps.
    
    Parameters:
        odefun    : function (t, x, u, params) -> (dxdt, y)
                    returns state derivative and output as arrays
        time      : time values at which the solution is computed.
                    p-dimensional array
        x0        : n-dimensional array, state at time[0].
        u         : (m x p) dimensional array
                    optional input sampled at each time step.
        params    : optional parameters passed to odefun.
        tolerance : desired tolerance constant (default 0.001)
                    may be a vector of length x0, or a scalar
        display   : 3=display lots of results; 2=display less; 1=even less; 0=none
        
    Returns:
        time      : ndarray, shape (1, p)
        x         : ndarray, shape (n, p)
        x_dot     : ndarray, shape (n, p)
        y         : ndarray, shape (m, p)
    """
    
    # create defaults if not provided
    time = np.asarray(time)
    x0 = np.asarray(x0).flatten()
    points = len(time)  # number of points in the time history
    
    if u is None:
        u = np.zeros((1, points))
    else:
        u = np.asarray(u)
    
    if params is None:
        params = 0
    
    tolerance = max(tolerance, 1e-12)
    display = int(np.clip(np.ceil(display), 0, 3))
    
    # state derivatives and outputs at time[0]
    dxdt0, y0 = odefun(time[0], x0, u[:, 0], params)
    
    n = x0.size                 # number of states
    m = np.asarray(y0).size     # number of outputs
    
    # verify inputs received are in 2D array shape
    if u.ndim == 1:
        u = u[np.newaxis, :]
    
    if u.shape[1] < points:
        pad_width = points - u.shape[1]
        u = np.pad(u, ((0, 0), (0, pad_width)), mode='constant')
    
    # allocate memory
    x = np.ones([n, points]) * np.nan
    x_dot = np.ones([n, points]) * np.nan
    y = np.ones([m, points]) * np.nan
    
    x[:, 0] = x0        # initial states
    x_dot[:, 0] = dxdt0 # initial rates
    y[:, 0] = y0        # initial outputs
    
    MaxSteps = 1000     # Maximum number of interior steps
    MaxNumSteps = 0     # Maximum number of sub-steps taken in an interval
    MaxError = 0        # Maximum sub-step truncation error
    
    # Cash-Karp coefficients for embedded Runge-Kutta steps
    a1 = 0
    a2 = 1/5
    a3 = 3/10
    a4 = 3/5
    a5 = 1
    a6 = 7/8
    
    b21 = 1/5
    b31 = 3/40
    b32 = 9/40
    b41 = 3/10
    b42 = -9/10
    b43 = 6/5
    b51 = -11/54
    b52 = 5/2
    b53 = -70/27
    b54 = 35/27
    b61 = 1631/55296
    b62 = 175/512
    b63 = 575/13824
    b64 = 44275/110592
    b65 = 253/4096
    
    XOLD = x0.copy()    # state vector at the start of each interior step
    fevals = 0          # running sum of function evaluations
    
    for p in range(points - 1):  # loop over all points in the time series
        
        t0 = time[p]            # time at start of the full step
        dt = time[p + 1] - t0   # size of the full step
        T0 = t0                 # time at start of an interior sub-step
        DT = dt                 # size of the interior sub-steps
        
        NumSteps = 1            # number of interior steps
        step = 0                # the current interior step number
        
        u0 = u[:, p]                      # forcing at start of full time step
        dudt = (u[:, p + 1] - u0) / dt    # change in forcing over time step
        
        while step < NumSteps:  # loop to adjust the integration time step
            
            dxdt1, _ = odefun(T0 + DT*a1, XOLD, u0 + dudt*(T0 + DT*a1 - t0), params)
            dxdt2, _ = odefun(T0 + DT*a2, XOLD + dxdt1*DT*b21, u0 + dudt*(T0 + DT*a2 - t0), params)
            dxdt3, _ = odefun(T0 + DT*a3, XOLD + dxdt1*DT*b31 + dxdt2*DT*b32, u0 + dudt*(T0 + DT*a3 - t0), params)
            dxdt4, _ = odefun(T0 + DT*a4, XOLD + dxdt1*DT*b41 + dxdt2*DT*b42 + dxdt3*DT*b43, u0 + dudt*(T0 + DT*a4 - t0), params)
            dxdt5, _ = odefun(T0 + DT*a5, XOLD + dxdt1*DT*b51 + dxdt2*DT*b52 + dxdt3*DT*b53 + dxdt4*DT*b54, u0 + dudt*(T0 + DT*a5 - t0), params)
            dxdt6, _ = odefun(T0 + DT*a6, XOLD + dxdt1*DT*b61 + dxdt2*DT*b62 + dxdt3*DT*b63 + dxdt4*DT*b64 + dxdt5*DT*b65, u0 + dudt*(T0 + DT*a6 - t0), params)
            
            # 5th order predictor
            x5 = XOLD + (dxdt1*37/378 + dxdt3*250/621 + dxdt4*125/594 + dxdt6*512/1771) * DT
            # 4th order predictor
            x4 = XOLD + (dxdt1*2825/27648 + dxdt3*18575/48384 + dxdt4*13525/55296 + dxdt5*277/14336 + dxdt6*1/4) * DT
            
            fevals = fevals + 6
            
            # evaluate the truncation error at the start of the full step
            if step == 0:
                TruncationError = np.abs(x5 - x4) / (np.abs(x5) + tolerance)
                Converged = np.all(TruncationError <= tolerance)
            
            if not np.all(np.isfinite(TruncationError)):
                break  # floating point error
            
            if Converged and NumSteps == 1:  # a single full dt step is ok
                XOLD = x5
                break
            
            elif not Converged and step == 0:  # look for a good sub-step size
                # Increase NumSteps using eq'n 16.2.10 of Numerical Recipes in C
                NS1 = NumSteps / np.min((tolerance / TruncationError)**0.25)
                # ... or increase NumSteps by 10 percent, ...
                NS2 = 1.1 * NumSteps
                # ... whichever is greater ...
                NumSteps = int(np.ceil(max(NS1, NS2)))
                DT = dt / NumSteps  # smaller interior sub-step size
            
            elif Converged and step < NumSteps:  # the step size is ok, continue
                step = step + 1     # increment the sub-step number
                T0 = t0 + step * DT # starting time for the next sub-step
                XOLD = x5           # state at start of next sub-step
            
            else:
                print(' uh-oh ... this should never happen ... check ode45u.py ...')
                return time, x, x_dot, y
            
            if display == 3 and step == 1:  # display intermediate information
                print('point   NS   s   error      time     fevals')
            if display == 3 and step > 0:   # display intermediate information
                print(f'{p:5d}  {NumSteps:3d} {step:3d} {np.max(TruncationError):9.2e} {T0:10.6f}   {fevals:4d}')
            
            if NumSteps > MaxSteps:  # too many interior steps
                print('ode45u: The required interior step size is too small')
                print(f'time= {p*dt:10.4f} NumSteps= {NumSteps:4d}  DT= {DT:9.2e}  Error= {np.max(TruncationError):9.2e}')
                if not np.isreal(np.max(TruncationError)):
                    return time, x, x_dot, y
                if tolerance < 1.00:
                    tolerance = tolerance * 1.5
                    print(f'Increasing the desired tolerance and trying again ... tolerance = {tolerance:10.2e}')
                elif MaxSteps < 5e3:
                    MaxSteps = int(MaxSteps * 1.1)
                    print(f'Increasing maximum allowable steps and trying again ... MaxSteps = {MaxSteps:9d}')
                else:
                    print(f'continuing  tolerance = {tolerance:10.2e}   MaxSteps = {MaxSteps:9d}')
                NumSteps = MaxSteps
        
        # end of loop to adjust the integration time step
        
        MaxError = max(np.max(TruncationError), MaxError)
        if NumSteps >= MaxNumSteps:
            MaxNumSteps = NumSteps
            pMax = p
        
        if display == 2 and NumSteps > 1:  # display intermediate information
            print(f'p = {p:6d}     NumSteps = {NumSteps:8d}     Error = {np.max(TruncationError):9.2e}')
        
        dxdt0, y0 = odefun(time[p + 1], XOLD, u[:, p + 1], params)
        
        x[:, p + 1] = XOLD      # state  solution
        x_dot[:, p] = dxdt0     #  rate  solution
        y[:, p] = y0            # output solution
        
        fevals = fevals + 1
        
        if np.linalg.norm(dxdt0) * dt > 1e9 * np.linalg.norm(XOLD):
            break  # break if unstable
        
        if display > 0:  # display overall solution statistics
            print(f'p = {p:6d}  MaxNumSteps = {MaxNumSteps:8d}  MaxError = {np.max(TruncationError):9.2e} fevals = {fevals:d}')
    
    # end of loop over all points in the time series
    
    return time, x, x_dot, y


#  References
#
#  Cash, J.R., and Karp, A.H. 1990,
#  A Variable Order Runge-Kutta Method for Initial-Value Problems with
#  Rapidly Varying Right-Hand Sides,
#  ACM Transactions on Mathematical Software, vol. 16, pp. 201-222.
#  http://www.duke.edu/~hpgavin/ce283/Cash-90.pdf
#
#  Press, W.H., Teukolsky, S.A., Vetterling, W.T., and Flannery, B.P.,
#  Numerical Recipes in C, Cambridge Univ Press, 1992, (ISBN 0-521-43108-5)
#  Section 16.1-16.2
#  http://www.nrbook.com/
#
# ODE45U ----------------------------------------------------------------------
# 2005, 2007, 2012, 2017, 2025-10-06
