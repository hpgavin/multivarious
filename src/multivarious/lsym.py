# lsym

def lsym(A,B,C,D,u,t,x0, ntrp): 

import numpy as np

'''
 y = lsym ( A, B, C, D, u, t, x0, ntrp )
 transient response of a continuous-time linear system to arbitrary inputs.
		      dx/dt = Ax + Bu
			y   = Cx + Du

	A    :     dynamics    matrix				 (n by n)
    B    :     input       matrix 				 (n by r)
 	C    :     output      matrix				 (m by n)
 	D    :     feedthrough matrix				 (m by r)
 	u    :     matrix of sampled inputs			 (r by p)
 	t    :     vector of uniformly spaced points in time	 (1 by p)
 	x0   :     vector of states at the first point in time   (n by 1)
    ntrp :     "zoh" zero order hold, "foh" first order hold (default)
 	y    :     matrix of the system outputs		 (m by p)
'''

    if (nargin < 8) , ntrp = 'foh' end

     n, r, m = abcd_dim(A,B,C,D)       %  matrix dimensions and compatability check 

    points = size(u,2)		   % number of data points 

    dt =  t[2] - t[1]                % uniform time-step value

# continuous-time to discrte-time conversion ...
    if strcmp(lower(ntrp),'zoh')      % zero-order hold on inputs
        M    = [ A B  zeros(r,n+r) ]
    else                              % first-order hold on inputs
        M    = [ A B zeros(n,r)  zeros(r,n+r) eye(r)  zeros(r,n+2*r) ]
    
    eMdt = expm(M*dt)                % matrix exponential
    Ad   = eMdt[1:n,1:n]             % discrete-time dynamics matrix
    Bd   = eMdt[1:n,n+1:n+r]         % discrete-time input matrix
    if strcmp(lower(ntrp),'zoh')
        Bd0  = Bd
        Bd1  = zeros(n,r)  
    else 
        Bd_  = eMdt(1:n,n+r+1:n+2*r)   % discrete-time input matrix
        Bd0  = Bd  -  Bd_ / dt         % discrete-time input matrix for time p
        Bd1  = Bd_ / dt                % discrete-time input matrix for time p+1
 
#   B and D for discrete time system
#   Bd_bar = Bd0 + Ad*Bd1
#   D_bar  = D   + C*Bd1

# Markov parameters for the discrete time system with ZOH
#   Y0 = D_bar
#   Y1 = C * Bd_bar
#   Y2 = C * Ad * Bd_bar
#   Y3 = C * Ad^2 * Bd_bar

# initial conditions are zero unless specified 

    if nargin < 7
        x0 = zeros(n,1)                % initial conditions are zero

    y = zeros(m,points)              % memory allocation for the output

    x = x0 + Bd1 * u[:,1]            % state at t(1) ... Kjell Ahlin

    y[:,1] = C * x  +  D * u[:,1]    % output for the initial condition

    for p = 2:points

        x      = Ad * x  +  Bd0 * u(:,p-1)  +  Bd1 * u(:,p)

        y(:,p) = C  * x  +  D * u(:,p)


    if max(max(abs(imag(y)))) < 1e-12
        y = real(y)  end  % strip out round-off

# ----------------------------------------------------------------- LSYM

#   w= logspace(-1,log10(100*pi/dt),200)
#   m1 = mimoBode( Ad, (Bd0+Ad*Bd1), C, (D+C*Bd1),  w, dt)
#   m2 = mimoBode( A, B, C, D,  w)
#   figure(111) loglog(w/2/pi, m1, w/2/pi,m2)
#   
#   pp = eig(Ad)
#   zz= syszeros ( Ad, (Bd0+Ad*Bd1), C, (D+C*Bd1) )
#   pz_plot(pp,zz,110,dt)

