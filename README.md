# multivarious

A multitude of various .py modules for multivariable things like digital signal processing, linear time invariant systems, optimization, ordinary differential equations and random variables. 

This repository is under development.  Updates to this README lag the addition of code.  Stay tuned!!! 

---------------------------------

## libraries 

### dsp . digitial signal processing

To generate, transform, and plot multivariate discrete time sequences 

| module       | description |
| ------------ | ------------ |
| **accel2displ** | acceleration, velocity and displacement without much bias or drift from acceleration |

### lti . linear time invariant systems

To analyze and transform a system of linear time invariant system defined by 
linear ordinary differential equations, _d**x**_/_dt_ = _*A x*_(_t_) + _*B u*_(_t_)
and corresponding system outputs, _*y*_(_t_) = _*C x*_(_t_) + _*D u*_(_t_) 

| module       | description |
| ------------ | ------------ |
| **abcd_dim** | dimensions of an LTI realization |
| **blk_hankel** | a block Hankel matrix from a matrix-valued time series |
| **blk_toeplitz** | a block Toeplitz matrix from a matrix-valued time series |
| **con2dis** | a discrete time LTI system from a continuous time LTI system  |
| **damp** | natural frequencies and damping ratios of a dynamics matrix  |
| **dis2con** | a continuous time LTI system from a discrete time LTI system  |
| **dlsym** | the response of a discrete time LTI system |
| **kalman_decomp** | Kalman canonical form from an LTI state-space realization  |
| **lsym** | the response of a continuous time LTI system |
| **mimo_bode** | the Bode spectrum of a MIMO LTI system |
| **mimo_tfe** | the estimate of a frequency response matrix from MIMO data |
| **pz_plot** | a plot of the poles and zeros on the complex plane  |
| **sys_zero** | MIMO system, invariant, transmissions, and decoupling zeros from (A,B,C,D) |
| **wiener_filter** | Markov parameters from time series: identification and simulation  |

### ode . ordinary differential equations

To solve a system of ordinary differential equations _dx_/_dt_ = _f_(_t_,_x_,_u_,_c_) where 
_t_ is time, _x_ is a state vector, _u_ is a time series of exogeneous inputs, and _c_ contains a set of system constants. 

| module       | description |
| ------------ | ------------ |
| **ode4u** | the ODE solution via a fixed-time step, 4th order method  |
| **ode45u** | the ODE solution via the adaptive time step, 4th-5th method by Cash and Karp  |
| **odef.py** | a library of multivariable ordinary differential equations, for testing |
| **ode_test** | a library of tests for ode4u and ode45u  |

### opt . optimization 

To minimize a function of multiple variables subject to a set of inequality constraints:
minimize _f_(_v_) such that _g_(_v_) < 0,
where _v_ is a vector of design variables,
_f_(_v_) is a scalar-valued objective function, and
_g_(_v_) is a vector of inequality constraints. 

| module       | description |
| ------------ | ------------ |

### rvs . random variables

To provide the probability density function, the cumulative distribution function, the inverse cumulative distribution function, and a random sample of various probabiity distributions.  
Correlated random samples of certain multivariate random variables as well.  

| module       | description |
| ------------ | ------------ |

---------------------------------
