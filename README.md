# multivarious

A multitude of various .py modules for multivariable things like digital signal processing, linear time invariant systems, optimization, ordinary differential equations and random variables. 

This repository is under development.  Updates to this README lag the addition of code.  Stay tuned!!! 

---------------------------------

## libraries 

### dsp . digitial signal processing

| **accel2displ** | acceleration, velocity and displacement without much bias or drift from acceleration |

### lti . linear time invariant systems

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

| **ode4u** | the ODE solution via a fixed-time step, 4th order method  |
| **ode45u** | the ODE solution via the adaptive time step, 4th-5th method by Cash and Karp  |
| **odef.py** | a library of multivariable ordinary differential equations, for testing |
| **ode_test** | a library of tests for ode4u and ode45u  |

### rvs . random variables

---------------------------------
