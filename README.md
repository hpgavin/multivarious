# multivarious

A multitude of various Python modules for multivariable things like digital signal processing, linear time invariant systems, optimization, ordinary differential equations and random variables. 

This repository is under development.  Updates to this README lag the addition of code.  Stay tuned!!! 

## dsp . digitial signal processing

To generate, transform, and plot multivariate discrete time sequences 

| module          | description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| **accel2displ** | acceleration, velocity and displacement without much bias or drift from acceleration |

## lti . linear time invariant systems

To analyze and transform linear time invariant systems
defined by 
linear differential equations, _d**x**_/_dt_ = _**A x**_(_t_) + _**B u**_(_t_)
or linear difference equations,  **x**(_k_+1) = _**A x**_(_k_) + _**B u**_(_k_)
and corresponding system outputs, _**y**_ = _**C x**_ + _**D u**_

| module            | description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| **abcd_dim**      | dimensions of an LTI realization                                           |
| **blk_hankel**    | a block Hankel matrix from a matrix-valued time series                     |
| **blk_toeplitz**  | a block Toeplitz matrix from a matrix-valued time series                   |
| **con2dis**       | a discrete time LTI system from a continuous time LTI system               |
| **damp**          | natural frequencies and damping ratios of a dynamics matrix                |
| **dis2con**       | a continuous time LTI system from a discrete time LTI system               |
| **dlsym**         | the response of a discrete time LTI system                                 |
| **kalman_decomp** | Kalman canonical form from an LTI state-space realization                  |
| **lsym**          | the response of a continuous time LTI system                               |
| **mimo_bode**     | the Bode spectrum of a MIMO LTI system                                     |
| **mimo_tfe**      | the estimate of a frequency response matrix from MIMO data                 |
| **pz_plot**       | a plot of the poles and zeros on the complex plane                         |
| **sys_zero**      | MIMO system, invariant, transmissions, and decoupling zeros from (A,B,C,D) |
| **wiener_filter** | Markov parameters from time series: identification and simulation          |

## ode . ordinary differential equations

To solve systems of ordinary differential equations _d**x**_/_dt_ = _f_(_t_,_**x**_,_**u**_,_c_) where 
_t_ is time, _**x**_ is a state vector, _**u**_ is a time series of exogeneous inputs, and _c_ contains a set of system constants. 

| module       | description                                                                  |
| ------------ | ---------------------------------------------------------------------------- |
| **ode4u**    | the ODE solution via a fixed-time step, 4th order method                     |
| **ode45u**   | the ODE solution via the adaptive time step, 4th-5th method by Cash and Karp |
| **odef**     | a library of multivariable ordinary differential equations, for testing      |
| **ode_test** | a library of tests for ode4u and ode45u                                      |

## opt . optimization

To minimize a function of multiple variables subject to a set of inequality constraints:
minimize _f_(_**v**_) such that _**g**_(_**v**_) < **0**,
where _**v**_ is a vector of design variables,
_f_(_**v**_) is a scalar-valued objective function, and
_**g**_(_**v**_) is a vector of inequality constraints. 

| module                 | description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| **avg_cov_func**       | estimate the value of an uncertain computation to desired precision       |
| **fsolve**             | solve a system of nonlinear algebraic equations                           |
| **L1_fit**             | linear least-squares curve fitting with l_1 regularization                |
| **L1_fit_test**        | test example for L1_fit                                                   |
| **L1_plots**           | plot results from L1_fit                                                  |
| **LP_analysis**        | to solve any LP using ors, nms, or sqp                                    |
| **mimoSHORSA**         | multi-input multi-output Stochastic High Order Response Surface Algorithm |
| **mimoSHORSA_example** | example of running mimoSHORSA                                             |
| **nms**                | nonlinear constrained optimization - Nelder Mead Simplex                  |
| **opt_example**        | example of runnin optimization codes ors, nms, sqp                        |
| **opt_options**        | adjust algorithmic options for ors, nms, and sqp                          |
| **ors**                | nonlinear constrained optimization - Optimized Random Search              |
| **plot_cvg_hist**      | plot the convergence historyof ors, nms and sqp                           |
| **plot_opt_surface**   | plot the objective function landscape in any selected 2D slice            |
| **poly_fit**           | power polynomial curve fitting with arbitrary exponents                   |
| **poly_fit_test**      | example function for poly_fit                                             |
| **sqp**                | nonlinear constrained optimization - Sequential Quadratic Programming     |

## rvs . random variables

To provide the probability density function, the cumulative distribution function, the inverse cumulative distribution function, and a random sample of various probabiity distributions.  
Correlated random samples of multivariate lognormal and beta random variables as well.  
methods: distribution.pdf(), distribution.cdf(), distribution.inv(), distribution.rnd()

| module               | description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **beta**             | [beta](http://en.wikipedia.org/wiki/Beta_distribution)                                              |
| **chi2**             | [chi-squared](http://en.wikipedia.org/wiki/Chi-squared_distribution)                                |
| **exponential**      | [exponential](http://en.wikipedia.org/wiki/Exponential_distribution)                                |
| **extreme_value_I**  | [type I extreme value]( )                                                                           |
| **extreme_value_II** | [type II extreme value]( )                                                                          |
| **gamma**            | [gamma](http://en.wikipedia.org/wiki/Gamma_distribution)                                            |
| **gev**              | [generalized extreme value](http://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)    |
| **laplace**          | [Laplace](http://en.wikipedia.org/wiki/Laplace_distribution)                                        |
| **lognormal**        | [lognormal](http://en.wikipedia.org/wiki/Log-normal_distribution)                                   |
| **normal**           | [Gaussian](http://en.wikipedia.org/wiki/Normal_distribution) (normal)                               |
| **plot_CDF_ci**      | plot the cumulative distribution function from data and its confidence interval                     |
| **poisson**          | [Poisson](http://en.wikipedia.org/wiki/Poisson_distribution)                                        |
| **quadratic**        | [quadratic](http://en.wikipedia.org/wiki/Beta_distribution) (special case of the beta distribution) |
| **rayleigh**         | [Rayleigh](http://en.wikipedia.org/wiki/Rayleigh_distribution)                                      |
| **students_t**       | [Student's t](https://en.wikipedia.org/wiki/Student%27s_t-distribution)                             |
| **triangular**       | [triangular](http://en.wikipedia.org/wiki/Triangular_distribution)                                  |
| **uniform**          | [uniform](http://en.wikipedia.org/wiki/Uniform_distribution_(continuous))                           |

## utl . utility functions

| module          | description                                    |
| --------------- | ---------------------------------------------- |
| **format_bank** | format a numerical string                      |
| **format_plot** | set the font size, line width, and marker size |
| **rainbow**     | rainbow line colors                            |

<!---
# installation for users
```bash
# start a virtual environment (venv) for a safe and clean installation
python3 -m venv .venv  
source .venv/bin/activate 

# choose (A) or (B) below
# (A) requires both PIP and GIT ... 
pip install git+https://github.com/hpgavin/multivarious.git  

# (B) requires PIP, but not GIT ...
pip install https://github.com/hpgavin/multivarious/archive/refs/heads/main.zip  
```

# installation for developers
```bash
# requires both PIP and GIT ... 
git clone https://github.com/hpgavin/multivarious.git
cd multivarious

# start a virtual environment (venv) for a safe and clean installation
python3 -m venv .venv
# on macOS / Linux
source .venv/bin/activate
# on Windows:
# .venv\Scripts\activate  

pip install -e .

```
-->

# installation 

```bash
# macOS and linux
# Python should already be installed 
# set PYTHONPATH in .profile ...
PYTHONPATH="$PYTHONPATH:/path/to/code/multivarious/"
export PYTHONPATH

# Windows 
# download and install Python from ... https://www.python.org/downloads/ 
# set PYTHONPATH through the Windows Control Panel
Control Panel -> System -> Advanced system settings -> Environment Variables -> User Variables -> PYTHONPATH
... add: 
"C:\full\path\to\code\multivarious;"
```
