# multivarious

A multitude of various Python modules for multivariable things like digital signal processing, fitting models to data, linear time invariant systems, optimization, ordinary differential equations and random variables. 

Multivarious is pedagogical.  It emphasizes a transparent implementation of numerical methods for multivariate problems with integrated visualization tools. 
The methods and code in this repository stem from decades of undergraduate and graduate engineering instruction.  

This repository is under development.  Updates to this README lead or lag the addition of code. 

Stay tuned and keep pulling!!! 

## dsp . digitial signal processing

To generate, transform, and plot multivariate discrete time sequences 

| module              | description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| **accel2displ**     | acceleration, velocity and displacement without much bias or drift from acceleration    |
| **autocorr**        | autocorrelation of a time series                                                        |
| **butter_synth_ss** | state space butterworth filter design using the matrix expnential                       |
| **cdiff**           | central differences along the rows of a 2D array                                        |
| **chrip**           | onomatopoeia for bird sounds and <br> sine-sweep signals with derivitives and integrals |
| **csd**             | estimate the cross-power spectra of a pair of signals and its chi^2 confidince interval |
| **eqgm_1d**         | a single axis of simulated earthquake ground motions                                    |
| **ftdsp**           | Fourier transform based digitial signal processing                                      |
| **lers_2d**         | response spectrum for biaxial motion                                                    |
| **psd**             | estimate the auto-power spectral density of a signal and its chi^2 confidence interval  |
| **taper**           | taper the ends of the rows of a 2D array - Planck or Tukey windows                      |

## fit . fit mathematical models to data

| module        | description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| **L1_fit**    | linear least-squares curve fitting with l_1 regularization             |
| **lm**        | Levenberg-Marquardt for nonlinear least squares curve-fitting problems |
| **mimo_rs**   | multi-input multi-output response surface                              |
| **poly_fit**  | power polynomial curve fitting with arbitrary exponents                |
| **prony_fit** | Prony function curve fitting with L1 regularization                    |

## lti . linear time invariant systems

To analyze and transform linear time invariant systems
defined by 
linear differential equations, _d**x**_/_dt_ = _**A x**_(_t_) + _**B u**_(_t_)
or linear difference equations,  _**x**_(_k_+1) = _**A x**_(_k_) + _**B u**_(_k_)
and corresponding system outputs, _**y**_ = _**C x**_ + _**D u**_

| module            | description                                                                |
| ----------------- | -------------------------------------------------------------------------- |
| **abcd_dim**      | dimensions of an LTI realization                                           |
| **blk_hankel**    | a block Hankel matrix from a matrix-valued time series                     |
| **blk_toeplitz**  | a block Toeplitz matrix from a matrix-valued time series                   |
| **ctrb**          | controllability gramian of (A,B) 
| **con2dis**       | a discrete time LTI system from a continuous time LTI system               |
| **damp**          | natural frequencies and damping ratios of a dynamics matrix                |
| **dis2con**       | a continuous time LTI system from a discrete time LTI system               |
| **dliap**         | wrapper for the ScyPi Sylvester equation (and Lyapunov equation) solver    |
| **dlsym**         | the response of a discrete time LTI system                                 |
| **kalman_decomp** | Kalman canonical form from an LTI state-space realization                  |
| **liap**          | wrapper for the ScyPi Sylvester equation (and Lyapunov equation) solver    |
| **lsym**          | the response of a continuous time LTI system                               |
| **mimo_bode**     | the Bode spectrum of a MIMO LTI system                                     |
| **mimo_tfe**      | the estimate of a frequency response matrix from MIMO data                 |
| **obsv**          | observability gramian of (A,C) 
| **pz_plot**       | a plot of the poles and zeros on the complex plane                         |
| **sys_zero**      | MIMO system, invariant, transmissions, and decoupling zeros from (A,B,C,D) |
| **wiener_filter** | Markov parameters from time series: identification and simulation          |

## ode . ordinary differential equations

To solve systems of ordinary differential equations _d**x**_/_dt_ = _f_(_t_,_**x**_,_**u**_,_c_) where 
_t_ is time, _**x**_ is a state vector, _**u**_ is a time series of exogeneous inputs, and _c_ contains a set of system constants. 

| module       | description                                                                                         |
| ------------ | --------------------------------------------------------------------------------------------------- |
| **ode4u**    | ODE solution via a fixed-time step, 4th order method                                                |
| **ode45u**   | ODE solution via the adaptive time step, 4th-5th method by Cash and Karp                            |
| **ode4ucc**  | ODE solution via a fixed-time step, 4th order method, and constraint correction                     |
| **ode45ucc** | ODE solution via the adaptive time step, 4th-5th method by Cash and Karp, and constraint correction |

## opt . optimization

To minimize a function of multiple variables subject to a set of inequality constraints:
minimize _f_(_**v**_) such that _**g**_(_**v**_) < **0**,
where _**v**_ is a vector of design variables,
_f_(_**v**_) is a scalar-valued objective function, and
_**g**_(_**v**_) is a vector of inequality constraints. 

| module     | description                                                           |
| ---------- | --------------------------------------------------------------------- |
| **fsolve** | solve a system of nonlinear algebraic equations                       |
| **nms**    | Nelder Mead Simplex              |
| **ors**    | Optimized Random Search          |
| **sqp**    | Sequential Quadratic Programming |

## rvs . random variables

To provide the probability density function, the cumulative distribution function, the inverse cumulative distribution function, and a random sample of various probabiity distributions.  
Correlated random samples of multivariate lognormal and beta random variables as well.  
methods: distribution.pdf(), distribution.cdf(), distribution.inv(), distribution.rnd()

| module               | description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| **beta**             | [beta](http://en.wikipedia.org/wiki/Beta_distribution)                                              |
| **binomial**         | [binomial](http://en.wikipedia.org/wiki/Binomial_distribution)                                              |
| **chi2**             | [chi-squared](http://en.wikipedia.org/wiki/Chi-squared_distribution)                                |
| **exponential**      | [exponential](http://en.wikipedia.org/wiki/Exponential_distribution)                                |
| **extreme_value_I**  | [type I extreme value]( )                                                                           |
| **extreme_value_II** | [type II extreme value]( )                                                                          |
| **gamma**            | [gamma](http://en.wikipedia.org/wiki/Gamma_distribution)                                            |
| **gev**              | [generalized extreme value](http://en.wikipedia.org/wiki/Generalized_extreme_value_distribution)    |
| **laplace**          | [Laplace](http://en.wikipedia.org/wiki/Laplace_distribution)                                        |
| **lognormal**        | [lognormal](http://en.wikipedia.org/wiki/Log-normal_distribution)                                   |
| **normal**           | [Gaussian](http://en.wikipedia.org/wiki/Normal_distribution) (normal)                               |
| **poisson**          | [Poisson](http://en.wikipedia.org/wiki/Poisson_distribution)                                        |
| **quadratic**        | [quadratic](http://en.wikipedia.org/wiki/Beta_distribution) (special case of the beta distribution) |
| **rayleigh**         | [Rayleigh](http://en.wikipedia.org/wiki/Rayleigh_distribution)                                      |
| **students_t**       | [Student's t](https://en.wikipedia.org/wiki/Student%27s_t-distribution)                             |
| **triangular**       | [triangular](http://en.wikipedia.org/wiki/Triangular_distribution)                                  |
| **uniform**          | [uniform](http://en.wikipedia.org/wiki/Uniform_distribution_(continuous))                           |

## utl . utility functions

| module               | description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| **L1_plots**         | plot results from L1_fit                                                        |
| **avg_cov_func**     | estimate the value of an uncertain computation to desired precision             |
| **format_bank**      | format a numerical string                                                       |
| **format_plot**      | set the font size, line width, and marker size                                  |
| **opt_options**      | adjust algorithmic options for ors, nms, and sqp                                |
| **plot_CDF_ci**      | plot the cumulative distribution function from data and its confidence interval |
| **plot_ensemble**    | plot three sets of corresponding ensemble time series                           |
| **plot_cvg_hist**    | plot the convergence history from ors, nms and sqp                              |
| **plot_lm**          | plot the converence history and fit statistics from an lm analysis              |
| **plot_opt_surface** | plot the objective function landscape in any selected 2D slice                  |
| **plot_spectra**     | plot three sets of corresponding ensemble spectra or transfer functions         |

# installation

If you have not yet installed git, Python or VS Code, follow the instructions in [section 2 of this Python tutorial](https://www.duke.edu/~hpgavin/pythonSkills+Debugging.pdf) 

A. Download or Update: Open any terminal (a VS Code Terminal or any terminal app on your computer) and: (1.) download a fresh copy or (2.) update an existing copy 
  1. download a fresh copy 

   ``` bash
   cd ~/Desktop/Code
   git clone https://github.com/hpgavin/multivarious
   ```
  2. update an existing copy 

   ``` bash
   cd ~/Desktop/Code/multivarious 
   git pull
   ```

B.  (1.) `pip install` for VS Code and/or (2.) set the `PYTHONPATH` or (3.) do something scarry or (4.) use a venv.   
  A chat on issues with Debian, python, pip, and PYTHONPATH is in multivarious/examples/doc/. 

  1. Within a VS Code Terminal,  (Terminal > New Terminal) 

   ``` bash
   cd ~/Desktop/Code/multivarious  
   pip install .  
   ```

  This will create directories `build/` and `multivarious.egg-info/.` You may delete the `build/` directory.  Keep the `multivarious.egg-info/` directory.  If `pip install` indicates the error: `error: externally-managed-environment` then set the `PYTHONPATH` (option (2.) below)

  2. (If you have completed (1.) above and are using Python only within VS Code, then this step is optional.)
     Open one of the following links and follow the instructions.  
   * [Windows-shell.txt](https://people.duke.edu/~hpgavin/Windows-shell.txt)  
   * [macOS-shell.txt](https://people.duke.edu/~hpgavin/macOS-shell.txt)   
   * [linux-shell.txt](https://people.duke.edu/~hpgavin/linux-shell.txt)

  3. If both (1.) and (2.) fail, please know that you can confidently install multivarious by doing this scarry thing

   ``` bash
   cd ~/Desktop/Code/multivarious 
   pip install --break-system-packages . 
   ```

  4. Use a venv 

C. Verify that VS Code has access to the multivarious library. 
  
   VS Code > Terminal > New Terminal 

   ``` bash
   python
   >>> import multivarious
   ```

   If this message ... `ModuleNotFoundError: No module named 'multivarious'` ... does not appear, you have installed `multivarious` correctly.

