# multivarious

A multitude of various Python modules for multivariable things like digital signal processing, linear time invariant systems, optimization, ordinary differential equations and random variables. 

This repository is under development.  Updates to this README lag the addition of code.  Stay tuned!!! 

## dsp . digitial signal processing

To generate, transform, and plot multivariate discrete time sequences 

| module          | description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| **accel2displ** | acceleration, velocity and displacement without much bias or drift from acceleration |
| **butter_synth_ss** | state space butterworth filter design using the matrix expnential |
| **cdiff** | central differences along the rows of a 2D array |
| **chrip** | generate a sine-sweep signal with its derivitive and integral |
| **eqgm_1d** | a single axis of simulated earthquake ground motions |
| **ftdsp** | Fourier transform based digitial signal processing | 
| **taper** | taper the ends of the rows of a 2D array - Planck or Tukey windows |

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
| **liap**          | wrapper for the ScyPi Sylvester equation (and Lyapunov equation) wrapper   |
| **lsym**          | the response of a continuous time LTI system                               |
| **mimo_bode**     | the Bode spectrum of a MIMO LTI system                                     |
| **mimo_tfe**      | the estimate of a frequency response matrix from MIMO data                 |
| **pz_plot**       | a plot of the poles and zeros on the complex plane                         |
| **sys_zero**      | MIMO system, invariant, transmissions, and decoupling zeros from (A,B,C,D) |
| **wiener_filter** | Markov parameters from time series: identification and simulation          |

## ode . ordinary differential equations

To solve systems of ordinary differential equations _d**x**_/_dt_ = _f_(_t_,_**x**_,_**u**_,_c_) where 
_t_ is time, _**x**_ is a state vector, _**u**_ is a time series of exogeneous inputs, and _c_ contains a set of system constants. 

| module     | description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| **ode4u**  | the ODE solution via a fixed-time step, 4th order method                     |
| **ode45u** | the ODE solution via the adaptive time step, 4th-5th method by Cash and Karp |

## opt . optimization

To minimize a function of multiple variables subject to a set of inequality constraints:
minimize _f_(_**v**_) such that _**g**_(_**v**_) < **0**,
where _**v**_ is a vector of design variables,
_f_(_**v**_) is a scalar-valued objective function, and
_**g**_(_**v**_) is a vector of inequality constraints. 

| module         | description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| **fsolve**     | solve a system of nonlinear algebraic equations                           |
| **L1_fit**     | linear least-squares curve fitting with l_1 regularization                |
| **mimoSHORSA** | multi-input multi-output Stochastic High Order Response Surface Algorithm |
| **nms**        | nonlinear constrained optimization - Nelder Mead Simplex                  |
| **ors**        | nonlinear constrained optimization - Optimized Random Search              |
| **poly_fit**   | power polynomial curve fitting with arbitrary exponents                   |
| **prony_fit**  | Prony function curve fitting with L1 regularization | 
| **sqp**        | nonlinear constrained optimization - Sequential Quadratic Programming     |

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
| **LP_analysis**      | to solve any LP using ors, nms, or sqp                                          |
| **avg_cov_func**     | estimate the value of an uncertain computation to desired precision             |
| **format_bank**      | format a numerical string                                                       |
| **format_plot**      | set the font size, line width, and marker size                                  |
| **opt_options**      | adjust algorithmic options for ors, nms, and sqp                                |
| **plot_CDF_ci**      | plot the cumulative distribution function from data and its confidence interval |
| **plot_cvg_hist**    | plot the convergence historyof ors, nms and sqp                                 |
| **plot_opt_surface** | plot the objective function landscape in any selected 2D slice                  |
| **rainbow**          | rainbow line colors                                                             |

# installation

* Install multivarious using git (for example, into a Code folder on your Desktop).
  
  Open a terminal window (Win11: (Win+X) and choose Windows Terminal) or (macOS: (Cmd+Space), type Terminal) and enter the two commands:
  
  ```
  cd ~/Desktop/Code
  git clone https://github.com/hpgavin/multivarious
  ```

* If you have not yet installed Python or VS Code, install Python (via [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)) and VS Code (via its [download page](https://code.visualstudio.com/Download)) 

* To configure VS Code, open VS Code and ...
  
  VS Code > File > Preferences > Settings > Search Settings ... <br> enter: python terminal execute ... <br> click the checkbox 
  
  VS Code > File > Preferences > Settings > Search Settings ... <br> enter: python terminal launch > Enter in settings.json ... <br> edit the following line, as shown below 
  
  ```
  "python.terminal.launchArgs": ["-i"] 
  ```
  
  Save your changes to this settings.json file and close the VS Code edit window. 

* Set the `PYTHONPATH` environment variable using (A) or (B) below 
  
  * **(A) For your VS Code installation, edit .vscode/settings.json :** 
    
    Open a terminal application and navigate to your multivarious/examples/.vscode directory
    
    ```bash
    cd ~/Desktop/Code/multivarious/examples/.vscode 
    ```
    
    open your settings.json file using one of these commands ... 
    
    ```bash
    notepad  settings.json     # Windows
    TextEdit settings.json     # macOS
    Gedit    settings.json     # linux
    ```
    
    and edit the section of your settings.json file corresponding to your OS by:
    
    1. changing `<USERNAME>` ... to the login ID for your computer
    
    2. changing `Desktop/Code` (or `Desktop\\Code`) to the path of your multivarious installation (e.g., `Desktop/stuff/Code`)
    
    Save and exit your editor. 
    
    Verify that your edits have taken effect by opening VS Code and: 
    
    1. VS Code > File > Open Folder ... > navigate to your multivarious/examples folder > Open
    
    2. VS Code > File Explorer > select verify_path_import.py > click the right arrow in the Edit menu [Run Python File]
    
    3. This will open a new Terminal Window which should display ...
       ```
       hello
       verifying that PYTHONPATH has been set ... 
       PYTHONPATH env: /home/hpgavin/Code/multivarious
       ... and yes, yes it has. Great!  
       verifying that multivarious can be imported ... 
       ... and yes, yes it can. Great!  
       ```
    
    4. [CTRL-D] - at the `>>>` Python prompt to exit the Python Interactive mode and return to the terminal prompt.   
    
    Copy your edited .vscode directory to any directory that will contain python code that uses the multivarious package. 
    
    Additional help on using settings.json to set the `PYTHONPATH` is provided in multivarious/examples/.vscode/SETTINGS_HELP.md 
  
  * **(B) For your computer, edit your profile directly :** 
    
    Open a terminal application and edit your profile using one of these commands ... 
    
    ```bash
    notepad     $PROFILE       # Windows
    TextEdit ~/.zprofile       # macOS
    Gedit     ~/.profile       # linux
    ```
    
    Copy one of the lines below and paste it into your profile. 
    
    ```bash
    $env:PYTHONPATH="$env:PYTHONPATH;%USERPROFILE%\Desktop\Code\multivarious/" # Windows
    export PYTHONPATH="$PYTHONPATH:$HOME/Desktop/Code/multivarious/"           # macOS and linux
    ```
    
    If you did not install multivarious into the Code directory on your Desktop, change `Desktop/Code` (or `Desktop\Code`) to match your selected installation directory. 
    
    Save your profile and close your editor.  
    
    Activate your edits by typing one of theses commands into the terminal ...  
    
    ```bash
    . $PROFILE                 # Windows
    source ~/.zprofile         # macOS
    source ~/.profile          # linux
    ```
    
    Verify the edits have taken effect by typing the one of these commands into the terminal ... 
    
    ```bash
    echo $env:PYTHONPATH       # Windows
    echo $PYTHONPATH           # macOS and linux 
    ```
    
    If the terminal shows that your `PYTHONPATH` is set as indended, it has worked. 
    VS Code will now find multivarious without the need for a .vscode/settings.json file.  
