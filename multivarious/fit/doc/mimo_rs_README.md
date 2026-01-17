# mimo_rs

**m**ulti-**i**nput **m**ulti-**o**tput  **r**esponse **s**urface 

This program fits a polynomial to multidimensional data by projecting the data onto a polynomial basis of oder up to 10 (or more).  Data may be scaled, standardized or decorrelated before fitting.  The polynomial basis may be Hermite, Legendre, or Power polynomials.  The model complexity is managed via L1 regularization. 

---------------------------------

## Usage

```
    order, coeff, meanX,meanY, trfrmX,trfrmY, testX,testY, testModelY = \
    mimo_rs(dataX, dataY, maxOrder=2, pTrain=70, pCull=0, cov_tol=0.10, scaling=1, L1_pnlty=1.0, basis_fctn='H', var_names=None ) 

  mimo_rs : multi-input multi-output response surface 

  approximates the data with a polynomial of arbitrary order,

     y(X) = a + \sum_{i=1}^n \sum_{j=1}^{k_i) b_ij X_i^j + 
            \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq}.

  INPUT       DESCRIPTION                                               DEFAULT
  --------    --------------------------------------------------------  -------
  dataX       m observations of nx input  features in a (nx x m) matrix
  dataY       m observations of ny output features in a (ny x m) vector
  maxOrder    maximum allowable polynomial order                            2
  pTrain      percentage of data for training (remaining for testing)      70
  scaling     scale the data before fitting                                 1
              scaling = 0: no scaling
              scaling = 1: subtract mean and divide by std.dev
              scaling = 2: subtract mean and decorrelate
              scaling = 3: log-transform, subtract mean and divide by std.dev
              scaling = 4: log-transform, subtract mean and decorrelate
  L1_pnlty    coefficient for L1 regularization                           1.0
  basis_fctn  basis function type                                         'H'
              'H': Hermite functions
              'L': Legendre polynomials
              'P': Power polynomials
  var_names   optional dictionary with keys 'X' and 'Y' containing
              lists of variable names for labeling                       None

  OUTPUT      DESCRIPTION
  --------    --------------------------------------------------------
   ordr       list of matrices of the orders of variables in each polynomial term 
   coeff      list of polynomial coefficients 
   meanX      mean vector of the scaled dataX
   meanY      mean vector of the scaled dataY
   invTX      inverse transformation matrix from dataX to dataZx (full col rank)
   TY         transformation matrix from dataZy to dataY (full row rank)
   testX      input  features for model testing 
   testY      output features for model testing 
   testModelY output features for model testing


```

---

## Package Contents

### Core Implementation

- **`mimo_rs.py`** - Complete Python implementation (~1000 lines)
  - Main algorithm with 16 functions
  - multiple scaling options
  - Hermite polynomial basis
  - Legendre polynomial basis
  - Power polynomial basis
  - L1 regularization
  - adaptive adjustment of the L1 regularization penalty factor

### Test & Examples

- **`mimiSHORSA_example.py`** - Quick start example
  
  - Minimal dependencies
  - Measured data loading
  - Visualization utilities

- **`mimo_rs_test.py`** - Comprehensive test suite
  
  - Synthetic data generation
  - Measured data loading
  - Visualization utilities

### Documentation

- **`mimo_rs_THEORY.md`** - Complete theoretical foundation
  - Mathematical derivations
  - Algorithm details
  - Application guidance

---

## Quick Start

### Installation

```bash
# Required packages
pip install numpy matplotlib
```

### Basic Usage

```python
import numpy as np
from multivarious.opt import mimo_rs

# mimo_rs parameters
maxOrder = 2    # maximum polynomial order for the model
pTrain = 70     # percentage of the data for training (remaining for testing)

pCull = 0       # percentage of the model to be culled
cov_tol = 0.10  # maximum desired coefficient of variation

#scaling = 0     # no scaling
scaling = 1     # subtract mean and divide by std.dev
#scaling = 2     # subtract mean and decorrelate
#scaling = 3     # log-transform, subtract mean and divide by std.dev
#scaling = 4     # log-transform, subtract mean and decorrelate

L1_pnlty = 1.0   # penalty factor for L1 regularization

# basis_function = 'P'  # power polynomials
# basis_function = 'L'  # Legendre polynomials
basis_function = 'H'  # Hermite functions


# Your data
# dataX: (nInp x mData) input features
# dataY: (nOut x mData) output features

# Run mimo_rs
order, coeff, meanX,meanY, trfrmX,trfrmY, testX,testY, testModelY = \
mimo_rs (
    dataX      ,        # the explanatory features, in rows ... (nInp x nData)
    dataY      ,        # the  dependent  features, in rows ... (nOut x nData)
    maxOrder   = 2,     # the largest product of exponennts in any single term
    pTrain     = 70,    # percentage of the data used for training
    pCull      = 0,     # percentage of the basis to cull 
    cov_tol    = 0.1,   # desired maximum model coefficient of variation
    scaling    = 1,     # five scaling options
    L1_pnlty   = 1.00,  # L1 regularization penalty factor
    basis_fctn = 'H' )  # 'H' Hermite, 'L' Legendre, 'P' power polynomial
```

### Run Examples

```bash
# Comprehensive test with synthetic data
python mimo_rs_test.py
```

---

## Reference

### Primary Publication

Gavin, H.P. and Yau, S.C., "High order limit state functions in the 
response surface method for structural reliability analysis,"
*Structural Safety*, Vol. 30, pp. 162-179, 2008.

### Authors

- Siu Chung Yau
- Henri P. Gavin

Department of Civil and Environmental Engineering  
Duke University  
2006, 2023, 2025

---

**Version**: 3.1  
**Date**: November 12, 2025  

---

*"Is this the fundamental problem statement of all data-science?"*
