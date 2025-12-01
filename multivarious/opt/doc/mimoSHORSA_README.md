# mimoSHORSA

**m**ulti-**i**nput **m**ulti-**o**tput **S**tochastic **H**igh **O**rder **R**esponse **S**urface **A**lgorithm

A sophisticated polynomial regression method using Hermite function basis with adaptive model reduction for high-dimensional data.

---------------------------------

## Usage

```
    order, coeff, meanX,meanY, trfrmX,trfrmY, testX,testY, testModelY = \
    mimoSHORSA(dataX, dataY, maxOrder=2, pTrain=70, pCull=0, cov_tol=0.10, scaling=1, L1_pnlty=1.0, basis_fctn='H') 

  mimoSHORSA
  multi-input multi-output Stochastic High Order Response Surface Algorithm

  This program fits a high order polynomial to multidimensional data via
  the stochastic high order response surface algorithm (mimoSHORSA) 

   mimoSHORSA approximates the data with a polynomial of arbitrary order,
     y(X) = a + \sum_{i=1}^n \sum_{j=1}^{k_i) b_ij X_i^j + 
            \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq}.
  The first stage of the algorithm determines the correct polynomial order,
  k_i in the response surace. Then the formulation of the mixed terms 
  \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq} are derived by the second stage
  based on previous results. In the third stage, the response surface 
  is approximated.

  INPUT       DESCRIPTION                                               DEFAULT
  --------    --------------------------------------------------------  -------
  dataX       m observations of n input  features in a (nx x m) matrix
  dataY       m observations of m output features in a (ny x m) vector
  maxOrder    maximum allowable polynomial order                            2
  pTrain      percentage of data for training (remaining for testing)      70
  pCull       maximum percentage of model which may be culled               0 
  cov_tol     desired maximum coefficient of variation of model coeff's  0.10
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


  OUTPUT      DESCRIPTION
  --------    --------------------------------------------------------
   order      matrix of the orders of variables in each term in the polynomial 
   coeff      polynomial coefficients 
   meanX      mean vector of the scaled dataX
   meanY      mean vector of the scaled dataY
   trfrmX     transformation matrix from dataZx to dataX
   trfrmY     transformation matrix from dataZy to dataY
   testX      input  features for model testing 
   testY      output features for model testing 
   testModelY output features for model testing
```

---

## Package Contents

### Core Implementation

- **`mimoSHORSA.py`** - Complete Python implementation (~1000 lines)
  - Main algorithm with 17 functions
  - Hermite polynomial basis
  - Legendre polynomial basis
  - Power polynomial basis
  - L1 regularization
  - optional iterative model reduction
  - adaptive adjustment of the L1 regularization penalty factor
  - multiple scaling options

### Test & Examples

- **`mimiSHORSA_example.py`** - Quick start example
  
  - Minimal dependencies
  - Measured data loading
  - Visualization utilities

- **`mimoSHORSA_test.py`** - Comprehensive test suite
  
  - Synthetic data generation
  - Measured data loading
  - Visualization utilities

### Documentation

- **`mimoSHORSA_THEORY.md`** - Complete theoretical foundation
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
from mimoSHORSA import mimoSHORSA

# mimoSHORSA parameters
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

# Run mimoSHORSA
order, coeff, meanX,meanY, trfrmX,trfrmY, testX,testY, testModelY = \
mimoSHORSA (
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
python mimoSHORSA_test.py
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
2006, 2023

---

**Version**: 1.1  
**Date**: November 12, 2025  
**Status**: Production Ready

*"Is this the fundamental problem statement of all data-science?"*
