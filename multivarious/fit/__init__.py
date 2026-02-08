"""
multivarious.fit: a multivarious subpackage for fitimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.fit.poly_fit() 
  from multivarious.dsp import poly_fit        poly_fit()
  from multivarious.rvs import *               poly_fit()

"""

# multivarious/fit/__init__.py

from .L1_fit import L1_fit
from .lm import lm
from .lm import levenberg_marquardt
from .mimo_rs import mimo_rs
from .poly_fit import poly_fit
from .prony_fit import prony_fit

__all__ = [
    "L1_fit",
    "lm", 
    "levenberg_marquardt", 
    "poly_fit",
    "prony_fit", 
]
