"""
multivarious.fit: a multivarious subpackage for fitimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.fit.polynomial() 
  from multivarious.dsp import polynomial      polynomial()
  from multivarious.rvs import *               polynomial()

"""

# multivarious/fit/__init__.py

from .l1 import l1
from .lm import lm
from .lm import levenberg_marquardt
from .mimo_srs import mimo_srs
from .polynomial import polynomial
from .prony import prony

__all__ = [
    "l1",
    "lm", 
    "levenberg_marquardt", 
    "mimo_srs", 
    "polynomial",
    "prony", 
]
