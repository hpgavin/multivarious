"""
multivarious.opt: a multivarious subpackage for optimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.opt.nms() 
  from multivarious.opt import nms             nms()
  from multivarious.opt import *               nms()

"""

# multivarious/opt/__init__.py

from .fsolve import fsolve
from .nms import nms
from .ors import ors
from .sqp import sqp

__all__ = [
    "fsolve",
    "nms",
    "ors",
    "sqp",
]
