"""
multivarious.ode: a multivarious subpackage for ordinary differential equations

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.ode.ode4u() 
  from multivarious.ode import lsym            ode4u()
  from multivarious.ode import *               ode4u()

"""

# multivarious/ode/__init__.py

from .ode4u import ode4u
from .ode45u import ode45u
from .ode4ucc import ode4ucc
from .ode45ucc import ode45ucc

__all__ = [ "ode4u", "ode45u", "ode4ucc", "ode45ucc", ]
