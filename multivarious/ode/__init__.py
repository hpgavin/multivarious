"""
multivarious.ode: a multivarious subpackage for ordinary differential equations
"""

# multivarious/ode/__init__.py

from .ode4u import ode4u
from .ode45u import ode45u
from .ode4ucc import ode4ucc
from .ode45ucc import ode45ucc

__all__ = [ "ode4u", "ode45u", "ode4ucc", "ode45ucc", ]
