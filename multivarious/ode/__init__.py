"""
multivarious.ode: a multivarious subpackage for ordinary differential equations
"""

# multivarious/ode/__init__.py

from .ode4u import ode4u
from .ode45u import ode45u

__all__ = [ "ode4u", "ode45u" ]
