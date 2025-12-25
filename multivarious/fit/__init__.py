"""
multivarious.fit: a multivarious subpackage for fitimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

    import multivarious as m
    m.fit.lm.lm(...)
    m.fit.poly_fit(...)

or:

    from multivarious.fit.lm import lm
    from multivarious.fit.lm import levenberg_marquardt

    y = sqp(...)

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
