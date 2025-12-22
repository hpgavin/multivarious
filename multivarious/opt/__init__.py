"""
multivarious.opt: a multivarious subpackage for optimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

    import multivarious as m
    m.utl.avg_cov_func(...)
    m.opt.sqp(...)
    m.opt.poly_fit(...)

or:

    from multivarious.opt import sqp
    y = sqp(...)

"""

# multivarious/opt/__init__.py

from .L1_fit import L1_fit
from .fsolve import fsolve
from .mimo_rs import mimo_rs
from .nms import nms
from .ors import ors
from .poly_fit import poly_fit
from .prony_fit import prony_fit
from .sqp import sqp

__all__ = [
    "L1_fit",
    "fsolve",
    "mimo_rs",
    "nms",
    "ors",
    "poly_fit",
    "prony_fit", 
    "sqp",
]
