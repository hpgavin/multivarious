"""
multivarious.opt: a multivarious subpackage for optimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

    import multivarious as m
    m.utl.avg_cov_func(...)
    m.opt.sqp(...)
    m.opt.poly_fit(...)
    m.opt.lm.levenberg_marquardt


or:

    from multivarious.opt import sqp
    from multivarious.opt.lm import levenberg_marquardt, lm

    y = sqp(...)

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
