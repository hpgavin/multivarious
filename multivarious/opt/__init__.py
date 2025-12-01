"""
multivarious.opt: a multivarious subpackage for optimization

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

    import multivarious as m
    m.opt.avg_cov_func(...)
    m.opt.sqp(...)
    m.opt.poly_fit(...)

or:

    from multivarious.opt import sqp
    y = sqp(...)

"""

# multivarious/opt/__init__.py

from .avg_cov_func import avg_cov_func
from .box_constraint import box_constraint
from .L1_fit import L1_fit
from .L1_plots import L1_plots
from .LP_analysis import LP_analysis
from .fsolve import fsolve
from .mimoSHORSA import mimoSHORSA
from .nms import nms
from .opt_options import opt_options
from .ors import ors
from .plot_cvg_hst import plot_cvg_hst
from .plot_opt_surface import plot_opt_surface
from .poly_fit import poly_fit
from .sqp import sqp

__all__ = [
    "avg_cov_func",
    "box_constraint",
    "L1_fit",
    "L1_plots",
    "LP_analysis",
    "fsolve",
    "mimoSHORSA",
    "nms",
    "opt_options",
    "ors",
    "plot_cvg_hst",
    "plot_opt_surface",
    "poly_fit",
    "sqp",
]
