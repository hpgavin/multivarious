"""
multivarious.utl: a multivarious subpackage for general utility functions

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

# multivarious/utl/__init__.py

from .avg_cov_func import avg_cov_func
from .box_constraint import box_constraint
from .format_bank import format_bank
from .format_plot import format_plot
from .opt_options import opt_options
from .plot_cvg_hst import plot_cvg_hst
from .plot_ensemble import plot_ensemble
from .plot_lm import plot_lm
from .plot_opt_surface import plot_opt_surface
from .plot_spectra import plot_spectra 
from .plot_CDF_ci import plot_CDF_ci
from .L1_plots import L1_plots


__all__ = [ "avg_cov_func", 
            "box_constraint", 
            "format_bank", 
            "format_plot", 
            "opt_options", 
            "plot_cvg_hst", 
            "plot_ensemble", 
            "plot_lm", 
            "plot_opt_surface", 
            "plot_spectra", 
            "plot_CDF_ci", 
            "L1_plots", 
          ]


