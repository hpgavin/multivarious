"""
multivarious.dsp: a multivarious subpackage for digital signal processing

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.dsp.chrip() 
  from multivarious.dsp import chrip           chrip()
  from multivarious.dsp import *               chrip()

"""

# multivarious/dsp/__init__.py

from . import accel2displ, autocorr, butter_synth_ss, cdiff, chrip, csd, eqgm_1d, ftdsp, lers_2d, psd, taper


from .accel2displ import accel2displ
from .autocorr import autocorr
from .butter_synth_ss import butter_synth_ss
from .cdiff import cdiff
from .chrip import chrip
from .csd import csd
from .eqgm_1d import eqgm_1d
from .ftdsp import ftdsp
from .lers_2d import lers_2d
from .psd import psd
from .taper import taper

__all__ = [ "accel2displ" , "autocorr", "butter_synth_ss", "cdiff", "chrip", "csd", "eqgm_1d", "ftdsp", "lers_2d", "psd", "taper", ]
