"""
multivarious.dsp: a multivarious subpackage for digital signal processing
"""

# multivarious/dsp/__init__.py

from .accel2displ import accel2displ
from .butter_synth_ss import butter_synth_ss
from .cdiff import cdiff
from .chirp import chirp
from .ftdsp import ftdsp
from .taper import taper

__all__ = [ "accel2displ" , "butter_synth_ss" , "cdiff" , "chirp" , "ftdsp' , "taper" ]
