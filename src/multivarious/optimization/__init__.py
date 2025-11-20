# multivarious/optimization/__init__.py
# multivarious/optimization/__init__.py

"""
Optimization algorithms: 
- ORS  (Optimized Random Search)
- NMS  (Nelder–Mead Simplex)
- SQP  (Sequential Quadratic Programming)
"""

# multivarious/optimization/__init__.py
from . import ors
from . import nms
from . import sqp

__all__ = ["ors", "nms", "sqp"]



