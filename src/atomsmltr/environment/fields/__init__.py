"""The ``atomsmltr.environment.fields`` subpackage provides definitions for vector fields

Currently, only magnetic fields are implemented, but this could be extented to electric fields

Examples
---------

Setup a magnetic field offset

>>> from atomsmltr.environment.fields import MagneticOffset
>>> offset_field = MagneticOffset(offset=(0,1,0), tag="offset")

"""

__all__ = [
    "MagneticGradient",
    "MagneticOffset",
    "MagneticQuadrupoleX",
    "MagneticQuadrupoleY",
    "MagneticQuadrupoleZ",
    "InterpMag1D1D",
]

from .magnetic import (
    MagneticGradient,
    MagneticOffset,
    MagneticQuadrupoleX,
    MagneticQuadrupoleY,
    MagneticQuadrupoleZ,
    InterpMag1D1D,
)
