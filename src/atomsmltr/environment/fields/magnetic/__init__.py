"""The ``atomsmltr.environment.fields.magnetic`` subpackage provides definitions for magnetic fields

Those are mostly direct implementations of generic `Fields` objects
from `atomsmltr.environment.fields.generic`

Examples
---------

Setup a magnetic field offset

.. code-block:: python

    from atomsmltr.environment.fields import MagneticOffset
    offset_field = MagneticOffset(offset=(0,1,0), tag="offset")

See also
--------
atomsmltr.environment.fields.generic
atomsmltr.environment.fields.interpolated

"""

__all__ = [
    "MagneticField",
    "MagneticGradient",
    "MagneticOffset",
    "MagneticQuadrupoleX",
    "MagneticQuadrupoleY",
    "MagneticQuadrupoleZ",
    "MagneticQuadrupole",
    "InterpMag1D1D",
    "InterpMag3D3D",
]
from .magnetic import (
    MagneticField,
    MagneticGradient,
    MagneticOffset,
    InterpMag1D1D,
    InterpMag3D3D,
    MagneticQuadrupoleX,
    MagneticQuadrupoleY,
    MagneticQuadrupoleZ,
    MagneticQuadrupole,
)
