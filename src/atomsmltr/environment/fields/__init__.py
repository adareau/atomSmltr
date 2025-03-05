"""The ``atomsmltr.environment.fields`` subpackage provides definitions for vector fields

Currently, only magnetic fields are implemented, but this could be extented to electric fields

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
atomsmltr.environment.fields.magnetic

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
