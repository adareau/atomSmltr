"""The ``atomsmltr.environment.fields`` subpackage provides definitions for vector fields

Currently, only magnetic fields are implemented, but this could be extented to electric fields

Examples
---------

Setup a magnetic field offset

.. code-block:: python

    from atomsmltr.environment.fields import MagneticOffset
    offset_field = MagneticOffset(field_value=(0,1,0), tag="offset")

See also
--------
atomsmltr.environment.fields.generic
atomsmltr.environment.fields.interpolated
atomsmltr.environment.fields.magnetic

"""

__all__ = [
    # - Magnetic Fields
    "MagneticGradient",
    "MagneticOffset",
    "MagneticQuadrupoleX",
    "MagneticQuadrupoleY",
    "MagneticQuadrupoleZ",
    "MagneticQuadrupole",
    "InterpMag1D1D",
    "InterpMag3D3D",
    # - Forces
    "Force",
    "GradientForce",
    "ConstantForce",
]

from .magnetic import (
    MagneticGradient,
    MagneticOffset,
    MagneticQuadrupoleX,
    MagneticQuadrupoleY,
    MagneticQuadrupoleZ,
    MagneticQuadrupole,
    InterpMag1D1D,
    InterpMag3D3D,
)

from .force import (
    Force,
    GradientForce,
    ConstantForce,
)
