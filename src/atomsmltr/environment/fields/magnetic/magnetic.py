"""
magnetic fields
=======================

This module implements Magnetic field classes, which are mostly bare implementations of
generic fields defined in ``atomsmltr.environment.fields``.

See also
---------
atomsmltr.environment.fields.generic
atomsmltr.environment.fields.interpolated
"""

# % LOCAL IMPORTS
from ..generic import (
    Field,
    GradientField,
    ConstantField,
    QuadrupoleFieldX,
    QuadrupoleFieldZ,
    QuadrupoleFieldY,
    QuadrupoleField,
)
from ..interpolated import InterpolatedField1D1D, InterpolatedField3D3D

# % CLASSES

# -- MAG FIELDs PARENT CLASS
#   not really used currently, but will be useful if we need to
#   implement features specific to mag. fields.
# > will also allow to check that the field is indeed a magnetic field
#   in the configuration object


class MagneticField(Field):
    """A generic magnetic field class. Used to set some properties common to all magnetic
    fields objects, and to have a way to identify magnetic field objects."""

    @property
    def type(self):
        return "magnetic field"

    @property
    def unit(self):
        return "T"


# -- PERFECT FIELDS CLASSES
class MagneticOffset(MagneticField, ConstantField):
    """A perfect magnetic field offset

    See also
    ---------
    atomsmltr.environment.fields.generic.ConstantField
    """

    @property
    def type(self):
        return "magnetic field offset"


class MagneticGradient(MagneticField, GradientField):
    """A perfect magnetic field gradient

    See also
    ---------
    atomsmltr.environment.fields.generic.GradientField
    """

    @property
    def type(self):
        return "magnetic field gradient"


class MagneticQuadrupoleX(MagneticField, QuadrupoleFieldX):
    """A perfect magnetic field quadrupole, with strong axis along X

    See also
    ---------
    atomsmltr.environment.fields.generic.QuadrupoleFieldX
    """

    @property
    def type(self):
        return "magnetic quadrupole x"


class MagneticQuadrupoleY(MagneticField, QuadrupoleFieldY):
    """A perfect magnetic field quadrupole, with strong axis along Y

    See also
    ---------
    atomsmltr.environment.fields.generic.QuadrupoleFieldY
    """

    @property
    def type(self):
        return "magnetic quadrupole y"


class MagneticQuadrupoleZ(MagneticField, QuadrupoleFieldZ):
    """A perfect magnetic field quadrupole, with strong axis along Z

    See also
    ---------
    atomsmltr.environment.fields.generic.QuadrupoleFieldZ
    """

    @property
    def type(self):
        return "magnetic quadrupole z"


class MagneticQuadrupole(MagneticField, QuadrupoleField):
    """A perfect magnetic field quadrupole, with strong axis along a given vector (x,y,z)

    See also
    ---------
    atomsmltr.environment.fields.generic.QuadrupoleField
    """

    @property
    def type(self):
        return "magnetic quadrupole"


# -- INTERPOLATED
class InterpMag1D1D(MagneticField, InterpolatedField1D1D):
    """An interpolated field 1D / 1D

    See also
    ---------
    atomsmltr.environment.fields.interpolated.InterpolatedField1D1D
    """

    @property
    def type(self):
        return "interpolated mag. field (1D-1D)"


# -- INTERPOLATED
class InterpMag3D3D(MagneticField, InterpolatedField3D3D):
    """An interpolated field 3D/3D

    See also
    ---------
    atomsmltr.environment.fields.interpolated.InterpolatedField3D3D
    """

    @property
    def type(self):
        return "interpolated mag. field (3D-3D)"
