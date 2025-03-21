"""
forces
=======================

This module implements Forces field classes, which are mostly bare implementations of
generic fields defined in ``atomsmltr.environment.fields``.

See also
---------
atomsmltr.environment.fields.generic
"""

# % LOCAL IMPORTS
from ..generic import (
    Field,
    GradientField,
    ConstantField,
)

# % CLASSES

# -- FORCE PARENT CLASS
#   not really used currently, but will be useful if we need to
#   implement features specific to forces fields.
# > will also allow to check that the field is indeed a force
#   in the configuration object


class Force(Field):
    """A generic force field class. Used to set some properties common
    to all forces objects, and to have a way to identify magnetic field objects."""

    @property
    def type(self):
        return "Force"

    @property
    def unit(self):
        return "N"


# -- PERFECT FIELDS CLASSES
class ConstantForce(Force, ConstantField):
    """A constant Force

    See also
    ---------
    atomsmltr.environment.fields.generic.ConstantField
    """

    @property
    def type(self):
        return "constant force"


class GradientForce(Force, GradientField):
    """A perfect force gradient

    See also
    ---------
    atomsmltr.environment.fields.generic.GradientField
    """

    @property
    def type(self):
        return "gradient force"
