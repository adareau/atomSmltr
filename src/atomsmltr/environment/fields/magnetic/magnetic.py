# -*- coding: utf-8 -*-
"""Defines the magnetic field class
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# % LOCAL IMPORTS
from ..generic import Field, GradientField, OffsetField
from ....utils.infostring import InfoString

# % CLASSES

# -- MAG FIELDs PARENT CLASS
#   not really used currently, but will be useful if we need to
#   implement features specific to mag. fields.
# > will also allow to check that the field is indeed a magnetic field
#   in the environment object


class MagneticField(Field):
    """Our magnetic field class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def type(self):
        return "magnetic field"

    @property
    def unit(self):
        return "T"


# -- PERFECT FIELDS CLASSES
class MagneticOffset(MagneticField, OffsetField):
    """Our magnetic field class"""

    def __init__(self, offset: float, tag: str = ""):
        """Generates a constant offset magnetic field

        Args:
            offset (npt.ArrayLike): offset of the field (array of size 3)
        """
        super(MagneticOffset, self).__init__(offset, tag)

    @property
    def type(self):
        return "magnetic field offset"


class MagneticGradient(MagneticField, GradientField):
    """Our magnetic field class"""

    def __init__(
        self,
        origin: npt.ArrayLike,
        slope: float,
        gradient_direction: npt.ArrayLike,
        field_direction: npt.ArrayLike,
        offset: float = 0.0,
        tag: str = "",
    ):
        """Magnetic field perfect gradient

        See below for arguments.

        Note that 'gradient_direction' and 'field_direction' are meant to be
        unit vectors, but the class will take care of normalizing any non normalized entry


        Args:
            origin (npt.ArrayLike): origin for the gradient (array of size 3)
            slope (float): the slope of the gradient (scalar)
            gradient_direction (npt.ArrayLike): the direction of the gradient (array of size 3)
            field_direction (npt.ArrayLike): the field direction (array of size 3)
            offset (float, optional): the field offset, at origin (scalar). Defaults to 0.0.
        """
        super(MagneticGradient, self).__init__(
            origin=origin,
            slope=slope,
            gradient_direction=gradient_direction,
            field_direction=field_direction,
            offset=offset,
            tag=tag,
        )

    @property
    def type(self):
        return "magnetic field gradient"
