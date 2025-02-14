# -*- coding: utf-8 -*-
"""Defines the magnetic field class
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import magpylib as magpy

# % LOCAL IMPORTS
from .generic import AbstractField, AbstractGradientField, AbstractOffsetField
from ...utils.infostring import InfoString

# % CLASSES

# -- MAG FIELDs PARENT CLASS
#   not really used currently, but will be useful if we need to
#   implement features specific to mag. fields.
# > will also allow to check that the field is indeed a magnetic field
#   in the environment object


class MagneticField(AbstractField):
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
class MagneticOffset(MagneticField, AbstractOffsetField):
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


class MagneticGradient(MagneticField, AbstractGradientField):
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


class MagpylibWrapper(MagneticField):
    """Our magnetic field class"""

    def __init__(self, magpy_object, tag: str = ""):
        """Generates a constant offset magnetic field

        Args:
            offset (npt.ArrayLike): offset of the field (array of size 3)
        """
        super(MagpylibWrapper, self).__init__(tag)
        self.magpy_object = magpy_object

    @property
    def type(self):
        return "magpylib object"

    # -- requested methods for AbstractField
    # pylint : disable=method_hidden
    @staticmethod
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `AbstractField` class
        """
        # let's call the magpy get_B function
        B = magpy.getB(self.magpy_object, position, squeeze=False)
        # sqeeeeeeze
        value = B.T
        while value.ndim > position.ndim:
            value = np.squeeze(value, axis=-1)
        return value.T

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "magpylib object")
        # TODO can we have more info ?
        return info
