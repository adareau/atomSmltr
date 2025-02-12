# -*- coding: utf-8 -*-
"""Defines the generic Field Class (for vector fields)
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from abc import abstractmethod

# % LOCAL IMPORTS
from ...utils.plotter import Plottable
from ...utils.infostring import InfoString


# % ABSTRACT CLASSES


class AbstractField(Plottable):
    """A generic, abstract class to handle fields (magnetic mostly)"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def value(self, x, y, z):
        """Will return the value of the field at point (x, y, z)"""
        pass

    # -- INFO STRING / OBJECT MANAGEMENT
    @abstractmethod
    def gen_infostring_obj(self):
        """should return the infostring object"""
        pass

    def gen_info_string(self, **kwargs):
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        print(self.gen_info_string())

    # -- PLOT
    # TODO > plot methods, at this level !!!
    def plot1D(self, ax=None):
        pass

    def plot2D(self, ax=None, plane="XY"):
        pass

    def plot3D(self, ax=None):
        pass


# % TOOL CLASSES


class AbstractOffsetField(AbstractField):
    """To generate perfect field offset"""

    def __init__(self, offset: npt.ArrayLike = (0, 0, 0)):
        """Generates a constant offset field

        Args:
            offset (npt.ArrayLike): offset of the field (array of size 3)
        """
        self.offset = offset

    # -- getters and setters
    @property
    @abstractmethod
    def type():
        """Type has to be defined in the concrete class"""
        pass

    @property
    def offset(self) -> npt.ArrayLike:
        return self.__offset

    @offset.setter
    def offset(self, value: npt.ArrayLike):
        value = np.asanyarray(value)
        if value.size != 3:
            raise ValueError("'offset' should be an array of size 3")
        if not np.all(np.isreal(value)):
            raise TypeError("'offset' should be an array of real numbers")
        self.__offset = value

    # -- requested methods for AbstractField
    def value(self, x, y, z):
        """Returns the value of the field at poins (x, y, z).
            Here we have an offset, so the field is constant
        Args:
            x (float): x position in lab frame
            y (float): y position in lab frame
            z (float): z position in lab frame

        Returns:
            value: the value of the field
        """
        return self.__offset

    def gen_infostring_obj(self):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "offset (constant field)")
        info.add_element("value", f"{self.offset}")
        info.add_element("norm", f"{np.linalg.norm(self.offset):.3g}")
        return info


class AbstractGradientField(AbstractField):
    """To generate perfect gradients"""

    def __init__(
        self,
        slope: float,
        offset: float,
        origin: npt.ArrayLike,
        gradient_direction: npt.ArrayLike,
        field_direction: npt.ArrayLike,
    ):
        pass
