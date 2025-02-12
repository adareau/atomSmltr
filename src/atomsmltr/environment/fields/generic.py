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
        super(AbstractField, self).__init__()

    @property
    @abstractmethod
    def type():
        """Type has to be defined in the concrete class"""
        pass

    @property
    @abstractmethod
    def unit():
        """unit has to be defined in the concrete class"""
        pass

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

    # -- USEFUL CHECK FUNCTIONS
    def _check_real_number(self, value, name):
        if np.asanyarray(value).size > 1:
            raise ValueError(f"'{name}' should be a scalar")
        if not np.isreal(value):
            raise TypeError(f"'{name}' should be a real numbers")
        return value

    def _check_3D_vector(self, value, name, norm=False):
        value = np.asanyarray(value)
        if value.size != 3:
            raise ValueError(f"'{name}' should be an array of size 3")
        if not np.all(np.isreal(value)):
            raise TypeError(f"'{name}' should be an array of real numbers")
        if norm:
            value = value / np.linalg.norm(value)
        return value


# % TOOL CLASSES


class AbstractOffsetField(AbstractField):
    """To generate perfect field offset"""

    def __init__(self, offset: npt.ArrayLike = (0, 0, 0)):
        """Generates a constant offset field

        Args:
            offset (npt.ArrayLike): offset of the field (array of size 3)
        """
        self.offset = offset
        super(AbstractField, self).__init__()

    # -- getters and setters

    @property
    def offset(self) -> npt.ArrayLike:
        return self.__offset

    @offset.setter
    def offset(self, value: npt.ArrayLike):
        self.__offset = self._check_3D_vector(value, "offset")

    # -- requested methods for AbstractField
    def value(self, x, y, z):
        """Returns the value of the field at point (x, y, z).
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
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "offset (constant field)")
        info.add_element(f"value ({unit})", f"{self.offset}")
        info.add_element(f"norm ({unit})", f"{np.linalg.norm(self.offset):.3g}")
        return info


class AbstractGradientField(AbstractField):
    """To generate perfect gradients"""

    def __init__(
        self,
        origin: npt.ArrayLike,
        slope: float,
        gradient_direction: npt.ArrayLike,
        field_direction: npt.ArrayLike,
        offset: float = 0.0,
    ):
        """Abstract Gradient

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
        self.slope = slope
        self.offset = offset
        self.origin = origin
        self.gradient_direction = gradient_direction
        self.field_direction = field_direction
        super(AbstractGradientField, self).__init__()

    # -- value
    def value(self, x, y, z):
        """Returns the value of the field at point (x, y, z).
        Args:
            x (float): x position in lab frame
            y (float): y position in lab frame
            z (float): z position in lab frame

        Returns:
            value: the value of the field
        """
        # - convert position to vector
        r = np.array([x, y, z])
        dr = r - self.origin

        # - distance to origin ?
        distance = np.dot(dr, self.gradient_direction)

        # - value at position
        value = (self.offset + distance * self.slope) * self.field_direction

        return value

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "perfect gradient")
        info.add_element(f"slope ({unit}/m)", f"{self.slope:.3g}")
        info.add_element("gradient direction", f"{self.gradient_direction}")
        info.add_element("field direction", f"{self.field_direction}")
        info.add_element(f"origin (m)", f"{self.origin}")
        info.add_element(f"offset ({unit})", f"{self.offset:3g}")
        return info

    # -- getters and setters
    # -
    @property
    def slope(self) -> npt.ArrayLike:
        return self.__slope

    @slope.setter
    def slope(self, value: npt.ArrayLike):
        self.__slope = self._check_real_number(value, "slope")

    # -
    @property
    def offset(self) -> npt.ArrayLike:
        return self.__offset

    @offset.setter
    def offset(self, value: npt.ArrayLike):
        self.__offset = self._check_real_number(value, "offset")

    # -
    @property
    def origin(self) -> npt.ArrayLike:
        return self.__origin

    @origin.setter
    def origin(self, value: npt.ArrayLike):
        self.__origin = self._check_3D_vector(value, "origin")

    # -
    @property
    def gradient_direction(self) -> npt.ArrayLike:
        return self.__gradient_direction

    @gradient_direction.setter
    def gradient_direction(self, value: npt.ArrayLike):
        self.__gradient_direction = self._check_3D_vector(
            value, "gradient_direction", norm=True
        )

    # -
    @property
    def field_direction(self) -> npt.ArrayLike:
        return self.__field_direction

    @field_direction.setter
    def field_direction(self, value: npt.ArrayLike):
        self.__field_direction = self._check_3D_vector(
            value, "field_direction", norm=True
        )
