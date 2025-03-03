# -*- coding: utf-8 -*-
"""Implements one-dimensional zones
"""

# % IMPORTS
import numpy as np

# % LOCAL IMPORTS
from .generic import Zone
from ...utils.infostring import InfoString

# % CLASSES


class SingleLimit(Zone):
    """docstring for UpperLimit."""

    def __init__(self, value: float, axis: int = 0, *args, **kwargs):
        super(SingleLimit, self).__init__(*args, **kwargs)
        self.axis = axis
        self.value = value

    # -- GETTERS & SETTERS

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = float(value)

    @property
    def axis(self):
        return self.__axis

    @axis.setter
    def axis(self, value):
        if value not in [0, 1, 2]:
            raise TypeError("'axis' should be 0, 1, 2")
        self.__axis = value

    # -- INFOSTRING

    def gen_infostring_obj(self):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        if isinstance(self, UpperLimit):
            info.add_element("type", "1D upper limit")
        elif isinstance(self, LowerLimit):
            info.add_element("type", "1D lower limit")
        info.add_element("tag", self.tag)
        info.add_element("target", self.target)
        info.add_element("action", self.action)
        info.add_element(f"value", f"{self.value}")
        info.add_element(f"axis", f"{self.axis}")
        info.add_element(f"inverted", f"{self.inverted}")
        return info

    # -- PLOT

    def plot1D(self, ax=None):
        pass

    def plot2D(self, ax=None):
        pass

    def plot3D(self, ax=None):
        pass


class UpperLimit(SingleLimit):
    """Zone defined by its upper limit, set by parameter 'value'"""

    def __init__(self, value: float, axis: int = 0, *args, **kwargs):
        """Defines a zone by its (1D) upper limit

        Args:
            value (float): the upper limit
            axis (int, optional): axis to consider (0:x, 1:y, 2:z)
        """
        super(UpperLimit, self).__init__(value, axis, *args, **kwargs)

    def _in_zone(self, vector):
        u = {}
        u[0], u[1], u[2] = vector.T
        in_zone = u[self.axis] < self.value
        return in_zone.T

    @property
    def type(self):
        return "Upper Limit"


class LowerLimit(SingleLimit):
    """Zone defined by its lower limit, set by parameter 'value'"""

    def __init__(self, value: float, axis: int = 0, *args, **kwargs):
        """Defines a zone by its (1D) lower limit

        Args:
            value (float): the lower limit
            axis (int, optional): axis to consider (0:x, 1:y, 2:z)
        """
        super(LowerLimit, self).__init__(value, axis, *args, **kwargs)

    def _in_zone(self, vector):
        u = {}
        u[0], u[1], u[2] = vector.T
        in_zone = u[self.axis] > self.value
        return in_zone.T

    @property
    def type(self):
        return "Upper Limit"


class Limits(Zone):
    """docstring for UpperLimit."""

    def __init__(self, min: float, max: float, axis: int = 0, *args, **kwargs):
        """Defines a 1D segment, with min / max value

        Args:
            min (float): minimum value
            max (float): maximum value
            axis (int, optional): axis to consider (0:x, 1:y, 2:z)
            *args and **kwargs sent to Zone()
        """
        super(Limits, self).__init__(*args, **kwargs)
        self.axis = axis
        self.min = min
        self.max = max

    # -- GETTERS & SETTERS
    @property
    def type(self):
        return "1D limits"

    @property
    def min(self):
        return self.__min

    @min.setter
    def min(self, min):
        self.__min = float(min)

    @property
    def max(self):
        return self.__max

    @max.setter
    def max(self, max):
        self.__max = float(max)

    @property
    def axis(self):
        return self.__axis

    @axis.setter
    def axis(self, value):
        if value not in [0, 1, 2]:
            raise TypeError("'axis' should be 0, 1, 2")
        self.__axis = value

    # -- ZONE

    def _in_zone(self, vector):
        u = {}
        u[0], u[1], u[2] = vector.T
        in_zone = (u[self.axis] > self.min) & (u[self.axis] < self.max)
        return in_zone.T

    # -- INFOSTRING

    def gen_infostring_obj(self):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "1D limits")
        info.add_element("tag", self.tag)
        info.add_element("target", self.target)
        info.add_element("action", self.action)
        info.add_element(f"min", f"{self.min}")
        info.add_element(f"max", f"{self.max}")
        info.add_element(f"axis", f"{self.axis}")
        info.add_element(f"inverted", f"{self.inverted}")
        return info

    # -- PLOT

    def plot1D(self, ax=None):
        pass

    def plot2D(self, ax=None):
        pass

    def plot3D(self, ax=None):
        pass
