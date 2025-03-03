# -*- coding: utf-8 -*-
"""Implements three-dimensional zones
"""

# % IMPORTS
import numpy as np

# % LOCAL IMPORTS
from .generic import Zone
from ...utils.infostring import InfoString

# % CLASSES


class Box(Zone):
    """docstring for UpperLimit."""

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        *args,
        **kwargs,
    ):
        """A 3D Box zone, along cartesian axes

        Args:
            xmin (float): minimum value for x
            xmax (float): maximum value for x
            ymin (float): minimum value for y
            ymax (float): maximum value for y
            zmin (float): minimum value for z
            zmax (float): maximum value for z
        """
        super(Box, self).__init__(*args, **kwargs)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    # -- GETTERS & SETTERS

    @property
    def type(self):
        return "3D Box"

    @property
    def xmin(self):
        return self.__xmin

    @xmin.setter
    def xmin(self, xmin):
        self.__xmin = float(xmin)

    @property
    def xmax(self):
        return self.__xmax

    @xmax.setter
    def xmax(self, xmax):
        self.__xmax = float(xmax)

    @property
    def ymin(self):
        return self.__ymin

    @ymin.setter
    def ymin(self, ymin):
        self.__ymin = float(ymin)

    @property
    def ymax(self):
        return self.__ymax

    @ymax.setter
    def ymax(self, ymax):
        self.__ymax = float(ymax)

    @property
    def zmin(self):
        return self.__zmin

    @zmin.setter
    def zmin(self, zmin):
        self.__zmin = float(zmin)

    @property
    def zmax(self):
        return self.__zmax

    @zmax.setter
    def zmax(self, zmax):
        self.__zmax = float(zmax)

    # -- ZONE

    def _in_zone(self, vector):
        x, y, z = vector.T
        in_zone = (
            (x > self.xmin)
            & (x < self.xmax)
            & (y > self.ymin)
            & (y < self.ymax)
            & (z > self.zmin)
            & (z < self.zmax)
        )
        return in_zone.T

    # -- INFOSTRING

    def gen_infostring_obj(self):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "3D Box")
        info.add_element("tag", self.tag)
        info.add_element("target", self.target)
        info.add_element("action", self.action)
        info.add_element(f"xmin, xmax", f"{self.xmin, self.xmax}")
        info.add_element(f"ymin, ymax", f"{self.ymin, self.ymax}")
        info.add_element(f"zmin, zmax", f"{self.zmin, self.zmax}")
        info.add_element(f"inverted", f"{self.inverted}")
        return info

    # -- PLOT

    def plot1D(self, ax=None):
        pass

    def plot2D(self, ax=None):
        pass

    def plot3D(self, ax=None):
        pass
