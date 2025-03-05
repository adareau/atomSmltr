"""
volumes - 1D zones
=======================

Here we implement 3D zones, namely ``Box``

.. code-block:: python

    from atomsmltr.environment.zones import Box

    pos_box = Box(
        xmin=-10,
        xmax=10,
        ymin=0,
        ymax=5,
        zmin=-8,
        zmax=100,
        target="position",
        action="tag",
        tag="position box",
    )
"""

# % IMPORTS
import numpy as np

# % LOCAL IMPORTS
from .generic import Zone
from ...utils.infostring import InfoString

# % CLASSES


class Box(Zone):
    """A 3D box with cartesian coordinates

    Parameters
    ----------
    xmin : float
        minimum value for x
    xmax : float
        maximum value for x
    ymin : float
        minimum value for y
    ymax : float
        maximum value for y
    zmin : float
        minimum value for z
    zmax : float
        maximum value for z
    target : str, optional
        the target for the zone, can be "position" or "speed", by default "position"
    action : str, optional
        the action associated to the zone.
        Currently only "stop" is implemented, by default "stop"
    tag : str, optional
        the zone tag

    Example
    -------

    .. code-block:: python

        from atomsmltr.environment.zones import Box

        pos_box = Box(
            xmin=-10,
            xmax=10,
            ymin=0,
            ymax=5,
            zmin=-8,
            zmax=100,
            target="position",
            action="tag",
            tag="position box",
        )

    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        target: str = "position",
        action: str = "stop",
        tag: str = None,
    ):

        super(Box, self).__init__(target=target, action=action, tag=tag)
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
        """float: the minimum value for x"""
        return self.__xmin

    @xmin.setter
    def xmin(self, xmin):
        self.__xmin = float(xmin)

    @property
    def xmax(self):
        """float: the maximum value for x"""
        return self.__xmax

    @xmax.setter
    def xmax(self, xmax):
        self.__xmax = float(xmax)

    @property
    def ymin(self):
        """float: the minimum value for y"""
        return self.__ymin

    @ymin.setter
    def ymin(self, ymin):
        self.__ymin = float(ymin)

    @property
    def ymax(self):
        """float: the maximum value for y"""
        return self.__ymax

    @ymax.setter
    def ymax(self, ymax):
        self.__ymax = float(ymax)

    @property
    def zmin(self):
        """float: the minimum value for z"""
        return self.__zmin

    @zmin.setter
    def zmin(self, zmin):
        self.__zmin = float(zmin)

    @property
    def zmax(self):
        """float: the maximum value for z"""
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
