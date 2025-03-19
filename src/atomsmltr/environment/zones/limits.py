"""
limits - 1D zones
=======================

Here we implement 1D zones, namely ``UpperLimit``, ``LowerLimit`` and ``Limits``

.. code-block:: python

    from atomsmltr.environment.zones import Limits

    xlim = Limits(min=-1, max=1, axis=0, target="position", action="stop", tag="xlim")
    vxlim = Limits(min=0, max=500, axis=0, target="speed", action="stop", tag="vxlim")
"""

# % IMPORTS
import numpy as np

# % LOCAL IMPORTS
from .generic import Zone
from ...utils.infostring import InfoString

# % CLASSES


class SingleLimit(Zone):
    """A generic class for SingleLimits"""

    def __init__(self, value: float, axis: int = 0, *args, **kwargs):
        super(SingleLimit, self).__init__(*args, **kwargs)
        self.axis = axis
        self.value = value

    # -- GETTERS & SETTERS

    @property
    def value(self) -> float:
        """float: the limit value"""
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = float(value)

    @property
    def axis(self) -> int:
        """int: the limit target axis"""
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
        info.add_element("in_tag", self.in_tag)
        info.add_element("out_tag", self.tag)
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
    """Defines a zone by its (1D) upper limit

    Parameters
    ----------
    value : float
        the upper limit
    axis : int, optional
        axis to consider (0:x, 1:y, 2:z), by default 0
    target : str, optional
        the target for the zone, can be "position" or "speed", by default "position"
    action : str, optional
        the action associated to the zone.
        implemented actions = ["stop", "ignore"], default is "stop"
    tag : str, optional
        the zone tag
    in_tag : str, optional
        tag for an object inside the zone, by default None
    out_tag : str, optional
        tag for an object inside the zone, by default None

    Example
    -------
    >>> up_x = UpperLimit(value=5, axis=0, tag="upx")

    """

    def __init__(
        self,
        value: float,
        axis: int = 0,
        target: str = "position",
        action: str = "stop",
        tag: str = None,
        in_tag: str = None,
        out_tag: str = None,
    ):
        super(UpperLimit, self).__init__(
            value=value,
            axis=axis,
            target=target,
            action=action,
            tag=tag,
            in_tag=in_tag,
            out_tag=out_tag,
        )

    def _in_zone(self, vector):
        u = {}
        u[0], u[1], u[2] = vector.T
        in_zone = u[self.axis] <= self.value
        return in_zone.T

    @property
    def type(self):
        return "Upper Limit"


class LowerLimit(SingleLimit):
    """Defines a zone by its (1D) upper limit

    Parameters
    ----------
    value : float
        the lower limit
    axis : int, optional
        axis to consider (0:x, 1:y, 2:z), by default 0
    target : str, optional
        the target for the zone, can be "position" or "speed", by default "position"
    action : str, optional
        the action associated to the zone.
        implemented actions = ["stop", "ignore"], default is "stop"
    tag : str, optional
        the zone tag
    in_tag : str, optional
        tag for an object inside the zone, by default None
    out_tag : str, optional
        tag for an object inside the zone, by default None

    Example
    -------
    >>> low_x = LowerLimit(value=-5, axis=0, tag="lowx")

    """

    def __init__(
        self,
        value: float,
        axis: int = 0,
        target: str = "position",
        action: str = "stop",
        tag: str = None,
        in_tag: str = None,
        out_tag: str = None,
    ):
        super(LowerLimit, self).__init__(
            value=value,
            axis=axis,
            target=target,
            action=action,
            tag=tag,
            in_tag=in_tag,
            out_tag=out_tag,
        )

    def _in_zone(self, vector):
        u = {}
        u[0], u[1], u[2] = vector.T
        in_zone = u[self.axis] >= self.value
        return in_zone.T

    @property
    def type(self):
        return "Upper Limit"


class Limits(Zone):
    """Defines a 1D segment, with min / max value

    Parameters
    ----------
    min : float
        minimum value
    max : float
        maximum value
    axis : int, optional
        axis to consider (0:x, 1:y, 2:z), by default 0
    target : str, optional
        the target for the zone, can be "position" or "speed", by default "position"
    action : str, optional
        the action associated to the zone.
        implemented actions = ["stop", "ignore"], default is "stop"
    tag : str, optional
        the zone tag
    in_tag : str, optional
        tag for an object inside the zone, by default None
    out_tag : str, optional
        tag for an object inside the zone, by default None


    Example
    --------
    >>> xlim = Limits(min=-1, max=1, axis=0, target="position", action="stop", tag="xlim")
    """

    def __init__(
        self,
        min: float,
        max: float,
        axis: int = 0,
        target: str = "position",
        action: str = "stop",
        tag: str = None,
        in_tag: str = None,
        out_tag: str = None,
    ):
        super(Limits, self).__init__(
            target=target, action=action, tag=tag, in_tag=in_tag, out_tag=out_tag
        )
        self.axis = axis
        self.min = min
        self.max = max

    # -- GETTERS & SETTERS
    @property
    def type(self):
        return "1D limits"

    @property
    def min(self) -> float:
        """float: the lower limit"""
        return self.__min

    @min.setter
    def min(self, min):
        self.__min = float(min)

    @property
    def max(self) -> float:
        """float: the upper limit"""
        return self.__max

    @max.setter
    def max(self, max):
        self.__max = float(max)

    @property
    def axis(self) -> int:
        """int: the target axis"""
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
        in_zone = (u[self.axis] >= self.min) & (u[self.axis] <= self.max)
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
        info.add_element("in_tag", self.in_tag)
        info.add_element("out_tag", self.tag)
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
