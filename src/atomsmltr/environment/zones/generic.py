"""
zones
=======================

Here we implement the generic ``Zone`` class, as well as a series of
``ZoneCollection`` classes that are used to combine several zones

Note
----
    the actual implementation of zones are in other modules

See Also
--------
atomsmltr.environment.zones.limits
atomsmltr.environment.zones.volumes

"""

# % IMPORTS
import numpy as np
from abc import abstractmethod
from copy import copy, deepcopy

# % LOCAL IMPORTS
from ..envbase import EnvObject
from ...utils.misc import check_position_speed_array
from ...utils.infostring import InfoString

# % ABSTRACT CLASSES

IMPLEMENTED_ACTIONS = ["stop", "ignore"]
IMPLEMENTED_TARGETS = ["position", "speed"]


# % -------------------------------
# % SIMPLE ZONES
# % -------------------------------


class Zone(EnvObject):
    """A generic Zone object

    Parameters
    ----------
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
    """

    def __init__(
        self,
        target: str = "position",
        action: str = "stop",
        tag: str = None,
        in_tag: str = None,
        out_tag: str = None,
    ):
        super(Zone, self).__init__(tag)
        self.inverted = False
        self.target = target
        self.action = action
        self.in_tag = in_tag
        self.out_tag = out_tag

    # -- GETTERS & SETTERS

    @property
    def vector(self):
        return False

    @property
    def inverted(self):
        """bool: if inverted, the zone logic is inverted"""
        return self.__inverted

    @inverted.setter
    def inverted(self, value):
        if not isinstance(value, bool):
            raise TypeError("'inverted' should be a boolean")
        self.__inverted = value

    @property
    def target(self):
        """str: the target for the zone. Can be "position" or "speed" """
        return self.__target

    @target.setter
    def target(self, value):
        if value not in IMPLEMENTED_TARGETS:
            raise ValueError(f"implemented targets are : {IMPLEMENTED_TARGETS}")
        self.__target = value

    @property
    def action(self):
        """str: the action associated with the zone. implemented actions = ["stop", "ignore"]."""
        return self.__action

    @action.setter
    def action(self, value):
        if value not in IMPLEMENTED_ACTIONS:
            raise ValueError(f"implemented actions are : {IMPLEMENTED_ACTIONS}")
        self.__action = value

    @property
    def in_tag(self) -> str:
        """str: tag for a object inside the zone"""
        return self._in_tag

    @in_tag.setter
    def in_tag(self, value: str) -> None:
        if not isinstance(value, str) and value is not None:
            raise TypeError("'in_tag' should be a string or None")
        self._in_tag = value

    @property
    def out_tag(self) -> str:
        """str: tag for a object outside the zone"""
        return self._out_tag

    @out_tag.setter
    def out_tag(self, value: str) -> None:
        if not isinstance(value, str) and value is not None:
            raise TypeError("'out_tag' should be a string or None")
        self._out_tag = value

    # -- functions

    def invert(self):
        """toggles the 'inverted' status"""
        self.__inverted = not self.__inverted

    def inverted_copy(self):
        """Returns an inverted copy of the object"""
        new_object = deepcopy(self)
        new_object.invert()
        return new_object

    # -- METHODS
    def get_value(self, vector: np.ndarray, nocheck: bool = False) -> np.ndarray:
        """Evaluates whether 'vector' is in the zone

        Parameters
        ----------
        vector : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the vectors in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        value : array of shape (1,) or (n1, n2, ..., 1)
            wheter the vector is 'in the zone'

        Notes
        -----

        ``vector`` should be an array of shape (...,3), where last axis contains
        the coordinates to evaluate.

        if the ``inverted`` property is set to true, ``get_value`` will return
        True outside the zone

        """
        vector = self._check_position_array(vector, nocheck)
        res = self._in_zone(vector)
        if self.inverted:
            res = np.logical_not(res)
        return res

    @abstractmethod
    def _in_zone(self, vector):
        """actual implementationf of 'get_value'"""

    # -- OPERATORS OVERLOADING

    def __and__(self, object):
        if isinstance(object, Zone):
            new_collection = ANDCollection()
            new_collection.add_zone(deepcopy(self))
            new_collection.add_zone(deepcopy(object))
            return new_collection
        else:
            raise TypeError("only 'Zones' objects can be combined")

    def __or__(self, object):
        if isinstance(object, Zone):
            new_collection = ORCollection()
            new_collection.add_zone(deepcopy(self))
            new_collection.add_zone(deepcopy(object))
            return new_collection
        else:
            raise TypeError("only 'Zones' objects can be combined")

    def __xor__(self, object):
        if isinstance(object, Zone):
            new_collection = XORCollection()
            new_collection.add_zone(deepcopy(self))
            new_collection.add_zone(deepcopy(object))
            return new_collection
        else:
            raise TypeError("only 'Zones' objects can be combined")


# % -------------------------------
# % ZONES COLLECTIONS
# % -------------------------------


class ZoneCollection(Zone):
    def __init__(self, *args, **kwargs):
        self.__zones = []
        super(ZoneCollection, self).__init__(*args, **kwargs)

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "Zone Collection"

    @property  # readonly
    def zones(self) -> list:
        """list: a list of the zones included in the collection"""
        return self.__zones

    def add_zone(self, zone: Zone):
        """adds a zone to the current collection

        Parameters
        ----------
        zone : Zone
            the zone to add
        """

        if not isinstance(zone, Zone):
            raise TypeError("'zone' should be a zone object")
        self.__zones.append(zone)

    def reset(self):
        """Resets the zone list"""
        self.__zones = []

    # -- INFOSTRING

    def gen_infostring_obj(self):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", self.type)
        info.add_element("tag", self.tag)
        info.add_element("in_tag", self.in_tag)
        info.add_element("out_tag", self.out_tag)
        info.add_element("target", self.target)
        info.add_element("action", self.action)
        info.add_element(f"zones", f"{[z.tag for z in self.zones]}")
        info.add_element(f"inverted", f"{self.inverted}")
        return info

    # -- PLOT

    def plot1D(self, ax=None):
        pass

    def plot2D(self, ax=None):
        pass

    def plot3D(self, ax=None):
        pass

    # -- OPERATORS OVERLOADING

    def __add__(self, object):
        # then operator acts on a new collection
        collection = self.__class__()
        for z in self.zones:
            collection.add_zone(deepcopy(z))
        return self.__add_operator__(object, collection)

    def __iadd__(self, object):
        """let's handle additions between zonecollections

        add behaves as a shorthand for "add_zones"
        we will only allow collections of same type to be added

        """
        # then operator acts on self
        # - recursive add if list
        if isinstance(object, (list, tuple)):
            for element in object:
                self.__add_operator__(element, self)
            return self
        # - otherwise
        return self.__add_operator__(object, self)

    def __add_operator__(self, object, coll):
        """a function to factor the __add__ and __iadd__ operators"""
        # case 1 > same type of zone, then we add all the zones
        if object:
            if isinstance(object, self.__class__):
                for z in object.zones:
                    new_zone = deepcopy(z)
                    if object.inverted:
                        new_zone.invert()
                    coll.add_zone(new_zone)
                return coll
            # case 2 > it is a zone, not a collection
            elif isinstance(object, Zone) and not isinstance(object, ZoneCollection):
                coll.add_zone(deepcopy(object))
                return coll
            else:
                raise TypeError(
                    "a ZoneCollection can only be added with a Zone or another ZoneCollection of same type"
                )


class ANDCollection(ZoneCollection):

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "AND Zone Collection"

    def _in_zone(self, vector):
        res_list = [zone.get_value(vector) for zone in self.zones]
        return np.logical_and.reduce(res_list)


class ORCollection(ZoneCollection):

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "OR Zone Collection"

    def _in_zone(self, vector):
        res_list = [zone.get_value(vector) for zone in self.zones]
        return np.logical_or.reduce(res_list)


class XORCollection(ZoneCollection):

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "XOR Zone Collection"

    def _in_zone(self, vector):
        res_list = [zone.get_value(vector) for zone in self.zones]
        return np.logical_xor.reduce(res_list)


# % -------------------------------
# % SUPER ZONE
# % -------------------------------

IMPLEMENTED_LOGIC = ["OR", "XOR", "AND"]


class SuperZone(ZoneCollection):
    """SuperZone is a zone collection for position/speed vectors

    Parameters
    ----------
    zones : list, optional
        list of zones to add at object creation, by default []
    logic : str, optional
        the logic of zone combination. Can be "OR", "AND", "XOR", by default "AND"
    action : str, optional
        the action to trigger
        implemented actions = ["stop", "ignore"], default is "stop"
    tag : str, optional
        the tag of the zone, by default None
    in_tag : str, optional
        tag for an object inside the zone, by default None
    out_tag : str, optional
        tag for an object inside the zone, by default None
    """

    def __init__(
        self,
        zones: list = [],
        logic: str = "AND",
        action: str = "stop",
        tag: str = None,
        in_tag: str = None,
        out_tag: str = None,
    ):

        super(SuperZone, self).__init__(
            action=action, tag=tag, in_tag=in_tag, out_tag=out_tag
        )
        self.logic = logic
        self.__iadd__(zones)

    # -- PROPERTIES

    @property
    def type(self):
        return "Super Zone collection"

    # - target is irrelevant
    @property
    def target(self):
        """not used in this case"""
        return None

    @target.setter
    def target(self, value):
        pass

    # - logic
    @property
    def logic(self):
        """str: the logic for the SuperZone combination ("OR", "XOR", "AND")"""
        return self.__logic

    @logic.setter
    def logic(self, value: str):
        if value not in IMPLEMENTED_LOGIC:
            raise ValueError(f"'logic' should be in {IMPLEMENTED_LOGIC}")
        self.__logic = value
        match value:
            case "OR":
                self.__logical_op = np.logical_or
            case "AND":
                self.__logical_op = np.logical_and
            case "XOR":
                self.__logical_op = np.logical_xor

    # -- METHODS

    def add_zones(self, zones: Zone | list):
        self.__iadd__(zones)

    def get_value(self, vector: np.ndarray, nocheck: bool = False) -> np.ndarray:
        """Evaluates whether 'vector' is in the zone

        Parameters
        ----------
        vector : array of shape (6,) or (n1, n2, ..., 6)
            cartesian coordinates of the vectors in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        value : array of shape (1,) or (n1, n2, ..., 1)
            whether the vector is 'in the zone'

        Notes
        -----

        This is a ``SuperZone`` object, so it acts on **position-speed** vectors
        of dimensions 6 !

        ``vector`` should be an array of shape (6,) or (n1, n2, .., 6), where last axis contains
        the coordinates (position & speed) to evaluate.

        In all cases, the last dimension contains cordinates (x, y, z, vx, vy, vz),

        if the ``inverted`` property is set to true, ``get_value`` will return
        True outside the zone

        """
        vector = check_position_speed_array(vector, nocheck)
        res = self._in_zone(vector)
        if self.inverted:
            res = np.logical_not(res)
        return res

    def _in_zone(self, vector: np.ndarray) -> np.ndarray:
        """The actual implementation of the ``in_zone`` method"""
        # -- separate speeds & positions
        x, y, z, vx, vy, vz = vector.T
        position = np.array([x, y, z]).T
        speed = np.array([vx, vy, vz]).T

        # -- separate zones according to target
        speed_zones = []
        position_zones = []
        for zone in self.zones:
            if zone.target == "speed":
                speed_zones.append(zone)
            elif zone.target == "position":
                position_zones.append(zone)

        # -- evaluate
        res_list = [zone.get_value(position) for zone in position_zones]
        res_list += [zone.get_value(speed) for zone in speed_zones]

        if res_list:
            res = self.__logical_op.reduce(res_list)
        else:
            res = x.T == x.T
        return res

    def gen_infostring_obj(self):
        info = super().gen_infostring_obj()
        info.add_element("logic", self.logic)
        info.rm_element("target")
        return info
