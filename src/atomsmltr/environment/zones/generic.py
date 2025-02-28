# -*- coding: utf-8 -*-
"""Defines the generic Zones classes
"""

# % IMPORTS
import numpy as np
from abc import abstractmethod
from copy import copy, deepcopy

# % LOCAL IMPORTS
from ..envbase import EnvObject
from ...utils.misc import check_position_array
from ...utils.infostring import InfoString

# % ABSTRACT CLASSES

IMPLEMENTED_ACTIONS = ["stop"]
IMPLEMENTED_TARGETS = ["position", "speed"]


class Zone(EnvObject):

    def __init__(self, target="position", action="stop", *args, **kwargs):
        super(Zone, self).__init__(*args, **kwargs)
        self.inverted = False
        self.target = target
        self.action = action

    # -- GETTERS & SETTERS

    @property
    def inverted(self):
        return self.__inverted

    @inverted.setter
    def inverted(self, value):
        if not isinstance(value, bool):
            raise TypeError("'inverted' should be a boolean")
        self.__inverted = value

    @property
    def target(self):
        return self.__target

    @target.setter
    def target(self, value):
        if value not in IMPLEMENTED_TARGETS:
            raise ValueError(f"implemented targets are : {IMPLEMENTED_TARGETS}")
        self.__target = value

    @property
    def action(self):
        return self.__action

    @action.setter
    def action(self, value):
        if value not in IMPLEMENTED_ACTIONS:
            raise ValueError(f"implemented actions are : {IMPLEMENTED_ACTIONS}")
        self.__action = value

    # -- functions

    def invert(self):
        """toggles the 'inverted' status"""
        self.__inverted = not self.__inverted

    def inverted_copy(self):
        new_object = deepcopy(self)
        new_object.invert()
        return new_object

    # -- METHODS
    def in_zone(self, vector, nocheck=False):
        """Evaluates whether 'vector' is in the zone

        vector should be an array of shape (...,3), where last axis contains
        the coordinates to evaluate.

        if the 'inverted' property is set to true, in_zone will return
        True outside the zone

        Args:
            vector (array, shape (...,3)): coordinates to evaluate

        Returns:
            res (array of booleans, shape (...,1)): whether coordinates are in the zone
        """
        vector = check_position_array(vector)
        res = self._in_zone(vector)
        if self.inverted:
            res = np.logical_not(res)
        return res

    @abstractmethod
    def _in_zone(self, vector):
        """actual implementationf of 'in_zone'"""

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


class ZoneCollection(Zone):
    def __init__(self, *args, **kwargs):
        self.__zones = []
        super(ZoneCollection, self).__init__(*args, **kwargs)

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "Zone Collection"

    @property  # readonly
    def zones(self):
        return self.__zones

    def add_zone(self, zone):
        if not isinstance(zone, Zone):
            raise TypeError("'zone' should be a zone object")
        self.__zones.append(zone)

    def reset(self):
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
        return self.__add_operator__(object, self)

    def __add_operator__(self, object, coll):
        """a function to factor the __add__ and __iadd__ operators"""
        # case 1 > same type of zone, then we add all the zones
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
        res_list = [zone.in_zone(vector) for zone in self.zones]
        return np.logical_and.reduce(res_list)


class ORCollection(ZoneCollection):

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "OR Zone Collection"

    def _in_zone(self, vector):
        res_list = [zone.in_zone(vector) for zone in self.zones]
        return np.logical_or.reduce(res_list)


class XORCollection(ZoneCollection):

    # -- METHODS AND PROPERTIES
    @property
    def type(self):
        return "XOR Zone Collection"

    def _in_zone(self, vector):
        res_list = [zone.in_zone(vector) for zone in self.zones]
        return np.logical_xor.reduce(res_list)
