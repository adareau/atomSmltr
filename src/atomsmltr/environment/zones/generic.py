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


class Zone(EnvObject):

    def __init__(self, *args, **kwargs):
        super(Zone, self).__init__(*args, **kwargs)
        self.inverted = False

    # -- GETTERS & SETTERS

    @property
    def inverted(self):
        return self.__inverted

    @inverted.setter
    def inverted(self, value):
        if not isinstance(value, bool):
            raise TypeError("'inverted' should be a boolean")
        self.__inverted = value

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

    def __add__(self, object):
        collection = ZoneCollection()
        collection.add_zone(self)
        collection.__add__(object)
        return collection

    def __iadd__(self, object):
        collection = ZoneCollection()
        collection.add_zone(self)
        collection.__add__(object)
        return collection


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

    def _in_zone(self, vector):
        res_list = [zone.in_zone(vector) for zone in self.zones]
        return np.logical_and.reduce(res_list)

    # -- INFOSTRING

    def gen_infostring_obj(self):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "zone collection")
        info.add_element("tag", self.tag)
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
        collection = ZoneCollection()
        for z in self.zones:
            collection.add_zone(copy(z))
        if isinstance(object, ZoneCollection):
            for z in object.zones:
                collection.add_zone(copy(z))
            return collection
        elif isinstance(object, Zone):
            collection.add_zone(copy(object))
            return collection
        else:
            raise TypeError(
                "a ZoneCollection can only be added with a Zone or another ZoneCollection"
            )

    def __iadd__(self, object):
        if isinstance(object, ZoneCollection):
            for z in object.zones:
                self.add_zone(copy(z))
            return self
        elif isinstance(object, Zone):
            self.add_zone(copy(object))
            return self
        else:
            raise TypeError(
                "a ZoneCollection can only be added with a Zone or another ZoneCollection"
            )
