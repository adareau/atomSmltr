# -*- coding: utf-8 -*-
"""Defines the generic Zones classes
"""

# % IMPORTS
from abc import abstractmethod
import numpy as np

# % LOCAL IMPORTS
from ..envbase import EnvObject
from ...utils.misc import check_position_array

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
