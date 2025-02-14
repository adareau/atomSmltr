# -*- coding: utf-8 -*-
"""Defines the Configuration classes,
to make a consistent configuration for the simulator
"""

# % LOCAL IMPORTS
from ..environment import AbstractLaserBeam, MagneticField
from ..atoms import Atom


# % DEFINE THE CLASS


class Configuration(object):
    def __init__(self):
        self.__lasers = {}
        self.__zones = {}
        self.__magfields = {}
        self.__atom = None
