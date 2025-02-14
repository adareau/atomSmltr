# -*- coding: utf-8 -*-
"""Defines the Configuration classes,
to make a consistent configuration for the simulator
"""

# % LOCAL IMPORTS
from ..environment import AbstractLaserBeam, MagneticField
from ..atoms import Atom


# % DEFINE THE CLASS


class Configuration(object):
    def __init__(self, object_list=[], atom=None):
        # - initialize collections
        self.__lasers = {}
        self.__zones = {}
        self.__magfields = {}
        self.__atomlight = {}
        self.__atom = None

        # - init atom
        if atom is not None:
            self.atom = atom

    # -- COLLECTION HANDLING METHOD

    # -- GETTERS & SETTERS
    @property
    def atom(self) -> Atom:
        return self.__atom

    @atom.setter
    def atom(self, atom: Atom):
        if not isinstance(atom, Atom):
            raise TypeError("'atom' should be an atom")
        self.__atom = atom
        # TODO here, check sanity of __atomlight ; warning
