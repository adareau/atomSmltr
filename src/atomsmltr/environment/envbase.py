# -*- coding: utf-8 -*-
"""Defines the base class for environment objects
"""

# % IMPORTS
from abc import abstractmethod

# % LOCAL IMPORTS
from ..utils.plotter import Plottable
from ..utils.misc import random_word

# % ABSTRACT CLASSES


class EnvObject(Plottable):
    """A generic class for environment objects.

    For the moment, only ensures that the "tag" property is defined
    """

    def __init__(self, tag: str = None):
        # init tag with random word if None
        if tag is None:
            tag = random_word()
        self.tag = tag
        super(EnvObject, self).__init__()

    @property
    def tag(self) -> str:
        return self._tag

    @tag.setter
    def tag(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("'tag' should be a string")
        self._tag = value

    # -- INFO STRING / OBJECT MANAGEMENT
    @abstractmethod
    def gen_infostring_obj(self):
        """should return the infostring object"""
        pass

    def gen_info_string(self, **kwargs):
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        print(self.gen_info_string())

    @property
    @abstractmethod
    def type():
        """Type has to be defined in the concrete class"""
        pass
