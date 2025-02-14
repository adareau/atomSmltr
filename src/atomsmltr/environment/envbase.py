# -*- coding: utf-8 -*-
"""Defines the base class for environment objects
"""

# % LOCAL IMPORTS
from ..utils.plotter import Plottable

# % ABSTRACT CLASSES


class EnvObject(Plottable):
    """A generic class for environment objects.

    For the moment, only ensures that the "tag" property is defined
    """

    def __init__(self, tag: str):
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
