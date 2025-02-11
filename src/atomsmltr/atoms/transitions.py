# -*- coding: utf-8 -*-
"""Defines the AtomicTransition class, later embedded in an Atom() object
"""

# % IMPORTS
from abc import ABC, abstractmethod
import scipy.constants as csts

# % LOCAL IMPORTS

# % ABSTRACT CLASSES


class AtomicTransition(ABC):
    def __init__(self, tag: str):
        self.__tag = tag
        super().__init__()

    @property
    def tag(self):
        return self.__tag
