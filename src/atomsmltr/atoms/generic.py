# -*- coding: utf-8 -*-
"""Generic atom classes ; specific implementations to be defined in separate files
"""

# % IMPORTS
from abc import ABC, abstractmethod
import scipy.constants as csts

# % LOCAL IMPORTS
from .transitions import AtomicTransition

# % ABSTRACT CLASSES


class Atom(ABC):
    def __init__(self, mass: float, name: str):
        self.__mass = mass
        self.__name = name
        self.__transitions = {}
        super().__init__()

    @property
    def mass(self) -> float:
        """The atom mass in kg"""
        return self.__mass

    @mass.setter
    def mass(self, value: float) -> None:
        """we do not want to mess with the mass on the fly..."""
        msg = "To avoid mistakes, you cannot change the mass once the atom object is instatiated"
        raise Warning(msg)

    @property
    def mass_au(self) -> float:
        """The atom mass in atomic units"""
        return self.__mass / csts.m_u

    @property
    def name(self) -> float:
        return self.__name

    def add_transition(self, transition: AtomicTransition, tag=None) -> None:
        """Adds a transition (`AtomicTransition` object) to the atom transitions collection"""
        # -- parse input
        # check transition type
        if not isinstance(transition, AtomicTransition):
            raise TypeError("`transition` should be an `AtomicTransition`object.")
        # if tag is non, use the atomic transition builtin tag
        tag = transition.tag if tag is None else tag
        # check that not in dictionnary
        msg = f"There is already a transition with the tag {tag} in the atom's collection."
        assert tag not in self.__transitions, msg
        # TODO: other checks ? warning if already same wavelength ?

        # -- add to collection
        self.__transitions.update({tag: transition})

    def list_transitions(self) -> list:
        """returns a list of all transitions tags"""
        return list(self.__transitions.keys())

    def rm_transition(self, tag: str):
        """removes a transition from the list"""
        del self.__transitions[tag]
