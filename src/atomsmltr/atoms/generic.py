"""
atoms
=======================

This module implements the generic ``Atom`` class, that can be used to define
a custom atom species.

Note
----
    In general, it is better to create new classes for specific
    atomic species, so that they be used by all users. Specifi atomic species
    are stored in ``atomsmltr.atoms.collection``

"""

# % IMPORTS
from abc import ABC, abstractmethod
from copy import copy

import scipy.constants as csts
import numpy as np

# % LOCAL IMPORTS
from .transitions import AtomicTransition
from ..utils.infostring import InfoString

# % ABSTRACT CLASSES


class Atom(ABC):
    """A generic class to define atomic species.

    Parameters
    ----------
    mass : float
        the atomic mass in kg
    name : str
        a name to describe the atom


    Example
    -------
    >>> from atomsmltr.atoms import Atom
    >>> from scipy import constants as csts
    >>> atom = Atom(mass=4 * csts.m_u, name="Helium")

    Note
    ----
    Specific atomic species are implemented in ``atomsmltr.atoms.collection``

    """

    def __init__(self, mass: float, name: str):
        self.__mass = mass
        self.__name = name
        self.__transitions = {}
        super().__init__()

    # - GETTERS AND SETTERS

    @property
    def mass(self) -> float:
        """float: the atom mass in kg"""
        return self.__mass

    @property
    def mass_au(self) -> float:
        """float: the atom mass in atomic units"""
        return self.__mass / csts.m_u

    @property
    def name(self) -> float:
        """str: the atom name"""
        return self.__name

    @property
    def transitions(self) -> list:
        """list: a list of the atom's transitions"""
        return self.__transitions.values()

    @property
    def trans(self) -> dict:
        """dict: a copy of the atom's transition dictionnary"""
        return copy(self.__transitions)

    # - RADIATION PRESSURE

    def get_radiation_pressure(
        self,
        transition: str,  # the transition tag
        intensity: float,  # the intensity in W/cm^2
        mag_field: float,  # the amplitude of the magnetic field
        polarization: np.ndarray,  # projection of laser polarization on (pi, sigma+, sigma-)
        detuning: float,  # laser detuning
    ) -> float:
        """Computes the radiation pressure.

        Parameters
        ----------
        transition : str
            The tag of the considered transition. The transition has to be defined in the atom's
            transition list, that can be accessed using the ``list_transitions()`` method.

        intensity: float
            The laser intensity, in W/m^2

        mag_field: float
            The *norm* of the magnetic field, in T (scalar).

        polarization: ndarray, shape (,3)
            Projection of the laser polarization on (pi, sigma+, sigma-)

        detuning: float
            The laser bare detuning, in rad/s

        Returns
        -------
        F_rad: float
            The radiation pressure in SI (scalar)

        Notes
        -------
        Yields the radiation pressure felt by an atom excited on a given transition, by
        a laser with given intensity. It takes into account the value of the magnetic field,
        the polarization of the laser and its bare detuning.

        The computation is based on the ``get_scattering_rate()`` method, implemented in the
        ``AtomicTransition`` class.

        See Also
        --------
        atomsmltr.atoms.transitions.AtomicTransition.get_scattering_rate()

        """
        # parameter check
        transition_list = self.list_transitions()
        msg = f"There is no transition with tag '{transition}'. Available transitions : {transition_list}"
        if transition not in transition_list:
            raise KeyError(msg)

        # get scattering rate
        trans = self.__transitions[transition]
        scattering_rate = trans.get_scattering_rate(
            intensity, mag_field, polarization, detuning
        )

        # convert to radiation pressure
        k = 2 * csts.pi / trans.wavelength
        F_rad = csts.hbar * k * scattering_rate

        return F_rad

    # - TRANSITIONS MANAGEMENT

    def get_transitions_copy(self) -> dict:
        """returns a copy of the transition dictionnary

        Returns
        -------
        dict
            a dict containing the atom's transitions
        """
        return copy(self.__transitions)

    def add_transition(self, transition: AtomicTransition, tag=None) -> None:
        """adds a transition to the atom transition dict.

        Parameters
        ----------
        transition : AtomicTransition
            an atomic transition (see ``atomsmltr.atoms.transitions.AtomicTransition``)
        tag : str, optional
            the tag identifying the transition in the transition dict.
            If nothing or ``None`` is given, we will use the ``.tag`` property
            of the ``AtomicTransition`` object

        See Also
        --------
        atomsmltr.atoms.transitions.AtomicTransition
        """
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
        """returns a list of the transition tags (str)

        Returns
        -------
        list of str
            list of transition tags
        """
        return list(self.__transitions.keys())

    def rm_transition(self, tag: str) -> None:
        """removes a transition for the atom's transition dict.

        Parameters
        ----------
        tag : str
            the tag of the transition to remove
        """
        del self.__transitions[tag]

    # - INFOSTRING

    def _gen_infostring_obj(self) -> InfoString:
        """generates an ``InfoString`` object.

        Returns
        -------
        InfoString
            an ``InfoString`` object

        See also
        --------
        atomsmltr.utils.infostring.InfoString
        """

        info = InfoString(title=self.name)
        info.add_section("Parameters")
        info.add_element("mass (kg)", f"{self.mass:.2e}")
        info.add_element("mass (au)", f"{self.mass_au:.2f}")

        info.add_section("Transition list")
        for trans_tag in self.list_transitions():
            info.add_element(trans_tag)

        for trans_tag, trans in self.__transitions.items():
            trans_info = trans.gen_infostring_obj()
            section_title = f"'{trans_tag}' transition"
            info.absorb_section(trans_info, "Parameters", section_title)

        return info

    def gen_infostring_obj(self) -> InfoString:
        """generates an ``InfoString`` object.

        Returns
        -------
        InfoString
            an ``InfoString`` object

        See also
        --------
        atomsmltr.utils.infostring.InfoString
        """
        return self._gen_infostring_obj()

    def gen_info_string(self, **kwargs) -> str:
        """generates an info string

        Returns
        -------
        info_string: str
            a string with information on the atom
        """
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self) -> None:
        """prints the atom infostring"""
        print(self.gen_info_string())
