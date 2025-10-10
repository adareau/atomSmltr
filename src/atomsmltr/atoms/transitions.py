"""
transitions
=======================

This module implements the ``AtomicTransition`` class, that is meant to be embedded
in the ``Atom`` class.

Example
-------

.. code-block:: python

    from atomsmltr.atoms import Atom
    from atomsmltr.atoms.transitions import DummyTransition
    from scipy import constants as csts
    atom = Atom(mass=4 * csts.m_u, name="Helium")
    transition = DummyTransition("transition", Gamma=1, wavelength=1)
    atom.add_transition(transition, tag="transition")

See Also
--------
atomsmltr.atoms.generic.Atom
"""

# % IMPORTS
from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as csts

# % LOCAL IMPORTS
from ..utils.infostring import InfoString

# % PHYSICS DEFINITIONS
"""In the following, we will define transitions using two parameters:
    > the wavelength in vacuum `lbda`
    > the natural linewidth `Gamma`

    All other parameters are derived from that
 """


def _w0(lbda: float) -> float:
    """returns the pulsation, in rad/s

    Parameters
    ----------
    lbda : float
        wavelength (m)

    Returns
    -------
    w0 : float
        pulsation (rad/s)
    """

    w0 = 2 * np.pi * csts.c / lbda
    return w0


def _Isat(lbda: float, Gamma: float) -> float:
    """Returns the saturation intensity, in W/m^2

    Parameters
    ----------
        lbda : float
            vacuum wavelength (in meters)
        Gamma : float
            natural linewidth (in rad/s)

    Returns
    -------
        Isat : float
            saturation intensity (in W/m^2)
    """
    w0 = _w0(lbda)
    Isat = csts.hbar * Gamma * w0**3 / 12 / np.pi / csts.c**2
    return Isat


def _Isat_mW_per_cm2(lbda: float, Gamma: float) -> float:
    """Returns the saturation intensity, in mW/cm^2

    Parameters
    ----------
        lbda : float
            vacuum wavelength (in meters)
        Gamma : float
            natural linewidth (in rad/s)

    Returns
    -------
        Isat: float
            saturation intensity (in mW/cm^2)
    """
    Isat_SI = _Isat(lbda, Gamma)
    Isat = Isat_SI * 1e3 / (1e2) ** 2
    return Isat


def _OmegaR(lbda: float, Gamma: float, I: float) -> float:
    """Returns the bare Rabi frequency for a two level system

    Parameters
    ----------
        lbda : float
            vacuum wavelength (in meters)
        Gamma : float
            natural linewidth (in rad/s)
        I : float
            saturation intensity (in W/m^2)

    Returns
    -------
        OmegaR : float
            the bare Rabi frequency (in rad/s)
    """
    Isat = _Isat(lbda, Gamma)
    OmegaR = Gamma * np.sqrt(I / 2 / Isat)
    return OmegaR


def _sat_param(lbda: float, Gamma: float, I: float, detuning: float) -> float:
    """Returns the saturation parameter for a two-level system.

    Beware, detuning is 2pi * (f_laser - f_transition)

    Parameters
    ----------
        lbda : float
            vacuum wavelength (in meters)
        Gamma : float
            natural linewidth (in rad/s)
        I : float
            saturation intensity (in W/m^2)
        detuning float
            laser detuning (in rad/s)

    Returns
    -------
        s : float
            the saturation parameter
    """
    Isat = _Isat(lbda, Gamma)
    s = (I / Isat) * (Gamma**2 / 4) / (detuning**2 + Gamma**2 / 4)
    return s


def _scattering_rate(lbda: float, Gamma: float, I: float, detuning: float) -> float:
    """Returns the scattering rate for a two-level system

    Beware, detuning is 2pi * (f_laser - f_transition)

    Parameters
    ----------
        lbda : float
            vacuum wavelength (in meters)
        Gamma : float
            natural linewidth (in rad/s)
        I : float
            saturation intensity (in W/m^2)
        detuning float
            laser detuning (in rad/s)

    Returns
    -------
        gamma_scatt : float
            the scattering rate (in /s)
    """
    s = _sat_param(lbda, Gamma, I, detuning)
    gamma_scatt = 0.5 * Gamma * s / (1 + s)
    return gamma_scatt


def _Doppler_temperature(Gamma: float, delta: float) -> float:
    """Returns the Doppler temperature for a transition


    Parameters
    ----------
        Gamma : float
            natural linewidth (in rad/s)

        delta : float
            laser detuning (in Hz)

    Returns
    -------
        T_Doppler : float
            Doppler temperature in K
    """
    T_Dopp = csts.hbar / 2 / csts.k * (delta**2 + Gamma**2 / 4) / np.abs(delta)
    return T_Dopp


# % ABSTRACT CLASSES


class AtomicTransition(ABC):
    """A generic (abstract) class to define electronic transitions in atoms

    Parameters
    ----------
    wavelength : float
        the vacuum wavelength (in m)
    Gamma : float
        the transition natural linewidth (in rad/s)
    tag : str
        a tag identifying the transition

    Notes
    ------
    This is an abstract class that cannot be used in simulations. For that purpose,
    see actual implementations of transitions.

    See also
    --------
    DummyTransition
    J0J1Transition
    """

    def __init__(self, wavelength: float, Gamma: float, tag: str):
        self.__tag = tag
        self.__wavelength = wavelength
        self.__Gamma = Gamma
        super().__init__()

    # -- READ-ONLY PROPERTIES
    # only with getters, no setters

    @property
    def tag(self):
        """str: transition identifier"""
        return self.__tag

    @property
    def wavelength(self):
        """float: transition vacuum wavelength (m)"""
        return self.__wavelength

    @property
    def Gamma(self):
        """float: transition natural linewidth (rad/s)"""
        return self.__Gamma

    @property
    def Isat(self):
        """float: transition saturation intensity (W/m^2)"""
        return _Isat(self.wavelength, self.Gamma)

    @property
    def Isat_mW_per_cm2(self):
        """float: transition saturation intensity (mW/cm^2)"""
        return _Isat_mW_per_cm2(self.wavelength, self.Gamma)

    @property
    def k(self):
        """float: transition wavenumber k = 2π / λ (m^-1)"""
        return 2 * np.pi / self.wavelength

    @property
    def Doppler_temperature(self):
        """float: Doppler temperature TD = hbar k / 2 / kB (K)"""
        return _Doppler_temperature(self.Gamma, -0.5 * self.Gamma)

    # -- METHODS

    def get_Doppler_temperature(self, detuning: float) -> float:
        """Returns the Doppler temperature for a given laser detuning

        Parameters
        ----------
        detuning : float
            laser detuning, in Hz

        Returns
        -------
        float
            the Doppler temperature, in K
        """
        return _Doppler_temperature(self.Gamma, detuning)

    def get_saturation_parameter(self, intensity: float, detuning: float) -> float:
        """Returns the saturation parameter (for a two-level system)

        Parameters
        ----------
            intensity : float
                laser intensity in W/m^2

            detuning : float
                laser detuning in rad/s

        Returns
        -------
            s : float
                the saturation parameter
        """
        s = _sat_param(self.wavelength, self.Gamma, intensity, detuning)
        return s

    @abstractmethod
    def get_scattering_rate(
        self,
        intensity: float,
        mag_field: float,
        polarization: np.ndarray,
        detuning: float,
    ):
        """Returns the scattering rate for a given laser / mag. field configuration

        Parameters
        ----------
        intensity : float
            laser intensity (W/m^)
        mag_field : float
            (scalar) magnetic field amplitude (T)
        polarization : ndarray, shape (,3)
            projection of the laser polarization on (pi, sigma+, sigma-)
        detuning : float
            laser detuning (rad/s)

        Returns
        -------
        scattering rate : float
            the transition scattering rate
        """
        pass

    @abstractmethod
    def get_resonant_speed(
        self,
        mag_field: float,
        polarization: str,
        detuning: float,
    ):
        """Returns the resonant speed for a given mag. field configuration

        Parameters
        ----------
        mag_field : float
            (scalar) magnetic field amplitude (T)
        polarization : str
            laser polarization : "pi", "sigma+" or "sigma-"
        detuning : float
            laser detuning (rad/s)

        Returns
        -------
        speed : float
            resonant speed (m/s)
        """
        pass

    def _gen_infostring_obj(self):
        """Generates an info string object"""
        info = InfoString(title=self.tag)
        info.add_section("Parameters")
        info.add_element("λ", f"{self.wavelength * 1e9:.2f} nm")
        info.add_element("Γ", f"2π × {self.Gamma / 2 / np.pi:.2e} Hz")
        info.add_element("Isat", f"{self.Isat_mW_per_cm2:.2f} mw/cm²")
        info.add_element("Doppler temp.", f"{self.Doppler_temperature:.2e} K")
        return info

    def gen_infostring_obj(self):
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

    def gen_info_string(self, **kwargs):
        """generates an info string

        Returns
        -------
        info_string: str
            a string with information on the atom
        """
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        """prints the atom infostring"""
        print(self.gen_info_string())


class DummyTransition(AtomicTransition):
    """Dummy class, only for testing purposes"""

    def get_scattering_rate(
        self, intensity: float, mag_field: float, polarization: str, detuning: float
    ):
        """Returns the scattering rate for the DummyTransition model

        This is just a two-level atom with no dependence on mag. field or polarization

        Parameters
        ----------
        intensity : float
            laser intensity (W/m^)
        mag_field : float
            (scalar) magnetic field amplitude (T)
        polarization : ndarray, shape (,3)
            projection of the laser polarization on (pi, sigma+, sigma-)
        detuning : float
            laser detuning (rad/s)

        Returns
        -------
        scattering rate : float
            the transition scattering rate
        """
        rate = _scattering_rate(self.__wavelength, self.__Gamma, intensity, detuning)
        return rate

    def get_resonant_speed(
        self,
        mag_field: float,  # the amplitude of the magnetic field
        polarization: str,  # "pi", "sigma+", "sigma-"
        detuning: float,  # laser detuning
    ):
        """Returns the resonant speed for the DummyTransition model

        This actually always return zero...

        Parameters
        ----------
        mag_field : float
            (scalar) magnetic field amplitude (T)
        polarization : str
            laser polarization : "pi", "sigma+" or "sigma-"
        detuning : float
            laser detuning (rad/s)

        Returns
        -------
        speed : float
            resonant speed (m/s)
        """
        return 0


# % REAL IMPLEMENTATIONS


class J0J1Transition(AtomicTransition):
    """A class to handle J=0 -> J=1 transitions

    Parameters
    ----------
    wavelength : float
        the vacuum wavelength (in m)
    Gamma : float
        the transition natural linewidth (in rad/s)
    lande_factor : float
        the lande g-factor for the transition
    tag : str
        a tag identifying the transition
    """

    def __init__(self, wavelength: float, Gamma: float, lande_factor: float, tag: str):
        self.__lande_factor = lande_factor
        super().__init__(wavelength=wavelength, Gamma=Gamma, tag=tag)

    @property
    def lande_factor(self):
        """float: the lande g-factor for the transition"""
        return self.__lande_factor

    def gen_infostring_obj(self):
        info = self._gen_infostring_obj()
        info.add_element("lande factor g", f"{self.lande_factor}")
        return info

    def get_scattering_rate(
        self,
        intensity: float,  # the intensity in W/cm^2
        mag_field: float,  # the amplitude of the magnetic field
        polarization: list,  # projection (squared) of laser polarization on (pi, sigma+, sigma-)
        detuning: float,  # laser detuning (in rad/s !!!!!!)
    ):
        # -- get projections
        # TODO : checks here
        polarization = np.asanyarray(polarization)
        proj_pi, proj_sigm_plus, proj_sigm_minus = polarization.T
        proj_pi = proj_pi.T
        proj_sigm_minus = proj_sigm_minus.T
        proj_sigm_plus = proj_sigm_plus.T

        # -- Zeeman effect
        # NB : detuning is 2 * pi * (f_laser - f_atom)
        # constants
        mu_B = csts.physical_constants["Bohr magneton"][0]
        mu = self.lande_factor * mu_B / csts.hbar

        # compute detuning
        det_pi = detuning
        det_sigm_minus = detuning + mu * mag_field
        det_sigm_plus = detuning - mu * mag_field

        # -- Compute scattering rate
        # NB : we assume that the transition is not saturated and we can sum
        # all the polarization components
        scatt_pi = _scattering_rate(
            self.wavelength, self.Gamma, intensity * proj_pi, det_pi
        )
        scatt_sigm_minus = _scattering_rate(
            self.wavelength, self.Gamma, intensity * proj_sigm_minus, det_sigm_minus
        )
        scatt_sigm_plus = _scattering_rate(
            self.wavelength, self.Gamma, intensity * proj_sigm_plus, det_sigm_plus
        )

        # sum
        scatt_total = scatt_pi + scatt_sigm_minus + scatt_sigm_plus

        return scatt_total

    def get_resonant_speed(
        self,
        mag_field: float,  # the amplitude of the magnetic field
        polarization: str,  # "pi", "sigma+", "sigma-"
        detuning: float,  # laser detuning
    ):
        # -- check input
        polar_list = ["pi", "sigma+", "sigma-"]
        msg = f"'polarization' should be in {polar_list}"
        assert polarization in polar_list, msg

        # -- factor
        mu_B = csts.physical_constants["Bohr magneton"][0]
        mu = self.lande_factor * mu_B / csts.hbar
        prefact = {"pi": 0, "sigma+": 1, "sigma-": -1}

        v_res = (detuning - mu * prefact[polarization] * mag_field) / self.k
        return v_res
