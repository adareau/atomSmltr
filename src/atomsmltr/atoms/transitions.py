# -*- coding: utf-8 -*-
"""Defines the AtomicTransition class, later embedded in an Atom() object
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
    """Returns the pulsation, in rad.s^-1

    Args:
        lbda (float): wavelength (in meters)

    Returns:
        w0 (float): pulsation (in rad/s)
    """
    w0 = 2 * np.pi * csts.c / lbda
    return w0


def _Isat(lbda: float, Gamma: float) -> float:
    """Returns the saturation intensity, in W/m^2

    Args:
        lbda (float): vacuum wavelength (in meters)
        Gamma (float): natural linewidth (in rad/s)

    Returns:
        Isat (float): saturation intensity (in W/m^2)
    """
    w0 = _w0(lbda)
    Isat = csts.hbar * Gamma * w0**3 / 12 / np.pi / csts.c**2
    return Isat


def _Isat_mW_per_cm2(lbda: float, Gamma: float) -> float:
    """Returns the saturation intensity, in mW/cm^2

    Args:
        lbda (float): vacuum wavelength (in meters)
        Gamma (float): natural linewidth (in rad/s)

    Returns:
        Isat (float): saturation intensity (in mW/cm^2)
    """
    Isat_SI = _Isat(lbda, Gamma)
    Isat = Isat_SI * 1e3 / (1e2) ** 2
    return Isat


def _OmegaR(lbda: float, Gamma: float, I: float) -> float:
    """Returns the bare Rabi frequency for a two level system

    Args:
        lbda (float): vacuum wavelength (in meters)
        Gamma (float): natural linewidth (in rad/s)
        I (float): saturation intensitu (in W/m^2)

    Returns:
        OmegaR (float): the bare Rabi frequency (in rad/s)
    """
    Isat = _Isat(lbda, Gamma)
    OmegaR = Gamma * np.sqrt(I / 2 / Isat)
    return OmegaR


def _sat_param(lbda: float, Gamma: float, I: float, detuning: float) -> float:
    """Returns the saturation parameter for a two-level system.

    Beware, detuning is 2pi * (f_laser - f_transition)

    Args:
        lbda (float): vacuum wavelength (in meters)
        Gamma (float): natural linewidth (in rad/s)
        I (float): saturation intensitu (in W/m^2)
        detuning (float): laser detuning (in rad/s) (!!!)

    Returns:
        s (float): the saturation parameter
    """
    Isat = _Isat(lbda, Gamma)
    s = (I / Isat) * (Gamma**2 / 4) / (detuning**2 + Gamma**2 / 4)
    return s


def _scattering_rate(lbda: float, Gamma: float, I: float, detuning: float) -> float:
    """Returns the scattering rate for a two-level system

    Beware, detuning is 2pi * (f_laser - f_transition)

    Args:
        lbda (float): vacuum wavelength (in meters)
        Gamma (float): natural linewidth (in rad/s)
        I (float): saturation intensitu (in W/m^2)
        detuning (float): laser detuning (in rad/s) (!!!)

    Returns:
        gamma_scatt (float): the scattering rate (in /s)
    """
    s = _sat_param(lbda, Gamma, I, detuning)
    gamma_scatt = 0.5 * Gamma * s / (1 + s)
    return gamma_scatt


# % ABSTRACT CLASSES


class AtomicTransition(ABC):
    def __init__(self, tag: str, Gamma: float, wavelength: float):
        self.__tag = tag
        self.__wavelength = wavelength
        self.__Gamma = Gamma
        super().__init__()

    # -- LOCKED PROPERTIES
    # only with getters, no setters

    @property
    def tag(self):
        return self.__tag

    @property
    def wavelength(self):
        return self.__wavelength

    @property
    def Gamma(self):
        return self.__Gamma

    @property
    def Isat(self):
        return _Isat(self.wavelength, self.Gamma)

    @property
    def Isat_mW_per_cm2(self):
        return _Isat_mW_per_cm2(self.wavelength, self.Gamma)

    @property
    def k(self):
        return 2 * np.pi / self.wavelength

    # -- METHODS

    def get_saturation_parameter(self, intensity: float) -> float:
        """Returns the saturation parameter (for a two-level system)

        Args:
            intensity (float): laser intensity in W/m^2

        Returns:
            s (float): the saturation parameter
        """
        s = _sat_param(self.wavelength, self.Gamma, intensity)
        return s

    @abstractmethod
    def get_scattering_rate(
        self,
        intensity: float,  # the intensity in W/cm^2
        mag_field: float,  # the amplitude of the magnetic field
        polarization: list,  # projection of laser polarization on (pi, sigma+, sigma-)
        detuning: float,  # laser detuning
    ):
        """To be defined for each implementation
        NOTE: the Doppler effect will be handled at the Atom() object level
              so it will be passed to this function in a "transparent" manner.
        """
        pass

    @abstractmethod
    def get_resonant_speed(
        self,
        mag_field: float,  # the amplitude of the magnetic field
        polarization: str,  # "pi", "sigma+", "sigma-"
        detuning: float,  # laser detuning
    ):
        """To be defined for each implementation"""
        pass

    def _gen_infostring_obj(self):
        """Generates an info string object"""
        info = InfoString(title=self.tag)
        info.add_section("Parameters")
        info.add_element("λ", f"{self.wavelength * 1e9:.2f} nm")
        info.add_element("Γ", f"2π × {self.Gamma / 2 / np.pi:.2e} Hz")
        info.add_element("Isat", f"{self.Isat_mW_per_cm2:.2f} mw/cm²")
        return info

    def gen_infostring_obj(self):
        return self._gen_infostring_obj()

    def gen_info_string(self, **kwargs):
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        print(self.gen_info_string())


class DummyTransition(AtomicTransition):
    """Dummy class, only for testing purposes"""

    def get_scattering_rate(self, intensity, mag_field, polarization, detuning):
        rate = _scattering_rate(self.__wavelength, self.__Gamma, intensity, detuning)
        return rate

    def get_resonant_speed(
        self,
        mag_field: float,  # the amplitude of the magnetic field
        polarization: str,  # "pi", "sigma+", "sigma-"
        detuning: float,  # laser detuning
    ):
        return 0


# % REAL IMPLEMENTATIONS


class J0J1Transition(AtomicTransition):
    """A common class for simple J=0 -> J=1 transitions"""

    def __init__(self, lande_factor: float, *args, **kwargs):
        self.__lande_factor = lande_factor
        super().__init__(*args, **kwargs)

    @property
    def lande_factor(self):
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
        assert (
            np.asanyarray(polarization).size == 3
        ), "`polarization` should be a list/array of size 3"
        assert np.allclose(
            np.sum(polarization), 1
        ), "the sum of all polarization amplitudes should be one"

        proj_pi, proj_sigm_plus, proj_sigm_minus = polarization

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
