# -*- coding: utf-8 -*-
"""Defines the AtomicTransition class, later embedded in an Atom() object
"""

# % IMPORTS
from abc import ABC, abstractmethod
import numpy as np
import scipy.constants as csts

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
    def __init__(self, tag: str):
        self.__tag = tag
        super().__init__()

    @property
    def tag(self):
        return self.__tag
