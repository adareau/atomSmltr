# -*- coding: utf-8 -*-
"""Defines the laser beam classes
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from enum import Enum, auto

# % GLOBAL DEFINITIONS

DIRECTION_TYPES = ["vector", "thetaphi"]  # allowed values for `direction_type``


# % TOOL FUNCTIONS


def _intensity_gauss(
    r: float, z: float, w0: float, P: float, wavelength: float
) -> float:
    """Computes intensity for a Gaussian beam of waist w0 and power P0 at position
    (r, z), in cynlindrical coordinates. The beam is propagating along z, and the waist
    is located at r = z = 0. Lengths should be given in meters, and powers in watts.
    Intensity is returned in W/m^2

    Args:
        r (float): radial coordinate (distance to beam axis) in _meters_
        z (float): axial coordinate (distance to beam waist) in _meters_
        w0 (float): Gaussian beam waist radius (1/e^2) in _meters_
        P (float): laser power in _Watts_
        wavelength (float): laser wavelength in _meters_

    Returns:
        intensity (float): laser intensity in _W/m^2_
    """

    zR = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + z**2 / zR**2)
    I0 = 2 * P / np.pi / w0**2
    intensity = I0 * (w0 / wz) ** 2 * np.exp(-2 * (r**2) / wz**2)
    return intensity


# % ABSTRACT CLASSES


class AbstractLaserBeam(ABC):
    """docstring for AbstractLaserBeam."""

    def __init__(
        self,
        wavelength: float,
        waist: float,
        power: float,
        waist_position: npt.ArrayLike,
        direction: npt.ArrayLike,
        direction_type: str = "vector",
    ):
        self.wavelength = wavelength
        self.waist = waist
        self.power = power
        self.waist_position = waist_position
        # /!\ direction_type has to be defined BEFORE direction !!
        self.direction_type = direction_type
        self.direction = direction

        super().__init__()

    # -- class properties setters & getters
    # - wavelength
    @property
    def wavelength(self) -> float:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        self._positive_float_check("wavelength", value)
        if value > 3e-6 or value < 100e-9:
            raise Warning(
                "Value given for wavelength is outside the 100nm-3µm range, which is rather strange. Check that you have given the wavelength value in _meters_"
            )
        self._wavelength = float(value)

    # - waist
    @property
    def waist(self) -> float:
        return self._waist

    @waist.setter
    def waist(self, value: float) -> None:
        self._positive_float_check("waist", value)
        self._waist = float(value)

    # - power
    @property
    def power(self) -> float:
        return self._power

    @power.setter
    def power(self, value: float) -> None:
        self._positive_float_check("power", value)
        self._power = float(value)

    # - waist position
    @property
    def waist_position(self) -> npt.ArrayLike:
        return self._waist_position

    @waist_position.setter
    def waist_position(self, value: npt.ArrayLike) -> None:
        value = np.asanyarray(value)
        if value.size != 3:
            raise ValueError("'waist_position' should be an array-like of size 3")
        self._waist_position = value

    # - direction_type
    @property
    def direction_type(self) -> str:
        return self._direction_type

    @direction_type.setter
    def direction_type(self, value: str) -> None:
        if value not in DIRECTION_TYPES:
            raise ValueError(f"'direction_type' should be in {DIRECTION_TYPES}")
        self._direction_type = value

    # - direction
    @property
    def direction(self) -> npt.ArrayLike:
        return self._direction

    @direction.setter
    def direction(self, value: npt.ArrayLike) -> None:
        # convert to array
        value = np.asanyarray(value)

        # check that the size is OK
        errormsg = "When 'direction_type' is set to '{direction_type}', 'direction' should be an array of size {size}"
        if self.direction_type == "vector" and value.size != 3:
            raise ValueError(errormsg.format(direction_type="vector", size=3))
        elif self.direction_type == "thetaphi" and value.size != 2:
            raise ValueError(errormsg.format(direction_type="thetaphi", size=2))

        # compute unit vector
        if self.direction_type == "vector":
            norm = np.linalg.norm(value)
            if norm == 0:
                raise ValueError("Wrong value for the unit vector: norm is zero")
            unit_vector = value / norm
        elif self.direction_type == "thetaphi":
            theta, phi = value
            unit_vector = np.array(
                [
                    np.sin(theta) * np.cos(phi),  # x
                    np.sin(theta) * np.sin(phi),  # y
                    np.cos(theta),  # z
                ]
            )
            pass

        # store
        self._unit_vector = unit_vector
        self._direction = value

    # -- hidden methods
    def _positive_float_check(self, param_name: str, value: float) -> None:
        """internal function to check that a parameter is a positive float, raises a `ValueError` if not.

        Args:
            param_name (str): name of the checked parameter, to give context in the exception
            value (float): value of the paramater to check
        """
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise ValueError(f"'{param_name}' has to be a float")
        if value < 0:
            raise ValueError(f"'{param_name}' has to be a positive")


# % IMPLEMENTED CLASSES


class GaussianLaserBeam(AbstractLaserBeam):
    """docstring for GaussianLaserBeam."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
