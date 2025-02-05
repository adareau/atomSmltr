# -*- coding: utf-8 -*-
"""Defines the laser beam classes
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from enum import Enum, auto

# % LOCAL IMPORTS
from .polarization import Vertical, AbstractPolarization

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
        polarization: AbstractPolarization = Vertical(),
    ):
        self.wavelength = wavelength
        self.waist = waist
        self.power = power
        self.waist_position = waist_position
        # /!\ direction_type has to be defined BEFORE direction !!
        self.direction_type = direction_type
        self.direction = direction
        self.polarization = polarization

        super().__init__()

    # -- COMMON METHODS DEFINED HERE
    def _convert_coordinates_to_laser_frame(self, x, y, z):
        """Converts lab frame cartesian coordinates to laser frame coordinates.
        The laser frame is centered at the laser waist, and has the z axis aligned
        with the laser propagation.

        The unit vector defining laser propagation is defined with two angles, theta
        and phi : theta is the angle between the unit vector and the z axis of the lab
        frame, and phi is the angle of the unit vector project on the (x, y) plane of the
        lab frame, w.r.t the x axis.

        To define the new coordinates (x_laser, y_laser, z_laser) in the laser frame, we
        proceed as follow:

        1) we shift the frame to center it on the waist position:
            (x, y, z) > (xc, yc, zc)
        2) we perform a rotation with an angle phi around the lab frame z axis:
            (xc, yc, zc) > (x', y', z')
        3) we perform a rotation with an angle theta around the y' axis of the new frame:
            (x', y', z') > (x_laser, y_laser, z_laser)

        For convenience reasons, we also return polar coordinates in the laser frame

        Note: in some cases (elliptical beams for instance) it might be interesting to include
        a final rotation around the laser propagation axis in the laser frame. We decided that
        this rotation will be handled in the `intensity()` method of the corresponding class.

        Args:
            x (float or array): x cartesian coordinate in the lab frame
            y (float or array): y cartesian coordinate in the lab frame
            z (float or array): z cartesian coordinate in the lab frame

        Returns:
            x_laser (float or array): x cartesian coordinate in the laser frame
            y_laser (float or array): y cartesian coordinate in the laser frame
            z_laser (float or array): z cartesian coordinate in the laser frame
            rho_laser (float or array): radial polar coordinate in the laser frame
            phi_laser (float or array): angular polar coordinate in the laser frame
        """

        # shift center
        x0, y0, z0 = self._waist_position
        xc = x - x0
        yc = y - y0
        zc = z - z0

        # rotate : phi around z axis, then theta along new y axis
        # see function docstring and documentation for rotation & frames definitions
        theta = self._unit_vector_theta
        phi = self._unit_vector_phi
        x_laser = (
            xc * np.cos(theta) * np.cos(phi)
            + yc * np.cos(theta) * np.sin(phi)
            - zc * np.sin(theta)
        )
        y_laser = -xc * np.sin(phi) + yc * np.cos(phi)
        z_laser = (
            xc * np.sin(theta) * np.cos(phi)
            + yc * np.sin(theta) * np.sin(phi)
            + zc * np.cos(theta)
        )

        # also yield cylindrical coordinates
        rho_laser = np.sqrt(x_laser**2 + y_laser**2)
        th_laser = np.arctan2(y_laser, x_laser)

        return x_laser, y_laser, z_laser, rho_laser, th_laser

    def _convert_vector_to_laser_frame(self, vec):
        """Rotates a vector from lab frame to laser frame.

        The unit vector defining laser propagation is defined with two angles, theta
        and phi : theta is the angle between the unit vector and the z axis of the lab
        frame, and phi is the angle of the unit vector project on the (x, y) plane of the
        lab frame, w.r.t the x axis.

        To perform a rotation from lab frame (x, y, z) to laser frame (x_laser, y_laser, z_laser):

        1) we perform a rotation with an angle phi around the lab frame z axis:
            (x, y, z) > (x', y', z')
        2) we perform a rotation with an angle theta around the y' axis of the new frame:
            (x', y', z') > (x_laser, y_laser, z_laser)

        Args:
            vec (array of size 3): vector cartesian coordinates in the lab frame (x, y, z)

        Returns:
            vec_laser (array of size 3): vector cartesian coordinates in the laser frame (x_laser, y_laser, z_laser)
        """
        # convert vec
        vec = np.asanyarray(vec)
        assert vec.size == 3, "`vec` should be an array of size 3"
        x, y, z = vec
        # rotate : phi around z axis, then theta along new y axis
        # see function docstring and documentation for rotation & frames definitions
        theta = self._unit_vector_theta
        phi = self._unit_vector_phi
        x_laser = (
            x * np.cos(theta) * np.cos(phi)
            + y * np.cos(theta) * np.sin(phi)
            - z * np.sin(theta)
        )
        y_laser = -x * np.sin(phi) + y * np.cos(phi)
        z_laser = (
            x * np.sin(theta) * np.cos(phi)
            + y * np.sin(theta) * np.sin(phi)
            + z * np.cos(theta)
        )

        vec_laser = np.array([x_laser, y_laser, z_laser])
        return vec_laser

    def _convert_vector_to_lab_frame(self, vec):
        """Rotates a vector from laser frame to lab frame.

        Realizes the reverse operation of `_convert_vector_to_laser_frame`.
        See `_convert_vector_to_laser_frame` docstring for more information

        Args:
            vec (array of size 3): vector cartesian coordinates in the laser frame (x_laser, y_laser, z_laser)

        Returns:
            vec_lab (array of size 3): vector cartesian coordinates in the lab frame (x, y, z)
        """
        # convert vec
        vec = np.asanyarray(vec)
        assert vec.size == 3, "`vec` should be an array of size 3"
        x, y, z = vec
        # rotate : phi around z axis, then theta along new y axis
        # see function docstring and documentation for rotation & frames definitions
        theta = self._unit_vector_theta
        phi = self._unit_vector_phi
        x_lab = (
            x * np.cos(theta) * np.cos(phi)
            - y * np.sin(phi)
            + z * np.sin(theta) * np.cos(phi)
        )
        y_lab = (
            x * np.cos(theta) * np.sin(phi)
            + y * np.cos(phi)
            + z * np.sin(theta) * np.sin(phi)
        )
        z_lab = -x * np.sin(theta) + z * np.cos(theta)

        vec_lab = np.array([x_lab, y_lab, z_lab])
        return vec_lab

    def get_polarization_vector_in_laser_frame(self):
        """Returns the polarization vector describing the current polarization state, in the **LASER** frame

        See documentation for the exact definition of the vector. In short :

        > p_vec = (1, 0, 0)  : linear polarization along x (vertical)
        > p_vec = (0, 1, 0)  : linear polarization along y (horizontal)
        > p_vec = (0, 0, 1)  : circular right polarization
        > p_vec = (0, 0, -1) : circular left polarization

        Returns:
            p_vec: numpy array of size 3, containing the cartesian coordinates of the polarization vector (laser frame)
        """
        return self.polarization.get_polarization_vector()

    def get_polarization_vector_in_lab_frame(self):
        """Returns the polarization vector describing the current polarization state, in the **LAB** frame

        See documentation for the exact definition of the vector. In short :

        > p_vec = (1, 0, 0)  : linear polarization along x (vertical)
        > p_vec = (0, 1, 0)  : linear polarization along y (horizontal)
        > p_vec = (0, 0, 1)  : circular right polarization
        > p_vec = (0, 0, -1) : circular left polarization

        Returns:
            p_vec: numpy array of size 3, containing the cartesian coordinates of the polarization vector (lab frame)
        """
        p_vec_laser_frame = self.polarization.get_polarization_vector()
        p_vec_lab_frame = self._convert_vector_to_lab_frame(p_vec_laser_frame)
        return p_vec_lab_frame

    # -- REQUIRED ABSTRACT METHODS
    @abstractmethod
    def get_intensity(self, x, y, z):
        """Returns laser intensity at point (x, y, z) in the lab frame
        ATTENTION: x, y, z must be floats or arrays of same size !!

        Args:
            x (float or array): x cartesian coordinate in the lab frame
            y (float or array): y cartesian coordinate in the lab frame
            z (float or array): z cartesian coordinate in the lab frame

        Returns:
            I (float or array): laser intensity at point (x, y, z)
        """
        pass

    # -- CLASS PROPERTIES GETTERS & SETTERS
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
            # first case : a unit vector is provided
            # 1 - normalize
            norm = np.linalg.norm(value)
            if norm == 0:
                raise ValueError("Wrong value for the unit vector: norm is zero")
            unit_vector = value / norm
            # 2 - compute theta and phi
            ux, uy, uz = unit_vector
            theta = np.arctan2(np.sqrt(ux**2 + uy**2), uz)
            phi = np.arctan2(uy, ux)

        elif self.direction_type == "thetaphi":
            # second case : theta and phi are provided
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
        self._unit_vector_phi = phi
        self._unit_vector_theta = theta
        self._direction = value

    # - polarization
    @property
    def polarization(self) -> AbstractPolarization:
        return self._polarization

    @polarization.setter
    def polarization(self, value: AbstractPolarization) -> None:
        if not isinstance(value, AbstractPolarization):
            msg = "`polarization` should be a Polarization object, from atomsmltr.environment.lasers.polarization"
            raise ValueError(msg)
        self._polarization = value

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

    # -- REQUIRED METHOD FOR LASER BEAM CLASSES
    def get_intensity(self, x, y, z):
        """Returns laser intensity at point (x, y, z) in the lab frame
        ATTENTION: x, y, z must be floats or arrays of same size !!

        Args:
            x (float or array): x cartesian coordinate in the lab frame
            y (float or array): y cartesian coordinate in the lab frame
            z (float or array): z cartesian coordinate in the lab frame

        Returns:
            I (float or array): laser intensity at point (x, y, z)
        """
        # - get coordinates in laser frame
        # NB : x, y and phi are not needed here
        _, _, z_laser, rho_laser, _ = self._convert_coordinates_to_laser_frame(x, y, z)

        # - compute gaussian beam intensity
        intensity = _intensity_gauss(
            rho_laser, z_laser, self.waist, self.power, self.wavelength
        )

        return intensity
