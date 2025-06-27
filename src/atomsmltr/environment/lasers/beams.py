"""Laser Beams
================

Here we implement the generic ``LaserBeam`` class, as well as some actual
laser beam classes

Examples
---------

Setup a Gaussian beam

.. code-block:: python

    from atomsmltr.environment.lasers import GaussianLaserBeam
    from atomsmltr.environment.lasers.polarization import CircularLeft


    beam = GaussianLaserBeam(
        wavelength=399e-9,
        waist=50e-6,
        power=30e-3,
        waist_position=(0, 0, 0),
        direction=(0, 0, 1),
        polarization=CircularLeft(),
    )

See also
--------
atomsmltr.environment.lasers.polarization

"""

# % IMPORTS
import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod

# % LOCAL IMPORTS
from .polarization import Vertical, Polarization
from ..envbase import EnvObject
from ...utils.infostring import InfoString


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


class LaserBeam(EnvObject):
    """Representing laser beams

    Parameters
    ----------
    wavelength : float, optional
        vacuum wavelength (m), by default 399e-9
    waist : float, optional
        1/e^2 waist radius (m), by default 1e-3
    power : float, optional
        laser power (W), by default 1e-3
    waist_position : array, shape (,3), optional
        cartesian coordinates of the waist / focus position,
        in meters and in the lab frame, by default (0, 0, 0)
    direction : array, shape (,3) or (,2), optional
        depending on 'direction type', a vector or a (theta, phi) couple
        giving the propagation direction of the beam
    direction_type : str, optional
        type of direction : "vector" or "thetaphi", by default "vector"
    polarization : Polarization, optional
        laser polarization, by default Vertical()
    tag : str, optional
        laser tag, by default None
    """

    def __init__(
        self,
        wavelength: float = 399e-9,
        waist: float = 1e-3,
        power: float = 1e-3,
        waist_position: np.ndarray = (0, 0, 0),
        direction: np.ndarray = (0, 0, 1),
        direction_type: str = "vector",
        polarization: Polarization = Vertical(),
        tag: str = None,
    ):

        self.wavelength = wavelength
        self.waist = waist
        self.power = power
        self.waist_position = waist_position
        # /!\ direction_type has to be defined BEFORE direction !!
        self.direction_type = direction_type
        self.direction = direction
        self.polarization = polarization

        super(LaserBeam, self).__init__(tag=tag)

    # -- REQUESTED PROPERTY FOR ENVOBJECTS
    @property
    def vector(self):
        return False

    # -- COMMON METHODS DEFINED HERE
    def _convert_coordinates_to_laser_frame(
        self, position: np.ndarray, nocheck=False
    ) -> np.ndarray:
        """Converts lab frame cartesian coordinates to laser frame coordinates.

        Parameters
        ----------
        position : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        position_laser : array of shape (3,) or (n1, n2, ..., 3) (same as position)
            cartesian coordinates in the laser frame

        Notes
        -------

        'position' should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

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


        Note
        -----

            Note: in some cases (elliptical beams for instance) it might be interesting to include
            a final rotation around the laser propagation axis in the laser frame. We decided that
            this rotation will be handled in the `intensity()` method of the corresponding class.
        """

        # convert to array if needed
        position = self._check_position_array(position, nocheck)
        # get coordinates
        x, y, z = position.T
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

        # also yield cylindrical coordinates - NOT ANYMORE
        # rho_laser = np.sqrt(x_laser**2 + y_laser**2)
        # th_laser = np.arctan2(y_laser, x_laser)
        position_laser = np.array([x_laser, y_laser, z_laser]).T
        return position_laser

    def _convert_vector_to_laser_frame(
        self, vec: np.ndarray, nocheck: bool = False
    ) -> np.ndarray:
        """Rotates a vector from lab frame to laser frame.

        Parameters
        ----------
        vec : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the vectors in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        vec_laser : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the vectors in the laser frame

        Notes
        -------

        'vec' should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains vector coordinates x, y, z

        The unit vector defining laser propagation is defined with two angles, theta
        and phi : theta is the angle between the unit vector and the z axis of the lab
        frame, and phi is the angle of the unit vector project on the (x, y) plane of the
        lab frame, w.r.t the x axis.

        To perform a rotation from lab frame (x, y, z) to laser frame (x_laser, y_laser, z_laser):

        1) we perform a rotation with an angle phi around the lab frame z axis:
            (x, y, z) > (x', y', z')
        2) we perform a rotation with an angle theta around the y' axis of the new frame:
            (x', y', z') > (x_laser, y_laser, z_laser)

        """

        # convert vec
        vec = self._check_position_array(vec, nocheck)
        x, y, z = vec.T
        # rotate : phi around z axis, then theta along new y axis
        # see function docstring and documentation for rotation & frames definitions
        # shorthands
        costheta = self.__costheta
        sintheta = self.__sintheta
        cosphi = self.__cosphi
        sinphi = self.__sinphi
        # compute
        x_laser = x * costheta * cosphi + y * costheta * sinphi - z * sintheta
        y_laser = -x * sinphi + y * cosphi
        z_laser = x * sintheta * cosphi + y * sintheta * sinphi + z * costheta

        vec_laser = np.array([x_laser, y_laser, z_laser]).T
        return vec_laser

    def _convert_vector_to_lab_frame(
        self, vec: np.ndarray, nocheck=False
    ) -> np.ndarray:
        """Rotates a vector from laser frame to lab frame.

        Parameters
        ----------
        vec : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the vectors in the laser frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        vec_lab : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the vectors in the lab frame

        Notes
        ------

        Realizes the reverse operation of `_convert_vector_to_laser_frame`.
        See `_convert_vector_to_laser_frame` docstring for more information

        """

        # convert vec
        vec = self._check_position_array(vec, nocheck)
        x, y, z = vec.T
        # rotate : phi around z axis, then theta along new y axis
        # see function docstring and documentation for rotation & frames definitions
        # shorthands
        costheta = self.__costheta
        sintheta = self.__sintheta
        cosphi = self.__cosphi
        sinphi = self.__sinphi
        # compute
        x_lab = x * costheta * cosphi - y * sinphi + z * sintheta * cosphi
        y_lab = x * costheta * sinphi + y * cosphi + z * sintheta * sinphi
        z_lab = -x * sintheta + z * costheta

        vec_lab = np.array([x_lab, y_lab, z_lab]).T
        return vec_lab

    def get_polarization_vector_in_laser_frame(self) -> np.ndarray:
        """Returns the polarization vector describing the current polarization state, in the **LASER** frame

        Returns
        -------
        p_vec : array of shape (,3)
            cartesian coordinates of the polarization vector (laser frame)

        Notes
        ------
         See documentation for the exact definition of the vector. In short :

        | ``> p_vec = (1, 0, 0)``  : linear polarization along x (vertical)
        | ``> p_vec = (0, 1, 0)``  : linear polarization along y (horizontal)
        | ``> p_vec = (0, 0, 1)``  : circular right polarization
        | ``> p_vec = (0, 0, -1)`` : circular left polarization
        """
        return self.polarization.vector

    def get_polarization_vector_in_lab_frame(self) -> np.ndarray:
        """Returns the polarization vector describing the current polarization state, in the **LAB** frame

        Returns
        -------
        p_vec : array of shape (,3)
            cartesian coordinates of the polarization vector (lab frame)

        Notes
        ------
         See documentation for the exact definition of the vector. In short :

        | ``> p_vec = (1, 0, 0)``  : linear polarization along x (vertical)
        | ``> p_vec = (0, 1, 0)``  : linear polarization along y (horizontal)
        | ``> p_vec = (0, 0, 1)``  : circular right polarization
        | ``> p_vec = (0, 0, -1)`` : circular left polarization
        """
        p_vec_laser_frame = self.polarization.vector
        p_vec_lab_frame = self._convert_vector_to_lab_frame(p_vec_laser_frame)
        return p_vec_lab_frame

    def get_polarization_quant_amplitude(
        self, quantization_axis: np.ndarray, nocheck: bool = False
    ) -> np.ndarray:
        """Returns the projection of the polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩, using
        the vector ``quantization_axis`` as a quantification axis. See documentation
        for a derivation of this projection.

        Parameters
        ----------
        quantization_axis : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the quantization axis vector in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        polar_amp : array of shape (3,) or (n1, n2, ..., 3)
            contains the polarization amplitude for π, σ+ and σ- components

        Notes
        -----

        The input ``quantization_axis`` should be an array of shape (3,) or (n1, n2, .., 3), where
        the cartesian coordinates of the quantization axis are stored in the last dimension (of size 3)

        the result ``polar_amp`` is an array whose size matches the one of ``quantization_axis``, where the last
        dimension of size 3 contains the projections of the polarization state on π, σ+ and σ-

        That is :

        >>> pi_amp, sigmaplus_amp, sigma_minus_amp = polar_amp.T

        With:

        | ``pi_amp`` =  〈Ψ|π⟩
        | ``sigmaplus_amp`` =  〈Ψ|σ+⟩
        | ``sigma_minus_amp`` =  〈Ψ|σ-⟩

        See Also
        --------
        get_polarization_quant()
        get_polarization_quant_amplitude_dict()
        get_polarization_quant_dict()

        """

        # -- process input
        # - check
        quantization_axis = self._check_position_array(quantization_axis, nocheck)

        # -- compute angles of B field w.r.t k vector, in the laser frame
        # 1) coordinates of uB in laser frame
        uB_laser = self._convert_vector_to_laser_frame(quantization_axis, nocheck)
        # 2) compute angles
        xl, yl, zl = uB_laser.T
        alpha = np.arctan2(np.sqrt(xl * xl + yl * yl), zl)  # polar angle
        beta = np.arctan2(yl, xl)  # azimuthal angle

        # -- get angles of polarization vector in the laser frame
        u, v = self.polarization.get_polarization_vector_angles()

        # -- projections of polarization state |Ψ⟩ on |x⟩ and |y⟩
        # >>> see documentation for explanation
        x_proj = (1 / np.sqrt(2)) * (
            np.exp(-1j * v) * np.cos(u / 2) + np.exp(1j * v) * np.sin(u / 2)
        )
        y_proj = (1j / np.sqrt(2)) * (
            np.exp(-1j * v) * np.cos(u / 2) - np.exp(1j * v) * np.sin(u / 2)
        )

        # -- projections of polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩
        # >>> see documentation for explanation
        # shorthands
        sinB = np.sin(beta)
        cosB = np.cos(beta)
        sinA = np.sin(alpha)
        cosA = np.cos(alpha)
        sq2 = np.sqrt(2)
        # |σ+⟩
        sigma_plus_proj = (cosB * cosA + 1j * sinB) / sq2 * x_proj
        sigma_plus_proj += (sinB * cosA - 1j * cosB) / sq2 * y_proj
        # |σ-⟩
        sigma_minus_proj = (cosB * cosA - 1j * sinB) / sq2 * x_proj
        sigma_minus_proj += (sinB * cosA + 1j * cosB) / sq2 * y_proj
        # |π⟩
        pi_proj = cosB * sinA * x_proj + sinB * sinA * y_proj

        # -- result
        polar_amp = np.array([pi_proj, sigma_plus_proj, sigma_minus_proj]).T

        return polar_amp

    def get_polarization_quant(
        self, quantization_axis: np.ndarray, nocheck: bool = False
    ) -> np.ndarray:
        """Returns **squared norm** the projection of the polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩, using
        the vector ``quantization_axis`` as a quantification axis. See documentation
        for a derivation of this projection.

        Parameters
        ----------
        quantization_axis : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the quantization axis vector in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        polar_norm : array of shape (3,) or (n1, n2, ..., 3)
            contains the polarization norm for π, σ+ and σ- components

        Notes
        -----

        The input ``quantization_axis`` should be an array of shape (3,) or (n1, n2, .., 3), where
        the cartesian coordinates of the quantization axis are stored in the last dimension (of size 3)

        the result ``polar_norm`` is an array whose size matches the one of ``quantization_axis``, where the last
        dimension of size 3 contains the projections of the polarization state on π, σ+ and σ-

        That is :

        >>> pi_amp, sigmaplus_amp, sigma_minus_amp = polar_norm.T

        With:

        | ``pi_amp`` =  | 〈Ψ|π⟩ | ** 2
        | ``sigmaplus_amp`` =  | 〈Ψ|σ+⟩ | ** 2
        | ``sigma_minus_amp`` =  | 〈Ψ|σ-⟩ | ** 2

        See Also
        --------
        get_polarization_quant_amplitude()
        get_polarization_quant_amplitude_dict()
        get_polarization_quant_dict()
        """

        polar_amp = self.get_polarization_quant_amplitude(quantization_axis, nocheck)
        polar_norm = np.abs(polar_amp) ** 2
        return polar_norm

    def get_polarization_quant_amplitude_dict(
        self, quantization_axis: np.ndarray
    ) -> dict:
        """Returns the projection of the polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩, using
        the vector ``quantization_axis`` as a quantification axis. See documentation
        for a derivation of this projection.

        Parameters
        ----------
        quantization_axis : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the quantization axis vector in the lab frame

        Returns
        -------
        res : dict
            dict containing the polarization amplitude for π, σ+ and σ- components

        Notes
        ------

        The result is returned as a dictionnary `res`, such as :

        | ``res["sigma+"]`` =  〈Ψ|σ+⟩
        | ``res["sigma-"]`` =  〈Ψ|σ-⟩
        | ``res["pi"]`` =  〈Ψ|π⟩

        See Also
        --------
        get_polarization_quant_amplitude()
        get_polarization_quant()
        get_polarization_quant_dict()
        """
        # -- get result in array form
        polar_amp = self.get_polarization_quant_amplitude(quantization_axis)
        pi_amp, sigma_plus_amp, sigma_minus_amp = polar_amp.T

        # -- result
        res = {"sigma+": sigma_plus_amp, "sigma-": sigma_minus_amp, "pi": pi_amp}

        return res

    def get_polarization_quant_dict(self, quantization_axis):
        """Returns the **squared norm** of projection of the polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩, using
        the magnetic field vector ``quantization_axis`` as a quantification axis. See documentation
        for a derivation of this projection.

        Parameters
        ----------
        quantization_axis : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates of the quantization axis vector in the lab frame

        Returns
        -------
        res : dict
            dict containing the polarization norm for π, σ+ and σ- components

        Notes
        ------

        The result is returned as a dictionnary `res`, such as :

        | ``res["sigma+"]`` =  | 〈Ψ|σ+⟩ | ** 2
        | ``res["sigma-"]`` =  | 〈Ψ|σ-⟩ | ** 2
        | ``res["pi"]`` =  | 〈Ψ|π⟩ | ** 2

        See Also
        --------
        get_polarization_quant_amplitude()
        get_polarization_quant()
        get_polarization_quant_amplitude_dict()

        """
        projection_amplitude = self.get_polarization_quant_amplitude_dict(
            quantization_axis
        )
        res = {}
        for k, v in projection_amplitude.items():
            res[k] = np.linalg.norm(v) ** 2
        return res

    # -- REQUIRED ABSTRACT METHODS

    def get_value(self, position: np.ndarray, nocheck=False) -> np.ndarray:
        """Returns laser intensity at a given position in the lab frame

        Parameters
        ----------
        position : array of shape (3,) or (n1, n2, ..., 3)
            cartesian coordinates in the lab frame
        nocheck : bool, optional
            if set to True, function will not check that the shape of position
            matches requirements, by default False

        Returns
        -------
        intensity : float or array of shape (n1, n2, ..., 1)
            laser intensity at position

        Notes
        -------
        position is an array_like object, with shape (3,) or (n1, n2, .., 3).
        In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame

        """
        # Check position
        position = self._check_position_array(position, nocheck)
        # call hidden function that actually does the computation
        return self._intensity_func(self, position)

    @abstractmethod
    def _intensity_func(self, position):
        """Actual method for field computation ; defined for each subclass"""

    @abstractmethod
    def set_power_from_I(self, target_I: float):
        """Sets the power to reach a target intensity

        Parameters
        ----------
        target_I : float
            target intensity (W/m^2)
        """

    @abstractmethod
    def set_waist_from_I(self, target_I: float):
        """Sets the waist radius to reach a target intensity

        Parameters
        ----------
        target_I : float
            target intensity (W/m^2)
        """

    # -- CLASS PROPERTIES GETTERS & SETTERS
    # - wavelength
    @property
    def wavelength(self) -> float:
        """float: laser vacuum wavelength (m)"""
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
        """float: laser 1/e^2 radius (m)"""
        return self._waist

    @waist.setter
    def waist(self, value: float) -> None:
        self._positive_float_check("waist", value)
        self._waist = float(value)

    # - power
    @property
    def power(self) -> float:
        """float: laser power (W)"""
        return self._power

    @power.setter
    def power(self, value: float) -> None:
        self._positive_float_check("power", value)
        self._power = float(value)

    # - waist position
    @property
    def waist_position(self) -> np.ndarray:
        """array of shape (,3): cartesian coordinates of the laser focus / waist position in the lab frame (m)"""
        return self._waist_position

    @waist_position.setter
    def waist_position(self, value: np.ndarray) -> None:
        value = np.asanyarray(value)
        if value.size != 3:
            raise ValueError("'waist_position' should be an array-like of size 3")
        self._waist_position = value

    # - direction_type
    @property
    def direction_type(self) -> str:
        """str: type of direction setting. Can be "vector" or "thetaphi" """
        return self._direction_type

    @direction_type.setter
    def direction_type(self, value: str) -> None:
        if value not in DIRECTION_TYPES:
            raise ValueError(f"'direction_type' should be in {DIRECTION_TYPES}")
        self._direction_type = value

    # - direction
    @property
    def direction(self) -> np.ndarray:
        """array: either a vector or a (theta, phi) tuple describing the laser direction"""
        return self._direction

    @direction.setter
    def direction(self, value: np.ndarray) -> None:
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
        # pre compute some values, for later
        self.__costheta = np.cos(theta)
        self.__sintheta = np.sin(theta)
        self.__cosphi = np.cos(phi)
        self.__sinphi = np.sin(phi)

    # - polarization
    @property
    def polarization(self) -> Polarization:
        """Polarization: laser polarization object"""
        return self._polarization

    @polarization.setter
    def polarization(self, value: Polarization) -> None:
        if not isinstance(value, Polarization):
            msg = "`polarization` should be a Polarization object, from atomsmltr.environment.lasers.polarization"
            raise TypeError(msg)
        self._polarization = value

    # - others
    @property
    def unit_vector(self) -> np.ndarray:
        """array of shape (,3): unit vector describing laser propagation"""
        return self._unit_vector

    @property
    def k(self) -> float:
        """float: laser wavenumber k = 2π / λ (m^-1)"""
        return 2 * np.pi / self.wavelength

    @property
    def kvec(self) -> np.ndarray:
        """array: vector version of the laser wavenumber k = 2π / λ (m^-1)"""
        return self.k * self.unit_vector

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

    # -- PLOT FUNCTIONS

    def plot1D(self):
        pass

    def plot2D(
        self,
        limits: np.ndarray,
        Npoints: np.ndarray,
        cut: float = 0,
        ax=None,
        plane: str = "XY",
        cmap=None,
        show: bool = False,
        space_scale: float = 1.0,
    ):
        """Plots a 2D cut of the laser intensity, using Matplotlib pcolormesh()

        Parameters
        ----------
        limits : array, shape (4,)
            an array of size 4, providing (xmin, xmax, ymin, ymax).
        Npoints : int or array of shape (2,)
            number of points for each dimension,
            either a int or an array of two ints (Nx, Ny).
        cut : float, optional
            coordinate of the third axis for the cut. Defaults to 0.
        ax : Matplotlib Axes, optional
            the matplotlib axis on which to plot.
            If None is given a new figure is created.
            Defaults to None.
        plane : str, optional
            the plane for the cut. Accepted values are "XY", "YZ" and "ZX". Defaults to "XY".
        cmap : Matplotlib cmap, optional
            passed to matplotlib pcolormesh() function
        show : bool, optional
            whether to show the figure after calling the method. Defaults to False.
        space_scale : float, optional
            space coordinates will be multiplied by this when plotting. Defaults to 1.

        Returns
        -------
        ax : Matplotlib Axes
            the axis on which the plot was performed.

        Notes
        ------
        The limits are given via an array of size 4 'limits', providing providing (xmin, xmax, ymin, ymax)
        Number of points are given with 'Npoints', either as an integer (same value for x and y) or an array of size 2
        the coordinate of the cut axis is given by 'cut'

        Examples
        ---------
        >>> beam.plot2D(limits=(-5, 5, -4, 4), Npoints=200)
        >>> beam.plot2D(limits=(-5, 5, -4, 4), Npoints=200, cut=-5)
        >>> beam.plot2D(limits=(-5, 5, -4, 4), Npoints=(200, 100))

        """
        # - process arguments using the Plottable builtin method
        ax, position, X, Y = self._process_2D_plot_args(
            ax=ax,
            plane=plane,
            limits=limits,
            Npoints=Npoints,
            cut=cut,
        )

        # - compute intensity
        intensity = self.get_value(position)

        # - plot
        ax.pcolormesh(X * space_scale, Y * space_scale, intensity, cmap=cmap)
        ax.set_xlabel(plane.upper()[0])
        ax.set_ylabel(plane.upper()[1])

        # - show ?
        if show:
            plt.show()

        return ax

    def plot3D(
        self,
        ax=None,
        color: str = None,
        name: str = None,
        vscale: float = None,
        show: bool = False,
    ):
        """plots a 3D reprensentation of the laser beam, including:
               - a line : laser axis
               - an arrow along the propagation direction
               - a point : laser focus position
               - a dotted arrow : laser polarization vector

        Parameters
        ----------
        ax : custom Axes3D, optional
            the axis in which to plot. If None is given (default value) a new ax is generated
        color : str, optional
            a matplotlib compatible color. Defaults to None.
        name : str, optional
            the name of the laser, passed as a label when plotting. If none is given, use the laser tag
        vscale : float, optional
            A scaling factor. Use it to tweak the arrow size if needed. Defaults to None.
        show : bool, optional
            Whether the show the figure after calling the method. Defaults to False.

        Returns
        -------
        ax : custom Axes3D
            the figure axis in which the laser is plotted.

        Note
        ----
            When providing an axis via the ``ax`` parameter, make sure to use our custom implementation of
            matplotlib ``Axes3D``, as this function uses custom arrow drawing methods. The class can be imported
            via ``from atomsmltr.utils.plotter import Axes3D``
        """
        # - init ax (if needed)
        ax = self._init_ax(ax, ax3D=True)

        # - get laser information
        unit_vector = np.asanyarray(self._unit_vector)
        polar_vector_laserframe = np.asanyarray(self.polarization.vector)
        polar_vector = self._convert_vector_to_lab_frame(polar_vector_laserframe)
        waist_position = np.asanyarray(self.waist_position)
        # - scale
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        zmin, zmax = ax.get_zlim()
        dr = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
        if vscale is None:
            vscale = np.max(dr) / 5

        # - PLOT
        # waist position
        label = self.tag if name is None else name
        ax.scatter(*waist_position, marker="o", color=color, label=label)

        # plot laser
        r1 = waist_position + dr * unit_vector * 5
        r2 = waist_position - dr * unit_vector * 5
        x = np.linspace(-100, 100, 1000)
        r = waist_position[:, np.newaxis] + (unit_vector * dr)[:, np.newaxis] * x
        ax.plot(r[0, :], r[1, :], r[2, :], color=color)

        # plot propagation vector
        epsilon = 0.2
        ax.arrow3D(
            *(waist_position - unit_vector * vscale * (1 + epsilon)),
            *(vscale * unit_vector),
            mutation_scale=15,
            arrowstyle="simple",
            ec="k",
            fc=color,
        )

        # plot polarisation vector
        ax.arrow3D(
            *(waist_position - unit_vector * vscale * (1 + epsilon)),
            *(vscale * polar_vector * 0.7),
            mutation_scale=20,
            arrowstyle="-|>",
            linestyle="dashed",
            color=color,
        )

        if show:
            plt.show()
        return ax

    # -- INFO STRING

    @property
    @abstractmethod
    def disp_type(self) -> str:
        return ""

    def gen_infostring_obj(self, show_polar=True):
        """Generates an info string object"""
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element(f"type", f"{self.disp_type}")
        info.add_element(f"tag", f"{self.tag}")
        info.add_element(f"waist (m)", f"{self.waist:.3g}")
        info.add_element(f"power (W)", f"{self.power:.3g}")
        info.add_element(f"waist position (m)", f"{self.waist_position}")
        info.add_element(f"direction type", f"{self.direction_type}")
        info.add_element(f"direction", f"{self.direction}")
        info.add_element(f"unit vector", f"{self._unit_vector}")
        info.add_element(f"unit vector phi", f"π × {self._unit_vector_phi / np.pi}")
        info.add_element(f"unit vector theta", f"π × {self._unit_vector_theta / np.pi}")

        if show_polar:
            info_polar = self.polarization.gen_infostring_obj()
            info.merge(info_polar, prefix="")

        return info

    def print_info(self, show_polar=True):
        info_str = self.gen_infostring_obj(show_polar)
        print(info_str.generate())

    def print_polar_proj(self, mag_field_vector):
        mag_field_vector = self._check_position_array(mag_field_vector)
        res = self.get_polarization_quant_dict(mag_field_vector)
        print("> Local polarization projection")
        print(f"   + B  = {mag_field_vector*1e4} (G)")
        print(f"   + π  = {res["pi"]:.2f}")
        print(f"   + σ+ = {res["sigma+"]:.2f}")
        print(f"   + σ- = {res["sigma-"]:.2f}")


# % IMPLEMENTED CLASSES


class GaussianLaserBeam(LaserBeam):
    """A Gaussian laser beam

    Parameters
    ----------
    wavelength : float, optional
        vacuum wavelength (m), by default 399e-9
    waist : float, optional
        1/e^2 waist radius (m), by default 1e-3
    power : float, optional
        laser power (W), by default 1e-3
    waist_position : array, shape (,3), optional
        cartesian coordinates of the waist / focus position,
        in meters and in the lab frame, by default (0, 0, 0)
    direction : array, shape (,3) or (,2), optional
        depending on 'direction type', a vector or a (theta, phi) couple
        giving the propagation direction of the beam
    direction_type : str, optional
        type of direction : "vector" or "thetaphi", by default "vector"
    polarization : Polarization, optional
        laser polarization, by default Vertical()
    tag : str, optional
        laser tag, by default None
    """

    @property
    def type(self):
        return "Gaussian Laser Beam"

    @property
    def disp_type(self) -> str:
        return "Gaussian beam"

    # -- REQUIRED METHOD FOR LASER BEAM CLASSES
    # pylint : disable=method_hidden
    @staticmethod
    def _intensity_func(self, position):
        """Returns laser intensity at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `LaserBeam` class
        """
        # - get coordinates in laser frame
        # NB : x, y and phi are not needed here
        position_laser = self._convert_coordinates_to_laser_frame(position)
        x_laser, y_laser, z_laser = position_laser.T
        rho_laser = np.sqrt(x_laser**2 + y_laser**2)
        # - compute gaussian beam intensity
        intensity = _intensity_gauss(
            rho_laser, z_laser, self.waist, self.power, self.wavelength
        )
        intensity = intensity.T
        return intensity

    def set_power_from_I(self, target_I: float) -> None:
        # NB, for a Gaussian beam : I0 = 2 * P / np.pi / w0**2
        power = target_I * self.waist**2 * np.pi / 2
        self.power = power

    def set_waist_from_I(self, target_I: float) -> None:
        # NB, for a Gaussian beam : I0 = 2 * P / np.pi / w0**2
        waist = np.sqrt(2 * self.power / np.pi * target_I)
        self.waist = waist

    @property
    def rayleigh_length(self) -> float:
        """float: the laser Rayleigh length"""
        return np.pi * self.waist**2 / self.wavelength

    def gen_infostring_obj(self, show_polar=True):
        info = super().gen_infostring_obj(show_polar)
        info.add_element(
            "Rayleigh length", f"{self.rayleigh_length:.2g} m", section="Parameters"
        )
        return info


class PlaneWaveLaserBeam(LaserBeam):
    """Implements a plane wave. For convenience, we still define the waist and power, and
    the intensity is constant and corresponds to the peak intensity of a Gaussian beam with
    same power and waist."""

    @property
    def type(self):
        return "Plane Wave Laser Beam"

    @property
    def disp_type(self) -> str:
        return "Plane wave beam"

    # -- REQUIRED METHOD FOR LASER BEAM CLASSES
    # pylint : disable=method_hidden
    @staticmethod
    def _intensity_func(self, position: np.ndarray) -> np.ndarray:
        """Returns laser intensity at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `LaserBeam` class
        """
        # - get coordinates in laser frame
        # NB : x, y and phi are not needed here
        x, _, _ = position.T
        x = x.T
        # - compute gaussian beam intensity
        intensity = _intensity_gauss(
            0 * x, 0 * x, self.waist, self.power, self.wavelength
        )

        return intensity

    def set_power_from_I(self, target_I: float) -> None:
        # NB, for a Gaussian beam : I0 = 2 * P / np.pi / w0**2
        power = target_I * self.waist**2 * np.pi / 2
        self.power = power

    def set_waist_from_I(self, target_I: float) -> None:
        # NB, for a Gaussian beam : I0 = 2 * P / np.pi / w0**2
        waist = np.sqrt(2 * self.power / np.pi * target_I)
        self.waist = waist

    def gen_infostring_obj(self, show_polar=True):
        info = super().gen_infostring_obj(show_polar)
        info.rm_element("waist position (m)", section="Parameters")
        return info
