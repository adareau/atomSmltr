# -*- coding: utf-8 -*-
"""Defines the laser beam classes
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from abc import abstractmethod

# % LOCAL IMPORTS
from .polarization import Vertical, AbstractPolarization
from ..envbase import EnvObject
from ...utils.misc import check_position_array
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


class AbstractLaserBeam(EnvObject):
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
        tag: str = "",
    ):
        self.wavelength = wavelength
        self.waist = waist
        self.power = power
        self.waist_position = waist_position
        # /!\ direction_type has to be defined BEFORE direction !!
        self.direction_type = direction_type
        self.direction = direction
        self.polarization = polarization

        super().__init__(tag=tag)

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

    def get_polarization_magnetic_projection(self, mag_field_vector):
        """Returns the projection of the polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩, using
        the magnetic field vector `mag_field_vector` as a quantification axis. See documentation
        for a derivation of this projection.

        The result is returned as a dictionnary `res`, such as :

        res["sigma+"] =  〈Ψ|σ+⟩
        res["sigma-"] =  〈Ψ|σ-⟩
        res["pi"] =  〈Ψ|π⟩

        Args:
            mag_field_vector (array, size 3): cartesian coordinates of the magnetic field in the lab frame

        Returns:
            res (dict): dict containing the projections, see above
        """
        # -- convert mag field to numpy array
        uB = np.asanyarray(mag_field_vector)
        assert uB.size == 3, "`mag_field_vector` should be an array of size 3"
        norm = np.linalg.norm(uB)
        assert norm > 0, "`mag_field_vector` nor should not be zero"
        uB = uB / norm

        # -- compute angles of B field w.r.t k vector, in the laser frame
        # 1) coordinates of uB in laser frame
        uB_laser = self._convert_vector_to_laser_frame(uB)
        # 2) compute angles
        xl, yl, zl = uB_laser
        alpha = np.arctan2(np.sqrt(xl**2 + yl**2), zl)  # polar angle
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
        res = {"sigma+": sigma_plus_proj, "sigma-": sigma_minus_proj, "pi": pi_proj}

        return res

    def get_polarization_magnetic_projection_norm(self, mag_field_vector):
        """Returns the **squared norm** of projection of the polarization state |Ψ⟩ on |σ+⟩, |σ-⟩ and |π⟩, using
        the magnetic field vector `mag_field_vector` as a quantification axis. See documentation
        for a derivation of this projection.

        The result is returned as a dictionnary `res`, such as :

        res["sigma+"] =  |〈Ψ|σ+⟩|**2
        res["sigma-"] =  |〈Ψ|σ-⟩|**2
        res["pi"] =  |〈Ψ|π⟩|**2

        Args:
            mag_field_vector (array, size 3): cartesian coordinates of the magnetic field in the lab frame

        Returns:
            res (dict): dict containing the projections, see above
        """
        projection_amplitude = self.get_polarization_magnetic_projection(
            mag_field_vector
        )
        res = {}
        for k, v in projection_amplitude.items():
            res[k] = np.linalg.norm(v) ** 2
        return res

    # -- REQUIRED ABSTRACT METHODS

    def get_intensity(self, position: np.ndarray) -> np.ndarray:
        """Returns laser intensity at a given position in the lab frame

            position is an array_like object, with shape (3,) or (n1, n2, .., 3).
            In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame

        Args:
            position (array_like, shape (3,) or (n,3)) : positions at which the intensity is computed

        Returns:
            intensity (float or array): laser intensity at positions, with dimension matching the 'position' input.
        """
        # Check position
        position = check_position_array(position)
        # call hidden function that actually does the computation
        return self._intensity_func(self, position)

    @abstractmethod
    def _intensity_func(self, position):
        """Actual method for field computation ; defined for each subclass"""

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
            raise TypeError(msg)
        self._polarization = value

    # - others
    @property
    def unit_vector(self):
        return self._unit_vector

    @property
    def k(self):
        return 2 * np.pi / self.wavelength

    @property
    def kvec(self):
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
        cut=0,
        ax=None,
        plane="XY",
        cmap=None,
        show=False,
        space_scale=1.0,
    ):
        """Plots a 2D cut of the laser intensity.

        The limits are given via an array of size 4 'limits', providing providing (xmin, xmax, ymin, ymax)
        Number of points are given with 'Npoints', either as an integer (same value for x and y) or an array of size 2
        the coordinate of the cut axis is given by 'cut'

        Examples:
            > beam.plot2D(limits=(-5, 5, -4, 4), Npoints=200)
            > beam.plot2D(limits=(-5, 5, -4, 4), Npoints=200, cut=-5, plane="YZ")
            > beam.plot2D(limits=(-5, 5, -4, 4), Npoints=(200, 100))


        Args:
            limits (array): An array of size 4, providing (xmin, xmax, ymin, ymax).
            Npoints (int or array): Number of points for each dimension. Either a int or an array of two ints (Nx, Ny).
            cut (float, optional): coordinate of the third axis for the cut. Defaults to 0.
            ax (matploblit ax, optional): The axis on which to plot. Defaults to None.
            plane (string, optional): The plane for the cut. Accepted values are "XY", "YZ" and "ZX". Defaults to "XY".
            cmap (colormap, optional): colormap used in pcolormesh. Defaults to None.
            show (bool, optional): whether to show the figure after calling the method. Defaults to False.
            space_scale (float, optional): space coordinates will be multiplied by this when plotting. Defaults to 1.


        Returns:
            ax (matplotlib axis): the axis
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
        intensity = self.get_intensity(position)

        # - plot
        ax.pcolormesh(X * space_scale, Y * space_scale, intensity, cmap=cmap)
        ax.set_xlabel(plane.upper()[0])
        ax.set_ylabel(plane.upper()[1])

        # - show ?
        if show:
            plt.show()

        return ax

    def plot3D(self, ax=None, color=None, name=None, vscale=None, show=False):
        """plots a 3D reprensentation of the laser beam, including:
               - a line : laser axis
               - an arrow along the propagation direction
               - a point : laser focus position
               - a dotted arrow : laser polarization vector

            When providing an axis via the `ax` parameter, make sure to use our custom implementation of
            matplotlib `Axes3D`, as this function uses custom arrow drawing methods. The class can be imported
            via `from atomsmltr.utils.plotter import Axes3D`


        Args:
            ax (custom Axes3D, optional): The axis in which to plot. If None is given (default value) a new ax is generated
            color (string, optional): A matplotlib compatible color. Defaults to None.
            name (string, optional): The name of the laser, passed as a label when plotting. If none is given, use the laser tag
            vscale (float, optional): A scaling factor. Use it to tweak the arrow size if needed. Defaults to None.
            show (bool, optional): Whether the show the figure after calling the method. Defaults to False.

        Returns:
            ax: the figure axis in which the laser is plotted.
        """

        # - init ax (if needed)
        ax = self._init_ax(ax, ax3D=True)

        # - get laser information
        unit_vector = np.asanyarray(self._unit_vector)
        polar_vector_laserframe = np.asanyarray(
            self.polarization.get_polarization_vector()
        )
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


# % IMPLEMENTED CLASSES


class GaussianLaserBeam(AbstractLaserBeam):
    """docstring for GaussianLaserBeam."""

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
            `AbstractLaserBeam` class
        """
        # - get coordinates in laser frame
        # NB : x, y and phi are not needed here
        x, y, z = position.T
        _, _, z_laser, rho_laser, _ = self._convert_coordinates_to_laser_frame(x, y, z)

        # - compute gaussian beam intensity
        intensity = _intensity_gauss(
            rho_laser, z_laser, self.waist, self.power, self.wavelength
        )

        return intensity

    @property
    def rayleigh_length(self) -> float:
        return np.pi * self.waist**2 / self.wavelength

    def gen_infostring_obj(self, show_polar=True):
        info = super().gen_infostring_obj(show_polar)
        info.add_element(
            "Rayleigh length", f"{self.rayleigh_length:.2g} m", section="Parameters"
        )
        return info
