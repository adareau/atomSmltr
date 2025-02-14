# -*- coding: utf-8 -*-
"""Defines the generic Field Class (for vector fields)
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from abc import abstractmethod

# % LOCAL IMPORTS
from ...utils.plotter import Plottable
from ...utils.infostring import InfoString
from ...utils.misc import check_position_array

# % ABSTRACT CLASSES


class AbstractField(Plottable):
    """A generic, abstract class to handle fields (magnetic mostly)"""

    def __init__(self):
        super(AbstractField, self).__init__()

    @property
    @abstractmethod
    def type():
        """Type has to be defined in the concrete class"""
        pass

    @property
    @abstractmethod
    def unit():
        """unit has to be defined in the concrete class"""
        pass

    def value(self, position: np.ndarray) -> np.ndarray:
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
        return self._field_value_func(self, position)

    @abstractmethod
    def _field_value_func(self, position):
        """Actual method for field computation ; defined for each subclass"""

    # -- INFO STRING / OBJECT MANAGEMENT
    @abstractmethod
    def gen_infostring_obj(self):
        """should return the infostring object"""
        pass

    def gen_info_string(self, **kwargs):
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        print(self.gen_info_string())

    # -- PLOT
    # TODO > plot methods, at this level !!!
    def plot1D(self, ax=None):
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
        """Plots a 2D cut of the field.

        The limits are given via an array of size 4 'limits', providing providing (xmin, xmax, ymin, ymax)
        Number of points are given with 'Npoints', either as an integer (same value for x and y) or an array of size 2
        the coordinate of the cut axis is given by 'cut'

        Examples:
            > field.plot2D(limits=(-5, 5, -4, 4), Npoints=200)
            > field.plot2D(limits=(-5, 5, -4, 4), Npoints=200, cut=-5)
            > field.plot2D(limits=(-5, 5, -4, 4), Npoints=(200, 100))


        Args:
            limits (array): An array of size 4, providing (xmin, xmax, ymin, ymax).
            Npoints (int or array, optional): Number of points for each dimension. Either a int or an array of two ints (Nx, Ny).
            cut (float, optional): coordinate of the third axis for the cut. Defaults to 0.
            ax (matploblit ax, optional): The axis on which to plot. Defaults to None.
            plane (string, optional): The plane for the cut. Accepted values are "XY", "YZ" and "ZX". Defaults to "XY".
            cmap (optional): passed to matplotlib streamplot() function
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
        # - compute field
        mag_field = self.value(position)
        Bx, By, Bz = mag_field.T
        # - get relevant part
        match plane.upper():
            case "XY":
                u = Bx
                v = By

            case "YZ":
                u = By
                v = Bz
            case "ZX":
                u = Bz
                v = Bx

        color = np.sqrt(Bx**2 + By**2 + Bz**2)
        # Transpose if needed, since streamplot is quite strict..
        if not np.allclose(X[0], X):
            X = X.T
            Y = Y.T
            u = u.T
            v = v.T
            color = color.T

        # - plot
        ax.streamplot(X * space_scale, Y * space_scale, u, v, color=color, cmap=cmap)
        ax.set_xlabel(plane.upper()[0])
        ax.set_ylabel(plane.upper()[1])

        # - show ?
        if show:
            plt.show()

        return ax

    def plot3D(
        self,
        limits,
        Npoints,
        ax=None,
        color=None,
        name=None,
        show=False,
        scale=1.0,
        normalize=False,
    ):
        """plots a 3D representation of the field.

        Args:
            limits (array of size 6): limits for the plot (xmin, xmax, ymin, ymax, zmin, zmax)
            Npoints (int or array): Number of points for each dimension. Either a int or an array of trhee ints (Nx, Ny, Nz)
            ax (custom Axes3D, optional): The axis in which to plot. If None is given (default value) a new ax is generated
            color (string, optional): A matplotlib compatible color. Defaults to None.
            name (string, optional): The name of the laser, passed as a label when plotting. Defaults to None.
            show (bool, optional): Whether the show the figure after calling the method. Defaults to False.
            scale (float, optional): A scale factor for plotting the arrows (defaults to 1)
            normalize (bool, optional): if set to True, we normalize the magnetic field to have a max value of 1 before plotting

        Returns:
            ax: the figure axis in which the laser is plotted.
        """
        # ------------------------- START ARGUMENT CHECKING ----------------
        # - check plot config
        assert ax is None or isinstance(ax, Axes), "'ax' should be a matplotlib axis."
        # - check axis config
        # limits
        assert np.asanyarray(limits).size == 6, "`limits` should be an array of size 6"
        # Npoints
        Npoints = np.asanyarray(Npoints)
        msg = "`Npoints` should be an int or a list of three ints"
        assert Npoints.size in [1, 3], msg
        assert issubclass(Npoints.dtype.type, np.integer), msg
        # ------------------------- STOP ARGUMENT CHECKING ----------------
        # - init ax (if needed)
        ax = self._init_ax(ax, ax3D=True)
        # - generate grid
        xmin, xmax, ymin, ymax, zmin, zmax = limits
        Nx, Ny, Nz = (Npoints, Npoints, Npoints) if Npoints.size == 1 else Npoints
        grid = np.mgrid[
            xmin : xmax : Nx * 1j, ymin : ymax : Ny * 1j, zmin : zmax : Nz * 1j
        ]
        X, Y, Z = grid
        position = grid.T
        # - get magnetic field
        B = self.value(position)
        # - normalize ?
        if normalize:
            B = B / np.max(np.abs(B.ravel()))
        B = B * scale
        Bx, By, Bz = B.T
        # - plot
        ax.quiver(
            X,
            Y,
            Z,
            Bx,
            By,
            Bz,
            label=name,
            color=color,
        )
        # - axes names
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # - show
        if show:
            plt.show()
        return ax

    # -- USEFUL CHECK FUNCTIONS
    def _check_real_number(self, value, name):
        if np.asanyarray(value).size > 1:
            raise ValueError(f"'{name}' should be a scalar")
        if not np.isreal(value):
            raise TypeError(f"'{name}' should be a real numbers")
        return value

    def _check_3D_vector(self, value, name, norm=False):
        value = np.asanyarray(value)
        if value.size != 3:
            raise ValueError(f"'{name}' should be an array of size 3")
        if not np.all(np.isreal(value)):
            raise TypeError(f"'{name}' should be an array of real numbers")
        if norm:
            value = value / np.linalg.norm(value)
        return value


# % TOOL CLASSES


class AbstractOffsetField(AbstractField):
    """To generate perfect field offset"""

    def __init__(self, offset: npt.ArrayLike = (0, 0, 0)):
        """Generates a constant offset field

        Args:
            offset (npt.ArrayLike): offset of the field (array of size 3)
        """
        self.offset = offset
        super(AbstractField, self).__init__()

    # -- getters and setters

    @property
    def offset(self) -> npt.ArrayLike:
        return self.__offset

    @offset.setter
    def offset(self, value: npt.ArrayLike):
        self.__offset = self._check_3D_vector(value, "offset")

    # -- requested methods for AbstractField
    # pylint : disable=method_hidden
    @staticmethod
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `AbstractField` class
        """
        # 'position' already has the right size here
        # as it contains 3D vectors (position)
        # so we can generate an homogeneous field quite easily
        value = position * 0.0 + self.__offset
        return value

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "offset (constant field)")
        info.add_element(f"value ({unit})", f"{self.offset}")
        info.add_element(f"norm ({unit})", f"{np.linalg.norm(self.offset):.3g}")
        return info


class AbstractGradientField(AbstractField):
    """To generate perfect gradients"""

    def __init__(
        self,
        origin: npt.ArrayLike,
        slope: float,
        gradient_direction: npt.ArrayLike,
        field_direction: npt.ArrayLike,
        offset: float = 0.0,
    ):
        """Abstract Gradient

        See below for arguments.

        Note that 'gradient_direction' and 'field_direction' are meant to be
        unit vectors, but the class will take care of normalizing any non normalized entry


        Args:
            origin (npt.ArrayLike): origin for the gradient (array of size 3)
            slope (float): the slope of the gradient (scalar)
            gradient_direction (npt.ArrayLike): the direction of the gradient (array of size 3)
            field_direction (npt.ArrayLike): the field direction (array of size 3)
            offset (float, optional): the field offset, at origin (scalar). Defaults to 0.0.
        """
        self.slope = slope
        self.offset = offset
        self.origin = origin
        self.gradient_direction = gradient_direction
        self.field_direction = field_direction
        super(AbstractGradientField, self).__init__()

    # -- value
    # pylint : disable=method_hidden
    @staticmethod
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `AbstractField` class
        """
        # - get X, Y, and Z
        x, y, z = position.T
        x, y, z = x.T, y.T, z.T

        # - get gradient vector angles
        theta = self.__gradient_theta
        phi = self.__gradient_phi

        # - get coordinates w.r.t origin
        x0, y0, z0 = self.origin
        xc = x - x0
        yc = y - y0
        zc = z - z0

        # compute coordinates in rotated frame
        # we want z
        z_rot = (
            xc * np.sin(theta) * np.cos(phi)
            + yc * np.sin(theta) * np.sin(phi)
            + zc * np.cos(theta)
        )

        # - value at position
        value = (self.offset + z_rot * self.slope)[
            ..., np.newaxis
        ] * self.field_direction

        return value

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "perfect gradient")
        info.add_element(f"slope ({unit}/m)", f"{self.slope:.3g}")
        info.add_element("gradient direction", f"{self.gradient_direction}")
        info.add_element("field direction", f"{self.field_direction}")
        info.add_element(f"origin (m)", f"{self.origin}")
        info.add_element(f"offset ({unit})", f"{self.offset:3g}")
        return info

    # -- getters and setters
    # -
    @property
    def slope(self) -> npt.ArrayLike:
        return self.__slope

    @slope.setter
    def slope(self, value: npt.ArrayLike):
        self.__slope = self._check_real_number(value, "slope")

    # -
    @property
    def offset(self) -> npt.ArrayLike:
        return self.__offset

    @offset.setter
    def offset(self, value: npt.ArrayLike):
        self.__offset = self._check_real_number(value, "offset")

    # -
    @property
    def origin(self) -> npt.ArrayLike:
        return self.__origin

    @origin.setter
    def origin(self, value: npt.ArrayLike):
        self.__origin = self._check_3D_vector(value, "origin")

    # -
    @property
    def gradient_direction(self) -> npt.ArrayLike:
        return self.__gradient_direction

    @gradient_direction.setter
    def gradient_direction(self, value: npt.ArrayLike):
        value = self._check_3D_vector(value, "gradient_direction", norm=True)
        assert np.allclose(
            np.linalg.norm(value), 1
        ), "We did not manage to normalize gradient_direction, something is fishy.."
        # compute angles
        ux, uy, uz = value
        theta = np.arctan2(np.sqrt(ux**2 + uy**2), uz)
        phi = np.arctan2(uy, ux)
        self.__gradient_direction = value
        self.__gradient_theta = theta
        self.__gradient_phi = phi

    # -
    @property
    def field_direction(self) -> npt.ArrayLike:
        return self.__field_direction

    @field_direction.setter
    def field_direction(self, value: npt.ArrayLike):
        self.__field_direction = self._check_3D_vector(
            value, "field_direction", norm=True
        )
