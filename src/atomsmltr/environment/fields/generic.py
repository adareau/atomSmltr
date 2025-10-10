"""
generic fields
=======================

This module implements the generic ``Field`` class, and some perfect fields
(Offset, Gradient, Quadrupole). Those are abstract classes, and actual implementations
are gathered in other packages

See also
---------
atomsmltr.environment.fields.magnetic
"""

# % IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from abc import abstractmethod

# % LOCAL IMPORTS
from ..envbase import EnvObject
from ...utils.infostring import InfoString

# % ABSTRACT CLASSES


class Field(EnvObject):
    """A generic class to describe Field objects."""

    def __init__(self, *args, **kwargs):
        super(Field, self).__init__(*args, **kwargs)

    @property
    def vector(self):
        return True

    @property
    @abstractmethod
    def unit():
        """str: returns the unit of the field"""
        pass

    def get_value(self, position: np.ndarray) -> np.ndarray:
        """returns the value of the field at a given position.

        Parameters
        ----------
        position : np.ndarray, shape (,3) or (n1, n2, .., 3)
            positions at which the intensity is computed, given as cartesian coordinates
            in the lab frame.

        Returns
        -------
        value : np.ndarray, same shape as position
            returns the (vector) field value

        Notes
        -------
        `position` is an array_like object, with shape (3,) or (n1, n2, .., 3).
        In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame

        The field value is returned as an array with the same shape as `position`.

        >>> field_value = field.get_value(position)
        >>> X, Y, Z = position.T
        >>> Fx, Fy, Fz = field_value.T
        """
        # Check position
        position = self._check_position_array(position)
        # call hidden function that actually does the computation
        return self._field_value_func(position)

    @abstractmethod
    def _field_value_func(self, position):
        """Actual method for field computation ; defined for each subclass"""

    def get_norm(self, position: np.ndarray) -> np.ndarray:
        """Returns the field norm at a given position in the lab frame

        Parameters
        ----------
        position : np.ndarray, shape (,3) or (n1, n2, .., 3)
            positions at which the intensity is computed, given as cartesian coordinates
            in the lab frame.

        Returns
        -------
        norm : np.ndarray, shape (,1) or (n1, n2, .., 1)
            returns the (scalar) field norm
        """
        F = self.get_value(position)
        Fx, Fy, Fz = F.T
        F_norm = np.sqrt(Fx**2 + Fy**2 + Fz**2).T
        return F_norm

    def plot1D(
        self,
        start: np.ndarray,
        stop: np.ndarray,
        Npoints: int = 100,
        component: str = "Bz",
        ax=None,
        show: bool = False,
        space_scale: float = 1.0,
    ):
        """Plots a 1D line cut of the magnetic field using Matplotlib

        Parameters
        ----------
        start : array-like, shape (3,)
            Starting point (x, y, z) of the line along which to sample the field.
        stop : array-like, shape (3,)
            Ending point (x, y, z) of the line along which to sample the field.
        Npoints : int, optional
            Number of points sampled along the line. Defaults to 100.
        component : str, optional
            Field component to plot. Accepted values are "Bx", "By", "Bz" for vector components,
            or "B" for total field magnitude. Defaults to "Bz".
        ax : Matplotlib Axes, optional
            The matplotlib axis on which to plot.
            If None is given, a new figure is created. Defaults to None.
        show : bool, optional
            Whether to show the figure after calling the method. Defaults to False.
        space_scale : float, optional
            Space coordinates will be multiplied by this when plotting. Defaults to 1.

        Returns
        -------
        ax : Matplotlib Axes
            The axis on which the plot was performed.

        Notes
        ------
        The field is sampled along a straight line between `start` and `stop` using
        `Npoints` equally spaced positions. The field component or magnitude is computed
        and plotted as a function of the distance along the line.

        Examples
        ---------
        >>> field.plot1D(start=[0, 0, -10], stop=[0, 0, 10], Npoints=200)
        >>> field.plot1D(start=[0, 0, -5], stop=[5, 0, 0], component="B", show=True)
        >>> field.plot1D(start=[-5, 0, 0], stop=[5, 0, 0], component="Bx", space_scale=1e-3)

        """
        # Create points along the line
        start = np.array(start)
        stop = np.array(stop)
        t = np.linspace(0, 1, Npoints)
        points = start[None, :] + t[:, None] * (stop - start)[None, :]

        # Compute field
        B = self.get_value(points)

        # Choose component
        if component == "Bx":
            values = B[:, 0]
        elif component == "By":
            values = B[:, 1]
        elif component == "Bz":
            values = B[:, 2]
        elif component == "B":
            values = np.linalg.norm(B, axis=1)
        else:
            raise ValueError(
                f"Unknown component '{component}'. Choose 'Bx', 'By', 'Bz', or 'B'."
            )

        # Compute position along the line (scaled if needed)
        distances = np.linspace(0, np.linalg.norm(stop - start), Npoints) * space_scale

        # Plot
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(distances, values, label=component)
        ax.set_xlabel("Distance along line")
        ax.set_ylabel(f"Magnetic field [{component}]")
        ax.set_title(f"Field along 1D line from {start} to {stop}")
        ax.grid(True)
        ax.legend()

        if show:
            plt.show()

        return ax

    def plot2D(
        self,
        limits: np.ndarray,
        Npoints: np.ndarray,
        cut: float = 0.0,
        ax=None,
        plane: str = "XY",
        cmap=None,
        show: bool = False,
        space_scale: float = 1.0,
    ):
        """Plots a 2D cut of the field, using Matplotlib streamplot()

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
            passed to matplotlib streamplot() function
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
        >>> field.plot2D(limits=(-5, 5, -4, 4), Npoints=200)
        >>> field.plot2D(limits=(-5, 5, -4, 4), Npoints=200, cut=-5)
        >>> field.plot2D(limits=(-5, 5, -4, 4), Npoints=(200, 100))

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
        mag_field = self.get_value(position)
        Bx, By, Bz = mag_field.T
        Bx = Bx.T
        By = By.T
        Bz = Bz.T
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
        """plots a 3D representation of the field, using Matplotlib quiver()

        Parameters
        ----------
        limits : array, shape (,6)
            limits for the plot (xmin, xmax, ymin, ymax, zmin, zmax)
        Npoints : int or array of shape (,3)
            number of points for each dimension,
            either a int or an array of three ints (Nx, Ny, Nz)
        ax : custom Axes3D, optional
            the axis in which to plot.
            If None is given (default value) a new ax is generated
        color : str, optional
            a matplotlib compatible color. Defaults to None.
        name : str, optional
            the name of the field, passed as a label when plotting. Defaults to None.
        show : bool, optional
            whether the show the figure after calling the method. Defaults to False.
        scale : float, optional
            a scale factor for plotting the arrows (defaults to 1)
        normalize : bool, optional
            if set to True, we normalize the magnetic field to have a max value of 1 before plotting.
            Defaults to False

        Returns
        -------
        ax : Matplotlib Axes
            the axis on which the plot was performed.

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
        B = self.get_value(position)
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


class ConstantField(Field):
    """Generates a constant field

    Parameters
    ----------
    field_value : np.ndarray, shape (3,), optional
        Constant field value, by default (0, 0, 0)
    tag : str, optional
        Field tag, by default None
    """

    def __init__(self, field_value: np.ndarray = (0, 0, 0), tag: str = None):
        self.field_value = field_value
        super(Field, self).__init__(tag)

    # -- getters and setters

    @property
    def field_value(self) -> np.ndarray:
        return self.__field_value

    @field_value.setter
    def field_value(self, value: np.ndarray):
        """field_value (array): constant field value"""
        self.__field_value = self._check_3D_vector(value, "value")

    # -- requested methods for Field
    # pylint : disable=method_hidden
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # 'position' already has the right size here
        # as it contains 3D vectors (position)
        # so we can generate an homogeneous field quite easily
        value = position * 0.0 + self.__field_value
        return value

    def gen_infostring_obj(self):
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "constant field")
        info.add_element("tag", self.tag)
        info.add_element(f"field_value ({unit})", f"{self.field_value}")
        info.add_element(f"norm ({unit})", f"{np.linalg.norm(self.field_value):.3g}")
        return info


class GradientField(Field):
    """A perfect linear field gradient

    Parameters
    ----------
    origin : array, shape (,3)
        origin for the gradient, in cartesian coordinates in the lab frame
    slope : float
        the slope of the gradient, in 'field unit' / m
    gradient_direction : array, shape (.3)
        the direction of the gradient
    field_direction : array, shae (,3)
        the direction of the field
    offset : float, optional
        an offset for the field amplitude at origin, in 'field unit', by default 0.0
    tag : str, optional
        field tag, by default None

    Notes
    -----
        Note that 'gradient_direction' and 'field_direction' are meant to be
        unit vectors, but the class will take care of normalizing any non normalized entry

    Examples
    ---------

    Create a field pointing along z, with a amplitude increasing linearly along x

    >>> gradient = GradientField(
    ...  origin=(0, 0, 0),
    ...  slope=1,
    ...  gradient_direction=(1, 0, 0),
    ...  field_direction=(0, 0, 1),
    ...  tag="gradient",
    ... )

    """

    def __init__(
        self,
        origin: np.ndarray,
        slope: float,
        gradient_direction: np.ndarray,
        field_direction: np.ndarray,
        offset: float = 0.0,
        tag: str = None,
    ):
        self.slope = slope
        self.offset = offset
        self.origin = origin
        self.gradient_direction = gradient_direction
        self.field_direction = field_direction
        super(GradientField, self).__init__(tag)

    # -- value
    # pylint : disable=method_hidden
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
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
        info.add_element("tag", self.tag)
        info.add_element(f"slope ({unit}/m)", f"{self.slope:.3g}")
        info.add_element("gradient direction", f"{self.gradient_direction}")
        info.add_element("field direction", f"{self.field_direction}")
        info.add_element(f"origin (m)", f"{self.origin}")
        info.add_element(f"offset ({unit})", f"{self.offset:3g}")
        return info

    # -- getters and setters
    # -
    @property
    def slope(self) -> float:
        """float: the field amplitude gradient slope"""
        return self.__slope

    @slope.setter
    def slope(self, value: float):
        self.__slope = self._check_real_number(value, "slope")

    # -
    @property
    def offset(self) -> float:
        """float: the field amplitude value at origin"""
        return self.__offset

    @offset.setter
    def offset(self, value: float):
        self.__offset = self._check_real_number(value, "offset")

    # -
    @property
    def origin(self) -> np.ndarray:
        """array, shape (,3): the origin for the gradient"""
        return self.__origin

    @origin.setter
    def origin(self, value: np.ndarray):
        self.__origin = self._check_3D_vector(value, "origin")

    # -
    @property
    def gradient_direction(self) -> np.ndarray:
        """array, shape (,3): the gradient direction"""
        return self.__gradient_direction

    @gradient_direction.setter
    def gradient_direction(self, value: np.ndarray):
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
    def field_direction(self) -> np.ndarray:
        """array, shape (,3): the field direction"""
        return self.__field_direction

    @field_direction.setter
    def field_direction(self, value: np.ndarray):
        self.__field_direction = self._check_3D_vector(
            value, "field_direction", norm=True
        )


class BaseQuadrupoleField(Field):
    """A perfect quadrupole field

    Parameters
    ----------
    origin : array, shape (,3)
        origin for the quadrupole, in cartesian coordinates in the lab frame
    strong_axis : array, shape (,3)
        the direction of the strong axis
    slope : float
        the gradient of the weak axes
    tag : str, optional
        field tag, by default None

    Notes
    ------

    Note that 'strong_axis' is meant to be a unit vector, but the class will take care of normalizing
    any non normalized entry

    """

    def __init__(
        self,
        origin: np.ndarray,
        strong_axis: np.ndarray,
        slope: float,
        tag: str = None,
    ):
        self.slope = slope
        self.origin = origin
        self.strong_axis = strong_axis

        super(BaseQuadrupoleField, self).__init__(tag)

    # -- getters and setters
    # -
    @property
    def slope(self) -> float:
        """float: the gradient of the weak axes"""
        return self.__slope

    @slope.setter
    def slope(self, value: float):
        self.__slope = self._check_real_number(value, "slope")

    # -
    @property
    def origin(self) -> np.ndarray:
        """array, shape (3,): the origin of the quadrupole"""
        return self.__origin

    @origin.setter
    def origin(self, value: np.ndarray):
        self.__origin = self._check_3D_vector(value, "origin")

    # -
    @property
    def strong_axis(self) -> np.ndarray:
        """array, shape (3,): the direction of the strong axis"""
        return self.__strong_axis

    @strong_axis.setter
    def strong_axis(self, value: np.ndarray):
        value = self._check_3D_vector(value, "strong_axis", norm=True)
        assert np.allclose(
            np.linalg.norm(value), 1
        ), "We did not manage to normalize strong_axis, something is fishy.."
        # compute angles
        ux, uy, uz = value
        theta = np.arctan2(np.sqrt(ux**2 + uy**2), uz)
        phi = np.arctan2(uy, ux)
        self.__strong_axis = value
        self.__theta = theta
        self.__phi = phi

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "perfect quadrupole")
        info.add_element("tag", self.tag)
        info.add_element(f"slope ({unit}/m)", f"{self.slope:.3g}")
        info.add_element("strong axis", f"{self.strong_axis}")
        info.add_element(f"origin (m)", f"{self.origin}")
        return info


class QuadrupoleFieldX(BaseQuadrupoleField):
    """A perfect quadrupole field with strong axis along X

    Parameters
    ----------
    origin : array, shape (,3)
        origin for the quadrupole, in cartesian coordinates in the lab frame
    slope : float
        the gradient of the weak axes
    tag : str, optional
        field tag, by default None

    Notes
    ------

    Note that 'strong_axis' is meant to be a unit vector, but the class will take care of normalizing
    any non normalized entry

    """

    def __init__(
        self,
        origin: np.ndarray,
        slope: float,
        tag: str = None,
    ):
        super(QuadrupoleFieldX, self).__init__(
            origin=origin, slope=slope, tag=tag, strong_axis=(1, 0, 0)
        )

    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # - get X, Y, and Z
        x, y, z = position.T

        # - get coordinates w.r.t origin
        x0, y0, z0 = self.origin
        xc, yc, zc = x - x0, y - y0, z - z0

        # - compute
        slope = self.slope
        value = np.array([-2 * slope * xc, slope * yc, slope * zc]).T

        return value


class QuadrupoleFieldY(BaseQuadrupoleField):
    """A perfect quadrupole field with strong axis along Y

    Parameters
    ----------
    origin : array, shape (,3)
        origin for the quadrupole, in cartesian coordinates in the lab frame
    slope : float
        the gradient of the weak axes
    tag : str, optional
        field tag, by default None

    Notes
    ------

    Note that 'strong_axis' is meant to be a unit vector, but the class will take care of normalizing
    any non normalized entry

    """

    def __init__(
        self,
        origin: np.ndarray,
        slope: float,
        tag: str = None,
    ):
        super(QuadrupoleFieldY, self).__init__(
            origin=origin, slope=slope, tag=tag, strong_axis=(0, 1, 0)
        )

    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # - get X, Y, and Z
        x, y, z = position.T

        # - get coordinates w.r.t origin
        x0, y0, z0 = self.origin
        xc, yc, zc = x - x0, y - y0, z - z0

        # - compute
        slope = self.slope
        value = np.array([slope * xc, -2 * slope * yc, slope * zc]).T

        return value


class QuadrupoleFieldZ(BaseQuadrupoleField):
    """A perfect quadrupole field with strong axis along Z

    Parameters
    ----------
    origin : array, shape (,3)
        origin for the quadrupole, in cartesian coordinates in the lab frame
    slope : float
        the gradient of the weak axes
    tag : str, optional
        field tag, by default None

    Notes
    ------

    Note that 'strong_axis' is meant to be a unit vector, but the class will take care of normalizing
    any non normalized entry

    """

    def __init__(
        self,
        origin: np.ndarray,
        slope: float,
        tag: str = None,
    ):
        super(QuadrupoleFieldZ, self).__init__(
            origin=origin, slope=slope, tag=tag, strong_axis=(0, 0, 1)
        )

    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # - get X, Y, and Z
        x, y, z = position.T

        # - get coordinates w.r.t origin
        x0, y0, z0 = self.origin
        xc, yc, zc = x - x0, y - y0, z - z0

        # - compute
        slope = self.slope
        value = np.array([slope * xc, slope * yc, -2 * slope * zc]).T

        return value


class QuadrupoleField(BaseQuadrupoleField):
    """A perfect quadrupole field with strong axis along a given vector

    Parameters
    ----------
    origin : array, shape (,3)
        origin for the quadrupole, in cartesian coordinates in the lab frame
    strong_axis : array, shape (,3)
        the direction of the strong axis
    slope : float
        the gradient of the weak axes
    tag : str, optional
        field tag, by default None

    Notes
    ------

    Note that 'strong_axis' is meant to be a unit vector, but the class will take care of normalizing
    any non normalized entry

    """

    def __init__(
        self,
        origin: np.ndarray,
        strong_axis: np.ndarray,
        slope: float,
        tag: str = None,
    ):
        super(QuadrupoleField, self).__init__(
            origin=origin, slope=slope, tag=tag, strong_axis=strong_axis
        )

    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # - get X, Y, and Z
        x, y, z = np.moveaxis(position, -1, 0)

        # - get coordinates w.r.t origin
        x0, y0, z0 = self.origin
        xc, yc, zc = x - x0, y - y0, z - z0
        rc = np.stack((xc, yc, zc), axis=-1)

        # - compute
        slope = self.slope
        s = self.strong_axis / np.linalg.norm(self.strong_axis)
        projection = np.sum(rc * s, axis=-1, keepdims=True)

        value = slope * projection * s - (slope / 2) * (rc - projection * s)

        return value
