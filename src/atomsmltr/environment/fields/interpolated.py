"""
interpolated fields
=======================

This module implements the generic ``InterpolatedField`` class

See also
---------
atomsmltr.environment.fields.magnetic
"""

# % IMPORTS
import numpy as np
from abc import abstractmethod

# % LOCAL IMPORTS
from .generic import Field
from ...utils.infostring import InfoString

# % CLASS


class InterpolatedField(Field):
    """Creates an interpolated field from input values.

    Parameters
    ----------
    data_x : array
        the 'position' data of the field to interpolate
    data_y : array
        the 'value' data of the field to interpolate
    origin : array, shape (3,), optional
        cartesian coordinates of the origin for the field.
        If not set to (0,0,0), it will shift the interpolated field
        w.r.t the initial data, by default (0, 0, 0)
    scale : float, optional
        a scale factor for the interpolated field, by default 1.0
    tag : str, optional
        the field tag, by default None
    """

    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        origin: np.ndarray = (0, 0, 0),
        scale: float = 1.0,
        tag: str = None,
    ):

        self.origin = origin
        self.scale = scale
        self.interpolate(data_x, data_y)
        super(InterpolatedField, self).__init__(tag)

    # -- requested method
    @abstractmethod
    def interpolate(self, data_x: np.ndarray, data_y: np.ndarray):
        """(re)initialize the interpolation

        Parameters
        ----------
        data_x : array
            the 'position' data of the field to interpolate
        data_y : array
            the 'value' data of the field to interpolate
        """
        pass

    @abstractmethod
    def _interp_fun(self, x: np.ndarray):
        """result of interpolation, has to be assigned via 'interpolate'"""
        pass

    # -- requested methods for Field
    # pylint : disable=method_hidden
    def _field_value_func(self, position):
        """Returns field value at point position

        position should be an array of shape (3,) or (n1,n2,..,3)
        last axis contains coordinates x, y, z

        NB: position is already checked and converted to an array in the
            `Field` class
        """
        # most of the work is done by the '__interp_fun' method
        # here we only translate and scale
        translated_pos = position - self.origin
        value = self._interp_fun(translated_pos) * self.scale
        return value

    # -- getters and setters
    # -
    @property
    def origin(self) -> np.ndarray:
        """array, shape (,3): the origin for the interpolated field"""
        return self.__origin

    @origin.setter
    def origin(self, value: np.ndarray):
        self.__origin = self._check_3D_vector(value, "origin")

    # -
    @property
    def scale(self) -> float:
        """float: a scale factor for the field"""
        return self.__scale

    @scale.setter
    def scale(self, value: float):
        self.__scale = self._check_real_number(value, "scale")


class InterpolatedField1D1D(InterpolatedField):
    """Creates an interpolated 1D field from 1D input values.

    Parameters
    ----------
    data_x : array, shape (,n)
        the 'position' data of the field to interpolate
    data_y : array, shape (,n)
        the amplitude of the field to interpolate
    field_direction : array, shape (,3)
        the direction to which the interpolated field will point at
    x_direction : array, shape (,3)
        the direction corresponding to the data_x 'position' array
    origin : array, shape (3,), optional
        cartesian coordinates of the origin for the field.
        If not set to (0,0,0), it will shift the interpolated field
        w.r.t the initial data, by default (0, 0, 0)
    scale : float, optional
        a scale factor for the interpolated field, by default 1.0
    tag : str, optional
        the field tag, by default None
    """

    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        field_direction: np.ndarray = (1, 0, 0),
        x_direction: np.ndarray = (1, 0, 0),
        origin: np.ndarray = (0, 0, 0),
        scale: float = 1.0,
        tag: str = None,
    ):
        self.field_direction = field_direction
        self.x_direction = x_direction

        super(InterpolatedField1D1D, self).__init__(
            data_x=data_x,
            data_y=data_y,
            origin=origin,
            scale=scale,
            tag=tag,
        )

    # -- Interp fun
    def _interp_fun(self, position: np.ndarray) -> np.ndarray:
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
        theta = self.__xdir_theta
        phi = self.__xdir_phi

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
        value = self.__fun(z_rot)
        value = value[..., np.newaxis] * self.field_direction
        return value

    # -- GETTERS
    # -
    @property
    def data_x(self) -> np.ndarray:
        """array: x data used for interpolation"""
        return self.__data_x

    # -
    @property
    def data_y(self) -> np.ndarray:
        """array: y data used for interpolation"""
        return self.__data_y

    # -
    @property
    def field_direction(self) -> np.ndarray:
        """array, shape (3,): direction of the field"""
        return self.__field_direction

    @field_direction.setter
    def field_direction(self, value: np.ndarray):
        value = self._check_3D_vector(value, "field_direction", norm=True)
        self.__field_direction = value

    # -
    @property
    def x_direction(self) -> np.ndarray:
        """array, shape (,3) : the direction corresponding to the data_x 'position' array"""
        return self.__x_direction

    @x_direction.setter
    def x_direction(self, value: np.ndarray):
        value = self._check_3D_vector(value, "x_direction", norm=True)
        # compute angles
        ux, uy, uz = value
        theta = np.arctan2(np.sqrt(ux**2 + uy**2), uz)
        phi = np.arctan2(uy, ux)
        self.__xdir_theta = theta
        self.__xdir_phi = phi
        self.__x_direction = value

    # -- INTERPOLATE
    def interpolate(self, data_x, data_y):
        # -- check data
        # squeeze
        data_x = np.squeeze(data_x)
        data_y = np.squeeze(data_y)
        # dimension
        msg = "'data_x' and 'data_y' should be 1D arrays with same size"
        if data_x.ndim > 1 or data_y.ndim > 1:
            raise ValueError(msg)
        if data_x.shape != data_y.shape:
            raise ValueError(msg)
        # sort
        i_sort = np.argsort(data_x)
        data_x = data_x[i_sort]
        data_y = data_y[i_sort]

        # -- store
        self.__data_x = data_x
        self.__data_y = data_y

    def __fun(self, x):
        return np.interp(x, self.__data_x, self.__data_y)

    def gen_infostring_obj(self):
        """Generates an info string object"""
        unit = self.unit
        title = self.type
        title = title[:1].upper() + title[1:]  # capitalize first letter
        info = InfoString(title=title)
        info.add_section("Parameters")
        info.add_element("type", "interpolated 1D-1D field")
        info.add_element("tag", self.tag)
        info.add_element("origin (m)", f"{self.origin}")
        info.add_element("field direction", f"{self.field_direction}")
        info.add_element("x direction", f"{self.x_direction}")
        info.add_element("scale", f"{self.scale:.3g}")
        info.add_element("x start", f"{self.data_x[0]:.3g}")
        info.add_element("x stop", f"{self.data_x[-1]:.3g}")

        return info
