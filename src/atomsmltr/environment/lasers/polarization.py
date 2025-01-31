# -*- coding: utf-8 -*-
"""Defines special classes and functions relative to laser polarization
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt


# % GLOBAL DEFINITIONS

POLARIZATION_TYPES_LONGNAMES = [
    "vertical",
    "horizontal",
    "linear",
    "circular left",
    "circular right",
    "vector",
]

POLARIZATION_TYPES_SHORTNAMES = ["V", "x", "H", "y", "lin", "R", "L", "vec"]

POLARIZATION_TYPES = POLARIZATION_TYPES_LONGNAMES + POLARIZATION_TYPES_SHORTNAMES
POLARIZATION_TYPES_UPPER = [type.upper() for type in POLARIZATION_TYPES]

# % CLASS


class Polarization(object):
    """Handles laser polarization"""

    def __init__(self, type: str, angle: float = None, vec: npt.ArrayLike = None):
        """An object to handle laser polarization.

        We define the polarization in the frame of the laser, with laser propagation along z.
        We denote x as the 'horizontal' axis and y as the 'vertical'axis.
        For circular polarizations, we take the observer convention.

        To have a combined formalism for all polarization, in the end we define a polarization vector `p_vec`,
        following the Poincarré formalism. ATTENTION: there might be different ways of defining this vector,
        refer to the package documentation for a thorough definition.

        Polarization `type`can take several values:

        1) special polarizations : 'vertical', 'horizontal', 'circular left', 'circular right'
           and corresponding shorthands: 'V' or 'x', 'H' or 'y', 'R', 'L'

        2) generic polarizations : 'linear' (short 'lin') or 'vector' (short 'vec')

        in the case of 'linear' polarization, an additionnal argument `angle` has to be provided, that gives
        the angle of the linear polarization with respect to the x axis. Hence `angle = 0` corresponds to a
        linear polarization along x, and `angle = pi/2` to a linear polarization along y

        in the case of 'vector' polarization, polarization vector has to be given with the `vec` argument. `vec`
        are the cartesian coordinates of the vector in the (x,y,z) basis. For instance :

        > vec = (1, 0, 0)  : linear polarization along x
        > vec = (0, 1, 0)  : linear polarization along y
        > vec = (0, 0, 1)  : circular right polarization
        > vec = (0, 0, -1) : circular left polarization

        ATTENTION : for ease of use, the vector does not have to be normalized, but the resulting one will
        be.

        Args:
            type (str): the type of polarization to de defined (see docstring)
            angle (float, optional): when type is set to 'linear' (short 'lin'), defines the orientation of the
                                     linear polarization in the (x, y). Defaults to None.
            vec (array, optional): needed when type is set to 'vector' (short 'vec'). Allows for a direct
                                   definition of the polarization vector (see docstring & documentation)
                                   Defaults to None
        """
        self.type = type
        self.angle = angle
        self.vec = vec

    # -- METHODS
    def get_polarization_vector(self):
        """Returns the polarization vector describing the current polarization state.

        See documentation for the exact definition of the vector. In short :

        > p_vec = (1, 0, 0)  : linear polarization along x (vertical)
        > p_vec = (0, 1, 0)  : linear polarization along y (horizontal)
        > p_vec = (0, 0, 1)  : circular right polarization
        > p_vec = (0, 0, -1) : circular left polarization

        Returns:
            p_vec: numpy array of size 3, containing the cartesian coordinates of the polarization vector
        """
        # get the polarization type once and for all
        pol_type = self.type.upper()
        # some sanity checks
        if pol_type == "VECTOR":
            try:
                assert self.vec.size == 3
            except:
                raise ValueError(
                    "The 'vec' property should be a vector of size 3 for 'vector' type"
                )
        if pol_type == "LINEAR" and not (
            isinstance(self.angle, int) or isinstance(self.angle, float)
        ):
            raise ValueError("The 'angle' property should be a float for 'linear' type")

        # compute vector
        match self._type.upper():
            case "VERTICAL":
                p_vec = (1, 0, 0)
            case "HORIZONTAL":
                p_vec = (0, 1, 0)
            case "LINEAR":
                p_vec = (np.cos(self.angle), np.sin(self.angle), 0)
            case "CIRCULAR LEFT":
                p_vec = (0, 0, -1)
            case "CIRCULAR RIGHT":
                p_vec = (0, 0, 1)
            case "VECTOR":
                p_vec = self.vec

        p_vec = np.array(p_vec)
        p_vec = p_vec / np.linalg.norm(p_vec)

        return p_vec

    def get_polarization_vector_angles(self):
        """Gives the angles describing the current polarization vector.

        (see documentation for thorough description)

        The polarization is decribed in the Poincarré/Bloch-like sphere as a vector.
        This function yields the angles u (polar) and v (azimuthal)

        Note that we do not use theta or phi as those angles are already used to
        describe the orientation of the laser propagation vector in the `LaserBeam`class

        Returns:
            u (float): the u angle (polar angle)
            v (float): the v angle (azimuthal angle)
        """
        x, y, z = self.get_polarization_vector()
        u = np.arctan2(np.sqrt(x**2 + y**2), z)
        v = np.arctan2(y, x)
        return u, v

    # -- CLASS PROPERTIES GETTERS & SETTERS
    # - type
    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        # check that we have a string
        if not isinstance(value, str):
            raise ValueError("Wrong type provided for `type`, we expect a string.")

        # check that the type is implemented
        if value.upper() not in POLARIZATION_TYPES_UPPER:
            msg = f"Wrong value '{value}' for polarization type.\n"
            msg += f"Available types are : {POLARIZATION_TYPES}"
            raise ValueError(msg)

        # convert
        if value.upper() in ["V", "X"]:
            value = "vertical"
        elif value.upper() in ["H", "Y"]:
            value = "horizontal"
        elif value.upper() == "R":
            value = "circular right"
        elif value.upper() == "L":
            value = "circular left"
        elif value.upper() == "LIN":
            value = "linear"
        elif value.upper() == "VEC":
            value = "vector"

        assert value.upper() in [t.upper() for t in POLARIZATION_TYPES_LONGNAMES]

        self._type = value.upper()

    # - angle
    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        """The angle parameter is only needed for 'linear' polarization
        type, so we adapt the checking method.
        """
        # convert int into float
        if isinstance(value, int):
            value = float(value)

        if not isinstance(value, float) and value is not None:
            raise ValueError("Angle must be a float or None")

        if self.type == "LINEAR" and not isinstance(value, float):
            msg = "You have to provide a float value for angle when "
            msg += "polarization type is 'linear'"
            raise ValueError(msg)

        if self.type != "LINEAR" and value is not None:
            msg = "Polarization type is not 'linear' but angle"
            msg += " is not None. Current value of angle won't be used in"
            msg += " the polarization definition !!!"
            raise Warning(msg)

        self._angle = value

    # - vector
    @property
    def vec(self) -> npt.ArrayLike:
        return self._vec

    @vec.setter
    def vec(self, value: npt.ArrayLike) -> None:
        # if 'VECTOR' type is selected
        if self.type == "VECTOR":
            # convert to array
            value = np.asanyarray(value)
            if value.size != 3:
                raise ValueError("'vec' should be of size 3")
            # normalize
            norm = np.linalg.norm(value)
            if norm == 0:
                raise ValueError("Wrong value for 'vec'': norm is zero")
            self._vec = value / norm
        # compute unit vector
        else:
            if value is not None:
                msg = "Polarization type is not 'vector' but 'vec'"
                msg += " is not None. Current value of 'vec' won't be used in"
                msg += " the polarization definition !!!"
                raise Warning(msg)
            self._vec = value
