# -*- coding: utf-8 -*-
"""Defines special classes and functions relative to laser polarization
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

# % GLOBAL DEFINITIONS


# % ABSTRACT CLASS


class AbstractPolarization(ABC):
    """Handles laser polarization"""

    def __init__(self):
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
        # we keep this method public to display the docstring
        # and hide the calculation in a private method that has to be implemented for each
        # polarization class
        return self._get_polarization_vector()

    @abstractmethod
    def _get_polarization_vector(self):
        """Has to be implemented for each specifica class.
        See `get_polarization_vector()`public method for more information
        """

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

    def get_polarization_vector_projection(self, target: str):
        """Returns the scalar projection of the current polarization vector on a target polarization state

        The polarization Psi is defined as :

            |Psi⟩ = exp(-i*v) cos(u/2) |R⟩ +  exp(i*v) sin(u/2) |L⟩

        with |R⟩, |L⟩ the right- and left-handed circular polarization states. We also have

            |x⟩ = |V⟩ = (1/sqrt(2)) (|L⟩ + |R⟩)
            |y⟩ = |H⟩ = (i/sqrt(2)) (|L⟩ - |L⟩)

        Target should refer to the special polarization states defined in the class :
            'vertical', 'horizontal', 'circular left', 'circular right'

        and corresponding shorthands:
            'V' or 'x', 'H' or 'y', 'R', 'L'

        Args:
            target (str): the state on which to project (see docstring)

        Returns:
            proj (float, complex): the projection
        """
        # get angle values
        u, v = self.get_polarization_vector_angles()
        # common calculations
        A = np.exp(-1j * v) * np.cos(u / 2)
        B = np.exp(1j * v) * np.sin(u / 2)
        # return projection on desired vector
        match target.upper():
            case "V" | "X" | "VERTICAL":
                proj = (A + B) / np.sqrt(2)
            case "H" | "Y" | "HORIZONTAL":
                proj = 1j * (A - B) / np.sqrt(2)
            case "R" | "CIRCULAR RIGHT":
                proj = A
            case "L" | "CIRCULAR LEFT":
                proj = B
            case _:
                GOOD = ["vertical", "horizontal", "circular left", "circular right"]
                raise ValueError(f"Wrong value for target state, shoud be in {GOOD}")

        return proj

    def get_polarization_vector_projection_norm(self, target: str):
        """Returns the squared norm of scalar projection of the current polarization vector on a target polarization state

        The polarization Psi is defined as :

            |Psi⟩ = exp(-i*v) cos(u/2) |R⟩ +  exp(i*v) sin(u/2) |L⟩

        with |R⟩, |L⟩ the right- and left-handed circular polarization states. We also have

            |x⟩ = |V⟩ = (1/sqrt(2)) (|L⟩ + |R⟩)
            |y⟩ = |H⟩ = (i/sqrt(2)) (|L⟩ - |L⟩)

        Target should refer to the special polarization states defined in the class :
            'vertical', 'horizontal', 'circular left', 'circular right'

        and corresponding shorthands:
            'V' or 'x', 'H' or 'y', 'R', 'L'

        Args:
            target (str): the state on which to project (see docstring)

        Returns:
            norm (float, real): the squared norm of the projection
        """
        # get angle values
        u, v = self.get_polarization_vector_angles()
        # return projection on desired vector
        match target.upper():
            case "V" | "X" | "VERTICAL":
                norm = 0.5 * (1 + 2 * np.cos(u / 2) * np.sin(u / 2) * np.cos(2 * v))
            case "H" | "Y" | "HORIZONTAL":
                norm = 0.5 * (1 - 2 * np.cos(u / 2) * np.sin(u / 2) * np.cos(2 * v))
            case "R" | "CIRCULAR RIGHT":
                norm = np.cos(u / 2) ** 2
            case "L" | "CIRCULAR LEFT":
                norm = np.sin(u / 2) ** 2
            case _:
                GOOD = ["vertical", "horizontal", "circular left", "circular right"]
                raise ValueError(f"Wrong value for target state, shoud be in {GOOD}")

        return norm

    def get_info_string(self):
        """Returns an info string for the current polarization state"""
        # - definitions
        HEADER = ". {} :\n"
        PARAM = "  ├── {} : {}\n"
        LPARAM = "  └── {} : {}\n\n"
        TITLE = "| POLARIZATION PROPERTIES |\n"
        LINE = "─" * len(TITLE) + "\n"

        # - generate info string
        out_str = LINE
        out_str += TITLE
        out_str += LINE
        # object settings
        out_str += HEADER.format("Settings")
        out_str += PARAM.format("type", self.type)
        out_str += PARAM.format(
            "angle", "None" if self.angle is None else f"{self.angle / np.pi:.2f} pi"
        )
        out_str += LPARAM.format("vec", self.vec)

        # vector
        u, v = self.get_polarization_vector_angles()
        out_str += HEADER.format("Polarization vector")
        x, y, z = self.get_polarization_vector()
        out_str += PARAM.format("coords", f"({x:.2f}, {y:.2f}, {z:.2f})")
        out_str += PARAM.format("polar angle u", f"{u/np.pi:.2f} pi")
        out_str += LPARAM.format("azimt angle v", f"{u/np.pi:.2f} pi")

        # Projections (amplitudes)
        u, v = self.get_polarization_vector_angles()
        out_str += HEADER.format("Projections (amplitudes)")
        for target in ["vertical", "horizontal", "circular left", "circular right"]:
            proj = self.get_polarization_vector_projection(target)
            if target == "circular right":
                out_str += LPARAM.format(target, f"{proj:.2f}")
            else:
                out_str += PARAM.format(target, f"{proj:.2f}")

        # Projections (norm)
        u, v = self.get_polarization_vector_angles()
        out_str += HEADER.format("Projections (squared norm)")
        for target in ["vertical", "horizontal", "circular left", "circular right"]:
            proj = self.get_polarization_vector_projection_norm(target)
            if target == "circular right":
                out_str += LPARAM.format(target, f"{proj:.2f}")
            else:
                out_str += PARAM.format(target, f"{proj:.2f}")

        return out_str

    def display_info_string(self):
        """Prints an info string for the current polarization state"""
        print(self.get_info_string())


# % ACTUAL IMPLEMENTATIONS


class Vertical(AbstractPolarization):
    """Vertical polarization (along x in the laser frame)"""

    def __init__(self):
        super().__init__()
        self.type = "Vertical"

    def _get_polarization_vector(self):
        """For vertical polarization (along x) > (1, 0, 0)"""
        return (1, 0, 0)


class Horizontal(AbstractPolarization):
    """Horizontal polarization (along y in the laser frame)"""

    def __init__(self):
        super().__init__()
        self.type = "Horizontal"

    def _get_polarization_vector(self):
        """For horizontal polarization (along y) > (0, 1, 0)"""
        return (0, 1, 0)


class CircularLeft(AbstractPolarization):
    """Circular Left polarization (observer point of vue)"""

    def __init__(self):
        super().__init__()
        self.type = "Circular Left"

    def _get_polarization_vector(self):
        """For circular left polarization > (0, 0, -1)"""
        return (0, 0, -1)


class CircularRight(AbstractPolarization):
    """Circular Right polarization (observer point of vue)"""

    def __init__(self):
        super().__init__()
        self.type = "Circular Right"

    def _get_polarization_vector(self):
        """For circular right polarization > (0, 0, 1)"""
        return (0, 0, 1)


class Linear(AbstractPolarization):
    """Arbitrary linear polarization"""

    def __init__(self, angle):
        """Arbitrary linear polarization. `angle` is the angle of the linear polarization with respect to the x axis.
        Hence `angle = 0` corresponds to a linear polarization along x, and `angle = pi/2` to a linear polarization along y

        Args:
            angle (float): angle of the arbitrary linear polarization w.r.t the x axis
        """
        super().__init__()
        self.type = "Linear"
        self.angle = angle

    def _get_polarization_vector(self):
        """For arbitrary linear polarization > (cos(theta), sin(theta), 1)"""
        return (np.cos(self.angle), np.sin(self.angle), 0)

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

        if not isinstance(value, float):
            raise ValueError("Angle must be a float")

        self._angle = value
