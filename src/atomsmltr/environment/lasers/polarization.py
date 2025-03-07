"""Polarization
==================

Here we implement the ``Polarization`` class that we use to define the polarization of lasers

Remarks
--------

We define the polarization in the **frame of the laser**, with laser propagation along z.

We denote **x** as the **'horizontal'** axis and **y** as the **'vertical'** axis.

For circular polarizations, we take the **observer convention**.

To have a combined formalism for all polarization, in the end we define a **polarization vector** ``p_vec``,
following the Poincarré formalism. **ATTENTION:** there might be different ways of defining this vector,
refer to the package documentation for a thorough definition.

In the case of 'linear' polarization, an additionnal argument ``angle`` has to be provided, that gives
the angle of the linear polarization with respect to the x axis. Hence ``angle = 0`` corresponds to a
linear polarization along x, and ``angle = pi/2`` to a linear polarization along y

In the case of 'vector' polarization, polarization vector has to be given with the ``vec`` argument. ``vec``
are the cartesian coordinates of the vector in the (x,y,z) basis. For instance :

| ``> vec = (1, 0, 0)``  : linear polarization along x
| ``> vec = (0, 1, 0)``  : linear polarization along y
| ``> vec = (0, 0, 1)``  : circular right polarization
| ``> vec = (0, 0, -1)`` : circular left polarization

**ATTENTION** : for ease of use, the vector does not have to be normalized, but the resulting one will
be.


"""

# % IMPORTS
import numpy as np
from abc import ABC, abstractmethod

# % LOCAL IMPORTS
from ...utils.infostring import InfoString


# % ABSTRACT CLASS


class Polarization(ABC):
    """An object to handle laser polarization."""

    def __init__(self):
        self._vector = None
        self._u = None
        self._v = None

    # -- PROPERTIES
    @property
    def vector(self):
        vector = np.asanyarray(self._vector)
        return vector

    # -- METHODS

    def get_polarization_vector_angles(self) -> tuple:
        """Returns the angles describing the current polarization vector.

        (see documentation for thorough description)

        Returns
        -------
        u, v : floats
            the u (polar) and v (azimuthal) angles

        Notes
        -----
        The polarization is decribed in the Poincarré/Bloch-like sphere as a vector.
        This function yields the angles u (polar) and v (azimuthal)

        Note that we do not use theta or phi as those angles are already used to
        describe the orientation of the laser propagation vector in the ``LaserBeam`` class

        """
        u = self._u
        v = self._v
        return u, v

    def refresh_polarization_vector_angles(self):
        """Updates the polarization vector angles u & v to match the current value of the
        polarization vector.
        """
        x, y, z = self.vector
        u = np.arctan2(np.sqrt(x**2 + y**2), z)
        v = np.arctan2(y, x)
        self._u = u
        self._v = v

    def get_polarization_vector_projection(self, target: str) -> complex:
        """Returns the scalar projection of the current polarization vector on a target polarization state

        Parameters
        ----------
        target : str
            the state on which to project (see docstring Notes)

        Returns
        -------
        proj: complex
            the projection

        Notes
        -----
        The polarization Psi is defined as :

            |Psi⟩ = exp(-i*v) cos(u/2) |R⟩ +  exp(i*v) sin(u/2) |L⟩

        with |R⟩, |L⟩ the right- and left-handed circular polarization states. We also have

            |x⟩ = |V⟩ = (1/sqrt(2)) (|L⟩ + |R⟩)
            |y⟩ = |H⟩ = (i/sqrt(2)) (|L⟩ - |L⟩)

        Target should refer to the special polarization states defined in the class :

        >>> 'vertical', 'horizontal', 'circular left', 'circular right'

        and corresponding shorthands:

        >>> 'V' or 'x', 'H' or 'y', 'R', 'L'

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

    def get_polarization_vector_projection_norm(self, target: str) -> float:
        """Returns the squared norm of scalar projection of the current polarization vector on a target polarization state

        Parameters
        ----------
        target : str
            the state on which to project (see docstring Notes)

        Returns
        -------
        norm: float
            the squared norm of the projection

        Notes
        -----
        The polarization Psi is defined as :

            |Psi⟩ = exp(-i*v) cos(u/2) |R⟩ +  exp(i*v) sin(u/2) |L⟩

        with |R⟩, |L⟩ the right- and left-handed circular polarization states. We also have

            |x⟩ = |V⟩ = (1/sqrt(2)) (|L⟩ + |R⟩)
            |y⟩ = |H⟩ = (i/sqrt(2)) (|L⟩ - |L⟩)

        Target should refer to the special polarization states defined in the class :

        >>> 'vertical', 'horizontal', 'circular left', 'circular right'

        and corresponding shorthands:

        >>> 'V' or 'x', 'H' or 'y', 'R', 'L'

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

    def gen_infostring_obj(self):
        """Returns an info string object for the current polarization state"""
        # - init InfoString object
        info = InfoString("POLARIZATION PROPERTIES")
        # - populate info string
        # object settings
        info.add_section("Polarization settings")
        if isinstance(self, Linear):
            info.add_element("type", self.type)
            info.add_element("angle", f"{self.angle / np.pi:.2f} pi")
        elif isinstance(self, Vector):
            info.add_element("type", self.type)
            info.add_element("vector", self.vector)
        else:
            info.add_element("type", self.type)

        # vector
        info.add_section("Polarization vector")
        u, v = self.get_polarization_vector_angles()
        x, y, z = self.vector
        info.add_element("coords", f"({x:.2f}, {y:.2f}, {z:.2f})")
        info.add_element("polar angle u", f"{u/np.pi:.2f} pi")
        info.add_element("azimt angle v", f"{v/np.pi:.2f} pi")

        # Projections (amplitudes)
        u, v = self.get_polarization_vector_angles()
        info.add_section("Projections (amplitudes)")
        for target in ["vertical", "horizontal", "circular left", "circular right"]:
            proj = self.get_polarization_vector_projection(target)
            if target == "circular right":
                info.add_element(target, f"{proj:.2f}")
            else:
                info.add_element(target, f"{proj:.2f}")

        # Projections (norm)
        u, v = self.get_polarization_vector_angles()
        info.add_section("Projections (squared norm)")
        for target in ["vertical", "horizontal", "circular left", "circular right"]:
            proj = self.get_polarization_vector_projection_norm(target)
            if target == "circular right":
                info.add_element(target, f"{proj:.2f}")
            else:
                info.add_element(target, f"{proj:.2f}")

        return info

    def gen_info_string(self, **kwargs):
        return self.gen_infostring_obj().generate(**kwargs)

    def print_info(self):
        print(self.gen_info_string())


# % ACTUAL IMPLEMENTATIONS


class Vertical(Polarization):
    """Vertical polarization (along x in the laser frame)"""

    def __init__(self):
        super(Vertical, self).__init__()
        self.type = "Vertical"
        self._vector = (1, 0, 0)
        self.refresh_polarization_vector_angles()


class Horizontal(Polarization):
    """Horizontal polarization (along y in the laser frame)"""

    def __init__(self):
        super(Horizontal, self).__init__()
        self.type = "Horizontal"
        self._vector = (0, 1, 0)
        self.refresh_polarization_vector_angles()


class CircularLeft(Polarization):
    """Circular Left polarization (observer point of vue)"""

    def __init__(self):
        super(CircularLeft, self).__init__()
        self.type = "Circular Left"
        self._vector = (0, 0, -1)
        self.refresh_polarization_vector_angles()


class CircularRight(Polarization):
    """Circular Right polarization (observer point of vue)"""

    def __init__(self):
        super(CircularRight, self).__init__()
        self.type = "Circular Right"
        self._vector = (0, 0, 1)
        self.refresh_polarization_vector_angles()


class Linear(Polarization):
    """Arbitrary linear polarization.

    Here, ``angle`` is the angle of the linear polarization with respect to the x axis.
    Hence ``angle = 0`` corresponds to a linear polarization along x, and ``angle = pi/2`` to a linear polarization along y

    Parameters
    ----------
    angle : float
        angle of the arbitrary linear polarization w.r.t the x axis (radians)
    """

    def __init__(self, angle: float):
        super(Linear, self).__init__()
        self.type = "Linear"
        self.angle = angle

    @property
    def angle(self) -> float:
        """float: angle of the arbitrary linear polarization w.r.t the x axis (radians)"""
        return self._angle

    @angle.setter
    def angle(self, value: float) -> None:
        # convert int into float
        if isinstance(value, int):
            value = float(value)

        if not isinstance(value, float):
            raise ValueError("Angle must be a float")

        self._angle = value
        self._vector = (np.cos(self.angle), np.sin(self.angle), 0)
        self.refresh_polarization_vector_angles()


class Vector(Polarization):
    """Allows to define an arbitrary polarization.

    The polarization vector has to be given with the ``vector`` argument.
    ``vector`` are the cartesian coordinates of the vector in the (x,y,z) basis.

    For instance :

    | ``> vec = (1, 0, 0)``  : linear polarization along x
    | ``> vec = (0, 1, 0)``  : linear polarization along y
    | ``> vec = (0, 0, 1)``  : circular right polarization
    | ``> vec = (0, 0, -1)`` : circular left polarization

    **ATTENTION** : for ease of use, the vector does not have to be normalized, but the resulting one will
    be.

    Parameters
    ----------
    vector : array of shape (,3)
        polarization vector cartesian coordinates in laser frame
        (see documentation for its exact definition)
    """

    def __init__(self, vector: np.ndarray):
        super(Vector, self).__init__()
        self.type = "Vector"
        self.vector = vector

    @property
    def vector(self):
        """array of shape (3,) : polarization vector cartesian coordinates in laser frame"""
        vector = np.asanyarray(self._vector)
        return vector

    @vector.setter
    def vector(self, value: np.ndarray) -> None:
        # convert to array
        value = np.asanyarray(value)
        if value.size != 3:
            raise ValueError("'vector' should be of size 3")
        # normalize
        norm = np.linalg.norm(value)
        if norm == 0:
            raise ValueError("Wrong value for 'vector'': norm is zero")
        self._vector = value / norm
        self.refresh_polarization_vector_angles()
