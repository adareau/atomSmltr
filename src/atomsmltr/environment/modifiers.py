"""
modifiers for environment objects
===================================

This module contains several modifiers (decorators) for environment
objects, that can be used to perform spatial translations or
rotation on those objects.

Warning
--------

(!) This module is still at an experimental stage (!)

Do not use it yet, as it might yield unexpected results


Examples
---------

Rotate a laser beam

.. code-block:: python

    import numpy as np
    from atomsmltr.environment import GaussianLaserBeam
    from atomsmltr.environment.modifiers import rotate

    laser = GaussianLaserBeam(direction=(1,0,0))
    rotate(laser, (0,0,1), np.pi/4)


Shift a magnetic field quadrupole

.. code-block:: python

    from atomsmltr.environment import MagneticQuadrupoleX
    from atomsmltr.environment.modifiers import shift

    mag_field = MagneticQuadrupoleX(origin=(0,0,0), slope=1)
    shift(mag_field, (-1,0,1))

"""

# % IMPORTS
import numpy as np
from functools import wraps

# % LOCAL IMPORTS
from .envbase import EnvObject
from ..utils.misc import check_position_array

# % USEFUL FUNCTIONS


def rotation_matrix(u: np.ndarray, theta: float) -> np.ndarray:
    """Generates 3D rotation matrix

    Parameters
    ----------
    u : array, shape (3,)
        the axis around which to perform the rotation
        it does not need to be normalized, the function will take
        care of it
    theta : float
        the angle for the rotation

    Returns
    -------
    rotmat : array, shape (3,)
        the rotation matrix

    References
    -----------
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

    """
    # - normalize vector
    u = np.asanyarray(u)
    assert u.shape == (3,), "'u' should be an array of shape (3,)"
    assert np.linalg.norm(u) > 0, "'u' should not be null"
    u = u / np.linalg.norm(u)

    # - prepare useful matrices
    I = np.eye(3)
    u_cross = np.cross(I, u)
    u_out = np.outer(u, u)

    # - generate rotation matrix
    rotmat = np.cos(theta) * I + np.sin(theta) * u_cross + (1 - np.cos(theta)) * u_out

    return rotmat


def rotate_position_vector(
    position: np.ndarray, u: np.ndarray, theta: float
) -> np.ndarray:
    """Rotates a position vector

    Parameters
    ----------
    position : array of shape (3,) or (n1, n2, ..., 3)
        cartesian coordinates in the lab frame
    u : array, shape (3,)
        the axis around which to perform the rotation
        it does not need to be normalized, the function will take
        care of it
    theta : float
        the angle for the rotation

    Returns
    -------
    rotated_position : array of shape (3,) or (n1, n2, ..., 3)
        rotated position vector

    Notes
    -------
    position is an array_like object, with shape (3,) or (n1, n2, .., 3).
    In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame
    """
    # - compute rotation matrix
    rotmat = rotation_matrix(u, theta)
    # - perform rotation
    rotated_position = np.tensordot(rotmat, position.T, axes=(1, 0)).T
    return rotated_position


def rotate_position_vector_alt(
    position: np.ndarray, u: np.ndarray, theta: float
) -> np.ndarray:
    """Rotates a position vector

    Alternative version to ``rotate_position_vector``, not using the
    numpy tensordot method, and therefore less performant. Included
    as a more transparent comparison, to make sure that tensordot is
    doing what we want.

    Parameters
    ----------
    position : array of shape (3,) or (n1, n2, ..., 3)
        cartesian coordinates in the lab frame
    u : array, shape (3,)
        the axis around which to perform the rotation
        it does not need to be normalized, the function will take
        care of it
    theta : float
        the angle for the rotation

    Returns
    -------
    rotated_position : array of shape (3,) or (n1, n2, ..., 3)
        rotated position vector

    Notes
    -------
    position is an array_like object, with shape (3,) or (n1, n2, .., 3).
    In all cases, the last dimension contains cordinates (x, y, z), in meter and in the lab frame
    """
    # - compute rotation matrix
    rotmat = rotation_matrix(u, theta)
    # - get coordinates
    X, Y, Z = position.T
    # - apply rotation matrix "by hand"
    Xrot = rotmat[0, 0] * X + rotmat[0, 1] * Y + rotmat[0, 2] * Z
    Yrot = rotmat[1, 0] * X + rotmat[1, 1] * Y + rotmat[1, 2] * Z
    Zrot = rotmat[2, 0] * X + rotmat[2, 1] * Y + rotmat[2, 2] * Z
    # - generate rotated position
    rotated_position = np.array([Xrot, Yrot, Zrot]).T
    return rotated_position


# % WRAPPERS


# -- ROTATION


def _rotation_get_value_decorator(get_value, u: np.ndarray, theta: float, vector: bool):
    """decorator for the get_value method for a rotation modifier

    Parameters
    ----------
    get_value : function
        the method to decorate
    u : array, shape (3,)
        the axis around which to perform the rotation
        it does not need to be normalized, the function will take
        care of it
    theta : float
        the angle for the rotation
        care of it
    vector : bool
        whether the result of 'get_value' is a vector
    """

    @wraps(get_value)
    def wrapped(position, nocheck=False, *args, **kwargs):
        position = check_position_array(position, nocheck)
        # note : we rotate the *coordinates* by -theta
        # to have the *object* rotated by +theta
        rotated_position = rotate_position_vector(position, u, -theta)
        value = get_value(rotated_position, *args, **kwargs)
        if vector:
            value = rotate_position_vector(value, u, theta)
        return value

    return wrapped


def _rotation_gen_infostring_obj_dectorator(
    gen_infostring_obj, u: np.ndarray, theta: float
):
    """decorator for the gen_infostring_obj method for a roation modifier
    see _rotation_get_value_decorator for the arguments definition
    """

    @wraps(gen_infostring_obj)
    def wrapped(*args, **kwargs):
        info = gen_infostring_obj(*args, **kwargs)
        info.add_element("<modifier> : rotation")
        info.add_element("<axis>", u)
        info.add_element("<theta>", theta)
        return info

    return wrapped


def rotate(obj: EnvObject, u: np.ndarray, theta: float):
    """Modifier for environment objects, performing a rotation

    Parameters
    ----------
    obj : EnvObject
        the object to rotate
    u : array, shape (3,)
        the axis around which to perform the rotation
        it does not need to be normalized, the function will take
        care of it
    theta : float
        the angle for the rotation

    Returns
    -------
    rotated_obj : EnvObject
        the rotated object
    """
    # - check
    if not isinstance(obj, EnvObject):
        raise TypeError("`obj` should be a EnvObject")

    # - decorate
    obj.get_value = _rotation_get_value_decorator(
        obj.get_value, u, theta, vector=obj.vector
    )
    obj.gen_infostring_obj = _rotation_gen_infostring_obj_dectorator(
        obj.gen_infostring_obj, u, theta
    )

    return obj


# -- SHIFT


def _shift_get_value_decorator(get_value, dr: np.ndarray):
    """decorator for the get_value method for a shift modifier

    Parameters
    ----------
    get_value : function
        the method to decorate
    dr : array, shape (3,)
        the value of the shift (cartesian coordinates)
    """

    @wraps(get_value)
    def wrapped(position, nocheck=False, *args, **kwargs):
        position = check_position_array(position, nocheck)
        # note : we shift the *coordinates* by -dr
        # to have the *object* shifted by +dr
        X, Y, Z = position.T
        dx, dy, dz = dr
        Xs = X - dx
        Ys = Y - dy
        Zs = Z - dz
        shifted_position = np.array([Xs, Ys, Zs]).T
        value = get_value(shifted_position, *args, **kwargs)
        return value

    return wrapped


def _shift_gen_infostring_obj_dectorator(gen_infostring_obj, dr: np.ndarray):
    """decorator for the gen_infostring_obj method for a shift modifier
    see _rotation_get_value_decorator for the arguments definition
    """

    @wraps(gen_infostring_obj)
    def wrapped(*args, **kwargs):
        info = gen_infostring_obj(*args, **kwargs)
        info.add_element("<modifier> : shift")
        info.add_element("<shift>", dr)
        return info

    return wrapped


def shift(obj: EnvObject, dr: np.ndarray):
    """Modifier for environment objects, performing a spatial shift

    Parameters
    ----------
    obj : EnvObject
        the object to rotate
    dr : array, shape (3,)
        the value of the shift (cartesian coordinates)

    Returns
    -------
    shifted_obj : EnvObject
        the shifted object
    """
    # - check
    if not isinstance(obj, EnvObject):
        raise TypeError("`obj` should be a EnvObject")

    # - decorate
    obj.get_value = _shift_get_value_decorator(obj.get_value, dr)
    obj.gen_infostring_obj = _shift_gen_infostring_obj_dectorator(
        obj.gen_infostring_obj, dr
    )

    return obj
