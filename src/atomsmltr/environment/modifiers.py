"""
modifiers for environment objects
===================================

This module contains several modifiers (decorators) for environment
objects, that can be used to perform spatial translations or
rotation on those objects.
"""

# % IMPORTS
import numpy as np

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
