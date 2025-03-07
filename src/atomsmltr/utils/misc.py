"""misc
==================

a collection of useful functions

"""

# % IMPORTS
import numpy as np
from random import choice

# % ARGUMENT PROCESSORS / CHECKERS


def check_positive_float(param_name: str, value: float) -> None:
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


def check_position_array(position, nocheck=False):
    """Checks that a position array matches our vectorization convention.

    It raises an error if the shape is not good.

    Parameters
    ----------
    position : array
        the array to check
    nocheck : bool, optional
        if set to True, the function is bypasses

    Returns
    -------
    position
        the array

    Notes
    ------
    positions array should have a shape (3,) or (n1, n2, .., 3).

    In all cases, the last dimension contains cordinates (x, y, z),
    in meter and in the lab frame
    """
    if nocheck:
        return position
    # convert to array
    position = np.asanyarray(position)
    # check that shape is fine : should be (3,) or (n,3)
    if not position.shape or position.shape[-1] != 3:
        raise ValueError("The position array should be of shape (3,) or (n, m, .., 3)")
    return position


def check_position_speed_array(position, nocheck=False):
    """Checks that a position array matches our vectorization convention.

    It raises an error if the shape is not good.

    Parameters
    ----------
    position : array
        the array to check
    nocheck : bool, optional
        if set to True, the function is bypasses

    Returns
    -------
    position
        the array

    Notes
    ------
    positions array should have a shape (6,) or (n1, n2, .., 6).

    In all cases, the last dimension contains cordinates (x, y, z, vx, vy, vz),
    in meter and in the lab frame
    """
    if nocheck:
        return position
    # convert to array
    position = np.asanyarray(position)
    # check that shape is fine : should be (6,) or (n,6)
    if not position.shape or position.shape[-1] != 6:
        raise ValueError("The position array should be of shape (6,) or (n, m, .., 6)")
    return position


def check_scalar_field_value_function(func):
    """Used in tests : checks that a function yielding values of a 3D **scalar** field
    field behaves correctly with numpy arrays.

    for input of shape (..., 1) should return shape (..., 1)

    Parameters
    ----------
    func : function
        the function to check
    """

    # - 1 check that it works with a single position
    position = (0, 0, 0)
    value = func(position)
    assert value.ndim == 0

    # - 2 with arrays
    # -
    position = np.mgrid[0:1:8j, 0:5:10j, 0:0:1j].T
    X, _, _ = position.T
    X = X.T
    value = func(position)
    assert value.shape == X.shape
    # -
    position = position[0]
    X, _, _ = position.T
    X = X.T
    value = func(position)
    assert value.shape == X.shape


def check_vector_field_value_function(func):
    """Used in tests : checks that a function yielding values of a 3D **vector** field
    field behaves correctly with numpy arrays.

    for input of shape (..., 3) should return shape (..., 3)

    Parameters
    ----------
    func : function
        the function to check
    """

    # - 1 check that it works with a single position
    position = (0, 0, 0)
    value = func(position)
    assert value.shape == (3,)

    # - 2 with arrays
    # -
    position = np.mgrid[0:1:8j, 0:5:10j, 0:0:1j].T
    value = func(position)
    assert value.shape == position.shape
    # -
    position = position[0]
    value = func(position)
    assert value.shape == position.shape


# % RANDOM TAG


def random_word(syl=3):
    """Generates a random word

    Parameters
    ----------
    syl : int, optional
        number of syllabs, by default 3

    Returns
    -------
    word: str
        the random word
    """
    voy = "aeiou"
    cons = "zrtpqsdfghklmwxvbn"
    res = ""
    for i in range(syl):
        res += choice(cons) + choice(voy)
    return res
