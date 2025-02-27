# -*- coding: utf-8 -*-
"""some useful functions
"""

# % IMPORTS
import numpy as np
from random import choice

# % ARGUMENT PROCESSORS / CHECKERS


def check_position_array(position, nocheck=False):
    """Checks position vectors required for laser intensity, mag field, etc.. value compuation"""
    if nocheck:
        return position
    # convert to array
    position = np.asanyarray(position)
    # check that shape is fine : should be (3,) or (n,3)
    if not position.shape or position.shape[-1] != 3:
        raise ValueError("The position array should be of shape (3,) or (n, m, .., 3)")
    return position


def check_scalar_field_value_function(func):
    """Checks that a function yielding values of a 3D field
    field behaves correctly with numpy arrays. Typically used to
    check intensities.

    for input of shape (..., 1) should return shape (..., 1)

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
    """Checks that a function yielding values of a 3D field
    field behaves correctly with numpy arrays.

    for input of shape (.., 3), should return (..., 3)
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
    voy = "aeiou"
    cons = "zrtpqsdfghklmwxvbn"
    res = ""
    for i in range(syl):
        res += choice(cons) + choice(voy)
    return res
