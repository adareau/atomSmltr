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


# % RANDOM TAG


def random_word(syl=3):
    voy = "aeiou"
    cons = "zrtpqsdfghklmwxvbn"
    res = ""
    for i in range(syl):
        res += choice(cons) + choice(voy)
    return res
