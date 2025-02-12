# -*- coding: utf-8 -*-
"""Defines the magnetic field class
"""

# % IMPORTS
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# % LOCAL IMPORTS
from .generic import AbstractField, AbstractGradientField, AbstractOffsetField


# % CLASSES

# -- MAG FIELDs PARENT CLASS
#   not really used currently, but will be useful if we need to
#   implement features specific to mag. fields.
# > will also allow to check that the field is indeed a magnetic field
#   in the environment object


class MagneticField(AbstractField):
    """Our magnetic field class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def type(self):
        return "magnetic field"


# -- PERFECT FIELDS CLASSES
class MagneticOffset(MagneticField, AbstractOffsetField):
    """Our magnetic field class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def type(self):
        return "magnetic field offset"


class MagneticGradient(MagneticField, AbstractOffsetField):
    """Our magnetic field class"""

    pass
