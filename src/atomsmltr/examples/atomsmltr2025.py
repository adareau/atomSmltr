"""
Examples: atomSmltr 2025
=======================

This example provides the configurations used to generate the figures of the atomSmltr
presentation paper from 2025. See the paper for more information.

>> from atomsmltr.examples.atomsmltr import config_1D_MOT_Yb

1D MOT for Ytterbium, used to benchmark the damped oscillation of an atom in a MOT
using analytical formulas for the fluid friction and harmonic oscillator stiffness.

"""

# % EXPERIENCE DESCRIPTION
description = """

Examples from the atomSmltr presentation paper (2025).


"""

# % IMPORTS

import magpylib as magpy
import numpy as np

# % LOCAL IMPORTS

from ..atoms import Ytterbium
from ..environment.lasers import PlaneWaveLaserBeam
from ..environment.fields import MagneticGradient
from ..environment.lasers import CircularRight
from ..simulation import Configuration

# % EXAMPLE 1 - 3D MOT SPRING MODEL

# - parameters
atom = Ytterbium()
transition = atom.trans["main"]
transition.print_info()
s = 0.02
b = 0.1  # T/m
delta = -0.5 * transition.Gamma

# lasers
laser_1 = PlaneWaveLaserBeam(
    wavelength=transition.wavelength,
    direction=(1, 0, 0),
    tag="399+",
    polarization=CircularRight(),
)
laser_2 = PlaneWaveLaserBeam(
    wavelength=transition.wavelength,
    direction=(-1, 0, 0),
    tag="399-",
    polarization=CircularRight(),
)
laser_1.set_power_from_I(s * transition.Isat)
laser_2.set_power_from_I(s * transition.Isat)

# magnetic field
mag_gradient = MagneticGradient(
    origin=(0, 0, 0),
    slope=b,
    gradient_direction=(1, 0, 0),
    field_direction=(1, 0, 0),
    tag="mag_gradient",
)
# config
config_1D_MOT_Yb = Configuration(atom=atom) + laser_1 + laser_2 + mag_gradient
config_1D_MOT_Yb.add_atomlight_coupling("399+", "main", detuning=delta)
config_1D_MOT_Yb.add_atomlight_coupling("399-", "main", detuning=delta)
config_1D_MOT_Yb.delta = delta
config_1D_MOT_Yb.s = s
config_1D_MOT_Yb.b = b
